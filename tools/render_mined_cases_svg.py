import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_open_world import get_args_parser, build_datasets
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from tools.figure_svg_utils import save_svg_image
from tools.visualize_deformable_queries import compute_query_scores
from tools.visualize_odqe_gates import ODQEGateRecorder, ordered_gate_records, plot_query_gate_curve, plot_query_gate_heatmap
from tools.visualize_odqe_query_mechanism import MSDeformAttnRecorder, ordered_attention_records, plot_joint_panel
from tools.visualize_deformable_queries_svg import to_numpy_image, draw_gt_boxes, plot_query_sampling
from tools.visualize_query_trajectory_svg import compute_per_layer_scores, plot_query_trajectory_panel


def parse_categories(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_render_modes(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def normalize_query_kind(entry, scores):
    if entry.get('category') == 'known':
        return 'known'
    if entry.get('category') == 'unknown':
        return 'unknown'
    query_index = int(entry['query_index'])
    return 'unknown' if float(scores['unknown_score'][query_index]) >= float(scores['known_score'][query_index]) else 'known'


def load_selected_entries(manifest_path, categories, per_category_limit):
    with open(manifest_path, 'r', encoding='utf-8') as file:
        manifest = json.load(file)
    selected = []
    category_map = manifest.get('categories', {})
    for category in categories:
        entries = category_map.get(category, [])
        selected.extend(entries[:per_category_limit])
    return manifest, selected


def group_entries_by_sample(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[int(entry['sample_index'])].append(entry)
    return grouped


def ensure_case_dir(base_dir, entry):
    category = entry.get('category', 'unknown')
    image_id = int(entry.get('image_id', entry['sample_index']))
    query_index = int(entry['query_index'])
    case_dir = base_dir / category / f'image_{image_id:012d}' / f'query_{query_index:03d}'
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def render_case(entry, case_dir, base_image, scores, attn_records, gate_records, layer_scores, image_hw, render_modes):
    query_index = int(entry['query_index'])
    query_kind = normalize_query_kind(entry, scores)
    rendered_files = []

    if 'sampling' in render_modes:
        output_path = case_dir / f'{query_kind}_query_{query_index:03d}_sampling.svg'
        plot_query_sampling(base_image, attn_records, query_kind, query_index, scores, output_path)
        rendered_files.append(str(output_path))

    if 'gate' in render_modes and gate_records:
        curve_path = case_dir / f'{query_kind}_query_{query_index:03d}_gate_curve.svg'
        heatmap_path = case_dir / f'{query_kind}_query_{query_index:03d}_gate_heatmap.svg'
        plot_query_gate_curve(query_kind, query_index, gate_records, scores, curve_path)
        plot_query_gate_heatmap(query_kind, query_index, gate_records, heatmap_path)
        rendered_files.extend([str(curve_path), str(heatmap_path)])

    if 'joint' in render_modes and attn_records and gate_records:
        output_path = case_dir / f'{query_kind}_query_{query_index:03d}_joint_mechanism.svg'
        plot_joint_panel(query_kind, query_index, base_image, attn_records, gate_records, scores, output_path)
        rendered_files.append(str(output_path))

    if 'trajectory' in render_modes:
        output_path = case_dir / f'{query_kind}_query_{query_index:03d}_trajectory.svg'
        plot_query_trajectory_panel(query_kind, query_index, base_image, layer_scores, gate_records, image_hw, output_path)
        rendered_files.append(str(output_path))

    return {
        'category': entry.get('category', 'unknown'),
        'sample_index': int(entry['sample_index']),
        'image_id': int(entry.get('image_id', entry['sample_index'])),
        'query_index': query_index,
        'query_kind': query_kind,
        'case_dir': str(case_dir),
        'rendered_files': rendered_files,
    }


def build_parser():
    parser = argparse.ArgumentParser('SVG batch renderer for mined representative cases', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--manifest', required=True, type=str, help='path to representative_case_manifest.json')
    parser.add_argument('--split', default=None, choices=['train', 'eval', None], help='override split in manifest')
    parser.add_argument('--categories', default='known,unknown,odqe_salient', type=str)
    parser.add_argument('--per_category_limit', default=3, type=int)
    parser.add_argument('--render_modes', default='sampling,gate,joint,trajectory', type=str)
    parser.add_argument('--output_subdir', default='infer/rendered_cases', type=str)
    return parser


def main(parsed_args):
    categories = parse_categories(parsed_args.categories)
    render_modes = parse_render_modes(parsed_args.render_modes)
    manifest, entries = load_selected_entries(parsed_args.manifest, categories, parsed_args.per_category_limit)
    if not entries:
        raise RuntimeError('No selected entries found in manifest for the requested categories.')

    split = parsed_args.split or manifest.get('split', 'eval')
    device = torch.device(parsed_args.device)
    model, _, _, _ = build_model(parsed_args, mode=parsed_args.model_type)
    checkpoint = torch.load(parsed_args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    train_dataset, eval_dataset = build_datasets(parsed_args)
    dataset = eval_dataset if split == 'eval' else train_dataset

    grouped_entries = group_entries_by_sample(entries)
    invalid_cls_logits = list(range(parsed_args.PREV_INTRODUCED_CLS + parsed_args.CUR_INTRODUCED_CLS, parsed_args.num_classes - 1))

    out_dir = Path(parsed_args.output_dir) / parsed_args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    render_manifest = {
        'manifest_source': str(parsed_args.manifest),
        'checkpoint': str(parsed_args.checkpoint),
        'split': split,
        'categories': categories,
        'per_category_limit': int(parsed_args.per_category_limit),
        'render_modes': render_modes,
        'cases': [],
    }

    for sample_index, sample_entries in grouped_entries.items():
        image, target = dataset[sample_index]
        samples = nested_tensor_from_tensor_list([image]).to(device)

        attn_recorder = MSDeformAttnRecorder(model)
        gate_recorder = ODQEGateRecorder(model)
        try:
            with torch.no_grad():
                outputs = model(samples)
        finally:
            attn_records = ordered_attention_records(attn_recorder.records)
            gate_records = ordered_gate_records(gate_recorder.records)
            attn_recorder.restore()
            gate_recorder.restore()

        scores = compute_query_scores(outputs, parsed_args, invalid_cls_logits)
        layer_scores = compute_per_layer_scores(outputs, parsed_args, invalid_cls_logits)

        image_hw = target['size'].tolist() if 'size' in target else image.shape[-2:]
        image_np = to_numpy_image(image, image_hw)
        unknown_label = int(parsed_args.num_classes - 1)
        base_image = draw_gt_boxes(image_np, target, unknown_label)

        image_id = int(target['image_id'].item()) if 'image_id' in target else sample_index
        image_dir = out_dir / f'image_{image_id:012d}'
        image_dir.mkdir(parents=True, exist_ok=True)
        base_image_path = image_dir / 'image_with_gt.svg'
        if not base_image_path.exists():
            save_svg_image(base_image, base_image_path)

        for entry in sample_entries:
            case_dir = ensure_case_dir(out_dir, entry)
            result = render_case(entry, case_dir, base_image, scores, attn_records, gate_records, layer_scores, image_hw, render_modes)
            render_manifest['cases'].append(result)

    with open(out_dir / 'render_manifest.json', 'w', encoding='utf-8') as file:
        json.dump(render_manifest, file, ensure_ascii=False, indent=2)

    print(f'Saved SVG rendered mined cases to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
