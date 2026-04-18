import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_open_world import get_args_parser, build_datasets
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from util.visual.cases import (
    MSDeformAttnRecorder,
    ODQEGateRecorder,
    compute_query_scores,
    ordered_attention_records,
    ordered_gate_records,
    plot_joint_panel,
    plot_selected_query_gate_overview,
    save_joint_statistics_csv,
    select_queries,
)
from util.visual.helper import draw_gt_boxes, save_svg_image, to_numpy_image


def build_parser():
    parser = argparse.ArgumentParser('SVG-first joint ODQE gate and query sampling visualization', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--split', default='eval', choices=['train', 'eval'])
    parser.add_argument('--sample_index', default=0, type=int)
    parser.add_argument('--num_known_queries', default=3, type=int)
    parser.add_argument('--num_unknown_queries', default=3, type=int)
    parser.add_argument('--output_subdir', default='infer/odqe_joint_mechanism', type=str)
    return parser


def main(parsed_args):
    device = torch.device(parsed_args.device)
    model, _, _, _ = build_model(parsed_args, mode=parsed_args.model_type)
    checkpoint = torch.load(parsed_args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    train_dataset, eval_dataset = build_datasets(parsed_args)
    dataset = eval_dataset if parsed_args.split == 'eval' else train_dataset
    image, target = dataset[parsed_args.sample_index]
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

    if not attn_records:
        raise RuntimeError('No deformable decoder cross-attention records were captured.')
    if not gate_records:
        raise RuntimeError('No ODQE gate records were captured. Check whether --uod_enable_odqe is enabled and the checkpoint includes gate_mlp.')
    if len(attn_records) != len(gate_records):
        raise RuntimeError(f'Layer count mismatch: attention={len(attn_records)} gate={len(gate_records)}')

    invalid_cls_logits = list(range(parsed_args.PREV_INTRODUCED_CLS + parsed_args.CUR_INTRODUCED_CLS, parsed_args.num_classes - 1))
    scores = compute_query_scores(outputs, parsed_args, invalid_cls_logits)
    selected = select_queries(scores, parsed_args.num_known_queries, parsed_args.num_unknown_queries)

    image_hw = target['size'].tolist() if 'size' in target else image.shape[-2:]
    image_np = to_numpy_image(image, image_hw)
    unknown_label = int(parsed_args.num_classes - 1)
    base_image = draw_gt_boxes(image_np, target, unknown_label)

    image_id = int(target['image_id'].item()) if 'image_id' in target else parsed_args.sample_index
    out_dir = Path(parsed_args.output_dir) / parsed_args.output_subdir / f'image_{image_id:012d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_svg_image(base_image, out_dir / 'image_with_gt.svg')
    save_joint_statistics_csv(selected, attn_records, gate_records, scores, out_dir / 'selected_query_joint_statistics.csv')
    plot_selected_query_gate_overview(selected, gate_records, out_dir / 'selected_query_effective_gate_overview.svg')

    for query_kind, query_index in selected:
        output_path = out_dir / f'{query_kind}_query_{int(query_index):03d}_joint_mechanism.svg'
        plot_joint_panel(query_kind, int(query_index), base_image, attn_records, gate_records, scores, output_path)

    print(f'Saved SVG joint ODQE/query mechanism visualizations to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
