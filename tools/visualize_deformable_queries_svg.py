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
from util.visual.cases import MSDeformAttnRecorder, compute_query_scores, plot_query_sampling, save_query_summary_csv, select_queries
from util.visual.helper import draw_gt_boxes, save_svg_image, to_numpy_image


def build_parser():
    parser = argparse.ArgumentParser('SVG-first Deformable DETR query visualization', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--split', default='eval', choices=['train', 'eval'])
    parser.add_argument('--sample_index', default=0, type=int)
    parser.add_argument('--num_known_queries', default=4, type=int)
    parser.add_argument('--num_unknown_queries', default=4, type=int)
    parser.add_argument('--output_subdir', default='infer/query_sampling', type=str)
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

    recorder = MSDeformAttnRecorder(model)
    try:
        with torch.no_grad():
            outputs = model(samples)
    finally:
        records = list(recorder.records)
        recorder.restore()

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
    save_query_summary_csv(selected, scores, out_dir / 'selected_queries.csv')

    for query_kind, query_index in selected:
        output_path = out_dir / f'{query_kind}_query_{int(query_index):03d}_sampling.svg'
        plot_query_sampling(base_image, records, query_kind, int(query_index), scores, output_path)

    print(f'Saved SVG query visualizations to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
