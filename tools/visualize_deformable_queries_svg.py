import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_open_world import get_args_parser, build_datasets
from models import build_model
from models.ops.modules import MSDeformAttn
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from tools.figure_svg_utils import save_svg_figure, save_svg_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LEVEL_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00897B']


def to_numpy_image(image_tensor, target_hw=None):
    image = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    if target_hw is not None:
        height, width = int(target_hw[0]), int(target_hw[1])
        image = image[:height, :width]
    return (image * 255).astype(np.uint8)


def draw_gt_boxes(image_np, target, unknown_label):
    image = Image.fromarray(image_np).convert('RGB')
    draw = ImageDraw.Draw(image)
    boxes = box_ops.box_cxcywh_to_xyxy(target['boxes'].detach().cpu()).numpy()
    h, w = image_np.shape[:2]
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    labels = target['labels'].detach().cpu().numpy()
    for box, label in zip(boxes, labels):
        color = (243, 156, 18) if int(label) == int(unknown_label) else (0, 188, 212)
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return np.array(image)


class MSDeformAttnRecorder:
    def __init__(self, model):
        self.records = []
        self.patches = []
        for name, module in model.named_modules():
            if isinstance(module, MSDeformAttn) and 'transformer.decoder.layers' in name and name.endswith('cross_attn'):
                self._patch_module(name, module)

    def _patch_module(self, name, module):
        original_forward = module.forward
        recorder = self

        def wrapped_forward(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
            N, Len_q, _ = query.shape
            value = module.value_proj(input_flatten)
            if input_padding_mask is not None:
                value = value.masked_fill(input_padding_mask[..., None], float(0))
            value = value.view(N, input_flatten.shape[1], module.n_heads, module.d_model // module.n_heads)
            sampling_offsets = module.sampling_offsets(query).view(N, Len_q, module.n_heads, module.n_levels, module.n_points, 2)
            attention_weights = module.attention_weights(query).view(N, Len_q, module.n_heads, module.n_levels * module.n_points)
            attention_weights = torch.softmax(attention_weights, -1).view(N, Len_q, module.n_heads, module.n_levels, module.n_points)
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            elif reference_points.shape[-1] == 4:
                sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / module.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                raise ValueError('Unsupported reference point shape')
            recorder.records.append({
                'name': name,
                'sampling_locations': sampling_locations.detach().cpu(),
                'attention_weights': attention_weights.detach().cpu(),
                'reference_points': reference_points.detach().cpu(),
            })
            return original_forward(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask)

        module.forward = wrapped_forward
        self.patches.append((module, original_forward))

    def restore(self):
        for module, original_forward in self.patches:
            module.forward = original_forward


def compute_query_scores(outputs, args, invalid_cls_logits):
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    obj_prob = torch.exp(-obj_temp * outputs['pred_obj'].detach()).clamp(min=1e-6, max=1.0)

    class_prob = outputs['pred_logits'].detach().sigmoid().clone()
    if invalid_cls_logits:
        class_prob[:, :, invalid_cls_logits] = 0.0
    if class_prob.shape[-1] > 0:
        class_prob[:, :, -1] = 0.0
    max_known = class_prob.max(-1).values

    if 'pred_known' in outputs and outputs['pred_known'] is not None:
        known_temp = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim
        known_prob = torch.exp(-known_temp * outputs['pred_known'].detach()).clamp(min=1e-6, max=1.0)
        unknown_prob = (1.0 - known_prob).clamp(min=0.0, max=1.0)
    else:
        known_prob = torch.ones_like(obj_prob)
        unknown_prob = torch.zeros_like(obj_prob)

    known_score = obj_prob * known_prob * max_known
    unknown_score = obj_prob * unknown_prob * float(getattr(args, 'uod_postprocess_unknown_scale', 20.0))
    return {
        'obj_prob': obj_prob[0].cpu().numpy(),
        'known_prob': known_prob[0].cpu().numpy(),
        'unknown_prob': unknown_prob[0].cpu().numpy(),
        'max_known': max_known[0].cpu().numpy(),
        'known_score': known_score[0].cpu().numpy(),
        'unknown_score': unknown_score[0].cpu().numpy(),
    }


def select_queries(scores, num_known, num_unknown):
    known_indices = np.argsort(-scores['known_score'])[:num_known].tolist()
    unknown_indices = np.argsort(-scores['unknown_score'])[:num_unknown].tolist()
    selected = [('known', idx) for idx in known_indices] + [('unknown', idx) for idx in unknown_indices if idx not in known_indices]
    return selected


def _layer_order(records):
    order = []
    for record in records:
        name = record['name']
        try:
            layer_id = int(name.split('transformer.decoder.layers.')[1].split('.cross_attn')[0])
        except Exception:
            layer_id = len(order)
        order.append((layer_id, record))
    order.sort(key=lambda x: x[0])
    return order


def plot_query_sampling(image_np, records, query_kind, query_index, scores, output_path):
    ordered = _layer_order(records)
    if not ordered:
        return

    n_layers = len(ordered)
    ncols = min(3, n_layers)
    nrows = int(math.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows), squeeze=False, constrained_layout=True)
    h, w = image_np.shape[:2]

    for axis in axes.flat:
        axis.set_visible(False)

    for axis, (layer_id, record) in zip(axes.flat, ordered):
        axis.set_visible(True)
        axis.imshow(image_np)
        axis.set_axis_off()
        axis.set_title(f'Layer {layer_id}', fontsize=13, pad=8)
        sampling_locations = record['sampling_locations'][0, query_index].numpy()
        attention_weights = record['attention_weights'][0, query_index].numpy()
        reference_points = record['reference_points'][0, query_index].numpy()
        if reference_points.ndim == 2:
            ref_xy = reference_points[:, :2].mean(0)
        else:
            ref_xy = reference_points[:2]
        axis.scatter(ref_xy[0] * w, ref_xy[1] * h, c='white', edgecolors='black', s=120, marker='x', linewidths=2.2, zorder=5)
        heads_mean_locations = sampling_locations.mean(0)
        heads_mean_weights = attention_weights.mean(0)
        for lvl in range(heads_mean_locations.shape[0]):
            pts = heads_mean_locations[lvl]
            ws = heads_mean_weights[lvl]
            color = LEVEL_COLORS[lvl % len(LEVEL_COLORS)]
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            sizes = 36.0 + 260.0 * ws
            axis.scatter(xs, ys, s=sizes, c=color, alpha=0.80, edgecolors='black', linewidths=0.45, zorder=4)

    qs = scores['known_score'][query_index] if query_kind == 'known' else scores['unknown_score'][query_index]
    fig.suptitle(
        f'Query {query_index} | type={query_kind} | select_score={qs:.3f} | obj={scores["obj_prob"][query_index]:.3f} | unk={scores["unknown_prob"][query_index]:.3f} | max_known={scores["max_known"][query_index]:.3f}',
        fontsize=15,
        y=1.02,
    )

    legend_handles = [
        Line2D([0], [0], marker='x', color='black', markerfacecolor='white', markersize=10, linewidth=0, label='reference point'),
    ]
    for lvl in range(min(len(LEVEL_COLORS), heads_mean_locations.shape[0])):
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=LEVEL_COLORS[lvl], markersize=8, linewidth=0, label=f'feature level {lvl}'))
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=min(4, len(legend_handles)), frameon=False, fontsize=10)
    fig.text(0.5, -0.055, 'Circle size represents mean sampling attention weight across heads.', ha='center', va='center', fontsize=10)
    save_svg_figure(fig, output_path, pad_inches=0.08)


def save_query_summary_csv(selected, scores, output_path):
    header = 'query_kind,query_index,obj_prob,known_prob,unknown_prob,max_known,known_score,unknown_score\n'
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(header)
        for query_kind, query_index in selected:
            row = [
                query_kind,
                str(query_index),
                f"{scores['obj_prob'][query_index]:.6f}",
                f"{scores['known_prob'][query_index]:.6f}",
                f"{scores['unknown_prob'][query_index]:.6f}",
                f"{scores['max_known'][query_index]:.6f}",
                f"{scores['known_score'][query_index]:.6f}",
                f"{scores['unknown_score'][query_index]:.6f}",
            ]
            file.write(','.join(row) + '\n')


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
