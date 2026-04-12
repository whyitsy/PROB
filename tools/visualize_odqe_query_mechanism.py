import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from PIL import Image, ImageDraw

from main_open_world import get_args_parser, build_datasets
from models import build_model
from models.ops.modules import MSDeformAttn
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LEVEL_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00897B']
KNOWN_COLOR = '#00A65A'
UNKNOWN_COLOR = '#D81B60'


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


class ODQEGateRecorder:
    def __init__(self, model):
        self.records = []
        self.patches = []
        self.decay = None
        if hasattr(model, 'odqe_layer_decay'):
            try:
                self.decay = model.odqe_layer_decay.detach().cpu().numpy().astype(np.float32)
            except Exception:
                self.decay = None
        for name, module in model.named_modules():
            if self._is_gate_module(name, module):
                self._patch_module(name, module)

    @staticmethod
    def _is_gate_module(name, module):
        return name.startswith('gate_mlp.') and hasattr(module, 'layers') and callable(getattr(module, 'forward', None))

    @staticmethod
    def _parse_layer_id(name):
        try:
            return int(name.split('gate_mlp.')[1].split('.')[0])
        except Exception:
            return -1

    def _patch_module(self, name, module):
        original_forward = module.forward
        recorder = self
        layer_id = self._parse_layer_id(name)

        def wrapped_forward(x):
            out = original_forward(x)
            gate = torch.sigmoid(out)
            decay_value = 1.0
            if recorder.decay is not None and 0 <= layer_id < len(recorder.decay):
                decay_value = float(recorder.decay[layer_id])
            recorder.records.append({
                'name': name,
                'layer_id': layer_id,
                'raw_gate': gate.detach().cpu(),
                'effective_gate': (gate * decay_value).detach().cpu(),
                'decay': decay_value,
            })
            return out

        module.forward = wrapped_forward
        self.patches.append((module, original_forward))

    def restore(self):
        for module, original_forward in self.patches:
            module.forward = original_forward


def ordered_attention_records(records):
    order = []
    for record in records:
        try:
            layer_id = int(record['name'].split('transformer.decoder.layers.')[1].split('.cross_attn')[0])
        except Exception:
            layer_id = len(order)
        order.append((layer_id, record))
    order.sort(key=lambda x: x[0])
    return [record for _, record in order]


def ordered_gate_records(records):
    ordered = sorted(records, key=lambda item: item['layer_id'])
    return [item for item in ordered if item['layer_id'] >= 0]


def save_joint_statistics_csv(selected, attn_records, gate_records, scores, output_path):
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'query_kind', 'query_index', 'layer_id', 'decay',
            'obj_prob', 'unknown_prob', 'max_known', 'known_score', 'unknown_score',
            'raw_gate_mean', 'raw_gate_std', 'effective_gate_mean', 'effective_gate_std',
            'attention_mean', 'attention_max'
        ])
        for query_kind, query_index in selected:
            for layer_id, (attn_record, gate_record) in enumerate(zip(attn_records, gate_records)):
                raw_gate = gate_record['raw_gate'][0, query_index].numpy()
                effective_gate = gate_record['effective_gate'][0, query_index].numpy()
                attn = attn_record['attention_weights'][0, query_index].numpy()
                writer.writerow([
                    query_kind,
                    int(query_index),
                    int(layer_id),
                    float(gate_record['decay']),
                    float(scores['obj_prob'][query_index]),
                    float(scores['unknown_prob'][query_index]),
                    float(scores['max_known'][query_index]),
                    float(scores['known_score'][query_index]),
                    float(scores['unknown_score'][query_index]),
                    float(raw_gate.mean()),
                    float(raw_gate.std()),
                    float(effective_gate.mean()),
                    float(effective_gate.std()),
                    float(attn.mean()),
                    float(attn.max()),
                ])


def plot_sampling_row(figure, grid, row_index, image_np, attn_records, query_index):
    h, w = image_np.shape[:2]
    for col_index, attn_record in enumerate(attn_records):
        axis = figure.add_subplot(grid[row_index, col_index])
        axis.imshow(image_np)
        axis.set_axis_off()
        sampling_locations = attn_record['sampling_locations'][0, query_index].numpy()
        attention_weights = attn_record['attention_weights'][0, query_index].numpy()
        reference_points = attn_record['reference_points'][0, query_index].numpy()
        if reference_points.ndim == 2:
            ref_xy = reference_points[:, :2].mean(0)
        else:
            ref_xy = reference_points[:2]
        axis.scatter(ref_xy[0] * w, ref_xy[1] * h, c='white', edgecolors='black', s=80, marker='x', linewidths=2)
        heads_mean_locations = sampling_locations.mean(0)
        heads_mean_weights = attention_weights.mean(0)
        for lvl in range(heads_mean_locations.shape[0]):
            pts = heads_mean_locations[lvl]
            ws = heads_mean_weights[lvl]
            color = LEVEL_COLORS[lvl % len(LEVEL_COLORS)]
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            sizes = 20.0 + 200.0 * ws
            axis.scatter(xs, ys, s=sizes, c=color, alpha=0.75, edgecolors='black', linewidths=0.35)
        axis.set_title(f'Layer {col_index}', fontsize=10)


def plot_joint_panel(query_kind, query_index, image_np, attn_records, gate_records, scores, output_path):
    n_layers = len(attn_records)
    if n_layers == 0 or len(gate_records) == 0:
        return
    color = KNOWN_COLOR if query_kind == 'known' else UNKNOWN_COLOR
    layers = list(range(n_layers))
    raw_mean = [record['raw_gate'][0, query_index].mean().item() for record in gate_records]
    eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in gate_records]
    raw_gate_map = np.stack([record['raw_gate'][0, query_index].numpy() for record in gate_records], axis=0)
    eff_gate_map = np.stack([record['effective_gate'][0, query_index].numpy() for record in gate_records], axis=0)

    fig = plt.figure(figsize=(4.0 * n_layers, 12.0))
    grid = GridSpec(3, n_layers, figure=fig, height_ratios=[1.2, 0.55, 0.65])

    plot_sampling_row(fig, grid, 0, image_np, attn_records, query_index)

    curve_ax = fig.add_subplot(grid[1, : max(2, n_layers // 2)])
    curve_ax.plot(layers, raw_mean, marker='o', linewidth=2.0, c=color, label='raw gate mean')
    curve_ax.plot(layers, eff_mean, marker='o', linewidth=2.0, linestyle='--', c='#1E88E5', label='effective gate mean')
    curve_ax.set_xlabel('decoder layer')
    curve_ax.set_ylabel('gate mean')
    curve_ax.set_title('ODQE gate dynamics')
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, fontsize=8)

    attn_ax = fig.add_subplot(grid[1, max(2, n_layers // 2):])
    attn_mean = [record['attention_weights'][0, query_index].mean().item() for record in attn_records]
    attn_max = [record['attention_weights'][0, query_index].max().item() for record in attn_records]
    attn_ax.plot(layers, attn_mean, marker='o', linewidth=2.0, c='#6C757D', label='attention mean')
    attn_ax.plot(layers, attn_max, marker='o', linewidth=2.0, linestyle='--', c='#FB8C00', label='attention max')
    attn_ax.set_xlabel('decoder layer')
    attn_ax.set_ylabel('attention weight')
    attn_ax.set_title('Sampling-weight dynamics')
    attn_ax.grid(alpha=0.2)
    attn_ax.legend(frameon=False, fontsize=8)

    raw_ax = fig.add_subplot(grid[2, : max(2, n_layers // 2)])
    raw_im = raw_ax.imshow(raw_gate_map, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    raw_ax.set_title('Raw gate heatmap')
    raw_ax.set_xlabel('channel')
    raw_ax.set_ylabel('layer')
    raw_ax.set_yticks(range(n_layers))
    raw_ax.set_yticklabels(layers)
    fig.colorbar(raw_im, ax=raw_ax, fraction=0.046, pad=0.04)

    eff_ax = fig.add_subplot(grid[2, max(2, n_layers // 2):])
    eff_im = eff_ax.imshow(eff_gate_map, aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
    eff_ax.set_title('Effective gate heatmap')
    eff_ax.set_xlabel('channel')
    eff_ax.set_ylabel('layer')
    eff_ax.set_yticks(range(n_layers))
    eff_ax.set_yticklabels(layers)
    fig.colorbar(eff_im, ax=eff_ax, fraction=0.046, pad=0.04)

    select_score = scores['known_score'][query_index] if query_kind == 'known' else scores['unknown_score'][query_index]
    fig.suptitle(
        f'Query {query_index} | type={query_kind} | select_score={select_score:.3f} | obj={scores["obj_prob"][query_index]:.3f} | unk={scores["unknown_prob"][query_index]:.3f} | max_known={scores["max_known"][query_index]:.3f}',
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_selected_overview(selected, gate_records, output_path):
    if not selected:
        return
    matrix = []
    row_labels = []
    for query_kind, query_index in selected:
        eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in gate_records]
        matrix.append(eff_mean)
        row_labels.append(f'{query_kind[0].upper()}-{int(query_index):03d}')
    matrix = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(1.2 * matrix.shape[1] + 2.4, 0.5 * matrix.shape[0] + 2.4))
    im = ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0.0, vmax=max(1e-6, float(matrix.max())))
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('selected query')
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(list(range(matrix.shape[1])))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('Selected-query effective gate overview')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser('Standalone joint ODQE gate and query sampling visualization', parents=[get_args_parser()])
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
    Image.fromarray(base_image).save(out_dir / 'image_with_gt.png')
    save_joint_statistics_csv(selected, attn_records, gate_records, scores, out_dir / 'selected_query_joint_statistics.csv')
    plot_selected_overview(selected, gate_records, out_dir / 'selected_query_effective_gate_overview.png')

    for query_kind, query_index in selected:
        output_path = out_dir / f'{query_kind}_query_{int(query_index):03d}_joint_mechanism.png'
        plot_joint_panel(query_kind, int(query_index), base_image, attn_records, gate_records, scores, output_path)

    print(f'Saved joint ODQE/query mechanism visualizations to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
