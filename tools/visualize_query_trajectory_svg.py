import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_open_world import get_args_parser, build_datasets
from models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from tools.figure_svg_utils import save_svg_figure, save_svg_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LAYER_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00897B', '#6D4C41', '#3949AB']
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


def _boxes_to_abs_xyxy(boxes, image_hw):
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().cpu())
    scale = torch.tensor([int(image_hw[1]), int(image_hw[0]), int(image_hw[1]), int(image_hw[0])], dtype=boxes_xyxy.dtype)
    return (boxes_xyxy * scale).numpy()


def _energy_to_prob(energy, temperature):
    return torch.exp(-temperature * energy.detach()).clamp(min=1e-6, max=1.0)


def compute_per_layer_scores(outputs, args, invalid_cls_logits):
    layers = []
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temp = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim

    aux_outputs = list(outputs.get('aux_outputs', []))
    all_outputs = aux_outputs + [{
        'pred_logits': outputs['pred_logits'],
        'pred_boxes': outputs['pred_boxes'],
        'pred_obj': outputs.get('pred_obj'),
        'pred_known': outputs.get('pred_known'),
    }]

    for layer_idx, out in enumerate(all_outputs):
        pred_logits = out['pred_logits'].detach()
        pred_boxes = out['pred_boxes'].detach()
        pred_obj = out.get('pred_obj')
        pred_known = out.get('pred_known')

        class_prob = pred_logits.sigmoid().clone()
        if invalid_cls_logits:
            class_prob[:, :, invalid_cls_logits] = 0.0
        if class_prob.shape[-1] > 0:
            class_prob[:, :, -1] = 0.0
        max_known = class_prob.max(-1).values
        argmax_known = class_prob.argmax(-1)

        if pred_obj is not None:
            obj_prob = _energy_to_prob(pred_obj, obj_temp)
        else:
            obj_prob = torch.ones_like(max_known)

        if pred_known is not None:
            known_prob = _energy_to_prob(pred_known, known_temp)
            unknown_prob = (1.0 - known_prob).clamp(min=0.0, max=1.0)
        else:
            known_prob = torch.ones_like(obj_prob)
            unknown_prob = torch.zeros_like(obj_prob)

        known_score = obj_prob * known_prob * max_known
        unknown_score = obj_prob * unknown_prob * float(getattr(args, 'uod_postprocess_unknown_scale', 20.0))
        layers.append({
            'layer_id': layer_idx,
            'boxes': pred_boxes[0].cpu().numpy(),
            'obj_prob': obj_prob[0].cpu().numpy(),
            'known_prob': known_prob[0].cpu().numpy(),
            'unknown_prob': unknown_prob[0].cpu().numpy(),
            'max_known': max_known[0].cpu().numpy(),
            'argmax_known': argmax_known[0].cpu().numpy(),
            'known_score': known_score[0].cpu().numpy(),
            'unknown_score': unknown_score[0].cpu().numpy(),
        })
    return layers


def select_queries(final_layer_scores, num_known, num_unknown):
    known_indices = np.argsort(-final_layer_scores['known_score'])[:num_known].tolist()
    unknown_indices = np.argsort(-final_layer_scores['unknown_score'])[:num_unknown].tolist()
    selected = [('known', idx) for idx in known_indices] + [('unknown', idx) for idx in unknown_indices if idx not in known_indices]
    return selected


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


def ordered_gate_records(records):
    ordered = sorted(records, key=lambda item: item['layer_id'])
    return [item for item in ordered if item['layer_id'] >= 0]


def save_trajectory_csv(selected, layer_scores, gate_records, output_path):
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'query_kind', 'query_index', 'layer_id',
            'cx', 'cy', 'w', 'h',
            'obj_prob', 'known_prob', 'unknown_prob', 'max_known', 'argmax_known',
            'known_score', 'unknown_score',
            'raw_gate_mean', 'effective_gate_mean', 'decay'
        ])
        for query_kind, query_index in selected:
            for layer_id, layer in enumerate(layer_scores):
                box = layer['boxes'][query_index]
                gate_mean = None
                effective_gate_mean = None
                decay = None
                if layer_id < len(gate_records):
                    gate_mean = float(gate_records[layer_id]['raw_gate'][0, query_index].mean().item())
                    effective_gate_mean = float(gate_records[layer_id]['effective_gate'][0, query_index].mean().item())
                    decay = float(gate_records[layer_id]['decay'])
                writer.writerow([
                    query_kind,
                    int(query_index),
                    int(layer_id),
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(layer['obj_prob'][query_index]),
                    float(layer['known_prob'][query_index]),
                    float(layer['unknown_prob'][query_index]),
                    float(layer['max_known'][query_index]),
                    int(layer['argmax_known'][query_index]),
                    float(layer['known_score'][query_index]),
                    float(layer['unknown_score'][query_index]),
                    gate_mean,
                    effective_gate_mean,
                    decay,
                ])


def plot_image_trajectory(ax, image_np, abs_boxes_per_layer, query_kind):
    ax.imshow(image_np)
    ax.set_axis_off()
    centers_x = []
    centers_y = []
    for layer_id, box in enumerate(abs_boxes_per_layer):
        x1, y1, x2, y2 = box
        color = LAYER_COLORS[layer_id % len(LAYER_COLORS)]
        rect = Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), fill=False, linewidth=2.0, edgecolor=color, alpha=0.95)
        ax.add_patch(rect)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        centers_x.append(cx)
        centers_y.append(cy)
        ax.scatter(cx, cy, s=42, c=color, edgecolors='black', linewidths=0.5)
        ax.text(cx + 2, cy + 2, str(layer_id), color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.60, pad=1))
    if len(centers_x) >= 2:
        ax.plot(centers_x, centers_y, linewidth=2.0, color=KNOWN_COLOR if query_kind == 'known' else UNKNOWN_COLOR, alpha=0.9)
    ax.set_title('Query box trajectory on image')


def plot_box_geometry(ax, abs_boxes_per_layer):
    layer_ids = list(range(len(abs_boxes_per_layer)))
    widths = [max(1e-6, box[2] - box[0]) for box in abs_boxes_per_layer]
    heights = [max(1e-6, box[3] - box[1]) for box in abs_boxes_per_layer]
    areas = [w * h for w, h in zip(widths, heights)]
    centers_x = [0.5 * (box[0] + box[2]) for box in abs_boxes_per_layer]
    centers_y = [0.5 * (box[1] + box[3]) for box in abs_boxes_per_layer]
    ax.plot(layer_ids, centers_x, marker='o', linewidth=2.0, label='center x')
    ax.plot(layer_ids, centers_y, marker='o', linewidth=2.0, label='center y')
    ax.plot(layer_ids, widths, marker='o', linewidth=1.8, linestyle='--', label='width')
    ax.plot(layer_ids, heights, marker='o', linewidth=1.8, linestyle='--', label='height')
    ax.plot(layer_ids, np.sqrt(np.asarray(areas)), marker='o', linewidth=1.8, linestyle=':', label='sqrt(area)')
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('pixels / proxy scale')
    ax.set_title('Box geometry trajectory')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))


def plot_score_trajectory(ax, layer_scores, query_index):
    layer_ids = list(range(len(layer_scores)))
    obj = [layer['obj_prob'][query_index] for layer in layer_scores]
    unk = [layer['unknown_prob'][query_index] for layer in layer_scores]
    max_known = [layer['max_known'][query_index] for layer in layer_scores]
    known_score = [layer['known_score'][query_index] for layer in layer_scores]
    unknown_score = [layer['unknown_score'][query_index] for layer in layer_scores]
    ax.plot(layer_ids, obj, marker='o', linewidth=2.0, label='obj prob')
    ax.plot(layer_ids, unk, marker='o', linewidth=2.0, label='unknown prob')
    ax.plot(layer_ids, max_known, marker='o', linewidth=2.0, label='max known prob')
    ax.plot(layer_ids, known_score, marker='o', linewidth=1.8, linestyle='--', label='known score')
    ax.plot(layer_ids, unknown_score, marker='o', linewidth=1.8, linestyle='--', label='unknown score')
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('score')
    ax.set_ylim(0.0, max(1.05, 1.05 * max(known_score + unknown_score + obj + unk + max_known)))
    ax.set_title('Score trajectory')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))


def plot_gate_trajectory(ax, gate_records, query_index):
    layer_ids = list(range(len(gate_records)))
    raw_mean = [record['raw_gate'][0, query_index].mean().item() for record in gate_records]
    eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in gate_records]
    raw_max = [record['raw_gate'][0, query_index].max().item() for record in gate_records]
    decay = [record['decay'] for record in gate_records]
    ax.plot(layer_ids, raw_mean, marker='o', linewidth=2.0, label='raw gate mean')
    ax.plot(layer_ids, eff_mean, marker='o', linewidth=2.0, linestyle='--', label='effective gate mean')
    ax.plot(layer_ids, raw_max, marker='o', linewidth=1.8, linestyle=':', label='raw gate max')
    ax.plot(layer_ids, decay, marker='s', linewidth=1.6, linestyle='-.', label='layer decay')
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('gate')
    ax.set_title('ODQE gate trajectory')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18))


def plot_layer_class_trace(ax, layer_scores, query_index):
    layer_ids = list(range(len(layer_scores)))
    labels = [int(layer['argmax_known'][query_index]) for layer in layer_scores]
    vals = [float(layer['max_known'][query_index]) for layer in layer_scores]
    ax.plot(layer_ids, vals, marker='o', linewidth=2.0)
    for x, y, label in zip(layer_ids, vals, labels):
        ax.annotate(f'c{label}', (x, y), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('max known prob')
    ax.set_title('Predicted known-class trace')
    ax.grid(alpha=0.2)
    ax.margins(y=0.18)


def plot_summary(ax, query_kind, query_index, layer_scores, gate_records):
    ax.axis('off')
    final = layer_scores[-1]
    summary_lines = [
        f'query type: {query_kind}',
        f'query index: {int(query_index)}',
        f'final obj: {float(final["obj_prob"][query_index]):.3f}',
        f'final unk: {float(final["unknown_prob"][query_index]):.3f}',
        f'final max_known: {float(final["max_known"][query_index]):.3f}',
        f'final known_score: {float(final["known_score"][query_index]):.3f}',
        f'final unknown_score: {float(final["unknown_score"][query_index]):.3f}',
        f'final known class: {int(final["argmax_known"][query_index])}',
    ]
    if gate_records:
        summary_lines += [
            f'final raw gate mean: {float(gate_records[-1]["raw_gate"][0, query_index].mean().item()):.3f}',
            f'final effective gate mean: {float(gate_records[-1]["effective_gate"][0, query_index].mean().item()):.3f}',
        ]
    ax.text(0.02, 0.98, '\n'.join(summary_lines), va='top', ha='left', fontsize=10, family='monospace', linespacing=1.5)
    ax.set_title('Trajectory summary')


def plot_query_trajectory_panel(query_kind, query_index, image_np, layer_scores, gate_records, image_hw, output_path):
    abs_boxes_per_layer = []
    for layer in layer_scores:
        box_abs = _boxes_to_abs_xyxy(torch.from_numpy(layer['boxes'][query_index:query_index + 1]), image_hw)[0]
        abs_boxes_per_layer.append(box_abs)

    if gate_records:
        mosaic = [
            ['image', 'geometry', 'score'],
            ['gate', 'class', 'summary'],
        ]
        fig = plt.figure(figsize=(16.5, 9.3), constrained_layout=True)
    else:
        mosaic = [
            ['image', 'geometry', 'score'],
            ['class', 'class', 'summary'],
        ]
        fig = plt.figure(figsize=(16.5, 8.4), constrained_layout=True)

    axes = fig.subplot_mosaic(mosaic)
    plot_image_trajectory(axes['image'], image_np, abs_boxes_per_layer, query_kind)
    plot_box_geometry(axes['geometry'], abs_boxes_per_layer)
    plot_score_trajectory(axes['score'], layer_scores, query_index)
    plot_layer_class_trace(axes['class'], layer_scores, query_index)
    plot_summary(axes['summary'], query_kind, query_index, layer_scores, gate_records)
    if gate_records:
        plot_gate_trajectory(axes['gate'], gate_records, query_index)

    fig.suptitle(f'Cross-layer trajectory of query {int(query_index)} ({query_kind})', y=1.02, fontsize=15)
    save_svg_figure(fig, output_path, pad_inches=0.08)


def plot_selected_overview(selected, layer_scores, gate_records, output_path):
    if not selected:
        return
    layer_ids = list(range(len(layer_scores)))
    matrix = []
    row_labels = []
    for query_kind, query_index in selected:
        row = [float(layer_scores[layer_id]['unknown_prob'][query_index]) for layer_id in layer_ids]
        if gate_records:
            row = [row[layer_id] * float(gate_records[layer_id]['effective_gate'][0, query_index].mean().item()) for layer_id in layer_ids]
        matrix.append(row)
        row_labels.append(f'{query_kind[0].upper()}-{int(query_index):03d}')
    matrix = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(1.25 * len(layer_ids) + 2.8, 0.6 * len(row_labels) + 2.6), constrained_layout=True)
    im = ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0.0, vmax=max(1e-6, float(matrix.max())))
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('selected query')
    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('Unknownness × effective-gate overview')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_svg_figure(fig, output_path, pad_inches=0.04)


def build_parser():
    parser = argparse.ArgumentParser('SVG-first cross-layer query trajectory visualization', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--split', default='eval', choices=['train', 'eval'])
    parser.add_argument('--sample_index', default=0, type=int)
    parser.add_argument('--num_known_queries', default=3, type=int)
    parser.add_argument('--num_unknown_queries', default=3, type=int)
    parser.add_argument('--output_subdir', default='infer/query_trajectory', type=str)
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

    gate_recorder = ODQEGateRecorder(model)
    try:
        with torch.no_grad():
            outputs = model(samples)
    finally:
        gate_records = ordered_gate_records(gate_recorder.records)
        gate_recorder.restore()

    invalid_cls_logits = list(range(parsed_args.PREV_INTRODUCED_CLS + parsed_args.CUR_INTRODUCED_CLS, parsed_args.num_classes - 1))
    layer_scores = compute_per_layer_scores(outputs, parsed_args, invalid_cls_logits)
    selected = select_queries(layer_scores[-1], parsed_args.num_known_queries, parsed_args.num_unknown_queries)

    image_hw = target['size'].tolist() if 'size' in target else image.shape[-2:]
    image_np = to_numpy_image(image, image_hw)
    unknown_label = int(parsed_args.num_classes - 1)
    base_image = draw_gt_boxes(image_np, target, unknown_label)

    image_id = int(target['image_id'].item()) if 'image_id' in target else parsed_args.sample_index
    out_dir = Path(parsed_args.output_dir) / parsed_args.output_subdir / f'image_{image_id:012d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_svg_image(base_image, out_dir / 'image_with_gt.svg')
    save_trajectory_csv(selected, layer_scores, gate_records, out_dir / 'selected_query_trajectory.csv')
    plot_selected_overview(selected, layer_scores, gate_records, out_dir / 'selected_query_unknown_gate_overview.svg')

    for query_kind, query_index in selected:
        plot_query_trajectory_panel(
            query_kind,
            int(query_index),
            base_image,
            layer_scores,
            gate_records,
            image_hw,
            out_dir / f'{query_kind}_query_{int(query_index):03d}_trajectory.svg'
        )

    print(f'Saved SVG cross-layer query trajectories to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
