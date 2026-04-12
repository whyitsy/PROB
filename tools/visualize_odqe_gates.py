import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from main_open_world import get_args_parser, build_datasets
from models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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


class ODQEGateRecorder:
    def __init__(self, model):
        self.model = model
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
        if not name.startswith('gate_mlp.'):
            return False
        return hasattr(module, 'layers') and callable(getattr(module, 'forward', None))

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



def ordered_gate_records(records):
    ordered = sorted(records, key=lambda item: item['layer_id'])
    return [item for item in ordered if item['layer_id'] >= 0]


def save_gate_statistics_csv(selected, ordered_records, scores, output_path):
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'query_kind', 'query_index', 'layer_id', 'decay',
            'raw_gate_mean', 'raw_gate_std', 'raw_gate_max',
            'effective_gate_mean', 'effective_gate_std', 'effective_gate_max',
            'obj_prob', 'unknown_prob', 'max_known', 'known_score', 'unknown_score'
        ])
        for query_kind, query_index in selected:
            for record in ordered_records:
                raw_gate = record['raw_gate'][0, query_index].numpy()
                effective_gate = record['effective_gate'][0, query_index].numpy()
                writer.writerow([
                    query_kind,
                    int(query_index),
                    int(record['layer_id']),
                    float(record['decay']),
                    float(raw_gate.mean()),
                    float(raw_gate.std()),
                    float(raw_gate.max()),
                    float(effective_gate.mean()),
                    float(effective_gate.std()),
                    float(effective_gate.max()),
                    float(scores['obj_prob'][query_index]),
                    float(scores['unknown_prob'][query_index]),
                    float(scores['max_known'][query_index]),
                    float(scores['known_score'][query_index]),
                    float(scores['unknown_score'][query_index]),
                ])


def plot_query_gate_curve(query_kind, query_index, ordered_records, scores, output_path):
    layers = [record['layer_id'] for record in ordered_records]
    raw_mean = [record['raw_gate'][0, query_index].mean().item() for record in ordered_records]
    eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in ordered_records]
    raw_max = [record['raw_gate'][0, query_index].max().item() for record in ordered_records]
    eff_max = [record['effective_gate'][0, query_index].max().item() for record in ordered_records]
    decay = [record['decay'] for record in ordered_records]

    color = KNOWN_COLOR if query_kind == 'known' else UNKNOWN_COLOR
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    axes[0].plot(layers, raw_mean, marker='o', linewidth=2.0, c=color, label='raw gate mean')
    axes[0].plot(layers, eff_mean, marker='o', linewidth=2.0, linestyle='--', c='#1E88E5', label='effective gate mean')
    axes[0].plot(layers, decay, marker='s', linewidth=1.6, linestyle=':', c='#6C757D', label='layer decay')
    axes[0].set_xlabel('decoder layer')
    axes[0].set_ylabel('mean gate value')
    axes[0].set_title('ODQE gate mean by layer')
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(layers, raw_max, marker='o', linewidth=2.0, c=color, label='raw gate max')
    axes[1].plot(layers, eff_max, marker='o', linewidth=2.0, linestyle='--', c='#1E88E5', label='effective gate max')
    axes[1].set_xlabel('decoder layer')
    axes[1].set_ylabel('max gate value')
    axes[1].set_title('ODQE gate peak by layer')
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    select_score = scores['known_score'][query_index] if query_kind == 'known' else scores['unknown_score'][query_index]
    fig.suptitle(
        f'Query {query_index} | type={query_kind} | select_score={select_score:.3f} | obj={scores["obj_prob"][query_index]:.3f} | unk={scores["unknown_prob"][query_index]:.3f} | max_known={scores["max_known"][query_index]:.3f}',
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_query_gate_heatmap(query_kind, query_index, ordered_records, output_path):
    raw_gate = np.stack([record['raw_gate'][0, query_index].numpy() for record in ordered_records], axis=0)
    eff_gate = np.stack([record['effective_gate'][0, query_index].numpy() for record in ordered_records], axis=0)
    layers = [record['layer_id'] for record in ordered_records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    im0 = axes[0].imshow(raw_gate, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    axes[0].set_title(f'Raw ODQE gate heatmap | query {query_index} ({query_kind})')
    axes[0].set_xlabel('channel')
    axes[0].set_ylabel('decoder layer')
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(eff_gate, aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
    axes[1].set_title(f'Effective ODQE gate heatmap | query {query_index} ({query_kind})')
    axes[1].set_xlabel('channel')
    axes[1].set_ylabel('decoder layer')
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_selected_query_overview(selected, ordered_records, output_path):
    if not selected:
        return
    matrix = []
    row_labels = []
    for query_kind, query_index in selected:
        eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in ordered_records]
        matrix.append(eff_mean)
        row_labels.append(f'{query_kind[0].upper()}-{int(query_index):03d}')
    matrix = np.asarray(matrix, dtype=np.float32)
    layers = [record['layer_id'] for record in ordered_records]
    fig, ax = plt.subplots(figsize=(1.4 * len(layers) + 3.2, 0.52 * len(selected) + 2.4))
    im = ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0.0, vmax=max(1e-6, float(matrix.max())))
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('selected query')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('Effective ODQE gate mean for selected queries')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser('Standalone ODQE gate visualization', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--split', default='eval', choices=['train', 'eval'])
    parser.add_argument('--sample_index', default=0, type=int)
    parser.add_argument('--num_known_queries', default=4, type=int)
    parser.add_argument('--num_unknown_queries', default=4, type=int)
    parser.add_argument('--output_subdir', default='infer/odqe_gate', type=str)
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

    recorder = ODQEGateRecorder(model)
    try:
        with torch.no_grad():
            outputs = model(samples)
    finally:
        records = ordered_gate_records(recorder.records)
        recorder.restore()

    if not records:
        raise RuntimeError('No ODQE gate records were captured. Check whether --uod_enable_odqe is enabled and the checkpoint uses gate_mlp.')

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
    save_gate_statistics_csv(selected, records, scores, out_dir / 'selected_query_gate_statistics.csv')
    plot_selected_query_overview(selected, records, out_dir / 'selected_query_gate_mean_heatmap.png')

    for query_kind, query_index in selected:
        plot_query_gate_curve(query_kind, int(query_index), records, scores, out_dir / f'{query_kind}_query_{int(query_index):03d}_gate_curve.png')
        plot_query_gate_heatmap(query_kind, int(query_index), records, out_dir / f'{query_kind}_query_{int(query_index):03d}_gate_heatmap.png')

    print(f'Saved ODQE gate visualizations to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
