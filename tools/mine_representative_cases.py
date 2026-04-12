import argparse
import csv
import json
from pathlib import Path
import sys

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CATEGORY_COLORS = {
    'known': (0, 166, 90),
    'unknown': (216, 27, 96),
    'odqe_salient': (30, 136, 229),
}


def to_numpy_image(image_tensor, target_hw=None):
    image = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    if target_hw is not None:
        height, width = int(target_hw[0]), int(target_hw[1])
        image = image[:height, :width]
    return (image * 255).astype(np.uint8)


def _energy_to_prob(energy, temperature):
    return torch.exp(-temperature * energy.detach()).clamp(min=1e-6, max=1.0)


def compute_final_scores(outputs, args, invalid_cls_logits):
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temp = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim

    pred_logits = outputs['pred_logits'].detach()
    pred_boxes = outputs['pred_boxes'].detach()
    pred_obj = outputs.get('pred_obj')
    pred_known = outputs.get('pred_known')

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

    return {
        'boxes': pred_boxes[0].cpu().numpy(),
        'obj_prob': obj_prob[0].cpu().numpy(),
        'known_prob': known_prob[0].cpu().numpy(),
        'unknown_prob': unknown_prob[0].cpu().numpy(),
        'max_known': max_known[0].cpu().numpy(),
        'argmax_known': argmax_known[0].cpu().numpy(),
        'known_score': known_score[0].cpu().numpy(),
        'unknown_score': unknown_score[0].cpu().numpy(),
    }


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

    def clear(self):
        self.records.clear()

    def restore(self):
        for module, original_forward in self.patches:
            module.forward = original_forward


def ordered_gate_records(records):
    ordered = sorted(records, key=lambda item: item['layer_id'])
    return [item for item in ordered if item['layer_id'] >= 0]


def make_case_entry(category, sample_index, target, scores, query_index, category_score, gate_records):
    image_id = int(target['image_id'].item()) if 'image_id' in target else int(sample_index)
    gate_mean = None
    gate_peak = None
    gate_depth_delta = None
    if gate_records:
        per_layer_gate = [float(record['effective_gate'][0, query_index].mean().item()) for record in gate_records]
        gate_mean = float(np.mean(per_layer_gate))
        gate_peak = float(np.max(per_layer_gate))
        gate_depth_delta = float(per_layer_gate[-1] - per_layer_gate[0]) if len(per_layer_gate) >= 2 else 0.0
    return {
        'category': category,
        'sample_index': int(sample_index),
        'image_id': image_id,
        'query_index': int(query_index),
        'category_score': float(category_score),
        'obj_prob': float(scores['obj_prob'][query_index]),
        'known_prob': float(scores['known_prob'][query_index]),
        'unknown_prob': float(scores['unknown_prob'][query_index]),
        'max_known': float(scores['max_known'][query_index]),
        'known_score': float(scores['known_score'][query_index]),
        'unknown_score': float(scores['unknown_score'][query_index]),
        'argmax_known': int(scores['argmax_known'][query_index]),
        'box_cxcywh': [float(v) for v in scores['boxes'][query_index].tolist()],
        'gate_mean': gate_mean,
        'gate_peak': gate_peak,
        'gate_depth_delta': gate_depth_delta,
    }


def update_top_cases(top_cases, entry, top_k):
    category = entry['category']
    top_cases[category].append(entry)
    top_cases[category] = sorted(top_cases[category], key=lambda item: item['category_score'], reverse=True)[:top_k]


def draw_case_tile(dataset, entry, category, tile_size=320):
    image, target = dataset[entry['sample_index']]
    image_hw = target['size'].tolist() if 'size' in target else image.shape[-2:]
    image_np = to_numpy_image(image, image_hw)
    h, w = image_np.shape[:2]
    box = np.array(entry['box_cxcywh'], dtype=np.float32)[None, :]
    box_xyxy = box_ops.box_cxcywh_to_xyxy(torch.from_numpy(box)).numpy()[0]
    box_xyxy[[0, 2]] *= w
    box_xyxy[[1, 3]] *= h
    pil = Image.fromarray(image_np).convert('RGB').resize((tile_size, tile_size))
    draw = ImageDraw.Draw(pil)
    sx = tile_size / max(float(w), 1.0)
    sy = tile_size / max(float(h), 1.0)
    x1, y1, x2, y2 = box_xyxy
    color = CATEGORY_COLORS[category]
    draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=color, width=3)
    text_lines = [
        f's{entry["sample_index"]} / img {entry["image_id"]}',
        f'q{entry["query_index"]} | score {entry["category_score"]:.3f}',
        f'obj {entry["obj_prob"]:.3f} | unk {entry["unknown_prob"]:.3f}',
        f'mk {entry["max_known"]:.3f} | ks {entry["known_score"]:.3f}',
    ]
    if entry.get('gate_mean') is not None:
        text_lines.append(f'gate {entry["gate_mean"]:.3f} | d {entry["gate_depth_delta"]:.3f}')
    text = '\n'.join(text_lines)
    draw.multiline_text((8, 8), text, fill=(255, 255, 255), spacing=2)
    return pil


def save_contact_sheet(dataset, entries, category, output_path, tile_size=320, cols=3):
    if not entries:
        return
    rows = int(np.ceil(len(entries) / cols))
    canvas = Image.new('RGB', (cols * tile_size, rows * tile_size), color=(25, 25, 25))
    for idx, entry in enumerate(entries):
        tile = draw_case_tile(dataset, entry, category, tile_size=tile_size)
        row = idx // cols
        col = idx % cols
        canvas.paste(tile, (col * tile_size, row * tile_size))
    canvas.save(output_path)


def save_category_csv(entries, output_path):
    if not entries:
        return
    fieldnames = [
        'category', 'sample_index', 'image_id', 'query_index', 'category_score',
        'obj_prob', 'known_prob', 'unknown_prob', 'max_known',
        'known_score', 'unknown_score', 'argmax_known',
        'box_cxcywh', 'gate_mean', 'gate_peak', 'gate_depth_delta'
    ]
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = entry.copy()
            row['box_cxcywh'] = json.dumps(row['box_cxcywh'])
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser('Standalone representative case mining utility', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--split', default='eval', choices=['train', 'eval'])
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--max_samples', default=300, type=int)
    parser.add_argument('--top_k', default=9, type=int)
    parser.add_argument('--output_subdir', default='infer/representative_cases', type=str)
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
    total = len(dataset)
    end_index = min(total, parsed_args.start_index + parsed_args.max_samples)

    gate_recorder = ODQEGateRecorder(model)
    top_cases = {
        'known': [],
        'unknown': [],
        'odqe_salient': [],
    }

    invalid_cls_logits = list(range(parsed_args.PREV_INTRODUCED_CLS + parsed_args.CUR_INTRODUCED_CLS, parsed_args.num_classes - 1))

    for sample_index in range(parsed_args.start_index, end_index):
        image, target = dataset[sample_index]
        samples = nested_tensor_from_tensor_list([image]).to(device)
        gate_recorder.clear()
        with torch.no_grad():
            outputs = model(samples)
        gate_records = ordered_gate_records(gate_recorder.records)
        scores = compute_final_scores(outputs, parsed_args, invalid_cls_logits)

        known_query = int(np.argmax(scores['known_score']))
        unknown_query = int(np.argmax(scores['unknown_score']))
        update_top_cases(
            top_cases,
            make_case_entry('known', sample_index, target, scores, known_query, scores['known_score'][known_query], gate_records),
            parsed_args.top_k,
        )
        update_top_cases(
            top_cases,
            make_case_entry('unknown', sample_index, target, scores, unknown_query, scores['unknown_score'][unknown_query], gate_records),
            parsed_args.top_k,
        )

        if gate_records:
            odqe_signal = []
            for query_index in range(len(scores['obj_prob'])):
                eff_gate_mean = float(np.mean([record['effective_gate'][0, query_index].mean().item() for record in gate_records]))
                signal = eff_gate_mean * float(scores['unknown_prob'][query_index]) * float(scores['obj_prob'][query_index])
                odqe_signal.append(signal)
            odqe_query = int(np.argmax(np.asarray(odqe_signal)))
            update_top_cases(
                top_cases,
                make_case_entry('odqe_salient', sample_index, target, scores, odqe_query, odqe_signal[odqe_query], gate_records),
                parsed_args.top_k,
            )

    gate_recorder.restore()

    out_dir = Path(parsed_args.output_dir) / parsed_args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'split': parsed_args.split,
        'start_index': int(parsed_args.start_index),
        'end_index': int(end_index),
        'top_k': int(parsed_args.top_k),
        'categories': top_cases,
    }
    with open(out_dir / 'representative_case_manifest.json', 'w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    for category, entries in top_cases.items():
        save_category_csv(entries, out_dir / f'{category}_top_cases.csv')
        save_contact_sheet(dataset, entries, category, out_dir / f'{category}_contact_sheet.png')

    print(f'Saved representative case mining results to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
