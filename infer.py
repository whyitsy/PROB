import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
from main_open_world import get_args_parser
from models import build_model
from util import box_ops
from util.log import setup_logging

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _load_checkpoint_args(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    parser_defaults = vars(get_args_parser().parse_args([]))
    saved_args = checkpoint.get('args', {}) or {}
    parser_defaults.update(saved_args)
    parser_defaults['device'] = device
    return checkpoint, argparse.Namespace(**parser_defaults)


def _collect_input_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return [path for path in sorted(input_path.rglob('*')) if path.suffix.lower() in IMAGE_EXTENSIONS]


def _prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    target = {
        'boxes': torch.zeros((0, 4), dtype=torch.float32),
        'labels': torch.zeros((0,), dtype=torch.int64),
        'area': torch.zeros((0,), dtype=torch.float32),
        'iscrowd': torch.zeros((0,), dtype=torch.uint8),
        'orig_size': torch.as_tensor([height, width]),
        'size': torch.as_tensor([height, width]),
    }
    transform = make_coco_transforms('test')[-1]
    image_tensor, target = transform(image, target)
    return image, image_tensor, target


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        iou = box_ops.box_iou(boxes[current].unsqueeze(0), boxes[order[1:]])[0].squeeze(0)
        order = order[1:][iou < float(iou_threshold)]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long)


def _is_valid_geometry_xyxy(box, width, height, min_area_ratio=0.002, min_side_ratio=0.03, max_aspect_ratio=5.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    min_area = float(width * height) * float(min_area_ratio)
    min_side = min(float(width), float(height)) * float(min_side_ratio)
    if area < min_area or min(bw, bh) < min_side:
        return False
    aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
    return aspect <= float(max_aspect_ratio)


def _display_score(raw_score, label, unknown_label, unknown_score_scale):
    raw_score = float(raw_score)
    if int(label) == int(unknown_label):
        return raw_score / max(float(unknown_score_scale), 1e-6)
    return raw_score


def _build_detection_records(predictions, class_names, unknown_label, unknown_score_scale):
    boxes = predictions['boxes'].detach().cpu().tolist()
    labels = predictions['labels'].detach().cpu().tolist()
    raw_scores = predictions['scores'].detach().cpu().tolist()

    detections = []
    for box_xyxy, label, raw_score in zip(boxes, labels, raw_scores):
        label = int(label)
        raw_score = float(raw_score)
        is_unknown = bool(label == int(unknown_label))
        label_name = 'unknown' if is_unknown else (class_names[label] if 0 <= label < len(class_names) else f'class_{label}')
        detections.append(
            {
                'label': label,
                'label_name': label_name,
                'box_xyxy': [float(v) for v in box_xyxy],
                'raw_score': raw_score,
                'display_score': _display_score(raw_score, label, unknown_label, unknown_score_scale),
                'is_unknown': is_unknown,
            }
        )
    return detections


def _filter_detections(detections, image_size, args, unified_across_labels=False):
    if not detections:
        return []

    width, height = image_size
    filtered = []
    for det in detections:
        score = float(det['display_score'])
        threshold = float(args.unknown_score_threshold) if det['is_unknown'] else float(args.known_score_threshold)
        if score < threshold:
            continue
        if not _is_valid_geometry_xyxy(
            det['box_xyxy'],
            width,
            height,
            min_area_ratio=args.min_box_area_ratio,
            min_side_ratio=args.min_box_side_ratio,
            max_aspect_ratio=args.max_box_aspect_ratio,
        ):
            continue
        filtered.append(det)

    if not filtered:
        return []

    if unified_across_labels:
        boxes_t = torch.as_tensor([det['box_xyxy'] for det in filtered], dtype=torch.float32)
        scores_t = torch.as_tensor([det['display_score'] for det in filtered], dtype=torch.float32)
        kept = _nms_xyxy(boxes_t, scores_t, iou_threshold=args.nms_iou_threshold)
        final_keep = [filtered[index] for index in kept.tolist()]
        final_keep.sort(key=lambda item: item['display_score'], reverse=True)
        return final_keep

    final_keep = []
    for select_unknown in [False, True]:
        subset = [det for det in filtered if det['is_unknown'] == select_unknown]
        if not subset:
            continue
        boxes_t = torch.as_tensor([det['box_xyxy'] for det in subset], dtype=torch.float32)
        scores_t = torch.as_tensor([det['raw_score'] for det in subset], dtype=torch.float32)
        kept = _nms_xyxy(boxes_t, scores_t, iou_threshold=args.nms_iou_threshold)
        final_keep.extend(subset[index] for index in kept.tolist())

    final_keep.sort(key=lambda item: item['raw_score'], reverse=True)
    return final_keep


def _draw_detections(image: Image.Image, detections):
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det['box_xyxy']]
        is_unknown = bool(det['is_unknown'])
        color = 'red' if is_unknown else 'lime'
        name = 'unknown' if is_unknown else det['label_name']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0.0, y1 - 12)), f"{name}:{float(det['display_score']):.2f}", fill=color)
    return rendered


def _build_payload(image_path, image_size, args, unknown_label, raw_count, detections, unified_detections):
    summary = {
        'num_raw': int(raw_count),
        'num_exported': int(len(detections)),
        'num_unified_exported': int(len(unified_detections)),
        'num_known_exported': int(sum(0 if det['is_unknown'] else 1 for det in detections)),
        'num_unknown_exported': int(sum(1 if det['is_unknown'] else 0 for det in detections)),
    }
    return {
        'image': str(image_path),
        'image_size': {'width': int(image_size[0]), 'height': int(image_size[1])},
        'unknown_label': int(unknown_label),
        'inference_filter': {
            'known_score_threshold': float(args.known_score_threshold),
            'unknown_score_threshold': float(args.unknown_score_threshold),
            'nms_iou_threshold': float(args.nms_iou_threshold),
            'min_box_area_ratio': float(args.min_box_area_ratio),
            'min_box_side_ratio': float(args.min_box_side_ratio),
            'max_box_aspect_ratio': float(args.max_box_aspect_ratio),
        },
        'summary': summary,
        'predictions': detections,
        'unified_predictions': unified_detections,
    }


def run_inference(args):
    output_dir = Path(args.output_dir)
    json_dir = output_dir / 'json'
    image_dir = output_dir / 'images'
    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output=str(output_dir), distributed_rank=0, abbrev_name='PROB-Infer')

    checkpoint, model_args = _load_checkpoint_args(args.checkpoint, args.device)
    vars(model_args).update({k: v for k, v in vars(args).items() if v is not None})
    model_args.uod_postprocess_unknown_scale = args.unknown_score_scale

    model, _, postprocessors, _ = build_model(model_args, mode=model_args.model_type)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(torch.device(args.device))
    model.eval()

    image_paths = _collect_input_images(args.input)
    logging.info('Found %s image(s) for inference', len(image_paths))

    class_names = list(VOC_COCO_CLASS_NAMES[model_args.dataset])
    unknown_label = int(model_args.num_classes - 1)
    unknown_score_scale = float(args.unknown_score_scale)

    with torch.no_grad():
        for image_path in image_paths:
            original_image, image_tensor, target = _prepare_image(image_path)
            image_tensor = image_tensor.to(torch.device(args.device))

            outputs = model([image_tensor])
            target_sizes = target['orig_size'].unsqueeze(0).to(torch.device(args.device))
            predictions = postprocessors['bbox'](outputs, target_sizes)[0]

            raw_detections = _build_detection_records(predictions, class_names, unknown_label, unknown_score_scale)
            export_detections = _filter_detections(raw_detections, image_size=original_image.size, args=args, unified_across_labels=False)
            unified_export_detections = _filter_detections(raw_detections, image_size=original_image.size, args=args, unified_across_labels=True)

            payload = _build_payload(
                image_path=image_path,
                image_size=original_image.size,
                args=args,
                unknown_label=unknown_label,
                raw_count=len(raw_detections),
                detections=export_detections,
                unified_detections=unified_export_detections,
            )
            (json_dir / f'{image_path.stem}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

            rendered = _draw_detections(original_image, export_detections)
            rendered.save(image_dir / f'{image_path.stem}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Standalone inference for PROB / UOD checkpoints')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--input', required=True, type=str, help='single image path or directory of images')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model_type', default='uod', type=str)
    parser.add_argument('--known_score_threshold', default=0.35, type=float)
    parser.add_argument('--unknown_score_threshold', default=0.20, type=float)
    parser.add_argument('--nms_iou_threshold', default=0.5, type=float)
    parser.add_argument('--min_box_area_ratio', default=0.002, type=float)
    parser.add_argument('--min_box_side_ratio', default=0.03, type=float)
    parser.add_argument('--max_box_aspect_ratio', default=5.0, type=float)
    parser.add_argument('--unknown_score_scale', default=10.0, type=float)
    run_inference(parser.parse_args())
