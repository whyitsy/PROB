import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
from models import build_model
from util import box_ops
from main_open_world import get_args_parser
from util.log import setup_logging

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def build_viz_cfg():
    return {
        'display_known_score_thresh': 0.35,
        'display_unknown_score_thresh': 0.20,
        'display_nms_iou': 0.5,
        'display_apply_geometry_filter': True,
        'display_min_area_ratio': 0.002,
        'display_min_side_ratio': 0.03,
        'display_max_aspect_ratio': 5.0,
    }


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


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.6) -> torch.Tensor:
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


def _filter_export_detections(
    detections,
    image_size,
    known_score_thresh,
    unknown_score_thresh,
    nms_iou,
    min_area_ratio,
    min_side_ratio,
    max_aspect_ratio,
    unified_across_labels=False,
):
    if not detections:
        return []

    width, height = image_size
    filtered = []
    for det in detections:
        score = float(det['display_score'])
        threshold = float(unknown_score_thresh) if det['is_unknown'] else float(known_score_thresh)
        if score < threshold:
            continue
        if not _is_valid_geometry_xyxy(
            det['box_xyxy'],
            width,
            height,
            min_area_ratio=min_area_ratio,
            min_side_ratio=min_side_ratio,
            max_aspect_ratio=max_aspect_ratio,
        ):
            continue
        filtered.append(det)

    if not filtered:
        return []

    if unified_across_labels:
        boxes_t = torch.as_tensor([det['box_xyxy'] for det in filtered], dtype=torch.float32)
        display_scores_t = torch.as_tensor([det['display_score'] for det in filtered], dtype=torch.float32)
        kept = _nms_xyxy(boxes_t, display_scores_t, iou_threshold=nms_iou)
        final_keep = [filtered[index] for index in kept.tolist()]
        final_keep.sort(key=lambda item: item['display_score'], reverse=True)
        return final_keep

    final_keep = []
    for select_unknown in [False, True]:
        subset = [det for det in filtered if det['is_unknown'] == select_unknown]
        if not subset:
            continue
        boxes_t = torch.as_tensor([det['box_xyxy'] for det in subset], dtype=torch.float32)
        raw_scores_t = torch.as_tensor([det['raw_score'] for det in subset], dtype=torch.float32)
        kept = _nms_xyxy(boxes_t, raw_scores_t, iou_threshold=nms_iou)
        for index in kept.tolist():
            final_keep.append(subset[index])

    final_keep.sort(key=lambda item: item['raw_score'], reverse=True)
    return final_keep


def _prepare_display_arrays(raw_detections, image_size, unknown_label, unknown_score_scale, viz_cfg):
    filtered = _filter_export_detections(
        raw_detections,
        image_size=image_size,
        known_score_thresh=float(viz_cfg['display_known_score_thresh']),
        unknown_score_thresh=float(viz_cfg['display_unknown_score_thresh']),
        nms_iou=float(viz_cfg['display_nms_iou']),
        min_area_ratio=float(viz_cfg['display_min_area_ratio']),
        min_side_ratio=float(viz_cfg['display_min_side_ratio']),
        max_aspect_ratio=float(viz_cfg['display_max_aspect_ratio']),
        unified_across_labels=False,
    )
    if not filtered:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    boxes = np.asarray([det['box_xyxy'] for det in filtered], dtype=np.float32)
    labels = np.asarray([det['label'] for det in filtered], dtype=np.int64)
    display_scores = np.asarray([det['display_score'] for det in filtered], dtype=np.float32)
    return boxes, labels, display_scores


def _prepare_unified_display_arrays(raw_detections, image_size, unknown_label, unknown_score_scale, viz_cfg):
    if not raw_detections:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    width, height = image_size
    kept = []
    for det in raw_detections:
        raw_score = float(det['raw_score'])
        label = int(det['label'])
        threshold = float(viz_cfg['display_unknown_score_thresh']) if label == int(unknown_label) else float(viz_cfg['display_known_score_thresh'])
        if raw_score < threshold:
            continue
        if viz_cfg['display_apply_geometry_filter'] and not _is_valid_geometry_xyxy(
            det['box_xyxy'],
            width,
            height,
            min_area_ratio=float(viz_cfg['display_min_area_ratio']),
            min_side_ratio=float(viz_cfg['display_min_side_ratio']),
            max_aspect_ratio=float(viz_cfg['display_max_aspect_ratio']),
        ):
            continue
        kept.append(det)

    if not kept:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    boxes = np.asarray([det['box_xyxy'] for det in kept], dtype=np.float32)
    labels = np.asarray([det['label'] for det in kept], dtype=np.int64)
    display_scores = np.asarray([det['display_score'] for det in kept], dtype=np.float32)
    kept_indices = _nms_xyxy(torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(display_scores, dtype=torch.float32), iou_threshold=float(viz_cfg['display_nms_iou']))
    kept_indices = kept_indices.detach().cpu().numpy().astype(np.int64) if kept_indices.numel() > 0 else np.zeros((0,), dtype=np.int64)
    if kept_indices.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    kept_indices = kept_indices[np.argsort(-display_scores[kept_indices])]
    return boxes[kept_indices], labels[kept_indices], display_scores[kept_indices]


def _subset_arrays(boxes, labels, scores, unknown_label, subset):
    if len(labels) == 0:
        return boxes, labels, scores
    if subset == 'known':
        mask = labels != int(unknown_label)
    elif subset == 'unknown':
        mask = labels == int(unknown_label)
    else:
        mask = np.ones_like(labels, dtype=bool)
    return boxes[mask], labels[mask], scores[mask]


def _save_rendered_detections(output_path, image_np, boxes, labels, scores, unknown_label):
    image = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        is_unknown = int(label) == int(unknown_label)
        color = 'red' if is_unknown else 'lime'
        text = f"{'unk' if is_unknown else int(label)}:{float(score):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0.0, y1 - 12)), text, fill=color)
    image.save(output_path)


def _save_visualizations(output_dir, image_stem, image_np, boxes, labels, scores, unknown_label, viz_cfg):
    del viz_cfg
    for subset in ['all', 'known', 'unknown']:
        subset_boxes, subset_labels, subset_scores = _subset_arrays(boxes, labels, scores, unknown_label, subset)
        _save_rendered_detections(
            output_dir / 'vis' / f'{image_stem}_{subset}.png',
            image_np,
            subset_boxes,
            subset_labels,
            subset_scores,
            unknown_label,
        )


def _save_unified_visualization(output_dir, image_stem, image_np, boxes, labels, scores, unknown_label, viz_cfg):
    del viz_cfg
    _save_rendered_detections(
        output_dir / 'vis' / f'{image_stem}_unified.png',
        image_np,
        boxes,
        labels,
        scores,
        unknown_label,
    )


def _save_layer_summary_svg(output_path, vis_debug):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    objectness = vis_debug.get('layer_objectness_probability', vis_debug.get('layer_obj_prob', None))
    knownness = vis_debug.get('layer_knownness_probability', vis_debug.get('layer_knownness_prob', None))
    unknownness = vis_debug.get('layer_unknown_probability', vis_debug.get('layer_unknown_prob', None))
    max_known = vis_debug.get('layer_max_known_class_probability', vis_debug.get('layer_cls_max', None))
    if objectness is None:
        return
    objectness = objectness.detach().mean(dim=(1, 2)).cpu().numpy()
    layers = list(range(len(objectness)))
    figure, axis = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    axis.plot(layers, objectness, marker='o', linewidth=2.0, label='objectness prob')
    if knownness is not None:
        axis.plot(layers, knownness.detach().mean(dim=(1, 2)).cpu().numpy(), marker='o', linewidth=2.0, label='knownness prob')
    if unknownness is not None:
        axis.plot(layers, unknownness.detach().mean(dim=(1, 2)).cpu().numpy(), marker='o', linewidth=2.0, label='unknown prob')
    if max_known is not None:
        axis.plot(layers, max_known.detach().mean(dim=(1, 2)).cpu().numpy(), marker='o', linewidth=2.0, label='max known prob')
    axis.set_xlabel('Decoder layer')
    axis.set_ylabel('Mean value')
    axis.set_title('Layer-wise prediction summary')
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def run_inference(args):
    output_dir = Path(args.output_dir)
    (output_dir / 'json').mkdir(parents=True, exist_ok=True)
    (output_dir / 'vis').mkdir(parents=True, exist_ok=True)
    (output_dir / 'debug').mkdir(parents=True, exist_ok=True)
    setup_logging(output=str(output_dir), distributed_rank=0, abbrev_name='PROB-Infer')

    checkpoint, model_args = _load_checkpoint_args(args.checkpoint, args.device)
    vars(model_args).update({k: v for k, v in vars(args).items() if v is not None})
    model, _, postprocessors, _ = build_model(model_args, mode=model_args.model_type)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(torch.device(args.device))
    model.eval()

    image_paths = _collect_input_images(args.input)
    logging.info('Found %s image(s) for inference', len(image_paths))

    dataset_name = model_args.dataset
    class_names = list(VOC_COCO_CLASS_NAMES[dataset_name])
    unknown_label = int(model_args.num_classes - 1)
    unknown_score_scale = float(model_args.uod_postprocess_unknown_scale)
    viz_cfg = build_viz_cfg()

    with torch.no_grad():
        for image_path in image_paths:
            original_image, image_tensor, target = _prepare_image(image_path)
            image_np = np.asarray(original_image)
            image_tensor = image_tensor.to(torch.device(args.device))
            try:
                outputs = model([image_tensor], return_vis_debug=args.save_layer_debug)
            except TypeError:
                outputs = model([image_tensor])

            target_sizes = target['orig_size'].unsqueeze(0).to(torch.device(args.device))
            predictions = postprocessors['bbox'](outputs, target_sizes)[0]
            raw_detections = _build_detection_records(predictions, class_names, unknown_label, unknown_score_scale)
            export_detections = _filter_export_detections(
                raw_detections,
                image_size=original_image.size,
                known_score_thresh=args.known_score_thresh,
                unknown_score_thresh=args.unknown_score_thresh,
                nms_iou=args.nms_iou,
                min_area_ratio=args.min_area_ratio,
                min_side_ratio=args.min_side_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                unified_across_labels=False,
            )
            unified_export_detections = _filter_export_detections(
                raw_detections,
                image_size=original_image.size,
                known_score_thresh=args.known_score_thresh,
                unknown_score_thresh=args.unknown_score_thresh,
                nms_iou=args.nms_iou,
                min_area_ratio=args.min_area_ratio,
                min_side_ratio=args.min_side_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                unified_across_labels=True,
            )

            display_boxes, display_labels, display_scores = _prepare_display_arrays(
                raw_detections,
                image_size=original_image.size,
                unknown_label=unknown_label,
                unknown_score_scale=unknown_score_scale,
                viz_cfg=viz_cfg,
            )
            unified_boxes, unified_labels, unified_scores = _prepare_unified_display_arrays(
                raw_detections,
                image_size=original_image.size,
                unknown_label=unknown_label,
                unknown_score_scale=unknown_score_scale,
                viz_cfg=viz_cfg,
            )
            _save_visualizations(output_dir, image_path.stem, image_np, display_boxes, display_labels, display_scores, unknown_label, viz_cfg)
            _save_unified_visualization(output_dir, image_path.stem, image_np, unified_boxes, unified_labels, unified_scores, unknown_label, viz_cfg)

            summary = {
                'num_raw': len(raw_detections),
                'num_exported': len(export_detections),
                'num_unified_exported': len(unified_export_detections),
                'num_displayed': int(len(display_labels)),
                'num_unified_displayed': int(len(unified_labels)),
                'num_known_exported': int(sum(0 if det['is_unknown'] else 1 for det in export_detections)),
                'num_unknown_exported': int(sum(1 if det['is_unknown'] else 0 for det in export_detections)),
                'num_known_unified_exported': int(sum(0 if det['is_unknown'] else 1 for det in unified_export_detections)),
                'num_unknown_unified_exported': int(sum(1 if det['is_unknown'] else 0 for det in unified_export_detections)),
            }
            payload = {
                'image': str(image_path),
                'image_size': {'width': int(original_image.size[0]), 'height': int(original_image.size[1])},
                'unknown_label': unknown_label,
                'export_filter': {
                    'known_score_thresh': float(args.known_score_thresh),
                    'unknown_score_thresh': float(args.unknown_score_thresh),
                    'nms_iou': float(args.nms_iou),
                    'min_area_ratio': float(args.min_area_ratio),
                    'min_side_ratio': float(args.min_side_ratio),
                    'max_aspect_ratio': float(args.max_aspect_ratio),
                },
                'display_filter': {
                    'display_known_score_thresh': float(viz_cfg['display_known_score_thresh']),
                    'display_unknown_score_thresh': float(viz_cfg['display_unknown_score_thresh']),
                    'display_nms_iou': float(viz_cfg['display_nms_iou']),
                    'display_apply_geometry_filter': bool(viz_cfg['display_apply_geometry_filter']),
                    'display_min_area_ratio': float(viz_cfg['display_min_area_ratio']),
                    'display_min_side_ratio': float(viz_cfg['display_min_side_ratio']),
                    'display_max_aspect_ratio': float(viz_cfg['display_max_aspect_ratio']),
                },
                'summary': summary,
                'predictions': export_detections,
                'unified_predictions': unified_export_detections,
            }
            (output_dir / 'json' / f'{image_path.stem}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            (output_dir / 'json' / f'{image_path.stem}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

            if args.save_layer_debug and outputs.get('vis_debug', None) is not None:
                _save_layer_summary_svg(output_dir / 'debug' / f'{image_path.stem}_layer_summary.svg', outputs['vis_debug'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Standalone inference for PROB / UOD checkpoints')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--input', required=True, type=str, help='single image path or directory of images')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model_type', default='uod', type=str)
    parser.add_argument('--known_score_thresh', default=0.35, type=float)
    parser.add_argument('--unknown_score_thresh', default=0.20, type=float)
    parser.add_argument('--nms_iou', default=0.5, type=float)
    parser.add_argument('--min_area_ratio', default=0.002, type=float)
    parser.add_argument('--min_side_ratio', default=0.03, type=float)
    parser.add_argument('--max_aspect_ratio', default=5.0, type=float)
    parser.add_argument('--uod_postprocess_unknown_scale', default=10.0, type=float)
    parser.add_argument('--uod_known_temp', default=8, type=float)
    parser.add_argument('--save_layer_debug', action='store_true')
    run_inference(parser.parse_args())
