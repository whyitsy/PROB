import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
from models import build_model
from tools.figure_svg_utils import save_svg_image
from util import box_ops
from util.log import setup_logging

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _load_checkpoint_args(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_args = checkpoint.get('args', {}) or {}
    saved_args['device'] = device
    return checkpoint, argparse.Namespace(**saved_args)


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


def _post_filter_predictions(boxes, labels, raw_scores, unknown_label, image_size, known_score_thresh, unknown_score_thresh, nms_iou, unknown_score_scale, min_area_ratio, min_side_ratio, max_aspect_ratio):
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.int64)
    raw_scores_t = torch.as_tensor(raw_scores, dtype=torch.float32)
    if boxes_t.numel() == 0:
        return []

    width, height = image_size
    display_scores_t = raw_scores_t.clone()
    unknown_mask = labels_t == int(unknown_label)
    display_scores_t[unknown_mask] = display_scores_t[unknown_mask] / max(float(unknown_score_scale), 1e-6)

    keep = []
    for idx in range(boxes_t.shape[0]):
        score = float(display_scores_t[idx].item())
        label = int(labels_t[idx].item())
        if label == int(unknown_label):
            if score < float(unknown_score_thresh):
                continue
        else:
            if score < float(known_score_thresh):
                continue
        if not _is_valid_geometry_xyxy(boxes_t[idx].tolist(), width, height, min_area_ratio, min_side_ratio, max_aspect_ratio):
            continue
        keep.append(idx)

    if not keep:
        return []
    keep = torch.as_tensor(keep, dtype=torch.long)
    boxes_t = boxes_t[keep]
    labels_t = labels_t[keep]
    raw_scores_t = raw_scores_t[keep]
    display_scores_t = display_scores_t[keep]

    final_keep = []
    for select_unknown in [False, True]:
        mask = (labels_t == int(unknown_label)) if select_unknown else (labels_t != int(unknown_label))
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        kept = _nms_xyxy(boxes_t[idx], raw_scores_t[idx], iou_threshold=nms_iou)
        if kept.numel() > 0:
            final_keep.append(idx[kept])
    if not final_keep:
        return []
    final_keep = torch.cat(final_keep, dim=0)
    final_keep = final_keep[torch.argsort(raw_scores_t[final_keep], descending=True)]

    detections = []
    for idx in final_keep.tolist():
        detections.append({
            'label': int(labels_t[idx].item()),
            'box_xyxy': [float(v) for v in boxes_t[idx].tolist()],
            'raw_score': float(raw_scores_t[idx].item()),
            'score': float(display_scores_t[idx].item()),
            'is_unknown': bool(int(labels_t[idx].item()) == int(unknown_label)),
        })
    return detections


def _draw_predictions(image, detections, unknown_label, class_names, subset='all'):
    base = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    short_side = min(base.size)
    font_size = max(18, int(short_side * 0.02))
    line_width = max(6, int(short_side * 0.008))
    text_pad_x = max(4, int(font_size * 0.2))
    text_pad_y = max(2, int(font_size * 0.1))
    text_bg_color = (10, 12, 18, 185)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        if subset == 'known' and det['is_unknown']:
            continue
        if subset == 'unknown' and not det['is_unknown']:
            continue
        x1, y1, x2, y2 = det['box_xyxy']
        label = int(det['label'])
        score = float(det['score'])
        if label == int(unknown_label):
            stroke_color = (255, 35, 120, 220)
            text = f'Unknown {score:.2f}'
        else:
            class_name = class_names[label] if 0 <= label < len(class_names) else f'class_{label}'
            stroke_color = (0, 220, 160, 220)
            text = f'{class_name} {score:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=stroke_color, width=line_width)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        bg_x1 = x1 + line_width / 2
        bg_y1 = y1 + line_width / 2
        bg_x2 = bg_x1 + tw + 2 * text_pad_x
        bg_y2 = bg_y1 + th + 2 * text_pad_y
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=text_bg_color)
        draw.text((bg_x1 + text_pad_x, bg_y1 + text_pad_y - 2), text, fill=stroke_color, font=font)
    return Image.alpha_composite(base, overlay).convert('RGB')


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
    setup_logging(output=str(output_dir), distributed_rank=0, abbrev_name='PROB-Infer-V2')

    checkpoint, model_args = _load_checkpoint_args(args.checkpoint, args.device)
    vars(model_args).update({k: v for k, v in vars(args).items() if v is not None})
    model, _, postprocessors, _ = build_model(model_args, mode=getattr(model_args, 'model_type', 'uod'))
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(torch.device(args.device))
    model.eval()

    image_paths = _collect_input_images(args.input)
    logging.info('Found %s image(s) for inference', len(image_paths))
    dataset_name = getattr(model_args, 'dataset', 'OWDETR')
    class_names = list(VOC_COCO_CLASS_NAMES[dataset_name])
    unknown_label = int(getattr(model_args, 'num_classes', len(class_names)) - 1)

    with torch.no_grad():
        for image_path in image_paths:
            original_image, image_tensor, target = _prepare_image(image_path)
            image_tensor = image_tensor.to(torch.device(args.device))
            try:
                outputs = model([image_tensor], return_vis_debug=args.save_layer_debug)
            except TypeError:
                outputs = model([image_tensor])
            target_sizes = target['orig_size'].unsqueeze(0).to(torch.device(args.device))
            predictions = postprocessors['bbox'](outputs, target_sizes)[0]
            boxes = predictions['boxes'].detach().cpu().tolist()
            labels = predictions['labels'].detach().cpu().tolist()
            raw_scores = predictions['scores'].detach().cpu().tolist()
            unknown_score_scale = float(getattr(model_args, 'uod_postprocess_unknown_scale', 15.0))

            detections = _post_filter_predictions(
                boxes=boxes,
                labels=labels,
                raw_scores=raw_scores,
                unknown_label=unknown_label,
                image_size=original_image.size,
                known_score_thresh=args.known_score_thresh,
                unknown_score_thresh=args.unknown_score_thresh,
                nms_iou=args.nms_iou,
                unknown_score_scale=unknown_score_scale,
                min_area_ratio=args.min_area_ratio,
                min_side_ratio=args.min_side_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
            )

            json_path = output_dir / 'json' / f'{image_path.stem}.json'
            json_path.write_text(json.dumps({'image': str(image_path), 'predictions': detections}, ensure_ascii=False, indent=2), encoding='utf-8')

            vis_all = _draw_predictions(original_image, detections, unknown_label, class_names, subset='all')
            vis_known = _draw_predictions(original_image, detections, unknown_label, class_names, subset='known')
            vis_unknown = _draw_predictions(original_image, detections, unknown_label, class_names, subset='unknown')
            save_svg_image(np.array(vis_all), output_dir / 'vis' / f'{image_path.stem}_all.svg')
            save_svg_image(np.array(vis_known), output_dir / 'vis' / f'{image_path.stem}_known.svg')
            save_svg_image(np.array(vis_unknown), output_dir / 'vis' / f'{image_path.stem}_unknown.svg')

            summary = {
                'num_all': len(detections),
                'num_known': int(sum(0 if det['is_unknown'] else 1 for det in detections)),
                'num_unknown': int(sum(1 if det['is_unknown'] else 0 for det in detections)),
            }
            (output_dir / 'json' / f'{image_path.stem}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

            if args.save_layer_debug and outputs.get('vis_debug', None) is not None:
                _save_layer_summary_svg(output_dir / 'debug' / f'{image_path.stem}_layer_summary.svg', outputs['vis_debug'])


def build_parser():
    parser = argparse.ArgumentParser('Pure inference v2 for PROB / UOD checkpoints')
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
    parser.add_argument('--save_layer_debug', action='store_true')
    return parser


if __name__ == '__main__':
    run_inference(build_parser().parse_args())
