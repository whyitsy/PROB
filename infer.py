import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datasets.coco import make_coco_transforms
from models import build_model
from util.log import setup_logging
from util import box_ops

from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_checkpoint_args(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_args = checkpoint.get('args', {}) or {}
    saved_args['device'] = device
    return checkpoint, argparse.Namespace(**saved_args)


def _collect_input_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    image_paths = []
    for path in sorted(input_path.rglob('*')):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
    return image_paths


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


def _post_filter_predictions(
    boxes,
    labels,
    raw_scores,
    unknown_label,
    known_score_thresh=0.05,     # 新增：已知类别阈值
    unknown_score_thresh=0.05,   # 新增：未知类别阈值
    nms_iou=0.6,
    unknown_score_scale=15.0,
):
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.int64)
    raw_scores_t = torch.as_tensor(raw_scores, dtype=torch.float32)

    if boxes_t.numel() == 0:
        return [], [], [], []

    # 用于显示和阈值过滤的分数：
    # known 直接用 raw score
    # unknown 用 raw score / unknown_scale，还原到更可解释的范围
    display_scores_t = raw_scores_t.clone()
    unknown_mask = labels_t == int(unknown_label)
    known_mask = ~unknown_mask  # 新增：已知类别的 mask
    
    if float(unknown_score_scale) > 0:
        display_scores_t[unknown_mask] = display_scores_t[unknown_mask] / float(unknown_score_scale)

    # 分别对已知和未知类别应用不同的阈值过滤
    keep_known = known_mask & (display_scores_t >= float(known_score_thresh))
    keep_unknown = unknown_mask & (display_scores_t >= float(unknown_score_thresh))
    keep = keep_known | keep_unknown  # 合并保留的索引
    
    boxes_t = boxes_t[keep]
    labels_t = labels_t[keep]
    raw_scores_t = raw_scores_t[keep]
    display_scores_t = display_scores_t[keep]

    if boxes_t.numel() == 0:
        return [], [], [], []

    # # known / unknown 分开做 NMS，避免互相压制
    # keep_indices = []

    # known_indices = torch.nonzero(labels_t != int(unknown_label), as_tuple=False).flatten()
    # if known_indices.numel() > 0:
    #     kept_known = _nms_xyxy(boxes_t[known_indices], raw_scores_t[known_indices], iou_threshold=nms_iou)
    #     keep_indices.append(known_indices[kept_known])

    # unknown_indices = torch.nonzero(labels_t == int(unknown_label), as_tuple=False).flatten()
    # if unknown_indices.numel() > 0:
    #     kept_unknown = _nms_xyxy(boxes_t[unknown_indices], raw_scores_t[unknown_indices], iou_threshold=nms_iou)
    #     keep_indices.append(unknown_indices[kept_unknown])

    # if not keep_indices:
    #     return [], [], [], []

    # keep = torch.cat(keep_indices, dim=0)
    # keep = keep[torch.argsort(raw_scores_t[keep], descending=True)]
    
    # return (
    #     boxes_t[keep].cpu().tolist(),
    #     labels_t[keep].cpu().tolist(),
    #     raw_scores_t[keep].cpu().tolist(),
    #     display_scores_t[keep].cpu().tolist(),
    # )
    
    # --- 统一 NMS (不分已知/未知) ---
    # 使用 raw_scores_t 进行 NMS 排序，确保置信度高的框被保留
    keep_idx = _nms_xyxy(boxes_t, raw_scores_t, iou_threshold=nms_iou)
    
    # 按照 raw_scores 降序排列最终结果
    keep_idx = keep_idx[torch.argsort(raw_scores_t[keep_idx], descending=True)]
    
    return (
            boxes_t[keep_idx].cpu().tolist(),
            labels_t[keep_idx].cpu().tolist(),
            raw_scores_t[keep_idx].cpu().tolist(),
            display_scores_t[keep_idx].cpu().tolist(),
        )

def _draw_predictions(image, boxes, labels, scores, unknown_label, class_names):
    base = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    short_side = min(base.size)
    font_size = max(18, int(short_side * 0.02))
    line_width = max(8, int(short_side * 0.01))
    
    # 适当调整内边距
    text_pad_x = max(4, int(font_size * 0.2))
    text_pad_y = max(2, int(font_size * 0.1))
    
    box_fill_alpha = 0
    # 背景颜色，可以根据需要调整透明度 (最后一个数值)
    text_bg_color = (10, 12, 18, 180) 

    try:
        font = ImageFont.truetype('DejaVuSans.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(v) for v in box]

        if int(label) == int(unknown_label):
            stroke_color = (255, 35, 120, 200)   # 洋红
        else:
            stroke_color = (0, 220, 160, 200)    # 青绿

        # 1. 绘制物体的大框
        draw.rectangle([x1, y1, x2, y2], outline=stroke_color, width=line_width)

        # 2. 准备文字内容
        class_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else f'class_{int(label)}'
        text = f'Unknown {score:.2f}' if int(label) == int(unknown_label) else f'{class_name} {score:.2f}'
        
        # 计算文字大小
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        # 3. 计算背景填充块的坐标 (紧贴边框内侧)
        # 起始点设在 x1 + line_width/2, y1 + line_width/2 使得背景块与边框内沿对齐
        bg_x1 = x1 + line_width / 2
        bg_y1 = y1 + line_width / 2
        bg_x2 = bg_x1 + tw + 2 * text_pad_x
        bg_y2 = bg_y1 + th + 2 * text_pad_y

        # 4. 绘制文字背景块 (仅填充，无外框)
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=text_bg_color)

        # 5. 绘制文字
        draw.text(
            (bg_x1 + text_pad_x, bg_y1 + text_pad_y - 2), # -2 是微调视觉上的垂直居中
            text,
            fill=stroke_color,
            font=font,
        )

    return Image.alpha_composite(base, overlay).convert('RGB')


def _save_layer_summary_svg(output_path, vis_debug):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    objectness = vis_debug.get('layer_objectness_probability', None)
    knownness = vis_debug.get('layer_knownness_probability', None)
    unknownness = vis_debug.get('layer_unknown_probability', None)
    max_known = vis_debug.get('layer_max_known_class_probability', None)
    if objectness is None:
        return
    objectness = objectness.detach().mean(dim=(1, 2)).cpu().numpy()
    layers = list(range(len(objectness)))
    figure, axis = plt.subplots(figsize=(8, 5))
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
    axis.legend(frameon=False)
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)


def run_inference(args):
    output_dir = Path(args.output_dir)
    (output_dir / 'json').mkdir(parents=True, exist_ok=True)
    (output_dir / 'vis').mkdir(parents=True, exist_ok=True)
    (output_dir / 'debug').mkdir(parents=True, exist_ok=True)
    setup_logging(output=str(output_dir), distributed_rank=0, abbrev_name='PROB-Infer')

    checkpoint, model_args = _load_checkpoint_args(args.checkpoint, args.device)
    vars(model_args).update(vars(args))
    model, _, postprocessors, _ = build_model(model_args, mode=getattr(model_args, 'model_type', 'prob'))
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(torch.device(args.device))
    model.eval()

    image_paths = _collect_input_images(args.input)
    logging.info('Found %s image(s) for inference', len(image_paths))
    dataset_name = getattr(model_args, 'dataset', 'OWDETR')
    class_names = list(VOC_COCO_CLASS_NAMES[dataset_name])
    logging.info('Using class names from dataset "%s": %s', dataset_name, class_names)
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
            boxes, labels, raw_scores, display_scores = _post_filter_predictions(
                boxes=boxes,
                labels=labels,
                raw_scores=raw_scores,
                unknown_label=unknown_label,
                known_score_thresh=args.known_score_thresh,     # 新增：已知类别阈值
                unknown_score_thresh=args.unknown_score_thresh,   # 新增：未知类别阈值
                nms_iou=args.nms_iou,
                unknown_score_scale=unknown_score_scale,
            )

            filtered = [
                {
                    'label': int(label),
                    'score': float(display_score),      # 用于展示/阈值的分数
                    'raw_score': float(raw_score),      # 保留原始排序分数
                    'box_xyxy': [float(value) for value in box],
                    'is_unknown': bool(int(label) == unknown_label),
                }
                for box, label, raw_score, display_score in zip(boxes, labels, raw_scores, display_scores)
            ]

            json_path = output_dir / 'json' / f'{image_path.stem}.json'
            json_path.write_text(
                json.dumps({'image': str(image_path), 'predictions': filtered}, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            vis_image = _draw_predictions(
                original_image,
                boxes,
                labels,
                display_scores,
                unknown_label,
                class_names
            )
            vis_image.save(output_dir / 'vis' / f'{image_path.stem}.png')
            vis_image.save(output_dir / 'vis' / f'{image_path.stem}.png')

            if args.save_layer_debug and outputs.get('vis_debug', None) is not None:
                _save_layer_summary_svg(output_dir / 'debug' / f'{image_path.stem}_layer_summary.svg', outputs['vis_debug'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Standalone inference for PROB / UOD checkpoints')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--input', required=True, type=str, help='single image path or a directory of images')
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--mode_type', default='uod', type=str)
    parser.add_argument('--known_score_thresh', default=0.4, type=float, help='Score threshold for known classes')
    parser.add_argument('--unknown_score_thresh', default=0.3, type=float, help='Score threshold for unknown classes')
    parser.add_argument('--uod_pseudo_bbox_loss_coef', default=None, type=float)
    parser.add_argument('--uod_pseudo_giou_loss_coef', default=None, type=float)
    parser.add_argument('--nms_iou', default=0.3, type=float)
    parser.add_argument('--save_layer_debug', action='store_true', help='save layer-wise score summary when the checkpoint/model supports vis_debug')
    run_inference(parser.parse_args())
