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



def _draw_predictions(image, boxes, labels, scores, unknown_label, score_threshold=0.0):
    canvas = image.copy().convert('RGB')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except Exception:
        font = ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        color = (216, 27, 96) if int(label) == int(unknown_label) else (0, 166, 90)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f'U {score:.2f}' if int(label) == int(unknown_label) else f'K[{int(label)}] {score:.2f}'
        bbox = draw.textbbox((x1 + 2, y1 + 2), text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(20, 20, 20))
        draw.text((x1 + 2, y1 + 2), text, fill=color, font=font)
    return canvas


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
    unknown_label = int(getattr(model_args, 'num_classes', 81) - 1)

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
            scores = predictions['scores'].detach().cpu().tolist()

            filtered = [
                {
                    'label': int(label),
                    'score': float(score),
                    'box_xyxy': [float(value) for value in box],
                    'is_unknown': bool(int(label) == unknown_label),
                }
                for box, label, score in zip(boxes, labels, scores)
                if float(score) >= args.score_thresh
            ]
            json_path = output_dir / 'json' / f'{image_path.stem}.json'
            json_path.write_text(json.dumps({'image': str(image_path), 'predictions': filtered}, ensure_ascii=False, indent=2), encoding='utf-8')

            vis_image = _draw_predictions(original_image, boxes, labels, scores, unknown_label, score_threshold=args.score_thresh)
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
    parser.add_argument('--score_thresh', default=0.0, type=float)
    parser.add_argument('--uod_pseudo_bbox_loss_coef', default=None, type=float)
    parser.add_argument('--uod_pseudo_giou_loss_coef', default=None, type=float)
    parser.add_argument('--save_layer_debug', action='store_true', help='save layer-wise score summary when the checkpoint/model supports vis_debug')
    run_inference(parser.parse_args())
