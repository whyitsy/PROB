import argparse
from pathlib import Path
import sys


root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from main_open_world import get_args_parser
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from util.eval_viz import _draw_boxes, _ensure_dir



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

"""
# 单张图片
python tools/visualize_infer.py \
    --model_type uod \
    --resume /path/to/checkpoint.pth \
    --output_dir /path/to/exp_uod \
    --input /path/to/demo.jpg \
    --save_debug \
    
    --with_box_refine \
    --uod_enable_unknown \
    --uod_enable_pseudo \
    --uod_enable_batch_dynamic \
    --uod_enable_cls_soft_attn \
    
    --uod_enable_decorr \
    --uod_enable_odqe \
  
# 多张图片
python tools/visualize_infer.py \
    --model_type uod \
    --resume /path/to/checkpoint.pth \
    --output_dir /path/to/exp_uod \
    --input_dir /path/to/images \
    --glob "*.jpg" \
    --save_debug \
    
    --with_box_refine \
    --uod_enable_unknown \
    --uod_enable_pseudo \
    --uod_enable_batch_dynamic \
    --uod_enable_cls_soft_attn \
  
    --uod_enable_decorr \
    --uod_enable_odqe \
"""

def parse_args():
    parser = argparse.ArgumentParser(
        'Standalone inference visualizer',
        parents=[get_args_parser()],
        add_help=True,
    )
    parser.add_argument('--input', type=str, default='', help='single image path')
    parser.add_argument('--input_dir', type=str, default='', help='directory of images')
    parser.add_argument('--glob', type=str, default='*.jpg', help='glob pattern inside input_dir')
    parser.add_argument('--score_thresh', type=float, default=0.0)
    parser.add_argument('--save_debug', action='store_true')
    return parser.parse_args()


def build_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def discover_inputs(args):
    if args.input:
        return [Path(args.input)]
    if args.input_dir:
        return sorted(Path(args.input_dir).glob(args.glob))
    raise ValueError('either --input or --input_dir must be provided')


def load_model(args, device):
    model, criterion, postprocessors, _ = build_model(args, mode=args.model_type)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('Missing keys:', missing)
    if unexpected:
        print('Unexpected keys:', unexpected)
    model.to(device)
    model.eval()
    return model, postprocessors


def to_nested_tensor(pil_img, transform, device):
    tensor = transform(pil_img).to(device)
    return nested_tensor_from_tensor_list([tensor]), tensor



def to_grid(vec):
    vec = np.asarray(vec, dtype=np.float32)
    n = int(vec.shape[0])
    side = int(np.ceil(np.sqrt(max(n, 1))))
    grid = np.zeros((side * side,), dtype=np.float32)
    grid[:n] = vec
    grid = grid.reshape(side, side)
    return grid


def save_heatmap(vec, title, out_path):
    grid = to_grid(vec)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(grid, cmap='magma')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def run_one(model, postprocessors, image_path, args, device, transform):
    final_dir = Path(args.output_dir) / 'infer' / 'run_latest' / 'final'
    debug_dir = Path(args.output_dir) / 'infer' / 'run_latest' / 'debug'
    raw_dir = Path(args.output_dir) / 'infer' / 'run_latest' / 'raw'
    _ensure_dir(str(final_dir))
    _ensure_dir(str(debug_dir))
    _ensure_dir(str(raw_dir))


    pil_img = Image.open(image_path).convert('RGB')
    np_img = np.array(pil_img)
    samples, _ = to_nested_tensor(pil_img, transform, device)
    outputs = model(samples, return_vis_debug=True)
    target_size = torch.tensor([[pil_img.height, pil_img.width]], dtype=torch.long, device=device)
    results = postprocessors['bbox'](outputs, target_size)[0]


    keep = results['scores'].detach().cpu().numpy() >= float(args.score_thresh)
    boxes = results['boxes'].detach().cpu().numpy()[keep]
    labels = results['labels'].detach().cpu().numpy()[keep]
    scores = results['scores'].detach().cpu().numpy()[keep]
    unk_label = int(getattr(args, 'num_classes', 81) - 1)
    known_mask = labels != unk_label if len(labels) > 0 else np.array([], dtype=bool)
    unk_mask = labels == unk_label if len(labels) > 0 else np.array([], dtype=bool)
    summary = f'pred={len(boxes)} unk_pred={int(unk_mask.sum())}'


    stem = image_path.stem
    overlay = _draw_boxes(np_img, pred_boxes=boxes, pred_labels=labels, pred_scores=scores,
                          unk_label=unk_label, title='Inference Overlay', summary_text=summary, show_legend=True)
    pred_known = _draw_boxes(np_img, pred_boxes=boxes[known_mask], pred_labels=labels[known_mask],
                             pred_scores=scores[known_mask], unk_label=unk_label, title='Known Predictions',
                             summary_text=summary, show_legend=True)
    pred_unknown = _draw_boxes(np_img, pred_boxes=boxes[unk_mask], pred_labels=labels[unk_mask],
                               pred_scores=scores[unk_mask], unk_label=unk_label, title='Unknown Predictions',
                               summary_text=summary, show_legend=True)
    Image.fromarray(overlay).save(final_dir / f'{stem}__overlay.png')
    Image.fromarray(pred_known).save(final_dir / f'{stem}__pred_known.png')
    Image.fromarray(pred_unknown).save(final_dir / f'{stem}__pred_unknown.png')


    vis_debug = outputs.get('vis_debug', {})
    if args.save_debug and vis_debug:
        layer_obj = vis_debug['layer_obj_prob'][:, 0].detach().cpu().numpy()
        layer_unk = vis_debug['layer_unknown_prob'][:, 0].detach().cpu().numpy()
        layer_cls = vis_debug['layer_cls_max'][:, 0].detach().cpu().numpy()
        shallow_idx = 0
        deep_idx = layer_obj.shape[0] - 1
        save_heatmap(layer_obj[shallow_idx], 'Shallow Objectness Query Map', debug_dir / f'{stem}__obj_shallow.png')
        save_heatmap(layer_obj[deep_idx], 'Deep Objectness Query Map', debug_dir / f'{stem}__obj_deep.png')
        save_heatmap(layer_obj.mean(axis=0), 'Fused Objectness Query Map', debug_dir / f'{stem}__obj_fused.png')
        save_heatmap(layer_unk[shallow_idx], 'Shallow Unknownness Query Map', debug_dir / f'{stem}__unk_shallow.png')
        save_heatmap(layer_unk[deep_idx], 'Deep Unknownness Query Map', debug_dir / f'{stem}__unk_deep.png')
        save_heatmap(layer_unk.mean(axis=0), 'Fused Unknownness Query Map', debug_dir / f'{stem}__unk_fused.png')
        save_heatmap(layer_cls[deep_idx], 'Deep Max-Known Query Map', debug_dir / f'{stem}__cls_deep.png')
        gate = vis_debug.get('gate_mean_per_layer', None)
        if gate is not None:
            gate = gate.detach().cpu().numpy().reshape(-1)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(np.arange(len(gate)), gate, marker='o')
            ax.set_xlabel('Decoder layer')
            ax.set_ylabel('Mean gate')
            ax.set_title('ODQE Gate Mean Per Layer')
            ax.grid(alpha=0.25)
            fig.savefig(debug_dir / f'{stem}__gate_curve.png', dpi=180, bbox_inches='tight')
            plt.close(fig)
        np.savez_compressed(
            raw_dir / f'{stem}__vis_debug.npz',
            layer_obj_prob=layer_obj,
            layer_unknown_prob=layer_unk,
            layer_cls_max=layer_cls,
        )


def main():
    args = parse_args()
    device = torch.device(args.device)
    transform = build_transform()
    model, postprocessors = load_model(args, device)
    paths = discover_inputs(args)
    for p in paths:
        run_one(model, postprocessors, p, args, device, transform)
    print('Done.')

if __name__ == '__main__':
    main()