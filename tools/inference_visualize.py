import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _load_repo_parser() -> argparse.ArgumentParser:
    mod = importlib.import_module('main_open_world')
    if not hasattr(mod, 'get_args_parser'):
        raise RuntimeError('main_open_world.py does not expose get_args_parser()')
    base = mod.get_args_parser()
    parser = argparse.ArgumentParser(
        'PROB/UOD inference visualization',
        parents=[base],
        conflict_handler='resolve'
    )
    parser.add_argument('--image_path', required=True, help='Path to the image to visualize')
    parser.add_argument('--resume', required=True, help='Checkpoint path')
    parser.add_argument('--output_dir_vis', default='viz_outputs', help='Directory to save generated figures')
    parser.add_argument('--score_thresh', type=float, default=0.0, help='Final detection score threshold after official postprocess')
    parser.add_argument('--topk', type=int, default=100, help='Maximum number of detections to draw after thresholding')
    parser.add_argument('--known_class_cutoff', type=int, default=None,
                        help='Labels < cutoff are treated as known; labels >= cutoff are treated as unknown. '
                             'Defaults to num_classes-1.')
    parser.add_argument('--unknown_label_id', type=int, default=None,
                        help='Optional explicit unknown label id. If set, labels == unknown_label_id are unknown.')
    parser.add_argument('--save_query_debug', action='store_true', help='Save per-query debug tensors when available')
    parser.add_argument('--save_attention_heatmaps', action='store_true', help='Save heatmaps for cls_soft_attn / odqe_gate when available')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    return parser


def _build_model(args: argparse.Namespace):
    models_mod = importlib.import_module('models')
    if not hasattr(models_mod, 'build_model'):
        raise RuntimeError('models.__init__ does not expose build_model(args)')
    built = models_mod.build_model(args)
    if isinstance(built, tuple):
        if len(built) == 3:
            model, criterion, postprocessors = built
        elif len(built) >= 4:
            model, criterion, postprocessors = built[:3]
        else:
            raise RuntimeError(f'Unexpected build_model return length: {len(built)}')
    else:
        raise RuntimeError('build_model(args) must return a tuple')
    return model, criterion, postprocessors


def _nested_tensor_from_pil(img: Image.Image, device: torch.device) -> Tuple[Any, Tuple[int, int], np.ndarray]:
    misc = importlib.import_module('util.misc')
    if not hasattr(misc, 'nested_tensor_from_tensor_list'):
        raise RuntimeError('util.misc.nested_tensor_from_tensor_list not found')
    orig_np = np.array(img)
    orig_h, orig_w = orig_np.shape[:2]
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    nested = misc.nested_tensor_from_tensor_list([tensor]).to(device)
    return nested, (orig_h, orig_w), orig_np


def _tensor_to_prob(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    # robust conversion: if already in [0, 1], keep; otherwise sigmoid.
    if float(x.min()) >= 0.0 and float(x.max()) <= 1.0:
        return x
    return x.sigmoid()


def _extract_query_debug(outputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    debug: Dict[str, np.ndarray] = {}
    if 'pred_obj' in outputs:
        obj = outputs['pred_obj'][0]
        obj_prob = _tensor_to_prob(obj).squeeze(-1)
        debug['obj_prob'] = obj_prob.detach().cpu().numpy()
    if 'pred_known' in outputs:
        known = outputs['pred_known'][0]
        known_prob = _tensor_to_prob(known).squeeze(-1)
        debug['known_prob'] = known_prob.detach().cpu().numpy()
        debug['unknown_prob'] = (1.0 - known_prob).detach().cpu().numpy()
    if 'pred_logits' in outputs:
        logits = outputs['pred_logits'][0]
        cls_prob = logits.sigmoid()
        if cls_prob.shape[-1] > 0:
            debug['known_max'] = cls_prob.max(dim=-1).values.detach().cpu().numpy()
            debug['cls_prob'] = cls_prob.detach().cpu().numpy()
    if 'debug' in outputs and isinstance(outputs['debug'], dict):
        for k, v in outputs['debug'].items():
            if torch.is_tensor(v):
                debug[k] = v.detach().cpu().numpy()
    return debug


def _reshape_vector_for_heatmap(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v).reshape(-1)
    cols = max(1, int(math.sqrt(len(v))))
    rows = math.ceil(len(v) / cols)
    padded = np.pad(v, (0, rows * cols - len(v)), constant_values=np.nan)
    return padded.reshape(rows, cols)


def _save_heatmap(arr: np.ndarray, title: str, save_path: Path, cmap: str = 'YlOrRd') -> None:
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(arr, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _draw_detections(image_np: np.ndarray,
                     results: Dict[str, torch.Tensor],
                     out_path: Path,
                     score_thresh: float,
                     topk: int,
                     known_class_cutoff: int,
                     unknown_label_id: Optional[int]) -> None:
    scores = results['scores'].detach().cpu()
    labels = results['labels'].detach().cpu()
    boxes = results['boxes'].detach().cpu()

    keep = scores >= score_thresh
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    if len(scores) > topk:
        order = torch.argsort(scores, descending=True)[:topk]
        scores = scores[order]
        labels = labels[order]
        boxes = boxes[order]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)

    for score, label, box in zip(scores, labels, boxes):
        x1, y1, x2, y2 = box.tolist()
        label_i = int(label)
        is_unknown = False
        if unknown_label_id is not None:
            is_unknown = label_i == unknown_label_id
        else:
            is_unknown = label_i >= known_class_cutoff
        color = '#32CD32' if not is_unknown else '#FFD700'
        tag = 'Known' if not is_unknown else 'Unknown'
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.3,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 4), f'{tag} {float(score):.2f}', fontsize=10, color='black',
                bbox=dict(facecolor=color, alpha=0.75, edgecolor='none', pad=2))

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = _load_repo_parser()
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    out_dir = Path(args.output_dir_vis)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_pil = Image.open(args.image_path).convert('RGB')
    samples, (orig_h, orig_w), image_np = _nested_tensor_from_pil(image_pil, device)

    model, _, postprocessors = _build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint.get('model', checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[INFO] loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}')
    model.eval()

    with torch.no_grad():
        outputs = model(samples)

    if not isinstance(postprocessors, dict) or 'bbox' not in postprocessors:
        raise RuntimeError('postprocessors["bbox"] not found; cannot run official detection postprocess')

    target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
    results = postprocessors['bbox'](outputs, target_sizes)[0]

    known_cutoff = args.known_class_cutoff
    if known_cutoff is None:
        known_cutoff = max(1, int(getattr(args, 'num_classes', 81)) - 1)

    image_stem = Path(args.image_path).stem
    vis_path = out_dir / f'{image_stem}_detections.png'
    _draw_detections(image_np, results, vis_path, args.score_thresh, args.topk, known_cutoff, args.unknown_label_id)
    print(f'[OK] saved detection figure: {vis_path}')

    debug = _extract_query_debug(outputs)
    if debug and args.save_query_debug:
        npz_path = out_dir / f'{image_stem}_query_debug.npz'
        np.savez_compressed(npz_path, **debug)
        summary_path = out_dir / f'{image_stem}_query_debug_summary.json'
        summary = {k: {'shape': list(np.asarray(v).shape), 'min': float(np.nanmin(v)), 'max': float(np.nanmax(v)), 'mean': float(np.nanmean(v))}
                   for k, v in debug.items() if np.asarray(v).size > 0}
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'[OK] saved query debug tensors: {npz_path}')
        print(f'[OK] saved query debug summary: {summary_path}')

        if args.save_attention_heatmaps:
            for name in ('cls_soft_attn', 'odqe_gate', 'obj_prob', 'unknown_prob', 'known_prob', 'known_max'):
                if name not in debug:
                    continue
                arr = np.asarray(debug[name])
                if arr.ndim == 1:
                    arr = _reshape_vector_for_heatmap(arr)
                elif arr.ndim > 2:
                    arr = np.squeeze(arr)
                    if arr.ndim == 1:
                        arr = _reshape_vector_for_heatmap(arr)
                heat_path = out_dir / f'{image_stem}_{name}_heatmap.png'
                _save_heatmap(arr, name, heat_path)
                print(f'[OK] saved heatmap: {heat_path}')
    else:
        print('[INFO] no debug tensors found in raw outputs; main detection figure is still valid.')


if __name__ == '__main__':
    main()
