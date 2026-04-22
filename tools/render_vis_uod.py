#!/usr/bin/env python3
"""Render qualitative/statistical plots from extract_vis_uod.py dumps."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from util import box_ops


KNOWN_COLOR = '#2ecc71'
UNKNOWN_COLOR = '#e74c3c'
BACKGROUND_COLOR = '#b0b0b0'
ERROR_COLOR = '#f1c40f'
STAGE_COLOR = '#ff5a36'
QUERY_COLOR = '#00d5ff'
LEVEL_COLORS = ['#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93', '#1982c4']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Render UOD visualizations from extraction dumps')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_dump_arg(p):
        p.add_argument('--dump_dir', required=True)
        p.add_argument('--output_dir', required=True)

    def add_image_selection_args(p):
        p.add_argument('--image_ids', nargs='*', type=int, default=None)
        p.add_argument('--image_ids_file', default=None)
        p.add_argument('--max_images', type=int, default=8)

    overlay = subparsers.add_parser('overlay')
    add_dump_arg(overlay)
    add_image_selection_args(overlay)
    overlay.add_argument('--score_thr', type=float, default=0.30)
    overlay.add_argument('--max_det', type=int, default=15)
    overlay.add_argument('--max_error_det', type=int, default=5)
    overlay.add_argument('--iou_thr', type=float, default=0.5)

    mining = subparsers.add_parser('mining')
    add_dump_arg(mining)
    add_image_selection_args(mining)

    hist = subparsers.add_parser('histograms')
    add_dump_arg(hist)
    hist.add_argument('--bins', type=int, default=40)
    hist.add_argument('--assign_iou_thr', type=float, default=0.3)
    hist.add_argument('--background_iou_thr', type=float, default=0.1)

    evolution = subparsers.add_parser('box_evolution')
    add_dump_arg(evolution)
    add_image_selection_args(evolution)
    evolution.add_argument('--query_index', type=int, default=None)

    odqe = subparsers.add_parser('odqe_sampling')
    add_dump_arg(odqe)
    add_image_selection_args(odqe)
    odqe.add_argument('--layer_index', type=int, default=-1)
    odqe.add_argument('--query_index', type=int, default=None)
    odqe.add_argument('--top_points', type=int, default=24)

    gain = subparsers.add_parser('gate_gain')
    add_dump_arg(gain)
    add_image_selection_args(gain)
    gain.add_argument('--layer_index', type=int, default=-1)
    gain.add_argument('--top_queries', type=int, default=12)

    decorr = subparsers.add_parser('decorr')
    add_dump_arg(decorr)
    decorr.add_argument('--max_points', type=int, default=15000)

    manifold = subparsers.add_parser('manifold')
    add_dump_arg(manifold)
    manifold.add_argument('--max_points', type=int, default=15000)
    manifold.add_argument('--ellipsoid_quantile', type=float, default=0.95)
    manifold.add_argument('--kde_grid_size', type=int, default=30)
    manifold.add_argument('--kde_density_quantile', type=float, default=0.05)
    manifold.add_argument('--kde_band_ratio', type=float, default=0.1)

    return parser


def read_manifest(dump_dir: Path) -> Dict[str, Any]:
    with open(dump_dir / 'manifest.json', 'r', encoding='utf-8') as handle:
        return json.load(handle)


def parse_image_ids(args, manifest: Dict[str, Any]) -> List[int]:
    if getattr(args, 'image_ids', None):
        return list(dict.fromkeys(args.image_ids))
    if getattr(args, 'image_ids_file', None):
        image_ids = []
        with open(args.image_ids_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    image_ids.append(int(line))
        return list(dict.fromkeys(image_ids))
    saved = [int(record['image_id']) for record in manifest.get('saved_records', [])]
    return saved[: max(1, int(getattr(args, 'max_images', len(saved))))]


def load_entry(dump_dir: Path, image_id: int) -> Dict[str, Any]:
    return torch.load(dump_dir / 'per_image' / f'{image_id}.pt', map_location='cpu')


def iter_entries(dump_dir: Path, manifest: Dict[str, Any]):
    for record in manifest.get('saved_records', []):
        yield load_entry(dump_dir, int(record['image_id']))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_entry_probs(entry: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    meta = entry['config_meta']
    pred_logits = entry['outputs']['pred_logits'].unsqueeze(0)
    pred_obj = entry['outputs']['pred_obj'].unsqueeze(0)
    pred_known = entry['outputs'].get('pred_known', None)
    if pred_known is not None:
        pred_known = pred_known.unsqueeze(0)

    invalid = meta['invalid_cls_logits']
    logits = pred_logits.clone()
    if len(invalid) > 0:
        logits[:, :, invalid] = -10e10
    class_prob = logits.sigmoid()
    if len(invalid) > 0:
        class_prob[:, :, invalid] = 0.0
    if class_prob.shape[-1] > 0:
        class_prob[:, :, -1] = 0.0

    obj_prob = torch.exp(-meta['obj_temperature'] * pred_obj).clamp(min=1e-6, max=1.0)
    if pred_known is None:
        knownness_prob = torch.ones_like(obj_prob)
    else:
        knownness_prob = torch.exp(-meta['known_temperature'] * pred_known).clamp(min=1e-6, max=1.0)
    unknown_prob = (1.0 - knownness_prob).clamp(min=0.0, max=1.0)
    if class_prob.shape[-1] > 1:
        max_known_cls_prob = class_prob[:, :, :-1].max(dim=-1).values
    elif class_prob.shape[-1] > 0:
        max_known_cls_prob = class_prob.squeeze(-1)
    else:
        max_known_cls_prob = torch.zeros_like(obj_prob)
    unknown_score = obj_prob * unknown_prob * float(meta['unknown_scale'])
    return {
        'obj_prob': obj_prob[0],
        'knownness_prob': knownness_prob[0],
        'unknown_prob': unknown_prob[0],
        'max_known_cls_prob': max_known_cls_prob[0],
        'unknown_score': unknown_score[0],
        'class_prob': class_prob[0],
    }


def to_abs_xyxy(boxes_cxcywh: torch.Tensor, orig_size_hw: torch.Tensor) -> torch.Tensor:
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)
    h, w = orig_size_hw.tolist()
    scale = torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype)
    return boxes_xyxy * scale


def open_image(entry: Dict[str, Any]) -> np.ndarray:
    return np.array(Image.open(entry['image_path']).convert('RGB'))


def clamp_box_xyxy(box_xyxy: Sequence[float], img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = min(max(x1, 0.0), float(img_w - 1))
    y1 = min(max(y1, 0.0), float(img_h - 1))
    x2 = min(max(x2, 0.0), float(img_w - 1))
    y2 = min(max(y2, 0.0), float(img_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def is_displayable_box(box_xyxy: Sequence[float], img_w: int, img_h: int) -> bool:
    box = clamp_box_xyxy(box_xyxy, img_w, img_h)
    w = float(box[2] - box[0])
    h = float(box[3] - box[1])
    if w < 12 or h < 12:
        return False
    aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))
    if aspect > 6.0:
        return False
    area_frac = (w * h) / max(float(img_w * img_h), 1.0)
    if area_frac < 0.002 or area_frac > 0.75:
        return False
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    if cx < 0.02 * img_w or cx > 0.98 * img_w:
        return False
    if cy < 0.02 * img_h or cy > 0.98 * img_h:
        return False
    return True


def draw_box(ax, box_xyxy, color: str, label: Optional[str] = None, linewidth: float = 2.5, alpha: float = 1.0):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    rect = patches.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(rect)
    if label:
        ax.text(x1, max(2, y1 - 3), label, color='white', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=0.85))


def classify_detections(entry: Dict[str, Any], score_thr: float, max_det: int, iou_thr: float) -> List[Dict[str, Any]]:
    preds = entry['postprocess']
    gt = entry['targets_raw_abs']
    scores = preds['scores']
    labels = preds['labels']
    boxes = preds['boxes_abs_xyxy']
    gt_boxes = gt['boxes_abs_xyxy']
    gt_labels = gt['labels']
    num_classes = int(entry['config_meta']['num_classes'])
    unknown_label = num_classes - 1

    order = torch.argsort(scores, descending=True)
    used_gt = set()
    records: List[Dict[str, Any]] = []
    for idx in order.tolist():
        if len(records) >= max_det:
            break
        score = float(scores[idx].item())
        if score < score_thr:
            continue
        box = boxes[idx]
        label = int(labels[idx].item())
        best_iou = 0.0
        best_gt = -1
        if gt_boxes.numel() > 0:
            ious = box_ops.box_iou(box.unsqueeze(0), gt_boxes)[0][0]
            best_iou, best_gt_tensor = ious.max(dim=0)
            best_iou = float(best_iou.item())
            best_gt = int(best_gt_tensor.item())
        kind = 'error'
        if best_gt >= 0 and best_iou >= iou_thr and best_gt not in used_gt and label == int(gt_labels[best_gt].item()):
            used_gt.add(best_gt)
            kind = 'unknown' if label == unknown_label else 'known'
        records.append({'box': box, 'label': label, 'score': score, 'kind': kind})
    return records


def categorize_entry_queries(
    entry: Dict[str, Any],
    assign_iou_thr: float = 0.3,
    background_iou_thr: float = 0.1,
) -> torch.Tensor:
    pred_boxes_abs = to_abs_xyxy(entry['outputs']['pred_boxes'], entry['orig_size_hw'])
    gt_boxes = entry['targets_raw_abs']['boxes_abs_xyxy']
    gt_labels = entry['targets_raw_abs']['labels']
    num_queries = pred_boxes_abs.shape[0]
    categories = torch.full((num_queries,), -1, dtype=torch.int64)
    if gt_boxes.numel() == 0:
        categories[:] = 2
        return categories
    num_classes = int(entry['config_meta']['num_classes'])
    unknown_label = num_classes - 1
    ious = box_ops.box_iou(pred_boxes_abs, gt_boxes)[0]
    max_iou_all, _ = ious.max(dim=1)

    known_mask = gt_labels != unknown_label
    unknown_mask = gt_labels == unknown_label

    max_iou_known = torch.zeros(num_queries, dtype=torch.float32)
    max_iou_unknown = torch.zeros(num_queries, dtype=torch.float32)
    if known_mask.any():
        max_iou_known = ious[:, known_mask].max(dim=1).values
    if unknown_mask.any():
        max_iou_unknown = ious[:, unknown_mask].max(dim=1).values

    unknown_q = (max_iou_unknown >= assign_iou_thr) & (max_iou_unknown >= max_iou_known)
    known_q = (max_iou_known >= assign_iou_thr) & (~unknown_q)
    background_q = (max_iou_all < background_iou_thr)

    categories[unknown_q] = 1
    categories[known_q] = 0
    categories[(categories < 0) & background_q] = 2
    return categories


def legend_patches(items: Sequence[Tuple[str, str]]) -> List[Any]:
    return [patches.Patch(facecolor='none', edgecolor=color, label=label, linewidth=2.0) for label, color in items]


def auto_query_index(entry: Dict[str, Any]) -> int:
    probs = compute_entry_probs(entry)
    final_boxes = to_abs_xyxy(entry['outputs']['pred_boxes'], entry['orig_size_hw']).numpy()
    img_h, img_w = [int(v) for v in entry['orig_size_hw'].tolist()]
    categories = categorize_entry_queries(entry).numpy()
    valid = np.array([is_displayable_box(final_boxes[i], img_w, img_h) for i in range(final_boxes.shape[0])], dtype=bool)

    unknown_candidates = np.where((categories == 1) & valid)[0]
    if len(unknown_candidates) > 0:
        scores = probs['unknown_score'][unknown_candidates]
        return int(unknown_candidates[int(torch.argmax(scores).item())])

    known_candidates = np.where((categories == 0) & valid)[0]
    if len(known_candidates) > 0:
        scores = probs['obj_prob'][known_candidates] * probs['max_known_cls_prob'][known_candidates]
        return int(known_candidates[int(torch.argmax(scores).item())])

    pseudo_pos = [int(q) for q in entry['pseudo_mining'].get('selected_pseudo_pos', []) if q < len(valid) and valid[q]]
    if len(pseudo_pos) > 0:
        pseudo_tensor = torch.as_tensor(pseudo_pos, dtype=torch.long)
        scores = probs['unknown_score'][pseudo_tensor]
        return int(pseudo_pos[int(torch.argmax(scores).item())])

    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 0:
        valid_tensor = torch.as_tensor(valid_indices, dtype=torch.long)
        scores = probs['unknown_score'][valid_tensor]
        return int(valid_indices[int(torch.argmax(scores).item())])

    return int(torch.argmax(probs['unknown_score']).item())


def render_overlay(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    image_ids = parse_image_ids(args, manifest)
    for image_id in image_ids:
        entry = load_entry(dump_dir, image_id)
        image = open_image(entry)
        dets = classify_detections(entry, score_thr=args.score_thr, max_det=args.max_det, iou_thr=args.iou_thr)
        matched = [det for det in dets if det['kind'] != 'error']
        errors = [det for det in dets if det['kind'] == 'error'][: max(0, int(args.max_error_det))]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.axis('off')
        for det in matched + errors:
            color = KNOWN_COLOR if det['kind'] == 'known' else UNKNOWN_COLOR if det['kind'] == 'unknown' else ERROR_COLOR
            label_text = f"{det['label']}:{det['score']:.2f}"
            draw_box(ax, det['box'], color=color, label=label_text)
        ax.legend(handles=legend_patches([
            ('Correct known detection', KNOWN_COLOR),
            ('Correct unknown detection', UNKNOWN_COLOR),
            ('High-score error', ERROR_COLOR),
        ]), loc='upper right', framealpha=0.9)
        ax.set_title(f'image_id={image_id}')
        fig.tight_layout()
        fig.savefig(output_dir / f'{image_id}_overlay.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def render_mining(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    image_ids = parse_image_ids(args, manifest)
    for image_id in image_ids:
        entry = load_entry(dump_dir, image_id)
        image = open_image(entry)
        pred_boxes_abs = to_abs_xyxy(entry['outputs']['pred_boxes'], entry['orig_size_hw'])
        mining = entry['pseudo_mining']
        stages = [
            ('IoU/geometry filtered', mining.get('valid_after_geom', [])),
            ('Unknown/known filtered', mining.get('valid_after_unknown', [])),
            ('Selected pseudo positives', mining.get('selected_pseudo_pos', [])),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (title, indices) in zip(axes, stages):
            ax.imshow(image)
            ax.axis('off')
            for query_idx in indices:
                draw_box(ax, pred_boxes_abs[query_idx], color=STAGE_COLOR, label=None, linewidth=2.0, alpha=0.95)
            ax.legend(handles=legend_patches([('Stage candidate boxes', STAGE_COLOR)]), loc='upper right', framealpha=0.9)
            ax.set_title(f'{title}\ncount={len(indices)}')
        fig.tight_layout()
        fig.savefig(output_dir / f'{image_id}_mining_panel.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def aggregate_per_image_stats(dump_dir: Path, manifest: Dict[str, Any], assign_iou_thr: float, background_iou_thr: float) -> Dict[str, np.ndarray]:
    obj_values = []
    unk_values = []
    cls_values = []
    groups = []
    for entry in iter_entries(dump_dir, manifest):
        probs = compute_entry_probs(entry)
        categories = categorize_entry_queries(entry, assign_iou_thr=assign_iou_thr, background_iou_thr=background_iou_thr)
        valid = categories >= 0
        if not valid.any():
            continue
        obj_values.append(probs['obj_prob'][valid].cpu().numpy())
        unk_values.append(probs['unknown_prob'][valid].cpu().numpy())
        cls_values.append(probs['max_known_cls_prob'][valid].cpu().numpy())
        groups.append(categories[valid].cpu().numpy())
    if len(groups) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {'obj_prob': empty, 'unknown_prob': empty, 'max_known_cls_prob': empty, 'group': np.zeros((0,), dtype=np.int64)}
    return {
        'obj_prob': np.concatenate(obj_values, axis=0),
        'unknown_prob': np.concatenate(unk_values, axis=0),
        'max_known_cls_prob': np.concatenate(cls_values, axis=0),
        'group': np.concatenate(groups, axis=0).astype(np.int64),
    }


def render_histograms(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    stats = aggregate_per_image_stats(dump_dir, manifest, float(args.assign_iou_thr), float(args.background_iou_thr))
    group = stats['group']
    metrics = [
        ('obj_prob', 'Objectness probability'),
        ('unknown_prob', 'Unknown probability'),
        ('max_known_cls_prob', 'Max known-class probability'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, metrics):
        values = stats[key]
        ax.hist(values[group == 0], bins=args.bins, density=True, alpha=0.75, label='Known', color=KNOWN_COLOR)
        ax.hist(values[group == 1], bins=args.bins, density=True, alpha=0.75, label='Unknown', color=UNKNOWN_COLOR)
        ax.hist(values[group == 2], bins=args.bins, density=True, alpha=0.75, label='Background', color=BACKGROUND_COLOR)
        ax.set_title(title)
        ax.set_xlabel(key)
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir / 'ch3_histograms.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def render_box_evolution(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    image_ids = parse_image_ids(args, manifest)
    for image_id in image_ids:
        entry = load_entry(dump_dir, image_id)
        image = open_image(entry)
        img_h, img_w = image.shape[:2]
        query_idx = int(args.query_index) if args.query_index is not None else auto_query_index(entry)
        aux_outputs = entry['aux_outputs']
        layers = [aux['pred_boxes'][query_idx] for aux in aux_outputs]
        layers.append(entry['outputs']['pred_boxes'][query_idx])
        abs_boxes = [clamp_box_xyxy(to_abs_xyxy(box.unsqueeze(0), entry['orig_size_hw'])[0], img_w, img_h) for box in layers]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.axis('off')
        legend_handles = []
        for layer_index, box in enumerate(abs_boxes):
            color = LEVEL_COLORS[layer_index % len(LEVEL_COLORS)]
            lw = 1.5 + 0.4 * layer_index
            if (box[2] - box[0]) >= 2 and (box[3] - box[1]) >= 2:
                draw_box(ax, box, color=color, label=f'L{layer_index + 1}', linewidth=lw, alpha=0.95)
            legend_handles.append(Line2D([0], [0], color=color, lw=lw, label=f'Decoder L{layer_index + 1}'))
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)
        ax.set_title(f'Query {query_idx} box evolution')
        fig.tight_layout()
        fig.savefig(output_dir / f'{image_id}_query{query_idx}_box_evolution.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def _sampling_locations_from_layer(entry: Dict[str, Any], layer_index: int, query_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hooks = entry['odqe_hooks']
    meta = hooks['meta']
    layer = hooks['layers'][layer_index]
    n_heads = int(meta['n_heads'])
    n_levels = int(meta['n_levels'])
    n_points = int(meta['n_points'])
    reference_points = layer['reference_points'][query_idx]
    offsets = layer['sampling_offsets_raw'][query_idx].view(n_heads, n_levels, n_points, 2)
    attn_logits = layer['attention_logits_raw'][query_idx].view(n_heads, n_levels * n_points)
    attn = torch.softmax(attn_logits, dim=-1).view(n_heads, n_levels, n_points)
    spatial_shapes = layer['spatial_shapes']
    if reference_points.shape[-1] == 2:
        normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1).float()
        locations = reference_points[None, :, None, :] + offsets / normalizer[None, :, None, :]
    else:
        locations = reference_points[None, :, None, :2] + offsets / float(n_points) * reference_points[None, :, None, 2:] * 0.5
    orig_h, orig_w = [float(v) for v in entry['orig_size_hw'].tolist()]
    xy = locations.clone()
    xy[..., 0] *= orig_w
    xy[..., 1] *= orig_h
    level_centroids = []
    level_spreads = []
    for lvl in range(n_levels):
        lvl_weights = attn[:, lvl, :].reshape(-1)
        lvl_points = xy[:, lvl, :, :].reshape(-1, 2)
        weight_sum = float(lvl_weights.sum().item()) + 1e-6
        centroid = (lvl_weights[:, None] * lvl_points).sum(dim=0) / weight_sum
        deltas = lvl_points - centroid[None, :]
        spread = torch.sqrt((lvl_weights[:, None] * (deltas ** 2)).sum(dim=0) / weight_sum)
        level_centroids.append(centroid.numpy())
        level_spreads.append(spread.numpy())
    return xy.numpy(), attn.numpy(), np.stack(level_centroids, axis=0), np.stack(level_spreads, axis=0)


def render_odqe_sampling(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    image_ids = parse_image_ids(args, manifest)
    for image_id in image_ids:
        entry = load_entry(dump_dir, image_id)
        if not entry['odqe_hooks']:
            continue
        image = open_image(entry)
        layer_count = len(entry['odqe_hooks']['layers'])
        layer_index = layer_count - 1 if int(args.layer_index) < 0 else int(args.layer_index)
        query_idx = int(args.query_index) if args.query_index is not None else auto_query_index(entry)
        points_xy, attn, centroids, spreads = _sampling_locations_from_layer(entry, layer_index, query_idx)
        flat_points = points_xy.reshape(-1, 2)
        flat_weights = attn.reshape(-1)
        topk = min(int(args.top_points), flat_weights.shape[0])
        order = np.argsort(flat_weights)[::-1][:topk]
        top_points = flat_points[order]
        top_weights = flat_weights[order]
        query_box = to_abs_xyxy(entry['outputs']['pred_boxes'][query_idx].unsqueeze(0), entry['orig_size_hw'])[0].numpy()
        qx = 0.5 * (query_box[0] + query_box[2])
        qy = 0.5 * (query_box[1] + query_box[3])

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax in axes:
            ax.imshow(image)
            ax.axis('off')
            draw_box(ax, query_box, color=QUERY_COLOR, label=f'Q{query_idx}', linewidth=2.5)
        for lvl, centroid in enumerate(centroids):
            color = LEVEL_COLORS[lvl % len(LEVEL_COLORS)]
            axes[0].scatter([centroid[0]], [centroid[1]], s=90, color=color, marker='o')
            axes[0].annotate('', xy=centroid, xytext=(qx, qy), arrowprops=dict(arrowstyle='->', color=color, lw=2.0))
            spread = spreads[lvl]
            ellipse = patches.Ellipse((centroid[0], centroid[1]), width=max(8.0, spread[0] * 4), height=max(8.0, spread[1] * 4),
                                      fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
            axes[0].add_patch(ellipse)
        axes[0].legend(handles=[
            Line2D([0], [0], color=QUERY_COLOR, lw=2.5, label='Selected query box'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=LEVEL_COLORS[0], markersize=8, label='Per-level centroid'),
        ], loc='upper right', framealpha=0.9)
        axes[0].set_title(f'Layer {layer_index + 1} aggregated context sampling')

        x1, y1, x2, y2 = query_box
        margin_x = max(40.0, 0.5 * (x2 - x1))
        margin_y = max(40.0, 0.5 * (y2 - y1))
        axes[1].set_xlim(max(0, x1 - margin_x), min(image.shape[1], x2 + margin_x))
        axes[1].set_ylim(min(image.shape[0], y2 + margin_y), max(0, y1 - margin_y))
        sizes = 40 + 400 * (top_weights / max(top_weights.max(), 1e-6))
        scatter = axes[1].scatter(top_points[:, 0], top_points[:, 1], s=sizes, c=top_weights, cmap='magma', alpha=0.8,
                                  edgecolors='white', linewidths=0.6)
        axes[1].scatter([qx], [qy], s=100, color=QUERY_COLOR, marker='x')
        axes[1].legend(handles=[
            Line2D([0], [0], marker='x', color=QUERY_COLOR, lw=0, markersize=9, label='Query center'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff884d', markersize=8, label='Top weighted sampling points'),
        ], loc='upper right', framealpha=0.9)
        axes[1].set_title('Zoomed top-weighted sampling points')
        fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04, label='Attention weight')
        fig.tight_layout()
        fig.savefig(output_dir / f'{image_id}_layer{layer_index + 1}_query{query_idx}_odqe_sampling.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def compute_gain_for_layer(entry: Dict[str, Any], layer_index: int) -> np.ndarray:
    hooks = entry['odqe_hooks']
    if not hooks:
        raise ValueError('No ODQE hook data present for this entry.')
    layer = hooks['layers'][layer_index]
    gate_logits = layer.get('gate_logits_raw', None)
    if gate_logits is None:
        raise ValueError('No gate logits found for this layer.')
    gate = torch.sigmoid(gate_logits)
    context = layer['context_output']
    decay = hooks['odqe_layer_decay'][layer_index] if len(hooks['odqe_layer_decay']) > layer_index else torch.tensor(1.0)
    gain = decay * gate * context
    return torch.norm(gain, dim=-1).numpy()


def render_gate_gain(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    image_ids = parse_image_ids(args, manifest)
    for image_id in image_ids:
        entry = load_entry(dump_dir, image_id)
        if not entry['odqe_hooks']:
            continue
        image = open_image(entry)
        layer_count = len(entry['odqe_hooks']['layers'])
        layer_index = layer_count - 1 if int(args.layer_index) < 0 else int(args.layer_index)
        gain = compute_gain_for_layer(entry, layer_index)
        topk = min(int(args.top_queries), gain.shape[0])
        order = np.argsort(gain)[::-1][:topk]
        abs_boxes = to_abs_xyxy(entry['outputs']['pred_boxes'], entry['orig_size_hw']).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(image)
        axes[0].axis('off')
        for rank, query_idx in enumerate(order):
            cmap_val = float(rank) / max(1, topk - 1)
            color = plt.cm.viridis(1.0 - cmap_val)
            label = f'Q{query_idx}:{gain[query_idx]:.2f}'
            draw_box(axes[0], abs_boxes[query_idx], color=color, label=label, linewidth=2.2)
        axes[0].legend(handles=[
            patches.Patch(facecolor='none', edgecolor=plt.cm.viridis(1.0), linewidth=2.2, label='Top-gain query boxes')
        ], loc='upper right', framealpha=0.9)
        axes[0].set_title(f'Layer {layer_index + 1} top-{topk} context gains')
        axes[1].bar(np.arange(topk), gain[order])
        axes[1].set_xticks(np.arange(topk))
        axes[1].set_xticklabels([f'Q{q}' for q in order], rotation=45, ha='right')
        axes[1].set_ylabel('||decay * gate * context||')
        axes[1].set_title('Query gain magnitude ranking')
        fig.tight_layout()
        fig.savefig(output_dir / f'{image_id}_layer{layer_index + 1}_gate_gain.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def sample_global_points(stats: Dict[str, np.ndarray], max_points: int) -> Dict[str, np.ndarray]:
    total = int(stats['group'].shape[0])
    if total <= max_points:
        indices = np.arange(total)
    else:
        rng = np.random.default_rng(0)
        indices = rng.choice(total, size=max_points, replace=False)
    return {key: value[indices] for key, value in stats.items()}


def render_decorr(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    stats = aggregate_per_image_stats(dump_dir, manifest, assign_iou_thr=0.3, background_iou_thr=0.1)
    sampled = sample_global_points(stats, max_points=int(args.max_points))
    colors = {0: KNOWN_COLOR, 1: UNKNOWN_COLOR, 2: BACKGROUND_COLOR}
    labels = {0: 'Known', 1: 'Unknown', 2: 'Background'}
    pairs = [
        ('max_known_cls_prob', 'unknown_prob', 'Cls max vs Unknown prob'),
        ('max_known_cls_prob', 'obj_prob', 'Cls max vs Obj prob'),
        ('obj_prob', 'unknown_prob', 'Obj prob vs Unknown prob'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (x_key, y_key, title) in zip(axes.flat[:3], pairs):
        for group_id in [0, 1, 2]:
            mask = sampled['group'] == group_id
            ax.scatter(sampled[x_key][mask], sampled[y_key][mask], s=8, alpha=0.45, color=colors[group_id], label=labels[group_id])
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(title)
        ax.legend(loc='upper right')
    data_mat = np.stack([stats['max_known_cls_prob'], stats['obj_prob'], stats['unknown_prob']], axis=0)
    corr = np.corrcoef(data_mat)
    im = axes[1, 1].imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_xticklabels(['cls_max', 'obj_prob', 'unk_prob'])
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(['cls_max', 'obj_prob', 'unk_prob'])
    axes[1, 1].set_title('Correlation heatmap')
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='black')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / 'decorr_2d_heatmap.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for group_id in [0, 1, 2]:
        mask = sampled['group'] == group_id
        ax.scatter(sampled['obj_prob'][mask], sampled['unknown_prob'][mask], sampled['max_known_cls_prob'][mask], s=8,
                   alpha=0.4, color=colors[group_id], label=labels[group_id])
    ax.set_xlabel('obj_prob')
    ax.set_ylabel('unknown_prob')
    ax.set_zlabel('cls_max')
    ax.set_title('3D semantic scatter')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir / 'decorr_3d_scatter.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def _mahalanobis_ellipsoid(points: np.ndarray, quantile: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = points.mean(axis=0)
    cov = np.cov(points.T)
    cov += 1e-6 * np.eye(3)
    inv_cov = np.linalg.inv(cov)
    centered = points - mu[None, :]
    d2 = np.einsum('ni,ij,nj->n', centered, inv_cov, centered)
    radius = math.sqrt(float(np.quantile(d2, quantile)))
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack([x, y, z], axis=-1)
    transform = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-8, None))) * radius
    ellipsoid = sphere.reshape(-1, 3) @ transform.T + mu[None, :]
    ellipsoid = ellipsoid.reshape(x.shape + (3,))
    return mu, cov, ellipsoid


def _kde_shell(points: np.ndarray, grid_size: int, density_quantile: float, band_ratio: float) -> np.ndarray:
    try:
        from scipy.stats import gaussian_kde
    except Exception as exc:
        raise RuntimeError('scipy is required for KDE shell rendering') from exc
    kde = gaussian_kde(points.T)
    mins = points.min(axis=0) - 0.1
    maxs = points.max(axis=0) + 0.1
    xs = np.linspace(mins[0], maxs[0], grid_size)
    ys = np.linspace(mins[1], maxs[1], grid_size)
    zs = np.linspace(mins[2], maxs[2], grid_size)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)
    grid_density = kde(grid.T)
    point_density = kde(points.T)
    threshold = float(np.quantile(point_density, density_quantile))
    band = max(1e-8, threshold * band_ratio)
    mask = np.abs(grid_density - threshold) <= band
    return grid[mask]


def render_manifold(args) -> None:
    dump_dir = Path(args.dump_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    manifest = read_manifest(dump_dir)
    stats = aggregate_per_image_stats(dump_dir, manifest, assign_iou_thr=0.3, background_iou_thr=0.1)
    sampled = sample_global_points(stats, max_points=int(args.max_points))
    points = np.stack([sampled['obj_prob'], sampled['unknown_prob'], sampled['max_known_cls_prob']], axis=1)
    groups = sampled['group']
    colors = {0: KNOWN_COLOR, 1: UNKNOWN_COLOR, 2: BACKGROUND_COLOR}
    labels = {0: 'Known', 1: 'Unknown', 2: 'Background'}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for group_id in [0, 1, 2]:
        mask = groups == group_id
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], s=8, alpha=0.35, color=colors[group_id], label=labels[group_id])
    ax.set_xlabel('obj_prob')
    ax.set_ylabel('unknown_prob')
    ax.set_zlabel('cls_max')
    ax.set_title('3D semantic manifold scatter')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir / 'manifold_scatter.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    known_points = points[groups == 0]
    unknown_points = points[groups == 1]
    background_points = points[groups == 2]
    if known_points.shape[0] >= 8:
        _mu, _cov, ellipsoid = _mahalanobis_ellipsoid(known_points, quantile=float(args.ellipsoid_quantile))
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2], color=KNOWN_COLOR, linewidth=0.7, alpha=0.45)
        ax.scatter(known_points[:, 0], known_points[:, 1], known_points[:, 2], s=6, alpha=0.25, color=KNOWN_COLOR, label='Known')
        if len(unknown_points) > 0:
            ax.scatter(unknown_points[:, 0], unknown_points[:, 1], unknown_points[:, 2], s=8, alpha=0.45, color=UNKNOWN_COLOR, label='Unknown')
        if len(background_points) > 0:
            ax.scatter(background_points[:, 0], background_points[:, 1], background_points[:, 2], s=6, alpha=0.2, color=BACKGROUND_COLOR, label='Background')
        ax.set_xlabel('obj_prob')
        ax.set_ylabel('unknown_prob')
        ax.set_zlabel('cls_max')
        ax.set_title('Mahalanobis ellipsoid over known manifold')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(output_dir / 'manifold_ellipsoid.png', dpi=220, bbox_inches='tight')
        plt.close(fig)

        shell_points = _kde_shell(
            known_points,
            grid_size=int(args.kde_grid_size),
            density_quantile=float(args.kde_density_quantile),
            band_ratio=float(args.kde_band_ratio),
        )
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if len(shell_points) > 0:
            ax.scatter(shell_points[:, 0], shell_points[:, 1], shell_points[:, 2], s=5, alpha=0.08, color=KNOWN_COLOR, label='KDE shell')
        ax.scatter(known_points[:, 0], known_points[:, 1], known_points[:, 2], s=5, alpha=0.12, color=KNOWN_COLOR, label='Known')
        if len(unknown_points) > 0:
            ax.scatter(unknown_points[:, 0], unknown_points[:, 1], unknown_points[:, 2], s=8, alpha=0.45, color=UNKNOWN_COLOR, label='Unknown')
        if len(background_points) > 0:
            ax.scatter(background_points[:, 0], background_points[:, 1], background_points[:, 2], s=6, alpha=0.15, color=BACKGROUND_COLOR, label='Background')
        ax.set_xlabel('obj_prob')
        ax.set_ylabel('unknown_prob')
        ax.set_zlabel('cls_max')
        ax.set_title('KDE density shell over known manifold')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(output_dir / 'manifold_kde_shell.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    command_to_fn = {
        'overlay': render_overlay,
        'mining': render_mining,
        'histograms': render_histograms,
        'box_evolution': render_box_evolution,
        'odqe_sampling': render_odqe_sampling,
        'gate_gain': render_gate_gain,
        'decorr': render_decorr,
        'manifold': render_manifold,
    }
    command_to_fn[args.command](args)


if __name__ == '__main__':
    main()