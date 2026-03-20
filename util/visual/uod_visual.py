import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor_images(samples: Any) -> torch.Tensor:
    if hasattr(samples, 'tensors'):
        return samples.tensors
    if torch.is_tensor(samples):
        return samples
    raise TypeError(f'Unsupported samples type for visualization: {type(samples)!r}')


def _unnormalize_image(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().float()
    if img.ndim != 3:
        raise ValueError(f'Expected image tensor [C,H,W], got {tuple(img.shape)}')
    mean = IMAGENET_MEAN.to(img)
    std = IMAGENET_STD.to(img)
    vis = (img * std + mean).clamp(0, 1)
    return vis.permute(1, 2, 0).numpy()


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def _scale_boxes_xyxy(boxes_xyxy: torch.Tensor, image_hw: torch.Tensor) -> torch.Tensor:
    h = float(image_hw[0].item())
    w = float(image_hw[1].item())
    scale = boxes_xyxy.new_tensor([w, h, w, h])
    return boxes_xyxy * scale


def _pairwise_iof(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)).clamp(min=eps)
    return inter / area1[:, None]


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)).clamp(min=eps)
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)).clamp(min=eps)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=eps)


def _compute_uod_snapshot(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], criterion, epoch: int) -> List[Dict[str, Any]]:
    if 'pred_obj' not in outputs or 'pred_boxes' not in outputs or 'pred_logits' not in outputs:
        return []

    outputs_without_aux = {
        k: v for k, v in outputs.items()
        if k not in ['aux_outputs', 'enc_outputs', 'pred_obj', 'pred_unk', 'samples', 'pred_proj', 'pred_embed']
    }
    indices = criterion.matcher(outputs_without_aux, targets)

    obj_scores = outputs['pred_obj'].detach()
    pred_boxes = outputs['pred_boxes'].detach()
    pred_probs = outputs['pred_logits'].detach().sigmoid().clone()
    invalid_cls = getattr(criterion, 'invalid_cls_logits', None)
    if invalid_cls is not None:
        pred_probs[:, :, invalid_cls] = 0.0

    num_queries = obj_scores.shape[1]
    pos_quantile = getattr(criterion, 'unk_label_pos_quantile', getattr(criterion.args, 'unk_label_pos_quantile', 0.5))
    known_reject_thresh = getattr(criterion, 'unk_cls_reject_thresh', getattr(criterion.args, 'unk_cls_reject_thresh', 0.25))
    max_pos_per_img = int(getattr(criterion.args, 'unk_pos_per_img', 1))
    max_neg_per_img = int(getattr(criterion.args, 'unk_neg_per_img', 1))
    obj_enabled = bool(getattr(criterion, 'enable_unk_label_obj', False) and getattr(criterion, 'unk_label_start_epoch', 0) <= epoch)

    snapshots: List[Dict[str, Any]] = []
    for b_idx, (src_idx, _) in enumerate(indices):
        matched = sorted(src_idx.tolist())
        matched_set = set(matched)
        unmatched = [q for q in range(num_queries) if q not in matched_set]

        gt_boxes = targets[b_idx].get('boxes', torch.empty((0, 4), device=pred_boxes.device))
        gt_xyxy = _cxcywh_to_xyxy(gt_boxes) if gt_boxes.numel() > 0 else gt_boxes.new_zeros((0, 4))
        pred_xyxy = _cxcywh_to_xyxy(pred_boxes[b_idx])

        if len(src_idx) > 0:
            matched_scores = obj_scores[b_idx, src_idx]
            base_thresh = torch.quantile(matched_scores, pos_quantile).item()
            pos_thresh = base_thresh * float(getattr(criterion, 'unk_label_obj_score_thresh', 1.0))
        else:
            pos_thresh = float(getattr(criterion.args, 'default_pos_energy_thresh', 1.0))
        neg_thresh = pos_thresh + float(getattr(criterion, 'bg_neg_score_margin', getattr(criterion.args, 'bg_neg_score_margin', 0.5)))

        unmatched_iou_map = {q: 0.0 for q in unmatched}
        unmatched_iof_map = {q: 0.0 for q in unmatched}
        rejected_gt, rejected_geom, valid_unmatched = [], [], []

        if len(unmatched) > 0:
            cand = pred_xyxy[unmatched]
            if gt_xyxy.numel() > 0:
                ious = _pairwise_iou(cand, gt_xyxy)
                iofs = _pairwise_iof(cand, gt_xyxy)
                max_ious = ious.max(dim=1)[0]
                max_iofs = iofs.max(dim=1)[0]
            else:
                max_ious = cand.new_zeros((cand.shape[0],))
                max_iofs = cand.new_zeros((cand.shape[0],))

            for j, q in enumerate(unmatched):
                unmatched_iou_map[q] = float(max_ious[j].item())
                unmatched_iof_map[q] = float(max_iofs[j].item())
                if unmatched_iou_map[q] >= float(getattr(criterion, 'unk_max_iou', getattr(criterion.args, 'unk_max_iou', 0.3))) or \
                   unmatched_iof_map[q] >= float(getattr(criterion, 'unk_max_iof', getattr(criterion.args, 'unk_max_iof', 0.6))):
                    rejected_gt.append(q)
                    continue
                geom_ok = criterion._is_valid_unknown_geometry(pred_boxes[b_idx, q]) if hasattr(criterion, '_is_valid_unknown_geometry') else True
                if not geom_ok:
                    rejected_geom.append(q)
                    continue
                valid_unmatched.append(q)

        known_prob = pred_probs[b_idx, :, :criterion.num_classes - 1]
        known_max = known_prob.max(dim=-1)[0]
        low_energy_queries = [q for q in valid_unmatched if obj_scores[b_idx, q].item() < pos_thresh and known_max[q].item() < known_reject_thresh]
        low_energy_known_mean = float(sum(known_max[q].item() for q in low_energy_queries) / max(len(low_energy_queries), 1))
        if hasattr(criterion, '_image_gate_open'):
            gate_open, valid_ratio, low_energy_ratio = criterion._image_gate_open(len(valid_unmatched), len(unmatched), len(low_energy_queries), low_energy_known_mean)
        else:
            gate_open, valid_ratio, low_energy_ratio = (True, float(len(valid_unmatched)) / max(len(unmatched), 1), float(len(low_energy_queries)) / max(len(valid_unmatched), 1))

        pos_candidates = []
        pos_weights = []
        if obj_enabled and gate_open:
            for q in low_energy_queries:
                energy_rel = max(0.0, min(1.0, (pos_thresh - obj_scores[b_idx, q].item()) / max(pos_thresh, 1e-6)))
                known_rel = max(0.0, min(1.0, (known_reject_thresh - known_max[q].item()) / max(known_reject_thresh, 1e-6)))
                iou_rel = 1.0 - max(0.0, min(1.0, unmatched_iou_map[q] / max(float(getattr(criterion, 'unk_max_iou', getattr(criterion.args, 'unk_max_iou', 0.3))), 1e-6)))
                conf = 0.5 * energy_rel + 0.3 * known_rel + 0.2 * iou_rel
                pos_candidates.append((q, conf))
            pos_candidates.sort(key=lambda x: (-x[1], obj_scores[b_idx, x[0]].item(), known_max[x[0]].item()))
            pos_candidates = pos_candidates[:max_pos_per_img]
            pos_weights = [float(w) for _, w in pos_candidates]
        dummy_pos = [q for q, _ in pos_candidates]

        neg_candidates = []
        if obj_enabled:
            neg_candidates = [q for q in valid_unmatched if obj_scores[b_idx, q].item() > neg_thresh and known_max[q].item() < known_reject_thresh]
            neg_candidates.sort(key=lambda q: (-obj_scores[b_idx, q].item(), known_max[q].item(), unmatched_iou_map[q]))
        dummy_neg = neg_candidates[:max_neg_per_img]

        snapshots.append({
            'batch_index': b_idx,
            'matched': matched,
            'unmatched': unmatched,
            'valid_unmatched': valid_unmatched,
            'rejected_gt': rejected_gt,
            'rejected_geom': rejected_geom,
            'low_energy': low_energy_queries,
            'dummy_pos': dummy_pos,
            'dummy_neg': dummy_neg,
            'dummy_pos_weights': pos_weights,
            'pos_thresh': float(pos_thresh),
            'neg_thresh': float(neg_thresh),
            'gate_open': bool(gate_open),
            'valid_ratio': float(valid_ratio),
            'low_energy_ratio': float(low_energy_ratio),
            'known_max': known_max.detach().cpu(),
            'obj_scores': obj_scores[b_idx].detach().cpu(),
            'iou_map': unmatched_iou_map,
            'iof_map': unmatched_iof_map,
        })
    return snapshots


def _draw_boxes(ax, boxes_xyxy: torch.Tensor, color: str, linewidth: float = 1.6, linestyle: str = '-', labels: Optional[List[str]] = None):
    if boxes_xyxy.numel() == 0:
        return
    boxes = boxes_xyxy.detach().cpu().numpy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = Rectangle((x1, y1), max(x2 - x1, 1.0), max(y2 - y1, 1.0), fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        ax.add_patch(rect)
        if labels is not None and idx < len(labels) and labels[idx]:
            ax.text(x1, max(0, y1 - 2), labels[idx], color=color, fontsize=7, bbox=dict(facecolor='black', alpha=0.35, pad=1, edgecolor='none'))


def _save_overlay_grid(images: torch.Tensor, targets: List[Dict[str, torch.Tensor]], outputs: Dict[str, torch.Tensor], snapshots: List[Dict[str, Any]], save_path: str, max_images: int = 4):
    n = min(max_images, images.shape[0], len(snapshots))
    if n <= 0:
        return
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for plot_idx in range(rows * cols):
        ax = axes.flat[plot_idx]
        if plot_idx >= n:
            ax.axis('off')
            continue
        img = _unnormalize_image(images[plot_idx])
        ax.imshow(img)
        ax.axis('off')

        h, w = img.shape[:2]
        hw = torch.tensor([h, w], dtype=outputs['pred_boxes'].dtype, device=outputs['pred_boxes'].device)
        pred_xyxy = _scale_boxes_xyxy(_cxcywh_to_xyxy(outputs['pred_boxes'][plot_idx].detach()), hw)
        gt_xyxy = _scale_boxes_xyxy(_cxcywh_to_xyxy(targets[plot_idx]['boxes'].detach()), hw) if len(targets[plot_idx]['boxes']) > 0 else torch.empty((0, 4))
        snap = snapshots[plot_idx]

        _draw_boxes(ax, gt_xyxy, color='lime', linewidth=2.0, labels=[f'gt:{int(x)}' for x in targets[plot_idx]['labels'].detach().cpu().tolist()])
        _draw_boxes(ax, pred_xyxy[snap['matched']] if len(snap['matched']) > 0 else pred_xyxy.new_zeros((0, 4)), color='cyan', linewidth=1.5, labels=[f'm{q}' for q in snap['matched']])
        _draw_boxes(ax, pred_xyxy[snap['dummy_pos']] if len(snap['dummy_pos']) > 0 else pred_xyxy.new_zeros((0, 4)), color='gold', linewidth=1.8, labels=[f'p{q}' for q in snap['dummy_pos']])
        _draw_boxes(ax, pred_xyxy[snap['dummy_neg']] if len(snap['dummy_neg']) > 0 else pred_xyxy.new_zeros((0, 4)), color='red', linewidth=1.4, labels=[f'n{q}' for q in snap['dummy_neg']])
        _draw_boxes(ax, pred_xyxy[snap['rejected_gt']] if len(snap['rejected_gt']) > 0 else pred_xyxy.new_zeros((0, 4)), color='magenta', linewidth=1.0, linestyle='--', labels=[f'i{q}' for q in snap['rejected_gt']])
        _draw_boxes(ax, pred_xyxy[snap['rejected_geom']] if len(snap['rejected_geom']) > 0 else pred_xyxy.new_zeros((0, 4)), color='orange', linewidth=1.0, linestyle=':', labels=[f'g{q}' for q in snap['rejected_geom']])

        title = (
            f"img={plot_idx} gate={int(snap['gate_open'])} "
            f"valid={len(snap['valid_unmatched'])}/{max(len(snap['unmatched']), 1)} "
            f"lowE={len(snap['low_energy'])} pos={len(snap['dummy_pos'])} neg={len(snap['dummy_neg'])}\n"
            f"posT={snap['pos_thresh']:.3f} negT={snap['neg_thresh']:.3f}"
        )
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _save_phase_plot(snapshots: List[Dict[str, Any]], save_path: str):
    xs, ys, colors, labels = [], [], [], []
    color_map = {
        'matched': '#00bcd4',
        'dummy_pos': '#f4c430',
        'dummy_neg': '#d62728',
        'valid_unmatched': '#9e9e9e',
        'rejected_gt': '#e040fb',
        'rejected_geom': '#ff9800',
    }

    for snap in snapshots:
        matched = set(snap['matched'])
        dpos = set(snap['dummy_pos'])
        dneg = set(snap['dummy_neg'])
        vset = set(snap['valid_unmatched'])
        rgt = set(snap['rejected_gt'])
        rgeom = set(snap['rejected_geom'])
        for q in range(len(snap['obj_scores'])):
            xs.append(float(snap['known_max'][q].item()))
            ys.append(float(snap['obj_scores'][q].item()))
            if q in matched:
                labels.append('matched')
            elif q in dpos:
                labels.append('dummy_pos')
            elif q in dneg:
                labels.append('dummy_neg')
            elif q in rgt:
                labels.append('rejected_gt')
            elif q in rgeom:
                labels.append('rejected_geom')
            elif q in vset:
                labels.append('valid_unmatched')
            else:
                labels.append('other')
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ordered = ['matched', 'dummy_pos', 'dummy_neg', 'valid_unmatched', 'rejected_gt', 'rejected_geom']
    for name in ordered:
        idxs = [i for i, lab in enumerate(labels) if lab == name]
        if not idxs:
            continue
        ax.scatter(np.array(xs)[idxs], np.array(ys)[idxs], s=18, alpha=0.65, c=color_map[name], label=name)
    ax.set_xlabel('max known score')
    ax.set_ylabel('objectness energy')
    ax.set_title('UOD query phase plot')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _save_energy_hist(snapshots: List[Dict[str, Any]], save_path: str):
    bins = 30
    data = {'matched': [], 'dummy_pos': [], 'dummy_neg': [], 'valid_unmatched': []}
    for snap in snapshots:
        arr = snap['obj_scores'].numpy()
        data['matched'].extend(arr[snap['matched']].tolist() if len(snap['matched']) > 0 else [])
        data['dummy_pos'].extend(arr[snap['dummy_pos']].tolist() if len(snap['dummy_pos']) > 0 else [])
        data['dummy_neg'].extend(arr[snap['dummy_neg']].tolist() if len(snap['dummy_neg']) > 0 else [])
        v_only = sorted(list(set(snap['valid_unmatched']) - set(snap['dummy_pos']) - set(snap['dummy_neg'])))
        data['valid_unmatched'].extend(arr[v_only].tolist() if len(v_only) > 0 else [])

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for name, vals in data.items():
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.45, density=True, label=name)
    ax.set_xlabel('objectness energy')
    ax.set_ylabel('density')
    ax.set_title('UOD energy distribution')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_training_uod_visualizations(samples: Any, targets: List[Dict[str, torch.Tensor]], outputs: Dict[str, torch.Tensor], criterion, epoch: int, global_step: int, output_dir: str, max_images: int = 4) -> Optional[Dict[str, str]]:
    if output_dir is None or output_dir == '':
        return None
    if 'pred_obj' not in outputs or 'pred_boxes' not in outputs or 'pred_logits' not in outputs:
        return None

    images = _to_tensor_images(samples)
    snapshots = _compute_uod_snapshot(outputs, targets, criterion, epoch)
    if len(snapshots) == 0:
        return None

    base_dir = Path(output_dir) / 'uod_vis' / 'train' / f'epoch_{epoch:03d}' / f'step_{global_step:07d}'
    base_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = str(base_dir / 'mining_overlay.png')
    phase_path = str(base_dir / 'query_phase.png')
    hist_path = str(base_dir / 'energy_hist.png')

    _save_overlay_grid(images, targets, outputs, snapshots, overlay_path, max_images=max_images)
    _save_phase_plot(snapshots, phase_path)
    _save_energy_hist(snapshots, hist_path)

    summary = {
        'overlay': overlay_path,
        'phase': phase_path,
        'hist': hist_path,
    }
    with open(base_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')
    return summary


def _save_eval_prediction_overlay(images: torch.Tensor, targets: List[Dict[str, torch.Tensor]], results: List[Dict[str, torch.Tensor]], class_names: Optional[List[str]], save_path: str, max_images: int = 8, unknown_class_index: Optional[int] = None, score_thresh: float = 0.0):
    n = min(max_images, images.shape[0], len(results))
    if n <= 0:
        return
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for plot_idx in range(rows * cols):
        ax = axes.flat[plot_idx]
        if plot_idx >= n:
            ax.axis('off')
            continue
        img = _unnormalize_image(images[plot_idx])
        ax.imshow(img)
        ax.axis('off')

        gt_boxes = targets[plot_idx].get('boxes', torch.empty((0, 4), device=images.device))
        gt_labels = targets[plot_idx].get('labels', torch.empty((0,), dtype=torch.long, device=images.device))
        h, w = img.shape[:2]
        hw = torch.tensor([h, w], dtype=gt_boxes.dtype if gt_boxes.numel() > 0 else torch.float32, device=images.device)
        gt_xyxy = _scale_boxes_xyxy(_cxcywh_to_xyxy(gt_boxes), hw) if gt_boxes.numel() > 0 else torch.empty((0, 4))
        gt_text = []
        for lab in gt_labels.detach().cpu().tolist():
            if class_names is not None and 0 <= lab < len(class_names):
                gt_text.append(f'gt:{class_names[lab]}')
            else:
                gt_text.append(f'gt:{lab}')
        _draw_boxes(ax, gt_xyxy, color='lime', linewidth=2.0, labels=gt_text)

        pred = results[plot_idx]
        boxes = pred.get('boxes', torch.empty((0, 4))).detach().cpu()
        labels = pred.get('labels', torch.empty((0,), dtype=torch.long)).detach().cpu()
        scores = pred.get('scores', torch.empty((0,))).detach().cpu()
        known_boxes = []
        known_text = []
        unk_boxes = []
        unk_text = []
        for b, l, s in zip(boxes, labels.tolist(), scores.tolist()):
            if s < score_thresh:
                continue
            name = class_names[l] if class_names is not None and 0 <= l < len(class_names) else str(l)
            if unknown_class_index is not None and l == unknown_class_index:
                unk_boxes.append(b)
                unk_text.append(f'u:{name}:{s:.2f}')
            else:
                known_boxes.append(b)
                known_text.append(f'k:{name}:{s:.2f}')
        if len(known_boxes) > 0:
            _draw_boxes(ax, torch.stack(known_boxes), color='deepskyblue', linewidth=1.5, labels=known_text)
        if len(unk_boxes) > 0:
            _draw_boxes(ax, torch.stack(unk_boxes), color='gold', linewidth=1.8, labels=unk_text)
        ax.set_title(f'img={plot_idx} gt={len(gt_text)} predK={len(known_boxes)} predU={len(unk_boxes)}', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_eval_uod_visualizations(samples: Any, targets: List[Dict[str, torch.Tensor]], outputs: Dict[str, torch.Tensor], results: List[Dict[str, torch.Tensor]], criterion, output_dir: str, eval_tag: str, class_names: Optional[List[str]] = None, max_images: int = 8, score_thresh: float = 0.0) -> Optional[Dict[str, str]]:
    if output_dir is None or output_dir == '':
        return None
    if 'pred_obj' not in outputs or 'pred_boxes' not in outputs or 'pred_logits' not in outputs:
        return None
    images = _to_tensor_images(samples)
    snapshots = _compute_uod_snapshot(outputs, targets, criterion, epoch=10**9)
    base_dir = Path(output_dir) / 'uod_vis' / 'eval' / str(eval_tag)
    base_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = str(base_dir / 'known_unknown_overlay.png')
    phase_path = str(base_dir / 'query_phase.png')
    hist_path = str(base_dir / 'energy_hist.png')
    _save_eval_prediction_overlay(
        images=images,
        targets=targets,
        results=results,
        class_names=class_names,
        save_path=overlay_path,
        max_images=max_images,
        unknown_class_index=(len(class_names) - 1) if class_names else None,
        score_thresh=score_thresh,
    )
    _save_phase_plot(snapshots, phase_path)
    _save_energy_hist(snapshots, hist_path)
    summary = {'overlay': overlay_path, 'phase': phase_path, 'hist': hist_path}
    with open(base_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')
    return summary
