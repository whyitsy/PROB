# ------------------------------------------------------------------------
# Training and evaluation loop for official PROB / UOD experiments.
# TensorBoard only. No wandb.
# Adds extensive qualitative and quantitative visualizations.
# ------------------------------------------------------------------------
import csv
import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import util.misc as utils
from util import box_ops
from datasets.data_prefetcher import data_prefetcher
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _call_criterion(criterion, outputs, targets, epoch):
    try:
        return criterion(outputs, targets, epoch)
    except TypeError:
        return criterion(outputs, targets)


def _safe_float(v):
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    return float(v)


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _to_numpy_image(img_tensor, target_hw=None):
    img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    if target_hw is not None:
        h, w = int(target_hw[0]), int(target_hw[1])
        img = img[:h, :w]
    return (img * 255).astype(np.uint8)


def _cxcywh_to_abs_xyxy(boxes, size_hw):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().cpu())
    h, w = int(size_hw[0]), int(size_hw[1])
    scale = torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype)
    boxes_xyxy = boxes_xyxy * scale
    return boxes_xyxy.numpy()


def _label_color(label, unk_label):
    if int(label) == int(unk_label):
        return (220, 40, 40)
    palette = [
        (52, 152, 219), (46, 204, 113), (155, 89, 182), (241, 196, 15),
        (230, 126, 34), (26, 188, 156), (231, 76, 60), (149, 165, 166)
    ]
    return palette[int(label) % len(palette)]


def _draw_boxes(image_np, pred_boxes=None, pred_labels=None, pred_scores=None,
                gt_boxes=None, gt_labels=None, unk_label=80, title=None):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)

    if gt_boxes is not None and len(gt_boxes) > 0:
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = int(gt_labels[i]) if gt_labels is not None else -1
            color = (255, 255, 0) if label != int(unk_label) else (255, 128, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), f'GT:{label}', fill=color)

    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = int(pred_labels[i]) if pred_labels is not None else -1
            score = float(pred_scores[i]) if pred_scores is not None else None
            color = _label_color(label, unk_label)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f'P:{label}' if score is None else f'P:{label} {score:.2f}'
            draw.text((x1 + 2, y1 + 2), text, fill=color)

    if title:
        draw.text((5, 5), title, fill=(255, 255, 255))
    return np.array(img)


def _save_image(np_img, out_path):
    Image.fromarray(np_img).save(out_path)


def _save_contact_sheet(image_paths, out_path, thumb_size=(320, 240), cols=2):
    if not image_paths:
        return
    imgs = []
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
            img.thumbnail(thumb_size)
            imgs.append(img)
        except Exception:
            continue
    if not imgs:
        return
    cols = max(1, cols)
    rows = int(math.ceil(len(imgs) / cols))
    sheet = Image.new('RGB', (cols * thumb_size[0], rows * thumb_size[1]), (20, 20, 20))
    for idx, img in enumerate(imgs):
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        sheet.paste(img, (x, y))
    sheet.save(out_path)


def _save_figure(fig, out_path, writer=None, tb_tag=None, step=0):
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    if writer is not None and tb_tag is not None:
        try:
            writer.add_figure(tb_tag, fig, global_step=step)
        except Exception:
            pass
    plt.close(fig)


def _plot_histograms(vis_state, out_dir, writer=None, step=0):
    if len(vis_state['obj_prob']) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(vis_state['obj_prob'], bins=40)
    axes[0].set_title('Objectness probability')
    axes[1].hist(vis_state['unk_prob'], bins=40)
    axes[1].set_title('Unknownness probability')
    axes[2].hist(vis_state['cls_max'], bins=40)
    axes[2].set_title('Max known-class probability')
    for ax in axes:
        ax.grid(alpha=0.2)
    _save_figure(fig, os.path.join(out_dir, 'hist_probabilities.png'), writer, 'eval_viz/hist_probabilities', step)


def _plot_scatter(vis_state, out_dir, writer=None, step=0):
    if len(vis_state['obj_prob']) == 0:
        return
    groups = np.array(vis_state['group'])
    colors = np.array(['tab:blue', 'tab:red', 'tab:gray'])
    group_names = ['matched-known', 'unmatched-highunk', 'unmatched-other']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for gid in range(3):
        mask = groups == gid
        if np.any(mask):
            axes[0].scatter(np.array(vis_state['obj_prob'])[mask], np.array(vis_state['unk_prob'])[mask],
                            s=8, alpha=0.5, c=colors[gid], label=group_names[gid])
            axes[1].scatter(np.array(vis_state['obj_prob'])[mask], np.array(vis_state['cls_max'])[mask],
                            s=8, alpha=0.5, c=colors[gid], label=group_names[gid])
    axes[0].set_xlabel('obj prob')
    axes[0].set_ylabel('unk prob')
    axes[0].set_title('Objectness vs Unknownness')
    axes[1].set_xlabel('obj prob')
    axes[1].set_ylabel('max known prob')
    axes[1].set_title('Objectness vs Max-known score')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    _save_figure(fig, os.path.join(out_dir, 'scatter_relationships.png'), writer, 'eval_viz/scatter_relationships', step)


def _plot_heatmap(vis_state, out_dir, writer=None, step=0):
    if len(vis_state['obj_prob']) < 4:
        return
    arr = np.stack([vis_state['obj_prob'], vis_state['unk_prob'], vis_state['cls_max']], axis=0)
    corr = np.corrcoef(arr)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['obj', 'unk', 'cls_max'])
    ax.set_yticklabels(['obj', 'unk', 'cls_max'])
    ax.set_title('Prediction correlation heatmap')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='white')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, os.path.join(out_dir, 'correlation_heatmap.png'), writer, 'eval_viz/correlation_heatmap', step)


def _project_2d(x, method='pca', perplexity=30):
    if x.shape[0] < 3:
        return None
    if method == 'tsne':
        perpl = min(perplexity, max(2, x.shape[0] // 4))
        return TSNE(n_components=2, perplexity=perpl, init='pca', learning_rate='auto', random_state=42).fit_transform(x)
    return PCA(n_components=2, random_state=42).fit_transform(x)


def _plot_feature_embeddings(vis_state, out_dir, writer=None, step=0):
    groups = np.array(vis_state['feat_group'])
    group_names = ['matched-known', 'unmatched-highunk', 'unmatched-other']
    colors = np.array(['tab:blue', 'tab:red', 'tab:gray'])
    feat_specs = [
        ('proj_obj', vis_state['proj_obj']),
        ('proj_unk', vis_state['proj_unk']),
        ('proj_cls', vis_state['proj_cls']),
    ]
    for method in ['pca', 'tsne']:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plotted_any = False
        for ax, (name, feat_list) in zip(axes, feat_specs):
            if len(feat_list) < 8:
                ax.set_axis_off()
                continue
            x = np.stack(feat_list, axis=0)
            max_points = 800 if method == 'tsne' else 2000
            if x.shape[0] > max_points:
                idx = np.linspace(0, x.shape[0] - 1, max_points).astype(np.int64)
                x = x[idx]
                g = groups[idx]
            else:
                g = groups
            try:
                emb = _project_2d(x, method=method, perplexity=30)
            except Exception:
                ax.set_axis_off()
                continue
            if emb is None:
                ax.set_axis_off()
                continue
            plotted_any = True
            for gid in range(3):
                mask = g == gid
                if np.any(mask):
                    ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5, c=colors[gid], label=group_names[gid])
            ax.set_title(f'{name} {method.upper()}')
            ax.grid(alpha=0.2)
        if plotted_any:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper center', ncol=3)
            _save_figure(fig, os.path.join(out_dir, f'feature_{method}.png'), writer, f'eval_viz/feature_{method}', step)
        else:
            plt.close(fig)


def _plot_metric_bar(metrics, out_dir, writer=None, step=0):
    if not metrics:
        return
    keys = [k for k in ['U_R50', 'U_AP50', 'WI', 'AOSA', 'K_AP50', 'PK_AP50'] if k in metrics and isinstance(metrics[k], (int, float))]
    if not keys:
        return
    vals = [float(metrics[k]) for k in keys]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(range(len(keys)), vals)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha='right')
    ax.set_title('Current evaluation metrics')
    ax.grid(axis='y', alpha=0.2)
    _save_figure(fig, os.path.join(out_dir, 'current_metrics_bar.png'), writer, 'eval_viz/current_metrics_bar', step)


def _save_query_stats_csv(vis_state, out_dir):
    if len(vis_state['obj_prob']) == 0:
        return
    csv_path = os.path.join(out_dir, 'query_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['obj_prob', 'unk_prob', 'cls_max', 'group'])
        for row in zip(vis_state['obj_prob'], vis_state['unk_prob'], vis_state['cls_max'], vis_state['group']):
            writer.writerow(row)


def _save_feature_npz(vis_state, out_dir):
    if len(vis_state['proj_obj']) == 0:
        return
    np.savez_compressed(
        os.path.join(out_dir, 'feature_samples.npz'),
        proj_obj=np.asarray(vis_state['proj_obj'], dtype=np.float32),
        proj_unk=np.asarray(vis_state['proj_unk'], dtype=np.float32),
        proj_cls=np.asarray(vis_state['proj_cls'], dtype=np.float32),
        feat_group=np.asarray(vis_state['feat_group'], dtype=np.int64),
    )


def _init_eval_viz_state(args):
    max_query_points = int(getattr(args, 'viz_max_query_points', 2500))
    max_feature_points = int(getattr(args, 'viz_max_feature_points', 2500))
    return {
        'saved_images': [],
        'num_saved': 0,
        'obj_prob': [],
        'unk_prob': [],
        'cls_max': [],
        'group': [],
        'proj_obj': [],
        'proj_unk': [],
        'proj_cls': [],
        'feat_group': [],
        'max_query_points': max_query_points,
        'max_feature_points': max_feature_points,
    }


def _append_limited(dst, values, max_len):
    remain = max_len - len(dst)
    if remain <= 0:
        return
    if len(values) > remain:
        values = values[:remain]
    dst.extend(values)


def _collect_eval_stats(vis_state, outputs, targets, criterion, args):
    if len(vis_state['obj_prob']) >= vis_state['max_query_points'] and len(vis_state['proj_obj']) >= vis_state['max_feature_points']:
        return

    obj_prob = torch.exp(-float(getattr(args, 'obj_temp', 1.0)) * outputs['pred_obj'].detach())
    pred_unk = outputs.get('pred_unk', None)
    if pred_unk is not None:
        unk_prob = torch.sigmoid(pred_unk.detach())
    else:
        unk_prob = torch.zeros_like(obj_prob)

    cls_prob = outputs['pred_logits'].detach().sigmoid().clone()
    invalid_cls = getattr(criterion, 'invalid_cls_logits', [])
    if len(invalid_cls) > 0:
        cls_prob[:, :, invalid_cls] = 0.0
    cls_prob[:, :, -1] = 0.0
    cls_max = cls_prob.max(-1).values

    outputs_for_match = {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
    indices = criterion.matcher(outputs_for_match, targets)
    matched_mask = torch.zeros_like(obj_prob, dtype=torch.bool)
    for b, (src, _) in enumerate(indices):
        if len(src) > 0:
            matched_mask[b, src] = True

    obj_np = obj_prob.flatten().cpu().numpy()
    unk_np = unk_prob.flatten().cpu().numpy()
    cls_np = cls_max.flatten().cpu().numpy()
    matched_np = matched_mask.flatten().cpu().numpy()
    group_np = np.where(matched_np, 0, np.where(unk_np > 0.5, 1, 2)).astype(np.int64)

    start = len(vis_state['obj_prob'])
    remain = vis_state['max_query_points'] - start
    if remain > 0:
        _append_limited(vis_state['obj_prob'], obj_np.tolist(), vis_state['max_query_points'])
        _append_limited(vis_state['unk_prob'], unk_np.tolist(), vis_state['max_query_points'])
        _append_limited(vis_state['cls_max'], cls_np.tolist(), vis_state['max_query_points'])
        _append_limited(vis_state['group'], group_np.tolist(), vis_state['max_query_points'])

    if 'proj_obj' in outputs and len(vis_state['proj_obj']) < vis_state['max_feature_points']:
        proj_obj = outputs['proj_obj'].detach().flatten(0, 1).cpu().numpy()
        proj_unk = outputs['proj_unk'].detach().flatten(0, 1).cpu().numpy()
        proj_cls = outputs['proj_cls'].detach().flatten(0, 1).cpu().numpy()
        group_feat = group_np
        remain = vis_state['max_feature_points'] - len(vis_state['proj_obj'])
        if remain < len(proj_obj):
            proj_obj = proj_obj[:remain]
            proj_unk = proj_unk[:remain]
            proj_cls = proj_cls[:remain]
            group_feat = group_feat[:remain]
        vis_state['proj_obj'].extend(list(proj_obj))
        vis_state['proj_unk'].extend(list(proj_unk))
        vis_state['proj_cls'].extend(list(proj_cls))
        vis_state['feat_group'].extend(group_feat.tolist())


def _save_eval_qualitative(vis_state, samples, targets, vis_results, args, out_dir, writer=None, step=0):
    max_samples = int(getattr(args, 'viz_num_samples', 12))
    tb_max = int(getattr(args, 'viz_tb_images', 4))
    unk_label = int(getattr(args, 'num_classes', 81) - 1)

    for i in range(len(targets)):
        if vis_state['num_saved'] >= max_samples:
            break
        img_hw = targets[i]['size'].tolist()
        img_np = _to_numpy_image(samples.tensors[i], img_hw)
        image_id = int(targets[i]['image_id'].item()) if 'image_id' in targets[i] else vis_state['num_saved']

        gt_boxes = _cxcywh_to_abs_xyxy(targets[i]['boxes'], img_hw)
        gt_labels = targets[i]['labels'].detach().cpu().numpy()

        pred = vis_results[i]
        keep = pred['scores'].detach().cpu().numpy() >= float(getattr(args, 'viz_score_thresh', 0.20))
        pred_boxes = pred['boxes'].detach().cpu().numpy()[keep]
        pred_labels = pred['labels'].detach().cpu().numpy()[keep]
        pred_scores = pred['scores'].detach().cpu().numpy()[keep]

        pred_only = _draw_boxes(img_np, pred_boxes=pred_boxes, pred_labels=pred_labels, pred_scores=pred_scores,
                                unk_label=unk_label, title='Predictions')
        gt_only = _draw_boxes(img_np, gt_boxes=gt_boxes, gt_labels=gt_labels,
                              unk_label=unk_label, title='Ground Truth')
        overlay = _draw_boxes(img_np, pred_boxes=pred_boxes, pred_labels=pred_labels, pred_scores=pred_scores,
                              gt_boxes=gt_boxes, gt_labels=gt_labels, unk_label=unk_label, title='Pred + GT')

        pred_path = os.path.join(out_dir, f'{image_id:012d}_pred.png')
        gt_path = os.path.join(out_dir, f'{image_id:012d}_gt.png')
        overlay_path = os.path.join(out_dir, f'{image_id:012d}_overlay.png')
        _save_image(pred_only, pred_path)
        _save_image(gt_only, gt_path)
        _save_image(overlay, overlay_path)
        vis_state['saved_images'].append(overlay_path)

        if writer is not None and vis_state['num_saved'] < tb_max:
            writer.add_image(f'eval_qualitative/{image_id:012d}_overlay', overlay, global_step=step, dataformats='HWC')
            writer.add_image(f'eval_qualitative/{image_id:012d}_pred', pred_only, global_step=step, dataformats='HWC')
            writer.add_image(f'eval_qualitative/{image_id:012d}_gt', gt_only, global_step=step, dataformats='HWC')

        vis_state['num_saved'] += 1


def _finalize_eval_visualizations(vis_state, output_dir, epoch, metrics, writer=None):
    out_dir = os.path.join(output_dir, 'visualizations', f'epoch_{int(epoch):04d}')
    _ensure_dir(out_dir)
    _save_query_stats_csv(vis_state, out_dir)
    _save_feature_npz(vis_state, out_dir)
    _plot_histograms(vis_state, out_dir, writer, epoch)
    _plot_scatter(vis_state, out_dir, writer, epoch)
    _plot_heatmap(vis_state, out_dir, writer, epoch)
    _plot_feature_embeddings(vis_state, out_dir, writer, epoch)
    _plot_metric_bar(metrics, out_dir, writer, epoch)
    _save_contact_sheet(vis_state['saved_images'], os.path.join(out_dir, 'qualitative_contact_sheet.png'))


@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    image_sorted_scores_reduced = {}
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
    logging.info('found a total of %s images', len(image_sorted_scores_reduced.keys()))
    return image_sorted_scores_reduced


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int,
                    max_norm: float = 0, writer=None, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    pseudo_start = int(getattr(args, 'uod_start_epoch', 8))
    neg_start = pseudo_start + int(getattr(args, 'uod_neg_warmup_epochs', 0))

    for step in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = _call_criterion(criterion, outputs, targets, epoch)
        weight_dict = deepcopy(criterion.weight_dict)

        if epoch < nc_epoch:
            for k in list(weight_dict.keys()):
                if 'NC' in k:
                    weight_dict[k] = 0

        if epoch < pseudo_start:
            for k in ('loss_obj_pseudo', 'loss_obj_neg', 'loss_unk_pseudo', 'loss_unk_neg'):
                if k in weight_dict:
                    weight_dict[k] = 0.0
        elif epoch < neg_start:
            for k in ('loss_obj_neg', 'loss_unk_neg'):
                if k in weight_dict:
                    weight_dict[k] = 0.0

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logging.error('Loss is %s, stopping training', loss_value)
            logging.error('Reduced loss dict: %s', loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        if writer is not None:
            global_step = epoch * len(data_loader) + step
            writer.add_scalar('train/total_loss', loss_value, global_step)
            for k, v in loss_dict_reduced_scaled.items():
                writer.add_scalar(f'train_scaled/{k}', _safe_float(v), global_step)
            for k, v in loss_dict_reduced_unscaled.items():
                writer.add_scalar(f'train_unscaled/{k}', _safe_float(v), global_step)

            for stat_key in [
                'stat_num_dummy_pos', 'stat_num_dummy_neg', 'stat_num_valid_unmatched',
                'stat_num_pos_candidates', 'stat_num_batch_selected_pos', 'stat_pos_thresh_mean'
            ]:
                if stat_key in loss_dict_reduced:
                    writer.add_scalar(f'train_stats/{stat_key}', _safe_float(loss_dict_reduced[stat_key]), global_step)

            if 'pred_obj' in outputs:
                writer.add_histogram('train_hist/pred_obj_energy_all', outputs['pred_obj'].detach().float().cpu(), global_step)
                try:
                    outputs_for_match = {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
                    indices_for_hist = criterion.matcher(outputs_for_match, targets)
                    batch_size, num_queries = outputs['pred_obj'].shape[:2]
                    obj_energy = outputs['pred_obj'].detach()
                    matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=obj_energy.device)
                    for b_idx, (src, _) in enumerate(indices_for_hist):
                        if len(src) > 0:
                            matched_mask[b_idx, src] = True
                    matched_energy = obj_energy[matched_mask]
                    unmatched_energy = obj_energy[~matched_mask]
                    if matched_energy.numel() > 0:
                        writer.add_histogram('train_hist/pred_obj_energy_matched', matched_energy.float().cpu(), global_step)
                    if unmatched_energy.numel() > 0:
                        writer.add_histogram('train_hist/pred_obj_energy_unmatched', unmatched_energy.float().cpu(), global_step)
                except Exception:
                    pass

                if 'pred_unk' in outputs:
                    writer.add_histogram('train_hist/pred_unk_logits_all', outputs['pred_unk'].detach().float().cpu(), global_step)
                    writer.add_histogram('train_hist/pred_unk_prob_all', torch.sigmoid(outputs['pred_unk'].detach()).float().cpu(), global_step)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

    metric_logger.synchronize_between_processes()
    logging.info('Averaged stats: %s', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, writer=None, epoch=0):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, 'panoptic_eval'),
        )

    vis_state = _init_eval_viz_state(args) if (getattr(args, 'viz', False) and utils.is_main_process()) else None
    vis_dir = None
    if vis_state is not None:
        vis_dir = os.path.join(output_dir, 'visualizations', f'epoch_{int(epoch):04d}', 'qualitative')
        _ensure_dir(vis_dir)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
        vis_target_sizes = torch.stack([t['size'] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        vis_results = postprocessors['bbox'](outputs, vis_target_sizes) if vis_state is not None else None

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t['size'] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors['panoptic'](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                file_name = f'{image_id:012d}.png'
                res_pano[i]['image_id'] = image_id
                res_pano[i]['file_name'] = file_name
            panoptic_evaluator.update(res_pano)

        if vis_state is not None:
            _collect_eval_stats(vis_state, outputs, targets, criterion, args)
            _save_eval_qualitative(vis_state, samples, targets, vis_results, args, vis_dir, writer=writer, step=epoch)

    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    else:
        res = {}

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics'] = res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res['All']
        stats['PQ_th'] = panoptic_res['Things']
        stats['PQ_st'] = panoptic_res['Stuff']

    if vis_state is not None:
        _finalize_eval_visualizations(vis_state, output_dir, epoch, res, writer=writer)

    return stats, coco_evaluator
