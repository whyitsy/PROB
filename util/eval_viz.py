import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util import box_ops


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'cyan': '#33BBEE',
    'red': '#CC3311',
    'green': '#009988',
    'magenta': '#EE3377',
    'yellow': '#EEDD44',
    'gray': '#6C757D',
    'matched': '#0077BB',
    'unk': '#EE3377',
    'other': '#6C757D',
    'gt_known': '#EEDD44',
    'gt_unk': '#EE7733',
    'candidate': '#33BBEE',
    'box_known': '#009988',
    'box_unk': '#CC3311',
}


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


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _label_color(label, unk_label):
    return _hex_to_rgb(PALETTE['box_unk'] if int(label) == int(unk_label) else PALETTE['box_known'])

def _pred_text(label, score, unk_label):
    if int(label) == int(unk_label):
        return f'U {score:.2f}' if score is not None else 'U'
    return f'K[{int(label)}] {score:.2f}' if score is not None else f'K[{int(label)}]'


def _gt_text(label, unk_label):
    if int(label) == int(unk_label):
        return 'GT-U'
    return f'GT-K[{int(label)}]'


def _draw_legend(draw):
    legend = [
        ('Pred Known', _hex_to_rgb(PALETTE['box_known'])),
        ('Pred Unknown', _hex_to_rgb(PALETTE['box_unk'])),
        ('GT Known', _hex_to_rgb(PALETTE['gt_known'])),
        ('GT Unknown', _hex_to_rgb(PALETTE['gt_unk'])),
    ]
    x0, y0 = 6, 22
    for idx, (txt, color) in enumerate(legend):
        y = y0 + idx * 14
        draw.rectangle([x0, y, x0 + 10, y + 10], outline=color, width=2)
        draw.text((x0 + 16, y - 1), txt, fill=(255, 255, 255))


def _case_stem(image_id, epoch, num_pred, num_gt, num_unk_pred, num_unk_gt, num_u2k, num_k2u):
    return (
        f'{int(image_id):012d}'
        f'__ep{int(epoch):04d}'
        f'__pred{int(num_pred)}'
        f'__gt{int(num_gt)}'
        f'__unkpred{int(num_unk_pred)}'
        f'__unkgt{int(num_unk_gt)}'
        f'__u2k{int(num_u2k)}'
        f'__k2u{int(num_k2u)}'
    )



def _draw_boxes(image_np, pred_boxes=None, pred_labels=None, pred_scores=None,
                gt_boxes=None, gt_labels=None, unk_label=80, title=None,
                summary_text=None, show_legend=False):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = int(gt_labels[i]) if gt_labels is not None else -1
            color = _hex_to_rgb(PALETTE['gt_unk'] if label == int(unk_label) else PALETTE['gt_known'])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), _gt_text(label, unk_label), fill=color)
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = int(pred_labels[i]) if pred_labels is not None else -1
            score = float(pred_scores[i]) if pred_scores is not None else None
            color = _label_color(label, unk_label)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = _pred_text(label, score, unk_label)
            draw.text((x1 + 2, y1 + 2), text, fill=color)
    if title:
        draw.text((5, 5), title, fill=(255, 255, 255))
    if summary_text:
        draw.text((5, 175), summary_text, fill=(255, 255, 255))
    if show_legend:
        _draw_legend(draw)
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


def _save_figure(fig, out_path, writer=None, tb_tag=None, step=0, top=0.94, right=0.96):
    fig.subplots_adjust(top=top, right=right)
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
    axes[0].hist(vis_state['obj_prob'], bins=40, color=PALETTE['blue'] if 'blue' in PALETTE else '#0077BB')
    axes[0].set_title('Objectness probability')
    axes[1].hist(vis_state['unk_prob'], bins=40, color=PALETTE['unk'])
    axes[1].set_title('Unknownness probability')
    axes[2].hist(vis_state['cls_max'], bins=40, color=PALETTE['matched'])
    axes[2].set_title('Max known-class probability')
    for ax in axes:
        ax.grid(alpha=0.2)
    _save_figure(fig, os.path.join(out_dir, 'hist_probabilities.png'), writer, 'eval_viz/hist_probabilities', step)


def _plot_scatter(vis_state, out_dir, writer=None, step=0):
    if len(vis_state['obj_prob']) == 0:
        return
    groups = np.array(vis_state['group'])
    colors = np.array([PALETTE['matched'], PALETTE['unk'], PALETTE['other']])
    group_names = ['matched-known', 'unmatched-highunk', 'unmatched-other']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for gid in range(3):
        mask = groups == gid
        if np.any(mask):
            axes[0].scatter(np.array(vis_state['obj_prob'])[mask], np.array(vis_state['unk_prob'])[mask],
                            s=10, alpha=0.55, c=colors[gid], label=group_names[gid])
            axes[1].scatter(np.array(vis_state['obj_prob'])[mask], np.array(vis_state['cls_max'])[mask],
                            s=10, alpha=0.55, c=colors[gid], label=group_names[gid])
    axes[0].set_xlabel('obj prob')
    axes[0].set_ylabel('unk prob')
    axes[0].set_title('Objectness vs Unknownness')
    axes[1].set_xlabel('obj prob')
    axes[1].set_ylabel('max known prob')
    axes[1].set_title('Objectness vs Max-known score')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, frameon=False)
    _save_figure(fig, os.path.join(out_dir, 'scatter_relationships.png'), writer, 'eval_viz/scatter_relationships', step)


def _plot_heatmap(vis_state, out_dir, writer=None, step=0):
    if len(vis_state['obj_prob']) < 4:
        return
    obj_arr = np.array(vis_state['obj_prob'])
    unk_arr = np.array(vis_state['unk_prob'])
    cls_arr = np.array(vis_state['cls_max'])
    arr_global = np.stack([obj_arr, unk_arr, cls_arr], axis=0)
    corr_global = np.corrcoef(arr_global)
    fg_mask = obj_arr > 0.05
    if fg_mask.sum() > 4:
        arr_fg = np.stack([obj_arr[fg_mask], unk_arr[fg_mask], cls_arr[fg_mask]], axis=0)
        corr_fg = np.corrcoef(arr_fg)
    else:
        corr_fg = np.zeros((3, 3))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    fig.subplots_adjust(right=0.86, wspace=0.35)
    for ax, corr, title in zip(axes, [corr_global, corr_fg], ['Global (with Background)', 'Foreground Only (obj > 0.05)']):
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['obj', 'unk', 'cls_max'])
        ax.set_yticklabels(['obj', 'unk', 'cls_max'])
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='black' if abs(corr[i, j]) > 0.45 else 'white')
    cax = fig.add_axes([0.88, 0.17, 0.02, 0.68])
    fig.colorbar(im, cax=cax)
    _save_figure(fig, os.path.join(out_dir, 'correlation_heatmap.png'), writer, 'eval_viz/correlation_heatmap', step, top=0.92, right=0.86)


def compute_decoupling_corr_metrics(vis_state):
    if len(vis_state['obj_prob']) < 4:
        return {}
    obj_arr = np.asarray(vis_state['obj_prob'], dtype=np.float64)
    unk_arr = np.asarray(vis_state['unk_prob'], dtype=np.float64)
    cls_arr = np.asarray(vis_state['cls_max'], dtype=np.float64)
    out = {}
    arr_global = np.stack([obj_arr, unk_arr, cls_arr], axis=0)
    corr_global = np.corrcoef(arr_global)
    out['corr_global_obj_unk'] = float(corr_global[0, 1])
    out['corr_global_obj_cls'] = float(corr_global[0, 2])
    out['corr_global_unk_cls'] = float(corr_global[1, 2])
    fg_mask = obj_arr > 0.05
    if fg_mask.sum() > 4:
        arr_fg = np.stack([obj_arr[fg_mask], unk_arr[fg_mask], cls_arr[fg_mask]], axis=0)
        corr_fg = np.corrcoef(arr_fg)
        out['corr_fg_obj_unk'] = float(corr_fg[0, 1])
        out['corr_fg_obj_cls'] = float(corr_fg[0, 2])
        out['corr_fg_unk_cls'] = float(corr_fg[1, 2])
    else:
        out['corr_fg_obj_unk'] = None
        out['corr_fg_obj_cls'] = None
        out['corr_fg_unk_cls'] = None
    return out


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
    colors = np.array([PALETTE['matched'], PALETTE['unk'], PALETTE['other']])
    feat_specs = [('proj_obj', vis_state['proj_obj']), ('proj_unk', vis_state['proj_unk']), ('proj_cls', vis_state['proj_cls'])]
    for method in ['pca', 'tsne']:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
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
                    ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.6, c=colors[gid], label=group_names[gid])
            ax.set_title(f'{name} {method.upper()}')
            ax.grid(alpha=0.2)
        if plotted_any:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
            _save_figure(fig, os.path.join(out_dir, f'feature_{method}.png'), writer, f'eval_viz/feature_{method}', step, top=0.82)
        else:
            plt.close(fig)


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


def _save_error_case_csv(vis_state, out_dir):
    rows = vis_state.get('error_rows', [])
    if not rows:
        return
    with open(os.path.join(out_dir, 'error_case_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'num_pred', 'num_gt', 'num_unknown_to_known', 'num_known_to_unknown', 'num_candidates', 'num_final_unknown_candidates'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def init_eval_viz_state(args):
    return {
        'saved_images': [],
        'saved_final_images': [],
        'saved_debug_images': [],
        'num_saved': 0,
        'obj_prob': [],
        'unk_prob': [],
        'cls_max': [],
        'group': [],
        'proj_obj': [],
        'proj_unk': [],
        'proj_cls': [],
        'feat_group': [],
        'max_query_points': int(getattr(args, 'viz_max_query_points', 2500)),
        'max_feature_points': int(getattr(args, 'viz_max_feature_points', 2500)),
        'saved_panels': [],
        'error_rows': [],
    }


def _append_limited(dst, values, max_len):
    remain = max_len - len(dst)
    if remain <= 0:
        return
    if len(values) > remain:
        values = values[:remain]
    dst.extend(values)

def _select_mining_aligned_candidates_from_criterion(outputs, targets, criterion, epoch, image_idx, img_hw, args):
    outputs_for_match = {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
    indices = criterion.matcher(outputs_for_match, targets)
    mine_out = criterion._mine_uod_pseudo(outputs, targets, indices, epoch)
    dummy_pos_indices, _, dummy_pos_weights, dummy_pos_boxes, _, _ = mine_out

    if image_idx >= len(dummy_pos_indices) or len(dummy_pos_indices[image_idx]) == 0:
        return np.zeros((0, 4), dtype=np.float32), [], [], [], []

    q_idx = torch.as_tensor(dummy_pos_indices[image_idx], dtype=torch.long, device=outputs['pred_boxes'].device)
    boxes_cxcywh = dummy_pos_boxes[image_idx].detach().cpu()
    boxes_xyxy = _cxcywh_to_abs_xyxy(boxes_cxcywh, img_hw)

    fused = _compute_vis_fused(outputs, image_idx, args, invalid_cls_logits=getattr(criterion, 'invalid_cls_logits', []))
    if fused is None:
        return boxes_xyxy, [0.0] * len(boxes_xyxy), [0.0] * len(boxes_xyxy), [0.0] * len(boxes_xyxy), dummy_pos_weights[image_idx]

    obj_scores = fused['obj_prob'][q_idx.cpu()].tolist()
    cls_scores = fused['cls_max'][q_idx.cpu()].tolist()
    unk_scores = fused['unk_prob'][q_idx.cpu()].tolist()
    sel_scores = list(dummy_pos_weights[image_idx])
    return boxes_xyxy, obj_scores, cls_scores, unk_scores, sel_scores

def collect_eval_stats(vis_state, outputs, targets, criterion, args):
    if len(vis_state['obj_prob']) >= vis_state['max_query_points'] and len(vis_state['proj_obj']) >= vis_state['max_feature_points']:
        return

    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temp = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim

    obj_prob = _energy_to_prob(outputs['pred_obj'].detach(), obj_temp)
    if 'pred_known' in outputs:
        knownness_prob = _energy_to_prob(outputs['pred_known'].detach(), known_temp)
        unk_prob = (1.0 - knownness_prob).clamp(min=0.0, max=1.0)
    else:
        pred_unk = outputs.get('pred_unk', None)
        unk_prob = torch.sigmoid(pred_unk.detach()) if pred_unk is not None else torch.zeros_like(obj_prob)

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

    _append_limited(vis_state['obj_prob'], obj_np.tolist(), vis_state['max_query_points'])
    _append_limited(vis_state['unk_prob'], unk_np.tolist(), vis_state['max_query_points'])
    _append_limited(vis_state['cls_max'], cls_np.tolist(), vis_state['max_query_points'])
    _append_limited(vis_state['group'], group_np.tolist(), vis_state['max_query_points'])

    if 'proj_obj' in outputs and len(vis_state['proj_obj']) < vis_state['max_feature_points']:
        proj_obj = outputs['proj_obj'].detach().flatten(0, 1).cpu().numpy()
        proj_known = outputs.get('proj_known', outputs.get('proj_unk')).detach().flatten(0, 1).cpu().numpy()
        proj_cls = outputs['proj_cls'].detach().flatten(0, 1).cpu().numpy()
        group_feat = group_np
        remain = vis_state['max_feature_points'] - len(vis_state['proj_obj'])
        if remain < len(proj_obj):
            proj_obj = proj_obj[:remain]
            proj_known = proj_known[:remain]
            proj_cls = proj_cls[:remain]
            group_feat = group_feat[:remain]
        vis_state['proj_obj'].extend(list(proj_obj))
        vis_state['proj_unk'].extend(list(proj_known))
        vis_state['proj_cls'].extend(list(proj_cls))
        vis_state['feat_group'].extend(group_feat.tolist())

def _box_iou_np(boxes1, boxes2):
    if boxes1 is None or boxes2 is None or len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    b1 = boxes1.astype(np.float32)
    b2 = boxes2.astype(np.float32)
    area1 = np.clip(b1[:, 2] - b1[:, 0], 0, None) * np.clip(b1[:, 3] - b1[:, 1], 0, None)
    area2 = np.clip(b2[:, 2] - b2[:, 0], 0, None) * np.clip(b2[:, 3] - b2[:, 1], 0, None)
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)



def _box_iou_torch(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    return box_ops.box_iou(boxes1, boxes2)[0]


def _candidate_nms_np(boxes, scores, iou_thr=0.6, topk=None):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if topk is not None and len(keep) >= int(topk):
            break
        if order.size == 1:
            break
        rest = order[1:]
        ious = _box_iou_np(boxes[i:i+1], boxes[rest]).reshape(-1)
        order = rest[ious < float(iou_thr)]
    return np.asarray(keep, dtype=np.int64)


def _compute_vis_fused(outputs, image_idx, args, invalid_cls_logits=None):
    pred_logits = outputs.get('pred_logits', None)
    pred_obj = outputs.get('pred_obj', None)
    pred_boxes = outputs.get('pred_boxes', None)
    if pred_logits is None or pred_obj is None or pred_boxes is None:
        return None
    invalid_cls_logits = list(invalid_cls_logits or [])
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temp = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim
    unknown_scale = float(getattr(args, 'uod_postprocess_unknown_scale', 15.0))

    logits = pred_logits[image_idx].detach().clone().cpu()
    if len(invalid_cls_logits) > 0:
        logits[:, invalid_cls_logits] = -10e10

    obj_prob = _energy_to_prob(pred_obj[image_idx].detach().cpu(), obj_temp)
    class_prob = logits.sigmoid()
    if len(invalid_cls_logits) > 0:
        class_prob[:, invalid_cls_logits] = 0.0
    if class_prob.shape[-1] > 0:
        class_prob[:, -1] = 0.0

    pred_known = outputs.get('pred_known', None)
    if pred_known is not None:
        knownness_prob = _energy_to_prob(pred_known[image_idx].detach().cpu(), known_temp)
    else:
        pred_unk = outputs.get('pred_unk', None)
        unk_prob_compat = torch.sigmoid(pred_unk[image_idx].detach().cpu()) if pred_unk is not None else torch.zeros_like(obj_prob)
        knownness_prob = (1.0 - unk_prob_compat).clamp(min=1e-6, max=1.0)

    unk_prob = (1.0 - knownness_prob).clamp(min=0.0, max=1.0)
    known_scores = obj_prob.unsqueeze(-1) * knownness_prob.unsqueeze(-1) * class_prob
    cls_max = class_prob.max(dim=-1).values if class_prob.shape[-1] > 0 else torch.zeros_like(obj_prob)
    unknown_score = obj_prob * unk_prob * unknown_scale
    return {
        'obj_prob': obj_prob,
        'known_prob': class_prob,
        'knownness_prob': knownness_prob,
        'cls_max': cls_max,
        'unk_prob': unk_prob,
        'unknown_score': unknown_score,
        'pred_boxes': pred_boxes[image_idx].detach().cpu(),
        'known_scores': known_scores,
    }


def _is_valid_geometry_np(box_cxcywh, args):
    w = float(box_cxcywh[2])
    h = float(box_cxcywh[3])
    area = w * h
    side = min(w, h)
    ar = max(w / max(h, 1e-6), h / max(w, 1e-6))
    return area >= float(getattr(args, 'uod_min_area', 0.002)) and side >= float(getattr(args, 'uod_min_side', 0.05)) and ar <= float(getattr(args, 'uod_max_aspect_ratio', 4.0))


def _select_mining_aligned_candidates(outputs, targets, match_src_idx, image_idx, img_hw, args, invalid_cls_logits=None):
    fused = _compute_vis_fused(outputs, image_idx, args, invalid_cls_logits=invalid_cls_logits)
    if fused is None:
        return np.zeros((0, 4), dtype=np.float32), [], [], [], []
    energy = outputs['pred_obj'][image_idx].detach().cpu() / float(getattr(args, 'hidden_dim', 256))
    pred_boxes = outputs['pred_boxes'][image_idx].detach().cpu()
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
    matched = set(torch.as_tensor(match_src_idx, dtype=torch.long).tolist())
    num_queries = int(pred_boxes.shape[0])
    unmatched = [q for q in range(num_queries) if q not in matched]
    if len(match_src_idx) > 0:
        matched_scores = energy[torch.as_tensor(match_src_idx, dtype=torch.long)]
        mu_obj = matched_scores.mean().item()
        std_obj = matched_scores.std().item() if len(match_src_idx) > 1 else 0.0
        pos_thresh = max(mu_obj + 3.0 * std_obj, float(getattr(args, 'uod_min_pos_thresh', 0.08)))
    else:
        pos_thresh = float(getattr(args, 'uod_min_pos_thresh', 0.08))
    gt_boxes = targets[image_idx]['boxes'].detach().cpu()
    gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes) if gt_boxes.numel() > 0 else gt_boxes.new_zeros((0, 4))
    iou_map = {q: 0.0 for q in unmatched}
    valid = unmatched
    if gt_xyxy.numel() > 0 and len(unmatched) > 0:
        cand_boxes = pred_xyxy[torch.as_tensor(unmatched, dtype=torch.long)]
        ious = _box_iou_torch(cand_boxes, gt_xyxy)
        lt = torch.max(cand_boxes[:, None, :2], gt_xyxy[:, :2])
        rb = torch.min(cand_boxes[:, None, 2:], gt_xyxy[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = ((cand_boxes[:, 2] - cand_boxes[:, 0]).clamp(min=0) * (cand_boxes[:, 3] - cand_boxes[:, 1]).clamp(min=0)).clamp(min=1e-6)
        iofs = inter / area1[:, None]
        max_iou = ious.max(dim=1).values
        max_iof = iofs.max(dim=1).values
        valid = []
        for j, q in enumerate(unmatched):
            iou_map[q] = float(max_iou[j].item())
            if float(max_iou[j].item()) < float(getattr(args, 'uod_max_iou', 0.2)) and float(max_iof[j].item()) < float(getattr(args, 'uod_max_iof', 0.4)):
                valid.append(q)
    valid = [q for q in valid if _is_valid_geometry_np(pred_boxes[q].tolist(), args)]
    known_max = fused['known_prob'].max(dim=-1).values
    candidates = []
    for q in valid:
        e = float(energy[q].item())
        k = float(known_max[q].item())
        reject = float(getattr(args, 'uod_known_reject_thresh', 0.15))
        if e < pos_thresh and k < reject:
            energy_rel = max(0.0, min(1.0, (pos_thresh - e) / max(pos_thresh, 1e-6)))
            known_rel = max(0.0, min(1.0, (reject - k) / max(reject, 1e-6)))
            max_iou = float(getattr(args, 'uod_max_iou', 0.2))
            iou_rel = 1.0 - max(0.0, min(1.0, iou_map[q] / max(max_iou, 1e-6)))
            conf = (energy_rel * known_rel * iou_rel) ** (1.0 / 3.0)
            candidates.append((q, conf, e, k, float(fused['unk_prob'][q].item())))
    candidates = sorted(candidates, key=lambda x: (-x[1], x[2], x[3]))
    if len(candidates) == 0:
        return np.zeros((0, 4), dtype=np.float32), [], [], [], []
    cand_idx = [q for q, _, _, _, _ in candidates]
    cand_scores = [conf for _, conf, _, _, _ in candidates]
    cand_boxes = _cxcywh_to_abs_xyxy(pred_boxes[torch.as_tensor(cand_idx, dtype=torch.long)], img_hw)
    keep = _candidate_nms_np(cand_boxes, cand_scores, iou_thr=float(getattr(args, 'viz_candidate_nms_iou', 0.6)), topk=int(getattr(args, 'viz_candidate_topk', 10)))
    cand_boxes = cand_boxes[keep]
    kept = [candidates[int(k)] for k in keep.tolist()]
    return cand_boxes, [1.0 - c[2] for c in kept], [c[3] for c in kept], [c[4] for c in kept], [c[1] for c in kept]


def _select_final_unknown_candidates(pred, unk_label, topk=10, nms_iou=0.6):
    if pred is None or len(pred.get('boxes', [])) == 0:
        return np.zeros((0, 4), dtype=np.float32), [], []
    boxes = pred['boxes'].detach().cpu().numpy()
    labels = pred['labels'].detach().cpu().numpy()
    scores = pred['scores'].detach().cpu().numpy()
    mask = labels == int(unk_label)
    if mask.sum() == 0:
        return np.zeros((0, 4), dtype=np.float32), [], []
    boxes = boxes[mask]
    scores = scores[mask]
    keep = _candidate_nms_np(boxes, scores, iou_thr=nms_iou, topk=topk)
    return boxes[keep], scores[keep].tolist(), scores[keep].tolist()


def _draw_candidate_boxes(image_np, boxes, obj_scores, cls_scores, unk_scores, title='Mining-aligned pseudo candidates'):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    color = _hex_to_rgb(PALETTE['candidate'])
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        txt = f'obj:{obj_scores[i]:.2f} cls:{cls_scores[i]:.2f}'
        if unk_scores is not None:
            txt += f' unk:{unk_scores[i]:.2f}'
        draw.text((x1 + 2, y1 + 2), txt, fill=color)
    draw.text((5, 5), title, fill=(255, 255, 255))
    return np.array(img)


def _make_panel(images_with_titles, out_path, tile_hw=(480, 320), cols=2):
    imgs = []
    for arr, title in images_with_titles:
        img = Image.fromarray(arr).convert('RGB')
        canvas = Image.new('RGB', img.size, (20, 20, 20))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((6, 6), title, fill=(255, 255, 255))
        canvas = canvas.resize(tile_hw)
        imgs.append(canvas)
    rows = int(math.ceil(len(imgs) / cols))
    sheet = Image.new('RGB', (cols * tile_hw[0], rows * tile_hw[1]), (15, 15, 15))
    for idx, img in enumerate(imgs):
        x = (idx % cols) * tile_hw[0]
        y = (idx // cols) * tile_hw[1]
        sheet.paste(img, (x, y))
    sheet.save(out_path)


def _extract_error_cases(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, unk_label, iou_thr=0.5):
    errors = {'unknown_to_known_pred_idx': [], 'unknown_to_known_gt_idx': [], 'known_to_unknown_pred_idx': [], 'known_to_unknown_gt_idx': []}
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return errors
    iou = _box_iou_np(pred_boxes, gt_boxes)
    if iou.size == 0:
        return errors
    for g in range(len(gt_boxes)):
        p = int(np.argmax(iou[:, g]))
        if iou[p, g] < iou_thr:
            continue
        gt_is_unk = int(gt_labels[g]) == int(unk_label)
        pred_is_unk = int(pred_labels[p]) == int(unk_label)
        if gt_is_unk and not pred_is_unk:
            errors['unknown_to_known_pred_idx'].append(p)
            errors['unknown_to_known_gt_idx'].append(g)
        if (not gt_is_unk) and pred_is_unk:
            errors['known_to_unknown_pred_idx'].append(p)
            errors['known_to_unknown_gt_idx'].append(g)
    return errors


def _select_high_obj_low_known_candidates(outputs, image_idx, img_hw, args):
    obj_thr = float(getattr(args, 'viz_candidate_obj_thresh', 0.50))
    cls_thr = float(getattr(args, 'viz_candidate_clsmax_thresh', 0.20))
    topk = int(getattr(args, 'viz_candidate_topk', 10))
    pred_obj = outputs.get('pred_obj', None)
    pred_logits = outputs.get('pred_logits', None)
    pred_boxes = outputs.get('pred_boxes', None)
    if pred_obj is None or pred_logits is None or pred_boxes is None:
        return np.zeros((0, 4), dtype=np.float32), [], [], []
    obj_temp = float(getattr(args, 'obj_temp', 1.0))
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_prob = torch.exp(-(obj_temp / hidden_dim) * pred_obj[image_idx].detach()).cpu()
    cls_prob = pred_logits[image_idx].detach().sigmoid().cpu()
    if cls_prob.shape[-1] > 0:
        cls_prob[:, -1] = 0.0
    cls_max = cls_prob.max(-1).values
    pred_unk = outputs.get('pred_unk', None)
    unk_prob = torch.sigmoid(pred_unk[image_idx].detach()).cpu() if pred_unk is not None else torch.zeros_like(obj_prob)
    mask = (obj_prob >= obj_thr) & (cls_max <= cls_thr)
    if mask.sum().item() == 0:
        return np.zeros((0, 4), dtype=np.float32), [], [], []
    score = obj_prob * (1.0 - cls_max) * torch.clamp(unk_prob + 0.2, min=0.2)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    idx = idx[torch.argsort(score[idx], descending=True)][:topk]
    boxes = _cxcywh_to_abs_xyxy(pred_boxes[image_idx][idx], img_hw)
    return boxes, obj_prob[idx].tolist(), cls_max[idx].tolist(), unk_prob[idx].tolist()

def _energy_to_prob(energy, temperature):
    return torch.exp(-temperature * energy).clamp(min=1e-6, max=1.0)

def save_eval_qualitative(vis_state, samples, targets, vis_results, outputs, criterion, args, out_dir, writer=None, step=0, epoch=0):
    max_samples = int(getattr(args, 'viz_num_samples', 12))
    tb_max = int(getattr(args, 'viz_tb_images', 4))
    unk_label = int(getattr(args, 'num_classes', 81) - 1)
    final_dir = os.path.join(out_dir, 'final')
    debug_dir = os.path.join(out_dir, 'debug')
    _ensure_dir(final_dir)
    _ensure_dir(debug_dir)
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
        known_mask = pred_labels != unk_label if len(pred_labels) > 0 else np.array([], dtype=bool)
        unknown_mask = pred_labels == unk_label if len(pred_labels) > 0 else np.array([], dtype=bool)
        num_unk_gt = int((gt_labels == unk_label).sum()) if len(gt_labels) > 0 else 0
        num_unk_pred = int(unknown_mask.sum()) if len(pred_labels) > 0 else 0

        summary_text = (
            f'ID={image_id} | ep={int(epoch):04d} | pred={len(pred_boxes)} gt={len(gt_boxes)} | '
            f'unk_pred={num_unk_pred} unk_gt={num_unk_gt}'
        )
        pred_all = _draw_boxes(img_np, pred_boxes=pred_boxes, pred_labels=pred_labels, pred_scores=pred_scores,
                            unk_label=unk_label, title='All Predictions', summary_text=summary_text, show_legend=True)
        pred_known = _draw_boxes(img_np, pred_boxes=pred_boxes[known_mask], pred_labels=pred_labels[known_mask], pred_scores=pred_scores[known_mask],
                                unk_label=unk_label, title='Known Predictions', summary_text=summary_text, show_legend=True)
        pred_unknown = _draw_boxes(img_np, pred_boxes=pred_boxes[unknown_mask], pred_labels=pred_labels[unknown_mask], pred_scores=pred_scores[unknown_mask],
                                unk_label=unk_label, title='Unknown Predictions', summary_text=summary_text, show_legend=True)
        gt_only = _draw_boxes(img_np, gt_boxes=gt_boxes, gt_labels=gt_labels, unk_label=unk_label, title='Ground Truth',
                            summary_text=summary_text, show_legend=True)
        overlay = _draw_boxes(img_np, pred_boxes=pred_boxes, pred_labels=pred_labels, pred_scores=pred_scores,
                            gt_boxes=gt_boxes, gt_labels=gt_labels, unk_label=unk_label, title='Pred + GT',
                            summary_text=summary_text, show_legend=True)

        errors = _extract_error_cases(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, unk_label, iou_thr=float(getattr(args, 'viz_error_iou_thresh', 0.5)))
        u2k_pred_idx = np.array(sorted(set(errors['unknown_to_known_pred_idx'])), dtype=np.int64)
        u2k_gt_idx = np.array(sorted(set(errors['unknown_to_known_gt_idx'])), dtype=np.int64)
        k2u_pred_idx = np.array(sorted(set(errors['known_to_unknown_pred_idx'])), dtype=np.int64)
        k2u_gt_idx = np.array(sorted(set(errors['known_to_unknown_gt_idx'])), dtype=np.int64)

        summary_with_errors = summary_text + f' | U2K={len(u2k_gt_idx)} K2U={len(k2u_gt_idx)}'
        err_unknown_to_known = _draw_boxes(img_np, pred_boxes=pred_boxes[u2k_pred_idx] if len(u2k_pred_idx) > 0 else None, pred_labels=pred_labels[u2k_pred_idx] if len(u2k_pred_idx) > 0 else None, pred_scores=pred_scores[u2k_pred_idx] if len(u2k_pred_idx) > 0 else None, gt_boxes=gt_boxes[u2k_gt_idx] if len(u2k_gt_idx) > 0 else None, gt_labels=gt_labels[u2k_gt_idx] if len(u2k_gt_idx) > 0 else None, unk_label=unk_label, title='Error: Unknown -> Known', summary_text=summary_with_errors, show_legend=True)
        err_known_to_unknown = _draw_boxes(img_np, pred_boxes=pred_boxes[k2u_pred_idx] if len(k2u_pred_idx) > 0 else None, pred_labels=pred_labels[k2u_pred_idx] if len(k2u_pred_idx) > 0 else None, pred_scores=pred_scores[k2u_pred_idx] if len(k2u_pred_idx) > 0 else None, gt_boxes=gt_boxes[k2u_gt_idx] if len(k2u_gt_idx) > 0 else None, gt_labels=gt_labels[k2u_gt_idx] if len(k2u_gt_idx) > 0 else None, unk_label=unk_label, title='Error: Known -> Unknown', summary_text=summary_with_errors, show_legend=True)
        
        cand_boxes, cand_obj, cand_cls, cand_unk, cand_sel = _select_mining_aligned_candidates_from_criterion(outputs, targets, criterion, epoch, i, img_hw, args)
        cand_img = _draw_candidate_boxes(img_np, cand_boxes, cand_obj, cand_cls, cand_unk, title='Mining-aligned pseudo candidates')

        final_unk_boxes, final_unk_scores, _ = _select_final_unknown_candidates(
            pred, unk_label, topk=int(getattr(args, 'viz_candidate_topk', 10)),
            nms_iou=float(getattr(args, 'viz_candidate_nms_iou', 0.6))
        )
        final_unk_img = _draw_boxes(
            img_np,
            pred_boxes=final_unk_boxes,
            pred_labels=np.full((len(final_unk_boxes),), unk_label, dtype=np.int64) if len(final_unk_boxes) > 0 else None,
            pred_scores=np.asarray(final_unk_scores, dtype=np.float32) if len(final_unk_scores) > 0 else None,
            unk_label=unk_label,
            title='Final Unknown Predictions (Top-K/NMS)',
            summary_text=summary_with_errors,
            show_legend=True,
        )
        
        stem_name = _case_stem(
            image_id, epoch, len(pred_boxes), len(gt_boxes), num_unk_pred, num_unk_gt,
            len(u2k_gt_idx), len(k2u_gt_idx)
        )
        stem_final = os.path.join(final_dir, stem_name)
        stem_debug = os.path.join(debug_dir, stem_name)
        _save_image(pred_all, stem_final + '__pred_all.png')
        _save_image(pred_known, stem_final + '__pred_known.png')
        _save_image(pred_unknown, stem_final + '__pred_unknown.png')
        _save_image(gt_only, stem_final + '__gt.png')
        _save_image(overlay, stem_final + '__overlay.png')
        _save_image(err_unknown_to_known, stem_final + '__error_unknown_to_known.png')
        _save_image(err_known_to_unknown, stem_final + '__error_known_to_unknown.png')
        _save_image(cand_img, stem_debug + '__candidate_mining_aligned.png')
        _save_image(final_unk_img, stem_debug + '__final_unknown_candidates.png')

        panel_path = stem_final + '__panel.png'
        _make_panel([
            (gt_only, 'Ground Truth'),
            (pred_all, 'All Predictions'),
            (pred_known, 'Known Predictions'),
            (pred_unknown, 'Unknown Predictions')
        ], panel_path, tile_hw=(420, 280), cols=2)

        error_panel_path = stem_final + '__panel_errors.png'
        _make_panel([
            (overlay, 'Pred + GT'),
            (err_unknown_to_known, 'Error: Unknown -> Known'),
            (err_known_to_unknown, 'Error: Known -> Unknown'),
            (final_unk_img, 'Final Unknown Predictions')
        ], error_panel_path, tile_hw=(420, 280), cols=2)

        vis_state['saved_images'].append(error_panel_path)
        vis_state['saved_final_images'].append(panel_path)
        vis_state['saved_debug_images'].append(stem_debug + '__candidate_mining_aligned.png')
        vis_state['saved_panels'].append(panel_path)
        vis_state['error_rows'].append({
            'image_id': image_id,
            'num_pred': int(len(pred_boxes)),
            'num_gt': int(len(gt_boxes)),
            'num_unknown_to_known': int(len(u2k_gt_idx)),
            'num_known_to_unknown': int(len(k2u_gt_idx)),
            'num_candidates': int(len(cand_boxes)),
            'num_final_unknown_candidates': int(len(final_unk_boxes)),
        })
        if writer is not None and vis_state['num_saved'] < tb_max:
            writer.add_image(f'eval_qualitative/{image_id:012d}_panel', np.array(Image.open(panel_path)), global_step=step, dataformats='HWC')
            writer.add_image(f'eval_qualitative/{image_id:012d}_panel_errors', np.array(Image.open(error_panel_path)), global_step=step, dataformats='HWC')
        vis_state['num_saved'] += 1
        

def finalize_eval_visualizations(vis_state, output_dir, epoch, writer=None):
    out_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{int(epoch):04d}')
    _ensure_dir(out_dir)
    stats_dir = os.path.join(out_dir, 'stats')
    final_dir = os.path.join(out_dir, 'final')
    debug_dir = os.path.join(out_dir, 'debug')
    _ensure_dir(stats_dir)
    _ensure_dir(final_dir)
    _ensure_dir(debug_dir)
    _save_query_stats_csv(vis_state, stats_dir)
    _save_feature_npz(vis_state, stats_dir)
    _save_error_case_csv(vis_state, stats_dir)
    _plot_histograms(vis_state, stats_dir, writer, epoch)
    _plot_scatter(vis_state, stats_dir, writer, epoch)
    _plot_heatmap(vis_state, stats_dir, writer, epoch)
    _plot_feature_embeddings(vis_state, stats_dir, writer, epoch)
    _save_contact_sheet(vis_state.get('saved_final_images', vis_state.get('saved_panels', [])), os.path.join(final_dir, 'qualitative_panels_contact_sheet.png'))
    _save_contact_sheet(vis_state.get('saved_images', []), os.path.join(final_dir, 'qualitative_error_panels_contact_sheet.png'))
    _save_contact_sheet(vis_state.get('saved_debug_images', []), os.path.join(debug_dir, 'debug_contact_sheet.png'))
