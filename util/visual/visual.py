import torch
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
绿色：GT
蓝色：matched query
橙色：dummy_pos
红色：dummy_neg
"""


def _denorm_image(img_tensor):
    """
    img_tensor: [3, H, W], normalized image tensor
    returns numpy image in [0,1], shape [H, W, 3]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).detach().cpu().numpy()


def _get_valid_image_size(samples, targets, b):
    """
    优先使用 target['size'] 作为当前图像的真实可视尺寸。
    若没有，再退回 samples.mask 或 tensor shape。
    返回: (img_h, img_w)
    """
    if targets is not None and 'size' in targets[b]:
        size = targets[b]['size']
        if isinstance(size, torch.Tensor):
            img_h = int(size[0].item())
            img_w = int(size[1].item())
            return img_h, img_w

    if hasattr(samples, 'mask') and samples.mask is not None:
        # mask: [B, H, W], False 表示有效区域，True 表示 padding
        valid_mask = ~samples.mask[b]
        rows = torch.where(valid_mask.any(dim=1))[0]
        cols = torch.where(valid_mask.any(dim=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            img_h = int(rows[-1].item()) + 1
            img_w = int(cols[-1].item()) + 1
            return img_h, img_w

    _, H, W = samples.tensors[b].shape
    return int(H), int(W)


def _clamp_boxes_xyxy(boxes_xyxy, img_w, img_h):
    """
    boxes_xyxy: Tensor [N, 4] in absolute coords
    clamp to image region
    """
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy
    boxes = boxes_xyxy.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, img_w - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, img_h - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, img_w - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, img_h - 1)
    return boxes


def _norm_cxcywh_to_abs_xyxy(boxes_cxcywh, img_w, img_h):
    """
    boxes_cxcywh: normalized [N,4] in [0,1]
    return absolute xyxy boxes
    """
    if boxes_cxcywh.numel() == 0:
        return torch.zeros((0, 4), device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    boxes_xyxy = boxes_xyxy * scale
    boxes_xyxy = _clamp_boxes_xyxy(boxes_xyxy, img_w, img_h)
    return boxes_xyxy


@torch.no_grad()
def _build_debug_groups(outputs, targets, criterion, epoch):
    """
    根据当前最终层输出，重建 matched / dummy_pos / dummy_neg，专用于可视化。
    与训练逻辑保持一致：
    - matched quantile 定义 pos_thresh
    - known_max < unk_cls_reject_thresh
    - max_pos_per_img / max_neg_per_img
    返回:
        indices, dummy_pos_indices, dummy_neg_indices, debug_meta
    debug_meta[b]:
        {
            'pos_thresh': float,
            'neg_thresh': float,
            'query_info': {
                q: {'energy':..., 'known_max':..., 'max_iou':...}
            }
        }
    """
    outputs_wo_aux = {
        k: v for k, v in outputs.items()
        if k not in ['aux_outputs', 'enc_outputs', 'pred_obj', 'samples', 'pred_proj']
    }
    indices = criterion.matcher(outputs_wo_aux, targets)

    batch_size = len(targets)
    dummy_pos_indices = [[] for _ in range(batch_size)]
    dummy_neg_indices = [[] for _ in range(batch_size)]
    debug_meta = [{'pos_thresh': None, 'neg_thresh': None, 'query_info': {}} for _ in range(batch_size)]

    if not getattr(criterion, 'enable_unk_label_obj', False):
        return indices, dummy_pos_indices, dummy_neg_indices, debug_meta
    if epoch < getattr(criterion, 'unk_label_start_epoch', 0):
        return indices, dummy_pos_indices, dummy_neg_indices, debug_meta

    obj_scores = outputs['pred_obj']      # [B, Q], energy
    pred_boxes = outputs['pred_boxes']    # [B, Q, 4]
    pred_logits = outputs['pred_logits']  # [B, Q, C]

    num_queries = obj_scores.shape[1]
    max_pos_per_img = getattr(criterion.args, 'unk_pos_per_img', 1)
    max_neg_per_img = getattr(criterion.args, 'unk_neg_per_img', 2)
    known_reject_thresh = getattr(criterion.args, 'unk_cls_reject_thresh', 0.25)
    pos_quantile = getattr(criterion.args, 'unk_label_pos_quantile', 0.5)

    pred_probs = pred_logits.detach().sigmoid().clone()
    pred_probs[:, :, criterion.invalid_cls_logits] = 0.0

    for i, (src_idx, _) in enumerate(indices):
        matched_scores = obj_scores[i, src_idx]

        if len(matched_scores) > 0:
            base_thresh = torch.quantile(matched_scores.detach(), pos_quantile).item()
            pos_thresh = base_thresh * criterion.unk_label_obj_score_thresh
        else:
            pos_thresh = getattr(criterion.args, 'default_pos_energy_thresh', 1.0)

        neg_thresh = pos_thresh + getattr(criterion.args, 'unk_label_neg_margin', 0.5)

        debug_meta[i]['pos_thresh'] = pos_thresh
        debug_meta[i]['neg_thresh'] = neg_thresh

        all_queries = set(range(num_queries))
        matched_set = set(src_idx.tolist())
        unmatched = list(all_queries - matched_set)

        # 先记录每个 query 的 energy / known_max / max_iou，方便后面绘图
        cur_known_prob = pred_probs[i, :, :criterion.num_classes - 1]
        known_max = cur_known_prob.max(dim=-1)[0]

        box_xyxy_all = box_cxcywh_to_xyxy(pred_boxes[i])
        gt_boxes_xyxy = box_cxcywh_to_xyxy(targets[i]['boxes'])

        if len(gt_boxes_xyxy) > 0:
            ious_all = box_iou(box_xyxy_all, gt_boxes_xyxy)[0]  # [Q, num_gt]
            max_iou_all = ious_all.max(dim=1)[0]
        else:
            max_iou_all = torch.zeros(num_queries, device=box_xyxy_all.device, dtype=box_xyxy_all.dtype)

        for q in range(num_queries):
            debug_meta[i]['query_info'][q] = {
                'energy': float(obj_scores[i, q].item()),
                'known_max': float(known_max[q].item()),
                'max_iou': float(max_iou_all[q].item())
            }

        if len(unmatched) == 0:
            continue

        valid_unmatched = [q for q in unmatched if max_iou_all[q].item() < 0.3]

        pos_candidates = [
            q for q in valid_unmatched
            if obj_scores[i, q].item() < pos_thresh
            and known_max[q].item() < known_reject_thresh
        ]
        neg_candidates = [
            q for q in valid_unmatched
            if obj_scores[i, q].item() > neg_thresh
            and known_max[q].item() < known_reject_thresh
        ]

        pos_candidates_sorted = sorted(pos_candidates, key=lambda q: obj_scores[i, q].item())
        neg_candidates_sorted = sorted(neg_candidates, key=lambda q: obj_scores[i, q].item(), reverse=True)

        dummy_pos_indices[i] = pos_candidates_sorted[:max_pos_per_img]
        dummy_neg_indices[i] = neg_candidates_sorted[:max_neg_per_img]

    return indices, dummy_pos_indices, dummy_neg_indices, debug_meta


def _draw_group_boxes(ax, boxes_xyxy, color, linewidth=2.0, linestyle='-',
                      labels=None, alpha=1.0):
    for j, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        rect = patches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=linewidth, edgecolor=color, facecolor='none',
            linestyle=linestyle, alpha=alpha
        )
        ax.add_patch(rect)

        if labels is not None and j < len(labels):
            ax.text(
                x1, max(0, y1 - 4), labels[j],
                fontsize=8, color=color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.0)
            )


def _make_query_label(prefix, q, meta):
    """
    meta: {'energy': float, 'known_max': float, 'max_iou': float}
    """
    return (
        f'{prefix}:{q} '
        f'E={meta["energy"]:.3f} '
        f'K={meta["known_max"]:.2f} '
        f'IoU={meta["max_iou"]:.2f}'
    )


@torch.no_grad()
def log_debug_visualizations(writer, samples, targets, outputs, criterion, epoch, global_step,
                             max_images=2, prefix='train_vis'):
    """
    画 matched / dummy_pos / dummy_neg / GT 四组框到 TensorBoard
    增强点：
    - 使用 target['size'] / samples.mask 处理真实可视尺寸，避免 padding 干扰
    - 伪标签重建逻辑与当前训练逻辑同步
    - 框上显示 query id / energy / known_max / max_iou
    """
    if writer is None:
        return

    indices, dummy_pos_indices, dummy_neg_indices, debug_meta = _build_debug_groups(
        outputs, targets, criterion, epoch
    )

    images = samples.tensors
    pred_boxes = outputs['pred_boxes']

    batch_size = min(images.shape[0], max_images)

    for b in range(batch_size):
        img_h, img_w = _get_valid_image_size(samples, targets, b)

        img_np = _denorm_image(images[b])
        img_np = img_np[:img_h, :img_w, :]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_np)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis('off')

        meta_b = debug_meta[b]
        pos_thresh = meta_b['pos_thresh']
        neg_thresh = meta_b['neg_thresh']

        # GT: green
        gt_boxes_abs = _norm_cxcywh_to_abs_xyxy(targets[b]['boxes'].detach(), img_w, img_h)
        gt_labels = [f'gt:{int(x)}' for x in targets[b]['labels'].detach().cpu().tolist()]
        _draw_group_boxes(
            ax, gt_boxes_abs.cpu(),
            color='green', linewidth=2.0, linestyle='-',
            labels=gt_labels, alpha=0.9
        )

        # matched: blue
        matched_q = indices[b][0]
        if len(matched_q) > 0:
            matched_boxes_abs = _norm_cxcywh_to_abs_xyxy(pred_boxes[b, matched_q].detach(), img_w, img_h)
            matched_labels = []
            for q in matched_q.detach().cpu().tolist():
                matched_labels.append(_make_query_label('m', q, meta_b['query_info'][q]))
            _draw_group_boxes(
                ax, matched_boxes_abs.cpu(),
                color='blue', linewidth=2.0, linestyle='-',
                labels=matched_labels, alpha=0.9
            )

        # dummy_pos: orange
        if len(dummy_pos_indices[b]) > 0:
            pos_q = torch.tensor(dummy_pos_indices[b], device=pred_boxes.device, dtype=torch.long)
            pos_boxes_abs = _norm_cxcywh_to_abs_xyxy(pred_boxes[b, pos_q].detach(), img_w, img_h)
            pos_labels = []
            for q in pos_q.detach().cpu().tolist():
                pos_labels.append(_make_query_label('pos', q, meta_b['query_info'][q]))
            _draw_group_boxes(
                ax, pos_boxes_abs.cpu(),
                color='orange', linewidth=2.6, linestyle='--',
                labels=pos_labels, alpha=0.95
            )

        # dummy_neg: red
        if len(dummy_neg_indices[b]) > 0:
            neg_q = torch.tensor(dummy_neg_indices[b], device=pred_boxes.device, dtype=torch.long)
            neg_boxes_abs = _norm_cxcywh_to_abs_xyxy(pred_boxes[b, neg_q].detach(), img_w, img_h)
            neg_labels = []
            for q in neg_q.detach().cpu().tolist():
                neg_labels.append(_make_query_label('neg', q, meta_b['query_info'][q]))
            _draw_group_boxes(
                ax, neg_boxes_abs.cpu(),
                color='red', linewidth=2.6, linestyle=':',
                labels=neg_labels, alpha=0.95
            )

        title_parts = [f'epoch={epoch}', f'step={global_step}', f'size=({img_h},{img_w})']
        if pos_thresh is not None:
            title_parts.append(f'pos_th={pos_thresh:.3f}')
        if neg_thresh is not None:
            title_parts.append(f'neg_th={neg_thresh:.3f}')
        title_parts.append(f'matched={len(matched_q)}')
        title_parts.append(f'dpos={len(dummy_pos_indices[b])}')
        title_parts.append(f'dneg={len(dummy_neg_indices[b])}')

        ax.set_title(' | '.join(title_parts), fontsize=10)

        writer.add_figure(f'{prefix}/image_{b}', fig, global_step)
        plt.close(fig)