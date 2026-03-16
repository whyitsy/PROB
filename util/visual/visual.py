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


@torch.no_grad()
def _build_debug_groups(outputs, targets, criterion, epoch):
    """
    根据当前最终层输出，重建 matched / dummy_pos / dummy_neg，专用于可视化。
    返回:
        indices, dummy_pos_indices, dummy_neg_indices
    """
    outputs_wo_aux = {
        k: v for k, v in outputs.items()
        if k not in ['aux_outputs', 'enc_outputs', 'pred_obj', 'samples', 'pred_proj']
    }
    indices = criterion.matcher(outputs_wo_aux, targets)

    dummy_pos_indices = [[] for _ in range(len(targets))]
    dummy_neg_indices = [[] for _ in range(len(targets))]

    if not getattr(criterion, 'enable_unk_label_obj', False):
        return indices, dummy_pos_indices, dummy_neg_indices
    if epoch < getattr(criterion, 'unk_label_start_epoch', 0):
        return indices, dummy_pos_indices, dummy_neg_indices

    obj_scores = outputs['pred_obj']  # [B, Q], energy
    pred_boxes = outputs['pred_boxes']

    for i, (src_idx, _) in enumerate(indices):
        matched_scores = obj_scores[i, src_idx]

        if len(matched_scores) > 0:
            pos_thresh = matched_scores.mean().item() * criterion.unk_label_obj_score_thresh
        else:
            pos_thresh = getattr(criterion.args, 'default_pos_energy_thresh', 1.0)

        neg_thresh = pos_thresh + getattr(criterion.args, 'unk_label_neg_margin', 0.5)

        num_queries = obj_scores.shape[1]
        all_queries = set(range(num_queries))
        matched_set = set(src_idx.tolist())
        unmatched = list(all_queries - matched_set)

        if len(unmatched) == 0:
            continue

        box_xyxy_all = box_cxcywh_to_xyxy(pred_boxes[i])
        gt_boxes_xyxy = box_cxcywh_to_xyxy(targets[i]['boxes'])

        if len(gt_boxes_xyxy) > 0:
            ious = box_iou(box_xyxy_all[unmatched], gt_boxes_xyxy)[0]
            max_ious = ious.max(dim=1)[0]
            valid_unmatched = [unmatched[j] for j, max_iou in enumerate(max_ious) if max_iou < 0.3]
        else:
            valid_unmatched = unmatched

        pos_candidates = [q for q in valid_unmatched if obj_scores[i, q].item() < pos_thresh]
        pos_candidates_sorted = sorted(pos_candidates, key=lambda q: obj_scores[i, q].item())
        dummy_pos_indices[i] = pos_candidates_sorted[:1]

        neg_candidates = [q for q in valid_unmatched if obj_scores[i, q].item() > neg_thresh]
        neg_candidates_sorted = sorted(neg_candidates, key=lambda q: obj_scores[i, q].item(), reverse=True)
        dummy_neg_indices[i] = neg_candidates_sorted[:2]

    return indices, dummy_pos_indices, dummy_neg_indices


def _draw_group_boxes(ax, boxes_xyxy, color, linewidth=2.0, linestyle='-',
                      labels=None, alpha=1.0):
    for j, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color, facecolor='none',
            linestyle=linestyle, alpha=alpha
        )
        ax.add_patch(rect)
        if labels is not None and j < len(labels):
            ax.text(
                x1, max(0, y1 - 3), labels[j],
                fontsize=8, color=color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.0)
            )


@torch.no_grad()
def log_debug_visualizations(writer, samples, targets, outputs, criterion, epoch, global_step,
                             max_images=2, prefix='train_vis'):
    """
    画 matched / dummy_pos / dummy_neg / GT 四组框到 TensorBoard
    """
    if writer is None:
        return

    indices, dummy_pos_indices, dummy_neg_indices = _build_debug_groups(outputs, targets, criterion, epoch)

    images = samples.tensors
    pred_boxes = outputs['pred_boxes']
    pred_obj = outputs['pred_obj']

    batch_size = min(images.shape[0], max_images)

    for b in range(batch_size):
        img_np = _denorm_image(images[b])
        H, W = img_np.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_np)
        ax.axis('off')

        # GT: green
        gt_boxes = box_cxcywh_to_xyxy(targets[b]['boxes'].detach())
        gt_boxes_abs = gt_boxes * torch.tensor([W, H, W, H], device=gt_boxes.device)
        gt_labels = [f'gt:{int(x)}' for x in targets[b]['labels'].detach().cpu().tolist()]
        _draw_group_boxes(ax, gt_boxes_abs.cpu(), color='green', linewidth=2.0, labels=gt_labels, alpha=0.9)

        # matched: blue
        matched_q = indices[b][0]
        if len(matched_q) > 0:
            matched_boxes = box_cxcywh_to_xyxy(pred_boxes[b, matched_q].detach())
            matched_boxes_abs = matched_boxes * torch.tensor([W, H, W, H], device=matched_boxes.device)
            matched_energy = pred_obj[b, matched_q].detach().cpu().tolist()
            matched_labels = [f'm:{q.item()} e={e:.3f}' for q, e in zip(matched_q, matched_energy)]
            _draw_group_boxes(ax, matched_boxes_abs.cpu(), color='blue', linewidth=2.0,
                              linestyle='-', labels=matched_labels, alpha=0.9)

        # dummy_pos: orange
        if len(dummy_pos_indices[b]) > 0:
            pos_q = torch.tensor(dummy_pos_indices[b], device=pred_boxes.device, dtype=torch.long)
            pos_boxes = box_cxcywh_to_xyxy(pred_boxes[b, pos_q].detach())
            pos_boxes_abs = pos_boxes * torch.tensor([W, H, W, H], device=pos_boxes.device)
            pos_energy = pred_obj[b, pos_q].detach().cpu().tolist()
            pos_labels = [f'pos:{q.item()} e={e:.3f}' for q, e in zip(pos_q, pos_energy)]
            _draw_group_boxes(ax, pos_boxes_abs.cpu(), color='orange', linewidth=2.5,
                              linestyle='--', labels=pos_labels, alpha=0.95)

        # dummy_neg: red
        if len(dummy_neg_indices[b]) > 0:
            neg_q = torch.tensor(dummy_neg_indices[b], device=pred_boxes.device, dtype=torch.long)
            neg_boxes = box_cxcywh_to_xyxy(pred_boxes[b, neg_q].detach())
            neg_boxes_abs = neg_boxes * torch.tensor([W, H, W, H], device=neg_boxes.device)
            neg_energy = pred_obj[b, neg_q].detach().cpu().tolist()
            neg_labels = [f'neg:{q.item()} e={e:.3f}' for q, e in zip(neg_q, neg_energy)]
            _draw_group_boxes(ax, neg_boxes_abs.cpu(), color='red', linewidth=2.5,
                              linestyle=':', labels=neg_labels, alpha=0.95)

        writer.add_figure(f'{prefix}/image_{b}', fig, global_step)
        plt.close(fig)