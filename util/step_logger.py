import json
from pathlib import Path

import torch


def _safe_float(v):
    if v is None:
        return None
    if torch.is_tensor(v):
        try:
            return float(v.detach().cpu().item())
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_div(num, den):
    num = _safe_float(num)
    den = _safe_float(den)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num / den)


def _append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _score_stats(outputs, targets, criterion, args):
    stats = {}
    pred_obj = outputs.get('pred_obj')
    pred_logits = outputs.get('pred_logits')
    if pred_obj is None or pred_logits is None:
        return stats
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0))
    obj_prob = torch.exp(-(obj_temp / hidden_dim) * pred_obj.detach())
    pred_unk = outputs.get('pred_unk')
    unk_prob = torch.sigmoid(pred_unk.detach()) if pred_unk is not None else torch.zeros_like(obj_prob)
    cls_prob = pred_logits.detach().sigmoid().clone()
    invalid_cls = getattr(criterion, 'invalid_cls_logits', [])
    if len(invalid_cls) > 0:
        cls_prob[:, :, invalid_cls] = 0.0
    cls_prob[:, :, -1] = 0.0
    cls_max = cls_prob.max(-1).values
    outputs_for_match = {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
    try:
        indices_for_hist = criterion.matcher(outputs_for_match, targets)
        batch_size, num_queries = outputs['pred_obj'].shape[:2]
        matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=obj_prob.device)
        for b_idx, (src, _) in enumerate(indices_for_hist):
            if len(src) > 0:
                matched_mask[b_idx, src] = True
        if matched_mask.any():
            stats['train_stats/obj_prob_matched_mean'] = _safe_float(obj_prob[matched_mask].mean())
        if (~matched_mask).any():
            stats['train_stats/obj_prob_unmatched_mean'] = _safe_float(obj_prob[~matched_mask].mean())
    except Exception:
        pass
    stats['train_stats/unk_prob_mean'] = _safe_float(unk_prob.mean())
    stats['train_stats/cls_max_mean'] = _safe_float(cls_max.mean())
    if 'known_unk_suppress_coeff' in outputs:
        stats['train_stats/known_unk_suppress_coeff'] = _safe_float(outputs['known_unk_suppress_coeff'])
    if 'unknown_known_suppress_coeff' in outputs:
        stats['train_stats/unknown_known_suppress_coeff'] = _safe_float(outputs['unknown_known_suppress_coeff'])
    if 'gate_mean' in outputs and outputs['gate_mean'] is not None:
        stats['train_stats/odqe_gate_mean'] = _safe_float(outputs['gate_mean'])
    gate_per_layer = outputs.get('gate_mean_per_layer')
    if gate_per_layer is not None and torch.is_tensor(gate_per_layer):
        for i, val in enumerate(gate_per_layer.detach().flatten()):
            stats[f'train_stats/odqe_gate_mean_l{i}'] = _safe_float(val)
    return stats


def log_train_step(writer, step_jsonl_path, global_step, epoch, local_step, optimizer, grad_total_norm,
                   outputs, targets, criterion, loss_value, loss_dict_reduced, loss_dict_reduced_scaled,
                   hist_every=100, args=None):
    record = {
        'global_step': int(global_step),
        'epoch': int(epoch),
        'local_step': int(local_step),
        'train/total_loss': float(loss_value),
        'train/lr': _safe_float(optimizer.param_groups[0]['lr']),
        'train/grad_norm': _safe_float(grad_total_norm),
    }
    if 'class_error' in loss_dict_reduced:
        record['train/class_error'] = _safe_float(loss_dict_reduced['class_error'])
    for k, v in loss_dict_reduced_scaled.items():
        record[f'train_scaled/{k}'] = _safe_float(v)
    for k, v in loss_dict_reduced.items():
        record[f'train_unscaled/{k}'] = _safe_float(v)
    for stat_key in [
        'stat_num_dummy_pos', 'stat_num_valid_unmatched', 'stat_num_pos_candidates',
        'stat_num_batch_selected_pos', 'stat_pos_thresh_mean', 'stat_cls_attn_mean', 'stat_num_cls_soft'
    ]:
        if stat_key in loss_dict_reduced:
            record[f'train_stats/{stat_key}'] = _safe_float(loss_dict_reduced[stat_key])
    record['train_stats/pseudo_selection_ratio'] = _safe_div(loss_dict_reduced.get('stat_num_batch_selected_pos'), loss_dict_reduced.get('stat_num_valid_unmatched'))
    record['train_stats/pseudo_accept_ratio'] = _safe_div(loss_dict_reduced.get('stat_num_batch_selected_pos'), loss_dict_reduced.get('stat_num_pos_candidates'))
    record.update(_score_stats(outputs, targets, criterion, args))
    _append_jsonl(Path(step_jsonl_path), record)
    if writer is None:
        return
    for key, value in record.items():
        if key in {'global_step', 'epoch', 'local_step'} or value is None:
            continue
        writer.add_scalar(key, value, global_step)
    if hist_every > 0 and (global_step % hist_every == 0):
        if 'pred_obj' in outputs:
            writer.add_histogram('train_hist/pred_obj_energy_all', outputs['pred_obj'].detach().float().cpu(), global_step)
        if 'pred_unk' in outputs:
            writer.add_histogram('train_hist/pred_unk_logits_all', outputs['pred_unk'].detach().float().cpu(), global_step)
            writer.add_histogram('train_hist/pred_unk_prob_all', torch.sigmoid(outputs['pred_unk'].detach()).float().cpu(), global_step)
