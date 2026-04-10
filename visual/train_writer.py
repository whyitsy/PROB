import json
from pathlib import Path

import torch


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu().item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_div(numerator, denominator):
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def _append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


def _compute_query_score_statistics(outputs, targets, criterion, args):
    stats = {}
    objectness_energy = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
    class_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits')
    if objectness_energy is None or class_logits is None:
        return stats

    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    objectness_temperature = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    objectness_probability = torch.exp(-objectness_temperature * objectness_energy.detach())

    unknown_logit = _get_output(outputs, 'pred_unknown_logit', 'pred_unk')
    if unknown_logit is not None:
        unknown_probability = torch.sigmoid(unknown_logit.detach())
    else:
        knownness_energy = _get_output(outputs, 'pred_knownness_energy', 'pred_known')
        if knownness_energy is not None:
            knownness_temperature = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim
            unknown_probability = 1.0 - torch.exp(-knownness_temperature * knownness_energy.detach())
        else:
            unknown_probability = torch.zeros_like(objectness_probability)

    class_probability = class_logits.detach().sigmoid().clone()
    invalid_class_indices = getattr(criterion, 'invalid_cls_logits', [])
    if len(invalid_class_indices) > 0:
        class_probability[:, :, invalid_class_indices] = 0.0
    if class_probability.shape[-1] > 0:
        class_probability[:, :, -1] = 0.0
    max_known_class_probability = class_probability.max(-1).values

    matcher_outputs = {
        'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'),
        'pred_boxes': outputs['pred_boxes'],
    }
    try:
        matched_indices = criterion.matcher(matcher_outputs, targets)
        batch_size, num_queries = objectness_probability.shape[:2]
        matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=objectness_probability.device)
        for batch_index, (source_indices, _) in enumerate(matched_indices):
            if len(source_indices) > 0:
                matched_mask[batch_index, source_indices] = True
        if matched_mask.any():
            stats['train/query_stats/matched_objectness_prob_mean'] = _safe_float(objectness_probability[matched_mask].mean())
        if (~matched_mask).any():
            stats['train/query_stats/unmatched_objectness_prob_mean'] = _safe_float(objectness_probability[~matched_mask].mean())
    except Exception:
        pass

    stats['train/query_stats/unknown_probability_mean'] = _safe_float(unknown_probability.mean())
    stats['train/query_stats/max_known_class_probability_mean'] = _safe_float(max_known_class_probability.mean())

    gate_mean = _get_output(outputs, 'odqe_gate_mean', 'gate_mean')
    if gate_mean is not None:
        stats['train/query_stats/odqe_gate_mean'] = _safe_float(gate_mean)

    gate_per_layer = _get_output(outputs, 'odqe_gate_mean_per_layer', 'gate_mean_per_layer')
    if gate_per_layer is not None and torch.is_tensor(gate_per_layer):
        for layer_index, value in enumerate(gate_per_layer.detach().flatten()):
            stats[f'train/query_stats/odqe_gate_mean_layer_{layer_index}'] = _safe_float(value)

    return stats


def write_train_step_artifacts(
    tb_writer,
    step_jsonl_path,
    global_step,
    epoch,
    local_step,
    optimizer,
    grad_total_norm,
    outputs,
    targets,
    criterion,
    total_loss,
    reduced_loss_dict,
    reduced_weighted_loss_dict,
    viz_cfg=None,
    args=None,
):
    record = {
        'global_step': int(global_step),
        'epoch': int(epoch),
        'local_step': int(local_step),
        'train/loss/total': float(total_loss),
        'train/optim/lr': _safe_float(optimizer.param_groups[0]['lr']),
        'train/optim/grad_norm': _safe_float(grad_total_norm),
    }

    if 'class_error' in reduced_loss_dict:
        record['train/quality/class_error'] = _safe_float(reduced_loss_dict['class_error'])

    for key, value in reduced_weighted_loss_dict.items():
        record[f'train/loss_weighted/{key}'] = _safe_float(value)
    for key, value in reduced_loss_dict.items():
        record[f'train/loss_raw/{key}'] = _safe_float(value)

    selection_count = reduced_loss_dict.get('num_selected_pseudo_positive_queries', reduced_loss_dict.get('stat_num_batch_selected_pos'))
    unmatched_count = reduced_loss_dict.get('num_unmatched_queries_after_filter', reduced_loss_dict.get('stat_num_valid_unmatched'))
    candidate_count = reduced_loss_dict.get('num_pseudo_positive_candidates', reduced_loss_dict.get('stat_num_pos_candidates'))
    record['train/pseudo/selection_ratio'] = _safe_div(selection_count, unmatched_count)
    record['train/pseudo/accept_ratio'] = _safe_div(selection_count, candidate_count)

    record.update(_compute_query_score_statistics(outputs, targets, criterion, args))
    _append_jsonl(Path(step_jsonl_path), record)

    if tb_writer is None or viz_cfg is None:
        return

    for key, value in record.items():
        if key in {'global_step', 'epoch', 'local_step'} or value is None:
            continue
        tb_writer.add_scalar(key, value, global_step)

    objectness_energy = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
    unknown_logit = _get_output(outputs, 'pred_unknown_logit', 'pred_unk')
    histogram_interval = 100
    if global_step % histogram_interval == 0:
        if objectness_energy is not None:
            tb_writer.add_histogram('train/distribution/objectness_energy', objectness_energy.detach().float().cpu(), global_step)
        if unknown_logit is not None:
            tb_writer.add_histogram('train/distribution/unknown_logit', unknown_logit.detach().float().cpu(), global_step)
            tb_writer.add_histogram('train/distribution/unknown_probability', torch.sigmoid(unknown_logit.detach()).float().cpu(), global_step)
