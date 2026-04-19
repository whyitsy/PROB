
import json
from pathlib import Path

import torch

from visual.postprocess_aligned_stats import compute_train_query_score_statistics


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


def _append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


def _add_numeric_items(record, prefix, values):
    for key, value in values.items():
        numeric_value = _safe_float(value)
        if numeric_value is not None:
            record[f'{prefix}/{key}'] = numeric_value


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
        'train/lr': _safe_float(optimizer.param_groups[0]['lr']) if optimizer is not None else None,
        'train/grad_norm': _safe_float(grad_total_norm),
    }

    _add_numeric_items(record, 'train/loss_raw', reduced_loss_dict)
    _add_numeric_items(record, 'train/loss_weighted', reduced_weighted_loss_dict)

    selected_count = reduced_loss_dict.get('num_selected_pseudo_positive_queries', reduced_loss_dict.get('stat_num_batch_selected_pos'))
    candidate_count = reduced_loss_dict.get('num_pseudo_positive_candidates', reduced_loss_dict.get('stat_num_pos_candidates'))
    unmatched_count = reduced_loss_dict.get('num_unmatched_queries_after_filter', reduced_loss_dict.get('stat_num_valid_unmatched'))
    reliable_background_count = reduced_loss_dict.get('num_selected_reliable_background_queries', reduced_loss_dict.get('stat_num_dummy_neg'))
    ignored_count = reduced_loss_dict.get('num_classification_ignored_queries', reduced_loss_dict.get('stat_num_ignore_queries'))

    record['train/pseudo/selected_queries'] = _safe_float(selected_count)
    record['train/pseudo/candidate_queries'] = _safe_float(candidate_count)
    record['train/pseudo/valid_unmatched_queries'] = _safe_float(unmatched_count)
    record['train/pseudo/reliable_background_queries'] = _safe_float(reliable_background_count)
    record['train/pseudo/ignored_queries'] = _safe_float(ignored_count)

    record.update(compute_train_query_score_statistics(outputs, targets, criterion, args))
    _append_jsonl(Path(step_jsonl_path), record)

    if tb_writer is None or viz_cfg is None:
        return

    for key, value in record.items():
        if key in {'global_step', 'epoch', 'local_step'} or value is None:
            continue
        tb_writer.add_scalar(key, value, global_step)

    objectness_energy = outputs.get('pred_obj', outputs.get('pred_objectness_energy', None))
    unknown_logit = outputs.get('pred_unk', outputs.get('pred_unknown_logit', None))
    histogram_interval = 100
    if global_step % histogram_interval == 0:
        if objectness_energy is not None:
            tb_writer.add_histogram('train/distribution/objectness_energy', objectness_energy.detach().float().cpu(), global_step)
        if unknown_logit is not None:
            tb_writer.add_histogram('train/distribution/unknown_logit', unknown_logit.detach().float().cpu(), global_step)
            tb_writer.add_histogram('train/distribution/unknown_probability', torch.sigmoid(unknown_logit.detach()).float().cpu(), global_step)
