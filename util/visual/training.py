
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from util.visual.evaluation import compute_train_query_score_statistics
from util.visual.helper import save_svg_figure


PALETTE = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'cyan': '#33BBEE',
    'red': '#CC3311',
    'green': '#009988',
    'magenta': '#EE3377',
    'yellow': '#EEDD44',
    'purple': '#7A52A5',
    'gray': '#6C757D',
}

MODEL_STAT_KEYS = {
    'stat_num_dummy_pos': 'train/model_stats/num_dummy_pos',
    'stat_num_dummy_neg': 'train/model_stats/num_dummy_neg',
    'stat_num_ignore_queries': 'train/model_stats/num_ignore_queries',
    'stat_num_valid_unmatched': 'train/model_stats/num_valid_unmatched',
    'stat_num_pos_candidates': 'train/model_stats/num_pos_candidates',
    'stat_num_neg_candidates': 'train/model_stats/num_neg_candidates',
    'stat_num_batch_selected_pos': 'train/model_stats/num_batch_selected_pos',
    'stat_pos_thresh_mean': 'train/model_stats/pos_thresh_mean',
    'stat_cls_attn_mean': 'train/model_stats/cls_attn_mean',
    'stat_num_cls_soft': 'train/model_stats/num_cls_soft',
    'gate_mean': 'train/model_stats/gate_mean',
}


def safe_float(value):
    """把值转成 float。"""
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


def safe_div(numerator, denominator):
    """安全做除法。"""
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def sum_optional_floats(*values):
    """把可转 float 的值求和。"""
    valid_values = []
    for value in values:
        numeric_value = safe_float(value)
        if numeric_value is not None:
            valid_values.append(numeric_value)
    return None if not valid_values else float(sum(valid_values))


def append_json_record(path: Path, record: dict):
    """向 jsonl 文件追加一条记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


def ema(values, alpha=0.15):
    """计算一维序列的 EMA 平滑。"""
    smoothed = []
    previous = None
    for value in values:
        if value is None:
            smoothed.append(previous)
            continue
        if previous is None:
            previous = value
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        smoothed.append(previous)
    return smoothed


def epoch_series(rows, key):
    """提取按 epoch 对齐的序列。"""
    xs = []
    ys = []
    for row in rows:
        epoch = row.get('epoch')
        value = safe_float(row.get(key))
        if epoch is None or value is None:
            continue
        xs.append(int(epoch))
        ys.append(value)
    return xs, ys


def build_train_epoch_record(epoch, train_stats, num_trainable_parameters):
    """构建训练 epoch 记录。"""
    train_stats = train_stats or {}
    return {
        'epoch': int(epoch),
        'num_trainable_parameters': num_trainable_parameters,
        'train_total_loss': safe_float(train_stats.get('loss')),
        'train_lr': safe_float(train_stats.get('lr')),
        'train_grad_norm': safe_float(train_stats.get('grad_norm')),
        'train_class_error': safe_float(train_stats.get('class_error')),
        'train_weighted_loss_ce': safe_float(train_stats.get('weighted_loss_ce')),
        'train_raw_loss_ce': safe_float(train_stats.get('raw_loss_ce')),
        'train_weighted_loss_bbox': safe_float(train_stats.get('weighted_loss_bbox')),
        'train_raw_loss_bbox': safe_float(train_stats.get('raw_loss_bbox')),
        'train_weighted_loss_giou': safe_float(train_stats.get('weighted_loss_giou')),
        'train_raw_loss_giou': safe_float(train_stats.get('raw_loss_giou')),
        'train_weighted_loss_obj_ll': safe_float(train_stats.get('weighted_loss_obj_ll')),
        'train_raw_loss_obj_ll': safe_float(train_stats.get('raw_loss_obj_ll')),
        'train_weighted_loss_unk_known': safe_float(train_stats.get('weighted_loss_unk_known')),
        'train_raw_loss_unk_known': safe_float(train_stats.get('raw_loss_unk_known')),
        'train_weighted_loss_obj_pseudo': safe_float(train_stats.get('weighted_loss_obj_pseudo')),
        'train_raw_loss_obj_pseudo': safe_float(train_stats.get('raw_loss_obj_pseudo')),
        'train_weighted_loss_obj_neg': safe_float(train_stats.get('weighted_loss_obj_neg')),
        'train_raw_loss_obj_neg': safe_float(train_stats.get('raw_loss_obj_neg')),
        'train_weighted_loss_unk_pseudo': safe_float(train_stats.get('weighted_loss_unk_pseudo')),
        'train_raw_loss_unk_pseudo': safe_float(train_stats.get('raw_loss_unk_pseudo')),
        'train_weighted_loss_decorr': safe_float(train_stats.get('weighted_loss_decorr')),
        'train_raw_loss_decorr': safe_float(train_stats.get('raw_loss_decorr')),
        'train_weighted_loss_bbox_pseudo_cons': safe_float(train_stats.get('weighted_loss_bbox_pseudo_cons')),
        'train_raw_loss_bbox_pseudo_cons': safe_float(train_stats.get('raw_loss_bbox_pseudo_cons')),
        'train_weighted_loss_giou_pseudo_cons': safe_float(train_stats.get('weighted_loss_giou_pseudo_cons')),
        'train_raw_loss_giou_pseudo_cons': safe_float(train_stats.get('raw_loss_giou_pseudo_cons')),
        'num_selected_pseudo_positive_queries': safe_float(train_stats.get('num_selected_pseudo_positive_queries', train_stats.get('stat_num_batch_selected_pos'))),
        'num_selected_reliable_background_queries': safe_float(train_stats.get('num_selected_reliable_background_queries', train_stats.get('stat_num_dummy_neg'))),
        'num_pseudo_positive_candidates': safe_float(train_stats.get('num_pseudo_positive_candidates', train_stats.get('stat_num_pos_candidates'))),
        'num_classification_ignored_queries': safe_float(train_stats.get('num_classification_ignored_queries', train_stats.get('stat_num_ignore_queries'))),
        'num_unmatched_queries_after_filter': safe_float(train_stats.get('num_unmatched_queries_after_filter', train_stats.get('stat_num_valid_unmatched'))),
        'train_gate_mean': safe_float(train_stats.get('gate_mean')),
        'train_pos_thresh_mean': safe_float(train_stats.get('stat_pos_thresh_mean')),
        'pseudo_positive_selection_ratio': safe_div(
            train_stats.get('num_selected_pseudo_positive_queries', train_stats.get('stat_num_batch_selected_pos')),
            train_stats.get('num_unmatched_queries_after_filter', train_stats.get('stat_num_valid_unmatched')),
        ),
        'pseudo_positive_accept_ratio': safe_div(
            train_stats.get('num_selected_pseudo_positive_queries', train_stats.get('stat_num_batch_selected_pos')),
            train_stats.get('num_pseudo_positive_candidates', train_stats.get('stat_num_pos_candidates')),
        ),
        'train_total_knownness_loss': sum_optional_floats(train_stats.get('raw_loss_unk_known'), train_stats.get('raw_loss_unk_pseudo')),
    }


def build_eval_epoch_record(epoch, eval_stats, num_trainable_parameters):
    """构建评估 epoch 记录。"""
    open_world_metrics = eval_stats.get('open_world_metrics', {}) if isinstance(eval_stats, dict) else {}
    return {
        'epoch': int(epoch),
        'num_trainable_parameters': num_trainable_parameters,
        'open_world_metrics': open_world_metrics,
    }


def write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch):
    """把评估指标写入 tensorboard。"""
    if viz_ctx is None or viz_ctx.tb_writer is None or not isinstance(eval_stats, dict):
        return
    for key, value in eval_stats.items():
        if key == 'open_world_metrics' and isinstance(value, dict):
            for metric_name, metric_value in value.items():
                if isinstance(metric_value, (int, float)):
                    tag = 'A-OSE' if metric_name == 'AOSA' else metric_name
                    viz_ctx.tb_writer.add_scalar(f'eval/metrics/{tag}', metric_value, epoch)
        elif isinstance(value, (int, float)):
            viz_ctx.tb_writer.add_scalar(f'eval/{key}', value, epoch)
