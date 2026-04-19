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


def step_series(rows, key):
    """提取按 global step 对齐的序列。"""
    xs = []
    ys = []
    for row in rows:
        step = row.get('global_step')
        value = safe_float(row.get(key))
        if step is None or value is None:
            continue
        xs.append(int(step))
        ys.append(value)
    return xs, ys


def aggregate_step_values(rows, keys, window_size=1000):
    """按 step 窗口聚合多条序列。"""
    valid_rows = [row for row in rows if row.get('global_step') is not None]
    if not valid_rows:
        return None
    valid_rows = sorted(valid_rows, key=lambda item: int(item['global_step']))
    grouped = {}
    counts = {}
    for row in valid_rows:
        step = int(row['global_step'])
        bucket_end = ((step // window_size) + 1) * window_size
        if bucket_end not in grouped:
            grouped[bucket_end] = {key: 0.0 for key in keys}
            counts[bucket_end] = {key: 0 for key in keys}
        for key in keys:
            value = safe_float(row.get(key))
            if value is None:
                continue
            grouped[bucket_end][key] += value
            counts[bucket_end][key] += 1
    xs = sorted(grouped.keys())
    data = {}
    for key in keys:
        data[key] = []
        for bucket in xs:
            count = counts[bucket][key]
            data[key].append(grouped[bucket][key] / max(count, 1))
    return xs, data


def plot_training_total_loss(rows, output_path):
    """绘制训练总 loss 曲线。"""
    xs, ys = epoch_series(rows, 'train_total_loss')
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, marker='o', linewidth=2.2, color=PALETTE['blue'], label='total_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Total Loss Trend')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def plot_training_base_loss_components(rows, output_path):
    """绘制基础 loss 组成曲线。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for label, key, color in [
        ('classification', 'train_raw_loss_ce', PALETTE['blue']),
        ('box_l1', 'train_raw_loss_bbox', PALETTE['orange']),
        ('giou', 'train_raw_loss_giou', PALETTE['green']),
    ]:
        xs, ys = epoch_series(rows, key)
        if xs:
            plotted = True
            ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw loss')
    ax.set_title('Base Detection Loss Components')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    return save_svg_figure(fig, output_path)


def plot_training_matched_objectness_loss_component(rows, output_path):
    """绘制 matched objectness loss 曲线。"""
    xs, ys = epoch_series(rows, 'train_raw_loss_obj_ll')
    if not xs:
        xs, ys = epoch_series(rows, 'train_weighted_loss_obj_ll')
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, marker='o', linewidth=2.2, color=PALETTE['cyan'], label='matched_objectness')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Matched Objectness Loss Component')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def plot_training_open_world_loss_components(rows, output_path):
    """绘制 open-world loss 组成曲线。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for label, key, color in [
        ('matched_known_knownness', 'train_raw_loss_unk_known', PALETTE['orange']),
        ('pseudo_positive_objectness', 'train_raw_loss_obj_pseudo', PALETTE['blue']),
        ('pseudo_unknown_knownness', 'train_raw_loss_unk_pseudo', PALETTE['magenta']),
        ('branch_decorrelation', 'train_raw_loss_decorr', PALETTE['green']),
    ]:
        xs, ys = epoch_series(rows, key)
        if xs:
            plotted = True
            ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Raw loss')
    ax.set_title('Open-World Loss Components')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def plot_pseudo_mining_counts(rows, output_path):
    """绘制 pseudo mining 数量曲线。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    for label, key, color in [
        ('selected_pseudo_positive_queries', 'num_selected_pseudo_positive_queries', PALETTE['blue']),
        ('reliable_background_queries', 'num_selected_reliable_background_queries', PALETTE['orange']),
        ('candidate_queries', 'num_pseudo_positive_candidates', PALETTE['green']),
        ('ignored_queries', 'num_classification_ignored_queries', PALETTE['magenta']),
    ]:
        xs, ys = epoch_series(rows, key)
        if xs:
            plotted = True
            ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Count')
    ax.set_title('Pseudo Mining Count Statistics')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def plot_pseudo_mining_efficiency(rows, output_path):
    """绘制 pseudo mining 效率曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for label, key, color in [
        ('selection_ratio', 'pseudo_positive_selection_ratio', PALETTE['cyan']),
        ('accept_ratio', 'pseudo_positive_accept_ratio', PALETTE['red']),
    ]:
        xs, ys = epoch_series(rows, key)
        if xs:
            plotted = True
            ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=label)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ratio')
    ax.set_title('Pseudo Mining Efficiency')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def plot_step_group(rows, output_path, *, title, ylabel, series):
    """绘制一组 step 曲线。"""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    plotted = False
    for label, key, color in series:
        xs, ys = step_series(rows, key)
        if xs:
            plotted = True
            ax.plot(xs, ys, alpha=0.18, linewidth=0.9, color=color)
            ax.plot(xs, ema(ys, alpha=0.08), linewidth=2.0, color=color, label=label)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Global Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    return save_svg_figure(fig, output_path)


def plot_step_total_loss(rows, output_path):
    """绘制 step 总 loss 曲线。"""
    return plot_step_group(rows, output_path, title='Step-level Total Loss', ylabel='Loss', series=[('total_loss', 'train/loss/total', PALETTE['blue'])])


def plot_step_base_losses(rows, output_path):
    """绘制 step 基础 loss 曲线。"""
    return plot_step_group(
        rows,
        output_path,
        title='Step-level Base Losses',
        ylabel='Raw loss',
        series=[
            ('classification', 'train/loss_raw/loss_ce', PALETTE['blue']),
            ('box_l1', 'train/loss_raw/loss_bbox', PALETTE['orange']),
            ('giou', 'train/loss_raw/loss_giou', PALETTE['green']),
            ('matched_objectness', 'train/loss_raw/loss_obj_ll', PALETTE['cyan']),
        ],
    )


def plot_step_open_world_losses(rows, output_path):
    """绘制 step open-world loss 曲线。"""
    return plot_step_group(
        rows,
        output_path,
        title='Step-level Open-World Losses',
        ylabel='Raw loss',
        series=[
            ('matched_known_knownness', 'train/loss_raw/loss_unk_known', PALETTE['orange']),
            ('pseudo_positive_objectness', 'train/loss_raw/loss_obj_pseudo', PALETTE['blue']),
            ('pseudo_unknown_knownness', 'train/loss_raw/loss_unk_pseudo', PALETTE['magenta']),
            ('branch_decorrelation', 'train/loss_raw/loss_decorr', PALETTE['green']),
        ],
    )


def plot_step_query_score_statistics(rows, output_path):
    """绘制 step query score 统计曲线。"""
    return plot_step_group(
        rows,
        output_path,
        title='Step-level Query Score Statistics',
        ylabel='Value',
        series=[
            ('matched_objectness_prob', 'train/query_stats/matched_objectness_prob_mean', PALETTE['blue']),
            ('unmatched_objectness_prob', 'train/query_stats/unmatched_objectness_prob_mean', PALETTE['orange']),
            ('unknown_probability', 'train/query_stats/unknown_probability_mean', PALETTE['magenta']),
            ('max_known_class_probability', 'train/query_stats/max_known_class_probability_mean', PALETTE['green']),
        ],
    )


def plot_step_pseudo_mining_statistics(rows, output_path):
    """绘制 step pseudo mining 统计曲线。"""
    return plot_step_group(
        rows,
        output_path,
        title='Step-level Pseudo Mining Statistics',
        ylabel='Value',
        series=[
            ('selected_queries', 'train/pseudo/selected_queries', PALETTE['blue']),
            ('candidate_queries', 'train/pseudo/candidate_queries', PALETTE['green']),
            ('reliable_background_queries', 'train/pseudo/reliable_background_queries', PALETTE['orange']),
            ('ignored_queries', 'train/pseudo/ignored_queries', PALETTE['magenta']),
        ],
    )


def plot_step_pseudo_mining_counts_bars(rows, output_path, step_bar_interval=1000):
    """绘制 step pseudo mining 柱状统计图。"""
    value_keys = [
        ('selected', 'train/pseudo/selected_queries', PALETTE['blue']),
        ('candidates', 'train/pseudo/candidate_queries', PALETTE['green']),
        ('reliable_bg', 'train/pseudo/reliable_background_queries', PALETTE['orange']),
        ('ignored', 'train/pseudo/ignored_queries', PALETTE['magenta']),
    ]
    aggregated = aggregate_step_values(rows, [key for _, key, _ in value_keys], window_size=step_bar_interval)
    if aggregated is None:
        return None
    xs, data = aggregated
    if len(xs) == 0:
        return None

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bar_width = step_bar_interval * 0.18
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, num=len(value_keys))
    for offset, (label, key, color) in zip(offsets, value_keys):
        ax.bar(np.asarray(xs) + offset, data[key], width=bar_width, color=color, alpha=0.9, label=label)
    ax.set_xlabel(f'Global step (window={step_bar_interval})')
    ax.set_ylabel('Average count per step')
    ax.set_title('Pseudo Mining Counts Aggregated by Step Window')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    return save_svg_figure(fig, output_path)


def plot_step_auxiliary_family(rows, output_path, *, title, prefixes):
    """绘制一组辅助 loss 曲线。"""
    keys = sorted({key for row in rows for key in row.keys() if any(key.startswith(prefix) for prefix in prefixes)})
    if not keys:
        return None

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['magenta'], PALETTE['cyan'], PALETTE['red'], PALETTE['purple']]
    plotted = False
    for index, key in enumerate(keys):
        xs, ys = step_series(rows, key)
        if xs:
            plotted = True
            ax.plot(xs, ema(ys, alpha=0.08), linewidth=1.8, color=colors[index % len(colors)], label=key.replace('train/loss_raw/', ''))
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Raw loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return save_svg_figure(fig, output_path)


def plot_step_aux_obj_pseudo_loss_trends(rows, output_path):
    """绘制 obj pseudo 辅助 loss 曲线。"""
    return plot_step_auxiliary_family(rows, output_path, title='Step-level Auxiliary Objectness Pseudo Loss Trends', prefixes=['train/loss_raw/loss_obj_pseudo_'])


def plot_step_aux_unk_pseudo_loss_trends(rows, output_path):
    """绘制 unk pseudo 辅助 loss 曲线。"""
    return plot_step_auxiliary_family(rows, output_path, title='Step-level Auxiliary Unknownness Pseudo Loss Trends', prefixes=['train/loss_raw/loss_unk_pseudo_'])


def plot_step_aux_decorr_loss_trends(rows, output_path):
    """绘制 decorr 辅助 loss 曲线。"""
    return plot_step_auxiliary_family(rows, output_path, title='Step-level Auxiliary Decorrelation Loss Trends', prefixes=['train/loss_raw/loss_decorr_'])


def plot_step_auxiliary_loss_trends(rows, output_path):
    """绘制全部辅助 loss 汇总曲线。"""
    return plot_step_auxiliary_family(
        rows,
        output_path,
        title='Step-level Auxiliary Loss Trends',
        prefixes=['train/loss_raw/loss_obj_pseudo_', 'train/loss_raw/loss_unk_pseudo_', 'train/loss_raw/loss_decorr_'],
    )


def _add_numeric_items(record, prefix, values):
    """把字典里的数值写入记录。"""
    for key, value in values.items():
        numeric_value = safe_float(value)
        if numeric_value is not None:
            record[f'{prefix}/{key}'] = numeric_value


def write_train_step_artifacts(tb_writer, step_jsonl_path, global_step, epoch, local_step, optimizer, grad_total_norm, outputs, targets, criterion, total_loss, reduced_loss_dict, reduced_weighted_loss_dict, viz_cfg=None, args=None):
    """写训练 step 的数值记录和 tensorboard 标量。"""
    record = {
        'global_step': int(global_step),
        'epoch': int(epoch),
        'local_step': int(local_step),
        'train/loss/total': float(total_loss),
        'train/lr': safe_float(optimizer.param_groups[0]['lr']) if optimizer is not None else None,
        'train/grad_norm': safe_float(grad_total_norm),
    }
    _add_numeric_items(record, 'train/loss_raw', reduced_loss_dict)
    _add_numeric_items(record, 'train/loss_weighted', reduced_weighted_loss_dict)

    selected_count = reduced_loss_dict.get('num_selected_pseudo_positive_queries', reduced_loss_dict.get('stat_num_batch_selected_pos'))
    candidate_count = reduced_loss_dict.get('num_pseudo_positive_candidates', reduced_loss_dict.get('stat_num_pos_candidates'))
    unmatched_count = reduced_loss_dict.get('num_unmatched_queries_after_filter', reduced_loss_dict.get('stat_num_valid_unmatched'))
    reliable_background_count = reduced_loss_dict.get('num_selected_reliable_background_queries', reduced_loss_dict.get('stat_num_dummy_neg'))
    ignored_count = reduced_loss_dict.get('num_classification_ignored_queries', reduced_loss_dict.get('stat_num_ignore_queries'))
    record['train/pseudo/selected_queries'] = safe_float(selected_count)
    record['train/pseudo/candidate_queries'] = safe_float(candidate_count)
    record['train/pseudo/valid_unmatched_queries'] = safe_float(unmatched_count)
    record['train/pseudo/reliable_background_queries'] = safe_float(reliable_background_count)
    record['train/pseudo/ignored_queries'] = safe_float(ignored_count)

    record.update(compute_train_query_score_statistics(outputs, targets, criterion, args))
    append_json_record(Path(step_jsonl_path), record)

    if tb_writer is None or viz_cfg is None:
        return
    for key, value in record.items():
        if key in {'global_step', 'epoch', 'local_step'} or value is None:
            continue
        tb_writer.add_scalar(key, value, global_step)

    objectness_energy = outputs.get('pred_obj', outputs.get('pred_objectness_energy', None))
    unknown_logit = outputs.get('pred_unk', outputs.get('pred_unknown_logit', None))
    if global_step % 100 == 0:
        if objectness_energy is not None:
            tb_writer.add_histogram('train/distribution/objectness_energy', objectness_energy.detach().float().cpu(), global_step)
        if unknown_logit is not None:
            tb_writer.add_histogram('train/distribution/unknown_logit', unknown_logit.detach().float().cpu(), global_step)
            tb_writer.add_histogram('train/distribution/unknown_probability', torch.sigmoid(unknown_logit.detach()).float().cpu(), global_step)


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
