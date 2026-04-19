import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

GROUP_NAMES = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
GROUP_COLORS = [PALETTE['green'], PALETTE['magenta'], PALETTE['gray']]


def safe_float(value):
    """把值转成 float。"""
    try:
        return float(value)
    except Exception:
        return None


def open_world_history(eval_rows):
    """整理 open-world 指标历史。"""
    history = []
    for row in eval_rows:
        epoch = row.get('epoch')
        metrics = row.get('open_world_metrics') or row.get('test_metrics') or {}
        if epoch is None:
            continue
        history.append(
            {
                'epoch': int(epoch),
                'current_ap50': safe_float(metrics.get('CK_AP50')),
                'known_ap50': safe_float(metrics.get('K_AP50')),
                'unknown_recall50': safe_float(metrics.get('U_R50')),
                'wilderness_impact': safe_float(metrics.get('WI')),
                'absolute_open_set_error': safe_float(metrics.get('AOSA', metrics.get('A-OSE'))),
            }
        )
    return history


def plot_open_world_percentage_metrics(eval_rows, output_path):
    """绘制 open-world 百分比指标图。"""
    history = open_world_history(eval_rows)
    if not history:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, key, color in [
        ('Current AP50', 'current_ap50', PALETTE['blue']),
        ('Known AP50', 'known_ap50', PALETTE['green']),
        ('Unknown Recall50', 'unknown_recall50', PALETTE['magenta']),
    ]:
        xs = [item['epoch'] for item in history if item[key] is not None]
        ys = [item[key] for item in history if item[key] is not None]
        if xs:
            ax.plot(xs, ys, marker='o', linewidth=2.2, color=color, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Open-World Detection Percentage Metrics')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def plot_open_world_error_metrics(eval_rows, output_path):
    """绘制 open-world 误差指标图。"""
    history = open_world_history(eval_rows)
    if not history:
        return None

    fig, left_ax = plt.subplots(figsize=(10, 6))
    lines = []
    labels = []

    xs = [item['epoch'] for item in history if item['wilderness_impact'] is not None]
    ys = [item['wilderness_impact'] for item in history if item['wilderness_impact'] is not None]
    if xs:
        line = left_ax.plot(xs, ys, marker='o', linewidth=2.2, color=PALETTE['orange'], label='WI@0.8')[0]
        lines.append(line)
        labels.append(line.get_label())

    right_ax = left_ax.twinx()
    xs = [item['epoch'] for item in history if item['absolute_open_set_error'] is not None]
    ys = [item['absolute_open_set_error'] for item in history if item['absolute_open_set_error'] is not None]
    if xs:
        line = right_ax.plot(xs, ys, marker='s', linewidth=2.2, color=PALETTE['red'], label='A-OSE')[0]
        lines.append(line)
        labels.append(line.get_label())

    left_ax.set_xlabel('Epoch')
    left_ax.set_ylabel('Wilderness Impact')
    right_ax.set_ylabel('Absolute Open-Set Error')
    left_ax.set_title('Open-World Error Metrics')
    left_ax.grid(True, alpha=0.25)
    if lines:
        left_ax.legend(lines, labels, frameon=False)
    return save_svg_figure(fig, output_path)


def plot_branch_correlation_trends(eval_rows, output_path):
    """绘制 branch correlation 趋势图。"""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = False
    for key, color in [
        ('corr_fg_obj_unk', PALETTE['blue']),
        ('corr_fg_obj_cls', PALETTE['orange']),
        ('corr_fg_unk_cls', PALETTE['green']),
        ('corr_global_obj_unk', PALETTE['magenta']),
        ('corr_global_obj_cls', PALETTE['cyan']),
        ('corr_global_unk_cls', PALETTE['red']),
    ]:
        xs = []
        ys = []
        for row in eval_rows:
            epoch = row.get('epoch')
            metrics = row.get('open_world_metrics') or row.get('test_metrics') or {}
            value = safe_float(metrics.get(key))
            if epoch is None or value is None:
                continue
            xs.append(int(epoch))
            ys.append(value)
        if xs:
            plotted = True
            ax.plot(xs, ys, marker='o', linewidth=2.0, color=color, label=key)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Branch Correlation Trends')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    return save_svg_figure(fig, output_path)


def plot_query_probability_histograms_by_group(state, output_path):
    """绘制 query 概率分布直方图。"""
    if not state['objectness_probability']:
        return None
    groups = np.asarray(state['query_group'], dtype=np.int64)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    hist_specs = [
        ('objectness_probability', 'Objectness probability', axes[0]),
        ('unknown_probability', 'Unknown probability', axes[1]),
        ('max_known_class_probability', 'Max known-class probability', axes[2]),
    ]
    for field, title, ax in hist_specs:
        values = np.asarray(state[field], dtype=np.float32)
        if values.size == 0:
            ax.set_axis_off()
            continue
        value_min = float(values.min())
        value_max = float(values.max())
        if math.isclose(value_min, value_max):
            value_max = value_min + 1e-3
        bins = np.linspace(value_min, value_max, 36)
        for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
            mask = groups == group_index
            if np.any(mask):
                ax.hist(values[mask], bins=bins, alpha=0.40, label=group_name, color=color, histtype='stepfilled')
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    return save_svg_figure(fig, output_path)


def plot_query_relationship_scatter(state, output_path):
    """绘制 query 关系散点图。"""
    if not state['objectness_probability']:
        return None
    groups = np.asarray(state['query_group'], dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))
    x_objectness = np.asarray(state['objectness_probability'])
    y_unknown = np.asarray(state['unknown_probability'])
    y_known = np.asarray(state['max_known_class_probability'])
    for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
        mask = groups == group_index
        if np.any(mask):
            axes[0].scatter(x_objectness[mask], y_unknown[mask], s=10, alpha=0.55, c=color, label=group_name)
            axes[1].scatter(x_objectness[mask], y_known[mask], s=10, alpha=0.55, c=color, label=group_name)
    axes[0].set_xlabel('objectness probability')
    axes[0].set_ylabel('unknown probability')
    axes[0].set_title('Objectness vs Unknownness')
    axes[1].set_xlabel('objectness probability')
    axes[1].set_ylabel('max known-class probability')
    axes[1].set_title('Objectness vs Max Known-Class Score')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    return save_svg_figure(fig, output_path)


def plot_branch_correlation_heatmap(state, output_path):
    """绘制 branch correlation 热力图。"""
    if len(state['objectness_probability']) < 4:
        return None
    objectness = np.asarray(state['objectness_probability'], dtype=np.float64)
    unknownness = np.asarray(state['unknown_probability'], dtype=np.float64)
    max_known = np.asarray(state['max_known_class_probability'], dtype=np.float64)
    global_corr = np.corrcoef(np.stack([objectness, unknownness, max_known], axis=0))
    foreground_mask = objectness > 0.05
    if foreground_mask.sum() > 4:
        foreground_corr = np.corrcoef(
            np.stack(
                [objectness[foreground_mask], unknownness[foreground_mask], max_known[foreground_mask]],
                axis=0,
            )
        )
    else:
        foreground_corr = np.zeros((3, 3), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    fig.subplots_adjust(right=0.86, wspace=0.35)
    for ax, corr, title in zip(axes, [global_corr, foreground_corr], ['Global', 'Foreground only']):
        heatmap = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['objectness', 'unknown', 'max_known'])
        ax.set_yticklabels(['objectness', 'unknown', 'max_known'])
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                text_color = 'black' if abs(corr[i, j]) > 0.45 else 'white'
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color=text_color)
    color_ax = fig.add_axes([0.88, 0.17, 0.02, 0.68])
    fig.colorbar(heatmap, cax=color_ax)
    return save_svg_figure(fig, output_path)


def plot_layer_prediction_summary(state, output_path):
    """绘制 layer 预测统计曲线。"""
    if not state['layer_debug']:
        return None
    per_layer_objectness = state['layer_debug'].get('layer_objectness_probability_mean', [])
    per_layer_knownness = state['layer_debug'].get('layer_knownness_probability_mean', [])
    per_layer_unknown = state['layer_debug'].get('layer_unknown_probability_mean', [])
    per_layer_clsmax = state['layer_debug'].get('layer_max_known_class_probability_mean', [])
    if not per_layer_objectness:
        return None

    layers = list(range(len(per_layer_objectness)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(layers, per_layer_objectness, marker='o', linewidth=2.0, color=PALETTE['green'], label='objectness prob')
    if per_layer_knownness:
        ax.plot(layers, per_layer_knownness, marker='o', linewidth=2.0, color=PALETTE['blue'], label='knownness prob')
    if per_layer_unknown:
        ax.plot(layers, per_layer_unknown, marker='o', linewidth=2.0, color=PALETTE['magenta'], label='unknown prob')
    if per_layer_clsmax:
        ax.plot(layers, per_layer_clsmax, marker='o', linewidth=2.0, color=PALETTE['gray'], label='max known prob')
    ax.set_xlabel('Decoder layer')
    ax.set_ylabel('Mean value')
    ax.set_title('Layer-wise Prediction Statistics')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return save_svg_figure(fig, output_path)


def compute_branch_correlation_metrics(state):
    """计算 branch correlation 指标。"""
    if len(state['objectness_probability']) < 4:
        return {}
    objectness = np.asarray(state['objectness_probability'], dtype=np.float64)
    unknownness = np.asarray(state['unknown_probability'], dtype=np.float64)
    max_known = np.asarray(state['max_known_class_probability'], dtype=np.float64)
    global_corr = np.corrcoef(np.stack([objectness, unknownness, max_known], axis=0))
    result = {
        'corr_global_obj_unk': float(global_corr[0, 1]),
        'corr_global_obj_cls': float(global_corr[0, 2]),
        'corr_global_unk_cls': float(global_corr[1, 2]),
    }
    foreground_mask = objectness > 0.05
    if foreground_mask.sum() > 4:
        foreground_corr = np.corrcoef(
            np.stack(
                [objectness[foreground_mask], unknownness[foreground_mask], max_known[foreground_mask]],
                axis=0,
            )
        )
        result['corr_fg_obj_unk'] = float(foreground_corr[0, 1])
        result['corr_fg_obj_cls'] = float(foreground_corr[0, 2])
        result['corr_fg_unk_cls'] = float(foreground_corr[1, 2])
    else:
        result['corr_fg_obj_unk'] = None
        result['corr_fg_obj_cls'] = None
        result['corr_fg_unk_cls'] = None
    return result
