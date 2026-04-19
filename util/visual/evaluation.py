import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.prob_deformable_detr_uod import _compute_uod_fused_probabilities
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

QUERY_METADATA_KEYS = [
    'is_matched',
    'matched_gt_label',
    'matched_gt_is_unknown',
    'pred_top1_label',
    'pred_top1_is_unknown',
    'top1_known_class',
    'image_id',
    'query_index',
]
FEATURE_METADATA_KEYS = [f'feature_{key}' for key in QUERY_METADATA_KEYS]


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


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _append_limited(destination, values, max_length):
    remaining = max_length - len(destination)
    if remaining <= 0:
        return 0
    if len(values) > remaining:
        values = values[:remaining]
    destination.extend(values)
    return len(values)


def _ensure_state_lists(state):
    for key in QUERY_METADATA_KEYS:
        state.setdefault(key, [])
    for key in FEATURE_METADATA_KEYS:
        state.setdefault(key, [])


def build_postprocess_aligned_probabilities(outputs, criterion, args):
    """按 postprocess 口径构建 query 概率。"""
    pred_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits')
    pred_obj = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
    if pred_logits is None or pred_obj is None:
        return None

    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temperature = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temperature = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim
    unknown_scale = float(getattr(args, 'uod_postprocess_unknown_scale', 15.0))
    invalid_class_indices = list(getattr(criterion, 'invalid_cls_logits', []))

    return _compute_uod_fused_probabilities(
        pred_logits=pred_logits,
        pred_obj=pred_obj,
        pred_known=_get_output(outputs, 'pred_knownness_energy', 'pred_known'),
        invalid_cls_logits=invalid_class_indices,
        obj_temperature=obj_temperature,
        known_temperature=known_temperature,
        unknown_scale=unknown_scale,
    )


def compute_train_query_score_statistics(outputs, targets, criterion, args):
    """计算训练阶段 query score 统计。"""
    fused = build_postprocess_aligned_probabilities(outputs, criterion, args)
    if fused is None:
        return {}

    stats = {
        'train/query_stats/unknown_probability_mean': safe_float(fused['unknown_prob'].mean()),
        'train/query_stats/max_known_class_probability_mean': safe_float(fused['max_known_cls_prob'].mean()),
    }

    matcher_outputs = {
        'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'),
        'pred_boxes': outputs['pred_boxes'],
    }
    try:
        matched_indices = criterion.matcher(matcher_outputs, targets)
    except Exception:
        return stats

    batch_size, num_queries = fused['obj_prob'].shape[:2]
    matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=fused['obj_prob'].device)
    for batch_index, (source_indices, _) in enumerate(matched_indices):
        if len(source_indices) > 0:
            matched_mask[batch_index, source_indices] = True

    if matched_mask.any():
        stats['train/query_stats/matched_objectness_prob_mean'] = safe_float(fused['obj_prob'][matched_mask].mean())
    if (~matched_mask).any():
        stats['train/query_stats/unmatched_objectness_prob_mean'] = safe_float(fused['obj_prob'][~matched_mask].mean())
    return stats


def _build_query_metadata(outputs, targets, matched_indices, fused, args):
    unknown_label = int(getattr(args, 'num_classes', 81) - 1)
    batch_size, num_queries = fused['obj_prob'].shape[:2]
    device = fused['obj_prob'].device

    matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=device)
    matched_gt_label = torch.full((batch_size, num_queries), -1, dtype=torch.int64, device=device)
    for batch_index, (source_indices, target_indices) in enumerate(matched_indices):
        if len(source_indices) == 0:
            continue
        matched_mask[batch_index, source_indices] = True
        matched_gt_label[batch_index, source_indices] = targets[batch_index]['labels'][target_indices].to(torch.int64)

    fused_prob = fused['fused_prob']
    pred_top1_label = fused_prob.argmax(dim=-1).to(torch.int64)
    pred_top1_is_unknown = pred_top1_label == int(unknown_label)

    if fused['known_scores'].shape[-1] > 1:
        top1_known_class = fused['known_scores'][..., :-1].argmax(dim=-1).to(torch.int64)
    elif fused['known_scores'].shape[-1] == 1:
        top1_known_class = torch.zeros((batch_size, num_queries), dtype=torch.int64, device=device)
    else:
        top1_known_class = torch.full((batch_size, num_queries), -1, dtype=torch.int64, device=device)

    image_ids = []
    query_indices = []
    for batch_index in range(batch_size):
        image_id = int(targets[batch_index]['image_id'].item()) if 'image_id' in targets[batch_index] else batch_index
        image_ids.append(torch.full((num_queries,), image_id, dtype=torch.int64, device=device))
        query_indices.append(torch.arange(num_queries, dtype=torch.int64, device=device))

    metadata = {
        'is_matched': matched_mask,
        'matched_gt_label': matched_gt_label,
        'matched_gt_is_unknown': matched_gt_label == int(unknown_label),
        'pred_top1_label': pred_top1_label,
        'pred_top1_is_unknown': pred_top1_is_unknown,
        'top1_known_class': top1_known_class,
        'image_id': torch.stack(image_ids, dim=0),
        'query_index': torch.stack(query_indices, dim=0),
    }
    return metadata, matched_mask


def collect_eval_visual_stats_aligned(state, outputs, targets, criterion, args):
    """收集评估可视化需要的 query 和 feature 统计。"""
    _ensure_state_lists(state)
    if len(state['objectness_probability']) >= state['max_query_samples'] and len(state['objectness_features']) >= state['max_feature_samples']:
        return

    fused = build_postprocess_aligned_probabilities(outputs, criterion, args)
    if fused is None:
        return

    matcher_outputs = {
        'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'),
        'pred_boxes': outputs['pred_boxes'],
    }
    matched_indices = criterion.matcher(matcher_outputs, targets)
    metadata_tensors, matched_mask = _build_query_metadata(outputs, targets, matched_indices, fused, args)

    objectness_np = fused['obj_prob'].flatten().detach().cpu().numpy()
    unknown_np = fused['unknown_prob'].flatten().detach().cpu().numpy()
    max_known_np = fused['max_known_cls_prob'].flatten().detach().cpu().numpy()
    matched_np = matched_mask.flatten().detach().cpu().numpy()
    group_np = np.where(matched_np, 0, np.where(unknown_np > 0.5, 1, 2)).astype(np.int64)

    appended = _append_limited(state['objectness_probability'], objectness_np.tolist(), state['max_query_samples'])
    if appended > 0:
        _append_limited(state['unknown_probability'], unknown_np.tolist(), state['max_query_samples'])
        _append_limited(state['max_known_class_probability'], max_known_np.tolist(), state['max_query_samples'])
        _append_limited(state['query_group'], group_np.tolist(), state['max_query_samples'])
        for key in QUERY_METADATA_KEYS:
            flat_values = metadata_tensors[key].flatten().detach().cpu().numpy().tolist()
            _append_limited(state[key], flat_values, state['max_query_samples'])

    objectness_features = _get_output(outputs, 'decoder_objectness_features', 'proj_obj')
    knownness_features = _get_output(outputs, 'decoder_knownness_features', 'proj_known', 'proj_unk')
    classification_features = _get_output(outputs, 'decoder_classification_features', 'proj_cls')
    if objectness_features is not None and knownness_features is not None and classification_features is not None:
        obj_feat = objectness_features.detach().flatten(0, 1).cpu().numpy()
        known_feat = knownness_features.detach().flatten(0, 1).cpu().numpy()
        cls_feat = classification_features.detach().flatten(0, 1).cpu().numpy()
        feature_groups = group_np
        feature_metadata = {f'feature_{key}': metadata_tensors[key].flatten().detach().cpu().numpy().tolist() for key in QUERY_METADATA_KEYS}

        remaining = state['max_feature_samples'] - len(state['objectness_features'])
        if remaining > 0:
            if obj_feat.shape[0] > remaining:
                obj_feat = obj_feat[:remaining]
                known_feat = known_feat[:remaining]
                cls_feat = cls_feat[:remaining]
                feature_groups = feature_groups[:remaining]
                for key in FEATURE_METADATA_KEYS:
                    feature_metadata[key] = feature_metadata[key][:remaining]
            state['objectness_features'].extend(list(obj_feat))
            state['knownness_features'].extend(list(known_feat))
            state['classification_features'].extend(list(cls_feat))
            state['feature_groups'].extend(feature_groups.tolist())
            for key in FEATURE_METADATA_KEYS:
                state[key].extend(feature_metadata[key])

    vis_debug = outputs.get('vis_debug', None)
    if vis_debug is not None:
        layer_objectness_probability = vis_debug.get('layer_objectness_probability', vis_debug.get('layer_obj_prob', None))
        layer_knownness_probability = vis_debug.get('layer_knownness_probability', vis_debug.get('layer_knownness_prob', None))
        layer_unknown_probability = vis_debug.get('layer_unknown_probability', vis_debug.get('layer_unknown_prob', None))
        layer_max_known_class_probability = vis_debug.get('layer_max_known_class_probability', vis_debug.get('layer_cls_max', None))
        layer_count = 0
        if layer_objectness_probability is not None:
            layer_objectness_probability = layer_objectness_probability.detach().mean(dim=(1, 2)).cpu().numpy()
            layer_count = 1
            if state['layer_debug']['layer_objectness_probability_sum'] is None:
                state['layer_debug']['layer_objectness_probability_sum'] = np.zeros_like(layer_objectness_probability, dtype=np.float64)
            state['layer_debug']['layer_objectness_probability_sum'] += layer_objectness_probability
        if layer_knownness_probability is not None:
            layer_knownness_probability = layer_knownness_probability.detach().mean(dim=(1, 2)).cpu().numpy()
            if state['layer_debug']['layer_knownness_probability_sum'] is None:
                state['layer_debug']['layer_knownness_probability_sum'] = np.zeros_like(layer_knownness_probability, dtype=np.float64)
            state['layer_debug']['layer_knownness_probability_sum'] += layer_knownness_probability
        if layer_unknown_probability is not None:
            layer_unknown_probability = layer_unknown_probability.detach().mean(dim=(1, 2)).cpu().numpy()
            if state['layer_debug']['layer_unknown_probability_sum'] is None:
                state['layer_debug']['layer_unknown_probability_sum'] = np.zeros_like(layer_unknown_probability, dtype=np.float64)
            state['layer_debug']['layer_unknown_probability_sum'] += layer_unknown_probability
        if layer_max_known_class_probability is not None:
            layer_max_known_class_probability = layer_max_known_class_probability.detach().mean(dim=(1, 2)).cpu().numpy()
            if state['layer_debug']['layer_max_known_class_probability_sum'] is None:
                state['layer_debug']['layer_max_known_class_probability_sum'] = np.zeros_like(layer_max_known_class_probability, dtype=np.float64)
            state['layer_debug']['layer_max_known_class_probability_sum'] += layer_max_known_class_probability
        state['layer_debug']['count'] += layer_count


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
        foreground_corr = np.corrcoef(np.stack([objectness[foreground_mask], unknownness[foreground_mask], max_known[foreground_mask]], axis=0))
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
        foreground_corr = np.corrcoef(np.stack([objectness[foreground_mask], unknownness[foreground_mask], max_known[foreground_mask]], axis=0))
        result['corr_fg_obj_unk'] = float(foreground_corr[0, 1])
        result['corr_fg_obj_cls'] = float(foreground_corr[0, 2])
        result['corr_fg_unk_cls'] = float(foreground_corr[1, 2])
    else:
        result['corr_fg_obj_unk'] = None
        result['corr_fg_obj_cls'] = None
        result['corr_fg_unk_cls'] = None
    return result
