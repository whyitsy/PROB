import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util import box_ops

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COLOR = {
    'prediction_known': '#00A65A',
    'prediction_unknown': '#D81B60',
    'ground_truth_known': '#00BCD4',
    'ground_truth_unknown': '#F39C12',
    'pseudo_positive_candidate': '#1E88E5',
    'pseudo_positive_selected': '#1565C0',
    'reliable_background_selected': '#8E24AA',
    'ignored_query': '#B0BEC5',
    'matched_known': '#00A65A',
    'high_unknown_unmatched': '#D81B60',
    'other_unmatched': '#6C757D',
}


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _to_numpy_image(image_tensor, target_hw=None):
    image = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    if target_hw is not None:
        height, width = int(target_hw[0]), int(target_hw[1])
        image = image[:height, :width]
    return (image * 255).astype(np.uint8)


def _cxcywh_to_abs_xyxy(boxes, image_hw):
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    if torch.is_tensor(boxes):
        if boxes.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().cpu())
        scale = torch.tensor([int(image_hw[1]), int(image_hw[0]), int(image_hw[1]), int(image_hw[0])], dtype=boxes_xyxy.dtype)
        return (boxes_xyxy * scale).numpy()
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    scale = np.asarray([int(image_hw[1]), int(image_hw[0]), int(image_hw[1]), int(image_hw[0])], dtype=np.float32)
    return box_ops.box_cxcywh_to_xyxy(torch.from_numpy(boxes)).numpy() * scale


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _get_font(image_np, font_scale, min_size):
    font_size = max(min_size, int(max(image_np.shape[0], image_np.shape[1]) * font_scale))
    try:
        return ImageFont.truetype('DejaVuSans.ttf', font_size)
    except Exception:
        return ImageFont.load_default()


def _compute_line_width(image_np, viz_cfg):
    return max(viz_cfg['min_line_width'], int(max(image_np.shape[0], image_np.shape[1]) * viz_cfg['line_width_scale']))


def _draw_text_with_background(draw, xy, text, font, fill, background_fill=(20, 20, 20)):
    bbox = draw.textbbox(xy, text, font=font)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=background_fill)
    draw.text(xy, text, font=font, fill=fill)


def _prediction_text(label, score, unknown_label):
    if int(label) == int(unknown_label):
        return f'U {score:.2f}' if score is not None else 'U'
    return f'K[{int(label)}] {score:.2f}' if score is not None else f'K[{int(label)}]'


def _ground_truth_text(label, unknown_label):
    if int(label) == int(unknown_label):
        return 'GT-U'
    return f'GT-K[{int(label)}]'


def _legend_items():
    return [
        ('Pred Known', COLOR['prediction_known']),
        ('Pred Unknown', COLOR['prediction_unknown']),
        ('GT Known', COLOR['ground_truth_known']),
        ('GT Unknown', COLOR['ground_truth_unknown']),
        ('Pseudo Pos', COLOR['pseudo_positive_selected']),
        ('Reliable BG', COLOR['reliable_background_selected']),
    ]


def _draw_legend(draw, image_np, viz_cfg, anchor_xy):
    font = _get_font(image_np, viz_cfg['legend_font_size_scale'], viz_cfg['min_font_size'])
    items = _legend_items()
    box_size = max(12, int(max(image_np.shape[0], image_np.shape[1]) * 0.014))
    x0, y0 = anchor_xy
    for index, (label, color_hex) in enumerate(items):
        color = _hex_to_rgb(color_hex)
        y = y0 + index * (box_size + 8)
        draw.rectangle([x0, y, x0 + box_size, y + box_size], outline=color, width=2, fill=(30, 30, 30))
        _draw_text_with_background(draw, (x0 + box_size + 8, y - 2), label, font, (255, 255, 255))


def _compose_canvas_with_sidebar(image_np, viz_cfg, title=None, info_lines=None, show_legend=False):
    h, w = image_np.shape[:2]
    sidebar_width = max(240, int(w * 0.34))
    footer_height = max(96, int(h * 0.14))
    canvas = Image.new('RGB', (w + sidebar_width, h + footer_height), (18, 18, 22))
    canvas.paste(Image.fromarray(image_np).convert('RGB'), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(np.asarray(canvas), viz_cfg['font_size_scale'], viz_cfg['min_font_size'])
    title_font = _get_font(np.asarray(canvas), max(viz_cfg['font_size_scale'] * 1.05, 0.03), viz_cfg['min_font_size'] + 2)

    right_x = w + 12
    top_y = 10
    if title:
        _draw_text_with_background(draw, (right_x, top_y), title, title_font, (255, 255, 255))
        top_y += 28

    if info_lines:
        for line in info_lines:
            _draw_text_with_background(draw, (right_x, top_y), str(line), font, (230, 230, 230))
            top_y += 22

    if show_legend:
        legend_y = max(top_y + 8, 112)
        _draw_text_with_background(draw, (right_x, legend_y - 24), 'Legend', title_font, (255, 255, 255))
        _draw_legend(draw, np.asarray(canvas), viz_cfg, (right_x, legend_y))

    return np.array(canvas)


def _draw_boxes(
    image_np,
    viz_cfg,
    prediction_boxes=None,
    prediction_labels=None,
    prediction_scores=None,
    ground_truth_boxes=None,
    ground_truth_labels=None,
    title=None,
    info_lines=None,
    unknown_label=80,
    show_legend=False,
):
    image = Image.fromarray(image_np).convert('RGB')
    draw = ImageDraw.Draw(image)
    box_width = _compute_line_width(image_np, viz_cfg)
    font = _get_font(image_np, viz_cfg['font_size_scale'], viz_cfg['min_font_size'])

    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        for index, box in enumerate(ground_truth_boxes):
            x1, y1, x2, y2 = [float(value) for value in box]
            label = int(ground_truth_labels[index]) if ground_truth_labels is not None else -1
            color = _hex_to_rgb(COLOR['ground_truth_unknown'] if label == int(unknown_label) else COLOR['ground_truth_known'])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
            _draw_text_with_background(draw, (x1 + 2, max(0, y1 - 18)), _ground_truth_text(label, unknown_label), font, color)

    if prediction_boxes is not None and len(prediction_boxes) > 0:
        for index, box in enumerate(prediction_boxes):
            x1, y1, x2, y2 = [float(value) for value in box]
            label = int(prediction_labels[index]) if prediction_labels is not None else -1
            score = float(prediction_scores[index]) if prediction_scores is not None else None
            color = _hex_to_rgb(COLOR['prediction_unknown'] if label == int(unknown_label) else COLOR['prediction_known'])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
            _draw_text_with_background(draw, (x1 + 2, y1 + 2), _prediction_text(label, score, unknown_label), font, color)

    return _compose_canvas_with_sidebar(
        np.array(image),
        viz_cfg=viz_cfg,
        title=title,
        info_lines=info_lines,
        show_legend=show_legend,
    )


def _draw_stage_boxes(image_np, viz_cfg, stage_boxes, title, color_hex, stage_texts=None, info_lines=None):
    image = Image.fromarray(image_np).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = _get_font(image_np, viz_cfg['font_size_scale'], viz_cfg['min_font_size'])
    box_width = _compute_line_width(image_np, viz_cfg)
    color = _hex_to_rgb(color_hex)
    for index, box in enumerate(stage_boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        if stage_texts is not None and index < len(stage_texts):
            _draw_text_with_background(draw, (x1 + 2, y1 + 2), stage_texts[index], font, color)
    return _compose_canvas_with_sidebar(np.array(image), viz_cfg=viz_cfg, title=title, info_lines=info_lines, show_legend=False)


def _save_image(np_image, output_path):
    Image.fromarray(np_image).save(output_path)


def _save_contact_sheet(image_paths, output_path, viz_cfg):
    if not image_paths:
        return
    tile_width = viz_cfg['panel_tile_width']
    tile_height = viz_cfg['panel_tile_height']
    cols = viz_cfg['panel_cols']
    valid_images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            image = image.resize((tile_width, tile_height))
            valid_images.append(image)
        except Exception:
            continue
    if not valid_images:
        return
    rows = int(math.ceil(len(valid_images) / cols))
    sheet = Image.new('RGB', (cols * tile_width, rows * tile_height), (20, 20, 20))
    for index, image in enumerate(valid_images):
        x = (index % cols) * tile_width
        y = (index // cols) * tile_height
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def _save_figure(figure, output_path, tb_writer=None, tb_tag=None, global_step=0):
    figure.savefig(output_path, bbox_inches='tight')
    if tb_writer is not None and tb_tag is not None:
        try:
            tb_writer.add_figure(tb_tag, figure, global_step=global_step)
        except Exception:
            pass
    plt.close(figure)


def _plot_histograms(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if not state['objectness_probability']:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(state['objectness_probability'], bins=40, color=COLOR['matched_known'])
    axes[0].set_title('Objectness probability')
    axes[1].hist(state['unknown_probability'], bins=40, color=COLOR['high_unknown_unmatched'])
    axes[1].set_title('Unknown probability')
    axes[2].hist(state['max_known_class_probability'], bins=40, color=COLOR['other_unmatched'])
    axes[2].set_title('Max known-class probability')
    for axis in axes:
        axis.grid(alpha=0.2)
    _save_figure(figure, os.path.join(output_dir, f'query_probability_histograms.{viz_cfg["figure_format"]}'), tb_writer, 'eval_viz/query_probability_histograms', global_step)


def _plot_scatter(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if not state['objectness_probability']:
        return
    groups = np.asarray(state['query_group'], dtype=np.int64)
    colors = np.array([COLOR['matched_known'], COLOR['high_unknown_unmatched'], COLOR['other_unmatched']])
    names = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for group_index in range(3):
        mask = groups == group_index
        if np.any(mask):
            axes[0].scatter(np.asarray(state['objectness_probability'])[mask], np.asarray(state['unknown_probability'])[mask], s=10, alpha=0.55, c=colors[group_index], label=names[group_index])
            axes[1].scatter(np.asarray(state['objectness_probability'])[mask], np.asarray(state['max_known_class_probability'])[mask], s=10, alpha=0.55, c=colors[group_index], label=names[group_index])
    axes[0].set_xlabel('objectness probability')
    axes[0].set_ylabel('unknown probability')
    axes[0].set_title('Objectness vs Unknownness')
    axes[1].set_xlabel('objectness probability')
    axes[1].set_ylabel('max known-class probability')
    axes[1].set_title('Objectness vs Max Known-Class Score')
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    _save_figure(figure, os.path.join(output_dir, f'query_relationship_scatter.{viz_cfg["figure_format"]}'), tb_writer, 'eval_viz/query_relationship_scatter', global_step)


def _plot_correlation_heatmap(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if len(state['objectness_probability']) < 4:
        return
    objectness = np.asarray(state['objectness_probability'], dtype=np.float64)
    unknownness = np.asarray(state['unknown_probability'], dtype=np.float64)
    max_known = np.asarray(state['max_known_class_probability'], dtype=np.float64)
    global_corr = np.corrcoef(np.stack([objectness, unknownness, max_known], axis=0))
    foreground_mask = objectness > 0.05
    if foreground_mask.sum() > 4:
        foreground_corr = np.corrcoef(np.stack([objectness[foreground_mask], unknownness[foreground_mask], max_known[foreground_mask]], axis=0))
    else:
        foreground_corr = np.zeros((3, 3), dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    figure.subplots_adjust(right=0.86, wspace=0.35)
    for axis, corr, title in zip(axes, [global_corr, foreground_corr], ['Global', 'Foreground only']):
        heatmap = axis.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        axis.set_xticks(range(3))
        axis.set_yticks(range(3))
        axis.set_xticklabels(['objectness', 'unknown', 'max_known'])
        axis.set_yticklabels(['objectness', 'unknown', 'max_known'])
        axis.set_title(title)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', color='black' if abs(corr[i, j]) > 0.45 else 'white')
    color_axis = figure.add_axes([0.88, 0.17, 0.02, 0.68])
    figure.colorbar(heatmap, cax=color_axis)
    _save_figure(figure, os.path.join(output_dir, f'branch_correlation_heatmap.{viz_cfg["figure_format"]}'), tb_writer, 'eval_viz/branch_correlation_heatmap', global_step)


def _project_2d(features, method='pca'):
    if features.shape[0] < 3:
        return None
    if method == 'tsne':
        perplexity = min(30, max(2, features.shape[0] // 4))
        return TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=42).fit_transform(features)
    return PCA(n_components=2, random_state=42).fit_transform(features)


def _plot_feature_embeddings(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if not state['feature_groups'] or not state['objectness_features']:
        return
    groups = np.asarray(state['feature_groups'], dtype=np.int64)
    names = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
    colors = np.array([COLOR['matched_known'], COLOR['high_unknown_unmatched'], COLOR['other_unmatched']])
    feature_specs = [
        ('objectness_features', state['objectness_features']),
        ('knownness_features', state['knownness_features']),
        ('classification_features', state['classification_features']),
    ]
    for method in ['pca', 'tsne']:
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        plotted_any = False
        for axis, (name, feature_list) in zip(axes, feature_specs):
            if len(feature_list) < 8:
                axis.set_axis_off()
                continue
            features = np.stack(feature_list, axis=0)
            max_points = 800 if method == 'tsne' else 2000
            if features.shape[0] > max_points:
                sampled_index = np.linspace(0, features.shape[0] - 1, max_points).astype(np.int64)
                features = features[sampled_index]
                feature_groups = groups[sampled_index]
            else:
                feature_groups = groups
            try:
                embedding = _project_2d(features, method=method)
            except Exception:
                axis.set_axis_off()
                continue
            if embedding is None:
                axis.set_axis_off()
                continue
            plotted_any = True
            for group_index in range(3):
                mask = feature_groups == group_index
                if np.any(mask):
                    axis.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.6, c=colors[group_index], label=names[group_index])
            axis.set_title(f'{name} {method.upper()}')
            axis.grid(alpha=0.2)
        if plotted_any:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
            _save_figure(figure, os.path.join(output_dir, f'feature_embedding_{method}.{viz_cfg["figure_format"]}'), tb_writer, f'eval_viz/feature_embedding_{method}', global_step)
        else:
            plt.close(figure)


def _plot_layer_debug_summary(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if not state['layer_debug']:
        return
    per_layer_objectness = state['layer_debug'].get('layer_objectness_probability_mean', [])
    per_layer_knownness = state['layer_debug'].get('layer_knownness_probability_mean', [])
    per_layer_unknown = state['layer_debug'].get('layer_unknown_probability_mean', [])
    per_layer_clsmax = state['layer_debug'].get('layer_max_known_class_probability_mean', [])
    if not per_layer_objectness:
        return
    layers = list(range(len(per_layer_objectness)))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(layers, per_layer_objectness, marker='o', linewidth=2.0, color=COLOR['matched_known'], label='objectness prob')
    if per_layer_knownness:
        axis.plot(layers, per_layer_knownness, marker='o', linewidth=2.0, color=COLOR['prediction_known'], label='knownness prob')
    if per_layer_unknown:
        axis.plot(layers, per_layer_unknown, marker='o', linewidth=2.0, color=COLOR['prediction_unknown'], label='unknown prob')
    if per_layer_clsmax:
        axis.plot(layers, per_layer_clsmax, marker='o', linewidth=2.0, color=COLOR['other_unmatched'], label='max known prob')
    axis.set_xlabel('Decoder layer')
    axis.set_ylabel('Mean value')
    axis.set_title('Layer-wise Prediction Statistics')
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save_figure(figure, os.path.join(output_dir, f'layer_prediction_summary.{viz_cfg["figure_format"]}'), tb_writer, 'eval_viz/layer_prediction_summary', global_step)


def compute_branch_correlation_metrics(state):
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


def init_eval_visual_state(viz_cfg):
    return {
        'saved_primary_panels': [],
        'saved_error_panels': [],
        'saved_stage_panels': [],
        'saved_case_count': 0,
        'objectness_probability': [],
        'unknown_probability': [],
        'max_known_class_probability': [],
        'query_group': [],
        'objectness_features': [],
        'knownness_features': [],
        'classification_features': [],
        'feature_groups': [],
        'max_query_samples': viz_cfg['max_query_samples'],
        'max_feature_samples': viz_cfg['max_feature_samples'],
        'error_rows': [],
        'layer_debug': {
            'layer_objectness_probability_sum': None,
            'layer_knownness_probability_sum': None,
            'layer_unknown_probability_sum': None,
            'layer_max_known_class_probability_sum': None,
            'count': 0,
            'layer_objectness_probability_mean': [],
            'layer_knownness_probability_mean': [],
            'layer_unknown_probability_mean': [],
            'layer_max_known_class_probability_mean': [],
        },
    }


def _box_iou_numpy(boxes1, boxes2):
    if boxes1 is None or boxes2 is None or len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def _extract_error_cases(prediction_boxes, prediction_labels, ground_truth_boxes, ground_truth_labels, unknown_label, iou_threshold):
    errors = {
        'unknown_to_known_prediction_indices': [],
        'unknown_to_known_ground_truth_indices': [],
        'known_to_unknown_prediction_indices': [],
        'known_to_unknown_ground_truth_indices': [],
    }
    iou = _box_iou_numpy(prediction_boxes, ground_truth_boxes)
    if iou.size == 0:
        return errors
    for gt_index in range(len(ground_truth_boxes)):
        pred_index = int(np.argmax(iou[:, gt_index]))
        if iou[pred_index, gt_index] < iou_threshold:
            continue
        gt_is_unknown = int(ground_truth_labels[gt_index]) == int(unknown_label)
        pred_is_unknown = int(prediction_labels[pred_index]) == int(unknown_label)
        if gt_is_unknown and not pred_is_unknown:
            errors['unknown_to_known_prediction_indices'].append(pred_index)
            errors['unknown_to_known_ground_truth_indices'].append(gt_index)
        if (not gt_is_unknown) and pred_is_unknown:
            errors['known_to_unknown_prediction_indices'].append(pred_index)
            errors['known_to_unknown_ground_truth_indices'].append(gt_index)
    return errors


def _nms_xyxy_numpy(boxes, scores, iou_threshold=0.5):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        area_i = max(0.0, boxes[i, 2] - boxes[i, 0]) * max(0.0, boxes[i, 3] - boxes[i, 1])
        area_rest = np.maximum(0.0, boxes[order[1:], 2] - boxes[order[1:], 0]) * np.maximum(0.0, boxes[order[1:], 3] - boxes[order[1:], 1])
        union = np.maximum(area_i + area_rest - inter, 1e-6)
        iou = inter / union
        order = order[1:][iou < float(iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def _panel_thresholds(args):
    return {
        'known_score_thresh': float(getattr(args, 'eval_known_panel_score_thresh', 0.35)),
        'unknown_score_thresh': float(getattr(args, 'eval_unknown_panel_score_thresh', 3.0)),
        'nms_iou': float(getattr(args, 'eval_panel_nms_iou', 0.30)),
    }
    

def _filter_prediction_subset(prediction_boxes, prediction_labels, prediction_scores, unknown_label, select_unknown, thresholds):
    if prediction_boxes is None or len(prediction_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    prediction_boxes = np.asarray(prediction_boxes, dtype=np.float32)
    prediction_labels = np.asarray(prediction_labels, dtype=np.int64)
    prediction_scores = np.asarray(prediction_scores, dtype=np.float32)

    subset_mask = prediction_labels == int(unknown_label) if select_unknown else prediction_labels != int(unknown_label)
    score_thresh = thresholds['unknown_score_thresh'] if select_unknown else thresholds['known_score_thresh']
    subset_mask &= prediction_scores >= float(score_thresh)

    boxes = prediction_boxes[subset_mask]
    labels = prediction_labels[subset_mask]
    scores = prediction_scores[subset_mask]
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    keep = _nms_xyxy_numpy(boxes, scores, thresholds['nms_iou'])
    return boxes[keep], labels[keep], scores[keep]


def _build_common_info_lines(image_id, epoch, num_predictions, num_ground_truth_boxes):
    return [
        f'image_id: {image_id}',
        f'epoch: {int(epoch):04d}',
        f'pred(all): {int(num_predictions)}',
        f'gt: {int(num_ground_truth_boxes)}',
    ]


def save_eval_qualitative_cases(state, samples, targets, postprocessed_predictions, outputs, criterion, args, output_dir, viz_cfg, tb_writer=None, global_step=0, epoch=0):
    epoch = max(int(epoch), 0)
    unknown_label = int(getattr(args, 'num_classes', 81) - 1)
    final_dir = os.path.join(output_dir, 'final')
    debug_dir = os.path.join(output_dir, 'debug')
    _ensure_dir(final_dir)
    _ensure_dir(debug_dir)

    mining_debug = None
    if hasattr(criterion, 'generate_pseudo_mining_debug'):
        try:
            mining_debug = criterion.generate_pseudo_mining_debug(outputs, targets, epoch=epoch)
        except Exception:
            mining_debug = None

    thresholds = _panel_thresholds(args)

    for batch_index in range(len(targets)):
        if state['saved_case_count'] >= viz_cfg['max_qualitative_cases']:
            break

        image_hw = targets[batch_index]['size'].tolist()
        image_np = _to_numpy_image(samples.tensors[batch_index], image_hw)
        image_id = int(targets[batch_index]['image_id'].item()) if 'image_id' in targets[batch_index] else state['saved_case_count']
        ground_truth_boxes = _cxcywh_to_abs_xyxy(targets[batch_index]['boxes'], image_hw)
        ground_truth_labels = targets[batch_index]['labels'].detach().cpu().numpy()

        prediction = postprocessed_predictions[batch_index]
        prediction_boxes = prediction['boxes'].detach().cpu().numpy()
        prediction_labels = prediction['labels'].detach().cpu().numpy()
        prediction_scores = prediction['scores'].detach().cpu().numpy()

        case_prefix = f'{image_id:012d}__epoch_{int(epoch):04d}'
        case_dir = os.path.join(final_dir, case_prefix)
        _ensure_dir(case_dir)

        common_info = _build_common_info_lines(
            image_id=image_id,
            epoch=epoch,
            num_predictions=len(prediction_boxes),
            num_ground_truth_boxes=len(ground_truth_boxes),
        )

        known_boxes, known_labels, known_scores = _filter_prediction_subset(
            prediction_boxes, prediction_labels, prediction_scores, unknown_label, select_unknown=False, thresholds=thresholds
        )
        unknown_boxes, unknown_labels, unknown_scores = _filter_prediction_subset(
            prediction_boxes, prediction_labels, prediction_scores, unknown_label, select_unknown=True, thresholds=thresholds
        )

        final_prediction_image = _draw_boxes(
            image_np,
            viz_cfg,
            prediction_boxes=prediction_boxes,
            prediction_labels=prediction_labels,
            prediction_scores=prediction_scores,
            ground_truth_boxes=ground_truth_boxes,
            ground_truth_labels=ground_truth_labels,
            title='Prediction vs Ground Truth',
            info_lines=common_info + ['panel: official postprocess output', 'legend shown once in this panel'],
            unknown_label=unknown_label,
            show_legend=True,
        )
        prediction_known_image = _draw_boxes(
            image_np,
            viz_cfg,
            prediction_boxes=known_boxes,
            prediction_labels=known_labels,
            prediction_scores=known_scores,
            title='Known Predictions',
            info_lines=common_info + [
                f'kept(known): {len(known_boxes)}',
                f'known_thresh: {thresholds["known_score_thresh"]:.2f}',
                f'nms_iou: {thresholds["nms_iou"]:.2f}',
            ],
            unknown_label=unknown_label,
            show_legend=False,
        )
        prediction_unknown_image = _draw_boxes(
            image_np,
            viz_cfg,
            prediction_boxes=unknown_boxes,
            prediction_labels=unknown_labels,
            prediction_scores=unknown_scores,
            title='Unknown Predictions',
            info_lines=common_info + [
                f'kept(unknown): {len(unknown_boxes)}',
                f'unknown_thresh: {thresholds["unknown_score_thresh"]:.2f}',
                f'nms_iou: {thresholds["nms_iou"]:.2f}',
            ],
            unknown_label=unknown_label,
            show_legend=False,
        )
        ground_truth_image = _draw_boxes(
            image_np,
            viz_cfg,
            ground_truth_boxes=ground_truth_boxes,
            ground_truth_labels=ground_truth_labels,
            title='Ground Truth',
            info_lines=common_info + ['panel: transformed-image display space', 'gt boxes are shown without prediction filtering'],
            unknown_label=unknown_label,
            show_legend=False,
        )

        ground_truth_path = os.path.join(case_dir, 'ground_truth.png')
        prediction_path = os.path.join(case_dir, 'prediction_vs_gt.png')
        known_path = os.path.join(case_dir, 'known_predictions.png')
        unknown_path = os.path.join(case_dir, 'unknown_predictions.png')
        _save_image(ground_truth_image, ground_truth_path)
        _save_image(final_prediction_image, prediction_path)
        _save_image(prediction_known_image, known_path)
        _save_image(prediction_unknown_image, unknown_path)
        state['saved_primary_panels'].append(prediction_path)

        if viz_cfg['save_error_panel']:
            errors = _extract_error_cases(prediction_boxes, prediction_labels, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg['error_match_iou'])
            u2k_pred = np.asarray(sorted(set(errors['unknown_to_known_prediction_indices'])), dtype=np.int64)
            u2k_gt = np.asarray(sorted(set(errors['unknown_to_known_ground_truth_indices'])), dtype=np.int64)
            k2u_pred = np.asarray(sorted(set(errors['known_to_unknown_prediction_indices'])), dtype=np.int64)
            k2u_gt = np.asarray(sorted(set(errors['known_to_unknown_ground_truth_indices'])), dtype=np.int64)

            unknown_to_known_image = _draw_boxes(
                image_np,
                viz_cfg,
                prediction_boxes=prediction_boxes[u2k_pred] if len(u2k_pred) > 0 else None,
                prediction_labels=prediction_labels[u2k_pred] if len(u2k_pred) > 0 else None,
                prediction_scores=prediction_scores[u2k_pred] if len(u2k_pred) > 0 else None,
                ground_truth_boxes=ground_truth_boxes[u2k_gt] if len(u2k_gt) > 0 else None,
                ground_truth_labels=ground_truth_labels[u2k_gt] if len(u2k_gt) > 0 else None,
                title='Error: Unknown -> Known',
                info_lines=common_info + [f'error_pairs: {len(u2k_gt)}'],
                unknown_label=unknown_label,
                show_legend=False,
            )
            known_to_unknown_image = _draw_boxes(
                image_np,
                viz_cfg,
                prediction_boxes=prediction_boxes[k2u_pred] if len(k2u_pred) > 0 else None,
                prediction_labels=prediction_labels[k2u_pred] if len(k2u_pred) > 0 else None,
                prediction_scores=prediction_scores[k2u_pred] if len(k2u_pred) > 0 else None,
                ground_truth_boxes=ground_truth_boxes[k2u_gt] if len(k2u_gt) > 0 else None,
                ground_truth_labels=ground_truth_labels[k2u_gt] if len(k2u_gt) > 0 else None,
                title='Error: Known -> Unknown',
                info_lines=common_info + [f'error_pairs: {len(k2u_gt)}'],
                unknown_label=unknown_label,
                show_legend=False,
            )

            u2k_path = os.path.join(case_dir, 'error_unknown_to_known.png')
            k2u_path = os.path.join(case_dir, 'error_known_to_unknown.png')
            _save_image(unknown_to_known_image, u2k_path)
            _save_image(known_to_unknown_image, k2u_path)

            if len(u2k_gt) > 0:
                state['saved_error_panels'].append(u2k_path)
            elif len(k2u_gt) > 0:
                state['saved_error_panels'].append(k2u_path)

            state['error_rows'].append({
                'image_id': image_id,
                'num_predictions': int(len(prediction_boxes)),
                'num_ground_truth_boxes': int(len(ground_truth_boxes)),
                'num_unknown_to_known_errors': int(len(u2k_gt)),
                'num_known_to_unknown_errors': int(len(k2u_gt)),
            })

        if viz_cfg['save_mining_stage_panel'] and mining_debug is not None and batch_index < len(mining_debug):
            debug_case_dir = os.path.join(debug_dir, case_prefix)
            _ensure_dir(debug_case_dir)
            debug_item = mining_debug[batch_index]
            stage_specs = [
                ('final_prediction.png', final_prediction_image),
                ('after_gt_overlap_filter.png', _draw_stage_boxes(
                    image_np, viz_cfg, debug_item.get('after_gt_overlap_filter_boxes', []),
                    'After GT-overlap filter', COLOR['pseudo_positive_candidate'],
                    info_lines=common_info,
                )),
                ('after_geometry_filter.png', _draw_stage_boxes(
                    image_np, viz_cfg, debug_item.get('after_geometry_filter_boxes', []),
                    'After geometry filter', COLOR['pseudo_positive_candidate'],
                    info_lines=common_info,
                )),
                ('pseudo_positive_candidates.png', _draw_stage_boxes(
                    image_np, viz_cfg, debug_item.get('candidate_boxes_before_selection', []),
                    'Pseudo-positive candidates', COLOR['pseudo_positive_candidate'],
                    stage_texts=debug_item.get('candidate_score_texts'),
                    info_lines=common_info,
                )),
                ('selected_pseudo_positives.png', _draw_stage_boxes(
                    image_np, viz_cfg, debug_item.get('selected_pseudo_positive_boxes', []),
                    'Selected pseudo positives', COLOR['pseudo_positive_selected'],
                    info_lines=common_info,
                )),
                ('reliable_background_queries.png', _draw_stage_boxes(
                    image_np, viz_cfg, debug_item.get('selected_reliable_background_boxes', []),
                    'Reliable background queries', COLOR['reliable_background_selected'],
                    info_lines=common_info,
                )),
            ]
            for filename, stage_image in stage_specs:
                stage_path = os.path.join(debug_case_dir, filename)
                _save_image(stage_image, stage_path)
            state['saved_stage_panels'].append(os.path.join(debug_case_dir, 'final_prediction.png'))

        if tb_writer is not None and state['saved_case_count'] < viz_cfg['max_tensorboard_cases']:
            tb_writer.add_image(f'eval_qualitative/{image_id:012d}_prediction_vs_gt', final_prediction_image, global_step=global_step, dataformats='HWC')
            tb_writer.add_image(f'eval_qualitative/{image_id:012d}_ground_truth', ground_truth_image, global_step=global_step, dataformats='HWC')

        state['saved_case_count'] += 1


def finalize_eval_visualizations(state, output_dir, epoch, viz_cfg, tb_writer=None):
    epoch = max(int(epoch), 0)
    output_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{int(epoch):04d}')
    stats_dir = os.path.join(output_dir, 'stats')
    final_dir = os.path.join(output_dir, 'final')
    debug_dir = os.path.join(output_dir, 'debug')
    _ensure_dir(stats_dir)
    _ensure_dir(final_dir)
    _ensure_dir(debug_dir)

    layer_count = state['layer_debug']['count']
    if layer_count > 0:
        if state['layer_debug']['layer_objectness_probability_sum'] is not None:
            state['layer_debug']['layer_objectness_probability_mean'] = (state['layer_debug']['layer_objectness_probability_sum'] / layer_count).tolist()
        if state['layer_debug']['layer_knownness_probability_sum'] is not None:
            state['layer_debug']['layer_knownness_probability_mean'] = (state['layer_debug']['layer_knownness_probability_sum'] / layer_count).tolist()
        if state['layer_debug']['layer_unknown_probability_sum'] is not None:
            state['layer_debug']['layer_unknown_probability_mean'] = (state['layer_debug']['layer_unknown_probability_sum'] / layer_count).tolist()
        if state['layer_debug']['layer_max_known_class_probability_sum'] is not None:
            state['layer_debug']['layer_max_known_class_probability_mean'] = (state['layer_debug']['layer_max_known_class_probability_sum'] / layer_count).tolist()

    if viz_cfg['save_query_stats_csv'] and state['objectness_probability']:
        with open(os.path.join(stats_dir, 'query_statistics.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['objectness_probability', 'unknown_probability', 'max_known_class_probability', 'query_group'])
            for row in zip(state['objectness_probability'], state['unknown_probability'], state['max_known_class_probability'], state['query_group']):
                writer.writerow(row)

    if viz_cfg['save_feature_npz'] and state['objectness_features']:
        np.savez_compressed(
            os.path.join(stats_dir, 'feature_samples.npz'),
            objectness_features=np.asarray(state['objectness_features'], dtype=np.float32),
            knownness_features=np.asarray(state['knownness_features'], dtype=np.float32),
            classification_features=np.asarray(state['classification_features'], dtype=np.float32),
            feature_groups=np.asarray(state['feature_groups'], dtype=np.int64),
        )

    if viz_cfg['save_error_summary_csv'] and state['error_rows']:
        with open(os.path.join(stats_dir, 'error_case_summary.csv'), 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['image_id', 'num_predictions', 'num_ground_truth_boxes', 'num_unknown_to_known_errors', 'num_known_to_unknown_errors'])
            writer.writeheader()
            for row in state['error_rows']:
                writer.writerow(row)

    if viz_cfg['save_query_distribution_plots']:
        _plot_histograms(state, stats_dir, viz_cfg, tb_writer, epoch)
        _plot_scatter(state, stats_dir, viz_cfg, tb_writer, epoch)
        _plot_correlation_heatmap(state, stats_dir, viz_cfg, tb_writer, epoch)
        _plot_layer_debug_summary(state, stats_dir, viz_cfg, tb_writer, epoch)

    if viz_cfg['save_feature_embedding_plots']:
        _plot_feature_embeddings(state, stats_dir, viz_cfg, tb_writer, epoch)

    if viz_cfg['save_contact_sheet']:
        _save_contact_sheet(state['saved_primary_panels'], os.path.join(final_dir, 'prediction_vs_gt_contact_sheet.png'), viz_cfg)
        _save_contact_sheet(state['saved_error_panels'], os.path.join(final_dir, 'error_cases_contact_sheet.png'), viz_cfg)
        _save_contact_sheet(state['saved_stage_panels'], os.path.join(debug_dir, 'mining_stage_panels_contact_sheet.png'), viz_cfg)
