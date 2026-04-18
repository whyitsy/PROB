import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from util import box_ops
from util.visual.cases import (
    filter_prediction_display,
    render_error_known_to_unknown,
    render_error_unknown_to_known,
    render_ground_truth,
    render_known_predictions,
    render_prediction_vs_gt,
    render_unknown_predictions,
    save_contact_sheet,
)
from util.visual.embeddings import plot_feature_embedding_views, plot_score_space_embedding_views
from util.visual.evaluation import (
    compute_branch_correlation_metrics,
    plot_branch_correlation_heatmap,
    plot_layer_prediction_summary,
    plot_query_probability_histograms_by_group,
    plot_query_relationship_scatter,
)
from util.visual.helper import cxcywh_to_abs_xyxy, ensure_parent, save_image, to_numpy_image


COLOR = {
    'pseudo_positive_candidate': '#1E88E5',
    'pseudo_positive_selected': '#1565C0',
    'reliable_background_selected': '#8E24AA',
}


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _get_font(image_np, font_scale, min_size):
    font_size = max(min_size, int(max(image_np.shape[0], image_np.shape[1]) * font_scale))
    try:
        return ImageFont.truetype('DejaVuSans.ttf', font_size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_with_background(draw, xy, text, font, fill, background_fill=(20, 20, 20)):
    bbox = draw.textbbox(xy, text, font=font)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=background_fill)
    draw.text(xy, text, font=font, fill=fill)


def _draw_stage_boxes(image_np, viz_cfg, stage_boxes, title, color_rgb, stage_texts=None):
    image = Image.fromarray(image_np).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = _get_font(image_np, viz_cfg['font_size_scale'], viz_cfg['min_font_size'])
    box_width = max(viz_cfg['min_line_width'], int(max(image_np.shape[0], image_np.shape[1]) * viz_cfg['line_width_scale']))
    for index, box in enumerate(stage_boxes):
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=box_width)
        if stage_texts is not None and index < len(stage_texts):
            _draw_text_with_background(draw, (x1 + 2, y1 + 2), stage_texts[index], font, color_rgb)
    _draw_text_with_background(draw, (8, 6), title, font, (255, 255, 255))
    return np.array(image)


def _save_panel(images_with_titles, output_path, viz_cfg):
    tile_width = viz_cfg['panel_tile_width']
    tile_height = viz_cfg['panel_tile_height']
    cols = viz_cfg['panel_cols']
    images = []
    for image_np, title in images_with_titles:
        image = Image.fromarray(image_np).convert('RGB').resize((tile_width, tile_height))
        canvas = Image.new('RGB', (tile_width, tile_height), (20, 20, 20))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = _get_font(np.asarray(canvas), viz_cfg['font_size_scale'], viz_cfg['min_font_size'])
        _draw_text_with_background(draw, (8, 8), title, font, (255, 255, 255))
        images.append(canvas)
    rows = int(math.ceil(len(images) / cols))
    sheet = Image.new('RGB', (cols * tile_width, rows * tile_height), (15, 15, 15))
    for index, image in enumerate(images):
        x = (index % cols) * tile_width
        y = (index // cols) * tile_height
        sheet.paste(image, (x, y))
    ensure_parent(output_path)
    sheet.save(output_path)


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
        'is_matched': [],
        'matched_gt_is_unknown': [],
        'objectness_features': [],
        'knownness_features': [],
        'classification_features': [],
        'feature_groups': [],
        'feature_is_matched': [],
        'feature_matched_gt_is_unknown': [],
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


def _append_limited(destination, values, max_length):
    remaining = max_length - len(destination)
    if remaining <= 0:
        return
    if len(values) > remaining:
        values = values[:remaining]
    destination.extend(values)


def collect_eval_visual_stats(state, outputs, targets, criterion, args):
    if len(state['objectness_probability']) >= state['max_query_samples'] and len(state['objectness_features']) >= state['max_feature_samples']:
        return

    objectness_energy = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
    class_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits')
    if objectness_energy is None or class_logits is None:
        return

    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    objectness_temperature = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    objectness_probability = torch.exp(-objectness_temperature * objectness_energy.detach())

    knownness_energy = _get_output(outputs, 'pred_knownness_energy', 'pred_known')
    unknown_logit = _get_output(outputs, 'pred_unknown_logit', 'pred_unk')
    if knownness_energy is not None:
        knownness_temperature = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim
        knownness_probability = torch.exp(-knownness_temperature * knownness_energy.detach())
        unknown_probability = (1.0 - knownness_probability).clamp(min=0.0, max=1.0)
    elif unknown_logit is not None:
        unknown_probability = torch.sigmoid(unknown_logit.detach())
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
    matched_indices = criterion.matcher(matcher_outputs, targets)
    matched_mask = torch.zeros_like(objectness_probability, dtype=torch.bool)
    matched_gt_is_unknown = torch.zeros_like(objectness_probability, dtype=torch.bool)
    unknown_label = int(getattr(args, 'num_classes', 81) - 1)
    for batch_index, (source_indices, target_indices) in enumerate(matched_indices):
        if len(source_indices) == 0:
            continue
        matched_mask[batch_index, source_indices] = True
        gt_labels = targets[batch_index]['labels'][target_indices]
        matched_gt_is_unknown[batch_index, source_indices] = gt_labels == unknown_label

    objectness_np = objectness_probability.flatten().cpu().numpy()
    unknown_np = unknown_probability.flatten().cpu().numpy()
    max_known_np = max_known_class_probability.flatten().cpu().numpy()
    matched_np = matched_mask.flatten().cpu().numpy()
    matched_gt_unknown_np = matched_gt_is_unknown.flatten().cpu().numpy()
    group_np = np.where(matched_np, 0, np.where(unknown_np > 0.5, 1, 2)).astype(np.int64)

    _append_limited(state['objectness_probability'], objectness_np.tolist(), state['max_query_samples'])
    _append_limited(state['unknown_probability'], unknown_np.tolist(), state['max_query_samples'])
    _append_limited(state['max_known_class_probability'], max_known_np.tolist(), state['max_query_samples'])
    _append_limited(state['query_group'], group_np.tolist(), state['max_query_samples'])
    _append_limited(state['is_matched'], matched_np.astype(np.int64).tolist(), state['max_query_samples'])
    _append_limited(state['matched_gt_is_unknown'], matched_gt_unknown_np.astype(np.int64).tolist(), state['max_query_samples'])

    objectness_features = _get_output(outputs, 'decoder_objectness_features', 'proj_obj')
    knownness_features = _get_output(outputs, 'decoder_knownness_features', 'proj_known', 'proj_unk')
    classification_features = _get_output(outputs, 'decoder_classification_features', 'proj_cls')
    if objectness_features is not None and knownness_features is not None and classification_features is not None:
        obj_feat = objectness_features.detach().flatten(0, 1).cpu().numpy()
        known_feat = knownness_features.detach().flatten(0, 1).cpu().numpy()
        cls_feat = classification_features.detach().flatten(0, 1).cpu().numpy()
        feature_groups = group_np
        feature_is_matched = matched_np.astype(np.int64)
        feature_matched_gt_is_unknown = matched_gt_unknown_np.astype(np.int64)
        remaining = state['max_feature_samples'] - len(state['objectness_features'])
        if remaining > 0:
            if obj_feat.shape[0] > remaining:
                obj_feat = obj_feat[:remaining]
                known_feat = known_feat[:remaining]
                cls_feat = cls_feat[:remaining]
                feature_groups = feature_groups[:remaining]
                feature_is_matched = feature_is_matched[:remaining]
                feature_matched_gt_is_unknown = feature_matched_gt_is_unknown[:remaining]
            state['objectness_features'].extend(list(obj_feat))
            state['knownness_features'].extend(list(known_feat))
            state['classification_features'].extend(list(cls_feat))
            state['feature_groups'].extend(feature_groups.tolist())
            state['feature_is_matched'].extend(feature_is_matched.tolist())
            state['feature_matched_gt_is_unknown'].extend(feature_matched_gt_is_unknown.tolist())

    vis_debug = outputs.get('vis_debug', None)
    if vis_debug is not None:
        layer_objectness_probability = vis_debug.get('layer_objectness_probability', None)
        layer_knownness_probability = vis_debug.get('layer_knownness_probability', None)
        layer_unknown_probability = vis_debug.get('layer_unknown_probability', None)
        layer_max_known_class_probability = vis_debug.get('layer_max_known_class_probability', None)
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

    for batch_index in range(len(targets)):
        if state['saved_case_count'] >= viz_cfg['max_qualitative_cases']:
            break

        image_hw = targets[batch_index]['size'].tolist()
        image_np = to_numpy_image(samples.tensors[batch_index], image_hw)
        image_id = int(targets[batch_index]['image_id'].item()) if 'image_id' in targets[batch_index] else state['saved_case_count']
        ground_truth_boxes = cxcywh_to_abs_xyxy(targets[batch_index]['boxes'], image_hw)
        ground_truth_labels = targets[batch_index]['labels'].detach().cpu().numpy()
        prediction = postprocessed_predictions[batch_index]
        prediction_boxes = prediction['boxes'].detach().cpu().numpy()
        prediction_labels = prediction['labels'].detach().cpu().numpy()
        prediction_scores = prediction['scores'].detach().cpu().numpy()

        filtered_boxes, filtered_labels, filtered_scores = filter_prediction_display(
            prediction_boxes,
            prediction_labels,
            prediction_scores,
            image_hw,
            unknown_label,
            viz_cfg,
        )

        summary_text = f'ID={image_id} | epoch={int(epoch):04d} | pred={len(filtered_boxes)} | gt={len(ground_truth_boxes)}'
        case_prefix = f'{image_id:012d}__epoch_{int(epoch):04d}'

        prediction_vs_gt_path = os.path.join(final_dir, f'{case_prefix}__prediction_vs_gt.png')
        known_predictions_path = os.path.join(final_dir, f'{case_prefix}__known_predictions.png')
        unknown_predictions_path = os.path.join(final_dir, f'{case_prefix}__unknown_predictions.png')
        ground_truth_path = os.path.join(final_dir, f'{case_prefix}__ground_truth.png')

        render_prediction_vs_gt(
            image_np,
            filtered_boxes,
            filtered_labels,
            filtered_scores,
            ground_truth_boxes,
            ground_truth_labels,
            unknown_label,
            viz_cfg,
            prediction_vs_gt_path,
        )
        render_known_predictions(image_np, filtered_boxes, filtered_labels, filtered_scores, unknown_label, viz_cfg, known_predictions_path)
        render_unknown_predictions(image_np, filtered_boxes, filtered_labels, filtered_scores, unknown_label, viz_cfg, unknown_predictions_path)
        render_ground_truth(image_np, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, ground_truth_path)

        final_prediction_panel = np.asarray(Image.open(prediction_vs_gt_path).convert('RGB'))
        prediction_known_panel = np.asarray(Image.open(known_predictions_path).convert('RGB'))
        prediction_unknown_panel = np.asarray(Image.open(unknown_predictions_path).convert('RGB'))
        ground_truth_panel = np.asarray(Image.open(ground_truth_path).convert('RGB'))

        primary_panel_path = os.path.join(final_dir, f'{case_prefix}__panel.png')
        _save_panel(
            [
                (ground_truth_panel, 'Ground Truth'),
                (final_prediction_panel, 'Prediction vs GT'),
                (prediction_known_panel, 'Known Predictions'),
                (prediction_unknown_panel, 'Unknown Predictions'),
            ],
            primary_panel_path,
            viz_cfg,
        )
        state['saved_primary_panels'].append(primary_panel_path)

        if viz_cfg['save_error_panel']:
            unknown_to_known_path = os.path.join(final_dir, f'{case_prefix}__unknown_to_known.png')
            known_to_unknown_path = os.path.join(final_dir, f'{case_prefix}__known_to_unknown.png')
            _, num_u2k = render_error_unknown_to_known(
                image_np,
                filtered_boxes,
                filtered_labels,
                filtered_scores,
                ground_truth_boxes,
                ground_truth_labels,
                unknown_label,
                viz_cfg,
                unknown_to_known_path,
            )
            _, num_k2u = render_error_known_to_unknown(
                image_np,
                filtered_boxes,
                filtered_labels,
                filtered_scores,
                ground_truth_boxes,
                ground_truth_labels,
                unknown_label,
                viz_cfg,
                known_to_unknown_path,
            )
            unknown_to_known_panel = np.asarray(Image.open(unknown_to_known_path).convert('RGB'))
            known_to_unknown_panel = np.asarray(Image.open(known_to_unknown_path).convert('RGB'))
            error_panel_path = os.path.join(final_dir, f'{case_prefix}__errors.png')
            _save_panel(
                [
                    (final_prediction_panel, 'Prediction vs GT'),
                    (unknown_to_known_panel, 'Error: Unknown -> Known'),
                    (known_to_unknown_panel, 'Error: Known -> Unknown'),
                    (prediction_unknown_panel, 'Unknown Predictions'),
                ],
                error_panel_path,
                viz_cfg,
            )
            state['saved_error_panels'].append(error_panel_path)
            state['error_rows'].append(
                {
                    'image_id': image_id,
                    'num_predictions': int(len(filtered_boxes)),
                    'num_ground_truth_boxes': int(len(ground_truth_boxes)),
                    'num_unknown_to_known_errors': int(num_u2k),
                    'num_known_to_unknown_errors': int(num_k2u),
                }
            )

        if viz_cfg['save_mining_stage_panel'] and mining_debug is not None and batch_index < len(mining_debug):
            debug_item = mining_debug[batch_index]
            stage_panel_path = os.path.join(debug_dir, f'{case_prefix}__mining_stages.png')
            stage_images = [
                (final_prediction_panel, 'Final prediction'),
                (_draw_stage_boxes(image_np, viz_cfg, debug_item.get('after_gt_overlap_filter_boxes', []), 'After GT-overlap filter', (30, 136, 229)), 'After GT-overlap filter'),
                (_draw_stage_boxes(image_np, viz_cfg, debug_item.get('after_geometry_filter_boxes', []), 'After geometry filter', (30, 136, 229)), 'After geometry filter'),
                (_draw_stage_boxes(image_np, viz_cfg, debug_item.get('candidate_boxes_before_selection', []), 'Pseudo-positive candidates', (30, 136, 229), debug_item.get('candidate_score_texts')), 'Pseudo-positive candidates'),
                (_draw_stage_boxes(image_np, viz_cfg, debug_item.get('selected_pseudo_positive_boxes', []), 'Selected pseudo positives', (21, 101, 192)), 'Selected pseudo positives'),
                (_draw_stage_boxes(image_np, viz_cfg, debug_item.get('selected_reliable_background_boxes', []), 'Reliable background queries', (142, 36, 170)), 'Reliable background queries'),
            ]
            _save_panel(stage_images, stage_panel_path, viz_cfg)
            state['saved_stage_panels'].append(stage_panel_path)

        if tb_writer is not None and state['saved_case_count'] < viz_cfg['max_tensorboard_cases']:
            tb_writer.add_image(f'eval_qualitative/{image_id:012d}_panel', np.array(Image.open(primary_panel_path)), global_step=global_step, dataformats='HWC')

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
        plot_query_probability_histograms_by_group(state, os.path.join(stats_dir, f'query_probability_histograms.{viz_cfg["figure_format"]}'))
        plot_query_relationship_scatter(state, os.path.join(stats_dir, f'query_relationship_scatter.{viz_cfg["figure_format"]}'))
        plot_branch_correlation_heatmap(state, os.path.join(stats_dir, f'branch_correlation_heatmap.{viz_cfg["figure_format"]}'))
        plot_layer_prediction_summary(state, os.path.join(stats_dir, f'layer_prediction_summary.{viz_cfg["figure_format"]}'))

    if viz_cfg['save_feature_embedding_plots']:
        output_dirs = {
            'feature_2d': os.path.join(stats_dir, 'embeddings', 'feature', '2d'),
            'feature_3d': os.path.join(stats_dir, 'embeddings', 'feature', '3d'),
            'score_2d': os.path.join(stats_dir, 'embeddings', 'score_space', '2d'),
            'score_3d': os.path.join(stats_dir, 'embeddings', 'score_space', '3d'),
        }
        plot_feature_embedding_views(state, output_dirs, viz_cfg)
        plot_score_space_embedding_views(state, output_dirs, viz_cfg)

    if viz_cfg['save_contact_sheet']:
        save_contact_sheet(state['saved_primary_panels'], os.path.join(final_dir, 'primary_panels_contact_sheet.png'), viz_cfg)
        save_contact_sheet(state['saved_error_panels'], os.path.join(final_dir, 'error_panels_contact_sheet.png'), viz_cfg)
        save_contact_sheet(state['saved_stage_panels'], os.path.join(debug_dir, 'mining_stage_panels_contact_sheet.png'), viz_cfg)
