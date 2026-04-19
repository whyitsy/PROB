import csv
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    FEATURE_METADATA_KEYS,
    QUERY_METADATA_KEYS,
    plot_branch_correlation_heatmap,
    plot_layer_prediction_summary,
    plot_query_probability_histograms_by_group,
    plot_query_relationship_scatter,
)
from util.visual.helper import cxcywh_to_abs_xyxy, ensure_parent, to_numpy_image


QUERY_STATS_COLUMNS = [
    'objectness_probability',
    'unknown_probability',
    'max_known_class_probability',
    'query_group',
    *QUERY_METADATA_KEYS,
]


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _get_font(pixel_reference, font_scale, min_size):
    if isinstance(pixel_reference, np.ndarray):
        ref = max(pixel_reference.shape[0], pixel_reference.shape[1])
    else:
        ref = int(pixel_reference)
    font_size = max(min_size, int(ref * font_scale))
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
    tiles = []
    for image_np, title in images_with_titles:
        image = Image.fromarray(image_np).convert('RGB').resize((tile_width, tile_height))
        canvas = Image.new('RGB', (tile_width, tile_height), (20, 20, 20))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        font = _get_font(np.asarray(canvas), viz_cfg['font_size_scale'], viz_cfg['min_font_size'])
        _draw_text_with_background(draw, (8, 8), title, font, (255, 255, 255))
        tiles.append(canvas)
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new('RGB', (cols * tile_width, rows * tile_height), (15, 15, 15))
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile_width
        y = (index // cols) * tile_height
        sheet.paste(tile, (x, y))
    ensure_parent(output_path)
    sheet.save(output_path)


def init_eval_visual_state(viz_cfg):
    state = {
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
    for key in QUERY_METADATA_KEYS:
        state[key] = []
    for key in FEATURE_METADATA_KEYS:
        state[key] = []
    return state


def save_eval_qualitative_cases(state, samples, targets, postprocessed_predictions, outputs, criterion, args, output_dir, viz_cfg, tb_writer=None, global_step=0, epoch=0):
    del tb_writer, global_step
    epoch = max(int(epoch), 0)
    unknown_label = int(getattr(args, 'num_classes', 81) - 1)
    final_dir = os.path.join(output_dir, 'final')
    debug_dir = os.path.join(output_dir, 'debug')
    _ensure_dir(final_dir)
    _ensure_dir(debug_dir)

    mining_debug = None
    if getattr(viz_cfg, 'save_mining_stage_panel', viz_cfg.get('save_mining_stage_panel', False)) and hasattr(criterion, 'generate_pseudo_mining_debug'):
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
        raw_boxes = prediction['boxes'].detach().cpu().numpy()
        raw_labels = prediction['labels'].detach().cpu().numpy()
        raw_scores = prediction['scores'].detach().cpu().numpy()
        filtered_boxes, filtered_labels, filtered_scores = filter_prediction_display(raw_boxes, raw_labels, raw_scores, image_hw, unknown_label, viz_cfg)

        case_prefix = f'{image_id:012d}__epoch_{epoch:04d}'
        prediction_vs_gt_path = os.path.join(final_dir, f'{case_prefix}__prediction_vs_gt.png')
        known_predictions_path = os.path.join(final_dir, f'{case_prefix}__known_predictions.png')
        unknown_predictions_path = os.path.join(final_dir, f'{case_prefix}__unknown_predictions.png')
        ground_truth_path = os.path.join(final_dir, f'{case_prefix}__ground_truth.png')

        render_prediction_vs_gt(image_np, filtered_boxes, filtered_labels, filtered_scores, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, prediction_vs_gt_path)
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

        if viz_cfg.get('save_error_panel', False):
            unknown_to_known_path = os.path.join(final_dir, f'{case_prefix}__error_unknown_to_known.png')
            known_to_unknown_path = os.path.join(final_dir, f'{case_prefix}__error_known_to_unknown.png')
            _, num_u2k = render_error_unknown_to_known(image_np, filtered_boxes, filtered_labels, filtered_scores, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, unknown_to_known_path)
            _, num_k2u = render_error_known_to_unknown(image_np, filtered_boxes, filtered_labels, filtered_scores, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, known_to_unknown_path)
            if num_u2k > 0:
                state['saved_error_panels'].append(unknown_to_known_path)
            if num_k2u > 0:
                state['saved_error_panels'].append(known_to_unknown_path)
            state['error_rows'].append(
                {
                    'image_id': image_id,
                    'num_predictions_raw': int(len(raw_boxes)),
                    'num_predictions_filtered': int(len(filtered_boxes)),
                    'num_ground_truth_boxes': int(len(ground_truth_boxes)),
                    'num_unknown_to_known_errors': int(num_u2k),
                    'num_known_to_unknown_errors': int(num_k2u),
                }
            )

        if viz_cfg.get('save_mining_stage_panel', False) and mining_debug is not None and batch_index < len(mining_debug):
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

        state['saved_case_count'] += 1


def finalize_eval_visualizations(state, output_dir, epoch, viz_cfg, tb_writer=None):
    del tb_writer
    epoch = max(int(epoch), 0)
    output_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{epoch:04d}')
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
            writer.writerow(QUERY_STATS_COLUMNS)
            rows = zip(*(state[column] for column in QUERY_STATS_COLUMNS))
            for row in rows:
                writer.writerow(row)

    if viz_cfg['save_feature_npz'] and state['objectness_features']:
        save_dict = {
            'objectness_features': np.asarray(state['objectness_features'], dtype=np.float32),
            'knownness_features': np.asarray(state['knownness_features'], dtype=np.float32),
            'classification_features': np.asarray(state['classification_features'], dtype=np.float32),
            'feature_groups': np.asarray(state['feature_groups'], dtype=np.int64),
        }
        for key in FEATURE_METADATA_KEYS:
            save_dict[key] = np.asarray(state[key], dtype=np.int64)
        np.savez_compressed(os.path.join(stats_dir, 'feature_samples.npz'), **save_dict)

    if viz_cfg['save_error_summary_csv'] and state['error_rows']:
        with open(os.path.join(stats_dir, 'error_case_summary.csv'), 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['image_id', 'num_predictions_raw', 'num_predictions_filtered', 'num_ground_truth_boxes', 'num_unknown_to_known_errors', 'num_known_to_unknown_errors'])
            writer.writeheader()
            for row in state['error_rows']:
                writer.writerow(row)

    if viz_cfg['save_query_distribution_plots']:
        plot_query_probability_histograms_by_group(state, os.path.join(stats_dir, f'query_probability_histograms_by_group.{viz_cfg["figure_format"]}'))
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
        save_contact_sheet(state['saved_primary_panels'], os.path.join(final_dir, 'prediction_vs_gt_contact_sheet.png'), viz_cfg)
        save_contact_sheet(state['saved_error_panels'], os.path.join(final_dir, 'error_cases_contact_sheet.png'), viz_cfg)
        save_contact_sheet(state['saved_stage_panels'], os.path.join(debug_dir, 'mining_stage_panels_contact_sheet.png'), viz_cfg)
