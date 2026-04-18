import csv
import math
import os
import textwrap
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
    'matched_known': '#00A65A',
    'high_unknown_unmatched': '#D81B60',
    'other_unmatched': '#6C757D',
    'semantic_known': '#00A65A',
    'semantic_unknown': '#F39C12',
}
GROUP_NAMES = ['matched-known', 'unmatched-high-unknown', 'other-unmatched']
GROUP_COLORS = [COLOR['matched_known'], COLOR['high_unknown_unmatched'], COLOR['other_unmatched']]
SEMANTIC_NAMES = ['matched-known-gt', 'matched-unknown-gt']
SEMANTIC_COLORS = [COLOR['semantic_known'], COLOR['semantic_unknown']]
QUERY_STATS_COLUMNS = [
    'objectness_probability',
    'unknown_probability',
    'max_known_class_probability',
    'query_group',
    'is_matched',
    'matched_gt_label',
    'matched_gt_is_unknown',
    'pred_top1_label',
    'pred_top1_is_unknown',
    'top1_known_class',
    'image_id',
    'query_index',
]
FEATURE_METADATA_COLUMNS = [
    'feature_groups',
    'feature_is_matched',
    'feature_matched_gt_label',
    'feature_matched_gt_is_unknown',
    'feature_pred_top1_label',
    'feature_pred_top1_is_unknown',
    'feature_top1_known_class',
    'feature_image_id',
    'feature_query_index',
]


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def _compute_line_width(image_np, viz_cfg):
    return max(viz_cfg['min_line_width'], int(max(image_np.shape[0], image_np.shape[1]) * viz_cfg['line_width_scale']))


def _draw_text_with_background(draw, xy, text, font, fill, background_fill=(20, 20, 20, 220), pad=3):
    bbox = draw.textbbox(xy, text, font=font)
    draw.rounded_rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], radius=4, fill=background_fill)
    draw.text(xy, text, font=font, fill=fill)
    return bbox


def _prediction_text(label, score, unknown_label):
    if int(label) == int(unknown_label):
        return f'U {score:.2f}' if score is not None else 'U'
    return f'K[{int(label)}] {score:.2f}' if score is not None else f'K[{int(label)}]'


def _ground_truth_text(label, unknown_label):
    if int(label) == int(unknown_label):
        return 'GT-U'
    return f'GT-K[{int(label)}]'


def _build_legend_items():
    return [
        ('Pred Known', COLOR['prediction_known']),
        ('Pred Unknown', COLOR['prediction_unknown']),
        ('GT Known', COLOR['ground_truth_known']),
        ('GT Unknown', COLOR['ground_truth_unknown']),
    ]


def _wrap_lines(draw, lines, font, width_limit):
    wrapped = []
    for line in lines:
        line = '' if line is None else str(line)
        if not line:
            wrapped.append('')
            continue
        words = line.split()
        current = words[0] if words else ''
        for word in words[1:]:
            candidate = f'{current} {word}'.strip()
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if (bbox[2] - bbox[0]) <= width_limit:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        if current:
            wrapped.append(current)
    return wrapped or ['']


def _estimate_header_height(image_np, header_sections, viz_cfg, include_legend):
    ref = max(image_np.shape[0], image_np.shape[1])
    title_font = _get_font(ref, viz_cfg['title_font_size_scale'], viz_cfg['min_font_size'] + 2)
    info_font = _get_font(ref, viz_cfg['info_font_size_scale'], viz_cfg['min_font_size'])
    dummy = ImageDraw.Draw(Image.new('RGB', (16, 16)))
    section_gap = int(viz_cfg.get('header_section_gap', 18))
    line_gap = int(viz_cfg.get('header_text_line_gap', 7))
    width = image_np.shape[1]
    section_widths = [
        int(width * 0.42),
        int(width * 0.30),
    ]
    max_text_height = 0
    for idx, lines in enumerate(header_sections):
        font = title_font if idx == 0 else info_font
        width_limit = section_widths[min(idx, len(section_widths) - 1)]
        wrapped = _wrap_lines(dummy, lines, font, width_limit)
        height = 18
        for line in wrapped:
            bbox = dummy.textbbox((0, 0), line or ' ', font=font)
            height += (bbox[3] - bbox[1]) + line_gap
        max_text_height = max(max_text_height, height)
    legend_height = 0
    if include_legend:
        legend_font = _get_font(ref, viz_cfg['legend_font_size_scale'], viz_cfg['min_font_size'])
        box_size = max(12, int(ref * 0.014))
        legend_height = 18 + len(_build_legend_items()) * (box_size + 8) + (legend_font.size if hasattr(legend_font, 'size') else 12)
    minimum = max(viz_cfg['header_min_height'], int(image_np.shape[0] * viz_cfg['header_height_ratio']))
    return max(minimum, max_text_height + 18, legend_height + 18, section_gap * 2)


def _draw_header(draw, canvas_width, header_height, image_np, header_sections, viz_cfg, include_legend):
    ref = max(image_np.shape[0], image_np.shape[1])
    title_font = _get_font(ref, viz_cfg['title_font_size_scale'], viz_cfg['min_font_size'] + 2)
    info_font = _get_font(ref, viz_cfg['info_font_size_scale'], viz_cfg['min_font_size'])
    legend_font = _get_font(ref, viz_cfg['legend_font_size_scale'], viz_cfg['min_font_size'])
    section_gap = int(viz_cfg.get('header_section_gap', 18))
    line_gap = int(viz_cfg.get('header_text_line_gap', 7))

    left_x = 14
    center_x = int(canvas_width * 0.42)
    legend_x = int(canvas_width * (1.0 - viz_cfg.get('header_legend_width_ratio', 0.24)))
    top_y = 12

    section_specs = [
        (header_sections[0], title_font, left_x, int(canvas_width * 0.36)),
        (header_sections[1], info_font, center_x, int(canvas_width * 0.26)),
    ]
    for lines, font, x0, width_limit in section_specs:
        y = top_y
        for line in _wrap_lines(draw, lines, font, width_limit):
            bbox = _draw_text_with_background(draw, (x0, y), line or ' ', font, (255, 255, 255))
            y = bbox[3] + line_gap

    if include_legend:
        draw.rounded_rectangle([legend_x - 8, top_y - 4, canvas_width - 12, header_height - 14], radius=10, outline=(78, 82, 90), width=2, fill=(18, 20, 28))
        y = top_y + 6
        _draw_text_with_background(draw, (legend_x, y), 'Legend', legend_font, (255, 255, 255))
        y += max(22, legend_font.size + 10 if hasattr(legend_font, 'size') else 22)
        box_size = max(12, int(ref * 0.014))
        for label, color_hex in _build_legend_items():
            color = _hex_to_rgb(color_hex)
            draw.rectangle([legend_x, y + 3, legend_x + box_size, y + 3 + box_size], outline=color, width=2, fill=(30, 30, 30))
            _draw_text_with_background(draw, (legend_x + box_size + 8, y), label, legend_font, (255, 255, 255))
            y += box_size + 8

    draw.line([(0, header_height - 2), (canvas_width, header_height - 2)], fill=(70, 70, 70), width=2)


def _nms_xyxy(boxes, scores, iou_threshold):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        current = boxes[i:i + 1]
        rest = boxes[order[1:]]
        ious = box_ops.box_iou(torch.from_numpy(current), torch.from_numpy(rest))[0].numpy()[0]
        order = order[1:][ious < float(iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def _is_valid_geometry_xyxy(box, image_hw, viz_cfg):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    h, w = int(image_hw[0]), int(image_hw[1])
    min_area = float(h * w) * float(viz_cfg['display_min_area_ratio'])
    min_side = min(float(w), float(h)) * float(viz_cfg['display_min_side_ratio'])
    if area < min_area or min(bw, bh) < min_side:
        return False
    aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
    return aspect <= float(viz_cfg['display_max_aspect_ratio'])


def _filter_prediction_display(prediction_boxes, prediction_labels, prediction_scores, image_hw, unknown_label, viz_cfg):
    if prediction_boxes is None or len(prediction_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(prediction_boxes, dtype=np.float32)
    labels = np.asarray(prediction_labels, dtype=np.int64)
    scores = np.asarray(prediction_scores, dtype=np.float32)
    keep = []
    for idx in range(boxes.shape[0]):
        label = int(labels[idx])
        score = float(scores[idx])
        threshold = float(viz_cfg['display_unknown_score_thresh']) if label == int(unknown_label) else float(viz_cfg['display_known_score_thresh'])
        if score < threshold:
            continue
        if viz_cfg['display_apply_geometry_filter'] and not _is_valid_geometry_xyxy(boxes[idx], image_hw, viz_cfg):
            continue
        keep.append(idx)
    if not keep:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    keep = np.asarray(keep, dtype=np.int64)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    final_keep = []
    for select_unknown in [False, True]:
        mask = labels == int(unknown_label) if select_unknown else labels != int(unknown_label)
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            continue
        kept_local = _nms_xyxy(boxes[idxs], scores[idxs], viz_cfg['display_nms_iou'])
        if kept_local.size > 0:
            final_keep.append(idxs[kept_local])
    if not final_keep:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    final_keep = np.concatenate(final_keep, axis=0)
    final_keep = final_keep[np.argsort(-scores[final_keep])]
    return boxes[final_keep], labels[final_keep], scores[final_keep]


def _draw_boxes(
    image_np,
    viz_cfg,
    prediction_boxes=None,
    prediction_labels=None,
    prediction_scores=None,
    ground_truth_boxes=None,
    ground_truth_labels=None,
    header_title=None,
    header_meta_lines=None,
    header_stat_lines=None,
    unknown_label=80,
    show_legend=False,
):
    image = Image.fromarray(image_np).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    box_width = _compute_line_width(image_np, viz_cfg)
    font = _get_font(image_np, viz_cfg['font_size_scale'], viz_cfg['min_font_size'])

    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        for index, box in enumerate(ground_truth_boxes):
            x1, y1, x2, y2 = [float(value) for value in box]
            label = int(ground_truth_labels[index]) if ground_truth_labels is not None else -1
            color = _hex_to_rgb(COLOR['ground_truth_unknown'] if label == int(unknown_label) else COLOR['ground_truth_known'])
            draw_overlay.rectangle([x1, y1, x2, y2], outline=color + (235,), width=box_width)
            _draw_text_with_background(draw_overlay, (x1 + 2, max(0, y1 - getattr(font, 'size', 12) - 4)), _ground_truth_text(label, unknown_label), font, color + (255,))

    if prediction_boxes is not None and len(prediction_boxes) > 0:
        for index, box in enumerate(prediction_boxes):
            x1, y1, x2, y2 = [float(value) for value in box]
            label = int(prediction_labels[index]) if prediction_labels is not None else -1
            score = float(prediction_scores[index]) if prediction_scores is not None else None
            color = _hex_to_rgb(COLOR['prediction_unknown'] if label == int(unknown_label) else COLOR['prediction_known'])
            draw_overlay.rectangle([x1, y1, x2, y2], outline=color + (235,), width=box_width)
            _draw_text_with_background(draw_overlay, (x1 + 2, y1 + 2), _prediction_text(label, score, unknown_label), font, color + (255,))

    composed = Image.alpha_composite(image, overlay).convert('RGB')
    header_meta_lines = list(header_meta_lines or [])
    header_stat_lines = list(header_stat_lines or [])
    header_sections = [
        [header_title] + header_meta_lines,
        header_stat_lines,
    ]
    header_height = _estimate_header_height(image_np, header_sections, viz_cfg, show_legend)
    canvas = Image.new('RGB', (composed.width, composed.height + header_height), (10, 12, 18))
    canvas.paste(composed, (0, header_height))
    header_draw = ImageDraw.Draw(canvas)
    _draw_header(header_draw, canvas.width, header_height, image_np, header_sections, viz_cfg, show_legend)
    return np.array(canvas)


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
    groups = np.asarray(state['query_group'], dtype=np.int64)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    hist_specs = [
        ('objectness_probability', 'Objectness probability', axes[0]),
        ('unknown_probability', 'Unknown probability', axes[1]),
        ('max_known_class_probability', 'Max known-class probability', axes[2]),
    ]
    for field, title, axis in hist_specs:
        values = np.asarray(state[field], dtype=np.float32)
        if values.size == 0:
            axis.set_axis_off()
            continue
        value_min = float(values.min())
        value_max = float(values.max())
        if math.isclose(value_min, value_max):
            value_max = value_min + 1e-3
        bins = np.linspace(value_min, value_max, 36)
        for group_index, (group_name, color) in enumerate(zip(GROUP_NAMES, GROUP_COLORS)):
            mask = groups == group_index
            if np.any(mask):
                axis.hist(values[mask], bins=bins, alpha=0.40, label=group_name, color=color, histtype='stepfilled')
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    _save_figure(
        figure,
        os.path.join(output_dir, f'query_probability_histograms_by_group.{viz_cfg["figure_format"]}'),
        tb_writer,
        'eval_viz/query_probability_histograms_by_group',
        global_step,
    )


def _plot_scatter(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if not state['objectness_probability']:
        return
    groups = np.asarray(state['query_group'], dtype=np.int64)
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))
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
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    _save_figure(
        figure,
        os.path.join(output_dir, f'query_relationship_scatter.{viz_cfg["figure_format"]}'),
        tb_writer,
        'eval_viz/query_relationship_scatter',
        global_step,
    )


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
    _save_figure(
        figure,
        os.path.join(output_dir, f'branch_correlation_heatmap.{viz_cfg["figure_format"]}'),
        tb_writer,
        'eval_viz/branch_correlation_heatmap',
        global_step,
    )


def _select_embedding_max_points(method, dim, viz_cfg):
    if method == 'tsne':
        return int(viz_cfg['embedding_tsne_max_points_2d'] if dim == 2 else viz_cfg['embedding_tsne_max_points_3d'])
    if method == 'umap':
        return int(viz_cfg['embedding_umap_max_points_2d'] if dim == 2 else viz_cfg['embedding_umap_max_points_3d'])
    return int(viz_cfg['embedding_generic_max_points_2d'] if dim == 2 else viz_cfg['embedding_generic_max_points_3d'])


def _subsample_evenly(features, labels, max_points):
    if features.shape[0] <= max_points:
        return features, labels
    indices = np.linspace(0, features.shape[0] - 1, max_points).astype(np.int64)
    return features[indices], labels[indices]


def _compute_embedding(features, method, dim, viz_cfg):
    random_state = int(viz_cfg.get('embedding_random_state', 42))
    if method == 'pca':
        return PCA(n_components=dim, random_state=random_state).fit_transform(features)
    if method == 'tsne':
        perplexity = min(int(viz_cfg.get('embedding_tsne_perplexity_cap', 30)), max(2, features.shape[0] // 4))
        return TSNE(
            n_components=dim,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto',
            random_state=random_state,
        ).fit_transform(features)
    if method == 'umap':
        try:
            import umap
        except Exception:
            return None
        n_neighbors = min(int(viz_cfg.get('embedding_umap_n_neighbors', 20)), max(2, features.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            min_dist=float(viz_cfg.get('embedding_umap_min_dist', 0.15)),
            random_state=random_state,
        )
        return reducer.fit_transform(features)
    raise ValueError(f'Unsupported embedding method: {method}')


def _scatter_embedding(axis, embedding, labels, names, colors, dim):
    for group_index, (name, color) in enumerate(zip(names, colors)):
        mask = labels == group_index
        if not np.any(mask):
            continue
        if dim == 3:
            axis.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], s=9, alpha=0.58, c=color, label=name)
        else:
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=9, alpha=0.58, c=color, label=name)
    axis.grid(alpha=0.2)
    if dim == 3:
        axis.set_xlabel('C1')
        axis.set_ylabel('C2')
        axis.set_zlabel('C3')
    else:
        axis.set_xlabel('C1')
        axis.set_ylabel('C2')


def _create_embedding_axes(dim):
    if dim == 3:
        figure = plt.figure(figsize=(17.8, 5.6))
        axes = [figure.add_subplot(1, 3, idx + 1, projection='3d') for idx in range(3)]
    else:
        figure, axes = plt.subplots(1, 3, figsize=(16.4, 5.1))
        axes = list(axes)
    return figure, axes


def _plot_embedding_views(feature_specs, metadata, output_dir, viz_cfg, tb_writer=None, global_step=0):
    methods = list(viz_cfg.get('embedding_methods', ['pca', 'tsne', 'umap']))
    dims = list(viz_cfg.get('embedding_dims', [2, 3]))

    query_groups = np.asarray(metadata['feature_groups'], dtype=np.int64)
    matched_gt_is_unknown = np.asarray(metadata['feature_matched_gt_is_unknown'], dtype=np.int64)
    feature_is_matched = np.asarray(metadata['feature_is_matched'], dtype=np.int64)

    view_specs = []
    if viz_cfg.get('save_embedding_group01_views', True):
        view_specs.append(('group01', np.isin(query_groups, [0, 1]), np.where(query_groups == 0, 0, 1), ['matched-known', 'unmatched-high-unknown'], [COLOR['matched_known'], COLOR['high_unknown_unmatched']]))
    if viz_cfg.get('save_embedding_group012_views', True):
        view_specs.append(('group012', np.ones_like(query_groups, dtype=bool), query_groups, GROUP_NAMES, GROUP_COLORS))
    if viz_cfg.get('save_embedding_semantic_views', True):
        semantic_mask = feature_is_matched.astype(bool)
        semantic_labels = matched_gt_is_unknown.astype(np.int64)
        view_specs.append(('semantic_known_unknown', semantic_mask, semantic_labels, SEMANTIC_NAMES, SEMANTIC_COLORS))

    for dim in dims:
        min_points = int(viz_cfg['embedding_min_points_2d'] if dim == 2 else viz_cfg['embedding_min_points_3d'])
        for method in methods:
            for view_name, base_mask, base_labels, view_names, view_colors in view_specs:
                figure, axes = _create_embedding_axes(dim)
                plotted_any = False
                legend_handles = None
                legend_labels = None
                for axis, (feature_key, title) in zip(axes, feature_specs):
                    feature_list = feature_specs[(feature_key, title)] if isinstance(feature_specs, dict) else None
                    if feature_list is None:
                        axis.set_axis_off()
                        continue
                    features = np.asarray(feature_list, dtype=np.float32)
                    mask = base_mask.copy()
                    if view_name == 'semantic_known_unknown':
                        mask = np.logical_and(mask, np.isin(base_labels, [0, 1]))
                    labels = base_labels[mask]
                    features = features[mask]
                    if features.shape[0] < min_points or np.unique(labels).size < 1:
                        axis.set_axis_off()
                        continue
                    if view_name == 'semantic_known_unknown' and np.unique(labels).size < 2:
                        axis.set_axis_off()
                        continue
                    max_points = _select_embedding_max_points(method, dim, viz_cfg)
                    features, labels = _subsample_evenly(features, labels, max_points)
                    try:
                        embedding = _compute_embedding(features, method, dim, viz_cfg)
                    except Exception:
                        axis.set_axis_off()
                        continue
                    if embedding is None:
                        axis.set_axis_off()
                        continue
                    _scatter_embedding(axis, embedding, labels, view_names, view_colors, dim)
                    axis.set_title(f'{title} · {method.upper()} · {dim}D · {view_name}')
                    handles, labels_text = axis.get_legend_handles_labels()
                    if handles:
                        legend_handles, legend_labels = handles, labels_text
                    plotted_any = True
                if plotted_any:
                    if legend_handles:
                        figure.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=max(2, len(legend_labels)), frameon=False)
                    filename = f'feature_embedding_{method}_{dim}d_{view_name}.{viz_cfg["figure_format"]}'
                    tb_tag = f'eval_viz/feature_embedding_{method}_{dim}d_{view_name}'
                    _save_figure(figure, os.path.join(output_dir, filename), tb_writer, tb_tag, global_step)
                else:
                    plt.close(figure)


def _plot_feature_embeddings(state, output_dir, viz_cfg, tb_writer=None, global_step=0):
    if not state['feature_groups'] or not state['objectness_features']:
        return
    feature_specs = {
        ('objectness_features', 'Objectness features'): state['objectness_features'],
        ('knownness_features', 'Knownness features'): state['knownness_features'],
        ('classification_features', 'Classification features'): state['classification_features'],
    }
    metadata = {
        'feature_groups': np.asarray(state['feature_groups'], dtype=np.int64),
        'feature_is_matched': np.asarray(state['feature_is_matched'], dtype=np.int64),
        'feature_matched_gt_is_unknown': np.asarray(state['feature_matched_gt_is_unknown'], dtype=np.int64),
    }
    _plot_embedding_views(feature_specs, metadata, output_dir, viz_cfg, tb_writer, global_step)


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
    _save_figure(
        figure,
        os.path.join(output_dir, f'layer_prediction_summary.{viz_cfg["figure_format"]}'),
        tb_writer,
        'eval_viz/layer_prediction_summary',
        global_step,
    )


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
        'saved_case_count': 0,
        'objectness_probability': [],
        'unknown_probability': [],
        'max_known_class_probability': [],
        'query_group': [],
        'is_matched': [],
        'matched_gt_label': [],
        'matched_gt_is_unknown': [],
        'pred_top1_label': [],
        'pred_top1_is_unknown': [],
        'top1_known_class': [],
        'image_id': [],
        'query_index': [],
        'objectness_features': [],
        'knownness_features': [],
        'classification_features': [],
        'feature_groups': [],
        'feature_is_matched': [],
        'feature_matched_gt_label': [],
        'feature_matched_gt_is_unknown': [],
        'feature_pred_top1_label': [],
        'feature_pred_top1_is_unknown': [],
        'feature_top1_known_class': [],
        'feature_image_id': [],
        'feature_query_index': [],
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


def _base_meta_lines(image_id, epoch):
    return [
        f'image_id: {image_id}',
        f'epoch: {int(epoch):04d}',
    ]


def _base_stat_lines(raw_pred_count, filtered_pred_count, gt_count, extra_lines=None):
    lines = [
        f'pred(raw): {raw_pred_count}',
        f'pred(filtered): {filtered_pred_count}',
        f'gt: {gt_count}',
    ]
    lines.extend(list(extra_lines or []))
    return lines


def save_eval_qualitative_cases(state, samples, targets, postprocessed_predictions, outputs, criterion, args, output_dir, viz_cfg, tb_writer=None, global_step=0, epoch=0):
    del outputs, criterion  # 当前定性图路径不再依赖 mining-stage debug

    epoch = max(int(epoch), 0)
    unknown_label = int(getattr(args, 'num_classes', 81) - 1)
    final_dir = os.path.join(output_dir, 'final')
    _ensure_dir(final_dir)

    for batch_index in range(len(targets)):
        if state['saved_case_count'] >= viz_cfg['max_qualitative_cases']:
            break
        image_hw = targets[batch_index]['size'].tolist()
        image_np = _to_numpy_image(samples.tensors[batch_index], image_hw)
        image_id = int(targets[batch_index]['image_id'].item()) if 'image_id' in targets[batch_index] else state['saved_case_count']
        ground_truth_boxes = _cxcywh_to_abs_xyxy(targets[batch_index]['boxes'], image_hw)
        ground_truth_labels = targets[batch_index]['labels'].detach().cpu().numpy()

        prediction = postprocessed_predictions[batch_index]
        raw_prediction_boxes = prediction['boxes'].detach().cpu().numpy()
        raw_prediction_labels = prediction['labels'].detach().cpu().numpy()
        raw_prediction_scores = prediction['scores'].detach().cpu().numpy()

        prediction_boxes, prediction_labels, prediction_scores = _filter_prediction_display(
            raw_prediction_boxes,
            raw_prediction_labels,
            raw_prediction_scores,
            image_hw=image_hw,
            unknown_label=unknown_label,
            viz_cfg=viz_cfg,
        )

        known_mask = prediction_labels != unknown_label if len(prediction_labels) > 0 else np.array([], dtype=bool)
        unknown_mask = prediction_labels == unknown_label if len(prediction_labels) > 0 else np.array([], dtype=bool)

        case_dir = os.path.join(final_dir, f'{image_id:012d}__epoch_{int(epoch):04d}')
        _ensure_dir(case_dir)

        final_prediction_image = _draw_boxes(
            image_np,
            viz_cfg,
            prediction_boxes=prediction_boxes,
            prediction_labels=prediction_labels,
            prediction_scores=prediction_scores,
            ground_truth_boxes=ground_truth_boxes,
            ground_truth_labels=ground_truth_labels,
            header_title='Prediction vs Ground Truth',
            header_meta_lines=_base_meta_lines(image_id, epoch),
            header_stat_lines=_base_stat_lines(
                len(raw_prediction_boxes),
                len(prediction_boxes),
                len(ground_truth_boxes),
                extra_lines=[
                    f'known_thr: {viz_cfg["display_known_score_thresh"]:.2f}',
                    f'unknown_thr: {viz_cfg["display_unknown_score_thresh"]:.2f}',
                    f'nms_iou: {viz_cfg["display_nms_iou"]:.2f}',
                    f'geometry_filter: {int(bool(viz_cfg["display_apply_geometry_filter"]))}',
                ],
            ),
            unknown_label=unknown_label,
            show_legend=True,
        )

        prediction_known_image = _draw_boxes(
            image_np,
            viz_cfg,
            prediction_boxes=prediction_boxes[known_mask] if len(prediction_boxes) > 0 else None,
            prediction_labels=prediction_labels[known_mask] if len(prediction_labels) > 0 else None,
            prediction_scores=prediction_scores[known_mask] if len(prediction_scores) > 0 else None,
            header_title='Known Predictions',
            header_meta_lines=_base_meta_lines(image_id, epoch),
            header_stat_lines=_base_stat_lines(
                len(raw_prediction_boxes),
                int(known_mask.sum()) if len(known_mask) > 0 else 0,
                len(ground_truth_boxes),
                extra_lines=['view: known-only'],
            ),
            unknown_label=unknown_label,
            show_legend=False,
        )

        prediction_unknown_image = _draw_boxes(
            image_np,
            viz_cfg,
            prediction_boxes=prediction_boxes[unknown_mask] if len(prediction_boxes) > 0 else None,
            prediction_labels=prediction_labels[unknown_mask] if len(prediction_labels) > 0 else None,
            prediction_scores=prediction_scores[unknown_mask] if len(prediction_scores) > 0 else None,
            header_title='Unknown Predictions',
            header_meta_lines=_base_meta_lines(image_id, epoch),
            header_stat_lines=_base_stat_lines(
                len(raw_prediction_boxes),
                int(unknown_mask.sum()) if len(unknown_mask) > 0 else 0,
                len(ground_truth_boxes),
                extra_lines=['view: unknown-only'],
            ),
            unknown_label=unknown_label,
            show_legend=False,
        )

        ground_truth_image = _draw_boxes(
            image_np,
            viz_cfg,
            ground_truth_boxes=ground_truth_boxes,
            ground_truth_labels=ground_truth_labels,
            header_title='Ground Truth',
            header_meta_lines=_base_meta_lines(image_id, epoch),
            header_stat_lines=_base_stat_lines(len(raw_prediction_boxes), len(prediction_boxes), len(ground_truth_boxes)),
            unknown_label=unknown_label,
            show_legend=False,
        )

        _save_image(ground_truth_image, os.path.join(case_dir, 'ground_truth.png'))
        prediction_path = os.path.join(case_dir, 'prediction_vs_gt.png')
        _save_image(final_prediction_image, prediction_path)
        _save_image(prediction_known_image, os.path.join(case_dir, 'known_predictions.png'))
        _save_image(prediction_unknown_image, os.path.join(case_dir, 'unknown_predictions.png'))
        state['saved_primary_panels'].append(prediction_path)

        if viz_cfg.get('save_error_panel', False):
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
                prediction_scores=prediction_scores[u2k_pred] if len(prediction_scores) > 0 else None,
                ground_truth_boxes=ground_truth_boxes[u2k_gt] if len(u2k_gt) > 0 else None,
                ground_truth_labels=ground_truth_labels[u2k_gt] if len(u2k_gt) > 0 else None,
                header_title='Error: Unknown -> Known',
                header_meta_lines=_base_meta_lines(image_id, epoch),
                header_stat_lines=_base_stat_lines(
                    len(raw_prediction_boxes),
                    len(prediction_boxes),
                    len(ground_truth_boxes),
                    extra_lines=[f'error_pairs: {len(u2k_gt)}'],
                ),
                unknown_label=unknown_label,
                show_legend=True,
            )
            known_to_unknown_image = _draw_boxes(
                image_np,
                viz_cfg,
                prediction_boxes=prediction_boxes[k2u_pred] if len(k2u_pred) > 0 else None,
                prediction_labels=prediction_labels[k2u_pred] if len(prediction_labels) > 0 else None,
                prediction_scores=prediction_scores[k2u_pred] if len(prediction_scores) > 0 else None,
                ground_truth_boxes=ground_truth_boxes[k2u_gt] if len(k2u_gt) > 0 else None,
                ground_truth_labels=ground_truth_labels[k2u_gt] if len(k2u_gt) > 0 else None,
                header_title='Error: Known -> Unknown',
                header_meta_lines=_base_meta_lines(image_id, epoch),
                header_stat_lines=_base_stat_lines(
                    len(raw_prediction_boxes),
                    len(prediction_boxes),
                    len(ground_truth_boxes),
                    extra_lines=[f'error_pairs: {len(k2u_gt)}'],
                ),
                unknown_label=unknown_label,
                show_legend=True,
            )
            u2k_path = os.path.join(case_dir, 'error_unknown_to_known.png')
            k2u_path = os.path.join(case_dir, 'error_known_to_unknown.png')
            _save_image(unknown_to_known_image, u2k_path)
            _save_image(known_to_unknown_image, k2u_path)
            if len(u2k_gt) > 0:
                state['saved_error_panels'].append(u2k_path)
            if len(k2u_gt) > 0:
                state['saved_error_panels'].append(k2u_path)
            state['error_rows'].append({
                'image_id': image_id,
                'num_predictions_raw': int(len(raw_prediction_boxes)),
                'num_predictions_filtered': int(len(prediction_boxes)),
                'num_ground_truth_boxes': int(len(ground_truth_boxes)),
                'num_unknown_to_known_errors': int(len(u2k_gt)),
                'num_known_to_unknown_errors': int(len(k2u_gt)),
            })

        if tb_writer is not None and state['saved_case_count'] < viz_cfg['max_tensorboard_cases']:
            tb_writer.add_image(f'eval_qualitative/{image_id:012d}_prediction_vs_gt', final_prediction_image, global_step=global_step, dataformats='HWC')
            tb_writer.add_image(f'eval_qualitative/{image_id:012d}_ground_truth', ground_truth_image, global_step=global_step, dataformats='HWC')
        state['saved_case_count'] += 1


def finalize_eval_visualizations(state, output_dir, epoch, viz_cfg, tb_writer=None):
    epoch = max(int(epoch), 0)
    output_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{int(epoch):04d}')
    stats_dir = os.path.join(output_dir, 'stats')
    final_dir = os.path.join(output_dir, 'final')
    _ensure_dir(stats_dir)
    _ensure_dir(final_dir)

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
        }
        for key in FEATURE_METADATA_COLUMNS:
            save_dict[key] = np.asarray(state[key], dtype=np.int64)
        np.savez_compressed(os.path.join(stats_dir, 'feature_samples.npz'), **save_dict)

    if viz_cfg['save_error_summary_csv'] and state['error_rows']:
        with open(os.path.join(stats_dir, 'error_case_summary.csv'), 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['image_id', 'num_predictions_raw', 'num_predictions_filtered', 'num_ground_truth_boxes', 'num_unknown_to_known_errors', 'num_known_to_unknown_errors'])
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
