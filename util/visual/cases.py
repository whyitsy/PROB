import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from util import box_ops
from util.visual.helper import save_image, save_svg_figure, to_numpy_image, write_gallery_svg


COLOR = {
    'prediction_known': '#00A65A',
    'prediction_unknown': '#D81B60',
    'ground_truth_known': '#00BCD4',
    'ground_truth_unknown': '#F39C12',
}
LEVEL_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00897B']
LAYER_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00897B', '#6D4C41', '#3949AB']
KNOWN_COLOR = '#00A65A'
UNKNOWN_COLOR = '#D81B60'
CATEGORY_COLORS = {
    'known': (0, 166, 90),
    'unknown': (216, 27, 96),
    'odqe_salient': (30, 136, 229),
}


def hex_to_rgb(hex_color):
    """把十六进制颜色转成 RGB。"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[index:index + 2], 16) for index in (0, 2, 4))


def get_font(pixel_reference, font_scale, min_size):
    """根据图像尺寸选择字体。"""
    if isinstance(pixel_reference, np.ndarray):
        ref = max(pixel_reference.shape[0], pixel_reference.shape[1])
    else:
        ref = int(pixel_reference)
    font_size = max(min_size, int(ref * font_scale))
    try:
        return ImageFont.truetype('DejaVuSans.ttf', font_size)
    except Exception:
        return ImageFont.load_default()


def compute_line_width(image_np, viz_cfg):
    """计算框线宽度。"""
    return max(viz_cfg['min_line_width'], int(max(image_np.shape[0], image_np.shape[1]) * viz_cfg['line_width_scale']))


def draw_text_with_background(draw, xy, text, font, fill, background_fill=(20, 20, 20, 220), pad=3):
    """绘制带背景的文本。"""
    bbox = draw.textbbox(xy, text, font=font)
    draw.rounded_rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], radius=4, fill=background_fill)
    draw.text(xy, text, font=font, fill=fill)
    return bbox


def prediction_text(label, score, unknown_label):
    """生成预测框文字标签。"""
    if int(label) == int(unknown_label):
        return f'U {score:.2f}' if score is not None else 'U'
    return f'K[{int(label)}] {score:.2f}' if score is not None else f'K[{int(label)}]'


def ground_truth_text(label, unknown_label):
    """生成 GT 框文字标签。"""
    if int(label) == int(unknown_label):
        return 'GT-U'
    return f'GT-K[{int(label)}]'


def legend_items_for_content(prediction_labels, ground_truth_labels, unknown_label):
    """根据内容生成图例。"""
    items = []
    if prediction_labels is not None and len(prediction_labels) > 0:
        prediction_labels = np.asarray(prediction_labels, dtype=np.int64)
        if np.any(prediction_labels != int(unknown_label)):
            items.append(('Pred Known', COLOR['prediction_known']))
        if np.any(prediction_labels == int(unknown_label)):
            items.append(('Pred Unknown', COLOR['prediction_unknown']))
    if ground_truth_labels is not None and len(ground_truth_labels) > 0:
        ground_truth_labels = np.asarray(ground_truth_labels, dtype=np.int64)
        if np.any(ground_truth_labels != int(unknown_label)):
            items.append(('GT Known', COLOR['ground_truth_known']))
        if np.any(ground_truth_labels == int(unknown_label)):
            items.append(('GT Unknown', COLOR['ground_truth_unknown']))
    return items


def draw_overlay_legend(base_image, legend_items, viz_cfg):
    """在右上角绘制图例。"""
    if not legend_items:
        return base_image
    image = base_image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ref = max(image.size)
    font = get_font(ref, viz_cfg['legend_font_size_scale'], viz_cfg['min_font_size'])
    box_size = max(12, int(ref * 0.014))
    line_gap = max(6, int(ref * 0.006))
    margin = max(10, int(ref * 0.012))

    dummy = ImageDraw.Draw(Image.new('RGBA', (16, 16), (0, 0, 0, 0)))
    row_heights = []
    row_widths = []
    for label, _ in legend_items:
        bbox = dummy.textbbox((0, 0), label, font=font)
        row_heights.append(max(box_size, bbox[3] - bbox[1]))
        row_widths.append(box_size + 8 + (bbox[2] - bbox[0]))

    max_row_width = max(row_widths)
    x2 = image.size[0] - margin
    x1 = x2 - max_row_width
    y = margin
    for (label, color_hex), row_height in zip(legend_items, row_heights):
        color = hex_to_rgb(color_hex)
        icon_y = y + max(0, (row_height - box_size) // 2)
        draw.rectangle([x1, icon_y, x1 + box_size, icon_y + box_size], outline=color + (255,), width=2)
        draw.text((x1 + box_size + 8, y - 1), label, font=font, fill=color + (255,))
        y += row_height + line_gap
    return Image.alpha_composite(image, overlay).convert('RGB')


def nms_xyxy(boxes, scores, iou_threshold):
    """对 xyxy 框做 NMS。"""
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        current_index = int(order[0])
        keep.append(current_index)
        if order.size == 1:
            break
        current = boxes[current_index:current_index + 1]
        rest = boxes[order[1:]]
        ious = box_ops.box_iou(torch.from_numpy(current), torch.from_numpy(rest))[0].numpy()[0]
        order = order[1:][ious < float(iou_threshold)]
    return np.asarray(keep, dtype=np.int64)


def is_valid_geometry_xyxy(box, image_hw, viz_cfg):
    """判断框几何是否合法。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    height, width = int(image_hw[0]), int(image_hw[1])
    min_area = float(height * width) * float(viz_cfg['display_min_area_ratio'])
    min_side = min(float(width), float(height)) * float(viz_cfg['display_min_side_ratio'])
    if area < min_area or min(bw, bh) < min_side:
        return False
    aspect = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
    return aspect <= float(viz_cfg['display_max_aspect_ratio'])


def filter_prediction_display(prediction_boxes, prediction_labels, prediction_scores, image_hw, unknown_label, viz_cfg):
    """过滤可视化用预测框。"""
    if prediction_boxes is None or len(prediction_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(prediction_boxes, dtype=np.float32)
    labels = np.asarray(prediction_labels, dtype=np.int64)
    scores = np.asarray(prediction_scores, dtype=np.float32)

    keep = []
    for index in range(boxes.shape[0]):
        label = int(labels[index])
        score = float(scores[index])
        threshold = float(viz_cfg['display_unknown_score_thresh']) if label == int(unknown_label) else float(viz_cfg['display_known_score_thresh'])
        if score < threshold:
            continue
        if viz_cfg['display_apply_geometry_filter'] and not is_valid_geometry_xyxy(boxes[index], image_hw, viz_cfg):
            continue
        keep.append(index)

    if not keep:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    keep = np.asarray(keep, dtype=np.int64)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    final_keep = []
    for select_unknown in [False, True]:
        mask = labels == int(unknown_label) if select_unknown else labels != int(unknown_label)
        selected = np.nonzero(mask)[0]
        if selected.size == 0:
            continue
        kept_local = nms_xyxy(boxes[selected], scores[selected], viz_cfg['display_nms_iou'])
        if kept_local.size > 0:
            final_keep.append(selected[kept_local])

    if not final_keep:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    final_keep = np.concatenate(final_keep, axis=0)
    final_keep = final_keep[np.argsort(-scores[final_keep])]
    return boxes[final_keep], labels[final_keep], scores[final_keep]


def draw_detection_panel(
    image_np,
    viz_cfg,
    *,
    prediction_boxes=None,
    prediction_labels=None,
    prediction_scores=None,
    ground_truth_boxes=None,
    ground_truth_labels=None,
    unknown_label=80,
):
    """绘制单张检测面板。"""
    image = Image.fromarray(image_np).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = compute_line_width(image_np, viz_cfg)
    font = get_font(image_np, viz_cfg['font_size_scale'], viz_cfg['min_font_size'])

    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        for index, box in enumerate(ground_truth_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = int(ground_truth_labels[index])
            color = hex_to_rgb(COLOR['ground_truth_unknown'] if label == int(unknown_label) else COLOR['ground_truth_known'])
            draw.rectangle([x1, y1, x2, y2], outline=color + (235,), width=line_width)
            draw_text_with_background(draw, (x1 + 2, max(0, y1 - getattr(font, 'size', 12) - 4)), ground_truth_text(label, unknown_label), font, color + (255,))

    if prediction_boxes is not None and len(prediction_boxes) > 0:
        for index, box in enumerate(prediction_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = int(prediction_labels[index])
            score = float(prediction_scores[index]) if prediction_scores is not None else None
            color = hex_to_rgb(COLOR['prediction_unknown'] if label == int(unknown_label) else COLOR['prediction_known'])
            draw.rectangle([x1, y1, x2, y2], outline=color + (235,), width=line_width)
            draw_text_with_background(draw, (x1 + 2, y1 + 2), prediction_text(label, score, unknown_label), font, color + (255,))

    composed = Image.alpha_composite(image, overlay).convert('RGB')
    legend_items = legend_items_for_content(prediction_labels, ground_truth_labels, unknown_label)
    return np.array(draw_overlay_legend(composed, legend_items, viz_cfg))


def render_ground_truth(image_np, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, output_path):
    """渲染 GT 图。"""
    image = draw_detection_panel(image_np, viz_cfg, ground_truth_boxes=ground_truth_boxes, ground_truth_labels=ground_truth_labels, unknown_label=unknown_label)
    return save_image(image, output_path)


def render_predictions(image_np, prediction_boxes, prediction_labels, prediction_scores, unknown_label, viz_cfg, output_path):
    """渲染预测图。"""
    image = draw_detection_panel(image_np, viz_cfg, prediction_boxes=prediction_boxes, prediction_labels=prediction_labels, prediction_scores=prediction_scores, unknown_label=unknown_label)
    return save_image(image, output_path)


def render_known_predictions(image_np, prediction_boxes, prediction_labels, prediction_scores, unknown_label, viz_cfg, output_path):
    """渲染 known 预测图。"""
    mask = prediction_labels != unknown_label if len(prediction_labels) > 0 else np.array([], dtype=bool)
    image = draw_detection_panel(
        image_np,
        viz_cfg,
        prediction_boxes=prediction_boxes[mask] if len(prediction_boxes) > 0 else None,
        prediction_labels=prediction_labels[mask] if len(prediction_labels) > 0 else None,
        prediction_scores=prediction_scores[mask] if len(prediction_scores) > 0 else None,
        unknown_label=unknown_label,
    )
    return save_image(image, output_path)


def render_unknown_predictions(image_np, prediction_boxes, prediction_labels, prediction_scores, unknown_label, viz_cfg, output_path):
    """渲染 unknown 预测图。"""
    mask = prediction_labels == unknown_label if len(prediction_labels) > 0 else np.array([], dtype=bool)
    image = draw_detection_panel(
        image_np,
        viz_cfg,
        prediction_boxes=prediction_boxes[mask] if len(prediction_boxes) > 0 else None,
        prediction_labels=prediction_labels[mask] if len(prediction_labels) > 0 else None,
        prediction_scores=prediction_scores[mask] if len(prediction_scores) > 0 else None,
        unknown_label=unknown_label,
    )
    return save_image(image, output_path)


def render_prediction_vs_gt(image_np, prediction_boxes, prediction_labels, prediction_scores, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, output_path):
    """渲染 prediction 与 GT 对比图。"""
    image = draw_detection_panel(
        image_np,
        viz_cfg,
        prediction_boxes=prediction_boxes,
        prediction_labels=prediction_labels,
        prediction_scores=prediction_scores,
        ground_truth_boxes=ground_truth_boxes,
        ground_truth_labels=ground_truth_labels,
        unknown_label=unknown_label,
    )
    return save_image(image, output_path)


def box_iou_numpy(boxes1, boxes2):
    """计算 numpy 版 IoU。"""
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


def extract_error_cases(prediction_boxes, prediction_labels, ground_truth_boxes, ground_truth_labels, unknown_label, iou_threshold):
    """提取 known/unknown 误判。"""
    errors = {
        'unknown_to_known_prediction_indices': [],
        'unknown_to_known_ground_truth_indices': [],
        'known_to_unknown_prediction_indices': [],
        'known_to_unknown_ground_truth_indices': [],
    }
    iou = box_iou_numpy(prediction_boxes, ground_truth_boxes)
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


def render_error_unknown_to_known(image_np, prediction_boxes, prediction_labels, prediction_scores, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, output_path):
    """渲染 unknown 被判成 known 的错误图。"""
    errors = extract_error_cases(prediction_boxes, prediction_labels, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg['error_match_iou'])
    pred_idx = np.asarray(sorted(set(errors['unknown_to_known_prediction_indices'])), dtype=np.int64)
    gt_idx = np.asarray(sorted(set(errors['unknown_to_known_ground_truth_indices'])), dtype=np.int64)
    image = draw_detection_panel(
        image_np,
        viz_cfg,
        prediction_boxes=prediction_boxes[pred_idx] if len(pred_idx) > 0 else None,
        prediction_labels=prediction_labels[pred_idx] if len(pred_idx) > 0 else None,
        prediction_scores=prediction_scores[pred_idx] if len(pred_idx) > 0 else None,
        ground_truth_boxes=ground_truth_boxes[gt_idx] if len(gt_idx) > 0 else None,
        ground_truth_labels=ground_truth_labels[gt_idx] if len(gt_idx) > 0 else None,
        unknown_label=unknown_label,
    )
    return save_image(image, output_path), int(len(gt_idx))


def render_error_known_to_unknown(image_np, prediction_boxes, prediction_labels, prediction_scores, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg, output_path):
    """渲染 known 被判成 unknown 的错误图。"""
    errors = extract_error_cases(prediction_boxes, prediction_labels, ground_truth_boxes, ground_truth_labels, unknown_label, viz_cfg['error_match_iou'])
    pred_idx = np.asarray(sorted(set(errors['known_to_unknown_prediction_indices'])), dtype=np.int64)
    gt_idx = np.asarray(sorted(set(errors['known_to_unknown_ground_truth_indices'])), dtype=np.int64)
    image = draw_detection_panel(
        image_np,
        viz_cfg,
        prediction_boxes=prediction_boxes[pred_idx] if len(pred_idx) > 0 else None,
        prediction_labels=prediction_labels[pred_idx] if len(pred_idx) > 0 else None,
        prediction_scores=prediction_scores[pred_idx] if len(pred_idx) > 0 else None,
        ground_truth_boxes=ground_truth_boxes[gt_idx] if len(gt_idx) > 0 else None,
        ground_truth_labels=ground_truth_labels[gt_idx] if len(gt_idx) > 0 else None,
        unknown_label=unknown_label,
    )
    return save_image(image, output_path), int(len(gt_idx))


def save_contact_sheet(image_paths, output_path, viz_cfg):
    """生成 PNG contact sheet。"""
    if not image_paths:
        return None
    tile_width = viz_cfg['panel_tile_width']
    tile_height = viz_cfg['panel_tile_height']
    cols = viz_cfg['panel_cols']
    valid_images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB').resize((tile_width, tile_height))
            valid_images.append(image)
        except Exception:
            continue
    if not valid_images:
        return None
    rows = int(math.ceil(len(valid_images) / cols))
    sheet = Image.new('RGB', (cols * tile_width, rows * tile_height), (20, 20, 20))
    for index, image in enumerate(valid_images):
        x = (index % cols) * tile_width
        y = (index // cols) * tile_height
        sheet.paste(image, (x, y))
    sheet.save(output_path)
    return output_path


def layer_order(records):
    """按 decoder layer 排序 attention 记录。"""
    ordered = []
    for record in records:
        name = record['name']
        try:
            layer_id = int(name.split('transformer.decoder.layers.')[1].split('.cross_attn')[0])
        except Exception:
            layer_id = len(ordered)
        ordered.append((layer_id, record))
    ordered.sort(key=lambda item: item[0])
    return ordered


def plot_query_sampling(image_np, records, query_kind, query_index, scores, output_path):
    """绘制单个 query 的 sampling 图。"""
    ordered = layer_order(records)
    if not ordered:
        return None
    n_layers = len(ordered)
    ncols = min(3, n_layers)
    nrows = int(math.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.6 * nrows), squeeze=False, constrained_layout=True)
    h, w = image_np.shape[:2]
    for ax in axes.flat:
        ax.set_visible(False)
    last_level_count = 0
    for ax, (layer_id, record) in zip(axes.flat, ordered):
        ax.set_visible(True)
        ax.imshow(image_np)
        ax.set_axis_off()
        ax.set_title(f'Layer {layer_id}', fontsize=13, pad=8)
        sampling_locations = record['sampling_locations'][0, query_index].numpy()
        attention_weights = record['attention_weights'][0, query_index].numpy()
        reference_points = record['reference_points'][0, query_index].numpy()
        ref_xy = reference_points[:, :2].mean(0) if reference_points.ndim == 2 else reference_points[:2]
        ax.scatter(ref_xy[0] * w, ref_xy[1] * h, c='white', edgecolors='black', s=120, marker='x', linewidths=2.2, zorder=5)
        heads_mean_locations = sampling_locations.mean(0)
        heads_mean_weights = attention_weights.mean(0)
        last_level_count = heads_mean_locations.shape[0]
        for level in range(heads_mean_locations.shape[0]):
            pts = heads_mean_locations[level]
            ws = heads_mean_weights[level]
            color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            sizes = 36.0 + 260.0 * ws
            ax.scatter(xs, ys, s=sizes, c=color, alpha=0.80, edgecolors='black', linewidths=0.45, zorder=4)
    select_score = scores['known_score'][query_index] if query_kind == 'known' else scores['unknown_score'][query_index]
    fig.suptitle(
        f'Query {query_index} | type={query_kind} | select_score={select_score:.3f} | obj={scores["obj_prob"][query_index]:.3f} | unk={scores["unknown_prob"][query_index]:.3f} | max_known={scores["max_known"][query_index]:.3f}',
        fontsize=15,
        y=1.02,
    )
    legend_handles = [Line2D([0], [0], marker='x', color='black', markerfacecolor='white', markersize=10, linewidth=0, label='reference point')]
    for level in range(min(len(LEVEL_COLORS), last_level_count)):
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=LEVEL_COLORS[level], markersize=8, linewidth=0, label=f'feature level {level}'))
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=min(4, len(legend_handles)), frameon=False, fontsize=10)
    fig.text(0.5, -0.055, 'Circle size represents mean sampling attention weight across heads.', ha='center', va='center', fontsize=10)
    return save_svg_figure(fig, output_path, pad_inches=0.08)


def plot_query_gate_curve(query_kind, query_index, ordered_records, scores, output_path):
    """绘制单个 query 的 gate 曲线图。"""
    layers = [record['layer_id'] for record in ordered_records]
    raw_mean = [record['raw_gate'][0, query_index].mean().item() for record in ordered_records]
    eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in ordered_records]
    raw_max = [record['raw_gate'][0, query_index].max().item() for record in ordered_records]
    eff_max = [record['effective_gate'][0, query_index].max().item() for record in ordered_records]
    decay = [record['decay'] for record in ordered_records]
    color = KNOWN_COLOR if query_kind == 'known' else UNKNOWN_COLOR
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].plot(layers, raw_mean, marker='o', linewidth=2.0, c=color, label='raw gate mean')
    axes[0].plot(layers, eff_mean, marker='o', linewidth=2.0, linestyle='--', c='#1E88E5', label='effective gate mean')
    axes[0].plot(layers, decay, marker='s', linewidth=1.6, linestyle=':', c='#6C757D', label='layer decay')
    axes[0].set_xlabel('decoder layer')
    axes[0].set_ylabel('mean gate value')
    axes[0].set_title('ODQE gate mean by layer')
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(layers, raw_max, marker='o', linewidth=2.0, c=color, label='raw gate max')
    axes[1].plot(layers, eff_max, marker='o', linewidth=2.0, linestyle='--', c='#1E88E5', label='effective gate max')
    axes[1].set_xlabel('decoder layer')
    axes[1].set_ylabel('max gate value')
    axes[1].set_title('ODQE gate peak by layer')
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    select_score = scores['known_score'][query_index] if query_kind == 'known' else scores['unknown_score'][query_index]
    fig.suptitle(
        f'Query {query_index} | type={query_kind} | select_score={select_score:.3f} | obj={scores["obj_prob"][query_index]:.3f} | unk={scores["unknown_prob"][query_index]:.3f} | max_known={scores["max_known"][query_index]:.3f}',
        y=1.03,
        fontsize=11,
    )
    return save_svg_figure(fig, output_path)


def plot_query_gate_heatmap(query_kind, query_index, ordered_records, output_path):
    """绘制单个 query 的 gate 热力图。"""
    raw_gate = np.stack([record['raw_gate'][0, query_index].numpy() for record in ordered_records], axis=0)
    eff_gate = np.stack([record['effective_gate'][0, query_index].numpy() for record in ordered_records], axis=0)
    layers = [record['layer_id'] for record in ordered_records]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    im0 = axes[0].imshow(raw_gate, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    axes[0].set_title(f'Raw ODQE gate heatmap | query {query_index} ({query_kind})')
    axes[0].set_xlabel('channel')
    axes[0].set_ylabel('decoder layer')
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(eff_gate, aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
    axes[1].set_title(f'Effective ODQE gate heatmap | query {query_index} ({query_kind})')
    axes[1].set_xlabel('channel')
    axes[1].set_ylabel('decoder layer')
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    return save_svg_figure(fig, output_path)


def boxes_to_abs_xyxy(boxes, image_hw):
    """把 query box 转成绝对坐标。"""
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().cpu())
    scale = torch.tensor([int(image_hw[1]), int(image_hw[0]), int(image_hw[1]), int(image_hw[0])], dtype=boxes_xyxy.dtype)
    return (boxes_xyxy * scale).numpy()


def plot_image_trajectory(ax, image_np, abs_boxes_per_layer, query_kind):
    """在图上绘制 box trajectory。"""
    ax.imshow(image_np)
    ax.set_axis_off()
    centers_x = []
    centers_y = []
    for layer_id, box in enumerate(abs_boxes_per_layer):
        x1, y1, x2, y2 = box
        color = LAYER_COLORS[layer_id % len(LAYER_COLORS)]
        rect = Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), fill=False, linewidth=2.0, edgecolor=color, alpha=0.95)
        ax.add_patch(rect)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        centers_x.append(cx)
        centers_y.append(cy)
        ax.scatter(cx, cy, s=42, c=color, edgecolors='black', linewidths=0.5)
        ax.text(cx + 2, cy + 2, str(layer_id), color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.60, pad=1))
    if len(centers_x) >= 2:
        ax.plot(centers_x, centers_y, linewidth=2.0, color=KNOWN_COLOR if query_kind == 'known' else UNKNOWN_COLOR, alpha=0.9)
    ax.set_title('Query box trajectory on image')


def plot_box_geometry(ax, abs_boxes_per_layer):
    """绘制 box geometry 轨迹。"""
    layer_ids = list(range(len(abs_boxes_per_layer)))
    widths = [max(1e-6, box[2] - box[0]) for box in abs_boxes_per_layer]
    heights = [max(1e-6, box[3] - box[1]) for box in abs_boxes_per_layer]
    areas = [w * h for w, h in zip(widths, heights)]
    centers_x = [0.5 * (box[0] + box[2]) for box in abs_boxes_per_layer]
    centers_y = [0.5 * (box[1] + box[3]) for box in abs_boxes_per_layer]
    ax.plot(layer_ids, centers_x, marker='o', linewidth=2.0, label='center x')
    ax.plot(layer_ids, centers_y, marker='o', linewidth=2.0, label='center y')
    ax.plot(layer_ids, widths, marker='o', linewidth=1.8, linestyle='--', label='width')
    ax.plot(layer_ids, heights, marker='o', linewidth=1.8, linestyle='--', label='height')
    ax.plot(layer_ids, np.sqrt(np.asarray(areas)), marker='o', linewidth=1.8, linestyle=':', label='sqrt(area)')
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('pixels / proxy scale')
    ax.set_title('Box geometry trajectory')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))


def plot_score_trajectory(ax, layer_scores, query_index):
    """绘制 score trajectory。"""
    layer_ids = list(range(len(layer_scores)))
    obj = [layer['obj_prob'][query_index] for layer in layer_scores]
    unk = [layer['unknown_prob'][query_index] for layer in layer_scores]
    max_known = [layer['max_known'][query_index] for layer in layer_scores]
    known_score = [layer['known_score'][query_index] for layer in layer_scores]
    unknown_score = [layer['unknown_score'][query_index] for layer in layer_scores]
    ax.plot(layer_ids, obj, marker='o', linewidth=2.0, label='obj prob')
    ax.plot(layer_ids, unk, marker='o', linewidth=2.0, label='unknown prob')
    ax.plot(layer_ids, max_known, marker='o', linewidth=2.0, label='max known prob')
    ax.plot(layer_ids, known_score, marker='o', linewidth=1.8, linestyle='--', label='known score')
    ax.plot(layer_ids, unknown_score, marker='o', linewidth=1.8, linestyle='--', label='unknown score')
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('score')
    ax.set_ylim(0.0, max(1.05, 1.05 * max(known_score + unknown_score + obj + unk + max_known)))
    ax.set_title('Score trajectory')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))


def plot_gate_trajectory(ax, gate_records, query_index):
    """绘制 gate trajectory。"""
    layer_ids = list(range(len(gate_records)))
    raw_mean = [record['raw_gate'][0, query_index].mean().item() for record in gate_records]
    eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in gate_records]
    raw_max = [record['raw_gate'][0, query_index].max().item() for record in gate_records]
    decay = [record['decay'] for record in gate_records]
    ax.plot(layer_ids, raw_mean, marker='o', linewidth=2.0, label='raw gate mean')
    ax.plot(layer_ids, eff_mean, marker='o', linewidth=2.0, linestyle='--', label='effective gate mean')
    ax.plot(layer_ids, raw_max, marker='o', linewidth=1.8, linestyle=':', label='raw gate max')
    ax.plot(layer_ids, decay, marker='s', linewidth=1.6, linestyle='-.', label='layer decay')
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('gate')
    ax.set_title('ODQE gate trajectory')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18))


def plot_layer_class_trace(ax, layer_scores, query_index):
    """绘制逐层类别 trace。"""
    layer_ids = list(range(len(layer_scores)))
    labels = [int(layer['argmax_known'][query_index]) for layer in layer_scores]
    vals = [float(layer['max_known'][query_index]) for layer in layer_scores]
    ax.plot(layer_ids, vals, marker='o', linewidth=2.0)
    for x, y, label in zip(layer_ids, vals, labels):
        ax.annotate(f'c{label}', (x, y), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('max known prob')
    ax.set_title('Predicted known-class trace')
    ax.grid(alpha=0.2)
    ax.margins(y=0.18)


def plot_summary(ax, query_kind, query_index, layer_scores, gate_records):
    """绘制 query 摘要。"""
    ax.axis('off')
    final = layer_scores[-1]
    summary_lines = [
        f'query type: {query_kind}',
        f'query index: {int(query_index)}',
        f'final obj: {float(final["obj_prob"][query_index]):.3f}',
        f'final unk: {float(final["unknown_prob"][query_index]):.3f}',
        f'final max_known: {float(final["max_known"][query_index]):.3f}',
        f'final known_score: {float(final["known_score"][query_index]):.3f}',
        f'final unknown_score: {float(final["unknown_score"][query_index]):.3f}',
        f'final known class: {int(final["argmax_known"][query_index])}',
    ]
    if gate_records:
        summary_lines += [
            f'final raw gate mean: {float(gate_records[-1]["raw_gate"][0, query_index].mean().item()):.3f}',
            f'final effective gate mean: {float(gate_records[-1]["effective_gate"][0, query_index].mean().item()):.3f}',
        ]
    ax.text(0.02, 0.98, '\n'.join(summary_lines), va='top', ha='left', fontsize=10, family='monospace', linespacing=1.5)
    ax.set_title('Trajectory summary')


def plot_query_trajectory_panel(query_kind, query_index, image_np, layer_scores, gate_records, image_hw, output_path):
    """绘制单个 query trajectory 面板。"""
    abs_boxes_per_layer = []
    for layer in layer_scores:
        box_abs = boxes_to_abs_xyxy(torch.from_numpy(layer['boxes'][query_index:query_index + 1]), image_hw)[0]
        abs_boxes_per_layer.append(box_abs)
    if gate_records:
        mosaic = [['image', 'geometry', 'score'], ['gate', 'class', 'summary']]
        fig = plt.figure(figsize=(16.5, 9.3), constrained_layout=True)
    else:
        mosaic = [['image', 'geometry', 'score'], ['class', 'class', 'summary']]
        fig = plt.figure(figsize=(16.5, 8.4), constrained_layout=True)
    axes = fig.subplot_mosaic(mosaic)
    plot_image_trajectory(axes['image'], image_np, abs_boxes_per_layer, query_kind)
    plot_box_geometry(axes['geometry'], abs_boxes_per_layer)
    plot_score_trajectory(axes['score'], layer_scores, query_index)
    plot_layer_class_trace(axes['class'], layer_scores, query_index)
    plot_summary(axes['summary'], query_kind, query_index, layer_scores, gate_records)
    if gate_records:
        plot_gate_trajectory(axes['gate'], gate_records, query_index)
    fig.suptitle(f'Cross-layer trajectory of query {int(query_index)} ({query_kind})', y=1.02, fontsize=15)
    return save_svg_figure(fig, output_path, pad_inches=0.08)


def plot_selected_query_unknown_gate_overview(selected, layer_scores, gate_records, output_path):
    """绘制已选 query 的 unknown×gate 总览。"""
    if not selected:
        return None
    layer_ids = list(range(len(layer_scores)))
    matrix = []
    row_labels = []
    for query_kind, query_index in selected:
        row = [float(layer_scores[layer_id]['unknown_prob'][query_index]) for layer_id in layer_ids]
        if gate_records:
            row = [row[layer_id] * float(gate_records[layer_id]['effective_gate'][0, query_index].mean().item()) for layer_id in layer_ids]
        matrix.append(row)
        row_labels.append(f'{query_kind[0].upper()}-{int(query_index):03d}')
    matrix = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(1.25 * len(layer_ids) + 2.8, 0.6 * len(row_labels) + 2.6), constrained_layout=True)
    im = ax.imshow(matrix, aspect='auto', cmap='magma', vmin=0.0, vmax=max(1e-6, float(matrix.max())))
    ax.set_xlabel('decoder layer')
    ax.set_ylabel('selected query')
    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title('Unknownness × effective-gate overview')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_svg_figure(fig, output_path, pad_inches=0.04)


def plot_sampling_row(fig, grid, row_index, image_np, attn_records, query_index):
    """绘制 joint panel 的 sampling 行。"""
    h, w = image_np.shape[:2]
    for col_index, attn_record in enumerate(attn_records):
        ax = fig.add_subplot(grid[row_index, col_index])
        ax.imshow(image_np)
        ax.set_axis_off()
        sampling_locations = attn_record['sampling_locations'][0, query_index].numpy()
        attention_weights = attn_record['attention_weights'][0, query_index].numpy()
        reference_points = attn_record['reference_points'][0, query_index].numpy()
        ref_xy = reference_points[:, :2].mean(0) if reference_points.ndim == 2 else reference_points[:2]
        ax.scatter(ref_xy[0] * w, ref_xy[1] * h, c='white', edgecolors='black', s=80, marker='x', linewidths=2)
        heads_mean_locations = sampling_locations.mean(0)
        heads_mean_weights = attention_weights.mean(0)
        for level in range(heads_mean_locations.shape[0]):
            pts = heads_mean_locations[level]
            ws = heads_mean_weights[level]
            color = LEVEL_COLORS[level % len(LEVEL_COLORS)]
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            sizes = 20.0 + 200.0 * ws
            ax.scatter(xs, ys, s=sizes, c=color, alpha=0.75, edgecolors='black', linewidths=0.35)
        ax.set_title(f'Layer {col_index}', fontsize=10)


def plot_joint_panel(query_kind, query_index, image_np, attn_records, gate_records, scores, output_path):
    """绘制 query 的 joint mechanism 面板。"""
    n_layers = len(attn_records)
    if n_layers == 0 or len(gate_records) == 0:
        return None
    color = KNOWN_COLOR if query_kind == 'known' else UNKNOWN_COLOR
    layers = list(range(n_layers))
    raw_mean = [record['raw_gate'][0, query_index].mean().item() for record in gate_records]
    eff_mean = [record['effective_gate'][0, query_index].mean().item() for record in gate_records]
    raw_gate_map = np.stack([record['raw_gate'][0, query_index].numpy() for record in gate_records], axis=0)
    eff_gate_map = np.stack([record['effective_gate'][0, query_index].numpy() for record in gate_records], axis=0)
    fig = plt.figure(figsize=(4.0 * n_layers, 12.0))
    grid = GridSpec(3, n_layers, figure=fig, height_ratios=[1.2, 0.55, 0.65])
    plot_sampling_row(fig, grid, 0, image_np, attn_records, query_index)
    curve_ax = fig.add_subplot(grid[1, : max(2, n_layers // 2)])
    curve_ax.plot(layers, raw_mean, marker='o', linewidth=2.0, c=color, label='raw gate mean')
    curve_ax.plot(layers, eff_mean, marker='o', linewidth=2.0, linestyle='--', c='#1E88E5', label='effective gate mean')
    curve_ax.set_xlabel('decoder layer')
    curve_ax.set_ylabel('gate mean')
    curve_ax.set_title('ODQE gate dynamics')
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, fontsize=8)
    attn_ax = fig.add_subplot(grid[1, max(2, n_layers // 2):])
    attn_mean = [record['attention_weights'][0, query_index].mean().item() for record in attn_records]
    attn_max = [record['attention_weights'][0, query_index].max().item() for record in attn_records]
    attn_ax.plot(layers, attn_mean, marker='o', linewidth=2.0, c='#6C757D', label='attention mean')
    attn_ax.plot(layers, attn_max, marker='o', linewidth=2.0, linestyle='--', c='#FB8C00', label='attention max')
    attn_ax.set_xlabel('decoder layer')
    attn_ax.set_ylabel('attention weight')
    attn_ax.set_title('Sampling-weight dynamics')
    attn_ax.grid(alpha=0.2)
    attn_ax.legend(frameon=False, fontsize=8)
    raw_ax = fig.add_subplot(grid[2, : max(2, n_layers // 2)])
    raw_im = raw_ax.imshow(raw_gate_map, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    raw_ax.set_title('Raw gate heatmap')
    raw_ax.set_xlabel('channel')
    raw_ax.set_ylabel('layer')
    raw_ax.set_yticks(range(n_layers))
    raw_ax.set_yticklabels(layers)
    fig.colorbar(raw_im, ax=raw_ax, fraction=0.046, pad=0.04)
    eff_ax = fig.add_subplot(grid[2, max(2, n_layers // 2):])
    eff_im = eff_ax.imshow(eff_gate_map, aspect='auto', cmap='magma', vmin=0.0, vmax=1.0)
    eff_ax.set_title('Effective gate heatmap')
    eff_ax.set_xlabel('channel')
    eff_ax.set_ylabel('layer')
    eff_ax.set_yticks(range(n_layers))
    eff_ax.set_yticklabels(layers)
    fig.colorbar(eff_im, ax=eff_ax, fraction=0.046, pad=0.04)
    select_score = scores['known_score'][query_index] if query_kind == 'known' else scores['unknown_score'][query_index]
    fig.suptitle(
        f'Query {query_index} | type={query_kind} | select_score={select_score:.3f} | obj={scores["obj_prob"][query_index]:.3f} | unk={scores["unknown_prob"][query_index]:.3f} | max_known={scores["max_known"][query_index]:.3f}',
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return save_svg_figure(fig, output_path)


def draw_case_tile(dataset, entry, category, tile_size=420):
    """渲染代表案例 tile。"""
    image, target = dataset[entry['sample_index']]
    image_hw = target['size'].tolist() if 'size' in target else image.shape[-2:]
    image_np = to_numpy_image(image, image_hw)
    h, w = image_np.shape[:2]
    box = np.array(entry['box_cxcywh'], dtype=np.float32)[None, :]
    box_xyxy = box_ops.box_cxcywh_to_xyxy(torch.from_numpy(box)).numpy()[0]
    box_xyxy[[0, 2]] *= w
    box_xyxy[[1, 3]] *= h
    pil = Image.fromarray(image_np).convert('RGB').resize((tile_size, tile_size))
    draw = ImageDraw.Draw(pil)
    sx = tile_size / max(float(w), 1.0)
    sy = tile_size / max(float(h), 1.0)
    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=CATEGORY_COLORS[category], width=4)
    return pil


def fmt_gt_iou(value):
    """格式化 gt_iou 文本。"""
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.3f}'
    except Exception:
        return 'n/a'


def save_case_contact_sheet_svg(dataset, entries, category, output_path):
    """生成代表案例 SVG contact sheet。"""
    if not entries:
        return None
    items = []
    for entry in entries:
        tile = draw_case_tile(dataset, entry, category)
        items.append(
            {
                'pil_image': tile,
                'label_lines': [
                    f'{category} | sample {entry["sample_index"]}',
                    f'img {entry["image_id"]} q{entry["query_index"]} s={entry["category_score"]:.3f}',
                    f'obj={entry["obj_prob"]:.3f} unk={entry["unknown_prob"]:.3f} gt_iou={fmt_gt_iou(entry.get("gt_overlap"))}',
                ],
            }
        )
    return write_gallery_svg(items, output_path, title=f'{category} representative cases', mode='sampling', cols=3, tile_width=420)
