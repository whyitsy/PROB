import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_manifest(input_dir: Path) -> List[Dict[str, str]]:
    manifest = input_dir / 'manifest.jsonl'
    rows = []
    with manifest.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_meta(meta_path: Path) -> Dict:
    return json.loads(meta_path.read_text(encoding='utf-8'))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _select_boxes(boxes: np.ndarray, mask: np.ndarray, max_boxes: int) -> np.ndarray:
    idx = np.where(mask.astype(bool))[0]
    if max_boxes > 0:
        idx = idx[:max_boxes]
    return boxes[idx]


def _draw_boxes(ax, boxes: np.ndarray, color: str, label: str, linewidth: float = 1.5, linestyle: str = '-', alpha: float = 0.9):
    if boxes is None or len(boxes) == 0:
        return
    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), fill=False,
                             edgecolor=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        ax.add_patch(rect)
    ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)


def _render_overlay(entry: Dict[str, str], output_dir: Path, max_boxes: int = 150):
    arr = np.load(entry['npz'])
    meta = _load_meta(Path(entry['meta']))
    image_path = meta.get('image_path', '')
    if not image_path or not Path(image_path).exists():
        return None

    image = Image.open(image_path).convert('RGB')
    boxes = arr['pred_boxes_xyxy']
    gt_boxes = arr['gt_boxes_xyxy']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    axes = axes.reshape(2, 2)

    # A: raw unmatched
    ax = axes[0, 0]
    ax.imshow(image)
    _draw_boxes(ax, gt_boxes, 'lime', 'GT', linewidth=2.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['unmatched_mask'], max_boxes), 'gray', 'Unmatched', linewidth=0.8, alpha=0.6)
    ax.set_title(f"A. Raw unmatched | image_id={meta['image_id']}")
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)

    # B: GT + geometry
    ax = axes[0, 1]
    ax.imshow(image)
    _draw_boxes(ax, gt_boxes, 'lime', 'GT', linewidth=2.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['drop_by_gt_mask'], max_boxes), 'red', 'Drop by GT', linewidth=1.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['drop_by_geom_mask'], max_boxes), 'orange', 'Drop by geom', linewidth=1.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['keep_after_geom_mask'], max_boxes), 'royalblue', 'Keep after geom', linewidth=0.9)
    ax.set_title('B. GT-overlap + geometry filtering')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)

    # C: threshold filtering
    ax = axes[1, 0]
    ax.imshow(image)
    _draw_boxes(ax, gt_boxes, 'lime', 'GT', linewidth=2.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['drop_by_unk_min_mask'], max_boxes), 'gold', 'Drop by unk_min', linewidth=1.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['drop_by_pos_thresh_mask'], max_boxes), 'magenta', 'Drop by pos_thresh', linewidth=1.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['drop_by_known_reject_mask'], max_boxes), 'cyan', 'Drop by known_reject', linewidth=1.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['pos_candidates_mask'], max_boxes), 'royalblue', 'Pos candidates', linewidth=1.1)
    ax.set_title(f"C. Threshold filtering | pos_thresh={meta['pos_thresh']:.4f}")
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)

    # D: final role
    ax = axes[1, 1]
    ax.imshow(image)
    _draw_boxes(ax, gt_boxes, 'deepskyblue', 'GT', linewidth=2.0)
    _draw_boxes(ax, _select_boxes(boxes, arr['selected_pos_mask'], max_boxes), 'lime', 'Selected pos', linewidth=2.2)
    _draw_boxes(ax, _select_boxes(boxes, arr['selected_neg_mask'], max_boxes), 'red', 'Selected neg', linewidth=2.2)
    _draw_boxes(ax, _select_boxes(boxes, arr['ignore_mask'], max_boxes), 'gray', 'Ignore', linewidth=1.0, linestyle='--')
    ax.set_title('D. Final mining result')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"{meta['stem']}_overlay.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _render_histograms(entry: Dict[str, str], output_dir: Path):
    arr = np.load(entry['npz'])
    meta = _load_meta(Path(entry['meta']))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.reshape(2, 2)

    specs = [
        ('energy', 'Energy', meta['pos_thresh']),
        ('unknown_prob', 'Unknown probability', None),
        ('known_max', 'Known max probability', None),
        ('conf', 'Composite confidence', None),
    ]
    groups = [
        ('unmatched_mask', 'unmatched'),
        ('keep_after_geom_mask', 'keep_after_geom'),
        ('pos_candidates_mask', 'pos_candidates'),
        ('selected_pos_mask', 'selected_pos'),
    ]

    for ax, (key, title, vline) in zip(axes.flat, specs):
        values = arr[key]
        for mask_key, label in groups:
            mask = arr[mask_key].astype(bool)
            if key == 'conf':
                mask = mask & (values >= 0)
            vals = values[mask]
            if vals.size == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.45, label=label)
        if vline is not None:
            ax.axvline(float(vline), linestyle='--', linewidth=1.5, color='black', label='pos_thresh')
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"{meta['stem']}_hist.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _render_aggregate(summary_csv: Path, output_dir: Path):
    import csv
    rows = []
    with summary_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    keys = [
        'drop_by_gt', 'drop_by_geom', 'drop_by_unk_min',
        'drop_by_pos_thresh', 'drop_by_known_reject', 'drop_by_candidate_nms',
        'pos_candidates', 'selected_pos', 'selected_neg', 'ignore'
    ]
    sums = [sum(int(float(r[k])) for r in rows) for k in keys]

    fig = plt.figure(figsize=(14, 6), dpi=150)
    ax = fig.add_subplot(111)
    x = np.arange(len(keys))
    ax.bar(x, sums)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha='right')
    ax.set_title('Aggregate mining counts across saved images')
    fig.tight_layout()
    out_path = output_dir / 'aggregate_counts.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser('Render UOD mining debug visualizations')
    parser.add_argument('--input_dir', required=True, help='Directory created by uod_collect_mining_debug.py')
    parser.add_argument('--output_dir', required=True, help='Directory to save visualizations')
    parser.add_argument('--max_images', type=int, default=100, help='Max images to render')
    parser.add_argument('--max_draw_boxes_per_group', type=int, default=150, help='Max boxes drawn for each group on overlay panels')
    parser.add_argument('--render_hist', action='store_true', help='Also render histogram panels per image')
    parser.add_argument('--render_overlay', action='store_true', help='Render overlay panels per image')
    args = parser.parse_args()

    if not args.render_hist and not args.render_overlay:
        args.render_hist = True
        args.render_overlay = True

    input_dir = Path(args.input_dir)
    output_dir = _ensure_dir(Path(args.output_dir))
    overlays_dir = _ensure_dir(output_dir / 'overlays')
    hists_dir = _ensure_dir(output_dir / 'histograms')

    entries = _load_manifest(input_dir)
    rendered = 0
    for entry in entries:
        if args.max_images > 0 and rendered >= args.max_images:
            break
        if args.render_overlay:
            _render_overlay(entry, overlays_dir, max_boxes=args.max_draw_boxes_per_group)
        if args.render_hist:
            _render_histograms(entry, hists_dir)
        rendered += 1
        if rendered % 20 == 0:
            print(f'[INFO] rendered {rendered} images')

    _render_aggregate(input_dir / 'image_summary.csv', output_dir)
    print(f'[OK] rendered {rendered} images to {output_dir}')


if __name__ == '__main__':
    main()
