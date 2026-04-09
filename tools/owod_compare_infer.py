#!/usr/bin/env python3
"""Compare baseline and ours OWOD checkpoints on one image or a folder.

Place this script at the repo root (or run with PYTHONPATH pointing to the repo)
so imports like `from models import build_model` resolve correctly.

Main features:
- single-image or folder inference
- baseline vs ours comparison
- optional VOC XML annotations for case mining
- explicit legend for GT known/unknown, predicted known/unknown
- exports per-image panels + CSV summary

Typical usage:
python tools/owod_compare_infer.py \
  --weights_base path/to/prob_baseline.pth \
  --weights_ours path/to/ours_uod.pth \
  --input_dir path/to/images \
  --ann_dir path/to/xmls \
  --output_dir runs/qual_compare \
  --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

# Repo imports. The script is intended to live/run from repo root.
from main_open_world import get_args_parser
from models import build_model
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import (
    VOC_COCO_CLASS_NAMES,
    VOC_CLASS_NAMES_COCOFIED,
    BASE_VOC_CLASS_NAMES,
)
import util.misc as utils

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLORS = {
    "gt_known": "#EEDD44",
    "gt_unknown": "#EE7733",
    "pred_known": "#009988",
    "pred_unknown": "#CC3311",
    "candidate_unknown": "#33BBEE",
    "text_bg": "#111111",
    "text_fg": "#FFFFFF",
}

LEGEND_ITEMS = [
    ("GT-K", COLORS["gt_known"], "ground-truth known object"),
    ("GT-U", COLORS["gt_unknown"], "ground-truth unknown object"),
    ("P-K", COLORS["pred_known"], "predicted known object"),
    ("P-U", COLORS["pred_unknown"], "predicted unknown object"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def load_checkpoint_args(ckpt_path: str, device: str, cli_overrides: argparse.Namespace) -> Tuple[argparse.Namespace, Dict]:
    parser = argparse.ArgumentParser(parents=[get_args_parser()], add_help=False)
    args = parser.parse_args([])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    if isinstance(saved_args, dict):
        for k, v in saved_args.items():
            setattr(args, k, v)

    for key in [
        "device", "dataset", "PREV_INTRODUCED_CLS", "CUR_INTRODUCED_CLS",
        "num_classes", "model_type", "obj_temp", "uod_postprocess_unknown_scale",
        "uod_postprocess_unknown_ratio", "uod_enable_unknown", "uod_enable_pseudo",
        "uod_enable_batch_dynamic", "uod_enable_cls_soft_attn", "uod_enable_odqe",
        "uod_enable_decorr",
    ]:
        value = getattr(cli_overrides, key, None)
        if value is not None:
            setattr(args, key, value)

    args.device = device
    args.eval = True
    args.output_dir = ""
    return args, ckpt


class ModelBundle:
    def __init__(self, name: str, ckpt_path: str, args: argparse.Namespace, ckpt: Dict, device: torch.device):
        self.name = name
        self.ckpt_path = ckpt_path
        self.args = args
        self.device = device
        self.model, _, self.postprocessors, _ = build_model(args, mode=getattr(args, "model_type", "prob"))
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.to(device)
        self.model.eval()
        self.missing = list(missing)
        self.unexpected = list(unexpected)
        self.class_names = VOC_COCO_CLASS_NAMES[getattr(args, "dataset", "OWDETR")]
        self.unk_label = int(getattr(args, "num_classes", len(self.class_names)) - 1)

    @torch.no_grad()
    def infer(self, image: Image.Image) -> Dict:
        _, transform = make_coco_transforms("test")
        w, h = image.size
        dummy_target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "image_id": torch.tensor([0], dtype=torch.int64),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "iscrowd": torch.zeros((0,), dtype=torch.uint8),
        }
        img_tensor, target = transform(image, dummy_target)
        samples = utils.nested_tensor_from_tensor_list([img_tensor]).to(self.device)
        outputs = self.model(samples)
        target_sizes = torch.as_tensor([target["orig_size"].tolist()], dtype=torch.float32, device=self.device)
        pred = self.postprocessors["bbox"](outputs, target_sizes)[0]
        return {
            "outputs": outputs,
            "pred": pred,
            "size": target["orig_size"].tolist(),
        }


def to_numpy_image(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def parse_voc_xml(xml_path: Path, class_names: Sequence[str], num_known: int, unk_label: int) -> Dict[str, np.ndarray]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: List[List[float]] = []
    labels: List[int] = []
    known_names = set(class_names[:num_known])
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name in VOC_CLASS_NAMES_COCOFIED:
            cls_name = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls_name)]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text) - 1.0
        ymin = float(bbox.find("ymin").text) - 1.0
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        if cls_name in known_names:
            labels.append(class_names.index(cls_name))
        else:
            labels.append(unk_label)
    return {
        "boxes": np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
    }


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1 = boxes1.astype(np.float32)
    b2 = boxes2.astype(np.float32)
    area1 = np.clip(b1[:, 2] - b1[:, 0], 0, None) * np.clip(b1[:, 3] - b1[:, 1], 0, None)
    area2 = np.clip(b2[:, 2] - b2[:, 0], 0, None) * np.clip(b2[:, 3] - b2[:, 1], 0, None)
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def label_name(label: int, class_names: Sequence[str], unk_label: int) -> str:
    if int(label) == int(unk_label):
        return "unknown"
    if 0 <= int(label) < len(class_names):
        return str(class_names[int(label)])
    return f"cls{int(label)}"


def filter_predictions(pred: Dict, score_thresh: float) -> Dict[str, np.ndarray]:
    boxes = pred["boxes"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    keep = scores >= float(score_thresh)
    return {
        "boxes": boxes[keep],
        "labels": labels[keep],
        "scores": scores[keep],
    }


def split_known_unknown(pred_np: Dict[str, np.ndarray], unk_label: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    mask_unk = pred_np["labels"] == int(unk_label)
    mask_known = ~mask_unk
    known = {k: v[mask_known] for k, v in pred_np.items()}
    unk = {k: v[mask_unk] for k, v in pred_np.items()}
    return known, unk


def add_legend(draw: ImageDraw.ImageDraw, x: int = 8, y: int = 8) -> None:
    box_w = 18
    line_h = 18
    panel_w = 290
    panel_h = 10 + line_h * len(LEGEND_ITEMS) + 6
    draw.rounded_rectangle([x, y, x + panel_w, y + panel_h], radius=4, fill=hex_to_rgb(COLORS["text_bg"]))
    for i, (tag, color, desc) in enumerate(LEGEND_ITEMS):
        yy = y + 6 + i * line_h
        draw.rectangle([x + 8, yy + 2, x + 8 + box_w, yy + 2 + 10], outline=hex_to_rgb(color), width=3)
        draw.text((x + 36, yy), f"{tag}: {desc}", fill=hex_to_rgb(COLORS["text_fg"]))


def draw_boxes(
    image_np: np.ndarray,
    class_names: Sequence[str],
    unk_label: int,
    pred: Optional[Dict[str, np.ndarray]] = None,
    gt: Optional[Dict[str, np.ndarray]] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    extra_caption: Optional[str] = None,
) -> np.ndarray:
    img = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(img)

    if gt is not None and len(gt["boxes"]) > 0:
        for box, label in zip(gt["boxes"], gt["labels"]):
            color = COLORS["gt_unknown"] if int(label) == int(unk_label) else COLORS["gt_known"]
            x1, y1, x2, y2 = [float(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=hex_to_rgb(color), width=2)
            tag = "GT-U" if int(label) == int(unk_label) else "GT-K"
            draw.text((x1 + 2, max(0.0, y1 - 14)), f"{tag}:{label_name(int(label), class_names, unk_label)}", fill=hex_to_rgb(color))

    if pred is not None and len(pred["boxes"]) > 0:
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            is_unk = int(label) == int(unk_label)
            color = COLORS["pred_unknown"] if is_unk else COLORS["pred_known"]
            x1, y1, x2, y2 = [float(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=hex_to_rgb(color), width=3)
            tag = "P-U" if is_unk else "P-K"
            draw.text((x1 + 2, y1 + 2), f"{tag}:{label_name(int(label), class_names, unk_label)} {float(score):.2f}", fill=hex_to_rgb(color))

    if title:
        draw.rounded_rectangle([6, 6, 6 + 12 * max(8, len(title)), 26], radius=4, fill=hex_to_rgb(COLORS["text_bg"]))
        draw.text((10, 10), title, fill=hex_to_rgb(COLORS["text_fg"]))
    if extra_caption:
        draw.rounded_rectangle([6, 30, 6 + 12 * max(8, len(extra_caption)), 50], radius=4, fill=hex_to_rgb(COLORS["text_bg"]))
        draw.text((10, 34), extra_caption, fill=hex_to_rgb(COLORS["text_fg"]))
    if show_legend:
        add_legend(draw, x=8, y=56 if title else 8)
    return np.asarray(img)


def make_panel(images_with_titles: Sequence[Tuple[np.ndarray, str]], out_path: Path, cols: int = 2, tile_hw: Tuple[int, int] = (520, 360)) -> None:
    tiles: List[Image.Image] = []
    for arr, title in images_with_titles:
        img = Image.fromarray(arr).convert("RGB").resize(tile_hw)
        canvas = Image.new("RGB", tile_hw, (18, 18, 18))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle([8, 8, 8 + 14 * max(6, len(title)), 28], radius=4, fill=(17, 17, 17))
        d.text((12, 12), title, fill=(255, 255, 255))
        tiles.append(canvas)
    rows = int(math.ceil(len(tiles) / cols))
    panel = Image.new("RGB", (cols * tile_hw[0], rows * tile_hw[1]), (12, 12, 12))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_hw[0]
        y = (idx // cols) * tile_hw[1]
        panel.paste(tile, (x, y))
    panel.save(out_path)


def collect_case_stats(gt: Dict[str, np.ndarray], base_pred: Dict[str, np.ndarray], ours_pred: Dict[str, np.ndarray], unk_label: int, iou_thr: float) -> Dict:
    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]
    unk_gt_idx = np.where(gt_labels == int(unk_label))[0]
    known_gt_idx = np.where(gt_labels != int(unk_label))[0]

    base_iou = box_iou_np(base_pred["boxes"], gt_boxes)
    ours_iou = box_iou_np(ours_pred["boxes"], gt_boxes)

    def any_hit(pred: Dict[str, np.ndarray], iou_mat: np.ndarray, gt_index: int, want_unknown: Optional[bool] = None) -> bool:
        if len(pred["boxes"]) == 0:
            return False
        keep = np.ones(len(pred["boxes"]), dtype=bool)
        if want_unknown is True:
            keep = pred["labels"] == int(unk_label)
        elif want_unknown is False:
            keep = pred["labels"] != int(unk_label)
        if keep.sum() == 0:
            return False
        return float(iou_mat[keep, gt_index].max(initial=0.0)) >= float(iou_thr)

    per_unknown = []
    for gt_idx in unk_gt_idx.tolist():
        ours_unknown_hit = any_hit(ours_pred, ours_iou, gt_idx, want_unknown=True)
        base_any_hit = any_hit(base_pred, base_iou, gt_idx, want_unknown=None)
        base_known_hit = any_hit(base_pred, base_iou, gt_idx, want_unknown=False)
        base_unknown_hit = any_hit(base_pred, base_iou, gt_idx, want_unknown=True)
        if ours_unknown_hit and not base_any_hit:
            case = "baseline_missed_ours_detected"
        elif ours_unknown_hit and base_known_hit:
            case = "baseline_known_ours_corrected"
        elif ours_unknown_hit and base_unknown_hit:
            case = "both_detect_unknown"
        elif (not ours_unknown_hit) and base_known_hit:
            case = "both_fail_base_known"
        else:
            case = "other"
        per_unknown.append({
            "gt_idx": gt_idx,
            "case": case,
            "ours_unknown_hit": bool(ours_unknown_hit),
            "base_any_hit": bool(base_any_hit),
            "base_known_hit": bool(base_known_hit),
            "base_unknown_hit": bool(base_unknown_hit),
        })

    # A-OSE-style, prediction-driven errors.
    u2k_pred_idx = []
    u2k_gt_idx = []
    if len(base_pred["boxes"]) > 0 and len(unk_gt_idx) > 0:
        pred_known = np.where(base_pred["labels"] != int(unk_label))[0]
        for p in pred_known.tolist():
            overlaps = base_iou[p, unk_gt_idx]
            if overlaps.size > 0 and float(overlaps.max(initial=0.0)) >= float(iou_thr):
                u2k_pred_idx.append(p)
                u2k_gt_idx.append(int(unk_gt_idx[int(np.argmax(overlaps))]))

    k2u_pred_idx = []
    k2u_gt_idx = []
    if len(ours_pred["boxes"]) > 0 and len(known_gt_idx) > 0:
        pred_unk = np.where(ours_pred["labels"] == int(unk_label))[0]
        for p in pred_unk.tolist():
            overlaps = ours_iou[p, known_gt_idx]
            if overlaps.size > 0 and float(overlaps.max(initial=0.0)) >= float(iou_thr):
                k2u_pred_idx.append(p)
                k2u_gt_idx.append(int(known_gt_idx[int(np.argmax(overlaps))]))

    return {
        "num_gt_unknown": int(len(unk_gt_idx)),
        "num_gt_known": int(len(known_gt_idx)),
        "per_unknown": per_unknown,
        "baseline_missed_ours_detected": int(sum(x["case"] == "baseline_missed_ours_detected" for x in per_unknown)),
        "baseline_known_ours_corrected": int(sum(x["case"] == "baseline_known_ours_corrected" for x in per_unknown)),
        "both_detect_unknown": int(sum(x["case"] == "both_detect_unknown" for x in per_unknown)),
        "u2k_pred_idx": u2k_pred_idx,
        "u2k_gt_idx": u2k_gt_idx,
        "k2u_pred_idx": k2u_pred_idx,
        "k2u_gt_idx": k2u_gt_idx,
    }


def build_error_overlay(image_np: np.ndarray, gt: Dict[str, np.ndarray], pred: Dict[str, np.ndarray], gt_indices: Sequence[int], pred_indices: Sequence[int], class_names: Sequence[str], unk_label: int, title: str) -> np.ndarray:
    sub_gt = {
        "boxes": gt["boxes"][list(gt_indices)] if len(gt_indices) else np.zeros((0, 4), dtype=np.float32),
        "labels": gt["labels"][list(gt_indices)] if len(gt_indices) else np.zeros((0,), dtype=np.int64),
    }
    sub_pred = {
        "boxes": pred["boxes"][list(pred_indices)] if len(pred_indices) else np.zeros((0, 4), dtype=np.float32),
        "labels": pred["labels"][list(pred_indices)] if len(pred_indices) else np.zeros((0,), dtype=np.int64),
        "scores": pred["scores"][list(pred_indices)] if len(pred_indices) else np.zeros((0,), dtype=np.float32),
    }
    return draw_boxes(image_np, class_names, unk_label, pred=sub_pred, gt=sub_gt, title=title, show_legend=True)


def save_prediction_csv(csv_path: Path, pred_np: Dict[str, np.ndarray], class_names: Sequence[str], unk_label: int) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "label_name", "score", "x1", "y1", "x2", "y2"])
        for box, label, score in zip(pred_np["boxes"], pred_np["labels"], pred_np["scores"]):
            writer.writerow([int(label), label_name(int(label), class_names, unk_label), float(score), *[float(v) for v in box]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare OWOD baseline vs ours on image(s)")
    ap.add_argument("--weights_base", required=True)
    ap.add_argument("--weights_ours", required=True)
    ap.add_argument("--image_path", default=None, help="single image")
    ap.add_argument("--input_dir", default=None, help="image folder")
    ap.add_argument("--ann_path", default=None, help="single VOC xml path")
    ap.add_argument("--ann_dir", default=None, help="VOC xml folder matched by stem")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_thresh", type=float, default=0.00, help="visualization score threshold")
    ap.add_argument("--iou_thr", type=float, default=0.50, help="case mining IoU threshold")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--PREV_INTRODUCED_CLS", type=int, default=None)
    ap.add_argument("--CUR_INTRODUCED_CLS", type=int, default=None)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--model_type", default=None)
    ap.add_argument("--obj_temp", type=float, default=None)
    ap.add_argument("--uod_postprocess_unknown_scale", type=float, default=None)
    ap.add_argument("--uod_postprocess_unknown_ratio", type=float, default=None)
    ap.add_argument("--uod_enable_unknown", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--uod_enable_pseudo", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--uod_enable_batch_dynamic", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--uod_enable_cls_soft_attn", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--uod_enable_odqe", type=lambda x: x.lower() == "true", default=None)
    ap.add_argument("--uod_enable_decorr", type=lambda x: x.lower() == "true", default=None)
    args = ap.parse_args()

    if not args.image_path and not args.input_dir:
        ap.error("one of --image_path or --input_dir is required")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    base_args, base_ckpt = load_checkpoint_args(args.weights_base, args.device, args)
    ours_args, ours_ckpt = load_checkpoint_args(args.weights_ours, args.device, args)

    device = torch.device(args.device)
    base = ModelBundle("baseline", args.weights_base, base_args, base_ckpt, device)
    ours = ModelBundle("ours", args.weights_ours, ours_args, ours_ckpt, device)

    # Favor ours metadata for class names / unknown label, but check consistency.
    class_names = ours.class_names
    unk_label = ours.unk_label
    num_known = int(getattr(ours.args, "PREV_INTRODUCED_CLS", 0) + getattr(ours.args, "CUR_INTRODUCED_CLS", 0))

    image_root = Path(args.image_path) if args.image_path else Path(args.input_dir)
    images = list_images(image_root)
    if not images:
        raise FileNotFoundError(f"No images found under: {image_root}")

    summary_rows: List[Dict] = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        image_np = to_numpy_image(image)
        base_out = base.infer(image)
        ours_out = ours.infer(image)
        base_np = filter_predictions(base_out["pred"], score_thresh=args.score_thresh)
        ours_np = filter_predictions(ours_out["pred"], score_thresh=args.score_thresh)
        base_known, base_unk = split_known_unknown(base_np, unk_label)
        ours_known, ours_unk = split_known_unknown(ours_np, unk_label)

        gt = None
        xml_path = None
        if args.ann_path:
            xml_path = Path(args.ann_path)
        elif args.ann_dir:
            cand = Path(args.ann_dir) / f"{img_path.stem}.xml"
            if cand.exists():
                xml_path = cand
        if xml_path is not None and xml_path.exists():
            gt = parse_voc_xml(xml_path, class_names, num_known=num_known, unk_label=unk_label)

        stem_dir = output_dir / img_path.stem
        ensure_dir(stem_dir)
        Image.fromarray(draw_boxes(image_np, class_names, unk_label, pred=None, gt=gt, title="Ground Truth", show_legend=True)).save(stem_dir / "gt.png") if gt is not None else None
        Image.fromarray(draw_boxes(image_np, class_names, unk_label, pred=base_np, gt=gt, title="Baseline: all predictions", show_legend=True)).save(stem_dir / "baseline_all.png")
        Image.fromarray(draw_boxes(image_np, class_names, unk_label, pred=base_unk, gt=gt, title="Baseline: unknown-only", show_legend=True)).save(stem_dir / "baseline_unknown_only.png")
        Image.fromarray(draw_boxes(image_np, class_names, unk_label, pred=ours_np, gt=gt, title="Ours: all predictions", show_legend=True)).save(stem_dir / "ours_all.png")
        Image.fromarray(draw_boxes(image_np, class_names, unk_label, pred=ours_unk, gt=gt, title="Ours: unknown-only", show_legend=True)).save(stem_dir / "ours_unknown_only.png")

        panel_images = []
        if gt is not None:
            panel_images.append((draw_boxes(image_np, class_names, unk_label, gt=gt, title="Ground Truth", show_legend=True), "Ground Truth"))
        panel_images.extend([
            (draw_boxes(image_np, class_names, unk_label, pred=base_np, gt=gt, title="Baseline: all predictions", show_legend=True), "Baseline all"),
            (draw_boxes(image_np, class_names, unk_label, pred=base_unk, gt=gt, title="Baseline: unknown-only", show_legend=True), "Baseline unknown-only"),
            (draw_boxes(image_np, class_names, unk_label, pred=ours_np, gt=gt, title="Ours: all predictions", show_legend=True), "Ours all"),
            (draw_boxes(image_np, class_names, unk_label, pred=ours_unk, gt=gt, title="Ours: unknown-only", show_legend=True), "Ours unknown-only"),
        ])
        make_panel(panel_images, stem_dir / "panel_compare.png", cols=2)

        case_stats = None
        if gt is not None:
            case_stats = collect_case_stats(gt, base_np, ours_np, unk_label=unk_label, iou_thr=args.iou_thr)
            err_u2k = build_error_overlay(
                image_np, gt, base_np,
                case_stats["u2k_gt_idx"], case_stats["u2k_pred_idx"],
                class_names, unk_label,
                title="A-OSE style error: Unknown GT overlapped by known prediction",
            )
            err_k2u = build_error_overlay(
                image_np, gt, ours_np,
                case_stats["k2u_gt_idx"], case_stats["k2u_pred_idx"],
                class_names, unk_label,
                title="Known -> Unknown error: Known GT overlapped by unknown prediction",
            )
            Image.fromarray(err_u2k).save(stem_dir / "error_unknown_to_known.png")
            Image.fromarray(err_k2u).save(stem_dir / "error_known_to_unknown.png")
            make_panel([
                (err_u2k, "A-OSE style U->K"),
                (err_k2u, "Known->Unknown"),
            ], stem_dir / "panel_errors.png", cols=2)
            with open(stem_dir / "cases.json", "w", encoding="utf-8") as f:
                json.dump(case_stats, f, indent=2)

        save_prediction_csv(stem_dir / "baseline_predictions.csv", base_np, class_names, unk_label)
        save_prediction_csv(stem_dir / "ours_predictions.csv", ours_np, class_names, unk_label)

        summary_rows.append({
            "image": img_path.name,
            "num_base_pred": int(len(base_np["boxes"])),
            "num_base_unknown_pred": int(len(base_unk["boxes"])),
            "num_ours_pred": int(len(ours_np["boxes"])),
            "num_ours_unknown_pred": int(len(ours_unk["boxes"])),
            "num_gt_known": int(case_stats["num_gt_known"]) if case_stats else "",
            "num_gt_unknown": int(case_stats["num_gt_unknown"]) if case_stats else "",
            "baseline_missed_ours_detected": int(case_stats["baseline_missed_ours_detected"]) if case_stats else "",
            "baseline_known_ours_corrected": int(case_stats["baseline_known_ours_corrected"]) if case_stats else "",
            "both_detect_unknown": int(case_stats["both_detect_unknown"]) if case_stats else "",
            "u2k_count": int(len(case_stats["u2k_pred_idx"])) if case_stats else "",
            "k2u_count": int(len(case_stats["k2u_pred_idx"])) if case_stats else "",
        })

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    meta = {
        "baseline_ckpt": args.weights_base,
        "ours_ckpt": args.weights_ours,
        "device": args.device,
        "score_thresh": args.score_thresh,
        "iou_thr": args.iou_thr,
        "class_names_count": len(class_names),
        "unknown_label": unk_label,
        "num_known_for_gt_mapping": num_known,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
