import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from tools.uod_debug_common import (
    build_model_bundle,
    build_repo_parser,
    checkpoint_load_to_model,
    ensure_dir,
    save_json,
    to_device_targets,
)
from tools.uod_collect_mining_debug import _bool_mask_from_indices, _write_npz, replay_uod_mining_debug
from tools.uod_render_mining_debug import _render_histograms, _render_overlay
from util.misc import nested_tensor_from_tensor_list
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES


def _parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    out = []
    for part in text.replace('\n', ',').split(','):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _resolve_image_paths(image_paths: Sequence[str], image_dir: str, pattern: str) -> List[Path]:
    out: List[Path] = []
    for p in image_paths:
        pp = Path(p)
        if pp.exists() and pp.is_file():
            out.append(pp)
    if image_dir:
        img_dir = Path(image_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f'image_dir not found: {img_dir}')
        pats = [x.strip() for x in pattern.split(',') if x.strip()]
        found: List[Path] = []
        for pat in pats:
            found.extend(sorted(img_dir.glob(pat)))
        for p in found:
            if p.is_file():
                out.append(p)
    # dedup preserve order
    uniq = []
    seen = set()
    for p in out:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _class_names_for_dataset(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name not in VOC_COCO_CLASS_NAMES:
        raise KeyError(f'Unsupported dataset for VOC xml label mapping: {dataset_name}')
    return VOC_COCO_CLASS_NAMES[dataset_name]


def _build_label_lookup(dataset_name: str) -> Dict[str, int]:
    names = _class_names_for_dataset(dataset_name)
    return {name: idx for idx, name in enumerate(names)}


def _parse_voc_xml_to_target(xml_path: Path, label_lookup: Dict[str, int], num_classes: int) -> Dict[str, Any]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def _txt(node, tag, default=''):
        child = node.find(tag)
        return child.text.strip() if child is not None and child.text else default

    size = root.find('size')
    w = int(float(_txt(size, 'width', '0'))) if size is not None else 0
    h = int(float(_txt(size, 'height', '0'))) if size is not None else 0

    boxes = []
    labels = []
    areas = []
    for obj in root.findall('object'):
        cls = _txt(obj, 'name', '')
        if cls not in label_lookup:
            # unknown / unseen classes are mapped to the repo's last class id when available
            label = num_classes - 1
        else:
            label = label_lookup[cls]
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        xmin = float(_txt(bbox, 'xmin', '0')) - 1.0
        ymin = float(_txt(bbox, 'ymin', '0')) - 1.0
        xmax = float(_txt(bbox, 'xmax', '0'))
        ymax = float(_txt(bbox, 'ymax', '0'))
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        areas.append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))

    target = {
        'labels': torch.tensor(labels, dtype=torch.int64),
        'boxes_xyxy_abs': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        'area': torch.tensor(areas, dtype=torch.float32),
        'orig_size': torch.tensor([h, w], dtype=torch.int64),
        'size': torch.tensor([h, w], dtype=torch.int64),
        'iscrowd': torch.zeros(len(labels), dtype=torch.uint8),
    }
    return target


def _xyxy_abs_to_cxcywh_norm(boxes_xyxy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5 / max(float(w), 1.0)
    cy = (y1 + y2) * 0.5 / max(float(h), 1.0)
    bw = (x2 - x1) / max(float(w), 1.0)
    bh = (y2 - y1) / max(float(h), 1.0)
    return torch.stack([cx, cy, bw, bh], dim=-1)


def _infer_image_id_from_path(path: Path, fallback: int) -> int:
    digits = ''.join(ch for ch in path.stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except Exception:
            pass
    return fallback


class ArbitraryImageDataset(Dataset):
    def __init__(
        self,
        args,
        image_paths: Sequence[Path],
        xml_dir: str = '',
        xml_paths: Sequence[str] = (),
        split: str = 't2_test',
        dataset_name: str = 'OWDETR',
    ):
        self.args = args
        self.image_paths = list(image_paths)
        self.transforms = make_coco_transforms(split)
        self.split = split
        self.dataset_name = dataset_name
        self.label_lookup = _build_label_lookup(dataset_name)
        self.num_classes = int(getattr(args, 'num_classes', len(_class_names_for_dataset(dataset_name))))

        self.xml_map: Dict[str, Path] = {}
        if xml_dir:
            xdir = Path(xml_dir)
            if not xdir.exists():
                raise FileNotFoundError(f'xml_dir not found: {xdir}')
            for xp in sorted(xdir.glob('*.xml')):
                self.xml_map[xp.stem] = xp
        for xp in xml_paths:
            p = Path(xp)
            if p.exists():
                self.xml_map[p.stem] = p

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(str(image_path)).convert('RGB')
        w, h = img.size
        image_id = _infer_image_id_from_path(image_path, index)
        stem = image_path.stem

        xml_path = self.xml_map.get(stem, None)
        has_gt = xml_path is not None and xml_path.exists()
        if has_gt:
            raw_target = _parse_voc_xml_to_target(xml_path, self.label_lookup, self.num_classes)
            labels = raw_target['labels']
            boxes_xyxy_abs = raw_target['boxes_xyxy_abs']
            boxes = _xyxy_abs_to_cxcywh_norm(boxes_xyxy_abs, h, w)
            area = raw_target['area']
        else:
            labels = torch.zeros((0,), dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            'image_id': torch.tensor([image_id], dtype=torch.int64),
            'org_image_id': torch.tensor([ord(c) for c in stem], dtype=torch.float32),
            'labels': labels,
            'area': area,
            'boxes': boxes,
            'orig_size': torch.tensor([h, w], dtype=torch.int64),
            'size': torch.tensor([h, w], dtype=torch.int64),
            'iscrowd': torch.zeros(labels.shape[0], dtype=torch.uint8),
            'debug_image_path': str(image_path),
            'debug_image_name': image_path.name,
            'debug_xml_path': str(xml_path) if has_gt else '',
            'debug_has_gt': bool(has_gt),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def _manual_collate(batch):
    images, targets = zip(*batch)
    samples = nested_tensor_from_tensor_list(list(images))
    return samples, list(targets)


def _write_manifest_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_nogt_payload(outputs, targets_cpu, image_paths: List[str]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    payloads = []
    pred_boxes = outputs['pred_boxes'].detach().cpu()
    pred_obj = outputs['pred_obj'].detach().cpu()
    pred_known = outputs.get('pred_known', None)
    pred_logits = outputs['pred_logits'].detach().cpu()

    hidden_dim = pred_obj.shape[-1] if pred_obj.dim() > 2 else 1
    energy = pred_obj.squeeze(-1)
    if energy.numel() > 0 and energy.abs().max().item() > 2.0:
        # repo-style normalized energy approximation for visualization only
        energy = energy / 256.0

    known_prob = None
    unknown_prob = None
    if pred_known is not None:
        kp = torch.exp(-pred_known.detach().cpu().squeeze(-1) * (1.3 / 256.0)).clamp(min=1e-6, max=1.0)
        known_prob = kp
        unknown_prob = (1.0 - kp).clamp(min=0.0, max=1.0)
    else:
        unknown_prob = torch.zeros_like(energy)

    cls_prob = pred_logits.sigmoid()
    known_max = cls_prob.max(dim=-1).values
    obj_prob = torch.exp(-energy * (1.3 / 256.0)).clamp(min=1e-6, max=1.0)

    for i, tgt in enumerate(targets_cpu):
        orig_h = int(tgt['orig_size'][0].item())
        orig_w = int(tgt['orig_size'][1].item())
        boxes_norm = pred_boxes[i]
        x_c, y_c, bw, bh = boxes_norm.unbind(-1)
        boxes_xyxy = torch.stack([x_c - 0.5 * bw, y_c - 0.5 * bh, x_c + 0.5 * bw, y_c + 0.5 * bh], dim=-1)
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=boxes_xyxy.dtype)
        boxes_abs = (boxes_xyxy * scale).numpy()
        nq = boxes_abs.shape[0]
        payload = {
            'pred_boxes_xyxy': boxes_abs,
            'gt_boxes_xyxy': np.zeros((0, 4), dtype=np.float32),
            'gt_labels': np.zeros((0,), dtype=np.int64),
            'energy': energy[i].numpy(),
            'obj_prob': obj_prob[i].numpy(),
            'unknown_prob': unknown_prob[i].numpy(),
            'unknown_score': (obj_prob[i] * unknown_prob[i]).numpy(),
            'known_max': known_max[i].numpy(),
            'conf': np.full((nq,), -1.0, dtype=np.float32),
            'max_iou_to_gt': np.zeros((nq,), dtype=np.float32),
            'max_iof_to_gt': np.zeros((nq,), dtype=np.float32),
            'matched_mask': np.zeros((nq,), dtype=np.uint8),
            'unmatched_mask': np.ones((nq,), dtype=np.uint8),
            'drop_by_gt_mask': np.zeros((nq,), dtype=np.uint8),
            'keep_after_gt_mask': np.ones((nq,), dtype=np.uint8),
            'drop_by_geom_mask': np.zeros((nq,), dtype=np.uint8),
            'keep_after_geom_mask': np.ones((nq,), dtype=np.uint8),
            'drop_by_unk_min_mask': np.zeros((nq,), dtype=np.uint8),
            'drop_by_pos_thresh_mask': np.zeros((nq,), dtype=np.uint8),
            'drop_by_known_reject_mask': np.zeros((nq,), dtype=np.uint8),
            'pos_candidates_pre_nms_mask': np.zeros((nq,), dtype=np.uint8),
            'drop_by_candidate_nms_mask': np.zeros((nq,), dtype=np.uint8),
            'pos_candidates_mask': np.zeros((nq,), dtype=np.uint8),
            'selected_pos_mask': np.zeros((nq,), dtype=np.uint8),
            'selected_neg_mask': np.zeros((nq,), dtype=np.uint8),
            'ignore_mask': np.zeros((nq,), dtype=np.uint8),
        }
        meta = {
            'image_id': int(tgt['image_id'].item()) if torch.is_tensor(tgt['image_id']) else int(tgt['image_id']),
            'image_path': image_paths[i],
            'image_name': Path(image_paths[i]).name,
            'xml_path': tgt.get('debug_xml_path', ''),
            'has_gt': False,
            'nogt_mode': True,
            'exact_mining_replay': False,
            'reason': 'No XML/GT provided, so matcher/GT-overlap based mining cannot be replayed exactly.',
            'orig_h': orig_h,
            'orig_w': orig_w,
            'pos_thresh': float('nan'),
            'matched_mu_obj': float('nan'),
            'matched_std_obj': float('nan'),
            'num_matched': 0,
            'num_unmatched': int(nq),
            'num_keep_after_geom': int(nq),
            'drop_by_gt': 0,
            'drop_by_geom': 0,
            'drop_by_unk_min': 0,
            'drop_by_pos_thresh': 0,
            'drop_by_known_reject': 0,
            'drop_by_candidate_nms': 0,
            'pos_candidates': 0,
            'selected_pos': 0,
            'selected_neg': 0,
            'ignore': 0,
        }
        payloads.append((payload, meta))
    return payloads


def main() -> None:
    parser = build_repo_parser()
    parser.add_argument('--image_paths', default='', help='Comma-separated arbitrary jpg/png file paths')
    parser.add_argument('--image_dir', default='', help='Directory of arbitrary images')
    parser.add_argument('--image_glob', default='*.jpg,*.jpeg,*.png,*.webp', help='Glob patterns used with --image_dir')
    parser.add_argument('--xml_dir', default='', help='Optional VOC XML annotation directory; matched by image stem')
    parser.add_argument('--xml_paths', default='', help='Optional comma-separated XML file paths')
    parser.add_argument('--render_overlay', action='store_true', help='Render overlay panels')
    parser.add_argument('--render_hist', action='store_true', help='Render histogram panels')
    parser.add_argument('--max_draw_boxes_per_group', type=int, default=150, help='Maximum boxes drawn per overlay group')
    parser.add_argument('--allow_no_xml', action='store_true', help='Allow running without XML; this becomes forward-only diagnostic mode, not exact mining replay')
    args = parser.parse_args()

    if not args.render_overlay and not args.render_hist:
        args.render_overlay = True
        args.render_hist = True

    image_paths = _resolve_image_paths(_parse_csv_list(args.image_paths), args.image_dir, args.image_glob)
    if not image_paths:
        raise RuntimeError('No images found from --image_paths / --image_dir')

    xml_paths = _parse_csv_list(args.xml_paths)
    has_any_xml = bool(args.xml_dir or xml_paths)
    if not has_any_xml and not args.allow_no_xml:
        raise RuntimeError('No XML provided. For exact mining replay you need GT XML. Use --allow_no_xml only if you accept forward-only diagnostic mode.')

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    out_root = ensure_dir(args.output_dir_debug)
    raw_dir = ensure_dir(str(Path(out_root) / 'raw'))
    meta_dir = ensure_dir(str(Path(out_root) / 'meta'))
    viz_dir = ensure_dir(str(Path(out_root) / 'viz'))
    overlays_dir = ensure_dir(str(Path(viz_dir) / 'overlays'))
    hists_dir = ensure_dir(str(Path(viz_dir) / 'histograms'))
    manifest_path = Path(out_root) / 'manifest.jsonl'
    summary_csv = Path(out_root) / 'image_summary.csv'
    if manifest_path.exists():
        manifest_path.unlink()

    model, criterion, _, _ = build_model_bundle(args)
    checkpoint = checkpoint_load_to_model(model, args.resume)
    model.to(device)
    criterion.to(device)
    model.eval()
    criterion.eval()

    dataset = ArbitraryImageDataset(
        args=args,
        image_paths=image_paths,
        xml_dir=args.xml_dir,
        xml_paths=xml_paths,
        split=args.split,
        dataset_name=args.dataset,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_manual_collate,
        pin_memory=True,
        drop_last=False,
    )

    run_meta = {
        'resume': args.resume,
        'split': args.split,
        'stage_epoch': int(args.stage_epoch),
        'stage_name': args.stage_name,
        'gt_overlap_mode': args.gt_overlap_mode,
        'batch_size': int(args.batch_size),
        'model_type': getattr(args, 'model_type', None),
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'image_count': len(image_paths),
        'xml_dir': args.xml_dir,
        'xml_count': len(dataset.xml_map),
        'allow_no_xml': bool(args.allow_no_xml),
        'notes': 'With XML: exact mining replay. Without XML: forward-only diagnostic mode, no GT/matcher-based replay.',
    }
    save_json(Path(out_root) / 'run_meta.json', run_meta)

    summary_rows: List[Dict[str, Any]] = []
    global_idx = 0

    with torch.no_grad():
        for batch_idx, (samples, targets_cpu) in enumerate(loader):
            samples = samples.to(device)
            targets = to_device_targets(targets_cpu, device)
            outputs = model(samples)

            batch_has_gt = any(bool(t.get('debug_has_gt', False)) for t in targets_cpu)
            if batch_has_gt:
                matcher_inputs = {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
                matched_indices = criterion.matcher(matcher_inputs, targets)
                image_debugs, _, _, _, _ = replay_uod_mining_debug(
                    outputs=outputs,
                    targets=targets,
                    indices=matched_indices,
                    criterion=criterion,
                    stage_epoch=args.stage_epoch,
                    gt_overlap_mode=args.gt_overlap_mode,
                )
                batch_payloads: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                for local_idx, (target_cpu, debug_img) in enumerate(zip(targets_cpu, image_debugs)):
                    orig_h = int(target_cpu['orig_size'][0].item())
                    orig_w = int(target_cpu['orig_size'][1].item())
                    pred_boxes_abs = np.asarray(debug_img['pred_boxes_xyxy_norm'], dtype=np.float32).copy()
                    if pred_boxes_abs.size > 0:
                        pred_boxes_abs[:, [0, 2]] *= orig_w
                        pred_boxes_abs[:, [1, 3]] *= orig_h
                    gt_boxes_abs = np.asarray(debug_img['gt_boxes_xyxy_norm'], dtype=np.float32).copy()
                    if gt_boxes_abs.size > 0:
                        gt_boxes_abs[:, [0, 2]] *= orig_w
                        gt_boxes_abs[:, [1, 3]] *= orig_h
                    nq = pred_boxes_abs.shape[0]
                    payload = {
                        'pred_boxes_xyxy': pred_boxes_abs,
                        'gt_boxes_xyxy': gt_boxes_abs,
                        'gt_labels': np.asarray(debug_img['gt_labels']),
                        'energy': np.asarray(debug_img['energy']),
                        'obj_prob': np.asarray(debug_img['obj_prob']),
                        'unknown_prob': np.asarray(debug_img['unknown_prob']),
                        'unknown_score': np.asarray(debug_img['unknown_score']),
                        'known_max': np.asarray(debug_img['known_max']),
                        'conf': np.asarray(debug_img['conf']),
                        'max_iou_to_gt': np.asarray(debug_img['max_iou_to_gt']),
                        'max_iof_to_gt': np.asarray(debug_img['max_iof_to_gt']),
                        'matched_mask': _bool_mask_from_indices(nq, debug_img['matched']),
                        'unmatched_mask': _bool_mask_from_indices(nq, debug_img['unmatched']),
                        'drop_by_gt_mask': _bool_mask_from_indices(nq, debug_img['drop_by_gt']),
                        'keep_after_gt_mask': _bool_mask_from_indices(nq, debug_img['keep_after_gt']),
                        'drop_by_geom_mask': _bool_mask_from_indices(nq, debug_img['drop_by_geom']),
                        'keep_after_geom_mask': _bool_mask_from_indices(nq, debug_img['keep_after_geom']),
                        'drop_by_unk_min_mask': _bool_mask_from_indices(nq, debug_img['drop_by_unk_min']),
                        'drop_by_pos_thresh_mask': _bool_mask_from_indices(nq, debug_img['drop_by_pos_thresh']),
                        'drop_by_known_reject_mask': _bool_mask_from_indices(nq, debug_img['drop_by_known_reject']),
                        'pos_candidates_pre_nms_mask': _bool_mask_from_indices(nq, debug_img['pos_candidates_pre_nms']),
                        'drop_by_candidate_nms_mask': _bool_mask_from_indices(nq, debug_img['drop_by_candidate_nms']),
                        'pos_candidates_mask': _bool_mask_from_indices(nq, debug_img['pos_candidates']),
                        'selected_pos_mask': _bool_mask_from_indices(nq, debug_img['selected_pos']),
                        'selected_neg_mask': _bool_mask_from_indices(nq, debug_img['selected_neg']),
                        'ignore_mask': _bool_mask_from_indices(nq, debug_img['ignore']),
                    }
                    meta = {
                        'image_id': int(target_cpu['image_id'].item()) if torch.is_tensor(target_cpu['image_id']) else int(target_cpu['image_id']),
                        'image_path': target_cpu.get('debug_image_path', ''),
                        'image_name': target_cpu.get('debug_image_name', Path(target_cpu.get('debug_image_path', '')).name),
                        'xml_path': target_cpu.get('debug_xml_path', ''),
                        'has_gt': bool(target_cpu.get('debug_has_gt', False)),
                        'nogt_mode': False,
                        'exact_mining_replay': True,
                        'orig_h': orig_h,
                        'orig_w': orig_w,
                        'pos_thresh': float(debug_img['pos_thresh']),
                        'matched_mu_obj': float(debug_img['matched_mu_obj']),
                        'matched_std_obj': float(debug_img['matched_std_obj']),
                        'num_matched': len(debug_img['matched']),
                        'num_unmatched': len(debug_img['unmatched']),
                        'num_keep_after_geom': len(debug_img['keep_after_geom']),
                        'drop_by_gt': len(debug_img['drop_by_gt']),
                        'drop_by_geom': len(debug_img['drop_by_geom']),
                        'drop_by_unk_min': len(debug_img['drop_by_unk_min']),
                        'drop_by_pos_thresh': len(debug_img['drop_by_pos_thresh']),
                        'drop_by_known_reject': len(debug_img['drop_by_known_reject']),
                        'drop_by_candidate_nms': len(debug_img['drop_by_candidate_nms']),
                        'pos_candidates': len(debug_img['pos_candidates']),
                        'selected_pos': len(debug_img['selected_pos']),
                        'selected_neg': len(debug_img['selected_neg']),
                        'ignore': len(debug_img['ignore']),
                    }
                    batch_payloads.append((payload, meta))
            else:
                batch_payloads = _build_nogt_payload(outputs, targets_cpu, [t.get('debug_image_path', '') for t in targets_cpu])

            for local_idx, (payload, meta) in enumerate(batch_payloads):
                stem = f'{batch_idx:05d}_{local_idx:02d}_{Path(meta["image_name"]).stem}'
                meta['stem'] = stem
                npz_path = Path(raw_dir) / f'{stem}.npz'
                meta_path = Path(meta_dir) / f'{stem}.json'
                _write_npz(npz_path, meta_path, payload, meta)
                _write_manifest_row(manifest_path, {'stem': stem, 'npz': str(npz_path), 'meta': str(meta_path)})
                summary_rows.append(meta)
                entry = {'stem': stem, 'npz': str(npz_path), 'meta': str(meta_path)}
                if args.render_overlay:
                    _render_overlay(entry, Path(overlays_dir), max_boxes=args.max_draw_boxes_per_group)
                if args.render_hist:
                    _render_histograms(entry, Path(hists_dir))
                global_idx += 1

    _write_csv(summary_csv, summary_rows)
    print(f'[OK] saved {len(summary_rows)} arbitrary-image debug records to {out_root}')
    print(f'[OK] with_gt={sum(1 for r in summary_rows if r.get("has_gt", False))}, no_gt={sum(1 for r in summary_rows if not r.get("has_gt", False))}')
    print(f'[OK] manifest: {manifest_path}')
    print(f'[OK] summary:  {summary_csv}')


if __name__ == '__main__':
    main()
