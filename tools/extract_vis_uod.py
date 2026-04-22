#!/usr/bin/env python3
"""Extract per-image and global visualization data for UOD qualitative/statistical analysis.

This script intentionally does not modify the training/evaluation pipeline. It builds the
model from the repo configuration, runs a deterministic evaluation-style forward pass,
optionally captures ODQE internals via hooks, replays the pseudo-unknown mining logic
for debugging/visualization, and writes:

- manifest.json
- global_stats.pt
- per_image/<image_id>.pt

The dump is later consumed by tools/render_vis_uod.py.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

import util.misc as utils
from main_open_world import get_args_parser as get_main_args_parser
from main_open_world import build_datasets
from models import build_model
from util import box_ops


def build_extraction_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        'Extract UOD visualization tensors', parents=[get_main_args_parser()], add_help=True
    )
    parser.set_defaults(eval=False, distributed=False, num_workers=4)
    parser.add_argument('--save_dir', required=True, help='directory where visualization dumps are written')
    parser.add_argument('--dump_image_ids_file', default=None, help='optional text file with one image_id per line to force-save')
    parser.add_argument('--dump_every_n', type=int, default=50, help='save every N-th image when dump_image_ids_file is not provided')
    parser.add_argument('--dump_max_images', type=int, default=100, help='global cap for saved per-image dumps')
    parser.add_argument('--dump_start_index', type=int, default=0, help='start applying dump_every_n from this global dataset index')
    parser.add_argument('--disable_odqe_hooks', action='store_true', help='disable ODQE hook extraction even when model enables ODQE')
    parser.add_argument('--background_iou_thr', type=float, default=0.1, help='IoU threshold below which unmatched queries are treated as background for global stats')
    parser.add_argument('--log_every', type=int, default=20)
    return parser


class HookRecorder:
    """Batch-local hook recorder for ODQE internals.

    We intentionally record per-call lists rather than indexing by module instance so shared
    modules (with_box_refine=False) still yield decoder-order events.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[Any] = []
        self.enabled = False
        self.batch_cache: Dict[str, List[torch.Tensor]] = {}
        self.meta: Dict[str, Any] = {}

    def clear(self) -> None:
        self.batch_cache = {
            'context_query': [],
            'context_reference_points': [],
            'context_output': [],
            'sampling_offsets_raw': [],
            'attention_logits_raw': [],
            'gate_logits_raw': [],
            'spatial_shapes': [],
            'level_start_index': [],
        }

    def register(self) -> None:
        if not hasattr(self.model, 'context_attn'):
            return

        unique_context_modules = []
        seen_ids = set()
        for module in list(self.model.context_attn):
            if id(module) in seen_ids:
                continue
            seen_ids.add(id(module))
            unique_context_modules.append(module)

        unique_gate_modules = []
        seen_ids = set()
        if hasattr(self.model, 'gate_mlp'):
            for module in list(self.model.gate_mlp):
                if id(module) in seen_ids:
                    continue
                seen_ids.add(id(module))
                unique_gate_modules.append(module)

        for module in unique_context_modules:
            self.handles.append(module.register_forward_pre_hook(self._context_pre_hook))
            self.handles.append(module.register_forward_hook(self._context_out_hook))
            self.handles.append(module.sampling_offsets.register_forward_hook(self._offsets_hook))
            self.handles.append(module.attention_weights.register_forward_hook(self._attn_hook))
            self.meta = {
                'n_heads': int(module.n_heads),
                'n_levels': int(module.n_levels),
                'n_points': int(module.n_points),
                'd_model': int(module.d_model),
            }
        for module in unique_gate_modules:
            self.handles.append(module.register_forward_hook(self._gate_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _context_pre_hook(self, module, inputs):
        if not self.enabled:
            return
        query, reference_points, _memory, spatial_shapes, level_start_index, _padding_mask = inputs
        self.batch_cache['context_query'].append(query.detach().cpu())
        self.batch_cache['context_reference_points'].append(reference_points.detach().cpu())
        self.batch_cache['spatial_shapes'].append(spatial_shapes.detach().cpu())
        self.batch_cache['level_start_index'].append(level_start_index.detach().cpu())

    def _context_out_hook(self, module, inputs, output):
        if not self.enabled:
            return
        self.batch_cache['context_output'].append(output.detach().cpu())

    def _offsets_hook(self, module, inputs, output):
        if not self.enabled:
            return
        self.batch_cache['sampling_offsets_raw'].append(output.detach().cpu())

    def _attn_hook(self, module, inputs, output):
        if not self.enabled:
            return
        self.batch_cache['attention_logits_raw'].append(output.detach().cpu())

    def _gate_hook(self, module, inputs, output):
        if not self.enabled:
            return
        self.batch_cache['gate_logits_raw'].append(output.detach().cpu())

    def package_image(self, batch_index: int) -> Dict[str, Any]:
        if not self.batch_cache:
            return {}
        packaged = {
            'meta': dict(self.meta),
            'odqe_layer_decay': getattr(self.model, 'odqe_layer_decay', torch.empty(0)).detach().cpu(),
            'layers': [],
        }
        num_layers = len(self.batch_cache['context_output'])
        for layer_index in range(num_layers):
            layer_entry = {
                'query': self.batch_cache['context_query'][layer_index][batch_index].clone(),
                'reference_points': self.batch_cache['context_reference_points'][layer_index][batch_index].clone(),
                'context_output': self.batch_cache['context_output'][layer_index][batch_index].clone(),
                'sampling_offsets_raw': self.batch_cache['sampling_offsets_raw'][layer_index][batch_index].clone(),
                'attention_logits_raw': self.batch_cache['attention_logits_raw'][layer_index][batch_index].clone(),
                'spatial_shapes': self.batch_cache['spatial_shapes'][layer_index].clone(),
                'level_start_index': self.batch_cache['level_start_index'][layer_index].clone(),
            }
            if layer_index < len(self.batch_cache['gate_logits_raw']):
                layer_entry['gate_logits_raw'] = self.batch_cache['gate_logits_raw'][layer_index][batch_index].clone()
            packaged['layers'].append(layer_entry)
        return packaged


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint_state_dict(path: str, map_location: str = 'cpu') -> Dict[str, Any]:
    checkpoint = (
        torch.hub.load_state_dict_from_url(path, map_location=map_location, check_hash=True)
        if isinstance(path, str) and path.startswith('https')
        else torch.load(path, map_location=map_location)
    )
    return checkpoint


def maybe_tensor_to_list(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: maybe_tensor_to_list(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [maybe_tensor_to_list(v) for v in value]
    return value


def read_requested_image_ids(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    image_ids = set()
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            image_ids.add(int(line))
    return image_ids


def select_for_dump(
    image_id: int,
    dataset_index: int,
    requested_ids: Optional[set],
    already_selected: int,
    every_n: int,
    max_images: int,
    start_index: int,
) -> bool:
    if already_selected >= max_images:
        return False
    if requested_ids is not None:
        return image_id in requested_ids
    if dataset_index < start_index:
        return False
    stride = max(1, every_n)
    return ((dataset_index - start_index) % stride) == 0


def energy_to_prob(energy: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.exp(-temperature * energy).clamp(min=1e-6, max=1.0)


def compute_fused_probabilities(
    pred_logits: torch.Tensor,
    pred_obj: torch.Tensor,
    pred_known: Optional[torch.Tensor],
    invalid_cls_logits: Sequence[int],
    obj_temperature: float,
    known_temperature: float,
    unknown_scale: float,
) -> Dict[str, torch.Tensor]:
    logits = pred_logits.clone()
    if len(invalid_cls_logits) > 0:
        logits[:, :, invalid_cls_logits] = -10e10
    class_prob = logits.sigmoid()
    if len(invalid_cls_logits) > 0:
        class_prob[:, :, invalid_cls_logits] = 0.0
    if class_prob.shape[-1] > 0:
        class_prob[:, :, -1] = 0.0

    obj_prob = energy_to_prob(pred_obj, obj_temperature)
    if pred_known is None:
        knownness_prob = torch.ones_like(obj_prob)
    else:
        knownness_prob = energy_to_prob(pred_known, known_temperature)
    unknown_prob = (1.0 - knownness_prob).clamp(min=0.0, max=1.0)
    known_scores = obj_prob.unsqueeze(-1) * class_prob * knownness_prob.unsqueeze(-1)
    if class_prob.shape[-1] > 1:
        max_known_cls_prob = class_prob[:, :, :-1].max(dim=-1).values
    elif class_prob.shape[-1] > 0:
        max_known_cls_prob = class_prob.squeeze(-1)
    else:
        max_known_cls_prob = torch.zeros_like(obj_prob)
    unknown_score = obj_prob * unknown_prob * float(unknown_scale)
    fused_prob = known_scores.clone()
    if fused_prob.shape[-1] > 0:
        fused_prob[:, :, -1] = unknown_score
    return {
        'obj_prob': obj_prob,
        'knownness_prob': knownness_prob,
        'unknown_prob': unknown_prob,
        'class_prob': class_prob,
        'known_scores': known_scores,
        'max_known_cls_prob': max_known_cls_prob,
        'unknown_score': unknown_score,
        'fused_prob': fused_prob,
    }


def pairwise_iof(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)).clamp(min=1e-6)
    return inter / area1[:, None]


def is_valid_geometry(box_cxcywh: torch.Tensor, args) -> bool:
    w = float(box_cxcywh[2].item())
    h = float(box_cxcywh[3].item())
    area = w * h
    side = min(w, h)
    aspect_ratio = max(w / max(h, 1e-6), h / max(w, 1e-6))
    return area >= args.uod_min_area and side >= args.uod_min_side and aspect_ratio <= args.uod_max_aspect_ratio


def deduplicate_pos_candidates(pred_boxes_img: torch.Tensor, candidates: List[Tuple], iou_thr: float) -> List[Tuple]:
    if len(candidates) <= 1 or iou_thr is None or iou_thr <= 0:
        return candidates
    candidates = sorted(candidates, key=lambda item: (-item[2], item[3], item[4]))
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_img)
    kept: List[Tuple] = []
    kept_queries: List[int] = []
    for item in candidates:
        query_index = item[1]
        if len(kept_queries) == 0:
            kept.append(item)
            kept_queries.append(query_index)
            continue
        compare_q = torch.as_tensor(kept_queries, dtype=torch.long, device=boxes_xyxy.device)
        ious = box_ops.box_iou(boxes_xyxy[query_index].unsqueeze(0), boxes_xyxy[compare_q])[0]
        if torch.any(ious >= iou_thr):
            continue
        kept.append(item)
        kept_queries.append(query_index)
    return kept


def filter_negatives_near_selected_pos(pred_xyxy: torch.Tensor, selected_q: Sequence[int], candidate_qs: Sequence[int], iou_thr: float) -> List[int]:
    if len(selected_q) == 0 or len(candidate_qs) == 0 or iou_thr <= 0:
        return list(candidate_qs)
    selected = torch.as_tensor(list(selected_q), dtype=torch.long, device=pred_xyxy.device)
    kept: List[int] = []
    for query_index in candidate_qs:
        q_idx = torch.as_tensor([query_index], dtype=torch.long, device=pred_xyxy.device)
        ious = box_ops.box_iou(pred_xyxy[q_idx], pred_xyxy[selected])[0]
        if torch.any(ious > iou_thr):
            continue
        kept.append(int(query_index))
    return kept


def replay_mine_uod_pseudo(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    args,
    hidden_dim: int,
    invalid_cls_logits: Sequence[int],
    epoch: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Replay the pseudo-unknown mining logic to expose visualization stages.

    Returns one dict per batch image with stage indices and metadata.
    """
    if epoch is None:
        epoch = int(args.uod_start_epoch)

    batch_size = len(targets)
    per_image_debug: List[Dict[str, Any]] = [
        {
            'valid_after_geom': [],
            'valid_after_unknown': [],
            'selected_pseudo_pos': [],
            'selected_pseudo_neg': [],
            'ignore_queries': [],
            'selected_pos_weights': [],
            'pos_thresh': float(args.uod_min_pos_thresh),
        }
        for _ in range(batch_size)
    ]
    stats = {
        'num_dummy_pos': 0.0,
        'num_dummy_neg': 0.0,
        'num_ignore_queries': 0.0,
        'num_valid_unmatched': 0.0,
        'num_pos_candidates': 0.0,
        'num_neg_candidates': 0.0,
        'num_batch_selected_pos': 0.0,
        'pos_thresh_sum': 0.0,
        'num_thresh': 0.0,
    }

    if (not bool(args.uod_enable_pseudo)) or epoch < int(args.uod_start_epoch):
        return per_image_debug, stats

    fused = compute_fused_probabilities(
        outputs['pred_logits'],
        outputs['pred_obj'],
        outputs.get('pred_known', None),
        invalid_cls_logits,
        obj_temperature=float(args.obj_temp) / float(hidden_dim),
        known_temperature=float(args.uod_known_temp) / float(hidden_dim),
        unknown_scale=float(args.uod_postprocess_unknown_scale),
    )
    energy = outputs['pred_obj'].detach() / float(hidden_dim)
    pred_boxes = outputs['pred_boxes'].detach()
    obj_prob = fused['obj_prob'].detach()
    unknown_prob = fused['unknown_prob'].detach()
    unknown_score = fused['unknown_score'].detach()
    known_max = fused['max_known_cls_prob'].detach()
    num_queries = energy.shape[1]

    all_pos_candidates: List[Tuple] = []
    per_img_pos_candidates: List[List[Tuple]] = []
    per_img_cache: List[Dict[str, Any]] = []

    for batch_index, (src_idx, _target_idx) in enumerate(indices):
        matched = set(src_idx.tolist())
        unmatched = [query_idx for query_idx in range(num_queries) if query_idx not in matched]

        if len(src_idx) > 0:
            matched_scores = energy[batch_index, src_idx]
            mu_obj = float(matched_scores.mean().item())
            std_obj = float(matched_scores.std().item()) if len(src_idx) > 1 else 0.0
            pos_thresh = max(mu_obj + 3.0 * std_obj, float(args.uod_min_pos_thresh))
        else:
            pos_thresh = float(args.uod_min_pos_thresh)
        per_image_debug[batch_index]['pos_thresh'] = pos_thresh
        stats['pos_thresh_sum'] += pos_thresh
        stats['num_thresh'] += 1.0

        pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[batch_index])
        gt_xyxy = box_ops.box_cxcywh_to_xyxy(targets[batch_index]['boxes'])
        valid = unmatched
        iou_map = {query_idx: 0.0 for query_idx in unmatched}

        if gt_xyxy.numel() > 0 and len(unmatched) > 0:
            cand_boxes = pred_xyxy[unmatched]
            ious = box_ops.box_iou(cand_boxes, gt_xyxy)[0]
            iofs = pairwise_iof(cand_boxes, gt_xyxy)
            max_iou = ious.max(dim=1)[0]
            max_iof = iofs.max(dim=1)[0]
            valid = []
            for j, query_idx in enumerate(unmatched):
                iou_map[query_idx] = float(max_iou[j].item())
                if float(max_iou[j].item()) < float(args.uod_max_iou) and float(max_iof[j].item()) < float(args.uod_max_iof):
                    valid.append(query_idx)
        valid = [query_idx for query_idx in valid if is_valid_geometry(pred_boxes[batch_index, query_idx], args)]
        per_image_debug[batch_index]['valid_after_geom'] = list(valid)
        stats['num_valid_unmatched'] += float(len(valid))

        pos_candidates: List[Tuple] = []
        stage2_queries: List[int] = []
        for query_idx in valid:
            e_val = float(energy[batch_index, query_idx].item())
            k_val = float(known_max[batch_index, query_idx].item())
            u_val = float(unknown_prob[batch_index, query_idx].item())
            us_val = float(unknown_score[batch_index, query_idx].item())
            if u_val < float(args.uod_pos_unk_min):
                continue
            if e_val < pos_thresh and k_val < float(args.uod_known_reject_thresh):
                energy_rel = max(0.0, min(1.0, (pos_thresh - e_val) / max(pos_thresh, 1e-6)))
                known_rel = max(0.0, min(1.0, (float(args.uod_known_reject_thresh) - k_val) / max(float(args.uod_known_reject_thresh), 1e-6)))
                iou_rel = 1.0 - max(0.0, min(1.0, iou_map[query_idx] / max(float(args.uod_max_iou), 1e-6)))
                unk_rel = max(0.0, min(1.0, u_val))
                conf = (energy_rel * known_rel * iou_rel * max(unk_rel, 1e-6)) ** 0.25
                pos_candidates.append((batch_index, query_idx, conf, e_val, k_val, u_val, us_val))
                stage2_queries.append(query_idx)
        pos_candidates = deduplicate_pos_candidates(pred_boxes[batch_index], pos_candidates, float(args.uod_candidate_nms_iou))
        stage2_set = {item[1] for item in pos_candidates}
        per_image_debug[batch_index]['valid_after_unknown'] = [query_idx for query_idx in stage2_queries if query_idx in stage2_set]
        all_pos_candidates.extend(pos_candidates)
        stats['num_pos_candidates'] += float(len(pos_candidates))
        per_img_pos_candidates.append(pos_candidates)
        per_img_cache.append({'valid': valid, 'pred_xyxy': pred_xyxy})

    dummy_pos_indices: List[List[int]] = [[] for _ in range(batch_size)]
    dummy_pos_weights: List[List[float]] = [[] for _ in range(batch_size)]
    if bool(args.uod_enable_batch_dynamic):
        all_pos_candidates.sort(key=lambda item: (-item[2], -item[6], -item[5], item[3], item[4]))
        topk = min(
            int(args.uod_batch_topk_max),
            max(1, int(math.ceil(float(args.uod_batch_topk_ratio) * max(len(all_pos_candidates), 1))))
        )
        per_img_count = [0 for _ in range(batch_size)]
        selected = []
        for item in all_pos_candidates:
            batch_index, query_idx, conf, e_val, k_val, u_val, us_val = item
            if len(selected) >= topk:
                break
            if int(args.uod_pos_per_img_cap) > 0 and per_img_count[batch_index] >= int(args.uod_pos_per_img_cap):
                continue
            selected.append(item)
            per_img_count[batch_index] += 1
        for batch_index, query_idx, conf, _e_val, _k_val, _u_val, _us_val in selected:
            dummy_pos_indices[batch_index].append(int(query_idx))
            dummy_pos_weights[batch_index].append(float(max(0.2, min(1.0, conf))))
        stats['num_batch_selected_pos'] = float(len(selected))
    else:
        for batch_index, pos_candidates in enumerate(per_img_pos_candidates):
            pos_candidates.sort(key=lambda item: (-item[2], -item[6], -item[5], item[3], item[4]))
            per_cap = int(args.uod_pos_per_img_cap)
            if per_cap > 0:
                pos_candidates = pos_candidates[:per_cap]
            dummy_pos_indices[batch_index] = [int(item[1]) for item in pos_candidates]
            dummy_pos_weights[batch_index] = [float(max(0.2, min(1.0, item[2]))) for item in pos_candidates]
        stats['num_batch_selected_pos'] = float(sum(len(item) for item in dummy_pos_indices))

    stats['num_dummy_pos'] = float(sum(len(item) for item in dummy_pos_indices))
    dummy_neg_indices: List[List[int]] = [[] for _ in range(batch_size)]
    if epoch >= int(args.uod_start_epoch) + int(args.uod_neg_warmup_epochs):
        for batch_index in range(batch_size):
            valid = per_img_cache[batch_index]['valid']
            pred_xyxy = per_img_cache[batch_index]['pred_xyxy']
            pos_selected = dummy_pos_indices[batch_index]
            remaining = [query_idx for query_idx in valid if query_idx not in set(pos_selected)]
            remaining = filter_negatives_near_selected_pos(pred_xyxy, pos_selected, remaining, float(args.uod_neg_max_pseudo_iou))
            neg_candidates = []
            for query_idx in remaining:
                k_val = float(known_max[batch_index, query_idx].item())
                obj_val = float(obj_prob[batch_index, query_idx].item())
                u_val = float(unknown_prob[batch_index, query_idx].item())
                e_val = float(energy[batch_index, query_idx].item())
                if k_val > float(args.uod_neg_known_max):
                    continue
                if u_val > float(args.uod_neg_unk_max):
                    continue
                neg_candidates.append((query_idx, obj_val, e_val, k_val, u_val))
            stats['num_neg_candidates'] += float(len(neg_candidates))
            neg_candidates.sort(key=lambda item: (-item[1], item[2], item[3], item[4]))
            neg_candidates = neg_candidates[:int(args.uod_neg_per_img)]
            dummy_neg_indices[batch_index] = [int(item[0]) for item in neg_candidates]
            stats['num_dummy_neg'] += float(len(dummy_neg_indices[batch_index]))

    for batch_index in range(batch_size):
        pos_set = set(dummy_pos_indices[batch_index])
        neg_set = set(dummy_neg_indices[batch_index])
        ignore_queries: List[int] = []
        valid = per_img_cache[batch_index]['valid']
        for query_idx in valid:
            if query_idx in pos_set or query_idx in neg_set:
                continue
            if (
                float(obj_prob[batch_index, query_idx].item()) > 0.05
                and float(unknown_prob[batch_index, query_idx].item()) >= float(args.uod_pos_unk_min)
                and float(known_max[batch_index, query_idx].item()) < float(args.uod_known_reject_thresh)
            ):
                ignore_queries.append(int(query_idx))
        per_image_debug[batch_index]['selected_pseudo_pos'] = list(dummy_pos_indices[batch_index])
        per_image_debug[batch_index]['selected_pos_weights'] = list(dummy_pos_weights[batch_index])
        per_image_debug[batch_index]['selected_pseudo_neg'] = list(dummy_neg_indices[batch_index])
        per_image_debug[batch_index]['ignore_queries'] = list(ignore_queries)
        stats['num_ignore_queries'] += float(len(ignore_queries))
    return per_image_debug, stats


def categorize_queries(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    background_iou_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign query labels: 0=known, 1=unknown, 2=background, -1=ignore."""
    batch_size, num_queries = outputs['pred_boxes'].shape[:2]
    categories = torch.full((batch_size, num_queries), -1, dtype=torch.int64, device=outputs['pred_boxes'].device)
    max_ious = torch.zeros((batch_size, num_queries), dtype=torch.float32, device=outputs['pred_boxes'].device)
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'])
    unknown_class = num_classes - 1

    for batch_index, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) > 0:
            matched_labels = targets[batch_index]['labels'][tgt_idx]
            categories[batch_index, src_idx] = torch.where(matched_labels == unknown_class, 1, 0)
        if targets[batch_index]['boxes'].numel() > 0:
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(targets[batch_index]['boxes'])
            ious = box_ops.box_iou(pred_xyxy[batch_index], gt_xyxy)[0]
            max_iou, _ = ious.max(dim=1)
            max_ious[batch_index] = max_iou
        unmatched_mask = categories[batch_index] < 0
        categories[batch_index, unmatched_mask & (max_ious[batch_index] < background_iou_thr)] = 2
    return categories, max_ious


def gather_raw_gt(eval_dataset, img_id: int, args) -> Dict[str, torch.Tensor]:
    _target_tree, instances = eval_dataset.load_instances(img_id)
    instances = eval_dataset.label_known_class_and_unknown(instances)
    boxes = torch.as_tensor([item['bbox'] for item in instances], dtype=torch.float32)
    labels = torch.as_tensor([item['category_id'] for item in instances], dtype=torch.int64)
    return {'boxes_abs_xyxy': boxes, 'labels': labels}


def simplify_aux_outputs(aux_outputs: Sequence[Dict[str, torch.Tensor]], batch_index: int) -> List[Dict[str, torch.Tensor]]:
    simplified = []
    keep_keys = {'pred_boxes', 'pred_obj', 'pred_known', 'pred_logits'}
    for aux in aux_outputs:
        simplified.append({
            key: value[batch_index].detach().cpu()
            for key, value in aux.items()
            if key in keep_keys and value is not None
        })
    return simplified


def append_global_stats(
    global_lists: Dict[str, List[torch.Tensor]],
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    args,
    invalid_cls_logits: Sequence[int],
    image_ids: torch.Tensor,
    query_categories: torch.Tensor,
) -> None:
    fused = compute_fused_probabilities(
        outputs['pred_logits'],
        outputs['pred_obj'],
        outputs.get('pred_known', None),
        invalid_cls_logits,
        obj_temperature=float(args.obj_temp) / float(args.hidden_dim),
        known_temperature=float(args.uod_known_temp) / float(args.hidden_dim),
        unknown_scale=float(args.uod_postprocess_unknown_scale),
    )
    batch_size, num_queries = outputs['pred_boxes'].shape[:2]
    query_idx = torch.arange(num_queries, device=outputs['pred_boxes'].device).unsqueeze(0).repeat(batch_size, 1)
    image_ids_broadcast = image_ids.unsqueeze(1).repeat(1, num_queries)
    valid_mask = query_categories >= 0
    if valid_mask.any():
        global_lists['obj_prob'].append(fused['obj_prob'][valid_mask].detach().cpu())
        global_lists['unknown_prob'].append(fused['unknown_prob'][valid_mask].detach().cpu())
        global_lists['knownness_prob'].append(fused['knownness_prob'][valid_mask].detach().cpu())
        global_lists['max_known_cls_prob'].append(fused['max_known_cls_prob'][valid_mask].detach().cpu())
        global_lists['group'].append(query_categories[valid_mask].detach().cpu())
        global_lists['image_id'].append(image_ids_broadcast[valid_mask].detach().cpu())
        global_lists['query_idx'].append(query_idx[valid_mask].detach().cpu())


def finalize_global_stats(global_lists: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    finalized = {}
    for key, values in global_lists.items():
        if len(values) == 0:
            finalized[key] = torch.empty(0)
        else:
            finalized[key] = torch.cat(values, dim=0)
    return finalized


def main() -> None:
    parser = build_extraction_parser()
    args = parser.parse_args()
    if not getattr(args, 'eval_checkpoint', None):
        parser.error('--eval_checkpoint is required')
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    utils.init_distributed_mode(args)
    if getattr(args, 'distributed', False):
        raise RuntimeError('extract_vis_uod.py only supports single-process execution.')

    set_seed(int(args.seed))
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    per_image_dir = save_dir / 'per_image'
    per_image_dir.mkdir(parents=True, exist_ok=True)

    requested_image_ids = read_requested_image_ids(args.dump_image_ids_file)
    model, criterion, postprocessors, _ = build_model(args, mode=args.model_type)
    model.to(device)
    model.eval()
    criterion.eval()

    checkpoint = load_checkpoint_state_dict(args.eval_checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = [key for key in unexpected_keys if not (key.endswith('total_params') or key.endswith('total_ops'))]
    if missing_keys:
        logging.info('Missing keys while loading checkpoint: %s', missing_keys)
    if unexpected_keys:
        logging.info('Unexpected keys while loading checkpoint: %s', unexpected_keys)

    _, eval_dataset = build_datasets(args)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=SequentialSampler(eval_dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS, args.num_classes - 1))
    hook_recorder = HookRecorder(model)
    enable_odqe_hooks = bool(getattr(args, 'uod_enable_odqe', False)) and not bool(args.disable_odqe_hooks)
    if enable_odqe_hooks:
        hook_recorder.register()

    global_lists: Dict[str, List[torch.Tensor]] = {
        'obj_prob': [],
        'unknown_prob': [],
        'knownness_prob': [],
        'max_known_cls_prob': [],
        'group': [],
        'image_id': [],
        'query_idx': [],
    }
    saved_records: List[Dict[str, Any]] = []
    total_selected = 0
    processed = 0

    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(eval_loader):
            samples = samples.to(device)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            batch_image_ids = torch.as_tensor([int(target['image_id'].item()) for target in targets], dtype=torch.int64)
            dataset_indices = list(range(processed, processed + len(targets)))
            processed += len(targets)

            selected_in_batch: List[bool] = []
            for local_index, image_id in enumerate(batch_image_ids.tolist()):
                should_dump = select_for_dump(
                    image_id=image_id,
                    dataset_index=dataset_indices[local_index],
                    requested_ids=requested_image_ids,
                    already_selected=total_selected + sum(selected_in_batch),
                    every_n=int(args.dump_every_n),
                    max_images=int(args.dump_max_images),
                    start_index=int(args.dump_start_index),
                )
                selected_in_batch.append(should_dump)

            hook_recorder.clear()
            hook_recorder.enabled = enable_odqe_hooks and any(selected_in_batch)

            outputs = model(samples)
            original_sizes = torch.stack([target['orig_size'] for target in targets], dim=0)
            results = postprocessors['bbox'](outputs, original_sizes)
            indices = criterion.matcher({'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}, targets)
            query_categories, _max_ious = categorize_queries(
                outputs, targets, indices, num_classes=args.num_classes, background_iou_thr=float(args.background_iou_thr)
            )
            append_global_stats(
                global_lists,
                outputs,
                targets,
                indices,
                args,
                invalid_cls_logits,
                image_ids=batch_image_ids.to(device),
                query_categories=query_categories,
            )
            pseudo_debug, pseudo_stats = replay_mine_uod_pseudo(
                outputs,
                targets,
                indices,
                args=args,
                hidden_dim=int(args.hidden_dim),
                invalid_cls_logits=invalid_cls_logits,
                epoch=int(args.uod_start_epoch),
            )
            del pseudo_stats

            for local_index, should_dump in enumerate(selected_in_batch):
                if not should_dump:
                    continue
                image_id = int(batch_image_ids[local_index].item())
                dataset_index = int(dataset_indices[local_index])
                image_path = str(eval_dataset.images[dataset_index])
                raw_gt = gather_raw_gt(eval_dataset, image_id, args)
                entry = {
                    'image_id': image_id,
                    'dataset_index': dataset_index,
                    'image_path': image_path,
                    'orig_size_hw': targets[local_index]['orig_size'].detach().cpu(),
                    'targets_transformed': {
                        'boxes': targets[local_index]['boxes'].detach().cpu(),
                        'labels': targets[local_index]['labels'].detach().cpu(),
                    },
                    'targets_raw_abs': raw_gt,
                    'outputs': {
                        'pred_logits': outputs['pred_logits'][local_index].detach().cpu(),
                        'pred_boxes': outputs['pred_boxes'][local_index].detach().cpu(),
                        'pred_obj': outputs['pred_obj'][local_index].detach().cpu(),
                        'pred_known': outputs.get('pred_known', None)[local_index].detach().cpu() if outputs.get('pred_known', None) is not None else None,
                        'proj_obj': outputs.get('proj_obj', None)[local_index].detach().cpu() if outputs.get('proj_obj', None) is not None else None,
                        'proj_known': outputs.get('proj_known', None)[local_index].detach().cpu() if outputs.get('proj_known', None) is not None else None,
                        'proj_cls': outputs.get('proj_cls', None)[local_index].detach().cpu() if outputs.get('proj_cls', None) is not None else None,
                    },
                    'aux_outputs': simplify_aux_outputs(outputs.get('aux_outputs', []), local_index),
                    'postprocess': {
                        'scores': results[local_index]['scores'].detach().cpu(),
                        'labels': results[local_index]['labels'].detach().cpu(),
                        'boxes_abs_xyxy': results[local_index]['boxes'].detach().cpu(),
                    },
                    'query_groups': {
                        'categories': query_categories[local_index].detach().cpu(),
                    },
                    'pseudo_mining': pseudo_debug[local_index],
                    'odqe_hooks': hook_recorder.package_image(local_index) if hook_recorder.enabled else {},
                    'config_meta': {
                        'num_classes': int(args.num_classes),
                        'hidden_dim': int(args.hidden_dim),
                        'invalid_cls_logits': list(invalid_cls_logits),
                        'obj_temperature': float(args.obj_temp) / float(args.hidden_dim),
                        'known_temperature': float(args.uod_known_temp) / float(args.hidden_dim),
                        'unknown_scale': float(args.uod_postprocess_unknown_scale),
                    },
                }
                out_path = per_image_dir / f'{image_id}.pt'
                torch.save(entry, out_path)
                total_selected += 1
                saved_records.append({'image_id': image_id, 'dataset_index': dataset_index, 'file': str(out_path.relative_to(save_dir)), 'image_path': image_path})
                logging.info('Saved per-image dump %s (%d/%d)', out_path.name, total_selected, args.dump_max_images)

            if (batch_index + 1) % max(1, int(args.log_every)) == 0:
                logging.info('Processed %d / %d images', processed, len(eval_dataset))

    if enable_odqe_hooks:
        hook_recorder.close()

    global_stats = finalize_global_stats(global_lists)
    torch.save(global_stats, save_dir / 'global_stats.pt')
    manifest = {
        'checkpoint': str(args.eval_checkpoint),
        'dataset': args.dataset,
        'test_set': args.test_set,
        'model_type': args.model_type,
        'save_dir': str(save_dir),
        'num_eval_images': len(eval_dataset),
        'num_saved_images': len(saved_records),
        'dump_strategy': {
            'requested_image_ids_file': args.dump_image_ids_file,
            'dump_every_n': int(args.dump_every_n),
            'dump_max_images': int(args.dump_max_images),
            'dump_start_index': int(args.dump_start_index),
        },
        'saved_records': saved_records,
        'config': maybe_tensor_to_list(vars(args)),
    }
    with open(save_dir / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    logging.info('Wrote manifest.json and global_stats.pt to %s', save_dir)


if __name__ == '__main__':
    main()
