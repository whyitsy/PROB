import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from util import box_ops
from datasets.data_prefetcher import data_prefetcher
import util.misc as utils


def _normalize(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return F.normalize(x, p=2, dim=-1)


def resolve_proto_memory_file(args) -> str:
    explicit = getattr(args, 'proto_memory_file', '') or ''
    if explicit and os.path.exists(explicit):
        return explicit
    pretrain = getattr(args, 'pretrain', '') or ''
    if pretrain:
        cand = os.path.join(os.path.dirname(pretrain), getattr(args, 'proto_memory_name', 'prototype_memory.pt'))
        if os.path.exists(cand):
            return cand
    return ''


def _compute_iof(candidates_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    # candidates: [N,4], gt:[M,4] -> IoF wrt candidate area [N,M]
    if candidates_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
        return torch.zeros((candidates_xyxy.shape[0], gt_xyxy.shape[0]), device=candidates_xyxy.device)
    lt = torch.max(candidates_xyxy[:, None, :2], gt_xyxy[None, :, :2])
    rb = torch.min(candidates_xyxy[:, None, 2:], gt_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    cand_area = ((candidates_xyxy[:, 2] - candidates_xyxy[:, 0]).clamp(min=1e-6) *
                 (candidates_xyxy[:, 3] - candidates_xyxy[:, 1]).clamp(min=1e-6))
    return inter / cand_area[:, None]


def mine_unknown_candidates_for_memory(outputs, targets, criterion, args):
    outputs_without_aux = {
        k: v for k, v in outputs.items()
        if k not in ['aux_outputs', 'enc_outputs', 'pred_obj', 'pred_unk', 'samples', 'pred_proj', 'pred_embed']
    }
    indices = criterion.matcher(outputs_without_aux, targets)

    obj_scores = outputs['pred_obj']
    pred_logits = outputs['pred_logits'].detach().sigmoid().clone()
    pred_logits[:, :, criterion.invalid_cls_logits] = 0.0
    pred_boxes = outputs['pred_boxes']
    pred_embed = outputs.get('pred_embed', None)

    max_pos_per_img = max(1, int(getattr(args, 'proto_candidate_topk_per_image', getattr(args, 'unk_pos_per_img', 1))))
    known_reject_thresh = getattr(args, 'unk_cls_reject_thresh', 0.25)
    pos_quantile = getattr(args, 'unk_label_pos_quantile', 0.5)
    unk_max_iou = getattr(args, 'unk_max_iou', 0.3)
    unk_max_iof = getattr(args, 'unk_max_iof', 0.6)
    proto_min_score = getattr(args, 'proto_min_score', 0.10)
    obj_temp = getattr(args, 'obj_temp', 1.0) / getattr(args, 'hidden_dim', 256)

    all_feats: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []
    all_meta: List[Dict] = []

    for i, (src_idx, _) in enumerate(indices):
        matched_scores = obj_scores[i, src_idx]
        if len(matched_scores) > 0:
            base_thresh = torch.quantile(matched_scores.detach(), pos_quantile).item()
            pos_thresh = base_thresh * getattr(args, 'unk_label_obj_score_thresh', 1.0)
        else:
            pos_thresh = getattr(args, 'default_pos_energy_thresh', 1.0)

        num_queries = obj_scores.shape[1]
        matched_set = set(src_idx.tolist())
        unmatched = [q for q in range(num_queries) if q not in matched_set]
        if not unmatched:
            continue

        box_xyxy_all = box_ops.box_cxcywh_to_xyxy(pred_boxes[i])
        gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(targets[i]['boxes']) if len(targets[i]['boxes']) > 0 else None

        valid_unmatched = unmatched
        unmatched_iou_map = {q: 0.0 for q in unmatched}
        if gt_boxes_xyxy is not None and len(gt_boxes_xyxy) > 0:
            cand_boxes = box_xyxy_all[unmatched]
            ious = box_ops.box_iou(cand_boxes, gt_boxes_xyxy)[0]
            max_ious = ious.max(dim=1)[0]
            iofs = _compute_iof(cand_boxes, gt_boxes_xyxy)
            max_iofs = iofs.max(dim=1)[0]
            valid_unmatched = [
                unmatched[j] for j in range(len(unmatched))
                if max_ious[j].item() < unk_max_iou and max_iofs[j].item() < unk_max_iof
            ]
            unmatched_iou_map = {unmatched[j]: max_ious[j].item() for j in range(len(unmatched))}

        geom_valid = []
        for q in valid_unmatched:
            if criterion._is_valid_unknown_geometry(pred_boxes[i, q]) and float(pred_boxes[i, q, 2]) >= criterion.unk_min_side and float(pred_boxes[i, q, 3]) >= criterion.unk_min_side:
                geom_valid.append(q)
        if not geom_valid:
            continue

        known_prob = pred_logits[i, :, :criterion.num_classes - 1]
        known_max = known_prob.max(dim=-1)[0]
        low_energy = [q for q in geom_valid if obj_scores[i, q].item() < pos_thresh]
        if not low_energy:
            continue

        num_unmatched = max(len(unmatched), 1)
        low_energy_known_mean = float(known_max[low_energy].mean().item()) if low_energy else 1.0
        gate_open = (
            len(geom_valid) / float(num_unmatched) >= getattr(args, 'image_gate_min_valid_ratio', 0.05)
            and len(low_energy) / float(num_unmatched) >= getattr(args, 'image_gate_min_low_energy_ratio', 0.02)
            and len(low_energy) >= getattr(args, 'image_gate_min_pos_candidates', 1)
            and low_energy_known_mean <= getattr(args, 'image_gate_known_mean_max', 0.25)
        )
        if not gate_open:
            continue

        candidates = []
        for q in low_energy:
            if known_max[q].item() >= known_reject_thresh:
                continue
            obj_prob = torch.exp(-obj_temp * obj_scores[i, q]).item()
            merged_score = obj_prob * (1.0 - known_max[q].item())
            if merged_score < proto_min_score:
                continue
            weight = max(0.05, min(1.0, 1.0 - known_max[q].item()))
            candidates.append((q, merged_score, weight, unmatched_iou_map.get(q, 0.0)))

        candidates.sort(key=lambda t: (-t[1], t[3]))
        candidates = candidates[:max_pos_per_img]
        if not candidates:
            continue
        if pred_embed is None:
            continue

        img_id = targets[i]['image_id'].item() if 'image_id' in targets[i] else i
        for q, score, weight, max_iou in candidates:
            all_feats.append(pred_embed[i, q].detach())
            all_scores.append(torch.tensor(score * weight, dtype=pred_embed.dtype, device=pred_embed.device))
            all_meta.append({'image_id': int(img_id), 'query_index': int(q), 'max_iou': float(max_iou), 'score': float(score)})

    if len(all_feats) == 0:
        return None

    feats = torch.stack(all_feats, dim=0)
    scores = torch.stack(all_scores, dim=0)
    return {'features': feats, 'scores': scores, 'meta': all_meta}


def greedy_cluster_prototypes(features: torch.Tensor, scores: torch.Tensor, max_prototypes: int, assign_cos: float):
    feats = _normalize(features)
    scores = scores.float()
    order = torch.argsort(scores, descending=True)
    feats = feats[order]
    scores = scores[order]

    centers: List[torch.Tensor] = []
    totals: List[float] = []
    counts: List[int] = []
    exemplar_indices: List[int] = []

    for idx in range(feats.shape[0]):
        feat = feats[idx]
        weight = float(scores[idx].item())
        if len(centers) == 0:
            centers.append(feat.clone())
            totals.append(weight)
            counts.append(1)
            exemplar_indices.append(int(order[idx].item()))
            continue
        stacked = torch.stack(centers, dim=0)
        sim = torch.matmul(stacked, feat)
        best_sim, best_idx = sim.max(dim=0)
        if best_sim.item() < assign_cos and len(centers) < max_prototypes:
            centers.append(feat.clone())
            totals.append(weight)
            counts.append(1)
            exemplar_indices.append(int(order[idx].item()))
        else:
            j = int(best_idx.item())
            new_center = _normalize((centers[j] * totals[j] + feat * weight).unsqueeze(0))[0]
            centers[j] = new_center
            totals[j] += weight
            counts[j] += 1
            if weight > totals[j] / max(counts[j], 1):
                exemplar_indices[j] = int(order[idx].item())

    centers_t = torch.stack(centers, dim=0) if centers else torch.empty((0, feats.shape[1]), device=features.device)
    totals_t = torch.tensor(totals, dtype=torch.float32, device=features.device)
    counts_t = torch.tensor(counts, dtype=torch.long, device=features.device)
    exemplar_t = torch.tensor(exemplar_indices, dtype=torch.long, device=features.device) if exemplar_indices else torch.empty((0,), dtype=torch.long, device=features.device)

    if centers_t.shape[0] > max_prototypes:
        order2 = torch.argsort(totals_t, descending=True)[:max_prototypes]
        centers_t = centers_t[order2]
        totals_t = totals_t[order2]
        counts_t = counts_t[order2]
        exemplar_t = exemplar_t[order2]

    return centers_t, totals_t, counts_t, exemplar_t


@torch.no_grad()
def build_unknown_prototype_memory(model, criterion, device, data_loader, args, output_path: str):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = '[ProtoMemory]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    feature_list = []
    score_list = []
    meta_all: List[Dict] = []
    max_candidates = int(getattr(args, 'proto_max_candidates', 4000))

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        if samples is None:
            break
        outputs = model(samples)
        mined = mine_unknown_candidates_for_memory(outputs, targets, criterion, args)
        if mined is not None:
            feature_list.append(mined['features'].detach().cpu())
            score_list.append(mined['scores'].detach().cpu())
            meta_all.extend(mined['meta'])
        if len(meta_all) >= max_candidates:
            break
        samples, targets = prefetcher.next()

    if len(feature_list) == 0:
        logging.warning('No unknown candidates found for prototype memory. Skip saving.')
        return ''

    features = torch.cat(feature_list, dim=0)
    scores = torch.cat(score_list, dim=0)
    if features.shape[0] > max_candidates:
        keep = torch.argsort(scores, descending=True)[:max_candidates]
        features = features[keep]
        scores = scores[keep]
        meta_all = [meta_all[int(i)] for i in keep.tolist()]

    centers, totals, counts, exemplar_indices = greedy_cluster_prototypes(
        features=features,
        scores=scores,
        max_prototypes=int(getattr(args, 'proto_num_prototypes', 64)),
        assign_cos=float(getattr(args, 'proto_assign_cos', 0.75)),
    )

    payload = {
        'prototypes': _normalize(centers).cpu(),
        'scores': totals.cpu(),
        'counts': counts.cpu(),
        'exemplar_indices': exemplar_indices.cpu(),
        'meta': meta_all,
        'source_train_set': getattr(args, 'train_set', ''),
        'source_output_dir': getattr(args, 'output_dir', ''),
        'num_candidates': int(features.shape[0]),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    logging.info('Saved prototype memory to %s with %d prototypes from %d candidates', output_path, int(payload['prototypes'].shape[0]), int(features.shape[0]))
    return output_path
