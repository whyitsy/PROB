import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from tools.uod_debug_common import (
    build_loader,
    build_model_bundle,
    build_owod_dataset,
    build_repo_parser,
    checkpoint_load_to_model,
    cxcywh_to_xyxy,
    ensure_dir,
    image_id_from_target,
    image_path_map_from_dataset,
    save_json,
    scale_xyxy_to_orig,
    tensor_to_list,
    to_device_targets,
)


def _bool_mask_from_indices(num_queries: int, idxs: List[int]) -> np.ndarray:
    mask = np.zeros(num_queries, dtype=np.uint8)
    if len(idxs) > 0:
        mask[np.asarray(idxs, dtype=np.int64)] = 1
    return mask


def _tensor_mask_from_indices(num_queries: int, idxs: List[int], device) -> torch.Tensor:
    mask = torch.zeros(num_queries, dtype=torch.bool, device=device)
    if len(idxs) > 0:
        mask[torch.as_tensor(idxs, dtype=torch.long, device=device)] = True
    return mask


def _compute_conf(e: float, k: float, u: float, pos_thresh: float, known_reject_thresh: float, iou_q: float, max_iou: float) -> float:
    energy_rel = max(0.0, min(1.0, (pos_thresh - e) / max(pos_thresh, 1e-6)))
    known_rel = max(0.0, min(1.0, (known_reject_thresh - k) / max(known_reject_thresh, 1e-6)))
    iou_rel = 1.0 - max(0.0, min(1.0, iou_q / max(max_iou, 1e-6)))
    unk_rel = max(0.0, min(1.0, u))
    return float((energy_rel * known_rel * iou_rel * max(unk_rel, 1e-6)) ** 0.25)


def replay_uod_mining_debug(outputs, targets, indices, criterion, stage_epoch: int = 999, gt_overlap_mode: str = 'iou_iof'):
    batch_size = len(targets)
    device = outputs['pred_boxes'].device
    num_queries = outputs['pred_boxes'].shape[1]
    hidden_dim = float(criterion.hidden_dim)

    dummy_pos_indices = [[] for _ in range(batch_size)]
    dummy_neg_indices = [[] for _ in range(batch_size)]
    dummy_pos_weights = [[] for _ in range(batch_size)]
    ignore_query_indices = [[] for _ in range(batch_size)]

    image_debugs: List[Dict[str, Any]] = []
    per_img_pos_candidates: List[List[Tuple[int, int, float, float, float, float, float]]] = []
    per_img_cache: List[Dict[str, Any]] = []
    all_pos_candidates: List[Tuple[int, int, float, float, float, float, float]] = []

    energy = outputs['pred_obj'].detach() / hidden_dim
    pred_boxes = outputs['pred_boxes'].detach()
    fused = criterion._compute_fused_probabilities(outputs)
    obj_prob = fused['obj_prob'].detach()
    unknown_prob = fused['unknown_prob'].detach()
    unknown_score = fused['unknown_score'].detach()
    known_max = fused['max_known_cls_prob'].detach()

    for i, (src_idx, _) in enumerate(indices):
        matched = sorted(set(src_idx.tolist()))
        unmatched = [q for q in range(num_queries) if q not in set(matched)]

        if len(src_idx) > 0:
            matched_scores = energy[i, src_idx]
            mu_obj = matched_scores.mean().item()
            std_obj = matched_scores.std().item() if len(src_idx) > 1 else 0.0
            pos_thresh = max(mu_obj + 3.0 * std_obj, criterion.uod_min_pos_thresh)
        else:
            mu_obj = 0.0
            std_obj = 0.0
            pos_thresh = criterion.uod_min_pos_thresh

        pred_xyxy = cxcywh_to_xyxy(pred_boxes[i])
        gt_xyxy = cxcywh_to_xyxy(targets[i]['boxes'])

        max_iou_to_gt = pred_boxes.new_zeros((num_queries,))
        max_iof_to_gt = pred_boxes.new_zeros((num_queries,))
        drop_by_gt: List[int] = []
        keep_after_gt: List[int] = list(unmatched)
        iou_map = {q: 0.0 for q in unmatched}

        if gt_xyxy.numel() > 0 and len(unmatched) > 0:
            cand_boxes = pred_xyxy[unmatched]
            ious = criterion._pairwise_iof(cand_boxes, gt_xyxy) * 0.0
            try:
                from util import box_ops
                ious = box_ops.box_iou(cand_boxes, gt_xyxy)[0]
            except Exception:
                pass
            iofs = criterion._pairwise_iof(cand_boxes, gt_xyxy)
            max_iou = ious.max(dim=1)[0]
            max_iof = iofs.max(dim=1)[0]
            keep_after_gt = []
            for j, q in enumerate(unmatched):
                iou_q = float(max_iou[j].item())
                iof_q = float(max_iof[j].item())
                iou_map[q] = iou_q
                max_iou_to_gt[q] = iou_q
                max_iof_to_gt[q] = iof_q
                if gt_overlap_mode == 'none':
                    keep = True
                elif gt_overlap_mode == 'iou_only':
                    keep = iou_q < criterion.uod_max_iou
                else:
                    keep = (iou_q < criterion.uod_max_iou) and (iof_q < criterion.uod_max_iof)
                if keep:
                    keep_after_gt.append(q)
                else:
                    drop_by_gt.append(q)

        drop_by_geom: List[int] = []
        keep_after_geom: List[int] = []
        for q in keep_after_gt:
            if criterion._is_valid_geometry(pred_boxes[i, q]):
                keep_after_geom.append(q)
            else:
                drop_by_geom.append(q)

        drop_by_unk_min: List[int] = []
        drop_by_pos_thresh: List[int] = []
        drop_by_known_reject: List[int] = []
        pos_candidates_pre_nms: List[Tuple[int, int, float, float, float, float, float]] = []
        conf_scores = pred_boxes.new_full((num_queries,), -1.0)

        for q in keep_after_geom:
            e = float(energy[i, q].item())
            k = float(known_max[i, q].item())
            u = float(unknown_prob[i, q].item())
            us = float(unknown_score[i, q].item())
            if u < criterion.uod_pos_unk_min:
                drop_by_unk_min.append(q)
                continue
            if e >= pos_thresh:
                drop_by_pos_thresh.append(q)
                continue
            if k >= criterion.uod_known_reject_thresh:
                drop_by_known_reject.append(q)
                continue
            conf = _compute_conf(
                e=e,
                k=k,
                u=u,
                pos_thresh=pos_thresh,
                known_reject_thresh=criterion.uod_known_reject_thresh,
                iou_q=iou_map.get(q, 0.0),
                max_iou=criterion.uod_max_iou,
            )
            conf_scores[q] = conf
            pos_candidates_pre_nms.append((i, q, conf, e, k, u, us))

        pos_candidates = criterion._deduplicate_pos_candidates(pred_boxes[i], pos_candidates_pre_nms, criterion.uod_candidate_nms_iou)
        kept_after_nms = {item[1] for item in pos_candidates}
        drop_by_candidate_nms = [item[1] for item in pos_candidates_pre_nms if item[1] not in kept_after_nms]

        per_img_pos_candidates.append(pos_candidates)
        per_img_cache.append({'valid': keep_after_geom, 'pred_xyxy': pred_xyxy})
        all_pos_candidates.extend(pos_candidates)

        image_debugs.append({
            'matched': matched,
            'unmatched': unmatched,
            'drop_by_gt': drop_by_gt,
            'keep_after_gt': keep_after_gt,
            'drop_by_geom': drop_by_geom,
            'keep_after_geom': keep_after_geom,
            'drop_by_unk_min': drop_by_unk_min,
            'drop_by_pos_thresh': drop_by_pos_thresh,
            'drop_by_known_reject': drop_by_known_reject,
            'pos_candidates_pre_nms': [item[1] for item in pos_candidates_pre_nms],
            'drop_by_candidate_nms': drop_by_candidate_nms,
            'pos_candidates': [item[1] for item in pos_candidates],
            'selected_pos': [],
            'selected_neg': [],
            'ignore': [],
            'dummy_pos_weights': [],
            'pos_thresh': float(pos_thresh),
            'matched_mu_obj': float(mu_obj),
            'matched_std_obj': float(std_obj),
            'energy': energy[i].detach().cpu().numpy(),
            'obj_prob': obj_prob[i].detach().cpu().numpy(),
            'unknown_prob': unknown_prob[i].detach().cpu().numpy(),
            'unknown_score': unknown_score[i].detach().cpu().numpy(),
            'known_max': known_max[i].detach().cpu().numpy(),
            'conf': conf_scores.detach().cpu().numpy(),
            'pred_boxes_xyxy_norm': pred_xyxy.detach().cpu().numpy(),
            'gt_boxes_xyxy_norm': gt_xyxy.detach().cpu().numpy(),
            'gt_labels': targets[i]['labels'].detach().cpu().numpy(),
            'max_iou_to_gt': max_iou_to_gt.detach().cpu().numpy(),
            'max_iof_to_gt': max_iof_to_gt.detach().cpu().numpy(),
        })

    pseudo_active = bool(criterion.enable_pseudo) and int(stage_epoch) >= int(getattr(criterion, 'uod_start_epoch', 0))
    neg_active = pseudo_active and int(stage_epoch) >= int(getattr(criterion, 'uod_start_epoch', 0)) + int(getattr(criterion, 'uod_neg_warmup_epochs', 0))

    if pseudo_active and criterion.enable_batch_dynamic:
        all_pos_candidates.sort(key=lambda x: (-x[2], -x[6], -x[5], x[3], x[4]))
        topk = min(
            criterion.uod_batch_topk_max,
            max(1, int(np.ceil(criterion.uod_batch_topk_ratio * max(len(all_pos_candidates), 1))))
        )
        per_img_count = [0 for _ in range(batch_size)]
        selected = []
        for item in all_pos_candidates:
            b_idx, q, conf, e, k, u, us = item
            if len(selected) >= topk:
                break
            if criterion.uod_pos_per_img_cap > 0 and per_img_count[b_idx] >= criterion.uod_pos_per_img_cap:
                continue
            selected.append(item)
            per_img_count[b_idx] += 1
        for b_idx, q, conf, e, k, u, us in selected:
            dummy_pos_indices[b_idx].append(q)
            dummy_pos_weights[b_idx].append(float(max(0.2, min(1.0, conf))))
    elif pseudo_active:
        for i, pos_candidates in enumerate(per_img_pos_candidates):
            pos_candidates.sort(key=lambda x: (-x[2], -x[6], -x[5], x[3], x[4]))
            if criterion.uod_pos_per_img_cap > 0:
                pos_candidates = pos_candidates[:criterion.uod_pos_per_img_cap]
            dummy_pos_indices[i] = [q for _, q, _, _, _, _, _ in pos_candidates]
            dummy_pos_weights[i] = [float(max(0.2, min(1.0, conf))) for _, _, conf, _, _, _, _ in pos_candidates]

    for i in range(batch_size):
        image_debugs[i]['selected_pos'] = list(dummy_pos_indices[i])
        image_debugs[i]['dummy_pos_weights'] = list(dummy_pos_weights[i])

    if neg_active:
        # The standalone collector assumes mining is active for the requested stage_epoch.
        for i in range(batch_size):
            valid = per_img_cache[i]['valid']
            pred_xyxy = per_img_cache[i]['pred_xyxy']
            pos_selected = dummy_pos_indices[i]
            pos_selected_set = set(pos_selected)
            remaining = [q for q in valid if q not in pos_selected_set]
            remaining = criterion._filter_negatives_near_selected_pos(pred_xyxy, pos_selected, remaining)

            neg_candidates = []
            for q in remaining:
                k = float(known_max[i, q].item())
                obj = float(obj_prob[i, q].item())
                e = float(energy[i, q].item())
                if k > criterion.uod_neg_known_max:
                    continue
                neg_candidates.append((q, obj, e, k))
            neg_candidates.sort(key=lambda x: (-x[1], x[2], x[3]))
            if criterion.uod_neg_per_img > 0:
                neg_candidates = neg_candidates[:criterion.uod_neg_per_img]
            dummy_neg_indices[i] = [q for q, _, _, _ in neg_candidates]
            image_debugs[i]['selected_neg'] = list(dummy_neg_indices[i])

            pos_set = set(dummy_pos_indices[i])
            neg_set = set(dummy_neg_indices[i])
            ignore = []
            for q in valid:
                if q in pos_set or q in neg_set:
                    continue
                if (
                    float(obj_prob[i, q].item()) > 0.05
                    and float(unknown_prob[i, q].item()) >= criterion.uod_pos_unk_min
                    and float(known_max[i, q].item()) < criterion.uod_known_reject_thresh
                ):
                    ignore.append(q)
            ignore_query_indices[i] = ignore
            image_debugs[i]['ignore'] = list(ignore)

    return image_debugs, dummy_pos_indices, dummy_neg_indices, dummy_pos_weights, ignore_query_indices


def _write_npz(npz_path: Path, meta_path: Path, payload: Dict[str, Any], meta: Dict[str, Any]):
    np_payload = {
        'pred_boxes_xyxy': payload['pred_boxes_xyxy'],
        'gt_boxes_xyxy': payload['gt_boxes_xyxy'],
        'gt_labels': payload['gt_labels'],
        'energy': payload['energy'],
        'obj_prob': payload['obj_prob'],
        'unknown_prob': payload['unknown_prob'],
        'unknown_score': payload['unknown_score'],
        'known_max': payload['known_max'],
        'conf': payload['conf'],
        'max_iou_to_gt': payload['max_iou_to_gt'],
        'max_iof_to_gt': payload['max_iof_to_gt'],
    }
    mask_keys = [
        'matched_mask', 'unmatched_mask', 'drop_by_gt_mask', 'keep_after_gt_mask',
        'drop_by_geom_mask', 'keep_after_geom_mask', 'drop_by_unk_min_mask',
        'drop_by_pos_thresh_mask', 'drop_by_known_reject_mask',
        'pos_candidates_pre_nms_mask', 'drop_by_candidate_nms_mask', 'pos_candidates_mask',
        'selected_pos_mask', 'selected_neg_mask', 'ignore_mask'
    ]
    for key in mask_keys:
        np_payload[key] = payload[key]
    np.savez_compressed(npz_path, **np_payload)
    save_json(meta_path, meta)


def main():
    parser = build_repo_parser()
    args = parser.parse_args()

    out_root = ensure_dir(args.output_dir_debug)
    raw_dir = ensure_dir(str(out_root / 'raw'))
    meta_dir = ensure_dir(str(out_root / 'meta'))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    model, criterion, _, _ = build_model_bundle(args)
    checkpoint = checkpoint_load_to_model(model, args.resume)
    model.to(device)
    criterion.to(device)
    model.eval()
    criterion.eval()

    dataset = build_owod_dataset(args, args.split)
    loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    imgid_to_path, imgid_to_name = image_path_map_from_dataset(dataset)

    image_summary_rows: List[Dict[str, Any]] = []
    query_jsonl_path = out_root / 'query_records.jsonl'
    manifest_path = out_root / 'manifest.jsonl'
    meta = {
        'resume': args.resume,
        'split': args.split,
        'stage_epoch': int(args.stage_epoch),
        'stage_name': args.stage_name,
        'gt_overlap_mode': args.gt_overlap_mode,
        'batch_size': int(args.batch_size),
        'num_workers': int(args.num_workers),
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'model_type': getattr(args, 'model_type', None),
        'uod_start_epoch': getattr(args, 'uod_start_epoch', None),
        'uod_neg_warmup_epochs': getattr(args, 'uod_neg_warmup_epochs', None),
        'uod_min_pos_thresh': getattr(args, 'uod_min_pos_thresh', None),
        'uod_known_reject_thresh': getattr(args, 'uod_known_reject_thresh', None),
        'uod_pos_unk_min': getattr(args, 'uod_pos_unk_min', None),
        'uod_pos_per_img_cap': getattr(args, 'uod_pos_per_img_cap', None),
        'uod_neg_per_img': getattr(args, 'uod_neg_per_img', None),
        'uod_batch_topk_max': getattr(args, 'uod_batch_topk_max', None),
        'uod_batch_topk_ratio': getattr(args, 'uod_batch_topk_ratio', None),
        'uod_max_iou': getattr(args, 'uod_max_iou', None),
        'uod_max_iof': getattr(args, 'uod_max_iof', None),
    }
    save_json(out_root / 'run_meta.json', meta)

    fallback_img_counter = 0
    saved_images = 0

    query_fh = query_jsonl_path.open('w', encoding='utf-8') if args.collect_query_jsonl else None
    manifest_fh = manifest_path.open('w', encoding='utf-8')

    with torch.no_grad():
        for batch_idx, (samples, targets_cpu) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if args.save_images_limit > 0 and saved_images >= args.save_images_limit:
                break

            samples = samples.to(device)
            targets = to_device_targets(targets_cpu, device)
            outputs = model(samples)
            indices = criterion.matcher(
                {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']},
                targets,
            )
            image_debugs, _, _, _, _ = replay_uod_mining_debug(
                outputs=outputs,
                targets=targets,
                indices=indices,
                criterion=criterion,
                stage_epoch=args.stage_epoch,
                gt_overlap_mode=args.gt_overlap_mode,
            )

            for local_idx, (target_cpu, target_dev, debug_img) in enumerate(zip(targets_cpu, targets, image_debugs)):
                if args.save_images_limit > 0 and saved_images >= args.save_images_limit:
                    break
                image_id = image_id_from_target(target_cpu, fallback_img_counter)
                fallback_img_counter += 1
                image_path = imgid_to_path.get(image_id, '')
                image_name = imgid_to_name.get(image_id, f'{image_id}.jpg')
                stem = f'{batch_idx:05d}_{local_idx:02d}_{image_id}'
                npz_path = raw_dir / f'{stem}.npz'
                json_path = meta_dir / f'{stem}.json'

                orig_h = int(target_cpu['orig_size'][0].item())
                orig_w = int(target_cpu['orig_size'][1].item())
                pred_boxes_abs = scale_xyxy_to_orig(
                    torch.as_tensor(debug_img['pred_boxes_xyxy_norm']), orig_h=orig_h, orig_w=orig_w
                ).cpu().numpy()
                gt_boxes_abs = scale_xyxy_to_orig(
                    torch.as_tensor(debug_img['gt_boxes_xyxy_norm']), orig_h=orig_h, orig_w=orig_w
                ).cpu().numpy()

                num_queries = pred_boxes_abs.shape[0]
                payload = {
                    'pred_boxes_xyxy': pred_boxes_abs,
                    'gt_boxes_xyxy': gt_boxes_abs,
                    'gt_labels': debug_img['gt_labels'],
                    'energy': debug_img['energy'],
                    'obj_prob': debug_img['obj_prob'],
                    'unknown_prob': debug_img['unknown_prob'],
                    'unknown_score': debug_img['unknown_score'],
                    'known_max': debug_img['known_max'],
                    'conf': debug_img['conf'],
                    'max_iou_to_gt': debug_img['max_iou_to_gt'],
                    'max_iof_to_gt': debug_img['max_iof_to_gt'],
                    'matched_mask': _bool_mask_from_indices(num_queries, debug_img['matched']),
                    'unmatched_mask': _bool_mask_from_indices(num_queries, debug_img['unmatched']),
                    'drop_by_gt_mask': _bool_mask_from_indices(num_queries, debug_img['drop_by_gt']),
                    'keep_after_gt_mask': _bool_mask_from_indices(num_queries, debug_img['keep_after_gt']),
                    'drop_by_geom_mask': _bool_mask_from_indices(num_queries, debug_img['drop_by_geom']),
                    'keep_after_geom_mask': _bool_mask_from_indices(num_queries, debug_img['keep_after_geom']),
                    'drop_by_unk_min_mask': _bool_mask_from_indices(num_queries, debug_img['drop_by_unk_min']),
                    'drop_by_pos_thresh_mask': _bool_mask_from_indices(num_queries, debug_img['drop_by_pos_thresh']),
                    'drop_by_known_reject_mask': _bool_mask_from_indices(num_queries, debug_img['drop_by_known_reject']),
                    'pos_candidates_pre_nms_mask': _bool_mask_from_indices(num_queries, debug_img['pos_candidates_pre_nms']),
                    'drop_by_candidate_nms_mask': _bool_mask_from_indices(num_queries, debug_img['drop_by_candidate_nms']),
                    'pos_candidates_mask': _bool_mask_from_indices(num_queries, debug_img['pos_candidates']),
                    'selected_pos_mask': _bool_mask_from_indices(num_queries, debug_img['selected_pos']),
                    'selected_neg_mask': _bool_mask_from_indices(num_queries, debug_img['selected_neg']),
                    'ignore_mask': _bool_mask_from_indices(num_queries, debug_img['ignore']),
                }
                meta_row = {
                    'stem': stem,
                    'batch_idx': batch_idx,
                    'local_idx': local_idx,
                    'image_id': image_id,
                    'image_name': image_name,
                    'image_path': image_path,
                    'orig_h': orig_h,
                    'orig_w': orig_w,
                    'num_queries': int(num_queries),
                    'num_gt': int(gt_boxes_abs.shape[0]),
                    'pos_thresh': float(debug_img['pos_thresh']),
                    'matched_mu_obj': float(debug_img['matched_mu_obj']),
                    'matched_std_obj': float(debug_img['matched_std_obj']),
                    'dummy_pos_weights': [float(x) for x in debug_img['dummy_pos_weights']],
                    'counts': {
                        'matched': len(debug_img['matched']),
                        'unmatched': len(debug_img['unmatched']),
                        'drop_by_gt': len(debug_img['drop_by_gt']),
                        'keep_after_gt': len(debug_img['keep_after_gt']),
                        'drop_by_geom': len(debug_img['drop_by_geom']),
                        'keep_after_geom': len(debug_img['keep_after_geom']),
                        'drop_by_unk_min': len(debug_img['drop_by_unk_min']),
                        'drop_by_pos_thresh': len(debug_img['drop_by_pos_thresh']),
                        'drop_by_known_reject': len(debug_img['drop_by_known_reject']),
                        'pos_candidates_pre_nms': len(debug_img['pos_candidates_pre_nms']),
                        'drop_by_candidate_nms': len(debug_img['drop_by_candidate_nms']),
                        'pos_candidates': len(debug_img['pos_candidates']),
                        'selected_pos': len(debug_img['selected_pos']),
                        'selected_neg': len(debug_img['selected_neg']),
                        'ignore': len(debug_img['ignore']),
                    },
                    'lists': {
                        'matched': debug_img['matched'],
                        'unmatched': debug_img['unmatched'],
                        'drop_by_gt': debug_img['drop_by_gt'],
                        'keep_after_gt': debug_img['keep_after_gt'],
                        'drop_by_geom': debug_img['drop_by_geom'],
                        'keep_after_geom': debug_img['keep_after_geom'],
                        'drop_by_unk_min': debug_img['drop_by_unk_min'],
                        'drop_by_pos_thresh': debug_img['drop_by_pos_thresh'],
                        'drop_by_known_reject': debug_img['drop_by_known_reject'],
                        'pos_candidates_pre_nms': debug_img['pos_candidates_pre_nms'],
                        'drop_by_candidate_nms': debug_img['drop_by_candidate_nms'],
                        'pos_candidates': debug_img['pos_candidates'],
                        'selected_pos': debug_img['selected_pos'],
                        'selected_neg': debug_img['selected_neg'],
                        'ignore': debug_img['ignore'],
                    },
                }

                _write_npz(npz_path, json_path, payload, meta_row)
                manifest_fh.write(json.dumps({'stem': stem, 'npz': str(npz_path), 'meta': str(json_path)}, ensure_ascii=False) + '\n')
                image_summary_rows.append({
                    'stem': stem,
                    'image_id': image_id,
                    'image_name': image_name,
                    'pos_thresh': float(debug_img['pos_thresh']),
                    'matched_mu_obj': float(debug_img['matched_mu_obj']),
                    'matched_std_obj': float(debug_img['matched_std_obj']),
                    **meta_row['counts'],
                })

                if query_fh is not None:
                    roles = np.full((num_queries,), 'none', dtype=object)
                    roles[payload['selected_pos_mask'].astype(bool)] = 'pos'
                    roles[payload['selected_neg_mask'].astype(bool)] = 'neg'
                    roles[payload['ignore_mask'].astype(bool)] = 'ignore'

                    removed_by = np.full((num_queries,), 'none', dtype=object)
                    stage_order = [
                        ('drop_by_gt_mask', 'gt_overlap'),
                        ('drop_by_geom_mask', 'geometry'),
                        ('drop_by_unk_min_mask', 'unk_min'),
                        ('drop_by_pos_thresh_mask', 'pos_thresh'),
                        ('drop_by_known_reject_mask', 'known_reject'),
                        ('drop_by_candidate_nms_mask', 'candidate_nms'),
                    ]
                    for mask_name, label in stage_order:
                        mask = payload[mask_name].astype(bool)
                        removed_by[mask] = label
                    removed_by[payload['pos_candidates_mask'].astype(bool)] = 'kept_candidate'
                    removed_by[payload['selected_pos_mask'].astype(bool)] = 'selected_pos'
                    removed_by[payload['selected_neg_mask'].astype(bool)] = 'selected_neg'
                    removed_by[payload['ignore_mask'].astype(bool)] = 'ignore'
                    removed_by[payload['matched_mask'].astype(bool)] = 'matched'

                    for q in range(num_queries):
                        row = {
                            'stem': stem,
                            'image_id': image_id,
                            'qid': q,
                            'energy': float(payload['energy'][q]),
                            'obj_prob': float(payload['obj_prob'][q]),
                            'unknown_prob': float(payload['unknown_prob'][q]),
                            'unknown_score': float(payload['unknown_score'][q]),
                            'known_max': float(payload['known_max'][q]),
                            'conf': float(payload['conf'][q]),
                            'max_iou_to_gt': float(payload['max_iou_to_gt'][q]),
                            'max_iof_to_gt': float(payload['max_iof_to_gt'][q]),
                            'removed_by': str(removed_by[q]),
                            'final_role': str(roles[q]),
                            'matched': int(payload['matched_mask'][q]),
                            'unmatched': int(payload['unmatched_mask'][q]),
                            'box_xyxy': [float(x) for x in payload['pred_boxes_xyxy'][q].tolist()],
                        }
                        query_fh.write(json.dumps(row, ensure_ascii=False) + '\n')

                saved_images += 1

            if batch_idx % 10 == 0:
                print(f'[INFO] processed batch {batch_idx}, saved_images={saved_images}')

    if query_fh is not None:
        query_fh.close()
    manifest_fh.close()

    summary_csv = out_root / 'image_summary.csv'
    fieldnames = [
        'stem', 'image_id', 'image_name', 'pos_thresh', 'matched_mu_obj', 'matched_std_obj',
        'matched', 'unmatched', 'drop_by_gt', 'keep_after_gt', 'drop_by_geom', 'keep_after_geom',
        'drop_by_unk_min', 'drop_by_pos_thresh', 'drop_by_known_reject',
        'pos_candidates_pre_nms', 'drop_by_candidate_nms', 'pos_candidates',
        'selected_pos', 'selected_neg', 'ignore'
    ]
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in image_summary_rows:
            writer.writerow(row)

    aggregate = {
        'num_images': len(image_summary_rows),
        'sum_selected_pos': int(sum(row['selected_pos'] for row in image_summary_rows)),
        'sum_selected_neg': int(sum(row['selected_neg'] for row in image_summary_rows)),
        'sum_ignore': int(sum(row['ignore'] for row in image_summary_rows)),
        'sum_drop_by_gt': int(sum(row['drop_by_gt'] for row in image_summary_rows)),
        'sum_drop_by_geom': int(sum(row['drop_by_geom'] for row in image_summary_rows)),
        'sum_drop_by_unk_min': int(sum(row['drop_by_unk_min'] for row in image_summary_rows)),
        'sum_drop_by_pos_thresh': int(sum(row['drop_by_pos_thresh'] for row in image_summary_rows)),
        'sum_drop_by_known_reject': int(sum(row['drop_by_known_reject'] for row in image_summary_rows)),
        'sum_drop_by_candidate_nms': int(sum(row['drop_by_candidate_nms'] for row in image_summary_rows)),
    }
    save_json(out_root / 'aggregate_summary.json', aggregate)
    print(f'[OK] wrote raw debug outputs for {len(image_summary_rows)} images to {out_root}')


if __name__ == '__main__':
    main()
