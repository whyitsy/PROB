import numpy as np
import torch

from models.prob_deformable_detr_uod import _compute_uod_fused_probabilities


QUERY_METADATA_KEYS = [
    'is_matched',
    'matched_gt_label',
    'matched_gt_is_unknown',
    'pred_top1_label',
    'pred_top1_is_unknown',
    'top1_known_class',
    'image_id',
    'query_index',
]

FEATURE_METADATA_KEYS = [f'feature_{key}' for key in QUERY_METADATA_KEYS]


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu().item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def _append_limited(destination, values, max_length):
    remaining = max_length - len(destination)
    if remaining <= 0:
        return 0
    if len(values) > remaining:
        values = values[:remaining]
    destination.extend(values)
    return len(values)


def _ensure_state_lists(state):
    for key in QUERY_METADATA_KEYS:
        state.setdefault(key, [])
    for key in FEATURE_METADATA_KEYS:
        state.setdefault(key, [])


def build_postprocess_aligned_probabilities(outputs, criterion, args):
    pred_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits')
    pred_obj = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
    if pred_logits is None or pred_obj is None:
        return None

    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temperature = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temperature = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim
    unknown_scale = float(getattr(args, 'uod_postprocess_unknown_scale', 15.0))
    invalid_class_indices = list(getattr(criterion, 'invalid_cls_logits', []))

    return _compute_uod_fused_probabilities(
        pred_logits=pred_logits,
        pred_obj=pred_obj,
        pred_known=_get_output(outputs, 'pred_knownness_energy', 'pred_known'),
        invalid_cls_logits=invalid_class_indices,
        obj_temperature=obj_temperature,
        known_temperature=known_temperature,
        unknown_scale=unknown_scale,
    )


def compute_train_query_score_statistics(outputs, targets, criterion, args):
    fused = build_postprocess_aligned_probabilities(outputs, criterion, args)
    if fused is None:
        return {}

    stats = {
        'train/query_stats/unknown_probability_mean': _safe_float(fused['unknown_prob'].mean()),
        'train/query_stats/max_known_class_probability_mean': _safe_float(fused['max_known_cls_prob'].mean()),
    }

    matcher_outputs = {
        'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'),
        'pred_boxes': outputs['pred_boxes'],
    }
    try:
        matched_indices = criterion.matcher(matcher_outputs, targets)
    except Exception:
        return stats

    batch_size, num_queries = fused['obj_prob'].shape[:2]
    matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=fused['obj_prob'].device)
    for batch_index, (source_indices, _) in enumerate(matched_indices):
        if len(source_indices) > 0:
            matched_mask[batch_index, source_indices] = True

    if matched_mask.any():
        stats['train/query_stats/matched_objectness_prob_mean'] = _safe_float(fused['obj_prob'][matched_mask].mean())
    if (~matched_mask).any():
        stats['train/query_stats/unmatched_objectness_prob_mean'] = _safe_float(fused['obj_prob'][~matched_mask].mean())
    return stats


def _build_query_metadata(outputs, targets, matched_indices, fused, args):
    unknown_label = int(getattr(args, 'num_classes', 81) - 1)
    batch_size, num_queries = fused['obj_prob'].shape[:2]
    device = fused['obj_prob'].device

    matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=device)
    matched_gt_label = torch.full((batch_size, num_queries), -1, dtype=torch.int64, device=device)

    for batch_index, (source_indices, target_indices) in enumerate(matched_indices):
        if len(source_indices) == 0:
            continue
        matched_mask[batch_index, source_indices] = True
        matched_gt_label[batch_index, source_indices] = targets[batch_index]['labels'][target_indices].to(torch.int64)

    fused_prob = fused['fused_prob']
    pred_top1_label = fused_prob.argmax(dim=-1).to(torch.int64)
    pred_top1_is_unknown = pred_top1_label == int(unknown_label)

    if fused['known_scores'].shape[-1] > 1:
        top1_known_class = fused['known_scores'][..., :-1].argmax(dim=-1).to(torch.int64)
    elif fused['known_scores'].shape[-1] == 1:
        top1_known_class = torch.zeros((batch_size, num_queries), dtype=torch.int64, device=device)
    else:
        top1_known_class = torch.full((batch_size, num_queries), -1, dtype=torch.int64, device=device)

    image_ids = []
    query_indices = []
    for batch_index in range(batch_size):
        image_id = int(targets[batch_index]['image_id'].item()) if 'image_id' in targets[batch_index] else batch_index
        image_ids.append(torch.full((num_queries,), image_id, dtype=torch.int64, device=device))
        query_indices.append(torch.arange(num_queries, dtype=torch.int64, device=device))

    metadata = {
        'is_matched': matched_mask,
        'matched_gt_label': matched_gt_label,
        'matched_gt_is_unknown': matched_gt_label == int(unknown_label),
        'pred_top1_label': pred_top1_label,
        'pred_top1_is_unknown': pred_top1_is_unknown,
        'top1_known_class': top1_known_class,
        'image_id': torch.stack(image_ids, dim=0),
        'query_index': torch.stack(query_indices, dim=0),
    }
    return metadata, matched_mask


def collect_eval_visual_stats_aligned(state, outputs, targets, criterion, args):
    _ensure_state_lists(state)

    if len(state['objectness_probability']) >= state['max_query_samples'] and len(state['objectness_features']) >= state['max_feature_samples']:
        return

    fused = build_postprocess_aligned_probabilities(outputs, criterion, args)
    if fused is None:
        return

    matcher_outputs = {
        'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'),
        'pred_boxes': outputs['pred_boxes'],
    }
    matched_indices = criterion.matcher(matcher_outputs, targets)
    metadata_tensors, matched_mask = _build_query_metadata(outputs, targets, matched_indices, fused, args)

    objectness_np = fused['obj_prob'].flatten().detach().cpu().numpy()
    unknown_np = fused['unknown_prob'].flatten().detach().cpu().numpy()
    max_known_np = fused['max_known_cls_prob'].flatten().detach().cpu().numpy()
    matched_np = matched_mask.flatten().detach().cpu().numpy()
    group_np = np.where(matched_np, 0, np.where(unknown_np > 0.5, 1, 2)).astype(np.int64)

    appended = _append_limited(state['objectness_probability'], objectness_np.tolist(), state['max_query_samples'])
    if appended > 0:
        _append_limited(state['unknown_probability'], unknown_np.tolist(), state['max_query_samples'])
        _append_limited(state['max_known_class_probability'], max_known_np.tolist(), state['max_query_samples'])
        _append_limited(state['query_group'], group_np.tolist(), state['max_query_samples'])
        for key in QUERY_METADATA_KEYS:
            flat_values = metadata_tensors[key].flatten().detach().cpu().numpy().tolist()
            _append_limited(state[key], flat_values, state['max_query_samples'])

    objectness_features = _get_output(outputs, 'decoder_objectness_features', 'proj_obj')
    knownness_features = _get_output(outputs, 'decoder_knownness_features', 'proj_known', 'proj_unk')
    classification_features = _get_output(outputs, 'decoder_classification_features', 'proj_cls')
    if objectness_features is not None and knownness_features is not None and classification_features is not None:
        obj_feat = objectness_features.detach().flatten(0, 1).cpu().numpy()
        known_feat = knownness_features.detach().flatten(0, 1).cpu().numpy()
        cls_feat = classification_features.detach().flatten(0, 1).cpu().numpy()
        feature_groups = group_np

        feature_metadata = {
            f'feature_{key}': metadata_tensors[key].flatten().detach().cpu().numpy().tolist()
            for key in QUERY_METADATA_KEYS
        }

        remaining = state['max_feature_samples'] - len(state['objectness_features'])
        if remaining > 0:
            if obj_feat.shape[0] > remaining:
                obj_feat = obj_feat[:remaining]
                known_feat = known_feat[:remaining]
                cls_feat = cls_feat[:remaining]
                feature_groups = feature_groups[:remaining]
                for key in FEATURE_METADATA_KEYS:
                    feature_metadata[key] = feature_metadata[key][:remaining]
            state['objectness_features'].extend(list(obj_feat))
            state['knownness_features'].extend(list(known_feat))
            state['classification_features'].extend(list(cls_feat))
            state['feature_groups'].extend(feature_groups.tolist())
            for key in FEATURE_METADATA_KEYS:
                state[key].extend(feature_metadata[key])

    vis_debug = outputs.get('vis_debug', None)
    if vis_debug is not None:
        layer_objectness_probability = vis_debug.get('layer_objectness_probability', vis_debug.get('layer_obj_prob', None))
        layer_knownness_probability = vis_debug.get('layer_knownness_probability', vis_debug.get('layer_knownness_prob', None))
        layer_unknown_probability = vis_debug.get('layer_unknown_probability', vis_debug.get('layer_unknown_prob', None))
        layer_max_known_class_probability = vis_debug.get('layer_max_known_class_probability', vis_debug.get('layer_cls_max', None))
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
