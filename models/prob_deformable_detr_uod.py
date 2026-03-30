# ------------------------------------------------------------------------
# UOD: Explicit Unknownness Modeling + Decoupled Optimization on PROB.
# Chapter 3: explicit unknownness + sparse pseudo supervision + batch dynamic allocation
# Chapter 4: add triplet decoupled optimization
# ------------------------------------------------------------------------
import copy
import math
import logging

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       inverse_sigmoid, is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)

from models.ops.modules import MSDeformAttn
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .prob_deformable_detr import ProbObjectnessHead, sigmoid_focal_loss
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss
from .segmentation import sigmoid_focal_loss as seg_sigmoid_focal_loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _inverse_sigmoid_clamped(x, eps=1e-4):
    x = float(max(eps, min(1 - eps, x)))
    return math.log(x / (1 - x))


def _compute_uod_fused_probabilities(pred_logits, pred_obj, pred_unk, invalid_cls_logits, temperature,
                                     known_unk_suppress_coeff, unknown_known_suppress_coeff):
    logits = pred_logits.clone()
    logits[:, :, invalid_cls_logits] = -10e10
    obj_prob = torch.exp(-temperature * pred_obj)
    known_prob = logits.sigmoid().clone()
    if len(invalid_cls_logits) > 0:
        known_prob[:, :, invalid_cls_logits] = 0.0
    if known_prob.shape[-1] > 0:
        known_prob[:, :, -1] = 0.0
    unk_prob = torch.sigmoid(pred_unk) if pred_unk is not None else torch.zeros_like(obj_prob)
    known_suppress = torch.clamp(1.0 - known_unk_suppress_coeff * unk_prob.unsqueeze(-1), min=0.0, max=1.0)
    known_scores = obj_prob.unsqueeze(-1) * known_prob * known_suppress
    if known_prob.shape[-1] > 1:
        max_known_prob = known_prob[:, :, :-1].max(dim=-1).values
    else:
        max_known_prob = torch.zeros_like(obj_prob)
    unknown_suppress = torch.clamp(1.0 - unknown_known_suppress_coeff * max_known_prob, min=0.0, max=1.0)
    unknown_score = obj_prob * unk_prob * unknown_suppress
    fused = known_scores.clone() 
    if fused.shape[-1] > 0:
        fused[:, :, -1] = unknown_score
    return {
        'obj_prob': obj_prob,
        'known_prob': known_prob,
        'unk_prob': unk_prob,
        'max_known_prob': max_known_prob,
        'known_scores': known_scores,
        'unknown_score': unknown_score,
        'fused_prob': fused,
    }


class DeformableDETRUOD(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.enable_odqe = bool(getattr(args, 'uod_enable_odqe', False))

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.unk_embed = nn.Linear(hidden_dim, 1)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)

        init_known_unk = _inverse_sigmoid_clamped(getattr(args, 'uod_known_unk_suppress_init', 0.5))
        init_unknown_known = _inverse_sigmoid_clamped(getattr(args, 'uod_unknown_known_suppress_init', 0.5))
        self.known_unk_suppress_logit = nn.Parameter(torch.tensor(init_known_unk, dtype=torch.float32))
        self.unknown_known_suppress_logit = nn.Parameter(torch.tensor(init_unknown_known, dtype=torch.float32))

        if self.enable_odqe:
            self.context_attn = MSDeformAttn(hidden_dim, num_feature_levels, n_heads=8, n_points=4)
            self.gate_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2)
            self.ffn_obj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.ffn_unk = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.ffn_cls = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        else:
            self.obj_proj = nn.Linear(hidden_dim, hidden_dim)
            self.unk_proj = nn.Linear(hidden_dim, hidden_dim)
            self.cls_proj = nn.Linear(hidden_dim, hidden_dim)

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
        self.backbone = backbone

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.unk_embed.bias.data.fill_(-2.0)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.unk_embed = _get_clones(self.unk_embed, num_pred)
            self.prob_obj_head = _get_clones(self.prob_obj_head, num_pred)
            if self.enable_odqe:
                self.context_attn = _get_clones(self.context_attn, num_pred)
                self.gate_mlp = _get_clones(self.gate_mlp, num_pred)
                self.ffn_obj = _get_clones(self.ffn_obj, num_pred)
                self.ffn_unk = _get_clones(self.ffn_unk, num_pred)
                self.ffn_cls = _get_clones(self.ffn_cls, num_pred)
            else:
                self.obj_proj = _get_clones(self.obj_proj, num_pred)
                self.unk_proj = _get_clones(self.unk_proj, num_pred)
                self.cls_proj = _get_clones(self.cls_proj, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.unk_embed = nn.ModuleList([self.unk_embed for _ in range(num_pred)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_pred)])
            if self.enable_odqe:
                self.context_attn = nn.ModuleList([self.context_attn for _ in range(num_pred)])
                self.gate_mlp = nn.ModuleList([self.gate_mlp for _ in range(num_pred)])
                self.ffn_obj = nn.ModuleList([self.ffn_obj for _ in range(num_pred)])
                self.ffn_unk = nn.ModuleList([self.ffn_unk for _ in range(num_pred)])
                self.ffn_cls = nn.ModuleList([self.ffn_cls for _ in range(num_pred)])
            else:
                self.obj_proj = nn.ModuleList([self.obj_proj for _ in range(num_pred)])
                self.unk_proj = nn.ModuleList([self.unk_proj for _ in range(num_pred)])
                self.cls_proj = nn.ModuleList([self.cls_proj for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if self.enable_odqe:
            decay_min = float(getattr(args, 'uod_odqe_decay_min', 0.1))
            decay_power = float(getattr(args, 'uod_odqe_decay_power', 1.0))
            if num_pred == 1:
                decay = torch.ones(1)
            else:
                positions = torch.linspace(0.0, 1.0, steps=num_pred)
                decay = 1.0 - (1.0 - decay_min) * positions.pow(decay_power)
            self.register_buffer('odqe_layer_decay', decay)
        else:
            self.register_buffer('odqe_layer_decay', torch.ones(num_pred))

    def _context_reference_input(self, reference, valid_ratios):
        ref_sig = reference.sigmoid()
        if ref_sig.shape[-1] == 4:
            return ref_sig[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        return ref_sig[:, :, None] * valid_ratios[:, None]

    def _calibration_coeffs(self):
        return torch.sigmoid(self.known_unk_suppress_logit), torch.sigmoid(self.unknown_known_suppress_logit)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight if not self.two_stage else None
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, enc_info = self.transformer(srcs, masks, pos, query_embeds)
        memory = enc_info['memory']
        spatial_shapes = enc_info['spatial_shapes']
        level_start_index = enc_info['level_start_index']
        valid_ratios = enc_info['valid_ratios']
        padding_mask = enc_info['padding_mask']

        outputs_classes, outputs_coords, outputs_objectness, outputs_unknownness = [], [], [], []
        outputs_obj_feats, outputs_unk_feats, outputs_cls_feats = [], [], []
        gate_means = []

        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            q = hs[lvl]
            if self.enable_odqe:
                ref_input = self._context_reference_input(reference, valid_ratios)
                c_q = self.context_attn[lvl](q, ref_input, memory, spatial_shapes, level_start_index, padding_mask)
                g_q = torch.sigmoid(self.gate_mlp[lvl](torch.cat([q, c_q], dim=-1)))
                layer_decay = self.odqe_layer_decay[min(lvl, len(self.odqe_layer_decay) - 1)].to(q.dtype)
                q_tilde = q + layer_decay * g_q * c_q
                gate_means.append((layer_decay * g_q).mean())
                obj_feat = self.ffn_obj[lvl](q_tilde)
                unk_feat = self.ffn_unk[lvl](q_tilde)
                cls_feat = self.ffn_cls[lvl](q_tilde)
                hs_for_bbox = q_tilde
            else:
                obj_feat = self.obj_proj[lvl](q)
                unk_feat = self.unk_proj[lvl](q)
                cls_feat = self.cls_proj[lvl](q)
                hs_for_bbox = q

            outputs_class = self.class_embed[lvl](cls_feat)
            outputs_objectness_lvl = self.prob_obj_head[lvl](obj_feat)
            outputs_unknownness_lvl = self.unk_embed[lvl](unk_feat).squeeze(-1)
            tmp = self.bbox_embed[lvl](hs_for_bbox)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objectness.append(outputs_objectness_lvl)
            outputs_unknownness.append(outputs_unknownness_lvl)
            outputs_obj_feats.append(obj_feat)
            outputs_unk_feats.append(unk_feat)
            outputs_cls_feats.append(cls_feat)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_objectness = torch.stack(outputs_objectness)
        outputs_unknownness = torch.stack(outputs_unknownness)
        known_coeff, unknown_coeff = self._calibration_coeffs()
        gate_mean_per_layer = torch.stack(gate_means) if gate_means else None

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_obj': outputs_objectness[-1],
            'pred_unk': outputs_unknownness[-1],
            'proj_obj': outputs_obj_feats[-1],
            'proj_unk': outputs_unk_feats[-1],
            'proj_cls': outputs_cls_feats[-1],
            'known_unk_suppress_coeff': known_coeff,
            'unknown_known_suppress_coeff': unknown_coeff,
        }
        if gate_mean_per_layer is not None:
            out['gate_mean_per_layer'] = gate_mean_per_layer
            out['gate_mean'] = gate_mean_per_layer.mean()
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_objectness, outputs_unknownness,
                outputs_obj_feats, outputs_unk_feats, outputs_cls_feats,
                known_coeff, unknown_coeff,
            )
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness, unknownness,
                      obj_feats, unk_feats, cls_feats, known_coeff, unknown_coeff):
        aux = []
        for a, b, c, d, po, pu, pc in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1], unknownness[:-1], obj_feats[:-1], unk_feats[:-1], cls_feats[:-1]):
            aux.append({
                'pred_logits': a,
                'pred_obj': b,
                'pred_unk': d,
                'pred_boxes': c,
                'proj_obj': po,
                'proj_unk': pu,
                'proj_cls': pc,
                'known_unk_suppress_coeff': known_coeff,
                'unknown_known_suppress_coeff': unknown_coeff,
            })
        return aux


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim,
                 focal_alpha=0.25, empty_weight=0.1, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.empty_weight = empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.hidden_dim = hidden_dim
        self.min_obj = -hidden_dim * math.log(0.9)
        self.args = args

        self.enable_unknown = bool(getattr(args, 'uod_enable_unknown', False))
        self.enable_pseudo = bool(getattr(args, 'uod_enable_pseudo', False))
        self.enable_batch_dynamic = bool(getattr(args, 'uod_enable_batch_dynamic', False))
        self.enable_decorr = bool(getattr(args, 'uod_enable_decorr', False))
        self.obj_temperature = float(getattr(args, 'obj_temp', 1.0)) / float(hidden_dim)
        self.num_aux_layers = max(int(getattr(args, 'dec_layers', 6)) - 1, 0)

        self.uod_start_epoch = int(getattr(args, 'uod_start_epoch', 8))
        self.uod_neg_warmup_epochs = int(getattr(args, 'uod_neg_warmup_epochs', 3))
        self.uod_pos_quantile = float(getattr(args, 'uod_pos_quantile', 0.25))
        self.uod_pos_scale = float(getattr(args, 'uod_pos_scale', 1.2))
        self.uod_min_pos_thresh = float(getattr(args, 'uod_min_pos_thresh', 0.08))
        self.uod_known_reject_thresh = float(getattr(args, 'uod_known_reject_thresh', 0.15))
        self.uod_neg_margin = float(getattr(args, 'uod_neg_margin', 0.8))
        self.uod_pos_per_img_cap = int(getattr(args, 'uod_pos_per_img_cap', 1))
        self.uod_neg_per_img = int(getattr(args, 'uod_neg_per_img', 1))
        self.uod_batch_topk_max = int(getattr(args, 'uod_batch_topk_max', 8))
        self.uod_batch_topk_ratio = float(getattr(args, 'uod_batch_topk_ratio', 0.25))
        self.uod_max_iou = float(getattr(args, 'uod_max_iou', 0.2))
        self.uod_max_iof = float(getattr(args, 'uod_max_iof', 0.4))
        self.uod_min_area = float(getattr(args, 'uod_min_area', 0.002))
        self.uod_min_side = float(getattr(args, 'uod_min_side', 0.05))
        self.uod_max_aspect_ratio = float(getattr(args, 'uod_max_aspect_ratio', 4.0))
        self.uod_candidate_nms_iou = float(getattr(args, 'uod_candidate_nms_iou', 0.6))
        self.uod_pos_unk_min = float(getattr(args, 'uod_pos_unk_min', 0.05))
        self.enable_cls_soft_attn = bool(getattr(args, 'uod_enable_cls_soft_attn', False))
        self.uod_cls_soft_attn_alpha = float(getattr(args, 'uod_cls_soft_attn_alpha', 0.5))
        self.uod_cls_soft_attn_min = float(getattr(args, 'uod_cls_soft_attn_min', 0.25))
        self.uod_neg_max_pseudo_iou =  float(getattr(args, 'uod_neg_max_pseudo_iou', 0.3))
        self.uod_neg_known_max = float(getattr(args, 'uod_neg_known_max', 0.7))
        self.uod_neg_unk_max = float(getattr(args, 'uod_neg_unk_max', 0.5))

    def _get_calibration_coeffs(self, outputs):
        known_coeff = outputs.get('known_unk_suppress_coeff', None)
        unknown_coeff = outputs.get('unknown_known_suppress_coeff', None)
        device = outputs['pred_logits'].device
        dtype = outputs['pred_logits'].dtype
        if known_coeff is None:
            known_coeff = torch.tensor(0.5, device=device, dtype=dtype)
        if unknown_coeff is None:
            unknown_coeff = torch.tensor(0.5, device=device, dtype=dtype)
        return known_coeff.to(device=device, dtype=dtype), unknown_coeff.to(device=device, dtype=dtype)

    def _compute_fused_probabilities(self, outputs):
        pred_unk = outputs.get('pred_unk', None)
        if pred_unk is None:
            zero = torch.zeros_like(outputs['pred_obj'])
            pred_unk = zero
        known_coeff, unknown_coeff = self._get_calibration_coeffs(outputs)
        return _compute_uod_fused_probabilities(
            outputs['pred_logits'],
            outputs['pred_obj'],
            pred_unk,
            self.invalid_cls_logits,
            self.obj_temperature,
            known_coeff,
            unknown_coeff,
        )

    def _aux_stage(self, layer_idx):
        if self.num_aux_layers <= 0:
            return 'high'
        low_end = max(1, self.num_aux_layers // 3)
        mid_end = max(low_end + 1, (2 * self.num_aux_layers + 2) // 3)
        if layer_idx < low_end:
            return 'low'
        if layer_idx < mid_end:
            return 'mid'
        return 'high'

    def _aux_losses_for_layer(self, layer_idx):
        stage = self._aux_stage(layer_idx)
        if stage == 'low':
            return ['labels', 'boxes', 'cardinality', 'obj_likelihood', 'obj_pseudo']
        if stage == 'mid':
            return ['labels', 'boxes', 'cardinality', 'obj_likelihood', 'obj_pseudo', 'unk_known', 'unk_pseudo']
        return ['labels', 'boxes', 'cardinality', 'obj_likelihood', 'obj_pseudo', 'unk_known', 'unk_pseudo', 'decorr']

    def _sigmoid_focal_loss_query_weight(self, inputs, targets, num_boxes, query_weights=None, alpha: float = 0.25, gamma: float = 2.0):
        prob = inputs.sigmoid()
        class_weight = torch.ones(inputs.shape[-1], dtype=prob.dtype, layout=prob.layout, device=prob.device)
        class_weight[-1] = self.empty_weight
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=class_weight)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        if query_weights is not None:
            loss = loss * query_weights.unsqueeze(-1)
        return loss.mean(1).sum() / num_boxes


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True,
                    dummy_pos_indices=None, dummy_pos_weights=None, **kwargs):
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:, :, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

        # target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1, dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o
        # target_classes_onehot = torch.zeros(src_logits.shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        C = src_logits.shape[-1]  # 当前 logits 维度，通常等于 self.num_classes

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,   # 额外背景索引
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], C + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # 去掉额外背景维度，unmatched 变成全 0
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        query_weights = None
        if self.enable_cls_soft_attn and dummy_pos_indices is not None:
            query_weights = torch.ones(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
            for b_idx, q_list in enumerate(dummy_pos_indices):
                if len(q_list) == 0:
                    continue
                q = torch.as_tensor(q_list, dtype=torch.long, device=src_logits.device)
                if dummy_pos_weights is not None and b_idx < len(dummy_pos_weights) and len(dummy_pos_weights[b_idx]) == len(q_list):
                    conf = torch.as_tensor(dummy_pos_weights[b_idx], dtype=src_logits.dtype, device=src_logits.device)
                else:
                    conf = torch.ones(len(q_list), dtype=src_logits.dtype, device=src_logits.device)
                
                attn = 1.0 - self.uod_cls_soft_attn_alpha * conf
                attn = torch.clamp(attn, min=self.uod_cls_soft_attn_min, max=1.0)
                # 两段式, 如果置信度很高, 直接硬豁免.
                hard_mask = conf >= 0.8
                attn[hard_mask] = 0.0
                
                query_weights[b_idx, q] = attn

        if query_weights is None:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                         num_classes=self.num_classes, empty_weight=self.empty_weight) * src_logits.shape[1]
        else:
            loss_ce = self._sigmoid_focal_loss_query_weight(
                src_logits, target_classes_onehot, num_boxes, query_weights=query_weights, alpha=self.focal_alpha
            ) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        if log and len(target_classes_o) > 0:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        elif log:
            losses['class_error'] = src_logits.sum() * 0.0
        return losses

    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        fused = self._compute_fused_probabilities(outputs)
        known_max = fused['known_scores'][:, :, :-1].max(dim=-1).values if fused['known_scores'].shape[-1] > 1 else fused['known_scores'].squeeze(-1)
        unk_score = fused['unknown_score']
        card_pred = ((known_max > 0.05) | (unk_score > 0.05)).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs['pred_masks']
        target_masks, valid = nested_tensor_from_tensor_list([t['masks'] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)
        src_masks = src_masks[src_idx]
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks[tgt_idx].flatten(1)

        return {
            'loss_mask': seg_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            'loss_dice': dice_loss(src_masks, target_masks, num_boxes),
        }

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        assert 'pred_obj' in outputs
        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs['pred_obj'][idx]
        if pred_obj.numel() == 0:
            return {'loss_obj_ll': outputs['pred_obj'].sum() * 0.0}
        return {'loss_obj_ll': torch.clamp(pred_obj, min=self.min_obj).sum() / num_boxes}

    def loss_unk_known(self, outputs, targets, indices, num_boxes, **kwargs):
        if (not self.enable_unknown) or 'pred_unk' not in outputs:
            zero = outputs['pred_logits'].sum() * 0.0
            return {'loss_unk_known': zero}
        idx = self._get_src_permutation_idx(indices)
        pred_unk = outputs['pred_unk'][idx]
        if pred_unk.numel() == 0:
            return {'loss_unk_known': outputs['pred_unk'].sum() * 0.0}
        raw_loss = F.binary_cross_entropy_with_logits(pred_unk, torch.zeros_like(pred_unk))
        fused = self._compute_fused_probabilities(outputs)
        batch_idx, src_idx = idx
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        fused_unk = fused['unknown_score'][batch_idx, src_idx]
        fused_known = fused['known_scores'][batch_idx, src_idx, target_classes_o]
        fused_unk_loss = F.binary_cross_entropy(fused_unk.clamp(1e-6, 1 - 1e-6), torch.zeros_like(fused_unk))
        fused_known_loss = F.binary_cross_entropy(fused_known.clamp(1e-6, 1 - 1e-6), torch.ones_like(fused_known))
        return {'loss_unk_known': raw_loss + 0.5 * (fused_unk_loss + fused_known_loss)}

    @staticmethod
    def _pairwise_iof(boxes1, boxes2):
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)).clamp(min=1e-6)
        return inter / area1[:, None]

    def _is_valid_geometry(self, box_cxcywh):
        w = box_cxcywh[2].item()
        h = box_cxcywh[3].item()
        area = w * h
        side = min(w, h)
        ar = max(w / max(h, 1e-6), h / max(w, 1e-6))
        return area >= self.uod_min_area and side >= self.uod_min_side and ar <= self.uod_max_aspect_ratio

    def _deduplicate_pos_candidates(self, pred_boxes_img, candidates, iou_thr):
        if len(candidates) <= 1 or iou_thr is None or iou_thr <= 0:
            return candidates
        candidates = sorted(candidates, key=lambda x: (-x[2], x[3], x[4]))
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_img)
        kept = []
        kept_q = []
        for item in candidates:
            q = item[1]
            if len(kept_q) == 0:
                kept.append(item)
                kept_q.append(q)
                continue
            ious = box_ops.box_iou(boxes_xyxy[q].unsqueeze(0), boxes_xyxy[torch.as_tensor(kept_q, dtype=torch.long, device=boxes_xyxy.device)])[0]
            if torch.any(ious >= iou_thr):
                continue
            kept.append(item)
            kept_q.append(q)
        return kept

    def _filter_negatives_near_selected_pos(self, pred_xyxy, selected_q, candidate_qs):
        if len(selected_q) == 0 or len(candidate_qs) == 0 or self.uod_neg_max_pseudo_iou <= 0:
            return candidate_qs
        selected = torch.as_tensor(selected_q, dtype=torch.long, device=pred_xyxy.device)
        kept = []
        for q in candidate_qs:
            q_idx = torch.as_tensor([q], dtype=torch.long, device=pred_xyxy.device)
            ious = box_ops.box_iou(pred_xyxy[q_idx], pred_xyxy[selected])[0]
            if torch.any(ious > self.uod_neg_max_pseudo_iou):
                continue
            kept.append(q)
        return kept

    @torch.no_grad()
    def _mine_uod_pseudo(self, outputs, targets, indices, epoch):
        batch_size = len(targets)
        dummy_pos_indices = [[] for _ in range(batch_size)]
        dummy_neg_indices = [[] for _ in range(batch_size)]
        dummy_pos_weights = [[] for _ in range(batch_size)]
        stats = {
            'num_dummy_pos': 0.0, 'num_dummy_neg': 0.0,
            'num_valid_unmatched': 0.0, 'num_pos_candidates': 0.0, 'num_neg_candidates': 0.0,
            'num_batch_selected_pos': 0.0, 'pos_thresh_sum': 0.0, 'num_thresh': 0.0,
        }

        if (not self.enable_pseudo) or epoch < self.uod_start_epoch:
            return dummy_pos_indices, dummy_neg_indices, dummy_pos_weights, stats

        energy = outputs['pred_obj'].detach() / float(self.hidden_dim)
        pred_boxes = outputs['pred_boxes'].detach()
        fused = self._compute_fused_probabilities(outputs)
        obj_prob = fused['obj_prob'].detach()
        unk_prob = fused['unk_prob'].detach()
        unknown_score = fused['unknown_score'].detach()
        known_max = fused['max_known_prob'].detach()
        num_queries = energy.shape[1]

        all_pos_candidates = []
        per_img_pos_candidates = []
        per_img_cache = []

        for i, (src_idx, _) in enumerate(indices):
            matched = set(src_idx.tolist())
            unmatched = [q for q in range(num_queries) if q not in matched]

            if len(src_idx) > 0:
                matched_scores = energy[i, src_idx]
                mu_obj = matched_scores.mean().item()
                std_obj = matched_scores.std().item() if len(src_idx) > 1 else 0.0
                pos_thresh = max(mu_obj + 3.0 * std_obj, self.uod_min_pos_thresh)
            else:
                pos_thresh = self.uod_min_pos_thresh

            stats['pos_thresh_sum'] += pos_thresh
            stats['num_thresh'] += 1.0

            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[i])
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(targets[i]['boxes'])
            valid = unmatched
            iou_map = {q: 0.0 for q in unmatched}

            if gt_xyxy.numel() > 0 and len(unmatched) > 0:
                cand_boxes = pred_xyxy[unmatched]
                ious = box_ops.box_iou(cand_boxes, gt_xyxy)[0]
                iofs = self._pairwise_iof(cand_boxes, gt_xyxy)
                max_iou = ious.max(dim=1)[0]
                max_iof = iofs.max(dim=1)[0]
                valid = []
                for j, q in enumerate(unmatched):
                    iou_map[q] = max_iou[j].item()
                    if max_iou[j].item() < self.uod_max_iou and max_iof[j].item() < self.uod_max_iof:
                        valid.append(q)

            valid = [q for q in valid if self._is_valid_geometry(pred_boxes[i, q])]
            stats['num_valid_unmatched'] += float(len(valid))

            pos_candidates = []
            for q in valid:
                e = energy[i, q].item()
                k = known_max[i, q].item()
                u = unk_prob[i, q].item()
                us = unknown_score[i, q].item()
                if u < self.uod_pos_unk_min:
                    continue
                if e < pos_thresh and k < self.uod_known_reject_thresh:
                    energy_rel = max(0.0, min(1.0, (pos_thresh - e) / max(pos_thresh, 1e-6)))
                    known_rel = max(0.0, min(1.0, (self.uod_known_reject_thresh - k) / max(self.uod_known_reject_thresh, 1e-6)))
                    iou_rel = 1.0 - max(0.0, min(1.0, iou_map[q] / max(self.uod_max_iou, 1e-6)))
                    unk_rel = max(0.0, min(1.0, u))
                    conf = (energy_rel * known_rel * iou_rel * max(unk_rel, 1e-6)) ** (1.0 / 4.0)
                    pos_candidates.append((i, q, conf, e, k, u, us))

            pos_candidates = self._deduplicate_pos_candidates(pred_boxes[i], pos_candidates, self.uod_candidate_nms_iou)
            all_pos_candidates.extend(pos_candidates)
            stats['num_pos_candidates'] += float(len(pos_candidates))
            per_img_pos_candidates.append(pos_candidates)
            per_img_cache.append({'valid': valid, 'pred_xyxy': pred_xyxy})

        if self.enable_batch_dynamic:
            all_pos_candidates.sort(key=lambda x: (-x[2], -x[6], -x[5], x[3], x[4]))
            topk = min(self.uod_batch_topk_max, max(1, int(math.ceil(self.uod_batch_topk_ratio * max(len(all_pos_candidates), 1)))))
            per_img_count = [0 for _ in range(batch_size)]
            selected = []
            for item in all_pos_candidates:
                b_idx, q, conf, e, k, u, us = item
                if len(selected) >= topk:
                    break
                if per_img_count[b_idx] >= self.uod_pos_per_img_cap:
                    continue
                selected.append(item)
                per_img_count[b_idx] += 1
            for b_idx, q, conf, e, k, u, us in selected:
                dummy_pos_indices[b_idx].append(q)
                dummy_pos_weights[b_idx].append(float(max(0.2, min(1.0, conf))))
            stats['num_batch_selected_pos'] = float(len(selected))
        else:
            for i, pos_candidates in enumerate(per_img_pos_candidates):
                pos_candidates.sort(key=lambda x: (-x[2], -x[6], -x[5], x[3], x[4]))
                pos_candidates = pos_candidates[:self.uod_pos_per_img_cap]
                dummy_pos_indices[i] = [q for _, q, _, _, _, _, _ in pos_candidates]
                dummy_pos_weights[i] = [float(max(0.2, min(1.0, conf))) for _, _, conf, _, _, _, _ in pos_candidates]
            stats['num_batch_selected_pos'] = float(sum(len(v) for v in dummy_pos_indices))

        stats['num_dummy_pos'] = float(sum(len(v) for v in dummy_pos_indices))

        if epoch >= self.uod_start_epoch + self.uod_neg_warmup_epochs:
            for i in range(batch_size):
                valid = per_img_cache[i]['valid']
                pred_xyxy = per_img_cache[i]['pred_xyxy']
                pos_selected = dummy_pos_indices[i]
                pos_selected_set = set(pos_selected)
                remaining = [q for q in valid if q not in pos_selected_set]
                remaining = self._filter_negatives_near_selected_pos(pred_xyxy, pos_selected, remaining)

                neg_candidates = []
                for q in remaining:
                    k = known_max[i, q].item()
                    u = unk_prob[i, q].item()
                    obj = obj_prob[i, q].item()
                    e = energy[i, q].item()
                    if k > self.uod_neg_known_max:
                        continue
                    if u > self.uod_neg_unk_max:
                        continue
                    neg_candidates.append((q, obj, e, k, u))

                stats['num_neg_candidates'] += float(len(neg_candidates))
                neg_candidates.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))
                neg_candidates = neg_candidates[:self.uod_neg_per_img]
                dummy_neg_indices[i] = [q for q, _, _, _, _ in neg_candidates]
                stats['num_dummy_neg'] += float(len(dummy_neg_indices[i]))

        return dummy_pos_indices, dummy_neg_indices, dummy_pos_weights, stats

    
    def loss_obj_pseudo(self, outputs, targets, indices, num_boxes,
                        dummy_pos_indices=None, dummy_pos_weights=None, **kwargs):
        assert 'pred_obj' in outputs
        energy = outputs['pred_obj'] / float(self.hidden_dim)
        device = energy.device
        dtype = energy.dtype
        zero = energy.sum() * 0.0
        if dummy_pos_indices is None:
            return {'loss_obj_pseudo': zero}

        sel_b, sel_q, sel_w = [], [], []
        for b_idx, q_list in enumerate(dummy_pos_indices):
            if len(q_list) == 0:
                continue
            sel_b.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
            sel_q.append(torch.as_tensor(q_list, dtype=torch.long, device=device))
            if dummy_pos_weights is not None and b_idx < len(dummy_pos_weights) and len(dummy_pos_weights[b_idx]) == len(q_list):
                w = torch.as_tensor(dummy_pos_weights[b_idx], dtype=dtype, device=device)
            else:
                w = torch.ones(len(q_list), dtype=dtype, device=device)
            sel_w.append(torch.clamp(w, min=0.2, max=1.0))

        if len(sel_b) == 0:
            return {'loss_obj_pseudo': zero}

        b = torch.cat(sel_b)
        q = torch.cat(sel_q)
        w = torch.cat(sel_w)
        loss = (w * energy[b, q]).sum() / (w.sum() + 1e-6)
        return {'loss_obj_pseudo': loss}

    def loss_obj_neg(self, outputs, targets, indices, num_boxes, dummy_neg_indices=None, **kwargs):
        assert 'pred_obj' in outputs
        energy = outputs['pred_obj'] / float(self.hidden_dim)
        device = energy.device
        zero = energy.sum() * 0.0
        if dummy_neg_indices is None:
            return {'loss_obj_neg': zero}

        sel_b, sel_q = [], []
        for b_idx, q_list in enumerate(dummy_neg_indices):
            if len(q_list) == 0:
                continue
            sel_b.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
            sel_q.append(torch.as_tensor(q_list, dtype=torch.long, device=device))
        if len(sel_b) == 0:
            return {'loss_obj_neg': zero}

        b = torch.cat(sel_b)
        q = torch.cat(sel_q)
        neg_energy = energy[b, q]
        return {'loss_obj_neg': F.relu(self.uod_neg_margin - neg_energy).mean()}

    def loss_unk_pseudo(self, outputs, targets, indices, num_boxes,
                        dummy_pos_indices=None, dummy_pos_weights=None, **kwargs):
        if (not self.enable_unknown) or 'pred_unk' not in outputs:
            zero = outputs['pred_logits'].sum() * 0.0
            return {'loss_unk_pseudo': zero}
        device = outputs['pred_unk'].device
        dtype = outputs['pred_unk'].dtype
        zero = outputs['pred_unk'].sum() * 0.0
        if dummy_pos_indices is None:
            return {'loss_unk_pseudo': zero}

        sel_b, sel_q, sel_w = [], [], []
        for b_idx, q_list in enumerate(dummy_pos_indices):
            if len(q_list) == 0:
                continue
            sel_b.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
            sel_q.append(torch.as_tensor(q_list, dtype=torch.long, device=device))
            if dummy_pos_weights is not None and b_idx < len(dummy_pos_weights) and len(dummy_pos_weights[b_idx]) == len(q_list):
                sel_w.append(torch.as_tensor(dummy_pos_weights[b_idx], dtype=dtype, device=device))
            else:
                sel_w.append(torch.ones(len(q_list), dtype=dtype, device=device))
        if len(sel_b) == 0:
            return {'loss_unk_pseudo': zero}
        b = torch.cat(sel_b)
        q = torch.cat(sel_q)
        w = torch.cat(sel_w)
        logits = outputs['pred_unk'][b, q]
        target = torch.ones_like(logits)
        raw_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        fused = self._compute_fused_probabilities(outputs)
        fused_unk = fused['unknown_score'][b, q].clamp(1e-6, 1 - 1e-6)
        fused_loss = F.binary_cross_entropy(fused_unk, torch.ones_like(fused_unk), reduction='none')
        loss = raw_loss + 0.5 * fused_loss
        return {'loss_unk_pseudo': (loss * w).sum() / (w.sum() + 1e-6)}

    def loss_unk_neg(self, outputs, targets, indices, num_boxes, dummy_neg_indices=None, **kwargs):
        if (not self.enable_unknown) or 'pred_unk' not in outputs:
            zero = outputs['pred_logits'].sum() * 0.0
            return {'loss_unk_neg': zero}
        device = outputs['pred_unk'].device
        zero = outputs['pred_unk'].sum() * 0.0
        if dummy_neg_indices is None:
            return {'loss_unk_neg': zero}
        sel_b, sel_q = [], []
        for b_idx, q_list in enumerate(dummy_neg_indices):
            if len(q_list) == 0:
                continue
            sel_b.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
            sel_q.append(torch.as_tensor(q_list, dtype=torch.long, device=device))
        if len(sel_b) == 0:
            return {'loss_unk_neg': zero}
        b = torch.cat(sel_b)
        q = torch.cat(sel_q)
        logits = outputs['pred_unk'][b, q]
        target = torch.zeros_like(logits)
        return {'loss_unk_neg': F.binary_cross_entropy_with_logits(logits, target)}

    def _feature_orth_loss(self, a, b):
        if a is None or b is None or a.numel() == 0 or b.numel() == 0:
            return a.sum() * 0.0 if a is not None else torch.tensor(0.0)
        a = F.normalize(a.reshape(-1, a.shape[-1]), dim=-1)
        b = F.normalize(b.reshape(-1, b.shape[-1]), dim=-1)
        return torch.abs((a * b).sum(-1)).mean() # 使用L1替换L2正则, 避免L2导致梯度消失

    def loss_orth(self, outputs, targets, indices, num_boxes, **kwargs):
        if (not self.enable_decorr) or ('proj_obj' not in outputs):
            zero = outputs['pred_logits'].sum() * 0.0
            return {'loss_orth': zero}
        z_obj = outputs['proj_obj']
        z_unk = outputs['proj_unk']
        z_cls = outputs['proj_cls']
        # loss = self._feature_orth_loss(z_obj, z_unk) + self._feature_orth_loss(z_obj, z_cls) + self._feature_orth_loss(z_unk, z_cls)
        
        ########## 增加 ############################################
        # 提取 objectness prob 作为前景掩码，阻断背景噪声的疯狂梯度
        obj_temp = float(getattr(self.args, 'obj_temp', 1.0))
        obj_prob = torch.exp(-(obj_temp / float(self.hidden_dim)) * outputs['pred_obj'])
        
        # 建议正交损失的掩码严格一点，保证只在确信的物体上做特征解耦
        fg_mask = obj_prob > 0.1 
        
        if fg_mask.sum() < 2:
            return {'loss_orth': outputs['pred_logits'].sum() * 0.0}

        # 只提取前景特征进行正交计算 [N_fg, C]
        z_obj_fg = z_obj[fg_mask]
        z_unk_fg = z_unk[fg_mask]
        z_cls_fg = z_cls[fg_mask]

        loss = self._feature_orth_loss(z_obj_fg, z_unk_fg) + \
               self._feature_orth_loss(z_obj_fg, z_cls_fg) + \
               self._feature_orth_loss(z_unk_fg, z_cls_fg)
        ########## 增加-结束 ############################################
        
        return {'loss_orth': loss / 3.0}

    def _corr_loss(self, x, y, mask=None):
        """计算皮尔逊相关系数平方，支持掩码过滤纯背景"""
        if x.numel() == 0 or y.numel() == 0:
            return x.sum() * 0.0
            
        if mask is not None:
            if mask.sum() < 2: # 样本太少无法算方差
                return x.sum() * 0.0
            x = x[mask]
            y = y[mask]
        else:
            x = x.reshape(-1)
            y = y.reshape(-1)
            
        x = x - x.mean()
        y = y - y.mean()
        denom = (x.std(unbiased=False) * y.std(unbiased=False) + 1e-6)
        corr = (x * y).mean() / denom
        return corr.pow(2)

    def loss_decorr(self, outputs, targets, indices, num_boxes, **kwargs):
        if (not self.enable_decorr) or ('pred_unk' not in outputs):
            zero = outputs['pred_logits'].sum() * 0.0
            return {'loss_decorr': zero}
        fused = self._compute_fused_probabilities(outputs)
        cls_max = fused['known_scores'][:, :, :-1].max(dim=-1).values if fused['known_scores'].shape[-1] > 1 else fused['known_scores'].squeeze(-1)
        obj_prob = fused['obj_prob']
        unk_prob = fused['unknown_score']
        fg_mask = obj_prob > 0.05
        loss_cls_unk = self._corr_loss(cls_max, unk_prob, mask=fg_mask)
        loss_cls_obj = self._corr_loss(cls_max, obj_prob, mask=fg_mask)
        loss_obj_unk = self._corr_loss(obj_prob, unk_prob, mask=fg_mask)
        loss = loss_cls_unk + loss_cls_obj + loss_obj_unk
        return {'loss_decorr': loss / 3.0}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) if len(indices) > 0 else torch.empty(0, dtype=torch.long)
        src_idx = torch.cat([src for (src, _) in indices]) if len(indices) > 0 else torch.empty(0, dtype=torch.long)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]) if len(indices) > 0 else torch.empty(0, dtype=torch.long)
        tgt_idx = torch.cat([tgt for (_, tgt) in indices]) if len(indices) > 0 else torch.empty(0, dtype=torch.long)
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
            'unk_known': self.loss_unk_known,
            'obj_pseudo': self.loss_obj_pseudo,
            'obj_neg': self.loss_obj_neg,
            'unk_pseudo': self.loss_unk_pseudo,
            'decorr': self.loss_decorr,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0):
        outputs_without_aux = {k: v for k, v in outputs.items() if k in ['pred_logits', 'pred_boxes']}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        dummy_pos_indices, dummy_neg_indices, dummy_pos_weights, mine_stats = self._mine_uod_pseudo(outputs, targets, indices, epoch)
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels' and self.enable_cls_soft_attn:
                kwargs.update({'dummy_pos_indices': dummy_pos_indices, 'dummy_pos_weights': dummy_pos_weights})
            if loss in ['obj_pseudo', 'unk_pseudo']:
                kwargs.update({'dummy_pos_indices': dummy_pos_indices, 'dummy_pos_weights': dummy_pos_weights})
            if loss == 'obj_neg':
                kwargs.update({'dummy_neg_indices': dummy_neg_indices})
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_aux = self.matcher({'pred_logits': aux_outputs['pred_logits'], 'pred_boxes': aux_outputs['pred_boxes']}, targets)
                stage = self._aux_stage(i)
                for loss in self._aux_losses_for_layer(i):
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                        if self.enable_cls_soft_attn and stage != 'low':
                            kwargs.update({'dummy_pos_indices': dummy_pos_indices, 'dummy_pos_weights': dummy_pos_weights})
                    if loss in ['obj_pseudo', 'unk_pseudo']:
                        kwargs.update({'dummy_pos_indices': dummy_pos_indices, 'dummy_pos_weights': dummy_pos_weights})
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices_enc = self.matcher(enc_outputs, bin_targets)
            for loss in ['labels', 'boxes', 'cardinality']:
                kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices_enc, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if self.enable_cls_soft_attn:
            attn_vals = []
            for q_weights in dummy_pos_weights:
                if len(q_weights) == 0:
                    continue
                conf = torch.as_tensor(q_weights, dtype=outputs['pred_logits'].dtype, device=outputs['pred_logits'].device)
                attn = 1.0 - self.uod_cls_soft_attn_alpha * conf
                attn = torch.clamp(attn, min=self.uod_cls_soft_attn_min, max=1.0)
                attn_vals.append(attn)
            if len(attn_vals) > 0:
                attn_vals = torch.cat(attn_vals)
                mine_stats['cls_attn_mean'] = float(attn_vals.mean().item())
                mine_stats['num_cls_soft'] = float(attn_vals.numel())
            else:
                mine_stats['cls_attn_mean'] = 1.0
                mine_stats['num_cls_soft'] = 0.0

        device = outputs['pred_logits'].device
        coeff_known, coeff_unknown = self._get_calibration_coeffs(outputs)
        gate_mean = outputs.get('gate_mean', None)
        losses.update({
            'stat_num_dummy_pos': torch.tensor(float(mine_stats.get('num_dummy_pos', 0.0)), device=device),
            'stat_num_dummy_neg': torch.tensor(float(mine_stats.get('num_dummy_neg', 0.0)), device=device),
            'stat_num_valid_unmatched': torch.tensor(float(mine_stats.get('num_valid_unmatched', 0.0)), device=device),
            'stat_num_pos_candidates': torch.tensor(float(mine_stats.get('num_pos_candidates', 0.0)), device=device),
            'stat_num_neg_candidates': torch.tensor(float(mine_stats.get('num_neg_candidates', 0.0)), device=device),
            'stat_num_batch_selected_pos': torch.tensor(float(mine_stats.get('num_batch_selected_pos', 0.0)), device=device),
            'stat_pos_thresh_mean': torch.tensor(float(mine_stats.get('pos_thresh_sum', 0.0)) / max(float(mine_stats.get('num_thresh', 0.0)), 1.0), device=device),
            'stat_cls_attn_mean': torch.tensor(float(mine_stats.get('cls_attn_mean', 1.0)), device=device),
            'stat_num_cls_soft': torch.tensor(float(mine_stats.get('num_cls_soft', 0.0)), device=device),
            'known_unk_suppress_coeff': coeff_known,
            'unknown_known_suppress_coeff': coeff_unknown,
            'gate_mean': gate_mean if gate_mean is not None else torch.tensor(0.0, device=device),
        })
        return losses


class PostProcess(nn.Module):
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100, unknown_routing_ratio=0.95):
        super().__init__()
        self.temperature = temperature
        self.invalid_cls_logits = invalid_cls_logits
        self.pred_per_im = pred_per_im
        self.unknown_routing_ratio = float(unknown_routing_ratio)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, pred_obj, out_bbox = outputs['pred_logits'], outputs['pred_obj'], outputs['pred_boxes']
        pred_unk = outputs.get('pred_unk', None)
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        # known_coeff = outputs.get('known_unk_suppress_coeff', torch.tensor(0.5, device=out_logits.device, dtype=out_logits.dtype))
        # unknown_coeff = outputs.get('unknown_known_suppress_coeff', torch.tensor(0.5, device=out_logits.device, dtype=out_logits.dtype))
        # ===== manual calibration for eval-only ablation =====
        force_known_unk = 1      # a
        force_unknown_known = 0  # b

        known_coeff = torch.tensor(force_known_unk, device=out_logits.device, dtype=out_logits.dtype)
        unknown_coeff = torch.tensor(force_unknown_known, device=out_logits.device, dtype=out_logits.dtype)
        fused = _compute_uod_fused_probabilities(
            out_logits,
            pred_obj,
            pred_unk if pred_unk is not None else torch.zeros_like(pred_obj),
            self.invalid_cls_logits,
            self.temperature,
            known_coeff.to(device=out_logits.device, dtype=out_logits.dtype),
            unknown_coeff.to(device=out_logits.device, dtype=out_logits.dtype),
        )

        known_scores = fused['known_scores'][:, :, :-1].clone() if fused['known_scores'].shape[-1] > 1 else fused['known_scores'].clone()
        if len(self.invalid_cls_logits) > 0 and known_scores.shape[-1] > 0:
            valid_invalid = [idx for idx in self.invalid_cls_logits if idx < known_scores.shape[-1]]
            if len(valid_invalid) > 0:
                known_scores[:, :, valid_invalid] = -1.0
        best_known_scores, best_known_labels = known_scores.max(dim=-1)
        best_known_scores = best_known_scores.clamp(min=0.0)
        unknown_scores = fused['unknown_score']

        choose_unknown = unknown_scores >= (self.unknown_routing_ratio * best_known_scores)
        query_scores = torch.where(choose_unknown, unknown_scores, best_known_scores)
        query_labels = best_known_labels.clone()
        query_labels[choose_unknown] = out_logits.shape[-1] - 1

        k = min(self.pred_per_im, query_scores.shape[1])
        scores, topk_queries = torch.topk(query_scores, k, dim=1)
        labels = torch.gather(query_labels, 1, topk_queries)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_queries.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


class ExemplarSelection(nn.Module):
    def __init__(self, args, num_classes, matcher, invalid_cls_logits, temperature=1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.invalid_cls_logits = invalid_cls_logits
        self.temperature = temperature
        print('running with exemplar_replay_selection')

    def calc_energy_per_image(self, outputs, targets, indices):
        known_coeff = outputs.get('known_unk_suppress_coeff', torch.tensor(0.5, device=outputs['pred_logits'].device, dtype=outputs['pred_logits'].dtype))
        unknown_coeff = outputs.get('unknown_known_suppress_coeff', torch.tensor(0.5, device=outputs['pred_logits'].device, dtype=outputs['pred_logits'].dtype))
        fused = _compute_uod_fused_probabilities(
            outputs['pred_logits'],
            outputs['pred_obj'],
            outputs.get('pred_unk', torch.zeros_like(outputs['pred_obj'])),
            self.invalid_cls_logits,
            self.temperature,
            known_coeff.to(device=outputs['pred_logits'].device, dtype=outputs['pred_logits'].dtype),
            unknown_coeff.to(device=outputs['pred_logits'].device, dtype=outputs['pred_logits'].dtype),
        )
        image_sorted_scores = {}
        for i in range(len(targets)):
            image_sorted_scores[''.join([chr(int(c)) for c in targets[i]['org_image_id']])] = {
                'labels': targets[i]['labels'].cpu().numpy(),
                'scores': fused['known_scores'][i, indices[i][0], targets[i]['labels']].detach().cpu().numpy(),
            }
        return [image_sorted_scores]

    def forward(self, samples, outputs, targets):
        outputs_without_aux = {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
        indices = self.matcher(outputs_without_aux, targets)
        return self.calc_energy_per_image(outputs, targets, indices)


def build(args):
    num_classes = args.num_classes
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS, num_classes - 1))
    print('Invalid class range: ' + str(invalid_cls_logits))
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = DeformableDETRUOD(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        args=args,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_obj_ll': args.obj_loss_coef,
    }
    losses = ['labels', 'boxes', 'cardinality', 'obj_likelihood']

    if getattr(args, 'uod_enable_unknown', False):
        weight_dict['loss_unk_known'] = getattr(args, 'unk_loss_coef', 0.3)
        losses.append('unk_known')
    if getattr(args, 'uod_enable_pseudo', False):
        weight_dict['loss_obj_pseudo'] = getattr(args, 'uod_pseudo_obj_loss_coef', 0.3)
        weight_dict['loss_obj_neg'] = getattr(args, 'uod_obj_neg_loss_coef', 1.0)
        weight_dict['loss_unk_pseudo'] = getattr(args, 'uod_pseudo_unk_loss_coef', 0.4)
        losses.extend(['obj_pseudo', 'obj_neg', 'unk_pseudo'])
    if getattr(args, 'uod_enable_decorr', False):
        weight_dict['loss_decorr'] = getattr(args, 'uod_decorr_loss_coef', 0.05)
        losses.append('decorr')

    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
        losses += ['masks']

    if args.aux_loss:
        aux_weight_dict = {}
        num_aux_layers = max(args.dec_layers - 1, 0)
        low_end = max(1, num_aux_layers // 3) if num_aux_layers > 0 else 0
        mid_end = max(low_end + 1, (2 * num_aux_layers + 2) // 3) if num_aux_layers > 0 else 0
        for i in range(num_aux_layers):
            stage = 'low' if i < low_end else ('mid' if i < mid_end else 'high')
            aux_weight_dict[f'loss_ce_{i}'] = weight_dict['loss_ce']
            aux_weight_dict[f'loss_bbox_{i}'] = weight_dict['loss_bbox']
            aux_weight_dict[f'loss_giou_{i}'] = weight_dict['loss_giou']
            aux_weight_dict[f'loss_obj_ll_{i}'] = weight_dict['loss_obj_ll']
            if 'loss_obj_pseudo' in weight_dict:
                if stage == 'low':
                    aux_weight_dict[f'loss_obj_pseudo_{i}'] = weight_dict['loss_obj_pseudo'] * float(getattr(args, 'uod_haux_low_obj_coef', 0.35))
                elif stage == 'mid':
                    aux_weight_dict[f'loss_obj_pseudo_{i}'] = weight_dict['loss_obj_pseudo'] * float(getattr(args, 'uod_haux_mid_unknown_coef', 0.45))
                else:
                    aux_weight_dict[f'loss_obj_pseudo_{i}'] = weight_dict['loss_obj_pseudo'] * float(getattr(args, 'uod_haux_high_unknown_coef', 0.7))
            if stage in ['mid', 'high'] and 'loss_unk_known' in weight_dict:
                coef = float(getattr(args, 'uod_haux_mid_unknown_coef', 0.45)) if stage == 'mid' else float(getattr(args, 'uod_haux_high_unknown_coef', 0.7))
                aux_weight_dict[f'loss_unk_known_{i}'] = weight_dict['loss_unk_known'] * coef
            if stage in ['mid', 'high'] and 'loss_unk_pseudo' in weight_dict:
                coef = float(getattr(args, 'uod_haux_mid_unknown_coef', 0.45)) if stage == 'mid' else float(getattr(args, 'uod_haux_high_unknown_coef', 0.7))
                aux_weight_dict[f'loss_unk_pseudo_{i}'] = weight_dict['loss_unk_pseudo'] * coef
            if stage == 'high' and 'loss_decorr' in weight_dict:
                aux_weight_dict[f'loss_decorr_{i}'] = weight_dict['loss_decorr'] * float(getattr(args, 'uod_haux_high_decorr_coef', 0.5))
        aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items() if k in ['loss_ce', 'loss_bbox', 'loss_giou']})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits,
                             args.hidden_dim, focal_alpha=args.focal_alpha, empty_weight=1, args=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(
        invalid_cls_logits,
        temperature=args.obj_temp / args.hidden_dim,
        pred_per_im=args.num_queries,
        unknown_routing_ratio=getattr(args, 'uod_postprocess_unknown_ratio', 0.95),
    )}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits,
                                           temperature=args.obj_temp / args.hidden_dim)
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == 'coco_panoptic':
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors['panoptic'] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    return model, criterion, postprocessors, exemplar_selection
