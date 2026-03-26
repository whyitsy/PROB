# ------------------------------------------------------------------------
# UOD: Explicit Unknownness Modeling + Decoupled Optimization on PROB.
# Chapter 3: explicit unknownness + sparse pseudo supervision + batch dynamic allocation
# Chapter 4: add triplet decoupled optimization
# ------------------------------------------------------------------------
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       inverse_sigmoid, is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)

from models.ops.modules import MSDeformAttn
from .backbone import build_backbone
from .deformable_transformer_CH4 import build_deforamble_transformer
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
        self.use_decorr = bool(getattr(args, 'uod_enable_decorr', False))
        self.enable_oadf = bool(getattr(args, 'uod_enable_oadf', False))

        # Shared heads for detection outputs.
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.unk_embed = nn.Linear(hidden_dim, 1)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)

        # ==========================================================
        # 【修改点 2】：CH4 OADF - 对象感知解耦框架模块定义
        # ==========================================================
        if self.enable_oadf:
            # 1. 上下文增强 (Cross Attention with Encoder Memory)
            self.context_attn = MSDeformAttn(hidden_dim, num_feature_levels, nheads=8, n_points=4)
            # 2. 门控融合 (Gated Fusion)
            self.gate_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2)
            # 3. 三分支任务细化 (Task-specific FFN Refinement)
            self.ffn_obj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.ffn_unk = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.ffn_cls = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        else:
            # 退回 CH3 的单层线性投影
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
            # 【修改点 3】：支持迭代框优化的模块克隆
            if self.enable_oadf:
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
            if self.enable_oadf:
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

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
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

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        # ==========================================================
        # 【修改点 4】：解包 Transformer 传出的 enc_info 并执行 OADF 逻辑
        # ==========================================================
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, enc_info = self.transformer(srcs, masks, pos, query_embeds)

        # 提取 OADF 所需的全图上下文结构变量
        memory = enc_info['memory']
        spatial_shapes = enc_info['spatial_shapes']
        level_start_index = enc_info['level_start_index']
        valid_ratios = enc_info['valid_ratios']
        padding_mask = enc_info['padding_mask']
        
        outputs_classes = []
        outputs_coords = []
        outputs_objectness = []
        outputs_unknownness = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            
            # ------ OADF 机制开始 ------
            q = hs[lvl]  # 当前层的 Query
            if self.enable_oadf:
                if lvl < 3: 
                    # 准备参考点：MSDeformAttn 要求输入的 reference_points 必须在 [0, 1] 范围内
                    ref_sig = reference.sigmoid()
                    if ref_sig.shape[-1] == 4:
                        ref_input = ref_sig[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                    else:
                        ref_input = ref_sig[:, :, None] * valid_ratios[:, None]

                    # 步骤一：提取物理上下文 (仅在底层)
                    c_q = self.context_attn[lvl](q, ref_input, memory, spatial_shapes, level_start_index, padding_mask)

                    # 步骤二：门控融合形成增强 Query
                    concat_feat = torch.cat([q, c_q], dim=-1)
                    g_q = torch.sigmoid(self.gate_mlp[lvl](concat_feat))
                    q_tilde = q + g_q * c_q
                else:
                    # 高层直接使用传递上来的特征，拒绝高级语义对轮廓信息的污染
                    q_tilde = q 
                    g_q = None
                
                if lvl == hs.shape[0] - 1 and g_q is not None:
                    gate_mean = g_q.mean()  # 存一下最后一层的门控均值，用于可视化
                elif lvl == hs.shape[0] - 1:
                    gate_mean = None # 或者保持上一次的值

                # 步骤三：三分支轻量细化 (Task-specific Refinement)
                obj_feat = self.ffn_obj[lvl](q_tilde)
                unk_feat = self.ffn_unk[lvl](q_tilde)
                cls_feat = self.ffn_cls[lvl](q_tilde)
                # ------ OADF 机制结束 ------
                # 供 bbox 回归使用
                hs_for_bbox = q_tilde
            else:
                # ------ 基础机制 (CH3) ------
                obj_feat = self.obj_proj[lvl](q)
                unk_feat = self.unk_proj[lvl](q)
                cls_feat = self.cls_proj[lvl](q)
                
                # 供 bbox 回归使用
                hs_for_bbox = q
                
                gate_mean=None

            # 将纯净细化后的特征送入最终预测头
            outputs_class = self.class_embed[lvl](cls_feat)
            outputs_objectness_lvl = self.prob_obj_head[lvl](obj_feat)
            outputs_unknownness_lvl = self.unk_embed[lvl](unk_feat).squeeze(-1)


            tmp = self.bbox_embed[lvl](hs_for_bbox)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objectness.append(outputs_objectness_lvl)
            outputs_unknownness.append(outputs_unknownness_lvl)

            # 保存最后一次循环的特征映射，用于 CH4 的特征正交 Loss
            if lvl == hs.shape[0] - 1:
                final_obj_feat = obj_feat
                final_unk_feat = unk_feat
                final_cls_feat = cls_feat
                
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_objectness = torch.stack(outputs_objectness)
        outputs_unknownness = torch.stack(outputs_unknownness)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_obj': outputs_objectness[-1],
            'pred_unk': outputs_unknownness[-1],
            'proj_obj': final_obj_feat,   
            'proj_unk': final_unk_feat,   
            'proj_cls': final_cls_feat,   
        }
        if gate_mean:
            out['gate_mean'] = gate_mean

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objectness, outputs_unknownness)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness, unknownness):
        return [
            {'pred_logits': a, 'pred_obj': b, 'pred_unk': d, 'pred_boxes': c}
            for a, b, c, d in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1], unknownness[:-1])
        ]


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
        # self.min_obj = -hidden_dim * math.log(0.9)
        self.min_obj = -hidden_dim * math.log(0.999)
        self.args = args

        self.enable_unknown = bool(getattr(args, 'uod_enable_unknown', False))
        self.enable_pseudo = bool(getattr(args, 'uod_enable_pseudo', False))
        self.enable_batch_dynamic = bool(getattr(args, 'uod_enable_batch_dynamic', False))
        self.enable_decorr = bool(getattr(args, 'uod_enable_decorr', False))

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
        self.enable_cls_soft_attn = bool(getattr(args, 'uod_enable_cls_soft_attn', False))
        self.uod_cls_soft_attn_alpha = float(getattr(args, 'uod_cls_soft_attn_alpha', 0.5))
        self.uod_cls_soft_attn_min = float(getattr(args, 'uod_cls_soft_attn_min', 0.25))

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

        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(src_logits.shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

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

    
    # def loss_labels(self, outputs, targets, indices, num_boxes, log=True,
    #                 dummy_pos_indices=None, dummy_pos_weights=None, **kwargs):
    #     assert 'pred_logits' in outputs
    #     temp_src_logits = outputs['pred_logits'].clone()
    #     temp_src_logits[:, :, self.invalid_cls_logits] = -10e10
    #     src_logits = temp_src_logits

    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

    #     # =====================================================================
    #     # 【架构重构：基于 Focal Loss 的纯净标签空间】
    #     # 默认初始化全为 0。在 Focal Loss 的语义下，全 0 就代表“纯背景”。
    #     # 我们不再强制给未匹配的 Query 赋予某个特定的“背景类别”索引。
    #     # =====================================================================
    #     target_classes_onehot = torch.zeros(src_logits.shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

    #     # 1. 已知类硬标签 (Hard Labels for Known Classes): Target = 1.0
    #     target_classes_onehot[idx[0], idx[1], target_classes_o] = 1.0

    #     # =====================================================================
    #     # 【核心创新：不确定性感知软目标 (Uncertainty-Aware Soft Target)】
    #     # 我们征用最后一个维度 (self.num_classes - 1) 作为专属的“未知类”通道。
    #     # 对于挖掘出的伪未知正样本，将几何平均置信度 (Conf) 作为它的正向监督信号！
    #     # =====================================================================
    #     unknown_class_index = self.num_classes - 1
        
    #     if dummy_pos_indices is not None and dummy_pos_weights is not None:
    #         for b_idx, q_list in enumerate(dummy_pos_indices):
    #             if len(q_list) > 0:
    #                 q_tensor = torch.as_tensor(q_list, dtype=torch.long, device=src_logits.device)
    #                 # 这里的 dummy_pos_weights 就是你在挖掘流程中算出的几何平均 Conf
    #                 conf_tensor = torch.as_tensor(dummy_pos_weights[b_idx], dtype=src_logits.dtype, device=src_logits.device)
                    
    #                 # 赋予正向的软标签增强！它不再是被压制的背景了！
    #                 target_classes_onehot[b_idx, q_tensor, unknown_class_index] = conf_tensor

    #     # 3. 直接计算 Focal Loss（不再需要繁琐的 query_weights 软掩码抑制了！）
    #     # 注意：这里的 empty_weight 刚好会作用在最后一个维度上，这非常完美地缓解了
    #     # 纯背景样本过多导致的类别不平衡问题。
    #     loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
    #                                  num_classes=self.num_classes, empty_weight=self.empty_weight) * src_logits.shape[1]

    #     losses = {'loss_ce': loss_ce}
        
    #     if log and len(target_classes_o) > 0:
    #         losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    #     elif log:
    #         losses['class_error'] = src_logits.sum() * 0.0
            
    #     return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
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
        target = torch.zeros_like(pred_unk)
        return {'loss_unk_known': F.binary_cross_entropy_with_logits(pred_unk, target)}

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

    @torch.no_grad()
    def _mine_uod_pseudo(self, outputs, targets, indices, epoch):
        batch_size = len(targets)
        dummy_pos_indices = [[] for _ in range(batch_size)]
        dummy_neg_indices = [[] for _ in range(batch_size)]
        dummy_pos_weights = [[] for _ in range(batch_size)]
        stats = {
            'num_dummy_pos': 0.0, 'num_dummy_neg': 0.0,
            'num_valid_unmatched': 0.0, 'num_pos_candidates': 0.0,
            'num_batch_selected_pos': 0.0, 'pos_thresh_sum': 0.0, 'num_thresh': 0.0,
        }

        if (not self.enable_pseudo) or epoch < self.uod_start_epoch:
            return dummy_pos_indices, dummy_neg_indices, dummy_pos_weights, stats

        # 注意：这里的 energy 实际上是马氏距离的变体，值越小说明越像物体
        energy = outputs['pred_obj'].detach() / float(self.hidden_dim)
        pred_boxes = outputs['pred_boxes'].detach()
        pred_probs = outputs['pred_logits'].detach().sigmoid().clone()
        pred_probs[:, :, self.invalid_cls_logits] = 0.0
        pred_probs[:, :, self.num_classes - 1] = 0.0
        num_queries = energy.shape[1]

        all_pos_candidates = []
        per_img_pos_candidates = []

        for i, (src_idx, _) in enumerate(indices):
            matched = set(src_idx.tolist())
            unmatched = [q for q in range(num_queries) if q not in matched]

            # =====================================================================
            # 创新点 1：基于高斯分布 3-Sigma 法则的数据驱动自适应阈值
            # 彻底摒弃 uod_pos_quantile 和 uod_pos_scale 等人工设定的经验比例
            # =====================================================================
            if len(src_idx) > 0:
                matched_scores = energy[i, src_idx]
                mu_obj = matched_scores.mean().item()
                std_obj = matched_scores.std().item() if len(src_idx) > 1 else 0.0
                
                # 统计学先验：正态分布下，mu + 3*std 覆盖了 99.7% 的有效物体分布
                pos_thresh = mu_obj + 3.0 * std_obj
                # 保底机制防止方差极度坍塌
                pos_thresh = max(pos_thresh, self.uod_min_pos_thresh)
            else:
                pos_thresh = self.uod_min_pos_thresh

            # 自适应负样本（背景）阈值：纯背景的能量应该大于未匹配 Queries 的均值
            if len(unmatched) > 0:
                unmatched_scores = energy[i, unmatched]
                mu_bg = unmatched_scores.mean().item()
                # 负样本的门槛：至少要达到纯背景的平均“距离”，且不能与正样本阈值重叠
                neg_thresh = max(mu_bg, pos_thresh + 0.5)
            else:
                neg_thresh = pos_thresh + 0.5

            stats['pos_thresh_sum'] += pos_thresh
            stats['num_thresh'] += 1.0

            if len(unmatched) == 0:
                per_img_pos_candidates.append([])
                continue

            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[i])
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(targets[i]['boxes'])
            valid = unmatched
            iou_map = {q: 0.0 for q in unmatched}

            if gt_xyxy.numel() > 0:
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
            known_max = pred_probs[i].max(dim=-1)[0]

            # =====================================================================
            # 创新点 2：非线性几何平均 (Geometric Mean) 替代硬编码线性加权
            # 彻底抛弃 0.5, 0.3, 0.2 的人工调参，引入“一票否决”机制
            # =====================================================================
            pos_candidates = []
            for q in valid:
                e = energy[i, q].item()
                k = known_max[q].item()
                if e < pos_thresh and k < self.uod_known_reject_thresh:
                    # 分别计算三个维度的相对置信度 (0 到 1 之间)
                    energy_rel = max(0.0, min(1.0, (pos_thresh - e) / max(pos_thresh, 1e-6)))
                    known_rel = max(0.0, min(1.0, (self.uod_known_reject_thresh - k) / max(self.uod_known_reject_thresh, 1e-6)))
                    iou_rel = 1.0 - max(0.0, min(1.0, iou_map[q] / max(self.uod_max_iou, 1e-6)))
                    
                    # 几何平均数：任何一个维度表现极差（接近0），整体置信度都会崩溃
                    # 这符合“木桶原理”，极其适合高质量伪标签的苛刻筛选
                    conf = (energy_rel * known_rel * iou_rel) ** (1.0 / 3.0)
                    
                    item = (i, q, conf, e, k)
                    pos_candidates.append(item)
                    all_pos_candidates.append(item)
            stats['num_pos_candidates'] += float(len(pos_candidates))
            per_img_pos_candidates.append(pos_candidates)

            if epoch >= self.uod_start_epoch + self.uod_neg_warmup_epochs:
                neg_candidates = []
                for q in valid:
                    e = energy[i, q].item()
                    k = known_max[q].item()
                    # 用自适应的 neg_thresh 替代原来死板的 margin
                    if e > neg_thresh and k < self.uod_known_reject_thresh:
                        neg_candidates.append((q, e, k))
                neg_candidates.sort(key=lambda x: (-x[1], x[2]))
                neg_candidates = neg_candidates[:self.uod_neg_per_img]
                dummy_neg_indices[i] = [q for q, _, _ in neg_candidates]
                stats['num_dummy_neg'] += float(len(dummy_neg_indices[i]))

        # Batch 动态分配逻辑保留，它根据我们算出的几何平均置信度进行排序截断
        if self.enable_batch_dynamic:
            all_pos_candidates.sort(key=lambda x: (-x[2], x[3], x[4]))
            topk = min(self.uod_batch_topk_max, max(1, int(math.ceil(self.uod_batch_topk_ratio * max(len(all_pos_candidates), 1)))))
            per_img_count = [0 for _ in range(batch_size)]
            selected = []
            for item in all_pos_candidates:
                b_idx, q, conf, e, k = item
                if len(selected) >= topk:
                    break
                if per_img_count[b_idx] >= self.uod_pos_per_img_cap:
                    continue
                selected.append(item)
                per_img_count[b_idx] += 1
            for b_idx, q, conf, e, k in selected:
                dummy_pos_indices[b_idx].append(q)
                dummy_pos_weights[b_idx].append(float(max(0.2, min(1.0, conf))))
            stats['num_batch_selected_pos'] = float(len(selected))
        else:
            for i, pos_candidates in enumerate(per_img_pos_candidates):
                pos_candidates.sort(key=lambda x: (-x[2], x[3], x[4]))
                pos_candidates = pos_candidates[:self.uod_pos_per_img_cap]
                dummy_pos_indices[i] = [q for _, q, _, _, _ in pos_candidates]
                dummy_pos_weights[i] = [float(max(0.2, min(1.0, conf))) for _, _, conf, _, _ in pos_candidates]
            stats['num_batch_selected_pos'] = float(sum(len(v) for v in dummy_pos_indices))

        stats['num_dummy_pos'] = float(sum(len(v) for v in dummy_pos_indices))
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
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
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
            
        logits = outputs['pred_logits'].sigmoid().clone()
        logits[:, :, self.invalid_cls_logits] = 0.0
        logits[:, :, self.num_classes - 1] = 0.0
        cls_max = logits.max(dim=-1)[0]
        
        obj_temp = float(getattr(self.args, 'obj_temp', 1.0))
        obj_prob = torch.exp(-(obj_temp / float(self.hidden_dim)) * outputs['pred_obj'])
        unk_prob = torch.sigmoid(outputs['pred_unk'])
        
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
            'unk_known': self.loss_unk_known, # 3, 显式未知性分支的已知类约束
            'obj_pseudo': self.loss_obj_pseudo, # 伪正样本监督, 让这些伪未知候选具备更高的 objectness。
            'obj_neg': self.loss_obj_neg, # 伪负样本监督, 把可靠背景从“像物体”这一侧推开，抑制背景误激活。
            'unk_pseudo': self.loss_unk_pseudo, # 伪正样本上的 unknownness 正监督, 将这些挖掘到的伪未知类作为正向监督信号，提升模型对未知性的识别能力。
            'unk_neg': self.loss_unk_neg, # 背景不是 unknown
            'orth': self.loss_orth, # 4, 表示层特征正交约束, 让proj_obj/proj_unk/proj_cls之间保持一定程度的正交，鼓励模型在不同的特征子空间中编码对象ness、未知性和类别信息，减少它们之间的干扰。
            'decorr': self.loss_decorr, # 预测层去相关损失, 减少obj_prob/unk_prob/cls_max三个决策分数在统计上的耦合。
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
                kwargs.update({
                    'dummy_pos_indices': dummy_pos_indices,
                    'dummy_pos_weights': dummy_pos_weights,
                })
            if loss in ['obj_pseudo', 'obj_neg', 'unk_pseudo', 'unk_neg']:
                kwargs.update({
                    'dummy_pos_indices': dummy_pos_indices,
                    'dummy_neg_indices': dummy_neg_indices,
                    'dummy_pos_weights': dummy_pos_weights,
                })
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_aux = self.matcher({'pred_logits': aux_outputs['pred_logits'], 'pred_boxes': aux_outputs['pred_boxes']}, targets)
                for loss in self.losses:
                    if loss in ['masks', 'obj_pseudo', 'obj_neg', 'unk_known', 'unk_pseudo', 'unk_neg', 'orth', 'decorr']:
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices_enc = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'obj_pseudo', 'obj_neg', 'unk_known', 'unk_pseudo', 'unk_neg', 'obj_likelihood', 'orth', 'decorr']:
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
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
        losses.update({
            'stat_num_dummy_pos': torch.tensor(float(mine_stats.get('num_dummy_pos', 0.0)), device=device),
            'stat_num_dummy_neg': torch.tensor(float(mine_stats.get('num_dummy_neg', 0.0)), device=device),
            'stat_num_valid_unmatched': torch.tensor(float(mine_stats.get('num_valid_unmatched', 0.0)), device=device),
            'stat_num_pos_candidates': torch.tensor(float(mine_stats.get('num_pos_candidates', 0.0)), device=device),
            'stat_num_batch_selected_pos': torch.tensor(float(mine_stats.get('num_batch_selected_pos', 0.0)), device=device),
            'stat_pos_thresh_mean': torch.tensor(float(mine_stats.get('pos_thresh_sum', 0.0)) / max(float(mine_stats.get('num_thresh', 0.0)), 1.0), device=device),
            'stat_cls_attn_mean': torch.tensor(float(mine_stats.get('cls_attn_mean', 1.0)), device=device),
            'stat_num_cls_soft': torch.tensor(float(mine_stats.get('num_cls_soft', 0.0)), device=device),
        })
        return losses


class PostProcess(nn.Module):
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100):
        super().__init__()
        self.temperature = temperature
        self.invalid_cls_logits = invalid_cls_logits
        self.pred_per_im = pred_per_im

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, pred_obj, out_bbox = outputs['pred_logits'], outputs['pred_obj'], outputs['pred_boxes']
        pred_unk = outputs.get('pred_unk', None)

        logits = out_logits.clone()
        logits[:, :, self.invalid_cls_logits] = -10e10
        assert len(logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = torch.exp(-self.temperature * pred_obj).unsqueeze(-1)
        prob = obj_prob * logits.sigmoid()
        prob[:, :, self.invalid_cls_logits] = 0.0

        if pred_unk is not None:
            unk_prob = torch.sigmoid(pred_unk)                    # [B, Q]

            # --- 护城河：绝对置信度拦截 (解决 A-OSE 暴涨的唯一解) ---
            known_prob = logits.sigmoid()
            # 必须大于 0.05（或0.1），否则原始概率太低，直接归零，禁止通过乘法泄漏
            valid_known_mask = known_prob > 0.2
            known_prob = known_prob * valid_known_mask.float()
            
            known_prob[:, :, self.invalid_cls_logits] = 0.0
            known_prob[:, :, -1] = 0.0                           # 最后一维不参与 known max

            max_known_prob = known_prob[:, :, :-1].max(dim=-1)[0]   # [B, Q]

            # 1) known 分数：保留你的 soft suppression
            beta = 1.5
            prob[:, :, :-1] = obj_prob * known_prob[:, :, :-1] * (1.0 - beta * unk_prob.unsqueeze(-1))

            # 2) unknown 分数：保留你的 known-aware 抑制
            prob[:, :, -1] = obj_prob.squeeze(-1) * unk_prob * (1.0 - max_known_prob)
            
            # --- 终极保险：强制排他 (可选，但强烈建议加上) ---
            # 即使经过软抑制，如果 unk 明显大于 known，还是建议把 known 清零
            # 避免评测脚本惩罚双重预测
            is_unknown = unk_prob > max_known_prob
            prob[:, :, :-1] = prob[:, :, :-1].masked_fill(is_unknown.unsqueeze(-1), 0.0)

        topk_values, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // logits.shape[2]
        labels = topk_indexes % logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


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
        out_logits, pred_obj = outputs['pred_logits'], outputs['pred_obj']
        pred_unk = outputs.get('pred_unk', None)
        logits = out_logits.clone()
        logits[:, :, self.invalid_cls_logits] = -10e10
        logits[:, :, self.num_classes - 1] = -10e10

        obj_prob = torch.exp(-self.temperature * pred_obj).unsqueeze(-1)
        prob = obj_prob * logits.sigmoid()
        if pred_unk is not None:
            prob = prob * (1.0 - torch.sigmoid(pred_unk).unsqueeze(-1))

        image_sorted_scores = {}
        for i in range(len(targets)):
            image_sorted_scores[''.join([chr(int(c)) for c in targets[i]['org_image_id']])] = {
                'labels': targets[i]['labels'].cpu().numpy(),
                'scores': prob[i, indices[i][0], targets[i]['labels']].detach().cpu().numpy(),
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
        weight_dict['loss_unk_pseudo'] = getattr(args, 'uod_pseudo_unk_loss_coef', 0.4)
        losses.extend(['obj_pseudo', 'unk_pseudo'])
        weight_dict['loss_obj_neg'] = getattr(args, 'uod_obj_neg_loss_coef', 0.2)
        weight_dict['loss_unk_neg'] = getattr(args, 'uod_bg_unk_loss_coef', 0.2)
        losses.extend(['obj_neg', 'unk_neg'])
    if getattr(args, 'uod_enable_decorr', False):
        weight_dict['loss_orth'] = getattr(args, 'uod_orth_loss_coef', 0.05)
        weight_dict['loss_decorr'] = getattr(args, 'uod_decorr_loss_coef', 0.05)
        losses.extend(['orth', 'decorr'])

    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
        losses += ['masks']

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            for k, v in list(weight_dict.items()):
                if k in ['loss_unk_known', 'loss_obj_pseudo', 'loss_obj_neg', 'loss_unk_pseudo', 'loss_unk_neg', 'loss_orth', 'loss_decorr']:
                    continue
                aux_weight_dict[k + f'_{i}'] = v
        aux_weight_dict.update({k + '_enc': v for k, v in weight_dict.items() if k in ['loss_ce', 'loss_bbox', 'loss_giou']})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits,
                             args.hidden_dim, focal_alpha=args.focal_alpha, args=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp / args.hidden_dim)}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits,
                                           temperature=args.obj_temp / args.hidden_dim)
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == 'coco_panoptic':
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors['panoptic'] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    return model, criterion, postprocessors, exemplar_selection
