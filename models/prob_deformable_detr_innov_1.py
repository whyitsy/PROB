# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
import logging
import copy
import clip
import logging

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss)
from .segmentation import sigmoid_focal_loss as seg_sigmoid_focal_loss
from .deformable_transformer import build_deforamble_transformer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, num_classes: int = 81, empty_weight: float = 0.1, valid_mask=None):
    prob = inputs.sigmoid()
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)
    
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if valid_mask is not None:
        loss = loss * valid_mask.unsqueeze(-1)
        
    return loss.mean(1).sum() / num_boxes


class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0, 1)  
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2 / x.shape[-1]


class DeformableDETR(nn.Module):
    def __init__(self, args, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)
        
        # 跨模态投影头 (由 args 控制是否激活)
        self.use_feature_align = getattr(args, 'use_feature_align', False)
        if self.use_feature_align:
            clip_dim = getattr(args, 'clip_dim', 512)
            self.proj_head = MLP(hidden_dim, hidden_dim, clip_dim, 2) 

        self.num_feature_levels = num_feature_levels

        # 只有在单阶段，或者 two-stage + TDQI 时，learned query 才真正会被用到
        if (not two_stage) or getattr(args, 'tdqi', False):
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        else:
            self.query_embed = None
        
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
                )])
                
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.prob_obj_head = _get_clones(self.prob_obj_head, num_pred)
            if self.use_feature_align:
                self.proj_head = _get_clones(self.proj_head, transformer.decoder.num_layers) 
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_pred)])
            if self.use_feature_align:
                self.proj_head = nn.ModuleList([self.proj_head for _ in range(transformer.decoder.num_layers)]) 
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

        query_embeds = None if self.query_embed is None else self.query_embed.weight
        clip_txt = getattr(self.args, 'clip_text_features', None)
        semantic_mask = getattr(self.args, 'semantic_mask', None)
        
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, query_embeds, tdqi=self.args.tdqi, tdqi_query_num=getattr(self.args, 'tdqi_query_num', 0), 
            clip_text_features=clip_txt, semantic_mask=semantic_mask)

        outputs_classes = []
        outputs_coords = []
        outputs_objectnesses = []
        outputs_projs = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            
            
            if (not self.args.etop) or (lvl <= getattr(self.args, 'etop_layer', 1)):
                outputs_objectness = self.prob_obj_head[lvl](hs[lvl])
            else:
                outputs_objectness = torch.zeros(
                    (hs.shape[1], hs.shape[2]),
                    device=hs.device,
                    dtype=hs.dtype
                )

            if self.use_feature_align:
                outputs_proj = self.proj_head[lvl](hs[lvl])
                outputs_projs.append(outputs_proj)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_objectnesses.append(outputs_objectness)
            
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_objectness = torch.stack(outputs_objectnesses)
        
        obj_layer = getattr(self.args, 'etop_layer', 1) if self.args.etop else -1
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 
               'pred_obj': outputs_objectness[obj_layer], 'samples': samples} 
               
        if self.use_feature_align:
            outputs_proj = torch.stack(outputs_projs)
            out['pred_proj'] = outputs_proj[-1]
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objectness, outputs_projs if self.use_feature_align else None)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness, outputs_proj):
        if outputs_proj is not None:
            return [{'pred_logits': a, 'pred_obj': b, 'pred_boxes': c, 'pred_proj': d}
                    for a, b, c, d in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1], outputs_proj[:-1])]
        else:
            return [{'pred_logits': a, 'pred_obj': b, 'pred_boxes': c}
                    for a, b, c in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1])]
            
class SetCriterion(nn.Module):
    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, empty_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        self.empty_weight = empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.min_obj = -hidden_dim * math.log(0.9)
        
        self.enable_unk_label_obj = getattr(args, 'enable_unk_label_obj', False)
        self.unk_label_obj_score_thresh = getattr(args, 'unk_label_obj_score_thresh', 1.0)
        self.unk_label_start_epoch = getattr(args, 'unk_label_start_epoch', 0)
        self.use_valid_mask = getattr(args, 'use_valid_mask', True) # 创新点1的豁免Mask
        self.use_feature_align = getattr(args, 'use_feature_align', False)
        self.use_vlm_distill = getattr(args, 'use_vlm_distill', False)
        self.args = args
        self.writer = args.writer if hasattr(args, 'writer') else None  # 从 args 获取 writer

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:, :, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        valid_mask = torch.ones([src_logits.shape[0], src_logits.shape[1]], dtype=src_logits.dtype, device=src_logits.device)
        
        if self.use_valid_mask:
            dummy_pos_indices = kwargs.get('dummy_pos_indices', None)
            if dummy_pos_indices is not None:
                for b_idx, q_list in enumerate(dummy_pos_indices):
                    if len(q_list) > 0:
                        valid_mask[b_idx, q_list] = 0.0
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, 
                                     num_classes=self.num_classes, empty_weight=self.empty_weight, valid_mask=valid_mask) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
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

    def loss_align(self, outputs, targets, indices, num_boxes, **kwargs):
        if not self.use_feature_align or 'pred_proj' not in outputs:
            return {}
            
        idx = self._get_src_permutation_idx(indices)
        src_proj = outputs['pred_proj'][idx]

        if 'clip_feat' not in targets[0] or len(src_proj) == 0:
            return {'loss_align': src_proj.sum() * 0.0} 

        target_clip = torch.cat([t['clip_feat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_clip = target_clip.to(src_proj.device).to(src_proj.dtype)

        src_proj_norm = F.normalize(src_proj, p=2, dim=-1)
        loss_align = 1 - (src_proj_norm * target_clip).sum(dim=-1)
        
        return {'loss_align': loss_align.sum() / num_boxes}

    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)
        src_masks = src_masks[src_idx]
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks[tgt_idx].flatten(1)
        return {
            "loss_mask": seg_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
    
    def _get_known_max_scores(self, logits):
        """
        logits: [num_queries, num_classes]
        return: [num_queries]
        """
        probs = logits.sigmoid().detach().clone()

        if hasattr(self, 'invalid_cls_logits') and self.invalid_cls_logits is not None:
            probs[:, self.invalid_cls_logits] = 0

        # 已知类范围：默认排除最后一个 unknown 类
        known_probs = probs[:, :self.num_classes - 1]

        if known_probs.numel() == 0:
            return torch.zeros(probs.shape[0], device=probs.device)

        return known_probs.max(dim=-1)[0]
    
    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes,
                        dummy_pos_indices=None, dummy_neg_indices=None,
                        dummy_vlm_weights=None, epoch=0):

        energy = outputs["pred_obj"]   # 越小越像物体
        device = energy.device
        dtype = energy.dtype

        pos_batch_idx, pos_query_idx, pos_weight_list = [], [], []
        neg_batch_idx, neg_query_idx = [], []

        zero = energy.sum() * 0.0

        # 1) 已知 matched 正样本：始终压低 energy
        if indices is not None:
            src_idx = self._get_src_permutation_idx(indices)
            if len(src_idx[0]) > 0:
                pos_batch_idx.append(src_idx[0].to(device))
                pos_query_idx.append(src_idx[1].to(device))
                pos_weight_list.append(torch.ones_like(src_idx[0], dtype=dtype, device=device))

        # 2) dummy_pos：加入 warmup，避免 unk_label_start_epoch 后立刻自举崩塌
        use_dummy_pos_loss = (
            epoch >= self.unk_label_start_epoch + getattr(self.args, 'unk_label_obj_warmup_epochs', 0)
        )

        if use_dummy_pos_loss and dummy_pos_indices is not None:
            for b_idx, q_list in enumerate(dummy_pos_indices):
                if len(q_list) == 0:
                    continue

                pos_batch_idx.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
                pos_query_idx.append(torch.tensor(q_list, dtype=torch.long, device=device))

                if (
                    self.use_vlm_distill
                    and dummy_vlm_weights is not None
                    and len(dummy_vlm_weights) > b_idx
                    and len(dummy_vlm_weights[b_idx]) == len(q_list)
                ):
                    w = torch.tensor(dummy_vlm_weights[b_idx], dtype=dtype, device=device)
                else:
                    w = torch.ones(len(q_list), dtype=dtype, device=device)

                pos_weight_list.append(w)

        # 3) dummy_neg：始终可用，用 hinge 往高 energy 推
        if dummy_neg_indices is not None:
            for b_idx, q_list in enumerate(dummy_neg_indices):
                if len(q_list) == 0:
                    continue
                neg_batch_idx.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
                neg_query_idx.append(torch.tensor(q_list, dtype=torch.long, device=device))

        loss_pos = zero
        loss_neg = zero

        if len(pos_batch_idx) > 0:
            pos_b = torch.cat(pos_batch_idx)
            pos_q = torch.cat(pos_query_idx)
            pos_w = torch.cat(pos_weight_list)
            pos_energy = energy[pos_b, pos_q]
            loss_pos = (pos_w * pos_energy).sum() / (pos_w.sum() + 1e-6)

        if len(neg_batch_idx) > 0:
            neg_b = torch.cat(neg_batch_idx)
            neg_q = torch.cat(neg_query_idx)
            neg_energy = energy[neg_b, neg_q]
            neg_margin = getattr(self.args, 'obj_neg_margin', 1.0)
            # margin-based learning / hinge loss 思想：负样本 energy 不只是比正样本大，而是要至少大到某个 margin.
            loss_neg = F.relu(neg_margin - neg_energy).mean()

        loss_obj = loss_pos + loss_neg
        return {'loss_obj': loss_obj}    
    
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
            'masks': self.loss_masks,
            'align': self.loss_align 
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _crop_and_encode(self, img_tensor, box_xyxy):
        device = img_tensor.device
        img = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device) + \
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        H, W = img.shape[1], img.shape[2]
        box_abs = box_xyxy * torch.tensor([W, H, W, H]).to(device)
        
        crops = []
        for box in box_abs:
            x1, y1, x2, y2 = box.int()
            x1, y1 = max(0, x1.item()), max(0, y1.item())
            x2, y2 = min(W, x2.item()), min(H, y2.item())
            if x2 <= x1 or y2 <= y1:
                crop = torch.zeros((3, 224, 224)).to(device)
            else:
                crop = img[:, y1:y2, x1:x2]
                crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            
            crop = (crop - torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)) / \
                   torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
            crops.append(crop)
            
        crops = torch.stack(crops)
        with torch.no_grad():
            clip_feat = self.args.clip_model.encode_image(crops)
            clip_feat = F.normalize(clip_feat, dim=-1)
        return clip_feat
    
    def _is_valid_unknown_geometry(self, box_cxcywh, eps=1e-6):
        """
        box_cxcywh: Tensor [4] in normalized coords
        return: bool
        """
        cx, cy, w, h = box_cxcywh.tolist()

        # 基本合法性
        if w <= 0 or h <= 0:
            return False

        area = w * h
        aspect = max(w / (h + eps), h / (w + eps))

        # 是否贴边
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        touch_border = (x1 < 0.01) or (y1 < 0.01) or (x2 > 0.99) or (y2 > 0.99)

        # 1) 极小面积过滤
        min_area = getattr(self.args, 'unk_min_area', 0.0015)
        if area < min_area:
            return False

        # 2) 极端长宽比过滤
        max_aspect = getattr(self.args, 'unk_max_aspect_ratio', 6.0)
        if aspect > max_aspect:
            return False

        # 3) 边缘+偏长 联合过滤
        border_aspect = getattr(self.args, 'unk_border_max_aspect_ratio', 3.5)
        if touch_border and aspect > border_aspect:
            return False

        return True

    def forward(self, outputs, targets, epoch):
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ['aux_outputs', 'enc_outputs', 'pred_obj', 'samples', 'pred_proj']}
        indices = self.matcher(outputs_without_aux, targets)
        
        dummy_pos_indices, dummy_neg_indices, dummy_vlm_weights = [], [], []
        samples = outputs.get('samples', None)
        has_clip = getattr(self.args, 'clip_model', None) is not None

        # 优化点 1：仅计算一次 GT 的 CLIP 特征供全局对齐使用
        if self.use_feature_align and has_clip and samples is not None and 'clip_feat' not in targets[0]:
            for i, t in enumerate(targets):
                if len(t['boxes']) == 0:
                    t['clip_feat'] = torch.empty((0, getattr(self.args, 'clip_dim', 512)), device=outputs['pred_logits'].device)
                else:
                    box_xyxy = box_ops.box_cxcywh_to_xyxy(t['boxes'])
                    t['clip_feat'] = self._crop_and_encode(samples.tensors[i], box_xyxy)
        
        
        stats = {
            'num_dummy_pos': 0.0,
            'num_dummy_neg': 0.0,
            'num_valid_unmatched': 0.0,
            'num_pos_candidates': 0.0,
            'num_neg_candidates': 0.0,

            'pos_energy_sum': 0.0,
            'neg_energy_sum': 0.0,
            'matched_energy_sum': 0.0,

            'dummy_pos_known_max_sum': 0.0,
            'dummy_neg_known_max_sum': 0.0,
            'dummy_pos_iou_sum': 0.0,
            'dummy_neg_iou_sum': 0.0,

            'pos_thresh_sum': 0.0,
            'neg_thresh_sum': 0.0,

            'num_pos_energy': 0,
            'num_neg_energy': 0,
            'num_matched_energy': 0,
            'num_dummy_pos_known': 0,
            'num_dummy_neg_known': 0,
            'num_dummy_pos_iou': 0,
            'num_dummy_neg_iou': 0,
            'num_thresh': 0,
        }
        # 在每张图上把 matched / dummy_pos / dummy_neg / threshold 都累计进 stats
        if self.enable_unk_label_obj and self.unk_label_start_epoch <= epoch:
            obj_scores = outputs['pred_obj']   # [B, Q], energy，越小越像物体
            num_queries = obj_scores.shape[1]

            max_pos_per_img = getattr(self.args, 'unk_pos_per_img', 1)
            max_neg_per_img = getattr(self.args, 'unk_neg_per_img', 2)
            known_reject_thresh = getattr(self.args, 'unk_cls_reject_thresh', 0.25)
            pos_quantile = getattr(self.args, 'unk_label_pos_quantile', 0.5)

            pred_probs = outputs['pred_logits'].detach().sigmoid().clone()
            pred_probs[:, :, self.invalid_cls_logits] = 0.0

            for i, (src_idx, _) in enumerate(indices):
                matched_scores = obj_scores[i, src_idx]

                if len(matched_scores) > 0:
                    base_thresh = torch.quantile(matched_scores.detach(), pos_quantile).item()
                    pos_thresh = base_thresh * self.unk_label_obj_score_thresh
                    stats['matched_energy_sum'] += matched_scores.sum().item()
                    stats['num_matched_energy'] += matched_scores.numel()
                else:
                    pos_thresh = getattr(self.args, 'default_pos_energy_thresh', 1.0)

                neg_thresh = pos_thresh + getattr(self.args, 'unk_label_neg_margin', 0.5)

                stats['pos_thresh_sum'] += pos_thresh
                stats['neg_thresh_sum'] += neg_thresh
                stats['num_thresh'] += 1

                all_queries = set(range(num_queries))
                matched_set = set(src_idx.tolist())
                unmatched = list(all_queries - matched_set)

                if not unmatched:
                    dummy_pos_indices.append([])
                    dummy_neg_indices.append([])
                    dummy_vlm_weights.append([])
                    continue

                box_xyxy_all = box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][i])
                gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(targets[i]['boxes'])

                if len(gt_boxes_xyxy) > 0:
                    ious = box_ops.box_iou(box_xyxy_all[unmatched], gt_boxes_xyxy)[0]
                    max_ious = ious.max(dim=1)[0]
                    valid_unmatched = [unmatched[j] for j, max_iou in enumerate(max_ious) if max_iou < 0.3]
                    unmatched_iou_map = {unmatched[j]: max_ious[j].item() for j in range(len(unmatched))}
                else:
                    valid_unmatched = unmatched
                    unmatched_iou_map = {q: 0.0 for q in unmatched}
                    
                geom_valid_pos = []
                for q in valid_unmatched:
                    box_q = outputs['pred_boxes'][i, q]   # cxcywh normalized
                    if self._is_valid_unknown_geometry(box_q):
                        geom_valid_pos.append(q)
                        
                valid_unmatched = geom_valid_pos
                stats['num_valid_unmatched'] += len(valid_unmatched)

                if len(valid_unmatched) == 0:
                    dummy_pos_indices.append([])
                    dummy_neg_indices.append([])
                    dummy_vlm_weights.append([])
                    continue

                cur_known_prob = pred_probs[i, :, :self.num_classes - 1]
                known_max = cur_known_prob.max(dim=-1)[0]

                
                        
                pos_candidates = [
                    q for q in geom_valid_pos
                    if obj_scores[i, q].item() < pos_thresh
                    and known_max[q].item() < known_reject_thresh
                ]
                
                neg_candidates = [
                    q for q in valid_unmatched
                    if obj_scores[i, q].item() > neg_thresh
                    and known_max[q].item() < known_reject_thresh
                ]

                stats['num_pos_candidates'] += len(pos_candidates)
                stats['num_neg_candidates'] += len(neg_candidates)

                pos_candidates_sorted = sorted(
                    pos_candidates,
                    key=lambda q: (
                        obj_scores[i, q].item(),         # energy 越低越优先
                        known_max[q].item(),             # 越不像已知越优先
                        unmatched_iou_map[q],            # 越远离 GT 越优先
                    )
                )
                
                neg_candidates_sorted = sorted(neg_candidates, key=lambda q: obj_scores[i, q].item(), reverse=True)

                dummy_pos = pos_candidates_sorted[:max_pos_per_img]
                dummy_neg = neg_candidates_sorted[:max_neg_per_img]

                dummy_pos_indices.append(dummy_pos)
                dummy_neg_indices.append(dummy_neg)

                stats['num_dummy_pos'] += len(dummy_pos)
                stats['num_dummy_neg'] += len(dummy_neg)

                if len(dummy_pos) > 0:
                    pos_energy_vals = obj_scores[i, dummy_pos]
                    stats['pos_energy_sum'] += pos_energy_vals.sum().item()
                    stats['num_pos_energy'] += pos_energy_vals.numel()

                    pos_known_vals = known_max[dummy_pos]
                    stats['dummy_pos_known_max_sum'] += pos_known_vals.sum().item()
                    stats['num_dummy_pos_known'] += pos_known_vals.numel()

                    pos_iou_vals = [unmatched_iou_map[q] for q in dummy_pos]
                    stats['dummy_pos_iou_sum'] += sum(pos_iou_vals)
                    stats['num_dummy_pos_iou'] += len(pos_iou_vals)

                if len(dummy_neg) > 0:
                    neg_energy_vals = obj_scores[i, dummy_neg]
                    stats['neg_energy_sum'] += neg_energy_vals.sum().item()
                    stats['num_neg_energy'] += neg_energy_vals.numel()

                    neg_known_vals = known_max[dummy_neg]
                    stats['dummy_neg_known_max_sum'] += neg_known_vals.sum().item()
                    stats['num_dummy_neg_known'] += neg_known_vals.numel()

                    neg_iou_vals = [unmatched_iou_map[q] for q in dummy_neg]
                    stats['dummy_neg_iou_sum'] += sum(neg_iou_vals)
                    stats['num_dummy_neg_iou'] += len(neg_iou_vals)

                if self.use_vlm_distill and len(dummy_pos) > 0 and has_clip and samples is not None:
                    box_cxcywh = outputs['pred_boxes'][i, dummy_pos]
                    box_xyxy = box_ops.box_cxcywh_to_xyxy(box_cxcywh)
                    clip_feat = self._crop_and_encode(samples.tensors[i], box_xyxy)

                    sim = clip_feat @ self.args.clip_text_features.T
                    sim_known = sim[:, :-1].max(dim=-1)[0]
                    sim_unk = sim[:, -1]

                    tau = getattr(self.args, 'vlm_tau', 0.1)
                    exp_unk = torch.exp(sim_unk / tau)
                    exp_known = torch.exp(sim_known / tau)
                    omega = exp_unk / (exp_unk + exp_known)

                    dummy_vlm_weights.append(omega.tolist())
                else:
                    dummy_vlm_weights.append([1.0] * len(dummy_pos))
                
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            kwargs = {
                'dummy_pos_indices': dummy_pos_indices,
                'dummy_neg_indices': dummy_neg_indices,
                'dummy_vlm_weights': dummy_vlm_weights,
                'epoch': epoch,
            }
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue

                    if loss == 'obj_likelihood' and self.args.etop and i > getattr(self.args, 'etop_layer', 1):
                        continue

                    # 关键：aux层不复用最终层伪标签
                    aux_kwargs = {
                        'dummy_pos_indices': None,
                        'dummy_neg_indices': None,
                        'dummy_vlm_weights': None,
                        'epoch': epoch,
                    }

                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes, **aux_kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets: bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'obj_likelihood', 'align']: continue
                kwargs = {'log': False} if loss == 'labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        device = outputs['pred_logits'].device

        stat_num_dummy_pos = torch.tensor(stats['num_dummy_pos'], dtype=torch.float32, device=device)
        stat_num_dummy_neg = torch.tensor(stats['num_dummy_neg'], dtype=torch.float32, device=device)
        stat_num_valid_unmatched = torch.tensor(stats['num_valid_unmatched'], dtype=torch.float32, device=device)
        stat_num_pos_candidates = torch.tensor(stats['num_pos_candidates'], dtype=torch.float32, device=device)
        stat_num_neg_candidates = torch.tensor(stats['num_neg_candidates'], dtype=torch.float32, device=device)

        stat_pos_energy_mean = torch.tensor(
            stats['pos_energy_sum'] / max(stats['num_pos_energy'], 1),
            dtype=torch.float32, device=device
        )
        stat_neg_energy_mean = torch.tensor(
            stats['neg_energy_sum'] / max(stats['num_neg_energy'], 1),
            dtype=torch.float32, device=device
        )
        stat_matched_energy_mean = torch.tensor(
            stats['matched_energy_sum'] / max(stats['num_matched_energy'], 1),
            dtype=torch.float32, device=device
        )

        stat_dummy_pos_known_max_mean = torch.tensor(
            stats['dummy_pos_known_max_sum'] / max(stats['num_dummy_pos_known'], 1),
            dtype=torch.float32, device=device
        )
        stat_dummy_neg_known_max_mean = torch.tensor(
            stats['dummy_neg_known_max_sum'] / max(stats['num_dummy_neg_known'], 1),
            dtype=torch.float32, device=device
        )
        stat_dummy_pos_iou_mean = torch.tensor(
            stats['dummy_pos_iou_sum'] / max(stats['num_dummy_pos_iou'], 1),
            dtype=torch.float32, device=device
        )
        stat_dummy_neg_iou_mean = torch.tensor(
            stats['dummy_neg_iou_sum'] / max(stats['num_dummy_neg_iou'], 1),
            dtype=torch.float32, device=device
        )

        stat_pos_thresh_mean = torch.tensor(
            stats['pos_thresh_sum'] / max(stats['num_thresh'], 1),
            dtype=torch.float32, device=device
        )
        stat_neg_thresh_mean = torch.tensor(
            stats['neg_thresh_sum'] / max(stats['num_thresh'], 1),
            dtype=torch.float32, device=device
        )

        losses.update({
            'stat_num_dummy_pos': stat_num_dummy_pos,
            'stat_num_dummy_neg': stat_num_dummy_neg,
            'stat_num_valid_unmatched': stat_num_valid_unmatched,
            'stat_num_pos_candidates': stat_num_pos_candidates,
            'stat_num_neg_candidates': stat_num_neg_candidates,

            'stat_pos_energy_mean': stat_pos_energy_mean,
            'stat_neg_energy_mean': stat_neg_energy_mean,
            'stat_matched_energy_mean': stat_matched_energy_mean,

            'stat_dummy_pos_known_max_mean': stat_dummy_pos_known_max_mean,
            'stat_dummy_neg_known_max_mean': stat_dummy_neg_known_max_mean,
            'stat_dummy_pos_iou_mean': stat_dummy_pos_iou_mean,
            'stat_dummy_neg_iou_mean': stat_dummy_neg_iou_mean,

            'stat_pos_thresh_mean': stat_pos_thresh_mean,
            'stat_neg_thresh_mean': stat_neg_thresh_mean,
        })
        return losses
    
class PostProcess(nn.Module):
    def __init__(
        self,
        invalid_cls_logits,
        num_classes,
        temperature=1.0,
        pred_per_im=100,
        known_thresh=0.05,
        unknown_thresh=0.05,
        enable_unknown_output=True,
    ):
        super().__init__()
        self.temperature = temperature
        self.invalid_cls_logits = invalid_cls_logits
        self.num_classes = num_classes
        self.pred_per_im = pred_per_im
        self.known_thresh = known_thresh
        self.unknown_thresh = unknown_thresh
        self.enable_unknown_output = enable_unknown_output

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits = outputs['pred_logits'].clone()
        pred_obj = outputs['pred_obj']
        out_bbox = outputs['pred_boxes']

        out_logits[:, :, self.invalid_cls_logits] = -10e10

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # [B, Q]
        obj_prob = torch.exp(-self.temperature * pred_obj)

        # 只看 known classes，不包含最后一个 unknown 类
        cls_prob_all = out_logits.sigmoid()
        known_prob = cls_prob_all[:, :, :self.num_classes - 1]
        max_known_scores, known_labels = known_prob.max(dim=-1)   # [B, Q], [B, Q]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        results = []

        unknown_label = self.num_classes - 1

        for b in range(out_logits.shape[0]):
            score_known = obj_prob[b] * max_known_scores[b]

            if self.enable_unknown_output:
                score_unknown = obj_prob[b] * (1.0 - max_known_scores[b])

                # 每个 query 只保留 known / unknown 更强的一路
                choose_unknown = score_unknown > score_known

                final_scores = torch.where(choose_unknown, score_unknown, score_known)
                final_labels = torch.where(
                    choose_unknown,
                    torch.full_like(known_labels[b], unknown_label),
                    known_labels[b]
                )

                keep = torch.where(
                    choose_unknown,
                    score_unknown > self.unknown_thresh,
                    score_known > self.known_thresh
                )
            else:
                final_scores = score_known
                final_labels = known_labels[b]
                keep = score_known > self.known_thresh

            final_scores = final_scores[keep]
            final_labels = final_labels[keep]
            final_boxes = boxes[b][keep]

            if final_scores.numel() > self.pred_per_im:
                topk_scores, topk_idx = torch.topk(final_scores, self.pred_per_im)
                final_scores = topk_scores
                final_labels = final_labels[topk_idx]
                final_boxes = final_boxes[topk_idx]

            img_h, img_w = target_sizes[b]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=final_boxes.device)
            final_boxes = final_boxes * scale_fct

            results.append({
                'scores': final_scores,
                'labels': final_labels,
                'boxes': final_boxes
            })

        return results

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
    
class ExemplarSelection(nn.Module):
    def __init__(self, args, num_classes, matcher, invalid_cls_logits, temperature=1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.invalid_cls_logits=invalid_cls_logits
        self.temperature=temperature
        logging.info('running with exemplar_replay_selection')
              
    def calc_energy_per_image(self, outputs, targets, indices):
        out_logits, pred_obj = outputs['pred_logits'], outputs['pred_obj']
        out_logits[:, :, self.invalid_cls_logits] = -10e10

        torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        logit_dist = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        prob = logit_dist*out_logits.sigmoid()

        image_sorted_scores = {}
        for i in range(len(targets)):
            image_sorted_scores[''.join([chr(int(c)) for c in targets[i]['org_image_id']])] = {'labels':targets[i]['labels'].cpu().numpy(),"scores": prob[i,indices[i][0],targets[i]['labels']].detach().cpu().numpy()}
        return [image_sorted_scores]

    def forward(self, samples, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj' and k != 'samples' and k != 'pred_proj'}
        indices = self.matcher(outputs_without_aux, targets)       
        return self.calc_energy_per_image(outputs, targets, indices)

def build(args):
    num_classes = args.num_classes
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS, num_classes-1))
    
    device = torch.device(args.device)
    
    # 动态初始化 CLIP 
    if getattr(args, 'use_feature_align', False):
        try:
            clip_model, _ = clip.load("ViT-B/32", device=device)
            for param in clip_model.parameters(): param.requires_grad = False
            args.clip_model = clip_model
            args.clip_dim = 512
            
            num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
            class_names = getattr(args, 'class_names', [f'object class {i}' for i in range(num_classes - 1)])
            safe_prompts = [f"a photo of a {class_names[i]}" if i < num_seen_classes else "a blank corrupted image" for i in range(num_classes - 1)]
            safe_prompts.append("a photo of an unknown generic object") 
            
            text_inputs = clip.tokenize(safe_prompts).to(device)
            with torch.no_grad():
                text_features = F.normalize(clip_model.encode_text(text_inputs), dim=-1)
            args.clip_text_features = text_features
            
            semantic_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
            if len(invalid_cls_logits) > 0: semantic_mask[invalid_cls_logits] = True
            args.semantic_mask = semantic_mask
        except Exception as e:
            logging.warning(f"Failed to load CLIP. VLM features disabled. Error: {e}")
            args.clip_model = None

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
        args, backbone, transformer, num_classes=num_classes, num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels, aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine, two_stage=args.two_stage,
    )
    
    matcher = build_matcher(args)
    
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef, 
        'loss_giou': args.giou_loss_coef, 
        'loss_obj': args.obj_loss_coef,
    }
    if getattr(args, 'use_feature_align', False):
        weight_dict['loss_align'] = getattr(args, 'align_loss_coef', 1.0) 

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'obj_likelihood'] 
    if getattr(args, 'use_feature_align', False): losses.append('align')

    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(
            invalid_cls_logits=invalid_cls_logits,
            num_classes=num_classes,
            temperature=args.obj_temp / args.hidden_dim,
            pred_per_im=args.pred_per_im,
            known_thresh=getattr(args, 'postproc_known_thresh', 0.05),
            unknown_thresh=getattr(args, 'postproc_unknown_thresh', 0.05),
            enable_unknown_output=getattr(args, 'enable_unknown_output', True),
        )
    }
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)
    
    return model, criterion, postprocessors, exemplar_selection