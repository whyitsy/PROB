# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss as sigmoid_focal_loss_torch
from torch import nn
import math
import logging
import copy
import clip  # [NEW] 引入 CLIP

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
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
    #### new 
    if valid_mask is not None:
        loss = loss * valid_mask.unsqueeze(-1)  # 将 valid_mask 应用于每个类别维度
        
    return loss.mean(1).sum() / num_boxes


class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)  
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
        
    def forward(self, x):
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2 / x.shape[-1]
     
    
class FullProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim=256, device='cpu'):
        super().__init__()
        self.flatten = nn.Flatten(0, 1)
        self.momentum = 0.1
        self.obj_mean=nn.Parameter(torch.ones(hidden_dim, device=device), requires_grad=False)
        self.obj_cov=nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        self.inv_obj_cov=nn.Parameter(torch.eye(hidden_dim, device=device), requires_grad=False)
        self.device=device
        self.hidden_dim=hidden_dim
            
    def update_params(self,x):
        out=self.flatten(x).detach()
        obj_mean=out.mean(dim=0)
        obj_cov=torch.cov(out.T)
        self.obj_mean.data = self.obj_mean*(1-self.momentum) + self.momentum*obj_mean
        self.obj_cov.data = self.obj_cov*(1-self.momentum) + self.momentum*obj_cov
        return
    
    def update_icov(self):
        self.inv_obj_cov.data = torch.pinverse(self.obj_cov.detach().cpu(), rcond=1e-6).to(self.device)
        return
        
    def mahalanobis(self, x):
        out=self.flatten(x)
        delta = out - self.obj_mean
        m = (delta * torch.matmul(self.inv_obj_cov, delta.T).T).sum(dim=-1)
        return m.unflatten(0, x.shape[:2])
    
    def set_momentum(self, m):
        self.momentum=m
        return
    
    def forward(self, x):
        if self.training:
            self.update_params(x)
        return self.mahalanobis(x)


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
        
        # [NEW] 修改投影头维度以对齐 CLIP，假设 args.clip_dim 默认 512
        clip_dim = getattr(args, 'clip_dim', 512)
        self.proj_head = MLP(hidden_dim, hidden_dim, clip_dim, 2) 

        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
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
            self.prob_obj_head =  _get_clones(self.prob_obj_head, num_pred)
            self.proj_head = _get_clones(self.proj_head, transformer.decoder.num_layers) # [NEW] 克隆投影头
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_pred)])
            self.proj_head = nn.ModuleList([self.proj_head for _ in range(transformer.decoder.num_layers)]) # [NEW]
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

        query_embeds = self.query_embed.weight
        clip_txt = getattr(self.args, 'clip_text_features', None)
        semantic_mask = getattr(self.args, 'semantic_mask', None)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, query_embeds, tdqi=self.args.tdqi, tdqi_query_num=self.args.tdqi_query_num, clip_text_features=clip_txt, semantic_mask=semantic_mask)

        outputs_classes = []
        outputs_coords = []
        outputs_objectnesses = []
        outputs_projs = [] # [NEW]

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.args.etop and lvl <= self.args.etop_layer:
                outputs_objectness = self.prob_obj_head[lvl](hs[lvl])
                outputs_objectness_shape = outputs_objectness.shape
                device = outputs_objectness.device
            else:
                outputs_objectness = torch.zeros(outputs_objectness_shape, device=device)

            # [NEW] 提取投影特征
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
        outputs_proj = torch.stack(outputs_projs) # [NEW]
        
        obj_layer = self.args.etop_layer if self.args.etop else -1
        
        # [NEW] 将 samples 和 pred_proj 加入输出字典，供 SetCriterion 在线截取图像
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 
               'pred_obj': outputs_objectness[obj_layer], 'pred_proj': outputs_proj[-1], 'samples': samples} 
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objectness, outputs_proj)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness, outputs_proj):
        aux_loss = [{'pred_logits': a, 'pred_obj': b, 'pred_boxes': c, 'pred_proj': d}
                for a, b, c, d in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1], outputs_proj[:-1])]
        return aux_loss


class SetCriterion(nn.Module):
    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, empty_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        self.empty_weight=empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.min_obj=-hidden_dim*math.log(0.9)
        
        self.enable_unk_label_obj = args.enable_unk_label_obj
        self.unk_label_obj_score_thresh = args.unk_label_obj_score_thresh
        self.unk_label_start_epoch = args.unk_label_start_epoch
        self.args = args

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:,:, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # [NEW: 核心豁免机制] ========================================
        # 构建一个有效性掩码 (Valid Mask)，默认全是 1.0 (全额计算 Loss)
        valid_mask = torch.ones([src_logits.shape[0], src_logits.shape[1]], dtype=src_logits.dtype, device=src_logits.device)
        
        # 如果某个 Query 被我们挖掘为“伪正样本 (未知)”，说明它不是背景！
        # 我们将其 valid_mask 设为 0，让主分类 Loss 不要把它当成背景来惩罚
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
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # [NEW] 跨模态特征对齐损失
    def loss_align(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_proj' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_proj = outputs['pred_proj'][idx]

        # 检查 target 中是否有预先通过 CLIP 在线提取好的 GT 特征
        if 'clip_feat' not in targets[0] or len(src_proj) == 0:
            return {'loss_align': src_proj.sum() * 0.0} 

        target_clip = torch.cat([t['clip_feat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_clip = target_clip.to(src_proj.device)
        target_clip = target_clip.to(src_proj.dtype)
        

        src_proj_norm = F.normalize(src_proj, p=2, dim=-1)
        # 余弦距离损失
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
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks[tgt_idx].flatten(1)
        losses = {
            "loss_mask": seg_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes,
                        dummy_pos_indices=None, dummy_neg_indices=None, dummy_vlm_weights=None):
        
        logits = outputs["pred_obj"].squeeze(-1)
        device = logits.device

        batch_idx_list = []
        query_idx_list = []
        label_list = []
        
        # 1. 已知正样本（权重始终为 1.0）
        if indices is not None:
            src_idx = self._get_src_permutation_idx(indices)
            if len(src_idx[0]) > 0:
                batch_idx_list.append(src_idx[0].to(device))
                query_idx_list.append(src_idx[1].to(device))
                label_list.append(torch.ones_like(src_idx[0], dtype=logits.dtype, device=device))

        # 2. 伪正样本（[NEW] 权重使用 VLM 提供的 omega）
        if dummy_pos_indices is not None:
            for b_idx, q_list in enumerate(dummy_pos_indices):
                if len(q_list) > 0:
                    batch_idx_list.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
                    query_idx_list.append(torch.tensor(q_list, dtype=torch.long, device=device))
                    
                    w = dummy_vlm_weights[b_idx] if dummy_vlm_weights else [1.0] * len(q_list)
                    label_list.append(torch.tensor(w, dtype=logits.dtype, device=device))

        # # 3. 伪负样本（权重为 0）
        # if dummy_neg_indices is not None:
        #     for b_idx, q_list in enumerate(dummy_neg_indices):
        #         if len(q_list) > 0:
        #             batch_idx_list.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
        #             query_idx_list.append(torch.tensor(q_list, dtype=torch.long, device=device))
        #             label_list.append(torch.zeros(len(q_list), dtype=logits.dtype, device=device))

        if len(batch_idx_list) == 0:
            zero_loss = torch.tensor(0.0, device=device, dtype=logits.dtype, requires_grad=True)
            return {'loss_obj': zero_loss}

        batch_idx = torch.cat(batch_idx_list)
        query_idx = torch.cat(query_idx_list)
        labels = torch.cat(label_list) 

        selected_logits = logits[batch_idx, query_idx].unsqueeze(1)  
        selected_labels = labels.unsqueeze(1)           

        total_loss = sigmoid_focal_loss_torch(
            inputs=selected_logits,   
            targets=selected_labels,  # 允许浮点软标签！
            alpha=self.focal_alpha,   
            gamma=2.0,                
            reduction="sum"           
        )
        return {'loss_obj': total_loss / num_boxes}
        
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
            'align': self.loss_align # [NEW]
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # [NEW] 核心在线裁剪辅助函数
    def _crop_and_encode(self, img_tensor, box_xyxy):
        """在线将 DETR 归一化图像还原、裁剪并送入 CLIP"""
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

    def forward(self, outputs, targets, epoch):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj' and k != 'samples' and k != 'pred_proj'}

        indices = self.matcher(outputs_without_aux, targets)
        
        dummy_pos_indices = []  
        dummy_neg_indices = []  
        dummy_vlm_weights = []  # [NEW]
        
        samples = outputs.get('samples', None)
        has_clip = getattr(self.args, 'clip_model', None) is not None

        # [NEW] 优化点 1：仅计算一次 GT 的 CLIP 特征供全局对齐使用
        if has_clip and samples is not None and 'clip_feat' not in targets[0]:
            for i, t in enumerate(targets):
                if len(t['boxes']) == 0:
                    t['clip_feat'] = torch.empty((0, self.args.clip_dim), device=outputs['pred_logits'].device)
                else:
                    box_xyxy = box_ops.box_cxcywh_to_xyxy(t['boxes'])
                    t['clip_feat'] = self._crop_and_encode(samples.tensors[i], box_xyxy)

        if self.enable_unk_label_obj and self.unk_label_start_epoch <= epoch:
            obj_scores = outputs['pred_obj'] 
            num_queries = obj_scores.shape[1]
            
            for i, (src_idx, _) in enumerate(indices):
                matched_scores = obj_scores[i, src_idx]  
                act_matched_scores = matched_scores.sigmoid()
                thresh = act_matched_scores.mean().item() * self.unk_label_obj_score_thresh 
                
                all_queries = set(range(num_queries))
                matched_set = set(src_idx.tolist())
                unmatched = list(all_queries - matched_set)

                if not unmatched:
                    dummy_pos_indices.append([])
                    dummy_neg_indices.append([])
                    dummy_vlm_weights.append([])
                    continue

                unmatched_scores = obj_scores[i, unmatched]  
                act_unmatched_scores = unmatched_scores.sigmoid()
                
                pos_candidates = [unmatched[j] for j, s in enumerate(act_unmatched_scores) if s > thresh]
                pos_candidates_sorted = sorted(pos_candidates, key=lambda idx: obj_scores[i, idx].item(), reverse=True)
                dummy_pos = pos_candidates_sorted[:1]  
                dummy_pos_indices.append(dummy_pos)

                neg_candidates = [unmatched[j] for j, s in enumerate(act_unmatched_scores) if s < (1 - thresh)]
                neg_candidates_sorted = sorted(neg_candidates, key=lambda idx: obj_scores[i, idx].item())
                dummy_neg = neg_candidates_sorted[:2]
                dummy_neg_indices.append(dummy_neg)

                # [NEW] 优化点 2：在线提取伪正样本 CLIP 特征，并计算蒸馏软标签 omega
                if len(dummy_pos) > 0 and has_clip and samples is not None:
                    box_cxcywh = outputs['pred_boxes'][i, dummy_pos]
                    box_xyxy = box_ops.box_cxcywh_to_xyxy(box_cxcywh)
                    
                    clip_feat = self._crop_and_encode(samples.tensors[i], box_xyxy)
                    
                    sim = clip_feat @ self.args.clip_text_features.T
                    sim_known = sim[:, :-1].max(dim=-1)[0]  # 已知类别的最高相似度
                    sim_unk = sim[:, -1]                    # 未知通配符的相似度
                    
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
            kwargs = {}
            kwargs['dummy_pos_indices'] = dummy_pos_indices
            kwargs['dummy_neg_indices'] = dummy_neg_indices
            kwargs['dummy_vlm_weights'] = dummy_vlm_weights # [NEW] 传递 VLM 软权重
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    if loss == 'obj_likelihood':
                        if self.args.etop and i > self.args.etop_layer:
                            continue
                    # [NEW] Aux loss 也要传递辅助权重
                    kwargs['dummy_pos_indices'] = dummy_pos_indices
                    kwargs['dummy_neg_indices'] = dummy_neg_indices
                    kwargs['dummy_vlm_weights'] = dummy_vlm_weights 
                    
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks' or loss == 'obj_likelihood' or loss == 'align': # [NEW] 排除 align
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100):
        super().__init__()
        self.temperature=temperature
        self.invalid_cls_logits=invalid_cls_logits
        self.pred_per_im=pred_per_im

    @torch.no_grad()
    def forward(self, outputs, target_sizes):      
        out_logits, pred_obj, out_bbox = outputs['pred_logits'], outputs['pred_obj'], outputs['pred_boxes']
        out_logits[:,:, self.invalid_cls_logits] = -10e10

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = torch.exp(-self.temperature*pred_obj).unsqueeze(-1)
        prob = obj_prob*out_logits.sigmoid()

        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im, dim=1)
        scores = topk_values
        
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
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
        print(f'running with exemplar_replay_selection')   
              
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
    logging.info("Invalid class range: " + str(invalid_cls_logits))
    
    device = torch.device(args.device)
    
    # [NEW] 极简且优雅的在线 CLIP 初始化与文本库构建
    try:
        clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in clip_model.parameters():
            param.requires_grad = False
        args.clip_model = clip_model
        args.clip_dim = 512
        
        # 1. 获取当前 Task 允许看见的类别数量
        num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        class_names = getattr(args, 'class_names', [f'object class {i}' for i in range(num_classes - 1)])
        
        # 2. 构造安全的 Prompts (防止文字层面的泄露)
        safe_prompts = []
        for i in range(num_classes - 1):
            if i < num_seen_classes:
                safe_prompts.append(f"a photo of a {class_names[i]}")
            else:
                # 对于未来的类别，用无意义的占位符代替，防止 CLIP 提取到有效语义
                safe_prompts.append("a blank corrupted image") 
                
        # 最后一个是未知物体的通配符 (对应背景/未知)
        safe_prompts.append("a photo of an unknown generic object") 
        
        text_inputs = clip.tokenize(safe_prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            text_features = F.normalize(text_features, dim=-1)
        args.clip_text_features = text_features
        
        # 3. [核心修复] 构造 Attention Mask 
        # True 表示该位置在 Attention 中被屏蔽 (忽略)，False 表示参与计算
        # 维度为 [num_classes]，其中 invalid_cls_logits 对应的位置设为 True
        semantic_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
        if len(invalid_cls_logits) > 0:
            semantic_mask[invalid_cls_logits] = True
        args.semantic_mask = semantic_mask
        
        logging.info(f"Loaded CLIP text bank. Seen classes: {num_seen_classes}. Masked classes: {len(invalid_cls_logits)}")
    except Exception as e:
        logging.warning(f"Failed to load CLIP. Online VSAD alignment will be disabled. Error: {e}")
        args.clip_model = None
        args.clip_text_features = None
        args.semantic_mask = None

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
        args, backbone, transformer, num_classes=num_classes, num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels, aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine, two_stage=args.two_stage,
    )
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        
    matcher = build_matcher(args)
    
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef, 
        'loss_giou': args.giou_loss_coef, 
        'loss_obj': args.obj_loss_coef,
        'loss_align': getattr(args, 'align_loss_coef', 1.0) # [NEW] 默认打开对齐损失
        }
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
        
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # [NEW] 注册 align
    losses = ['labels', 'boxes', 'cardinality','obj_likelihood', 'align'] 
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)
    
    return model, criterion, postprocessors, exemplar_selection