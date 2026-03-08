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
import copy

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=43215, stdout_to_server=True, stderr_to_server=True)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, num_classes: int = 81, empty_weight: float = 0.1):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    W = torch.ones(num_classes, dtype=prob.dtype, layout=prob.layout, device=prob.device)
    W[-1] = empty_weight
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=W)
    
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)  # 将输入张量展平，从第0维到第1维-[bs, num_queries, hidden_dim] -> [bs*num_queries, hidden_dim]
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
        
    def forward(self, x):
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2 / x.shape[-1]  # 计算每个query的特征向量的L2范数平方，并归一化 ### 重要的归一化
     
    
class FullProbObjectnessHead(nn.Module):
    """没有使用这个Head, 而是使用简单的L2距离替换马氏距离的计算, 发现计算量降低速度增加, 性能差不多, 更稳定. """
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
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)

        self.num_feature_levels = num_feature_levels
        if not two_stage:
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

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.prob_obj_head =  _get_clones(self.prob_obj_head, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
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
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_objectnesses = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_objectness = self.prob_obj_head[lvl](hs[lvl])

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

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_obj':outputs_objectness[-1]} 
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_objectness)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, objectness):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_obj': b, 'pred_boxes': c,}
                for a, b, c in zip(outputs_class[:-1], objectness[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, empty_weight=0.1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        
        self.empty_weight=empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.min_obj=-hidden_dim*math.log(0.9)
        
        # 基于物体性(分数)的自适应伪标签筛选
        self.enable_unk_label_obj = args.enable_unk_label_obj
        self.unk_label_obj_score_thresh = args.unk_label_obj_score_thresh
        self.unk_label_start_epoch = args.unk_label_start_epoch


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
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

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, 
                                     num_classes=self.num_classes, empty_weight=self.empty_weight) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, **kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
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

    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
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
                        dummy_pos_indices=None, dummy_neg_indices=None):
        
        logits = outputs["pred_obj"].squeeze(-1)  # [bs, query_num]
        device = logits.device

        # ---------- 收集所有参与计算的样本索引及标签 ----------
        batch_idx_list = []
        query_idx_list = []
        label_list = []
        
        # 1. 已知正样本（匈牙利匹配）
        if indices is not None:
            src_idx = self._get_src_permutation_idx(indices)  # (batch_idx, query_idx)
            if len(src_idx[0]) > 0:
                batch_idx_list.append(src_idx[0].to(device))
                query_idx_list.append(src_idx[1].to(device))
                label_list.append(torch.ones_like(src_idx[0], dtype=logits.dtype, device=device))

        # 2. 伪正样本（标签为1）
        if dummy_pos_indices is not None:
            for b_idx, q_list in enumerate(dummy_pos_indices):
                if len(q_list) > 0:
                    batch_idx_list.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
                    query_idx_list.append(torch.tensor(q_list, dtype=torch.long, device=device))
                    label_list.append(torch.ones(len(q_list), dtype=logits.dtype, device=device))

        # 3. 伪负样本（标签为0）
        if dummy_neg_indices is not None:
            for b_idx, q_list in enumerate(dummy_neg_indices):
                if len(q_list) > 0:
                    batch_idx_list.append(torch.full((len(q_list),), b_idx, dtype=torch.long, device=device))
                    query_idx_list.append(torch.tensor(q_list, dtype=torch.long, device=device))
                    label_list.append(torch.zeros(len(q_list), dtype=logits.dtype, device=device))

        # 如果没有样本参与，返回0损失（需要梯度，因此创建requires_grad的零张量）
        if len(batch_idx_list) == 0:
            zero_loss = torch.tensor(0.0, device=device, dtype=logits.dtype, requires_grad=True)
            return {'loss_obj': zero_loss}

        # 合并所有索引和标签
        batch_idx = torch.cat(batch_idx_list)
        query_idx = torch.cat(query_idx_list)
        labels = torch.cat(label_list)  # [N]

        # 提取对应的logits
        selected_logits = logits[batch_idx, query_idx]  # [N]

        # 扩展为 [N, 1] 以适应 sigmoid_focal_loss 的输入格式
        selected_logits = selected_logits.unsqueeze(1)  # [N, 1]
        selected_labels = labels.unsqueeze(1)           # [N, 1]

        # # --- Debug Log Start ---
        # with torch.no_grad():
        #     prob = selected_logits.sigmoid()
        #     logging.info(f"--- Loss Diagnosis ---")
        #     logging.info(f"Logits range: [{selected_logits.min():.4f}, {selected_logits.max():.4f}]") # 之前的logits范围没有经过归一化, 导致非常大. 
        #     logging.info(f"Prob range: [{prob.min():.4f}, {prob.max():.4f}]")
        #     logging.info(f"Labels unique: {torch.unique(selected_labels).tolist()}, Dtype: {selected_labels.dtype}")
        #     logging.info(f"Alpha: {self.focal_alpha}, Num Samples: {len(selected_labels)}")
            
        #     # 模拟计算一个不带 Focal 权重的标准 BCE
        #     standard_bce = F.binary_cross_entropy_with_logits(selected_logits, selected_labels, reduction="sum")
        #     logging.info(f"Standard BCE (Sum): {standard_bce.item():.6f}")
        # # --- Debug Log End ---
        
        total_loss = sigmoid_focal_loss_torch(
            inputs=selected_logits,   # [N, 1], 原始 logits
            targets=selected_labels,  # [N, 1], 0 或 1
            alpha=self.focal_alpha,   # 默认 0.25
            gamma=2.0,                # 默认 2
            reduction="sum"           # 先求和，方便后面除以 num_boxes
        )
        # logging.info(f"loss_obj_likelihood: total_loss={total_loss.item():.4f}, num_boxes={num_boxes}, avg_loss={total_loss.item() / num_boxes:.4f}")
        return {'loss_obj': total_loss / num_boxes}
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             epoch: 当前训练的epoch数, 用于控制某些损失的启用
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) # bs,(src_query_idx, tgt_instance_idx)
        
        dummy_pos_indices = []  # 每个 batch 的伪正样本 query 索引
        dummy_neg_indices = []  # 每个 batch 的伪负样本 query 索引
        
        if self.enable_unk_label_obj and self.unk_label_start_epoch <= epoch:
            # Add a dummy target for unknown objects
            # 1. 计算匹配上的query的objectness的平均值乘以系数得到阈值. 
            #    未匹配上的query的objectness高于阈值为高置信度未知伪正样本, 远低于阈值为高置信度负样本.
            obj_scores = outputs['pred_obj'] # bs, num_queries, 1
            num_queries = obj_scores.shape[1]
            
            for i, (src_idx, _) in enumerate(indices):
                # 1. 计算当前 batch 的阈值：匹配 query 得分的均值 * 系数
                matched_scores = obj_scores[i, src_idx]  # GPU
                act_matched_scores = matched_scores.sigmoid()
                thresh = act_matched_scores.mean().item() * self.unk_label_obj_score_thresh # item()得到python数据类型, 与设备无关, 始终在CPU上, 运算时框架会处理.
                # logging.info(f"act_matched_scores.mean(): {act_matched_scores.mean().item():.4f}, thresh: {thresh:.4f}")
                # 2. 找出未匹配的 query 索引
                all_queries = set(range(num_queries))
                matched_set = set(src_idx.tolist())
                unmatched = list(all_queries - matched_set)

                if not unmatched:
                    dummy_pos_indices.append([])
                    dummy_neg_indices.append([])
                    continue

                # 3. 根据 objectness 得分筛选伪正/负样本
                unmatched_scores = obj_scores[i, unmatched]  # (num_unmatched, 1)
                act_unmatched_scores = unmatched_scores.sigmoid()
                # 伪正样本：得分 > thresh，取最高的1个
                pos_candidates = [unmatched[j] for j, s in enumerate(act_unmatched_scores) if s > thresh]
                pos_candidates_sorted = sorted(pos_candidates, key=lambda idx: obj_scores[i, idx].item(), reverse=True)
                dummy_pos = pos_candidates_sorted[:1]  # 最多取一个
                dummy_pos_indices.append(dummy_pos)

                # 伪负样本：得分 < (1 - thresh)，取最低的两个
                neg_candidates = [unmatched[j] for j, s in enumerate(act_unmatched_scores) if s < (1 - thresh)]
                neg_candidates_sorted = sorted(neg_candidates, key=lambda idx: obj_scores[i, idx].item())
                dummy_neg = neg_candidates_sorted[:2]
                if len(dummy_neg) > 0:
                    logging.info(f"Epoch {epoch}: Batch {i}, thresh={thresh:.4f}, taking {len(dummy_pos)} dummy pos and {len(dummy_neg)} dummy neg samples.")
                dummy_neg_indices.append(dummy_neg)
                

        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            kwargs['dummy_pos_indices'] = dummy_pos_indices
            kwargs['dummy_neg_indices'] = dummy_neg_indices
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
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
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                if loss == 'obj_likelihood':
                    # Objectness loss is only applied to the decoder outputs, not the encoder outputs.
                    continue
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, invalid_cls_logits, temperature=1, pred_per_im=100):
        super().__init__()
        self.temperature=temperature
        self.invalid_cls_logits=invalid_cls_logits
        self.pred_per_im=pred_per_im

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """        
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
    """ Very simple multi-layer perceptron (also called FFN)"""

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
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs' and k !='pred_obj'}
        indices = self.matcher(outputs_without_aux, targets)       
        return self.calc_energy_per_image(outputs, targets, indices)


def build(args):
    num_classes = args.num_classes
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS, num_classes-1))
    logging.info("Invalid class range: " + str(invalid_cls_logits))
    
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        
    matcher = build_matcher(args)
    ####### 这里添加损失权重, 在engine中可以通过epoch控制是否启用该Loss #########
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef, 
        'loss_giou': args.giou_loss_coef, 
        'loss_obj': args.obj_loss_coef,
        }
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality','obj_likelihood']
    if args.masks:
        losses += ["masks"]

        
    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp/args.hidden_dim)
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, exemplar_selection
