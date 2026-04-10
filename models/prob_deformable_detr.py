# ------------------------------------------------------------------------
# PROB: Probabilistic Objectness for Open World Object Detection.
# This refactor removes segmentation-only branches from the main detection path
# and adds semantic output aliases without breaking checkpoint parameter names.
# ------------------------------------------------------------------------
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher


def _get_clones(module, num_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 81, empty_weight: float = 0.1):
    probabilities = inputs.sigmoid()
    class_weight = torch.ones(num_classes, dtype=probabilities.dtype, layout=probabilities.layout, device=probabilities.device)
    class_weight[-1] = empty_weight
    binary_cross_entropy = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=class_weight)
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = binary_cross_entropy * ((1.0 - pt) ** gamma)
    if alpha >= 0:
        alpha_term = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_term * loss
    return loss.mean(1).sum() / num_boxes


class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0, 1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()

    def forward(self, features):
        flat_features = self.flatten(features)
        normalized = self.objectness_bn(flat_features).unflatten(0, features.shape[:2])
        return normalized.norm(dim=-1) ** 2


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))

    def forward(self, features):
        for layer_index, layer in enumerate(self.layers):
            features = F.relu(layer(features)) if layer_index < self.num_layers - 1 else layer(features)
        return features


class DeformableDETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # Keep the original attribute names for checkpoint compatibility.
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outputs = len(backbone.strides)
            input_projection_layers = []
            for level_index in range(num_backbone_outputs):
                in_channels = backbone.num_channels[level_index]
                input_projection_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outputs):
                input_projection_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_projection_layers)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for input_projection in self.input_proj:
            nn.init.xavier_uniform_(input_projection[0].weight, gain=1)
            nn.init.constant_(input_projection[0].bias, 0)

        num_prediction_layers = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_prediction_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, num_prediction_layers)
            self.prob_obj_head = _get_clones(self.prob_obj_head, num_prediction_layers)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_prediction_layers)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_prediction_layers)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_prediction_layers)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_head in self.bbox_embed:
                nn.init.constant_(box_head.layers[-1].bias.data[2:], 0.0)

    @property
    def classification_head(self):
        return self.class_embed

    @property
    def box_regression_head(self):
        return self.bbox_embed

    @property
    def objectness_energy_head(self):
        return self.prob_obj_head

    def forward(self, samples: NestedTensor, return_vis_debug: bool = False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, positional_embeddings = self.backbone(samples)

        encoder_inputs = []
        encoder_masks = []
        for level_index, feature in enumerate(features):
            source, mask = feature.decompose()
            encoder_inputs.append(self.input_proj[level_index](source))
            encoder_masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(encoder_inputs):
            current_length = len(encoder_inputs)
            for level_index in range(current_length, self.num_feature_levels):
                if level_index == current_length:
                    source = self.input_proj[level_index](features[-1].tensors)
                else:
                    source = self.input_proj[level_index](encoder_inputs[-1])
                full_mask = samples.mask
                resized_mask = F.interpolate(full_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                positional_feature = self.backbone[1](NestedTensor(source, resized_mask)).to(source.dtype)
                encoder_inputs.append(source)
                encoder_masks.append(resized_mask)
                positional_embeddings.append(positional_feature)

        query_embeddings = None if self.two_stage else self.query_embed.weight
        decoder_hidden_states, initial_reference_points, intermediate_reference_points, encoder_class_logits, encoder_box_coordinates, encoder_info = self.transformer(
            encoder_inputs,
            encoder_masks,
            positional_embeddings,
            query_embeddings,
        )

        per_layer_class_logits = []
        per_layer_box_coordinates = []
        per_layer_objectness_energies = []
        for layer_index in range(decoder_hidden_states.shape[0]):
            reference = initial_reference_points if layer_index == 0 else intermediate_reference_points[layer_index - 1]
            reference = inverse_sigmoid(reference)
            layer_hidden = decoder_hidden_states[layer_index]
            class_logits = self.class_embed[layer_index](layer_hidden)
            objectness_energy = self.prob_obj_head[layer_index](layer_hidden)
            box_delta = self.bbox_embed[layer_index](layer_hidden)
            if reference.shape[-1] == 4:
                box_delta += reference
            else:
                box_delta[..., :2] += reference
            box_coordinates = box_delta.sigmoid()
            per_layer_class_logits.append(class_logits)
            per_layer_box_coordinates.append(box_coordinates)
            per_layer_objectness_energies.append(objectness_energy)

        per_layer_class_logits = torch.stack(per_layer_class_logits)
        per_layer_box_coordinates = torch.stack(per_layer_box_coordinates)
        per_layer_objectness_energies = torch.stack(per_layer_objectness_energies)

        output = {
            'pred_class_logits': per_layer_class_logits[-1],
            'pred_boxes': per_layer_box_coordinates[-1],
            'pred_objectness_energy': per_layer_objectness_energies[-1],
            # Backward-compatible aliases.
            'pred_logits': per_layer_class_logits[-1],
            'pred_obj': per_layer_objectness_energies[-1],
        }
        if self.aux_loss:
            output['aux_outputs'] = self._build_auxiliary_outputs(per_layer_class_logits, per_layer_box_coordinates, per_layer_objectness_energies)
        if self.two_stage:
            output['enc_outputs'] = {
                'pred_class_logits': encoder_class_logits,
                'pred_boxes': encoder_box_coordinates.sigmoid(),
                'pred_logits': encoder_class_logits,
            }
        if return_vis_debug:
            hidden_dim = float(per_layer_objectness_energies.shape[-1]) if per_layer_objectness_energies.ndim > 2 else 256.0
            output['vis_debug'] = {
                'layer_objectness_probability': torch.exp(-(1.0 / hidden_dim) * per_layer_objectness_energies.detach()),
                'layer_max_known_class_probability': per_layer_class_logits.detach().sigmoid().max(dim=-1).values,
            }
        return output

    @torch.jit.unused
    def _build_auxiliary_outputs(self, per_layer_class_logits, per_layer_box_coordinates, per_layer_objectness_energies):
        aux_outputs = []
        for class_logits, objectness_energy, box_coordinates in zip(per_layer_class_logits[:-1], per_layer_objectness_energies[:-1], per_layer_box_coordinates[:-1]):
            aux_outputs.append({
                'pred_class_logits': class_logits,
                'pred_boxes': box_coordinates,
                'pred_objectness_energy': objectness_energy,
                'pred_logits': class_logits,
                'pred_obj': objectness_energy,
            })
        return aux_outputs


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, empty_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.empty_weight = empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.min_objectness_energy = -hidden_dim * math.log(0.9)

    def _get_output(self, outputs, *keys):
        for key in keys:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        raise KeyError(f'None of the requested keys exist: {keys}')

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        class_logits = self._get_output(outputs, 'pred_class_logits', 'pred_logits').clone()
        class_logits[:, :, self.invalid_cls_logits] = -10e10
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        target_classes = torch.full(class_logits.shape[:2], self.num_classes - 1, dtype=torch.int64, device=class_logits.device)
        matched_target_classes = torch.cat([target['labels'][target_indices] for target, (_, target_indices) in zip(targets, indices)])
        target_classes[(matched_batch_indices, matched_query_indices)] = matched_target_classes
        target_onehot = torch.zeros(class_logits.shape, dtype=class_logits.dtype, layout=class_logits.layout, device=class_logits.device)
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        classification_loss = sigmoid_focal_loss(class_logits, target_onehot, num_boxes, alpha=self.focal_alpha, num_classes=self.num_classes, empty_weight=self.empty_weight) * class_logits.shape[1]
        losses = {'loss_ce': classification_loss}
        if log:
            losses['class_error'] = 100 - accuracy(class_logits[(matched_batch_indices, matched_query_indices)], matched_target_classes)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        class_logits = self._get_output(outputs, 'pred_class_logits', 'pred_logits')
        target_lengths = torch.as_tensor([len(target['labels']) for target in targets], device=class_logits.device)
        cardinality_prediction = (class_logits.argmax(-1) != class_logits.shape[-1] - 1).sum(1)
        return {'cardinality_error': F.l1_loss(cardinality_prediction.float(), target_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][(matched_batch_indices, matched_query_indices)]
        target_boxes = torch.cat([target['boxes'][target_indices] for target, (_, target_indices) in zip(targets, indices)], dim=0)
        box_l1_loss = F.l1_loss(src_boxes, target_boxes, reduction='none')
        generalized_iou_loss = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {
            'loss_bbox': box_l1_loss.sum() / num_boxes,
            'loss_giou': generalized_iou_loss.sum() / num_boxes,
        }

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        objectness_energy = self._get_output(outputs, 'pred_objectness_energy', 'pred_obj')[(matched_batch_indices, matched_query_indices)]
        return {'loss_obj_ll': torch.clamp(objectness_energy, min=self.min_objectness_energy).sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        batch_indices = torch.cat([torch.full_like(source_indices, batch_index) for batch_index, (source_indices, _) in enumerate(indices)])
        source_indices = torch.cat([source_indices for (source_indices, _) in indices])
        return batch_indices, source_indices

    def get_loss(self, loss_name, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
        }
        assert loss_name in loss_map, f'Unsupported loss: {loss_name}'
        return loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        matching_outputs = {
            'pred_logits': self._get_output(outputs, 'pred_class_logits', 'pred_logits'),
            'pred_boxes': outputs['pred_boxes'],
        }
        indices = self.matcher(matching_outputs, targets)
        num_boxes = sum(len(target['labels']) for target in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss_name in self.losses:
            losses.update(self.get_loss(loss_name, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for layer_index, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher({'pred_logits': self._get_output(aux_outputs, 'pred_class_logits', 'pred_logits'), 'pred_boxes': aux_outputs['pred_boxes']}, targets)
                for loss_name in self.losses:
                    kwargs = {'log': False} if loss_name == 'labels' else {}
                    aux_loss = self.get_loss(loss_name, aux_outputs, targets, aux_indices, num_boxes, **kwargs)
                    losses.update({f'{key}_{layer_index}': value for key, value in aux_loss.items()})

        if 'enc_outputs' in outputs:
            encoder_outputs = outputs['enc_outputs']
            binary_targets = copy.deepcopy(targets)
            for target in binary_targets:
                target['labels'] = torch.zeros_like(target['labels'])
            encoder_indices = self.matcher({'pred_logits': self._get_output(encoder_outputs, 'pred_class_logits', 'pred_logits'), 'pred_boxes': encoder_outputs['pred_boxes']}, binary_targets)
            for loss_name in ['labels', 'boxes']:
                kwargs = {'log': False} if loss_name == 'labels' else {}
                encoder_loss = self.get_loss(loss_name, encoder_outputs, binary_targets, encoder_indices, num_boxes, **kwargs)
                losses.update({f'{key}_enc': value for key, value in encoder_loss.items()})
        return losses


class PostProcess(nn.Module):
    def __init__(self, invalid_cls_logits, temperature=1.0, pred_per_im=100):
        super().__init__()
        self.temperature = temperature
        self.invalid_cls_logits = invalid_cls_logits
        self.pred_per_im = pred_per_im

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        class_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits').clone()
        objectness_energy = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
        predicted_boxes = outputs['pred_boxes']
        class_logits[:, :, self.invalid_cls_logits] = -10e10
        objectness_probability = torch.exp(-self.temperature * objectness_energy).unsqueeze(-1)
        fused_probability = objectness_probability * class_logits.sigmoid()
        topk_values, topk_indices = torch.topk(fused_probability.view(class_logits.shape[0], -1), self.pred_per_im, dim=1)
        scores = topk_values
        topk_boxes = topk_indices // class_logits.shape[2]
        labels = topk_indices % class_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(predicted_boxes)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        image_height, image_width = target_sizes.unbind(1)
        scale = torch.stack([image_width, image_height, image_width, image_height], dim=1)
        boxes = boxes * scale[:, None, :]
        return [{'scores': score, 'labels': label, 'boxes': box} for score, label, box in zip(scores, labels, boxes)]


class ExemplarSelection(nn.Module):
    def __init__(self, args, num_classes, matcher, invalid_cls_logits, temperature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.invalid_cls_logits = invalid_cls_logits
        self.temperature = temperature
        print('running with exemplar_replay_selection')

    def calc_energy_per_image(self, outputs, targets, indices):
        class_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits').clone()
        objectness_energy = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
        class_logits[:, :, self.invalid_cls_logits] = -10e10
        fused_scores = torch.exp(-self.temperature * objectness_energy).unsqueeze(-1) * class_logits.sigmoid()
        image_scores = {}
        for batch_index in range(len(targets)):
            image_scores[''.join([chr(int(char)) for char in targets[batch_index]['org_image_id']])] = {
                'labels': targets[batch_index]['labels'].cpu().numpy(),
                'scores': fused_scores[batch_index, indices[batch_index][0], targets[batch_index]['labels']].detach().cpu().numpy(),
            }
        return [image_scores]

    def forward(self, samples, outputs, targets):
        matching_outputs = {
            'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'),
            'pred_boxes': outputs['pred_boxes'],
        }
        indices = self.matcher(matching_outputs, targets)
        return self.calc_energy_per_image(outputs, targets, indices)


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    raise KeyError(f'Missing output keys: {keys}')


def build(args):
    num_classes = args.num_classes
    invalid_cls_logits = list(range(args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS, num_classes - 1))
    print('Invalid class range: ' + str(invalid_cls_logits))
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

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_obj_ll': args.obj_loss_coef,
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for layer_index in range(args.dec_layers - 1):
            aux_weight_dict.update({f'{key}_{layer_index}': value for key, value in weight_dict.items()})
        aux_weight_dict.update({f'{key}_enc': value for key, value in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'obj_likelihood']
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(invalid_cls_logits, temperature=args.obj_temp / args.hidden_dim)}
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp / args.hidden_dim)
    return model, criterion, postprocessors, exemplar_selection
