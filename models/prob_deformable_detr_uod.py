# ------------------------------------------------------------------------
# UOD: Explicit unknownness modeling on top of PROB.
# This refactor removes segmentation branches from the main detection path,
# adds semantic output aliases, and exposes faithful pseudo-mining debug data
# for visualization.
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
from models.ops.modules import MSDeformAttn
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .matcher import build_matcher
from .prob_deformable_detr import ProbObjectnessHead, sigmoid_focal_loss


def _get_clones(module, num_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def _energy_to_prob(energy, temperature):
    return torch.exp(-temperature * energy).clamp(min=1e-6, max=1.0)


def _unknown_logit_from_known_energy(knownness_energy, temperature):
    known_prob = _energy_to_prob(knownness_energy, temperature).clamp(min=1e-6, max=1 - 1e-6)
    unknown_prob = (1.0 - known_prob).clamp(min=1e-6, max=1 - 1e-6)
    return torch.log(unknown_prob / (1.0 - unknown_prob))


def _compute_uod_fused_probabilities(class_logits, objectness_energy, knownness_energy, invalid_class_indices, objectness_temperature, knownness_temperature=None, unknown_scale=15.0):
    if knownness_temperature is None:
        knownness_temperature = objectness_temperature
    logits = class_logits.clone()
    logits[:, :, invalid_class_indices] = -10e10
    objectness_probability = _energy_to_prob(objectness_energy, objectness_temperature)
    class_probability = logits.sigmoid().clone()
    if len(invalid_class_indices) > 0:
        class_probability[:, :, invalid_class_indices] = 0.0
    if class_probability.shape[-1] > 0:
        class_probability[:, :, -1] = 0.0
    if knownness_energy is None:
        knownness_probability = torch.ones_like(objectness_probability)
    else:
        knownness_probability = _energy_to_prob(knownness_energy, knownness_temperature)
    unknown_probability = (1.0 - knownness_probability).clamp(min=0.0, max=1.0)
    known_scores = objectness_probability.unsqueeze(-1) * knownness_probability.unsqueeze(-1) * class_probability
    if class_probability.shape[-1] > 1:
        max_known_class_probability = class_probability[:, :, :-1].max(dim=-1).values
    elif class_probability.shape[-1] > 0:
        max_known_class_probability = class_probability.squeeze(-1)
    else:
        max_known_class_probability = torch.zeros_like(objectness_probability)
    unknown_score = objectness_probability * unknown_probability * unknown_scale
    fused_scores = known_scores.clone()
    if fused_scores.shape[-1] > 0:
        fused_scores[:, :, -1] = unknown_score
    return {
        'objectness_probability': objectness_probability,
        'class_probability': class_probability,
        'knownness_probability': knownness_probability,
        'unknown_probability': unknown_probability,
        'max_known_class_probability': max_known_class_probability,
        'known_scores': known_scores,
        'unknown_score': unknown_score,
        'fused_scores': fused_scores,
    }


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


class _ZeroBBoxDelta(nn.Module):
    def forward(self, features):
        return features.new_zeros(*features.shape[:-1], 4)


class DeformableDETRUOD(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False, two_stage=False, args=None):
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
        self.objectness_temperature = float(getattr(args, 'obj_temp', 1.0)) / float(hidden_dim)
        self.knownness_temperature = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / float(hidden_dim)
        self.unknown_score_scale = float(getattr(args, 'uod_postprocess_unknown_scale', 15.0))

        # Keep original attribute names for checkpoint compatibility.
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.prob_obj_head = ProbObjectnessHead(hidden_dim)
        self.known_energy_head = ProbObjectnessHead(hidden_dim)

        if self.enable_odqe:
            self.context_attn = MSDeformAttn(hidden_dim, num_feature_levels, n_heads=8, n_points=4)
            self.gate_mlp = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2)
            self.ffn_obj = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.ffn_known = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
            self.ffn_cls = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        else:
            self.obj_proj = nn.Linear(hidden_dim, hidden_dim)
            self.known_proj = nn.Linear(hidden_dim, hidden_dim)
            self.cls_proj = nn.Linear(hidden_dim, hidden_dim)

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

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for input_projection in self.input_proj:
            nn.init.xavier_uniform_(input_projection[0].weight, gain=1)
            nn.init.constant_(input_projection[0].bias, 0)

        num_decoder_prediction_layers = transformer.decoder.num_layers
        num_total_prediction_layers = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_total_prediction_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, num_total_prediction_layers)
            self.prob_obj_head = _get_clones(self.prob_obj_head, num_decoder_prediction_layers)
            self.known_energy_head = _get_clones(self.known_energy_head, num_decoder_prediction_layers)
            if self.enable_odqe:
                self.context_attn = _get_clones(self.context_attn, num_decoder_prediction_layers)
                self.gate_mlp = _get_clones(self.gate_mlp, num_decoder_prediction_layers)
                self.ffn_obj = _get_clones(self.ffn_obj, num_decoder_prediction_layers)
                self.ffn_known = _get_clones(self.ffn_known, num_decoder_prediction_layers)
                self.ffn_cls = _get_clones(self.ffn_cls, num_decoder_prediction_layers)
            else:
                self.obj_proj = _get_clones(self.obj_proj, num_decoder_prediction_layers)
                self.known_proj = _get_clones(self.known_proj, num_decoder_prediction_layers)
                self.cls_proj = _get_clones(self.cls_proj, num_decoder_prediction_layers)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_total_prediction_layers)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_total_prediction_layers)])
            self.prob_obj_head = nn.ModuleList([self.prob_obj_head for _ in range(num_decoder_prediction_layers)])
            self.known_energy_head = nn.ModuleList([self.known_energy_head for _ in range(num_decoder_prediction_layers)])
            if self.enable_odqe:
                self.context_attn = nn.ModuleList([self.context_attn for _ in range(num_decoder_prediction_layers)])
                self.gate_mlp = nn.ModuleList([self.gate_mlp for _ in range(num_decoder_prediction_layers)])
                self.ffn_obj = nn.ModuleList([self.ffn_obj for _ in range(num_decoder_prediction_layers)])
                self.ffn_known = nn.ModuleList([self.ffn_known for _ in range(num_decoder_prediction_layers)])
                self.ffn_cls = nn.ModuleList([self.ffn_cls for _ in range(num_decoder_prediction_layers)])
            else:
                self.obj_proj = nn.ModuleList([self.obj_proj for _ in range(num_decoder_prediction_layers)])
                self.known_proj = nn.ModuleList([self.known_proj for _ in range(num_decoder_prediction_layers)])
                self.cls_proj = nn.ModuleList([self.cls_proj for _ in range(num_decoder_prediction_layers)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_head in self.bbox_embed:
                nn.init.constant_(box_head.layers[-1].bias.data[2:], 0.0)
            if not with_box_refine:
                proposal_box_heads = nn.ModuleList([_ZeroBBoxDelta() for _ in range(num_total_prediction_layers - 1)] + [self.bbox_embed[-1]])
                self.transformer.decoder.bbox_embed = proposal_box_heads

        if self.enable_odqe:
            decay_min = float(getattr(args, 'uod_odqe_decay_min', 0.1))
            decay_power = float(getattr(args, 'uod_odqe_decay_power', 1.0))
            if num_decoder_prediction_layers == 1:
                layer_decay = torch.ones(1)
            else:
                positions = torch.linspace(0.0, 1.0, steps=num_decoder_prediction_layers)
                layer_decay = 1.0 - (1.0 - decay_min) * positions.pow(decay_power)
            self.register_buffer('odqe_layer_decay', layer_decay)
        else:
            self.register_buffer('odqe_layer_decay', torch.ones(num_decoder_prediction_layers))

    @property
    def classification_head(self):
        return self.class_embed

    @property
    def box_regression_head(self):
        return self.bbox_embed

    @property
    def objectness_energy_head(self):
        return self.prob_obj_head

    @property
    def knownness_energy_head(self):
        return self.known_energy_head

    def _context_reference_input(self, reference_points, valid_ratios):
        reference_sigmoid = reference_points.sigmoid()
        if reference_sigmoid.shape[-1] == 4:
            return reference_sigmoid[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        return reference_sigmoid[:, :, None] * valid_ratios[:, None]

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

        query_embeddings = self.query_embed.weight if not self.two_stage else None
        decoder_hidden_states, initial_reference_points, intermediate_reference_points, encoder_class_logits, encoder_box_coordinates, encoder_info = self.transformer(
            encoder_inputs,
            encoder_masks,
            positional_embeddings,
            query_embeddings,
        )
        encoder_memory = encoder_info['memory']
        spatial_shapes = encoder_info['spatial_shapes']
        level_start_index = encoder_info['level_start_index']
        valid_ratios = encoder_info['valid_ratios']
        padding_mask = encoder_info['padding_mask']

        per_layer_class_logits = []
        per_layer_box_coordinates = []
        per_layer_objectness_energy = []
        per_layer_knownness_energy = []
        per_layer_objectness_features = []
        per_layer_knownness_features = []
        per_layer_classification_features = []
        per_layer_gate_means = []

        for layer_index in range(decoder_hidden_states.shape[0]):
            reference = initial_reference_points if layer_index == 0 else intermediate_reference_points[layer_index - 1]
            reference = inverse_sigmoid(reference)
            layer_hidden = decoder_hidden_states[layer_index]
            if self.enable_odqe:
                context_reference_input = self._context_reference_input(reference, valid_ratios)
                context_features = self.context_attn[layer_index](layer_hidden, context_reference_input, encoder_memory, spatial_shapes, level_start_index, padding_mask)
                gate = torch.sigmoid(self.gate_mlp[layer_index](torch.cat([layer_hidden, context_features], dim=-1)))
                layer_decay = self.odqe_layer_decay[min(layer_index, len(self.odqe_layer_decay) - 1)].to(layer_hidden.dtype)
                enhanced_hidden = layer_hidden + layer_decay * gate * context_features
                per_layer_gate_means.append((layer_decay * gate).mean())
                objectness_features = self.ffn_obj[layer_index](enhanced_hidden)
                knownness_features = self.ffn_known[layer_index](enhanced_hidden)
                classification_features = self.ffn_cls[layer_index](enhanced_hidden)
                box_features = enhanced_hidden
            else:
                objectness_features = self.obj_proj[layer_index](layer_hidden)
                knownness_features = self.known_proj[layer_index](layer_hidden)
                classification_features = self.cls_proj[layer_index](layer_hidden)
                box_features = layer_hidden

            class_logits = self.class_embed[layer_index](classification_features)
            objectness_energy = self.prob_obj_head[layer_index](objectness_features)
            knownness_energy = self.known_energy_head[layer_index](knownness_features)
            box_delta = self.bbox_embed[layer_index](box_features)
            if reference.shape[-1] == 4:
                box_delta += reference
            else:
                box_delta[..., :2] += reference
            box_coordinates = box_delta.sigmoid()

            per_layer_class_logits.append(class_logits)
            per_layer_box_coordinates.append(box_coordinates)
            per_layer_objectness_energy.append(objectness_energy)
            per_layer_knownness_energy.append(knownness_energy)
            per_layer_objectness_features.append(objectness_features)
            per_layer_knownness_features.append(knownness_features)
            per_layer_classification_features.append(classification_features)

        per_layer_class_logits = torch.stack(per_layer_class_logits)
        per_layer_box_coordinates = torch.stack(per_layer_box_coordinates)
        per_layer_objectness_energy = torch.stack(per_layer_objectness_energy)
        per_layer_knownness_energy = torch.stack(per_layer_knownness_energy)
        gate_mean_per_layer = torch.stack(per_layer_gate_means) if per_layer_gate_means else None

        unknown_logit = _unknown_logit_from_known_energy(per_layer_knownness_energy[-1], self.knownness_temperature)
        output = {
            'pred_class_logits': per_layer_class_logits[-1],
            'pred_boxes': per_layer_box_coordinates[-1],
            'pred_objectness_energy': per_layer_objectness_energy[-1],
            'pred_knownness_energy': per_layer_knownness_energy[-1],
            'pred_unknown_logit': unknown_logit,
            'decoder_objectness_features': per_layer_objectness_features[-1],
            'decoder_knownness_features': per_layer_knownness_features[-1],
            'decoder_classification_features': per_layer_classification_features[-1],
            # Backward-compatible aliases
            'pred_logits': per_layer_class_logits[-1],
            'pred_obj': per_layer_objectness_energy[-1],
            'pred_known': per_layer_knownness_energy[-1],
            'pred_unk': unknown_logit,
            'proj_obj': per_layer_objectness_features[-1],
            'proj_known': per_layer_knownness_features[-1],
            'proj_unk': per_layer_knownness_features[-1],
            'proj_cls': per_layer_classification_features[-1],
        }
        if gate_mean_per_layer is not None:
            output['odqe_gate_mean_per_layer'] = gate_mean_per_layer
            output['odqe_gate_mean'] = gate_mean_per_layer.mean()
            output['gate_mean_per_layer'] = gate_mean_per_layer
            output['gate_mean'] = gate_mean_per_layer.mean()
        if return_vis_debug:
            class_prob_layers = per_layer_class_logits.detach().sigmoid()
            if class_prob_layers.shape[-1] > 0:
                class_prob_layers[..., -1] = 0.0
            output['vis_debug'] = {
                'layer_objectness_probability': _energy_to_prob(per_layer_objectness_energy.detach(), self.objectness_temperature),
                'layer_knownness_probability': _energy_to_prob(per_layer_knownness_energy.detach(), self.knownness_temperature),
                'layer_unknown_probability': (1.0 - _energy_to_prob(per_layer_knownness_energy.detach(), self.knownness_temperature)).clamp(min=0.0, max=1.0),
                'layer_max_known_class_probability': class_prob_layers[..., :-1].max(dim=-1).values if class_prob_layers.shape[-1] > 1 else class_prob_layers.squeeze(-1),
                'odqe_gate_mean_per_layer': gate_mean_per_layer.detach() if gate_mean_per_layer is not None else None,
            }
        if self.aux_loss:
            output['aux_outputs'] = self._build_auxiliary_outputs(
                per_layer_class_logits,
                per_layer_box_coordinates,
                per_layer_objectness_energy,
                per_layer_knownness_energy,
                per_layer_objectness_features,
                per_layer_knownness_features,
                per_layer_classification_features,
            )
        if self.two_stage:
            output['enc_outputs'] = {
                'pred_class_logits': encoder_class_logits,
                'pred_boxes': encoder_box_coordinates.sigmoid(),
                'pred_logits': encoder_class_logits,
            }
        return output

    @torch.jit.unused
    def _build_auxiliary_outputs(self, per_layer_class_logits, per_layer_box_coordinates, per_layer_objectness_energy, per_layer_knownness_energy, per_layer_objectness_features, per_layer_knownness_features, per_layer_classification_features):
        aux_outputs = []
        for class_logits, objectness_energy, box_coordinates, knownness_energy, objectness_features, knownness_features, classification_features in zip(
            per_layer_class_logits[:-1],
            per_layer_objectness_energy[:-1],
            per_layer_box_coordinates[:-1],
            per_layer_knownness_energy[:-1],
            per_layer_objectness_features[:-1],
            per_layer_knownness_features[:-1],
            per_layer_classification_features[:-1],
        ):
            unknown_logit = _unknown_logit_from_known_energy(knownness_energy, self.knownness_temperature)
            aux_outputs.append({
                'pred_class_logits': class_logits,
                'pred_boxes': box_coordinates,
                'pred_objectness_energy': objectness_energy,
                'pred_knownness_energy': knownness_energy,
                'pred_unknown_logit': unknown_logit,
                'decoder_objectness_features': objectness_features,
                'decoder_knownness_features': knownness_features,
                'decoder_classification_features': classification_features,
                'pred_logits': class_logits,
                'pred_obj': objectness_energy,
                'pred_known': knownness_energy,
                'pred_unk': unknown_logit,
                'proj_obj': objectness_features,
                'proj_known': knownness_features,
                'proj_unk': knownness_features,
                'proj_cls': classification_features,
            })
        return aux_outputs


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, invalid_cls_logits, hidden_dim, focal_alpha=0.25, empty_weight=0.1, args=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.empty_weight = empty_weight
        self.invalid_cls_logits = invalid_cls_logits
        self.hidden_dim = hidden_dim
        self.min_objectness_energy = -hidden_dim * math.log(0.9)
        self.args = args

        self.enable_unknown = bool(getattr(args, 'uod_enable_unknown', False))
        self.enable_pseudo = bool(getattr(args, 'uod_enable_pseudo', False))
        self.enable_batch_dynamic = bool(getattr(args, 'uod_enable_batch_dynamic', False))
        self.enable_decorr = bool(getattr(args, 'uod_enable_decorr', False))
        self.enable_cls_soft_attn = bool(getattr(args, 'uod_enable_cls_soft_attn', False))
        self.objectness_temperature = float(getattr(args, 'obj_temp', 1.0)) / float(hidden_dim)
        self.knownness_temperature = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / float(hidden_dim)
        self.num_aux_layers = max(int(getattr(args, 'dec_layers', 6)) - 1, 0)

        self.uod_start_epoch = int(getattr(args, 'uod_start_epoch', 8))
        self.uod_neg_warmup_epochs = int(getattr(args, 'uod_neg_warmup_epochs', 3))
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
        self.uod_cls_soft_attn_alpha = float(getattr(args, 'uod_cls_soft_attn_alpha', 0.5))
        self.uod_cls_soft_attn_min = float(getattr(args, 'uod_cls_soft_attn_min', 0.25))
        self.uod_neg_max_pseudo_iou = float(getattr(args, 'uod_neg_max_pseudo_iou', 0.3))
        self.uod_neg_known_max = float(getattr(args, 'uod_neg_known_max', 0.7))
        self.uod_neg_unk_max = float(getattr(args, 'uod_neg_unk_max', 0.1))

    def _get_output(self, outputs, *keys):
        for key in keys:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
        raise KeyError(f'Missing outputs for keys: {keys}')

    def _compute_fused_probabilities(self, outputs):
        return _compute_uod_fused_probabilities(
            self._get_output(outputs, 'pred_class_logits', 'pred_logits'),
            self._get_output(outputs, 'pred_objectness_energy', 'pred_obj'),
            outputs.get('pred_knownness_energy', outputs.get('pred_known', None)),
            self.invalid_cls_logits,
            self.objectness_temperature,
            self.knownness_temperature,
            float(getattr(self.args, 'uod_postprocess_unknown_scale', 15.0)),
        )

    def _aux_stage(self, layer_index):
        if self.num_aux_layers <= 0:
            return 'high'
        low_end = max(1, self.num_aux_layers // 3)
        mid_end = max(low_end + 1, (2 * self.num_aux_layers + 2) // 3)
        if layer_index < low_end:
            return 'low'
        if layer_index < mid_end:
            return 'mid'
        return 'high'

    def _aux_losses_for_layer(self, layer_index):
        stage = self._aux_stage(layer_index)
        losses = ['labels', 'boxes', 'cardinality', 'obj_likelihood']
        if self.enable_unknown:
            losses.append('unk_known')
        if self.enable_pseudo:
            losses.append('obj_pseudo')
        if stage in ['mid', 'high']:
            if self.enable_pseudo and self.enable_unknown:
                losses.append('unk_pseudo')
            if self.enable_pseudo:
                losses.append('boxes_pseudo_cons')
        if stage == 'high' and self.enable_decorr:
            losses.append('decorr')
        return losses

    def _sigmoid_focal_loss_query_weight(self, inputs, targets, num_boxes, query_weights=None, alpha: float = 0.25, gamma: float = 2.0):
        probabilities = inputs.sigmoid()
        class_weight = torch.ones(inputs.shape[-1], dtype=probabilities.dtype, layout=probabilities.layout, device=probabilities.device)
        class_weight[-1] = self.empty_weight
        binary_cross_entropy = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=class_weight)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = binary_cross_entropy * ((1.0 - pt) ** gamma)
        if alpha >= 0:
            alpha_term = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_term * loss
        if query_weights is not None:
            loss = loss * query_weights.unsqueeze(-1)
        return loss.mean(1).sum() / num_boxes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, selected_pseudo_positive_query_indices=None, selected_pseudo_positive_confidences=None, classification_ignored_query_indices=None, **kwargs):
        class_logits = self._get_output(outputs, 'pred_class_logits', 'pred_logits').clone()
        class_logits[:, :, self.invalid_cls_logits] = -10e10
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        matched_target_classes = torch.cat([target['labels'][target_indices] for target, (_, target_indices) in zip(targets, indices)]) if len(indices) > 0 else torch.empty(0, dtype=torch.long, device=class_logits.device)
        num_classes = class_logits.shape[-1]
        target_classes = torch.full(class_logits.shape[:2], self.num_classes, dtype=torch.int64, device=class_logits.device)
        if len(matched_target_classes) > 0:
            target_classes[(matched_batch_indices, matched_query_indices)] = matched_target_classes
        target_onehot = torch.zeros([class_logits.shape[0], class_logits.shape[1], num_classes + 1], dtype=class_logits.dtype, layout=class_logits.layout, device=class_logits.device)
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        query_weights = None
        needs_query_weights = (classification_ignored_query_indices is not None) or (self.enable_cls_soft_attn and selected_pseudo_positive_query_indices is not None)
        if needs_query_weights:
            query_weights = torch.ones(class_logits.shape[:2], dtype=class_logits.dtype, device=class_logits.device)
            if classification_ignored_query_indices is not None:
                for batch_index, query_list in enumerate(classification_ignored_query_indices):
                    if len(query_list) > 0:
                        query_weights[batch_index, torch.as_tensor(query_list, dtype=torch.long, device=class_logits.device)] = 0.0
            if self.enable_cls_soft_attn and selected_pseudo_positive_query_indices is not None:
                for batch_index, query_list in enumerate(selected_pseudo_positive_query_indices):
                    if len(query_list) == 0:
                        continue
                    query_tensor = torch.as_tensor(query_list, dtype=torch.long, device=class_logits.device)
                    if selected_pseudo_positive_confidences is not None and len(selected_pseudo_positive_confidences[batch_index]) == len(query_list):
                        confidence = torch.as_tensor(selected_pseudo_positive_confidences[batch_index], dtype=class_logits.dtype, device=class_logits.device)
                    else:
                        confidence = torch.ones(len(query_list), dtype=class_logits.dtype, device=class_logits.device)
                    attenuation = 1.0 - self.uod_cls_soft_attn_alpha * confidence
                    attenuation = torch.clamp(attenuation, min=self.uod_cls_soft_attn_min, max=1.0)
                    attenuation[confidence >= 0.8] = 0.0
                    query_weights[batch_index, query_tensor] = torch.minimum(query_weights[batch_index, query_tensor], attenuation)

        if query_weights is None:
            classification_loss = sigmoid_focal_loss(class_logits, target_onehot, num_boxes, alpha=self.focal_alpha, num_classes=self.num_classes, empty_weight=self.empty_weight) * class_logits.shape[1]
        else:
            classification_loss = self._sigmoid_focal_loss_query_weight(class_logits, target_onehot, num_boxes, query_weights=query_weights, alpha=self.focal_alpha) * class_logits.shape[1]
        losses = {'loss_ce': classification_loss}
        if log and len(matched_target_classes) > 0:
            losses['class_error'] = 100 - accuracy(class_logits[(matched_batch_indices, matched_query_indices)], matched_target_classes)[0]
        elif log:
            losses['class_error'] = class_logits.sum() * 0.0
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        class_logits = self._get_output(outputs, 'pred_class_logits', 'pred_logits')
        target_lengths = torch.as_tensor([len(target['labels']) for target in targets], device=class_logits.device)
        fused = self._compute_fused_probabilities(outputs)
        if fused['known_scores'].shape[-1] > 1:
            known_max = fused['known_scores'][:, :, :-1].max(dim=-1).values
        else:
            known_max = fused['known_scores'].max(dim=-1).values
        unknown_score = fused['unknown_score']
        predicted_count = ((known_max > 0.05) | (unknown_score > 0.05)).sum(1)
        return {'cardinality_error': F.l1_loss(predicted_count.float(), target_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        if matched_batch_indices.numel() == 0:
            zero = outputs['pred_boxes'].sum() * 0.0
            return {'loss_bbox': zero, 'loss_giou': zero}
        src_boxes = outputs['pred_boxes'][(matched_batch_indices, matched_query_indices)]
        target_boxes = torch.cat([target['boxes'][target_indices] for target, (_, target_indices) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {'loss_bbox': loss_bbox.sum() / num_boxes, 'loss_giou': loss_giou.sum() / num_boxes}

    def loss_obj_likelihood(self, outputs, targets, indices, num_boxes):
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        objectness_energy = self._get_output(outputs, 'pred_objectness_energy', 'pred_obj')[(matched_batch_indices, matched_query_indices)]
        if objectness_energy.numel() == 0:
            return {'loss_obj_ll': self._get_output(outputs, 'pred_objectness_energy', 'pred_obj').sum() * 0.0}
        return {'loss_obj_ll': torch.clamp(objectness_energy, min=self.min_objectness_energy).sum() / num_boxes}

    def loss_unk_known(self, outputs, targets, indices, num_boxes, **kwargs):
        if not self.enable_unknown:
            return {'loss_unk_known': self._get_output(outputs, 'pred_class_logits', 'pred_logits').sum() * 0.0}
        matched_batch_indices, matched_query_indices = self._get_src_permutation_idx(indices)
        knownness_energy = self._get_output(outputs, 'pred_knownness_energy', 'pred_known')[(matched_batch_indices, matched_query_indices)]
        if knownness_energy.numel() == 0:
            return {'loss_unk_known': self._get_output(outputs, 'pred_knownness_energy', 'pred_known').sum() * 0.0}
        return {'loss_unk_known': torch.clamp(knownness_energy, min=self.min_objectness_energy).sum() / num_boxes}

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
        width = box_cxcywh[2].item()
        height = box_cxcywh[3].item()
        area = width * height
        shortest_side = min(width, height)
        aspect_ratio = max(width / max(height, 1e-6), height / max(width, 1e-6))
        return area >= self.uod_min_area and shortest_side >= self.uod_min_side and aspect_ratio <= self.uod_max_aspect_ratio

    def _deduplicate_pos_candidates(self, predicted_boxes_for_image, candidate_items, iou_threshold):
        if len(candidate_items) <= 1 or iou_threshold is None or iou_threshold <= 0:
            return candidate_items
        candidate_items = sorted(candidate_items, key=lambda item: (-item[2], item[3], item[4]))
        predicted_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(predicted_boxes_for_image)
        kept_items = []
        kept_query_indices = []
        for item in candidate_items:
            query_index = item[1]
            if not kept_query_indices:
                kept_items.append(item)
                kept_query_indices.append(query_index)
                continue
            ious = box_ops.box_iou(predicted_boxes_xyxy[query_index].unsqueeze(0), predicted_boxes_xyxy[torch.as_tensor(kept_query_indices, dtype=torch.long, device=predicted_boxes_xyxy.device)])[0]
            if torch.any(ious >= iou_threshold):
                continue
            kept_items.append(item)
            kept_query_indices.append(query_index)
        return kept_items

    def _filter_negatives_near_selected_pos(self, predicted_boxes_xyxy, selected_pseudo_positive_queries, candidate_queries):
        if len(selected_pseudo_positive_queries) == 0 or len(candidate_queries) == 0 or self.uod_neg_max_pseudo_iou <= 0:
            return candidate_queries
        selected_query_tensor = torch.as_tensor(selected_pseudo_positive_queries, dtype=torch.long, device=predicted_boxes_xyxy.device)
        kept_queries = []
        for query_index in candidate_queries:
            query_tensor = torch.as_tensor([query_index], dtype=torch.long, device=predicted_boxes_xyxy.device)
            ious = box_ops.box_iou(predicted_boxes_xyxy[query_tensor], predicted_boxes_xyxy[selected_query_tensor])[0]
            if torch.any(ious > self.uod_neg_max_pseudo_iou):
                continue
            kept_queries.append(query_index)
        return kept_queries

    def _mine_uod_pseudo(self, outputs, targets, indices, epoch, return_debug=False):
        batch_size = len(targets)
        selected_pseudo_positive_queries = [[] for _ in range(batch_size)]
        selected_reliable_background_queries = [[] for _ in range(batch_size)]
        selected_pseudo_positive_confidences = [[] for _ in range(batch_size)]
        selected_pseudo_positive_boxes = [outputs['pred_boxes'].new_zeros((0, 4)) for _ in range(batch_size)]
        classification_ignored_queries = [[] for _ in range(batch_size)]
        stats = {
            'num_selected_pseudo_positive_queries': 0.0,
            'num_selected_reliable_background_queries': 0.0,
            'num_classification_ignored_queries': 0.0,
            'num_unmatched_queries_after_filter': 0.0,
            'num_pseudo_positive_candidates': 0.0,
            'num_reliable_background_candidates': 0.0,
            'num_batch_selected_pseudo_positive_queries': 0.0,
            'pseudo_positive_threshold_sum': 0.0,
            'num_thresholds': 0.0,
        }
        debug_items = [] if return_debug else None

        if (not self.enable_pseudo) or epoch < self.uod_start_epoch:
            if return_debug:
                for batch_index in range(batch_size):
                    debug_items.append({
                        'after_gt_overlap_filter_boxes': [],
                        'after_geometry_filter_boxes': [],
                        'candidate_boxes_before_selection': [],
                        'candidate_score_texts': [],
                        'selected_pseudo_positive_boxes': [],
                        'selected_reliable_background_boxes': [],
                    })
                return selected_pseudo_positive_queries, selected_reliable_background_queries, selected_pseudo_positive_confidences, selected_pseudo_positive_boxes, classification_ignored_queries, stats, debug_items
            return selected_pseudo_positive_queries, selected_reliable_background_queries, selected_pseudo_positive_confidences, selected_pseudo_positive_boxes, classification_ignored_queries, stats

        objectness_energy = self._get_output(outputs, 'pred_objectness_energy', 'pred_obj').detach() / float(self.hidden_dim)
        predicted_boxes = outputs['pred_boxes'].detach()
        fused = self._compute_fused_probabilities(outputs)
        objectness_probability = fused['objectness_probability'].detach()
        unknown_probability = fused['unknown_probability'].detach()
        unknown_score = fused['unknown_score'].detach()
        max_known_class_probability = fused['max_known_class_probability'].detach()
        num_queries = objectness_energy.shape[1]

        all_candidate_items = []
        per_image_candidate_items = []
        per_image_cache = []
        per_image_debug = []

        for batch_index, (matched_query_indices, _) in enumerate(indices):
            matched_query_set = set(matched_query_indices.tolist())
            unmatched_queries = [query_index for query_index in range(num_queries) if query_index not in matched_query_set]
            if len(matched_query_indices) > 0:
                matched_energy = objectness_energy[batch_index, matched_query_indices]
                mean_energy = matched_energy.mean().item()
                std_energy = matched_energy.std().item() if len(matched_query_indices) > 1 else 0.0
                positive_threshold = max(mean_energy + 3.0 * std_energy, self.uod_min_pos_thresh)
            else:
                positive_threshold = self.uod_min_pos_thresh
            stats['pseudo_positive_threshold_sum'] += positive_threshold
            stats['num_thresholds'] += 1.0

            predicted_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(predicted_boxes[batch_index])
            gt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(targets[batch_index]['boxes'])
            valid_queries = unmatched_queries
            iou_map = {query_index: 0.0 for query_index in unmatched_queries}

            after_gt_overlap_filter = unmatched_queries
            if gt_boxes_xyxy.numel() > 0 and len(unmatched_queries) > 0:
                candidate_boxes = predicted_boxes_xyxy[unmatched_queries]
                ious = box_ops.box_iou(candidate_boxes, gt_boxes_xyxy)[0]
                iofs = self._pairwise_iof(candidate_boxes, gt_boxes_xyxy)
                max_iou = ious.max(dim=1)[0]
                max_iof = iofs.max(dim=1)[0]
                valid_queries = []
                for item_index, query_index in enumerate(unmatched_queries):
                    iou_map[query_index] = max_iou[item_index].item()
                    if max_iou[item_index].item() < self.uod_max_iou and max_iof[item_index].item() < self.uod_max_iof:
                        valid_queries.append(query_index)
                after_gt_overlap_filter = list(valid_queries)

            after_geometry_filter = [query_index for query_index in valid_queries if self._is_valid_geometry(predicted_boxes[batch_index, query_index])]
            valid_queries = list(after_geometry_filter)
            stats['num_unmatched_queries_after_filter'] += float(len(valid_queries))

            candidate_items = []
            candidate_before_unknown_filter = []
            candidate_before_known_reject = []
            for query_index in valid_queries:
                energy_value = objectness_energy[batch_index, query_index].item()
                known_max_value = max_known_class_probability[batch_index, query_index].item()
                unknown_probability_value = unknown_probability[batch_index, query_index].item()
                unknown_score_value = unknown_score[batch_index, query_index].item()
                if unknown_probability_value >= self.uod_pos_unk_min:
                    candidate_before_unknown_filter.append(query_index)
                else:
                    continue
                if energy_value < positive_threshold and known_max_value < self.uod_known_reject_thresh:
                    candidate_before_known_reject.append(query_index)
                    energy_rel = max(0.0, min(1.0, (positive_threshold - energy_value) / max(positive_threshold, 1e-6)))
                    known_rel = max(0.0, min(1.0, (self.uod_known_reject_thresh - known_max_value) / max(self.uod_known_reject_thresh, 1e-6)))
                    iou_rel = 1.0 - max(0.0, min(1.0, iou_map[query_index] / max(self.uod_max_iou, 1e-6)))
                    unknown_rel = max(0.0, min(1.0, unknown_probability_value))
                    confidence = (energy_rel * known_rel * iou_rel * max(unknown_rel, 1e-6)) ** (1.0 / 4.0)
                    candidate_items.append((batch_index, query_index, confidence, energy_value, known_max_value, unknown_probability_value, unknown_score_value))

            candidate_items = self._deduplicate_pos_candidates(predicted_boxes[batch_index], candidate_items, self.uod_candidate_nms_iou)
            all_candidate_items.extend(candidate_items)
            per_image_candidate_items.append(candidate_items)
            per_image_cache.append({'valid_queries': valid_queries, 'predicted_boxes_xyxy': predicted_boxes_xyxy})
            stats['num_pseudo_positive_candidates'] += float(len(candidate_items))

            if return_debug:
                per_image_debug.append({
                    'after_gt_overlap_filter_boxes': predicted_boxes_xyxy[torch.as_tensor(after_gt_overlap_filter, dtype=torch.long, device=predicted_boxes_xyxy.device)].detach().cpu().tolist() if len(after_gt_overlap_filter) > 0 else [],
                    'after_geometry_filter_boxes': predicted_boxes_xyxy[torch.as_tensor(after_geometry_filter, dtype=torch.long, device=predicted_boxes_xyxy.device)].detach().cpu().tolist() if len(after_geometry_filter) > 0 else [],
                    'candidate_boxes_before_selection': predicted_boxes_xyxy[torch.as_tensor([item[1] for item in candidate_items], dtype=torch.long, device=predicted_boxes_xyxy.device)].detach().cpu().tolist() if len(candidate_items) > 0 else [],
                    'candidate_score_texts': [f'conf={item[2]:.2f} obj={1.0 - item[3]:.2f} unk={item[5]:.2f}' for item in candidate_items],
                    'selected_pseudo_positive_boxes': [],
                    'selected_reliable_background_boxes': [],
                })

        if self.enable_batch_dynamic:
            all_candidate_items.sort(key=lambda item: (-item[2], -item[6], -item[5], item[3], item[4]))
            topk = min(self.uod_batch_topk_max, max(1, int(math.ceil(self.uod_batch_topk_ratio * max(len(all_candidate_items), 1)))))
            per_image_counts = [0 for _ in range(batch_size)]
            selected_items = []
            for item in all_candidate_items:
                batch_index, query_index, confidence, _, _, _, _ = item
                if len(selected_items) >= topk:
                    break
                if self.uod_pos_per_img_cap > 0 and per_image_counts[batch_index] >= self.uod_pos_per_img_cap:
                    continue
                selected_items.append(item)
                per_image_counts[batch_index] += 1
            for batch_index, query_index, confidence, _, _, _, _ in selected_items:
                selected_pseudo_positive_queries[batch_index].append(query_index)
                selected_pseudo_positive_confidences[batch_index].append(float(max(0.2, min(1.0, confidence))))
            stats['num_batch_selected_pseudo_positive_queries'] = float(len(selected_items))
        else:
            for batch_index, candidate_items in enumerate(per_image_candidate_items):
                candidate_items.sort(key=lambda item: (-item[2], -item[6], -item[5], item[3], item[4]))
                candidate_items = candidate_items[:self.uod_pos_per_img_cap]
                selected_pseudo_positive_queries[batch_index] = [item[1] for item in candidate_items]
                selected_pseudo_positive_confidences[batch_index] = [float(max(0.2, min(1.0, item[2]))) for item in candidate_items]
            stats['num_batch_selected_pseudo_positive_queries'] = float(sum(len(query_list) for query_list in selected_pseudo_positive_queries))

        for batch_index in range(batch_size):
            if len(selected_pseudo_positive_queries[batch_index]) > 0:
                query_tensor = torch.as_tensor(selected_pseudo_positive_queries[batch_index], dtype=torch.long, device=predicted_boxes.device)
                selected_pseudo_positive_boxes[batch_index] = predicted_boxes[batch_index, query_tensor].detach()
            if return_debug:
                predicted_boxes_xyxy = per_image_cache[batch_index]['predicted_boxes_xyxy']
                if len(selected_pseudo_positive_queries[batch_index]) > 0:
                    query_tensor = torch.as_tensor(selected_pseudo_positive_queries[batch_index], dtype=torch.long, device=predicted_boxes_xyxy.device)
                    per_image_debug[batch_index]['selected_pseudo_positive_boxes'] = predicted_boxes_xyxy[query_tensor].detach().cpu().tolist()

        stats['num_selected_pseudo_positive_queries'] = float(sum(len(query_list) for query_list in selected_pseudo_positive_queries))

        if epoch >= self.uod_start_epoch + self.uod_neg_warmup_epochs:
            for batch_index in range(batch_size):
                valid_queries = per_image_cache[batch_index]['valid_queries']
                predicted_boxes_xyxy = per_image_cache[batch_index]['predicted_boxes_xyxy']
                positive_queries = selected_pseudo_positive_queries[batch_index]
                remaining_queries = [query_index for query_index in valid_queries if query_index not in set(positive_queries)]
                remaining_queries = self._filter_negatives_near_selected_pos(predicted_boxes_xyxy, positive_queries, remaining_queries)
                reliable_background_candidates = []
                for query_index in remaining_queries:
                    known_max_value = max_known_class_probability[batch_index, query_index].item()
                    objectness_prob_value = objectness_probability[batch_index, query_index].item()
                    unknown_prob_value = unknown_probability[batch_index, query_index].item()
                    objectness_energy_value = objectness_energy[batch_index, query_index].item()
                    if known_max_value > self.uod_neg_known_max:
                        continue
                    if unknown_prob_value > self.uod_neg_unk_max:
                        continue
                    reliable_background_candidates.append((query_index, objectness_prob_value, objectness_energy_value, known_max_value, unknown_prob_value))
                stats['num_reliable_background_candidates'] += float(len(reliable_background_candidates))
                reliable_background_candidates.sort(key=lambda item: (-item[1], item[2], item[3], item[4]))
                reliable_background_candidates = reliable_background_candidates[:self.uod_neg_per_img]
                selected_reliable_background_queries[batch_index] = [item[0] for item in reliable_background_candidates]
                stats['num_selected_reliable_background_queries'] += float(len(selected_reliable_background_queries[batch_index]))
                if return_debug and len(selected_reliable_background_queries[batch_index]) > 0:
                    query_tensor = torch.as_tensor(selected_reliable_background_queries[batch_index], dtype=torch.long, device=predicted_boxes_xyxy.device)
                    per_image_debug[batch_index]['selected_reliable_background_boxes'] = predicted_boxes_xyxy[query_tensor].detach().cpu().tolist()

        for batch_index in range(batch_size):
            positive_set = set(selected_pseudo_positive_queries[batch_index])
            negative_set = set(selected_reliable_background_queries[batch_index])
            ignored_queries = []
            valid_queries = per_image_cache[batch_index]['valid_queries']
            for query_index in valid_queries:
                if query_index in positive_set or query_index in negative_set:
                    continue
                if objectness_probability[batch_index, query_index].item() > 0.05 and unknown_probability[batch_index, query_index].item() >= self.uod_pos_unk_min and max_known_class_probability[batch_index, query_index].item() < self.uod_known_reject_thresh:
                    ignored_queries.append(query_index)
            classification_ignored_queries[batch_index] = ignored_queries
            stats['num_classification_ignored_queries'] += float(len(ignored_queries))

        if return_debug:
            return selected_pseudo_positive_queries, selected_reliable_background_queries, selected_pseudo_positive_confidences, selected_pseudo_positive_boxes, classification_ignored_queries, stats, per_image_debug
        return selected_pseudo_positive_queries, selected_reliable_background_queries, selected_pseudo_positive_confidences, selected_pseudo_positive_boxes, classification_ignored_queries, stats

    def generate_pseudo_mining_debug(self, outputs, targets, epoch=0):
        matching_outputs = {'pred_logits': self._get_output(outputs, 'pred_class_logits', 'pred_logits'), 'pred_boxes': outputs['pred_boxes']}
        indices = self.matcher(matching_outputs, targets)
        _, _, _, _, _, _, debug_items = self._mine_uod_pseudo(outputs, targets, indices, epoch, return_debug=True)
        return debug_items

    def loss_obj_pseudo(self, outputs, targets, indices, num_boxes, selected_pseudo_positive_query_indices=None, selected_pseudo_positive_confidences=None, **kwargs):
        objectness_energy = self._get_output(outputs, 'pred_objectness_energy', 'pred_obj') / float(self.hidden_dim)
        device = objectness_energy.device
        dtype = objectness_energy.dtype
        zero = objectness_energy.sum() * 0.0
        if selected_pseudo_positive_query_indices is None:
            return {'loss_obj_pseudo': zero}
        batch_indices = []
        query_indices = []
        weights = []
        for batch_index, query_list in enumerate(selected_pseudo_positive_query_indices):
            if len(query_list) == 0:
                continue
            batch_indices.append(torch.full((len(query_list),), batch_index, dtype=torch.long, device=device))
            query_indices.append(torch.as_tensor(query_list, dtype=torch.long, device=device))
            if selected_pseudo_positive_confidences is not None and len(selected_pseudo_positive_confidences[batch_index]) == len(query_list):
                weight_tensor = torch.as_tensor(selected_pseudo_positive_confidences[batch_index], dtype=dtype, device=device)
            else:
                weight_tensor = torch.ones(len(query_list), dtype=dtype, device=device)
            weights.append(torch.clamp(weight_tensor, min=0.2, max=1.0))
        if not batch_indices:
            return {'loss_obj_pseudo': zero}
        batch_indices = torch.cat(batch_indices)
        query_indices = torch.cat(query_indices)
        weights = torch.cat(weights)
        loss = (weights * objectness_energy[batch_indices, query_indices]).sum() / (weights.sum() + 1e-6)
        return {'loss_obj_pseudo': loss}

    def loss_obj_neg(self, outputs, targets, indices, num_boxes, selected_reliable_background_query_indices=None, **kwargs):
        objectness_energy = self._get_output(outputs, 'pred_objectness_energy', 'pred_obj') / float(self.hidden_dim)
        device = objectness_energy.device
        zero = objectness_energy.sum() * 0.0
        if selected_reliable_background_query_indices is None:
            return {'loss_obj_neg': zero}
        batch_indices = []
        query_indices = []
        for batch_index, query_list in enumerate(selected_reliable_background_query_indices):
            if len(query_list) == 0:
                continue
            batch_indices.append(torch.full((len(query_list),), batch_index, dtype=torch.long, device=device))
            query_indices.append(torch.as_tensor(query_list, dtype=torch.long, device=device))
        if not batch_indices:
            return {'loss_obj_neg': zero}
        batch_indices = torch.cat(batch_indices)
        query_indices = torch.cat(query_indices)
        negative_energy = objectness_energy[batch_indices, query_indices]
        return {'loss_obj_neg': F.relu(self.uod_neg_margin - negative_energy).mean()}

    def loss_unk_pseudo(self, outputs, targets, indices, num_boxes, selected_pseudo_positive_query_indices=None, selected_pseudo_positive_confidences=None, **kwargs):
        if not self.enable_unknown:
            return {'loss_unk_pseudo': self._get_output(outputs, 'pred_class_logits', 'pred_logits').sum() * 0.0}
        knownness_energy = self._get_output(outputs, 'pred_knownness_energy', 'pred_known')
        device = knownness_energy.device
        dtype = knownness_energy.dtype
        zero = knownness_energy.sum() * 0.0
        if selected_pseudo_positive_query_indices is None:
            return {'loss_unk_pseudo': zero}
        batch_indices = []
        query_indices = []
        weights = []
        for batch_index, query_list in enumerate(selected_pseudo_positive_query_indices):
            if len(query_list) == 0:
                continue
            batch_indices.append(torch.full((len(query_list),), batch_index, dtype=torch.long, device=device))
            query_indices.append(torch.as_tensor(query_list, dtype=torch.long, device=device))
            if selected_pseudo_positive_confidences is not None and len(selected_pseudo_positive_confidences[batch_index]) == len(query_list):
                weights.append(torch.as_tensor(selected_pseudo_positive_confidences[batch_index], dtype=dtype, device=device))
            else:
                weights.append(torch.ones(len(query_list), dtype=dtype, device=device))
        if not batch_indices:
            return {'loss_unk_pseudo': zero}
        batch_indices = torch.cat(batch_indices)
        query_indices = torch.cat(query_indices)
        weights = torch.cat(weights)
        knownness_probability = _energy_to_prob(knownness_energy[batch_indices, query_indices], self.knownness_temperature)
        return {'loss_unk_pseudo': (knownness_probability * weights).sum() / (weights.sum() + 1e-6)}

    def loss_boxes_pseudo_cons(self, outputs, targets, indices, num_boxes, selected_pseudo_positive_query_indices=None, selected_pseudo_positive_boxes=None, **kwargs):
        zero = outputs['pred_boxes'].sum() * 0.0
        if selected_pseudo_positive_query_indices is None or selected_pseudo_positive_boxes is None:
            return {'loss_bbox_pseudo_cons': zero, 'loss_giou_pseudo_cons': zero}
        batch_indices = []
        query_indices = []
        target_boxes = []
        device = outputs['pred_boxes'].device
        for batch_index, query_list in enumerate(selected_pseudo_positive_query_indices):
            if len(query_list) == 0 or batch_index >= len(selected_pseudo_positive_boxes):
                continue
            target_box_tensor = selected_pseudo_positive_boxes[batch_index]
            if target_box_tensor is None or target_box_tensor.numel() == 0:
                continue
            query_tensor = torch.as_tensor(query_list, dtype=torch.long, device=device)
            target_box_tensor = target_box_tensor.to(device=device, dtype=outputs['pred_boxes'].dtype)
            num_items = min(query_tensor.shape[0], target_box_tensor.shape[0])
            if num_items <= 0:
                continue
            batch_indices.append(torch.full((num_items,), batch_index, dtype=torch.long, device=device))
            query_indices.append(query_tensor[:num_items])
            target_boxes.append(target_box_tensor[:num_items])
        if not batch_indices:
            return {'loss_bbox_pseudo_cons': zero, 'loss_giou_pseudo_cons': zero}
        batch_indices = torch.cat(batch_indices)
        query_indices = torch.cat(query_indices)
        target_boxes = torch.cat(target_boxes, dim=0)
        predicted_boxes = outputs['pred_boxes'][batch_indices, query_indices]
        num_pseudo_boxes = max(float(target_boxes.shape[0]), 1.0)
        l1_loss = F.l1_loss(predicted_boxes, target_boxes, reduction='none').sum() / num_pseudo_boxes
        giou_loss = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(predicted_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {'loss_bbox_pseudo_cons': l1_loss, 'loss_giou_pseudo_cons': giou_loss.sum() / num_pseudo_boxes}

    def _corr_loss(self, x, y, mask=None):
        if x.numel() == 0 or y.numel() == 0:
            return x.sum() * 0.0
        if mask is not None:
            if mask.sum() < 2:
                return x.sum() * 0.0
            x = x[mask]
            y = y[mask]
        else:
            x = x.reshape(-1)
            y = y.reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        denominator = (x.std(unbiased=False) * y.std(unbiased=False) + 1e-6)
        correlation = (x * y).mean() / denominator
        return correlation.pow(2)

    def loss_decorr(self, outputs, targets, indices, num_boxes, **kwargs):
        if not self.enable_decorr:
            return {'loss_decorr': self._get_output(outputs, 'pred_class_logits', 'pred_logits').sum() * 0.0}
        fused = self._compute_fused_probabilities(outputs)
        max_known_class_probability = fused['max_known_class_probability']
        objectness_probability = fused['objectness_probability']
        unknown_probability = fused['unknown_probability']
        foreground_mask = objectness_probability > 0.05
        loss_cls_unknown = self._corr_loss(max_known_class_probability, unknown_probability, mask=foreground_mask)
        loss_cls_objectness = self._corr_loss(max_known_class_probability, objectness_probability, mask=foreground_mask)
        loss_objectness_unknown = self._corr_loss(objectness_probability, unknown_probability, mask=foreground_mask)
        return {'loss_decorr': (loss_cls_unknown + loss_cls_objectness + loss_objectness_unknown) / 3.0}

    def _get_src_permutation_idx(self, indices):
        if len(indices) == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        batch_indices = torch.cat([torch.full_like(source_indices, batch_index) for batch_index, (source_indices, _) in enumerate(indices)])
        source_indices = torch.cat([source_indices for (source_indices, _) in indices])
        return batch_indices, source_indices

    def get_loss(self, loss_name, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'obj_likelihood': self.loss_obj_likelihood,
            'unk_known': self.loss_unk_known,
            'obj_pseudo': self.loss_obj_pseudo,
            'obj_neg': self.loss_obj_neg,
            'unk_pseudo': self.loss_unk_pseudo,
            'boxes_pseudo_cons': self.loss_boxes_pseudo_cons,
            'decorr': self.loss_decorr,
        }
        assert loss_name in loss_map, f'Unsupported loss: {loss_name}'
        return loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0):
        matching_outputs = {'pred_logits': self._get_output(outputs, 'pred_class_logits', 'pred_logits'), 'pred_boxes': outputs['pred_boxes']}
        indices = self.matcher(matching_outputs, targets)
        num_boxes = sum(len(target['labels']) for target in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        selected_pseudo_positive_queries, selected_reliable_background_queries, selected_pseudo_positive_confidences, selected_pseudo_positive_boxes, classification_ignored_queries, mining_stats = self._mine_uod_pseudo(outputs, targets, indices, epoch)
        losses = {}
        for loss_name in self.losses:
            kwargs = {}
            if loss_name == 'labels':
                kwargs['classification_ignored_query_indices'] = classification_ignored_queries
                if self.enable_cls_soft_attn:
                    kwargs.update({
                        'selected_pseudo_positive_query_indices': selected_pseudo_positive_queries,
                        'selected_pseudo_positive_confidences': selected_pseudo_positive_confidences,
                    })
            if loss_name in ['obj_pseudo', 'unk_pseudo']:
                kwargs.update({
                    'selected_pseudo_positive_query_indices': selected_pseudo_positive_queries,
                    'selected_pseudo_positive_confidences': selected_pseudo_positive_confidences,
                })
            if loss_name == 'obj_neg':
                kwargs['selected_reliable_background_query_indices'] = selected_reliable_background_queries
            if loss_name == 'boxes_pseudo_cons':
                kwargs.update({
                    'selected_pseudo_positive_query_indices': selected_pseudo_positive_queries,
                    'selected_pseudo_positive_boxes': selected_pseudo_positive_boxes,
                })
            losses.update(self.get_loss(loss_name, outputs, targets, indices, num_boxes, **kwargs))

        if 'aux_outputs' in outputs:
            for layer_index, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher({'pred_logits': self._get_output(aux_outputs, 'pred_class_logits', 'pred_logits'), 'pred_boxes': aux_outputs['pred_boxes']}, targets)
                stage = self._aux_stage(layer_index)
                for loss_name in self._aux_losses_for_layer(layer_index):
                    kwargs = {}
                    if loss_name == 'labels':
                        kwargs['log'] = False
                        kwargs['classification_ignored_query_indices'] = classification_ignored_queries
                        if self.enable_cls_soft_attn and stage != 'low':
                            kwargs.update({
                                'selected_pseudo_positive_query_indices': selected_pseudo_positive_queries,
                                'selected_pseudo_positive_confidences': selected_pseudo_positive_confidences,
                            })
                    if loss_name in ['obj_pseudo', 'unk_pseudo']:
                        kwargs.update({
                            'selected_pseudo_positive_query_indices': selected_pseudo_positive_queries,
                            'selected_pseudo_positive_confidences': selected_pseudo_positive_confidences,
                        })
                    if loss_name == 'boxes_pseudo_cons':
                        kwargs.update({
                            'selected_pseudo_positive_query_indices': selected_pseudo_positive_queries,
                            'selected_pseudo_positive_boxes': selected_pseudo_positive_boxes,
                        })
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

        if self.enable_cls_soft_attn:
            attenuation_values = []
            for confidence_list in selected_pseudo_positive_confidences:
                if len(confidence_list) == 0:
                    continue
                confidence = torch.as_tensor(confidence_list, dtype=self._get_output(outputs, 'pred_class_logits', 'pred_logits').dtype, device=self._get_output(outputs, 'pred_class_logits', 'pred_logits').device)
                attenuation = 1.0 - self.uod_cls_soft_attn_alpha * confidence
                attenuation = torch.clamp(attenuation, min=self.uod_cls_soft_attn_min, max=1.0)
                attenuation_values.append(attenuation)
            if attenuation_values:
                attenuation_values = torch.cat(attenuation_values)
                mining_stats['mean_classification_attenuation'] = float(attenuation_values.mean().item())
                mining_stats['num_classification_attenuated_queries'] = float(attenuation_values.numel())
            else:
                mining_stats['mean_classification_attenuation'] = 1.0
                mining_stats['num_classification_attenuated_queries'] = 0.0

        device = self._get_output(outputs, 'pred_class_logits', 'pred_logits').device
        odqe_gate_mean = outputs.get('odqe_gate_mean', outputs.get('gate_mean', None))
        losses.update({
            'num_selected_pseudo_positive_queries': torch.tensor(float(mining_stats.get('num_selected_pseudo_positive_queries', 0.0)), device=device),
            'num_selected_reliable_background_queries': torch.tensor(float(mining_stats.get('num_selected_reliable_background_queries', 0.0)), device=device),
            'num_classification_ignored_queries': torch.tensor(float(mining_stats.get('num_classification_ignored_queries', 0.0)), device=device),
            'num_unmatched_queries_after_filter': torch.tensor(float(mining_stats.get('num_unmatched_queries_after_filter', 0.0)), device=device),
            'num_pseudo_positive_candidates': torch.tensor(float(mining_stats.get('num_pseudo_positive_candidates', 0.0)), device=device),
            'num_reliable_background_candidates': torch.tensor(float(mining_stats.get('num_reliable_background_candidates', 0.0)), device=device),
            'num_batch_selected_pseudo_positive_queries': torch.tensor(float(mining_stats.get('num_batch_selected_pseudo_positive_queries', 0.0)), device=device),
            'mean_pseudo_positive_threshold': torch.tensor(float(mining_stats.get('pseudo_positive_threshold_sum', 0.0)) / max(float(mining_stats.get('num_thresholds', 0.0)), 1.0), device=device),
            'mean_classification_attenuation': torch.tensor(float(mining_stats.get('mean_classification_attenuation', 1.0)), device=device),
            'num_classification_attenuated_queries': torch.tensor(float(mining_stats.get('num_classification_attenuated_queries', 0.0)), device=device),
            'odqe_gate_mean': odqe_gate_mean if odqe_gate_mean is not None else torch.tensor(0.0, device=device),
            # Backward-compatible aliases for existing metric logging code.
            'stat_num_dummy_pos': torch.tensor(float(mining_stats.get('num_selected_pseudo_positive_queries', 0.0)), device=device),
            'stat_num_dummy_neg': torch.tensor(float(mining_stats.get('num_selected_reliable_background_queries', 0.0)), device=device),
            'stat_num_ignore_queries': torch.tensor(float(mining_stats.get('num_classification_ignored_queries', 0.0)), device=device),
            'stat_num_valid_unmatched': torch.tensor(float(mining_stats.get('num_unmatched_queries_after_filter', 0.0)), device=device),
            'stat_num_pos_candidates': torch.tensor(float(mining_stats.get('num_pseudo_positive_candidates', 0.0)), device=device),
            'stat_num_neg_candidates': torch.tensor(float(mining_stats.get('num_reliable_background_candidates', 0.0)), device=device),
            'stat_num_batch_selected_pos': torch.tensor(float(mining_stats.get('num_batch_selected_pseudo_positive_queries', 0.0)), device=device),
            'stat_pos_thresh_mean': torch.tensor(float(mining_stats.get('pseudo_positive_threshold_sum', 0.0)) / max(float(mining_stats.get('num_thresholds', 0.0)), 1.0), device=device),
            'stat_cls_attn_mean': torch.tensor(float(mining_stats.get('mean_classification_attenuation', 1.0)), device=device),
            'stat_num_cls_soft': torch.tensor(float(mining_stats.get('num_classification_attenuated_queries', 0.0)), device=device),
            'gate_mean': odqe_gate_mean if odqe_gate_mean is not None else torch.tensor(0.0, device=device),
        })
        return losses


class PostProcess(nn.Module):
    def __init__(self, invalid_cls_logits, objectness_temperature=1.0, knownness_temperature=None, pred_per_im=100, unknown_scale=15.0):
        super().__init__()
        self.objectness_temperature = objectness_temperature
        self.knownness_temperature = objectness_temperature if knownness_temperature is None else knownness_temperature
        self.invalid_cls_logits = invalid_cls_logits
        self.pred_per_im = pred_per_im
        self.unknown_scale = float(unknown_scale)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        class_logits = _get_output(outputs, 'pred_class_logits', 'pred_logits')
        objectness_energy = _get_output(outputs, 'pred_objectness_energy', 'pred_obj')
        predicted_boxes = outputs['pred_boxes']
        knownness_energy = outputs.get('pred_knownness_energy', outputs.get('pred_known', None))
        fused = _compute_uod_fused_probabilities(class_logits, objectness_energy, knownness_energy, self.invalid_cls_logits, self.objectness_temperature, self.knownness_temperature, self.unknown_scale)
        probability = fused['fused_scores']
        k = min(self.pred_per_im, probability.shape[1] * max(probability.shape[2], 1))
        topk_values, topk_indices = torch.topk(probability.view(probability.shape[0], -1), k, dim=1)
        scores = topk_values
        topk_boxes = topk_indices // probability.shape[2]
        labels = topk_indices % probability.shape[2]
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
        self.args = args
        print('running with exemplar_replay_selection')

    def calc_energy_per_image(self, outputs, targets, indices):
        fused = _compute_uod_fused_probabilities(
            _get_output(outputs, 'pred_class_logits', 'pred_logits'),
            _get_output(outputs, 'pred_objectness_energy', 'pred_obj'),
            outputs.get('pred_knownness_energy', outputs.get('pred_known', None)),
            self.invalid_cls_logits,
            self.temperature,
            float(getattr(self.args, 'uod_known_temp', getattr(self.args, 'obj_temp', 1.0))) / float(getattr(self.args, 'hidden_dim', 256)),
            float(getattr(self.args, 'uod_postprocess_unknown_scale', 15.0)),
        )
        image_scores = {}
        for batch_index in range(len(targets)):
            image_scores[''.join([chr(int(char)) for char in targets[batch_index]['org_image_id']])] = {
                'labels': targets[batch_index]['labels'].cpu().numpy(),
                'scores': fused['known_scores'][batch_index, indices[batch_index][0], targets[batch_index]['labels']].detach().cpu().numpy(),
            }
        return [image_scores]

    def forward(self, samples, outputs, targets):
        matching_outputs = {'pred_logits': _get_output(outputs, 'pred_class_logits', 'pred_logits'), 'pred_boxes': outputs['pred_boxes']}
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
        if getattr(args, 'uod_enable_unknown', False):
            weight_dict['loss_unk_pseudo'] = getattr(args, 'uod_pseudo_unk_loss_coef', 0.4)
        pseudo_bbox_coef = args.uod_pseudo_bbox_loss_coef if args.uod_pseudo_bbox_loss_coef is not None else args.bbox_loss_coef * 0.5
        pseudo_giou_coef = args.uod_pseudo_giou_loss_coef if args.uod_pseudo_giou_loss_coef is not None else args.giou_loss_coef * 0.25
        weight_dict['loss_bbox_pseudo_cons'] = pseudo_bbox_coef
        weight_dict['loss_giou_pseudo_cons'] = pseudo_giou_coef
        losses.extend(['obj_pseudo', 'obj_neg'])
        if getattr(args, 'uod_enable_unknown', False):
            losses.append('unk_pseudo')
    if getattr(args, 'uod_enable_decorr', False):
        weight_dict['loss_decorr'] = getattr(args, 'uod_decorr_loss_coef', 0.05)
        losses.append('decorr')

    if args.aux_loss:
        aux_weight_dict = {}
        num_aux_layers = max(args.dec_layers - 1, 0)
        low_end = max(1, num_aux_layers // 3) if num_aux_layers > 0 else 0
        mid_end = max(low_end + 1, (2 * num_aux_layers + 2) // 3) if num_aux_layers > 0 else 0
        for layer_index in range(num_aux_layers):
            stage = 'low' if layer_index < low_end else ('mid' if layer_index < mid_end else 'high')
            aux_weight_dict[f'loss_ce_{layer_index}'] = weight_dict['loss_ce']
            aux_weight_dict[f'loss_bbox_{layer_index}'] = weight_dict['loss_bbox']
            aux_weight_dict[f'loss_giou_{layer_index}'] = weight_dict['loss_giou']
            aux_weight_dict[f'loss_obj_ll_{layer_index}'] = weight_dict['loss_obj_ll']
            if 'loss_unk_known' in weight_dict:
                aux_weight_dict[f'loss_unk_known_{layer_index}'] = weight_dict['loss_unk_known']
            if 'loss_obj_pseudo' in weight_dict:
                if stage == 'low':
                    aux_weight_dict[f'loss_obj_pseudo_{layer_index}'] = weight_dict['loss_obj_pseudo'] * float(getattr(args, 'uod_haux_low_obj_coef', 0.35))
                elif stage == 'mid':
                    aux_weight_dict[f'loss_obj_pseudo_{layer_index}'] = weight_dict['loss_obj_pseudo'] * float(getattr(args, 'uod_haux_mid_unknown_coef', 0.45))
                else:
                    aux_weight_dict[f'loss_obj_pseudo_{layer_index}'] = weight_dict['loss_obj_pseudo'] * float(getattr(args, 'uod_haux_high_unknown_coef', 0.7))
            if stage in ['mid', 'high'] and 'loss_unk_pseudo' in weight_dict:
                coef = float(getattr(args, 'uod_haux_mid_unknown_coef', 0.45)) if stage == 'mid' else float(getattr(args, 'uod_haux_high_unknown_coef', 0.7))
                aux_weight_dict[f'loss_unk_pseudo_{layer_index}'] = weight_dict['loss_unk_pseudo'] * coef
            if stage in ['mid', 'high'] and 'loss_bbox_pseudo_cons' in weight_dict:
                coef = float(getattr(args, 'uod_haux_mid_unknown_coef', 0.45)) if stage == 'mid' else float(getattr(args, 'uod_haux_high_unknown_coef', 0.7))
                aux_weight_dict[f'loss_bbox_pseudo_cons_{layer_index}'] = weight_dict['loss_bbox_pseudo_cons'] * coef
                aux_weight_dict[f'loss_giou_pseudo_cons_{layer_index}'] = weight_dict['loss_giou_pseudo_cons'] * coef
            if stage == 'high' and 'loss_decorr' in weight_dict:
                aux_weight_dict[f'loss_decorr_{layer_index}'] = weight_dict['loss_decorr'] * float(getattr(args, 'uod_haux_high_decorr_coef', 0.5))
        aux_weight_dict.update({f'{key}_enc': value for key, value in weight_dict.items() if key in ['loss_ce', 'loss_bbox', 'loss_giou']})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, invalid_cls_logits, args.hidden_dim, focal_alpha=args.focal_alpha, empty_weight=1, args=args)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(
            invalid_cls_logits,
            objectness_temperature=args.obj_temp / args.hidden_dim,
            knownness_temperature=float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / args.hidden_dim,
            pred_per_im=args.num_queries,
            unknown_scale=float(getattr(args, 'uod_postprocess_unknown_scale', 15.0)),
        )
    }
    exemplar_selection = ExemplarSelection(args, num_classes, matcher, invalid_cls_logits, temperature=args.obj_temp / args.hidden_dim)
    return model, criterion, postprocessors, exemplar_selection
