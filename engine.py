# ------------------------------------------------------------------------
# Training and evaluation loop for official PROB / UOD experiments.
# Refactored to keep step logging and evaluation visualization modular.
# ------------------------------------------------------------------------
import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torch

import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.eval_viz import (
    _ensure_dir,
    collect_eval_stats,
    compute_decoupling_corr_metrics,
    finalize_eval_visualizations,
    init_eval_viz_state,
    save_eval_qualitative,
)
from util.step_logger import log_train_step


def _call_criterion(criterion, outputs, targets, epoch):
    try:
        return criterion(outputs, targets, epoch)
    except TypeError:
        return criterion(outputs, targets)


def _safe_float(v):
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    return float(v)


def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)

    samples, targets = prefetcher.next()

    image_sorted_scores_reduced = {}
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
    logging.info('found a total of %s images', len(image_sorted_scores_reduced.keys()))
    return image_sorted_scores_reduced


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int,
                    max_norm: float = 0, writer=None, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    pseudo_start = int(getattr(args, 'uod_start_epoch', 8))
    neg_warmup = int(getattr(args, 'uod_neg_warmup_epochs', 0))

    for step in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = _call_criterion(criterion, outputs, targets, epoch)
        weight_dict = deepcopy(criterion.weight_dict)

        if epoch < nc_epoch:
            for k in list(weight_dict.keys()):
                if 'NC' in k:
                    weight_dict[k] = 0

        if epoch < pseudo_start:
            for k in ('loss_obj_pseudo', 'loss_unk_pseudo', 'loss_obj_neg'):
                if k in weight_dict:
                    weight_dict[k] = 0.0
        elif epoch < pseudo_start + neg_warmup:
            if 'loss_obj_neg' in weight_dict:
                weight_dict['loss_obj_neg'] = 0.0

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logging.error('Loss is %s, stopping training', loss_value)
            logging.error('Reduced loss dict: %s', loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        if writer is not None or getattr(args, 'output_dir', ''):
            global_step = epoch * len(data_loader) + step
            step_jsonl_path = Path(getattr(args, 'output_dir', '')) / getattr(args, 'step_metrics_jsonl', 'metrics_step.jsonl') if getattr(args, 'output_dir', '') else Path('metrics_step.jsonl')
            log_train_step(
                writer=writer,
                step_jsonl_path=step_jsonl_path,
                global_step=global_step,
                epoch=epoch,
                local_step=step,
                optimizer=optimizer,
                grad_total_norm=grad_total_norm,
                outputs=outputs,
                targets=targets,
                criterion=criterion,
                loss_value=loss_value,
                loss_dict_reduced=loss_dict_reduced,
                loss_dict_reduced_scaled=loss_dict_reduced_scaled,
                hist_every=int(getattr(args, 'step_histogram_every', 100)),
                args=args,
            )
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        extra_meter_keys = [
            'known_unk_suppress_coeff', 'unknown_known_suppress_coeff', 'gate_mean',
            'stat_num_dummy_pos', 'stat_num_dummy_neg', 'stat_num_valid_unmatched',
            'stat_num_pos_candidates', 'stat_num_neg_candidates',
            'stat_num_batch_selected_pos', 'stat_pos_thresh_mean', 'stat_cls_attn_mean', 'stat_num_cls_soft'
        ]
        for key in extra_meter_keys:
            if key in loss_dict_reduced:
                metric_logger.update(**{key: loss_dict_reduced[key]})
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

    metric_logger.synchronize_between_processes()
    logging.info('Averaged stats: %s', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args, writer=None, epoch=0):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, 'panoptic_eval'),
        )

    vis_state = init_eval_viz_state(args) if (getattr(args, 'viz', False) and utils.is_main_process()) else None
    vis_dir = None
    if vis_state is not None:
        vis_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{int(epoch):04d}', 'qualitative')
        _ensure_dir(vis_dir)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
        vis_target_sizes = torch.stack([t['size'] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        vis_results = postprocessors['bbox'](outputs, vis_target_sizes) if vis_state is not None else None

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t['size'] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors['panoptic'](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                file_name = f'{image_id:012d}.png'
                res_pano[i]['image_id'] = image_id
                res_pano[i]['file_name'] = file_name
            panoptic_evaluator.update(res_pano)

        if vis_state is not None:
            collect_eval_stats(vis_state, outputs, targets, criterion, args)
            vis_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{int(epoch):04d}', 'qualitative')
            save_eval_qualitative(vis_state, samples, targets, vis_results, outputs, args=args, out_dir=vis_dir, writer=writer, step=epoch)

    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    else:
        res = {}

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics'] = dict(res) if isinstance(res, dict) else {}
    if vis_state is not None:
        corr_metrics = compute_decoupling_corr_metrics(vis_state)
        for k, v in corr_metrics.items():
            if v is not None:
                stats['metrics'][k] = v

    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res['All']
        stats['PQ_th'] = panoptic_res['Things']
        stats['PQ_st'] = panoptic_res['Stuff']

    if vis_state is not None:
        finalize_eval_visualizations(vis_state, output_dir, epoch, writer=writer)

    return stats, coco_evaluator