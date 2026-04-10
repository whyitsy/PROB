# ------------------------------------------------------------------------
# Training and evaluation loops for official PROB / UOD experiments.
# This refactor keeps the execution order linear while moving visualization
# and TensorBoard writing into dedicated visual/* helpers.
# ------------------------------------------------------------------------
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable
import os

import torch

import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
from datasets.open_world_eval import OWEvaluator
from visual.eval_visualizer import (
    collect_eval_visual_stats,
    compute_branch_correlation_metrics,
    finalize_eval_visualizations,
    init_eval_visual_state,
    save_eval_qualitative_cases,
)
from visual.train_writer import write_train_step_artifacts


def _get_output(outputs, *keys):
    for key in keys:
        if key in outputs and outputs[key] is not None:
            return outputs[key]
    return None


def _call_criterion(criterion, outputs, targets, epoch):
    try:
        return criterion(outputs, targets, epoch)
    except TypeError:
        return criterion(outputs, targets)


def _forward_model_for_evaluation(model, samples, enable_visual_debug):
    if not enable_visual_debug:
        return model(samples)
    try:
        return model(samples, return_vis_debug=True)
    except TypeError:
        return model(samples)


def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = '[ExemplarReplay]'
    print_frequency = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)

    samples, targets = prefetcher.next()
    image_sorted_scores = {}
    for _ in metric_logger.log_every(range(len(data_loader)), print_frequency, header):
        outputs = model(samples)
        per_batch_scores = exemplar_selection(samples, outputs, targets)
        for item in utils.combine_dict(per_batch_scores):
            image_sorted_scores.update(item[0])
        metric_logger.update(processed_images=len(image_sorted_scores.keys()))
        samples, targets = prefetcher.next()
    logging.info('Collected exemplar scores for %s images', len(image_sorted_scores.keys()))
    return image_sorted_scores


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    tb_writer=None,
    output_dir='',
    step_metrics_file='train/metrics_step.jsonl',
    viz_cfg=None,
    args=None,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = f'Epoch: [{epoch}]'
    print_frequency = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    pseudo_start_epoch = int(getattr(args, 'uod_start_epoch', 8))
    reliable_background_warmup = int(getattr(args, 'uod_neg_warmup_epochs', 0))

    for local_step in metric_logger.log_every(range(len(data_loader)), print_frequency, header):
        outputs = model(samples)
        loss_dict = _call_criterion(criterion, outputs, targets, epoch)
        weight_dict = deepcopy(criterion.weight_dict)

        if epoch < pseudo_start_epoch:
            for key in ('loss_obj_pseudo', 'loss_unk_pseudo', 'loss_obj_neg', 'loss_bbox_pseudo_cons', 'loss_giou_pseudo_cons'):
                if key in weight_dict:
                    weight_dict[key] = 0.0
        elif epoch < pseudo_start_epoch + reliable_background_warmup:
            if 'loss_obj_neg' in weight_dict:
                weight_dict['loss_obj_neg'] = 0.0

        total_loss = sum(loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict)
        reduced_loss_dict = utils.reduce_dict(loss_dict)
        reduced_raw_loss_dict = {key: value for key, value in reduced_loss_dict.items() if key in weight_dict or key.startswith('stat_') or key.startswith('num_')}
        reduced_weighted_loss_dict = {key: value * weight_dict[key] for key, value in reduced_loss_dict.items() if key in weight_dict}
        reduced_total_loss = sum(reduced_weighted_loss_dict.values())
        total_loss_value = reduced_total_loss.item()

        if not math.isfinite(total_loss_value):
            logging.error('Loss is %s, stopping training', total_loss_value)
            logging.error('Reduced loss dict: %s', reduced_loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        if tb_writer is not None or output_dir:
            global_step = epoch * len(data_loader) + local_step
            step_jsonl_path = Path(output_dir) / step_metrics_file if output_dir else Path(step_metrics_file)
            write_train_step_artifacts(
                tb_writer=tb_writer,
                step_jsonl_path=step_jsonl_path,
                global_step=global_step,
                epoch=epoch,
                local_step=local_step,
                optimizer=optimizer,
                grad_total_norm=grad_total_norm,
                outputs=outputs,
                targets=targets,
                criterion=criterion,
                total_loss=total_loss_value,
                reduced_loss_dict=reduced_raw_loss_dict,
                reduced_weighted_loss_dict=reduced_weighted_loss_dict,
                viz_cfg=viz_cfg,
                args=args,
            )

        metric_logger.update(loss=total_loss_value, **reduced_weighted_loss_dict)
        for key, value in reduced_raw_loss_dict.items():
            metric_logger.update(**{key: value})
        if 'class_error' in reduced_loss_dict:
            metric_logger.update(class_error=reduced_loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

    metric_logger.synchronize_between_processes()
    logging.info('Averaged stats: %s', metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_dataset,
    device,
    output_dir,
    args,
    viz_cfg=None,
    tb_writer=None,
    epoch=0,
):
    epoch = max(int(epoch), 0)
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    iou_types = ('bbox',)
    evaluator = OWEvaluator(base_dataset, iou_types, args=args)

    visual_state = init_eval_visual_state(viz_cfg) if (viz_cfg is not None and utils.is_main_process()) else None

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = _forward_model_for_evaluation(model, samples, enable_visual_debug=(visual_state is not None))

        original_sizes = torch.stack([target['orig_size'] for target in targets], dim=0)
        visual_sizes = torch.stack([target['size'] for target in targets], dim=0)
        results = postprocessors['bbox'](outputs, original_sizes)
        visual_results = postprocessors['bbox'](outputs, visual_sizes) if visual_state is not None else None

        result_by_image_id = {target['image_id'].item(): output for target, output in zip(targets, results)}
        evaluator.update(result_by_image_id)

        if visual_state is not None:
            collect_eval_visual_stats(visual_state, outputs, targets, criterion, args)
            visual_output_dir = os.path.join(output_dir, 'eval', 'visualizations', f'epoch_{int(epoch):04d}')
            save_eval_qualitative_cases(
                visual_state,
                samples,
                targets,
                visual_results,
                outputs,
                criterion,
                args,
                visual_output_dir,
                viz_cfg,
                tb_writer=tb_writer,
                global_step=epoch,
                epoch=epoch,
            )

    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    open_world_metrics = evaluator.summarize()

    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    stats['open_world_metrics'] = dict(open_world_metrics) if isinstance(open_world_metrics, dict) else {}
    if visual_state is not None:
        for key, value in compute_branch_correlation_metrics(visual_state).items():
            if value is not None:
                stats['open_world_metrics'][key] = value
        finalize_eval_visualizations(visual_state, output_dir, epoch, viz_cfg, tb_writer=tb_writer)

    if 'bbox' in postprocessors:
        stats['coco_eval_bbox'] = evaluator.coco_eval['bbox'].stats.tolist()

    return stats, evaluator
