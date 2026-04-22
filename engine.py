import logging
import math
import sys
from copy import deepcopy
from typing import Iterable

import torch

import util.misc as utils
from datasets.data_prefetcher import data_prefetcher
from datasets.open_world_eval import OWEvaluator

@torch.inference_mode()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    model.eval()
    exemplar_selection.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    header = '[ExempReplay]'
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=False)
    samples, targets = prefetcher.next()

    image_sorted_scores_reduced = {}

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        if samples is None:
            break

        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)

        for item in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(item[0])

        metric_logger.update(processed_images=len(image_sorted_scores_reduced.keys()))

        del outputs, image_sorted_scores
        samples, targets = prefetcher.next()

    logging.info('found a total of %s images', len(image_sorted_scores_reduced.keys()))
    return image_sorted_scores_reduced


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    viz_ctx=None,
    args=None,
):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    model_stat_logger = utils.MetricLogger(delimiter='  ')

    header = f'Epoch: [{epoch}]'
    print_frequency = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    pseudo_start_epoch = int(getattr(args, 'uod_start_epoch', 8))
    reliable_background_warmup = int(getattr(args, 'uod_neg_warmup_epochs', 0))

    for local_step in metric_logger.log_every(range(len(data_loader)), print_frequency, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets, epoch)
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

        reduced_model_stat_dict = {
            key: value
            for key, value in reduced_loss_dict.items()
            if key.startswith('stat_') or key.startswith('num_') or key == 'gate_mean'
        }
        reduced_raw_loss_dict = {
            key: value
            for key, value in reduced_loss_dict.items()
            if key in weight_dict
        }
        reduced_weighted_loss_dict = {
            key: value * weight_dict[key]
            for key, value in reduced_loss_dict.items()
            if key in weight_dict
        }
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

        global_step = epoch * len(data_loader) + local_step
        if viz_ctx is not None and getattr(viz_ctx, 'tb_writer', None) is not None:
            viz_ctx.tb_writer.add_scalar('train/loss/total', total_loss_value, global_step)
            viz_ctx.tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            viz_ctx.tb_writer.add_scalar('train/grad_norm', grad_total_norm, global_step)
            for key, value in reduced_raw_loss_dict.items():
                viz_ctx.tb_writer.add_scalar(f'train/loss_raw/{key}', value, global_step)
            for key, value in reduced_weighted_loss_dict.items():
                viz_ctx.tb_writer.add_scalar(f'train/loss_weighted/{key}', value, global_step)

        metric_logger.update(loss=total_loss_value)
        metric_logger.update(**{f'weighted_{key}': value for key, value in reduced_weighted_loss_dict.items()})
        metric_logger.update(**{f'raw_{key}': value for key, value in reduced_raw_loss_dict.items()})
        if 'class_error' in reduced_loss_dict:
            metric_logger.update(class_error=reduced_loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(grad_norm=grad_total_norm)

        if reduced_model_stat_dict:
            model_stat_logger.update(**reduced_model_stat_dict)

        samples, targets = prefetcher.next()

    metric_logger.synchronize_between_processes()
    model_stat_logger.synchronize_between_processes()
    logging.info('Averaged stats: %s', metric_logger)

    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    stats.update({key: meter.global_avg for key, meter in model_stat_logger.meters.items()})
    return stats


@torch.inference_mode()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_dataset,
    device,
    args,
    viz_ctx=None,
    epoch=0,
):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    iou_types = ('bbox',)
    evaluator = OWEvaluator(base_dataset, iou_types, args=args)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = model(samples)

        original_sizes = torch.stack([target['orig_size'] for target in targets], dim=0)
        results = postprocessors['bbox'](outputs, original_sizes)
        result_by_image_id = {target['image_id'].item(): output for target, output in zip(targets, results)}
        evaluator.update(result_by_image_id)

    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    open_world_metrics = evaluator.summarize()

    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    stats['open_world_metrics'] = dict(open_world_metrics) if isinstance(open_world_metrics, dict) else {}

    if 'bbox' in postprocessors:
        stats['coco_eval_bbox'] = evaluator.coco_eval['bbox'].stats.tolist()

    return stats, evaluator
