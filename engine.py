# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
 
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
import util.misc as utils
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from copy import deepcopy
from util.visual.visual import *

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0, writer=None, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets, epoch) 
        weight_dict = deepcopy(criterion.weight_dict)
        
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.
        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
        if epoch < args.unk_label_start_epoch:
            for k, v in weight_dict.items():
                if 'loss_obj' in k or 'loss_unk' in k:
                    weight_dict[k] = 0
         
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        
        if writer is not None:
            global_step = epoch * len(data_loader) + _
            writer.add_scalar('train/total_loss', loss_value, global_step)
            for k, v in loss_dict_reduced_scaled.items():
                writer.add_scalar(f'train_scaled/{k}', v, global_step)
            for k, v in loss_dict_reduced_unscaled.items():
                writer.add_scalar(f'train_unscaled/{k}', v, global_step)

            # 每 200 step 画一次图，避免TensorBoard过大
            if global_step % 200 == 0:
                try:
                    log_debug_visualizations(
                        writer=writer,
                        samples=samples,
                        targets=targets,
                        outputs=outputs,
                        criterion=criterion,
                        epoch=epoch,
                        global_step=global_step,
                        max_images=2,
                        prefix='train_vis'
                    )
                except Exception as e:
                    print(f'[warn] visualization failed at step {global_step}: {e}')

            # 重点监控 objectness / 伪标签统计
            stat_keys = [
                'stat_num_dummy_pos',
                'stat_num_dummy_neg',
                'stat_pos_energy_mean',
                'stat_neg_energy_mean',
                'stat_matched_energy_mean',
                'stat_pos_thresh_mean',
                'stat_neg_thresh_mean',
            ]
            for k in stat_keys:
                if k in loss_dict_reduced:
                    v = loss_dict_reduced[k]
                    writer.add_scalar(f'train_stats/{k}', v.item() if torch.is_tensor(v) else v, global_step)

            # 记录 pred_obj energy 直方图
            if 'pred_obj' in outputs:
                writer.add_histogram('train_hist/pred_obj_energy_all',
                                    outputs['pred_obj'].detach().float().cpu(),
                                    global_step)

                # matched / unmatched 分开画，更有用
                try:
                    outputs_without_aux = {k: v for k, v in outputs.items()
                                        if k not in ['aux_outputs', 'enc_outputs', 'pred_obj', 'samples', 'pred_proj']}
                    indices_for_hist = criterion.matcher(outputs_without_aux, targets)

                    batch_size, num_queries = outputs['pred_obj'].shape[:2]
                    obj_energy = outputs['pred_obj'].detach()

                    matched_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=obj_energy.device)
                    for b_idx, (src, _) in enumerate(indices_for_hist):
                        if len(src) > 0:
                            matched_mask[b_idx, src] = True

                    matched_energy = obj_energy[matched_mask]
                    unmatched_energy = obj_energy[~matched_mask]

                    if matched_energy.numel() > 0:
                        writer.add_histogram('train_hist/pred_obj_energy_matched',
                                            matched_energy.float().cpu(),
                                            global_step)
                    if unmatched_energy.numel() > 0:
                        writer.add_histogram('train_hist/pred_obj_energy_unmatched',
                                            unmatched_energy.float().cpu(),
                                            global_step)
                except Exception:
                    pass
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
 
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
 
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
 
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
 
            panoptic_evaluator.update(res_pano)
 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
 
    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics']=res
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
 
    
@torch.no_grad()
def get_exemplar_replay(model, exemplar_selection, device, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '[ExempReplay]'
    print_freq = 10
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    image_sorted_scores_reduced={}
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        image_sorted_scores = exemplar_selection(samples, outputs, targets)
        for i in utils.combine_dict(image_sorted_scores):
            image_sorted_scores_reduced.update(i[0])
            
        metric_logger.update(loss=len(image_sorted_scores_reduced.keys()))
        samples, targets = prefetcher.next()
        
    print(f'found a total of {len(image_sorted_scores_reduced.keys())} images')
    return image_sorted_scores_reduced