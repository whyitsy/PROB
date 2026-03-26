# ------------------------------------------------------------------------
# PROB: Probabilistic Objectness for Open World Object Detection
# Official training entry adapted for tensorboard-only logging and UOD methods.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import datasets.samplers as samplers
import util.misc as utils
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from engine import evaluate, get_exemplar_replay, train_one_epoch
from util.log import setup_logging
from models import build_model


METRICS_JSONL = 'metrics_log.jsonl'


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _sanitize_for_checkpoint(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_checkpoint(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_checkpoint(v) for k, v in obj.items()}
    if isinstance(obj, argparse.Namespace):
        return {k: _sanitize_for_checkpoint(v) for k, v in vars(obj).items() if not k.startswith('_')}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return repr(obj)


def _build_checkpoint_args(args):
    return _sanitize_for_checkpoint(args)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _read_metrics_history(metrics_jsonl_path: Path):
    history = []
    if not metrics_jsonl_path.exists():
        return history

    for line in metrics_jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue

        metrics = record.get('test_metrics', {}) or {}
        epoch = record.get('epoch', None)
        if epoch is None or not isinstance(metrics, dict) or not metrics:
            continue

        history.append({
            'epoch': int(epoch),
            'Current AP50': _safe_float(metrics.get('CK_AP50')),
            'Known AP50': _safe_float(metrics.get('K_AP50')),
            'Unknown Recall50': _safe_float(metrics.get('U_R50')),
            'WI@0.8': _safe_float(metrics.get('WI')),
            'A-OSE': _safe_float(metrics.get('AOSA', metrics.get('A-OSE'))),
        })

    history.sort(key=lambda x: x['epoch'])
    return history

def _filter_valid_series(history, key):
    xs, ys = [], []
    for item in history:
        val = item.get(key)
        if val is None:
            continue
        xs.append(item['epoch'])
        ys.append(val)
    return xs, ys


def _plot_percent_metric_trends(history, plots_dir: Path):
    keys = ['Current AP50', 'Known AP50', 'Unknown Recall50']
    has_any = any(_filter_valid_series(history, k)[0] for k in keys)
    if not has_any:
        return

    plt.figure(figsize=(10, 6))
    for key in keys:
        xs, ys = _filter_valid_series(history, key)
        if xs:
            plt.plot(xs, ys, marker='o', linewidth=2, label=key)

    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.title('Open-World Percentage Metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'open_world_metric_trends_percent.png', dpi=200)
    plt.close()
    
def _plot_openworld_metric_trends(history, plots_dir: Path):
    xs_wi, ys_wi = _filter_valid_series(history, 'WI@0.8')
    xs_ose, ys_ose = _filter_valid_series(history, 'A-OSE')
    if not xs_wi and not xs_ose:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    lines = []
    labels = []

    if xs_wi:
        l1 = ax1.plot(xs_wi, ys_wi, marker='o', linewidth=2,
                      label='WI@Recall0.8, IoU50')
        lines += l1
        labels += [line.get_label() for line in l1]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Wilderness Impact (lower is better)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if xs_ose:
        l2 = ax2.plot(xs_ose, ys_ose, marker='s', linewidth=2,
                      label='A-OSE@IoU50')
        lines += l2
        labels += [line.get_label() for line in l2]
    ax2.set_ylabel('Absolute Open-Set Error (lower is better)')

    if lines:
        ax1.legend(lines, labels, loc='best')

    plt.title('Open-World Error Metrics')
    fig.tight_layout()
    fig.savefig(plots_dir / 'open_world_metric_trends_openworld.png', dpi=200)
    plt.close(fig)

def _plot_current_metric_bars(history, plots_dir: Path):
    if not history:
        return
    latest = history[-1]

    # 百分比指标
    percent_items = [
        ('Current AP50', latest.get('Current AP50')),
        ('Known AP50', latest.get('Known AP50')),
        ('Unknown Recall50', latest.get('Unknown Recall50')),
    ]
    percent_items = [(k, v) for k, v in percent_items if v is not None]
    if percent_items:
        plt.figure(figsize=(8, 5))
        labels = [k for k, _ in percent_items]
        values = [v for _, v in percent_items]
        plt.bar(labels, values)
        plt.ylabel('Percentage (%)')
        plt.title(f'Current Percentage Metrics (Epoch {latest["epoch"]})')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(plots_dir / 'current_metrics_bar_percent.png', dpi=200)
        plt.close()

    # 开放世界误差指标
    wi_val = latest.get('WI@0.8')
    ose_val = latest.get('A-OSE')
    if wi_val is not None or ose_val is not None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['WI@0.8', 'A-OSE'], rotation=15)
        ax1.set_ylabel('WI@0.8 (lower is better)')

        if wi_val is not None:
            ax1.bar([0], [wi_val], width=0.5)

        if ose_val is not None:
            ax2 = ax1.twinx()
            ax2.bar([1], [ose_val], width=0.5)
            ax2.set_ylabel('A-OSE (lower is better)')

        fig.suptitle(f'Current Open-World Error Metrics (Epoch {latest["epoch"]})')
        fig.tight_layout()
        fig.savefig(plots_dir / 'current_metrics_bar_openworld.png', dpi=200)
        plt.close(fig)
        
        
def _plot_training_loss_trends(metrics_jsonl_path: Path, plots_dir: Path):
    if not metrics_jsonl_path.exists():
        return

    epochs = []
    total_losses = []
    objectness_losses = []
    unknown_losses = []
    pseudo_losses = []
    negative_losses = []

    for line in metrics_jsonl_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue

        epoch = record.get('epoch')
        if epoch is None:
            continue

        train_loss = record.get('train_loss')
        if train_loss is not None:
            epochs.append(epoch)
            total_losses.append(float(train_loss))

            obj_loss = record.get('train_loss_obj_ll_unscaled', record.get('train_loss_obj_ll'))
            unk_loss = record.get('train_loss_unk_unscaled', record.get('train_loss_unk'))
            pseudo_loss = record.get('train_loss_obj_pseudo_unscaled', record.get('train_loss_obj_pseudo'))
            neg_loss = record.get('train_loss_obj_neg_unscaled', record.get('train_loss_obj_neg'))

            objectness_losses.append(None if obj_loss is None else float(obj_loss))
            unknown_losses.append(None if unk_loss is None else float(unk_loss))
            pseudo_losses.append(None if pseudo_loss is None else float(pseudo_loss))
            negative_losses.append(None if neg_loss is None else float(neg_loss))

    if not epochs:
        return

    # 1) 总损失
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, marker='o', linewidth=2, label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Total Loss Trend')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'training_loss_total_trend.png', dpi=200)
    plt.close()

    # 2) 开放世界相关损失分量
    component_series = {
        # 'objectness_ll': objectness_losses,
        'unknown': unknown_losses,
        'pseudo_pos': pseudo_losses,
        'pseudo_neg': negative_losses,
    }
    has_component = any(any(v is not None for v in vals) for vals in component_series.values())
    if has_component:
        plt.figure(figsize=(10, 6))
        for name, vals in component_series.items():
            xs = [e for e, v in zip(epochs, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if xs:
                plt.plot(xs, ys, marker='o', linewidth=1.8, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Open-World Loss Components')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_loss_openworld_components.png', dpi=200)
        plt.close()
        
        # 这里单独画objectness_ll是因为它的数值范围可能和其他loss分量差很多，放在一起可能不好看
        plt.figure(figsize=(10, 6))
        xs = [e for e, v in zip(epochs, objectness_losses) if v is not None]
        ys = [v for v in objectness_losses if v is not None]
        plt.plot(xs, ys, marker='o', linewidth=1.8, label='objectness_ll')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Objectness Loss Trend')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_loss_objectness_trend.png', dpi=200)
        plt.close()

def _plot_pseudo_stat_trends(metrics_jsonl_path: Path, plots_dir: Path):
    if not metrics_jsonl_path.exists():
        return

    count_keys = [
        'train_stat_num_dummy_pos',
        'train_stat_num_dummy_neg',
        'train_stat_num_valid_unmatched',
        'train_stat_num_pos_candidates',
        'train_stat_num_batch_selected_pos',
    ]
    thresh_keys = ['train_stat_pos_thresh_mean']

    count_data = {k: {'x': [], 'y': []} for k in count_keys}
    thresh_data = {k: {'x': [], 'y': []} for k in thresh_keys}

    for line in metrics_jsonl_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue

        epoch = record.get('epoch')
        if epoch is None:
            continue

        for key in count_keys:
            val = record.get(key)
            if val is not None:
                count_data[key]['x'].append(epoch)
                count_data[key]['y'].append(float(val))

        for key in thresh_keys:
            val = record.get(key)
            if val is not None:
                thresh_data[key]['x'].append(epoch)
                thresh_data[key]['y'].append(float(val))

    # 1) 数量型统计
    if any(v['x'] for v in count_data.values()):
        plt.figure(figsize=(11, 7))
        display_names = {
            'train_stat_num_dummy_pos': 'dummy_pos',
            'train_stat_num_dummy_neg': 'dummy_neg',
            'train_stat_num_valid_unmatched': 'valid_unmatched',
            'train_stat_num_pos_candidates': 'pos_candidates',
            'train_stat_num_batch_selected_pos': 'batch_selected_pos',
        }
        for key, v in count_data.items():
            if v['x']:
                plt.plot(v['x'], v['y'], marker='o', linewidth=1.8, label=display_names.get(key, key))
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.title('Pseudo-Supervision Count Statistics')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pseudo_stat_counts.png', dpi=200)
        plt.close()

    # 2) 阈值型统计
    if any(v['x'] for v in thresh_data.values()):
        plt.figure(figsize=(10, 6))
        for key, v in thresh_data.items():
            if v['x']:
                plt.plot(v['x'], v['y'], marker='o', linewidth=2, label='pos_thresh_mean')
        plt.xlabel('Epoch')
        plt.ylabel('Threshold')
        plt.title('Pseudo-Supervision Threshold Trend')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'pseudo_stat_thresholds.png', dpi=200)
        plt.close()

def _refresh_metric_plots(output_dir: Path):
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl_path = output_dir / METRICS_JSONL

    history = _read_metrics_history(metrics_jsonl_path)
    _plot_percent_metric_trends(history, plots_dir)
    _plot_openworld_metric_trends(history, plots_dir)
    _plot_current_metric_bars(history, plots_dir)
    _plot_training_loss_trends(metrics_jsonl_path, plots_dir)
    _plot_pseudo_stat_trends(metrics_jsonl_path, plots_dir)


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    ################ Deformable DETR ################
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=41, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--masks', default=False, action='store_true', help='Train segmentation head if the flag is provided')
    parser.add_argument('--backbone', default='dino_resnet50', type=str, help='Name of the convolutional backbone to use')

    parser.add_argument('--frozen_weights', type=str, default=None, help='Path to the pretrained model. If set, only the mask head will be trained')
    parser.add_argument('--dilation', action='store_true', help='If true, we replace stride with dilation in the last convolutional block (DC5)')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help='Type of positional embedding to use on top of the image features')
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help='position / size * scale')
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    parser.add_argument('--enc_layers', default=6, type=int, help='Number of encoding layers in the transformer')
    parser.add_argument('--dec_layers', default=6, type=int, help='Number of decoding layers in the transformer')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='Intermediate size of the feedforward layers in the transformer blocks')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout applied in the transformer')
    parser.add_argument('--nheads', default=8, type=int, help='Number of attention heads inside the transformer attentions')
    parser.add_argument('--num_queries', default=100, type=int, help='Number of query slots')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help='Disables auxiliary decoding losses')
    parser.add_argument('--set_cost_class', default=2, type=float, help='Class coefficient in the matching cost')
    parser.add_argument('--set_cost_bbox', default=5, type=float, help='L1 box coefficient in the matching cost')
    parser.add_argument('--set_cost_giou', default=2, type=float, help='giou box coefficient in the matching cost')
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', default=True, action='store_true')
    parser.add_argument('--viz_num_samples', default=12, type=int)
    parser.add_argument('--viz_tb_images', default=4, type=int)
    parser.add_argument('--eval_every', default=5, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    ################ OW-DETR ################
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--unmatched_boxes', default=False, action='store_true')
    parser.add_argument('--top_unk', default=5, type=int)
    parser.add_argument('--featdim', default=1024, type=int)
    parser.add_argument('--invalid_cls_logits', default=False, action='store_true', help='owod setting')
    parser.add_argument('--NC_branch', default=False, action='store_true')
    parser.add_argument('--bbox_thresh', default=0.3, type=float)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--nc_loss_coef', default=2, type=float)
    parser.add_argument('--train_set', default='', help='training txt files')
    parser.add_argument('--test_set', default='', help='testing txt files')
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--nc_epoch', default=0, type=int)
    parser.add_argument('--dataset', default='OWDETR', help='defines which dataset is used. Built for: {TOWOD, OWDETR, VOC2007}')
    parser.add_argument('--data_root', default='/mnt/data/kky/datasets/owdetr/data/OWOD', type=str)
    parser.add_argument('--unk_conf_w', default=1.0, type=float)

    ################ PROB OWOD ################
    parser.add_argument('--model_type', default='prob', type=str, choices=['prob', 'uod', 'uod_paramless'])
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--wandb_project', default='', type=str)

    parser.add_argument('--obj_loss_coef', default=8e-4, type=float)
    parser.add_argument('--obj_temp', default=1.3, type=float)
    parser.add_argument('--freeze_prob_model', default=False, action='store_true', help='freeze probabilistic estimation')

    parser.add_argument('--num_inst_per_class', default=50, type=int, help='number of instances per class')
    parser.add_argument('--exemplar_replay_selection', default=False, action='store_true', help='use learned exemplar selection')
    parser.add_argument('--exemplar_replay_max_length', default=int(1e10), type=int, help='max number of images that can be saved')
    parser.add_argument('--exemplar_replay_dir', default='', type=str, help='directory of exemplar replay txt files')
    parser.add_argument('--exemplar_replay_prev_file', default='', type=str, help='path to previous ft file')
    parser.add_argument('--exemplar_replay_cur_file', default='', type=str, help='path to current ft file')
    parser.add_argument('--exemplar_replay_random', default=False, action='store_true', help='make selection random')

    ################ UOD (Chapter 3 / Chapter 4) ################
    parser.add_argument('--uod_enable_unknown', default=False, action='store_true', help='enable explicit unknownness branch')
    parser.add_argument('--uod_enable_pseudo', default=False, action='store_true', help='enable pseudo unknown supervision')
    parser.add_argument('--uod_enable_batch_dynamic', default=False, action='store_true', help='enable batch-level dynamic pseudo allocation')
    parser.add_argument('--uod_enable_decorr', default=False, action='store_true', help='enable triplet decoupled optimization')
    parser.add_argument('--uod_enable_cls_soft_attn', default=False, action='store_true', help='attenuate classification loss on pseudo-positive queries')
    parser.add_argument('--uod_enable_odqe', default=False, action='store_true', help='enable object-aware task decouple query enhancement(ODQE) module')

    parser.add_argument('--unk_loss_coef', default=0.3, type=float, help='weight of matched-known negative unknownness loss')
    parser.add_argument('--uod_pseudo_unk_loss_coef', default=0.4, type=float, help='weight of pseudo-unknown unknownness loss')
    parser.add_argument('--uod_bg_unk_loss_coef', default=0.2, type=float, help='weight of reliable-background unknownness negative loss')
    parser.add_argument('--uod_pseudo_obj_loss_coef', default=0.3, type=float, help='weight of pseudo-positive objectness loss')
    parser.add_argument('--uod_obj_neg_loss_coef', default=0.2, type=float, help='weight of reliable-background objectness loss')
    parser.add_argument('--uod_orth_loss_coef', default=0.05, type=float, help='weight of feature orthogonality loss')
    parser.add_argument('--uod_decorr_loss_coef', default=0.05, type=float, help='weight of prediction decorrelation loss')

    parser.add_argument('--uod_start_epoch', default=3, type=int, help='epoch to start pseudo supervision')
    parser.add_argument('--uod_neg_warmup_epochs', default=2, type=int, help='delay reliable-background losses after pseudo start')
    parser.add_argument('--uod_pos_quantile', default=0.25, type=float, help='matched energy quantile used as threshold base')
    parser.add_argument('--uod_pos_scale', default=1.2, type=float, help='positive threshold scale')
    parser.add_argument('--uod_min_pos_thresh', default=0.08, type=float, help='minimum pseudo-positive energy threshold')
    parser.add_argument('--uod_known_reject_thresh', default=0.15, type=float, help='maximum known score for pseudo-unknown candidates')
    parser.add_argument('--uod_neg_margin', default=0.8, type=float, help='margin for reliable background negative mining')
    parser.add_argument('--uod_pos_per_img_cap', default=1, type=int, help='max pseudo positives per image')
    parser.add_argument('--uod_neg_per_img', default=1, type=int, help='max reliable background negatives per image')
    parser.add_argument('--uod_batch_topk_max', default=8, type=int, help='max pseudo positives selected per batch')
    parser.add_argument('--uod_batch_topk_ratio', default=0.25, type=float, help='dynamic ratio for batch-level pseudo selection')
    parser.add_argument('--uod_max_iou', default=0.2, type=float, help='max IoU with GT for pseudo candidates')
    parser.add_argument('--uod_max_iof', default=0.4, type=float, help='max IoF with GT for pseudo candidates')
    parser.add_argument('--uod_min_area', default=0.002, type=float, help='min normalized area for pseudo candidates')
    parser.add_argument('--uod_min_side', default=0.05, type=float, help='min normalized side length for pseudo candidates')
    parser.add_argument('--uod_max_aspect_ratio', default=4.0, type=float, help='max aspect ratio for pseudo candidates')
    parser.add_argument('--uod_cls_soft_attn_alpha', default=0.8, type=float, help='strength of pseudo-positive classification attenuation')
    parser.add_argument('--uod_cls_soft_attn_min', default=0.1, type=float, help='minimum query weight under classification attenuation')

    return parser


def get_datasets(args):
    logging.info('Dataset: %s', args.dataset)
    dataset_train = OWDetection(args, args.data_root, image_set=args.train_set,
                                transforms=make_coco_transforms(args.train_set), dataset=args.dataset)
    dataset_val = OWDetection(args, args.data_root, image_set=args.test_set,
                              dataset=args.dataset, transforms=make_coco_transforms(args.test_set))
    logging.info('Train split: %s', args.train_set)
    logging.info('Test split: %s', args.test_set)
    logging.info('%s', dataset_train)
    logging.info('%s', dataset_val)
    return dataset_train, dataset_val


def create_ft_dataset(args, image_sorted_scores):
    logging.info('found a total of %s images', len(image_sorted_scores.keys()))
    tmp_dir = args.data_root + '/ImageSets/' + args.dataset + '/' + args.exemplar_replay_dir + '/'

    class_sorted_scores = {}
    imgs_per_class = {}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS):
        class_sorted_scores[str(i)] = []
        imgs_per_class[str(i)] = []

    for _, v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            class_sorted_scores[str(v['labels'][j])].append(v['scores'][j])

    class_threshold = {}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS):
        tmp = np.array(class_sorted_scores[str(i)])
        tmp.sort()
        tmp = torch.Tensor(tmp)
        if len(tmp) > args.num_inst_per_class and not args.exemplar_replay_random:
            max_val = tmp[-args.num_inst_per_class // 2]
            min_val = tmp[args.num_inst_per_class // 2]
        else:
            if args.exemplar_replay_random:
                logging.info('using random exemplar selection')
            else:
                logging.info('only found %s imgs in class %s', len(tmp), i)
            max_val = tmp.min()
            min_val = tmp.max()
        class_threshold[str(i)] = (min_val, max_val)

    save_imgs = []
    for k, v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            label = str(v['labels'][j])
            if (v['scores'][j] <= class_threshold[label][0].numpy() or v['scores'][j] >= class_threshold[label][1].numpy()) and (len(imgs_per_class[label]) <= args.num_inst_per_class + 2):
                save_imgs.append(k)
                imgs_per_class[label].append(k)

    logging.info('found %s images in run', len(np.unique(save_imgs)))
    if len(args.exemplar_replay_prev_file) > 0:
        previous_ft = open(tmp_dir + args.exemplar_replay_prev_file, 'r').read().splitlines()
        save_imgs += previous_ft

    save_imgs = np.unique(save_imgs)
    np.random.shuffle(save_imgs)
    if len(save_imgs) > args.exemplar_replay_max_length:
        save_imgs = save_imgs[:args.exemplar_replay_max_length]

    os.makedirs(tmp_dir, exist_ok=True)
    with open(tmp_dir + args.exemplar_replay_cur_file, 'w') as f:
        for line in save_imgs:
            f.write(line)
            f.write('\n')


def _log_test_stats_to_tb(writer, test_stats, epoch):
    if writer is None:
        return
    for key, val in test_stats.items():
        if key == 'metrics' and isinstance(val, dict):
            for mk, mv in val.items():
                if isinstance(mv, (int, float)):
                    tag = 'A-OSE' if mk == 'AOSA' else mk
                    writer.add_scalar(f'eval_metrics/{tag}', mv, epoch)
        elif isinstance(val, (int, float)):
            writer.add_scalar(f'eval/{key}', val, epoch)

def rprint(msg):
    rank = int(os.environ.get("RANK", -1))
    print(f"[rank {rank}] {msg}", flush=True)
    
def main(args):
    utils.init_distributed_mode(args)
    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output=args.output_dir, distributed_rank=utils.get_rank(), abbrev_name='PROB')
    logging.info('Arguments:\n%s', args)

    writer = None
    if utils.is_main_process() and args.output_dir:
        run_name = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
        tb_log_dir = os.path.join(args.output_dir, 'tensorboard_logs', run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        writer.add_text('args', str(args), 0)
        logging.info('TensorBoard log dir: %s', tb_log_dir)
    args.writer = writer

    logging.info('git\n  %s\n', utils.get_sha())
    if args.frozen_weights is not None:
        assert args.masks, 'Frozen training is meant for segmentation only'

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, exemplar_selection = build_model(args, mode=args.model_type)
    model.to(device)

    model_without_ddp = model
    logging.info('%s', model_without_ddp)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('number of params: %s', n_parameters)

    dataset_train, dataset_val = get_datasets(args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers, pin_memory=True)

    def match_name_keywords(n, name_keywords):
        return any(b in n for b in name_keywords)

    param_dicts = [
        {
            'params': [p for n, p in model_without_ddp.named_parameters()
                       if not match_name_keywords(n, args.lr_backbone_names)
                       and not match_name_keywords(n, args.lr_linear_proj_names)
                       and p.requires_grad],
            'lr': args.lr,
        },
        {
            'params': [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            'lr': args.lr_backbone,
        },
        {
            'params': [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            'lr': args.lr * args.lr_linear_proj_mult,
        }
    ]
    optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay) if args.sgd else torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.pretrain:
        logging.info('Initialized from the pre-training model: %s', args.pretrain)
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logging.info('%s', msg)
        args.start_epoch = checkpoint.get('epoch', -1) + 1
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args, epoch=args.start_epoch-1)
            _log_test_stats_to_tb(writer, test_stats, args.start_epoch)
            return

    if args.resume:
        checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True) if args.resume.startswith('https') else torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            logging.info('Missing Keys: %s', missing_keys)
        if len(unexpected_keys) > 0:
            logging.info('Unexpected Keys: %s', unexpected_keys)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                logging.info('Override resumed lr_drop with current args.lr_drop=%s', args.lr_drop)
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args, epoch=args.start_epoch-1)
            _log_test_stats_to_tb(writer, test_stats, args.start_epoch)
            if args.output_dir and coco_evaluator is not None:
                utils.save_on_master(coco_evaluator.coco_eval['bbox'].eval, output_dir / 'eval.pth')
            return

    
    if args.freeze_prob_model:
        if isinstance(model_without_ddp.prob_obj_head, torch.nn.ModuleList):
            for obj_head in model_without_ddp.prob_obj_head:
                if hasattr(obj_head, 'freeze_prob_model'):
                    obj_head.freeze_prob_model()
        else:
            if hasattr(model_without_ddp.prob_obj_head, 'freeze_prob_model'):
                model_without_ddp.prob_obj_head.freeze_prob_model()

    logging.info('Start training from epoch %s to %s', args.start_epoch, args.epochs)
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      args.nc_epoch, args.clip_max_norm, writer=writer, args=args)
        lr_scheduler.step()

        test_stats = {}
        coco_evaluator = None
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            should_eval = (epoch + 1) % args.lr_drop == 0 or (epoch % args.eval_every == 0 or epoch == 0 or epoch == 1 or (args.epochs - epoch) < 1)
            if should_eval:
                test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args, epoch=epoch)
                _log_test_stats_to_tb(writer, test_stats, epoch)
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            elif epoch > args.epochs - 6:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            ckpt_args = _build_checkpoint_args(args)
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': ckpt_args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / METRICS_JSONL).open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
            
            record = {
                'epoch': epoch,
                'train_loss': float(train_stats.get('loss', 0.0)) if isinstance(train_stats, dict) else None,
                'train_loss_obj_ll': train_stats.get('loss_obj_ll') if isinstance(train_stats, dict) else None,
                'train_loss_unk': train_stats.get('loss_unk') if isinstance(train_stats, dict) else None,
                'train_loss_obj_pseudo': train_stats.get('loss_obj_pseudo') if isinstance(train_stats, dict) else None,
                'train_loss_obj_neg': train_stats.get('loss_obj_neg') if isinstance(train_stats, dict) else None,
                'train_stat_num_dummy_pos': train_stats.get('stat_num_dummy_pos') if isinstance(train_stats, dict) else None,
                'train_stat_num_dummy_neg': train_stats.get('stat_num_dummy_neg') if isinstance(train_stats, dict) else None,
                'train_stat_num_valid_unmatched': train_stats.get('stat_num_valid_unmatched') if isinstance(train_stats, dict) else None,
                'train_stat_num_pos_candidates': train_stats.get('stat_num_pos_candidates') if isinstance(train_stats, dict) else None,
                'train_stat_num_batch_selected_pos': train_stats.get('stat_num_batch_selected_pos') if isinstance(train_stats, dict) else None,
                'train_stat_pos_thresh_mean': train_stats.get('stat_pos_thresh_mean') if isinstance(train_stats, dict) else None,
                'test_metrics': test_stats.get('metrics', {}) if isinstance(test_stats, dict) else {},
            }
            with (output_dir / METRICS_JSONL).open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
            if args.viz:
                try:
                    _refresh_metric_plots(output_dir)
                except Exception as e:
                    logging.error('Failed to refresh metric plots: %s', e)
            if args.dataset in ['owod', 'owdetr', 'TOWOD', 'OWDETR'] and epoch % args.eval_every == 0 and epoch > 0:
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if 'bbox' in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval['bbox'].eval, output_dir / 'eval' / name)

    if args.exemplar_replay_selection:
        image_sorted_scores = get_exemplar_replay(model, exemplar_selection, device, data_loader_train)
        create_ft_dataset(args, image_sorted_scores)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time %s', total_time_str)
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)