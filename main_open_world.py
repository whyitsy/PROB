# ------------------------------------------------------------------------
# PROB / UOD official training entry.
# This refactor keeps the linear execution order in one file, removes stale
# segmentation / NC arguments, and moves visualization-specific policy into
# visual/viz_config.py.
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat # 美化输出
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets.samplers as samplers
import util.misc as utils
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from engine import evaluate, get_exemplar_replay, train_one_epoch
from models import build_model
from util.log import setup_logging


@dataclass
class VizContext:
    output_dir: Optional[Path]
    tb_writer: Optional[SummaryWriter]
    checkpoint_dir_name: str = 'train/checkpoints'
    tensorboard_dir_name: str = 'train/tensorboard'

    @classmethod
    def from_args(cls, args):
        output_dir = Path(args.output_dir) if getattr(args, 'output_dir', None) else None
        viz_ctx = cls(output_dir=output_dir, tb_writer=None)
        viz_ctx._build_output_structure()
        if bool(getattr(args, 'viz', False)) and utils.is_main_process() and viz_ctx.tensorboard_dir is not None:
            run_name = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
            log_dir = viz_ctx.tensorboard_dir / run_name
            log_dir.mkdir(parents=True, exist_ok=True)
            viz_ctx.tb_writer = SummaryWriter(log_dir=str(log_dir))
            logging.info('TensorBoard log dir: %s', log_dir)
        return viz_ctx

    @property
    def checkpoint_dir(self):
        return None if self.output_dir is None else self.output_dir / self.checkpoint_dir_name

    @property
    def tensorboard_dir(self):
        return None if self.output_dir is None else self.output_dir / self.tensorboard_dir_name

    def _build_output_structure(self):
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.tensorboard_dir is not None:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None


def write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch):
    if viz_ctx is None or viz_ctx.tb_writer is None or not isinstance(eval_stats, dict):
        return
    for key, value in eval_stats.items():
        if isinstance(value, (int, float)):
            viz_ctx.tb_writer.add_scalar(f'eval/{key}', value, epoch)


def write_epoch_reports(viz_ctx, epoch, train_stats, eval_stats, num_trainable_parameters, eval_evaluator=None, args=None):
    del train_stats, num_trainable_parameters, eval_evaluator, args
    write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch)


def _sanitize_for_checkpoint(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_checkpoint(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _sanitize_for_checkpoint(value) for key, value in obj.items()}
    if isinstance(obj, argparse.Namespace):
        return {key: _sanitize_for_checkpoint(value) for key, value in vars(obj).items() if not key.startswith('_')}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return repr(obj)


def get_args_parser():
    parser = argparse.ArgumentParser('PROB / UOD Detector', add_help=False)

    # ---------------- Basic optimization ----------------
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--eval_batch_size', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=41, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--sgd', action='store_true')

    # ---------------- Deformable DETR backbone / transformer ----------------
    parser.add_argument('--with_box_refine', action='store_true')
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--backbone', default='dino_resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)

    # ---------------- Matcher / base detection loss ----------------
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # ---------------- Runtime ----------------
    parser.add_argument('--output_dir', default='/mnt/data/kky/output/PROB/exps/MOWODB/TEMP', help='directory for checkpoints, logs and visual outputs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', help='resume full training state from checkpoint')
    parser.add_argument('--eval_checkpoint', help='checkpoint path used only for eval mode (model weights only)')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true', help='enable TensorBoard and qualitative/statistical visualization')
    parser.add_argument('--eval_every', default=5, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true')

    # ---------------- Open-world dataset ----------------
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=20, type=int)
    parser.add_argument('--train_set', default='owod_t1_train', help='training txt split name')
    parser.add_argument('--test_set', default='owod_all_task_test', help='evaluation txt split name')
    parser.add_argument('--num_classes', default=81, type=int)
    parser.add_argument('--dataset', default='TOWOD', help='one of {TOWOD, OWDETR, VOC2007}')
    parser.add_argument('--data_root', default='/mnt/data/kky/datasets/owdetr/data/OWOD', type=str)

    # ---------------- Model mode ----------------
    parser.add_argument('--model_type', default='prob', type=str, choices=['prob', 'uod'])
    parser.add_argument('--obj_loss_coef', default=8e-4, type=float)
    parser.add_argument('--obj_temp', default=1.3, type=float)
    parser.add_argument('--uod_known_temp', default=1.3, type=float, help='knownness-energy temperature used by UOD scoring and visualization')
    parser.add_argument('--freeze_prob_model', default=False, action='store_true')

    # ---------------- Exemplar replay ----------------
    parser.add_argument('--num_inst_per_class', default=50, type=int)
    parser.add_argument('--exemplar_replay_selection', default=False, action='store_true')
    parser.add_argument('--exemplar_replay_max_length', default=int(1e10), type=int)
    parser.add_argument('--exemplar_replay_dir', type=str)
    parser.add_argument('--exemplar_replay_prev_file', type=str)
    parser.add_argument('--exemplar_replay_cur_file', type=str)
    parser.add_argument('--exemplar_replay_random', default=False, action='store_true')

    # ---------------- UOD ----------------
    parser.add_argument('--uod_enable_unknown', default=False, action='store_true')
    parser.add_argument('--uod_enable_pseudo', default=False, action='store_true')
    parser.add_argument('--uod_enable_batch_dynamic', default=False, action='store_true')
    parser.add_argument('--uod_enable_decorr', default=False, action='store_true')
    parser.add_argument('--uod_enable_cls_soft_attn', default=False, action='store_true')
    parser.add_argument('--uod_enable_odqe', default=False, action='store_true')

    parser.add_argument('--uod_odqe_decay_min', default=0.1, type=float)
    parser.add_argument('--uod_odqe_decay_power', default=1.0, type=float)
    parser.add_argument('--uod_haux_low_obj_coef', default=0.35, type=float)
    parser.add_argument('--uod_haux_mid_unknown_coef', default=0.45, type=float)
    parser.add_argument('--uod_haux_high_unknown_coef', default=0.7, type=float)
    parser.add_argument('--uod_haux_high_decorr_coef', default=0.5, type=float)

    parser.add_argument('--unk_loss_coef', default=8e-4, type=float)
    parser.add_argument('--uod_pseudo_unk_loss_coef', default=0.5, type=float)
    parser.add_argument('--uod_pseudo_obj_loss_coef', default=1.0, type=float)
    parser.add_argument('--uod_obj_neg_loss_coef', default=1.0, type=float)
    parser.add_argument('--uod_decorr_loss_coef', default=2.0, type=float)
    parser.add_argument('--uod_pseudo_bbox_loss_coef', default=3, type=float)
    parser.add_argument('--uod_pseudo_giou_loss_coef', default=1, type=float)

    parser.add_argument('--uod_start_epoch', default=12, type=int)
    parser.add_argument('--uod_neg_warmup_epochs', default=2, type=int)
    parser.add_argument('--uod_min_pos_thresh', default=0.08, type=float)
    parser.add_argument('--uod_known_reject_thresh', default=0.25, type=float)
    parser.add_argument('--uod_neg_margin', default=0.12, type=float)
    parser.add_argument('--uod_pos_per_img_cap', default=0, type=int)
    parser.add_argument('--uod_neg_per_img', default=2, type=int)
    parser.add_argument('--uod_neg_known_max', default=0.12, type=float)
    parser.add_argument('--uod_neg_unk_max', default=0.10, type=float)
    parser.add_argument('--uod_neg_max_pseudo_iou', default=0.25, type=float)
    parser.add_argument('--uod_batch_topk_max', default=16, type=int)
    parser.add_argument('--uod_batch_topk_ratio', default=0.25, type=float)
    parser.add_argument('--uod_max_iou', default=0.2, type=float)
    parser.add_argument('--uod_max_iof', default=0.4, type=float)
    parser.add_argument('--uod_min_area', default=0.02, type=float)
    parser.add_argument('--uod_min_side', default=0.05, type=float)
    parser.add_argument('--uod_max_aspect_ratio', default=10.0, type=float)
    parser.add_argument('--uod_candidate_nms_iou', default=0.6, type=float)
    parser.add_argument('--uod_cls_soft_attn_alpha', default=0.5, type=float)
    parser.add_argument('--uod_cls_soft_attn_min', default=0.25, type=float)
    parser.add_argument('--uod_pos_unk_min', default=0.05, type=float)
    parser.add_argument('--uod_postprocess_unknown_scale', default=20.0, type=float)
    return parser


def build_datasets(args):
    logging.info('Dataset: %s', args.dataset)
    train_dataset = OWDetection( args, args.data_root, image_set=args.train_set, transforms=make_coco_transforms(args.train_set), dataset=args.dataset)
    eval_dataset = OWDetection( args, args.data_root, image_set=args.test_set, transforms=make_coco_transforms(args.test_set), dataset=args.dataset)
    logging.info('Train split: %s', args.train_set)
    logging.info('Eval split: %s', args.test_set)
    return train_dataset, eval_dataset


def create_ft_dataset(args, image_sorted_scores):
    logging.info('Collected exemplar replay scores for %s images', len(image_sorted_scores.keys()))
    replay_output_dir = Path(args.data_root) / 'ImageSets' / args.dataset / args.exemplar_replay_dir
    replay_output_dir.mkdir(parents=True, exist_ok=True)

    per_class_scores = {}
    per_class_images = {}
    current_class_range = range(args.PREV_INTRODUCED_CLS, args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS)
    for class_index in current_class_range:
        per_class_scores[str(class_index)] = []
        per_class_images[str(class_index)] = []

    for item in image_sorted_scores.values():
        for label, score in zip(item['labels'], item['scores']):
            per_class_scores[str(label)].append(score)

    class_thresholds = {}
    for class_index in current_class_range:
        values = np.array(per_class_scores[str(class_index)])
        values.sort()
        values = torch.as_tensor(values)
        if len(values) > args.num_inst_per_class and not args.exemplar_replay_random:
            max_value = values[-args.num_inst_per_class // 2]
            min_value = values[args.num_inst_per_class // 2]
        else:
            if args.exemplar_replay_random:
                logging.info('Using random exemplar replay selection')
            else:
                logging.info('Only found %s images for class %s', len(values), class_index)
            max_value = values.min() if len(values) > 0 else torch.tensor(0.0)
            min_value = values.max() if len(values) > 0 else torch.tensor(0.0)
        class_thresholds[str(class_index)] = (min_value, max_value)

    selected_images = []
    for image_id, item in image_sorted_scores.items():
        for label, score in zip(item['labels'], item['scores']):
            label_key = str(label)
            min_value, max_value = class_thresholds[label_key]
            if (score <= min_value.numpy() or score >= max_value.numpy()) and len(per_class_images[label_key]) <= args.num_inst_per_class + 2:
                selected_images.append(image_id)
                per_class_images[label_key].append(image_id)

    if args.exemplar_replay_prev_file:
        previous_file = replay_output_dir / args.exemplar_replay_prev_file
        if previous_file.exists():
            selected_images += previous_file.read_text(encoding='utf-8').splitlines()

    selected_images = np.unique(selected_images)
    np.random.shuffle(selected_images)
    if len(selected_images) > args.exemplar_replay_max_length:
        selected_images = selected_images[:args.exemplar_replay_max_length]

    current_file = replay_output_dir / args.exemplar_replay_cur_file
    with current_file.open('w', encoding='utf-8') as file:
        for image_id in selected_images:
            file.write(str(image_id))
            file.write('\n')


def main(args):
    utils.init_distributed_mode(args)
    viz_ctx = VizContext.from_args(args)
    output_dir = viz_ctx.output_dir
    setup_logging(output=args.output_dir, distributed_rank=utils.get_rank(), abbrev_name='UOD')
    logging.info('Arguments:\n%s', pformat(vars(args), indent=2, width=100, compact=True, sort_dicts=False)) # vars() 将 Namespace 转为 dict

    try:
        device = torch.device(args.device)
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, criterion, postprocessors, exemplar_selection = build_model(args, mode=args.model_type)
        model.to(device)
        model_without_ddp = model
        logging.info('%s', model_without_ddp)
        num_trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        logging.info('Number of trainable parameters: %s', f'{num_trainable_parameters:,}')

        train_dataset, eval_dataset = build_datasets(args)

        if args.distributed:
            if args.cache_mode:
                train_sampler = samplers.NodeDistributedSampler(train_dataset)
                eval_sampler = samplers.NodeDistributedSampler(eval_dataset, shuffle=False)
            else:
                train_sampler = samplers.DistributedSampler(train_dataset)
                eval_sampler = samplers.DistributedSampler(eval_dataset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)

        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
        eval_loader = DataLoader(eval_dataset, args.eval_batch_size, sampler=eval_sampler, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

        def match_name_keywords(parameter_name, keyword_list):
            return any(keyword in parameter_name for keyword in keyword_list)

        parameter_groups = [
            {
                'params': [parameter for name, parameter in model_without_ddp.named_parameters() if not match_name_keywords(name, args.lr_backbone_names) and not match_name_keywords(name, args.lr_linear_proj_names) and parameter.requires_grad],
                'lr': args.lr,
            },
            {
                'params': [parameter for name, parameter in model_without_ddp.named_parameters() if match_name_keywords(name, args.lr_backbone_names) and parameter.requires_grad],
                'lr': args.lr_backbone,
            },
            {
                'params': [parameter for name, parameter in model_without_ddp.named_parameters() if match_name_keywords(name, args.lr_linear_proj_names) and parameter.requires_grad],
                'lr': args.lr * args.lr_linear_proj_mult,
            },
        ]

        if args.sgd:
            optimizer = torch.optim.SGD(parameter_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        if args.freeze_prob_model and hasattr(model_without_ddp, 'prob_obj_head'):
            probability_head = model_without_ddp.prob_obj_head
            if isinstance(probability_head, torch.nn.ModuleList):
                for module in probability_head:
                    if hasattr(module, 'freeze_prob_model'):
                        module.freeze_prob_model()
            elif hasattr(probability_head, 'freeze_prob_model'):
                probability_head.freeze_prob_model()

            if hasattr(model_without_ddp, 'known_energy_head'):
                knownness_head = model_without_ddp.known_energy_head
                if isinstance(knownness_head, torch.nn.ModuleList):
                    for module in knownness_head:
                        if hasattr(module, 'freeze_prob_model'):
                            module.freeze_prob_model()
                elif hasattr(knownness_head, 'freeze_prob_model'):
                    knownness_head.freeze_prob_model()

        if args.eval:
            logging.info('Evaluating from checkpoint: %s', args.eval_checkpoint)
            checkpoint = (
                torch.hub.load_state_dict_from_url(args.eval_checkpoint, map_location='cpu', check_hash=True)
                if isinstance(args.eval_checkpoint, str) and args.eval_checkpoint.startswith('https')
                else torch.load(args.eval_checkpoint, map_location='cpu')
            )
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            unexpected_keys = [key for key in unexpected_keys if not (key.endswith('total_params') or key.endswith('total_ops'))]
            if missing_keys:
                logging.info('Missing keys while loading eval checkpoint: %s', missing_keys)
            if unexpected_keys:
                logging.info('Unexpected keys while loading eval checkpoint: %s', unexpected_keys)
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1

            eval_stats, _ = evaluate(model, criterion, postprocessors, eval_loader, eval_dataset, device, args, viz_ctx=viz_ctx, epoch=args.start_epoch)
            write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, args.start_epoch)
        else:
            if args.resume:
                logging.info('Resuming from checkpoint: %s', args.resume)
                checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True) if args.resume.startswith('https') else torch.load(args.resume, map_location='cpu')
                missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                unexpected_keys = [key for key in unexpected_keys if not (key.endswith('total_params') or key.endswith('total_ops'))]
                if missing_keys:
                    logging.info('Missing keys while resuming: %s', missing_keys)
                if unexpected_keys:
                    logging.info('Unexpected keys while resuming: %s', unexpected_keys)

                logging.info('epoch resumed from %d', checkpoint['epoch'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                    import copy
                    optimizer_param_groups = copy.deepcopy(optimizer.param_groups)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    for current_group, original_group in zip(optimizer.param_groups, optimizer_param_groups):
                        current_group['lr'] = original_group['lr']
                        current_group['initial_lr'] = original_group['initial_lr']
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
                    lr_scheduler.step(lr_scheduler.last_epoch)

            logging.info('Start training from epoch %s to %s', args.start_epoch, args.epochs)
            training_start_time = time.time()

            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch,  max_norm=args.clip_max_norm, viz_ctx=viz_ctx, args=args)
                lr_scheduler.step()

                eval_stats = {}
                eval_evaluator = None
                checkpoint_paths = []
                if output_dir is not None:
                    checkpoint_paths.append(viz_ctx.checkpoint_dir / 'checkpoint_latest.pth')
                    should_run_evaluation = ((epoch + 1) % args.lr_drop == 0) or (epoch == 0) or (epoch == 1) or ((epoch + 1) % args.eval_every == 0 or (epoch == args.epochs - 1))
                    if should_run_evaluation:
                        eval_stats, eval_evaluator = evaluate(model, criterion, postprocessors, eval_loader, eval_dataset, device, args, viz_ctx=viz_ctx, epoch=epoch)
                        write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch)
                        checkpoint_paths.append(viz_ctx.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth')
                    elif epoch > args.epochs - 6:
                        checkpoint_paths.append(viz_ctx.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth')

                    checkpoint_args = _sanitize_for_checkpoint(args) # 处理args中可能存在的不可序列化对象.
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': checkpoint_args,
                        }, checkpoint_path)

                write_epoch_reports(viz_ctx=viz_ctx, epoch=epoch, train_stats=train_stats, eval_stats=eval_stats, 
                                    num_trainable_parameters=num_trainable_parameters,
                                    eval_evaluator=eval_evaluator, args=args)

            logging.info("Training finished. Releasing training resources to free VRAM...")

            if 'optimizer' in locals():
                del optimizer
            if 'lr_scheduler' in locals():
                del lr_scheduler
                
            if 'outputs' in locals():
                del outputs
            if 'loss_dict' in locals():
                del loss_dict

            torch.cuda.empty_cache()

            logging.info("Resources released. Proceeding to exemplar replay/evaluation.")

            if args.exemplar_replay_selection:
                logging.info('running with exemplar_replay_selection')
                exemplar_scores = get_exemplar_replay(model, exemplar_selection, device, train_loader)
                create_ft_dataset(args, exemplar_scores)

            total_training_time = time.time() - training_start_time
            logging.info('Training time %s', str(datetime.timedelta(seconds=int(total_training_time))))
    finally:
        viz_ctx.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PROB / UOD training and evaluation script', parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        main(parsed_args)
    except Exception as error:
        logging.error('An error occurred during execution: %s', error, exc_info=True)
        raise
    finally:
        import torch.distributed as dist
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logging.info('Distributed process group destroyed successfully.')
        except Exception as cleanup_error:
            logging.error('Failed to destroy process group cleanly: %s', cleanup_error, exc_info=True)
