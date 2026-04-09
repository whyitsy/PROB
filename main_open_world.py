# ------------------------------------------------------------------------
# PROB: Probabilistic Objectness for Open World Object Detection
# Official training entry adapted for tensorboard-only logging and UOD methods.
# Refactored to keep plotting / visualization logic in util modules.
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import os
import random
import time
from pathlib import Path

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
from util.plot_metrics import append_json_record, refresh_metric_plots
from models import build_model


METRICS_JSONL = 'metrics_log.jsonl'
STEP_METRICS_JSONL = 'metrics_step.jsonl'


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


def _sum_optional_floats(*vals):
    xs = []
    for v in vals:
        v = _safe_float(v)
        if v is not None:
            xs.append(v)
    return None if len(xs) == 0 else float(sum(xs))


def _safe_div(num, den):
    num = _safe_float(num)
    den = _safe_float(den)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num / den)


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
    parser.add_argument('--viz_score_thresh', default=0.0, type=float, help='display threshold for qualitative pred_* visualizations; set to 0.0 to align with postprocess outputs')
    parser.add_argument('--viz_error_iou_thresh', default=0.5, type=float, help='IoU threshold used to mark qualitative open-world error cases')
    parser.add_argument('--viz_candidate_mode', default='both', type=str, choices=['mining', 'final_unknown', 'both'], help='candidate visualization mode: pseudo-mining aligned, final unknown aligned, or both')
    parser.add_argument('--viz_candidate_topk', default=10, type=int, help='maximum number of candidate boxes shown per image')
    parser.add_argument('--viz_candidate_nms_iou', default=0.6, type=float, help='IoU threshold for deduplicating visualization candidates')
    parser.add_argument('--viz_max_query_points', default=2500, type=int, help='maximum query samples cached for visualization plots')
    parser.add_argument('--viz_max_feature_points', default=2500, type=int, help='maximum feature samples cached for PCA/t-SNE visualizations')
    parser.add_argument('--step_metrics_jsonl', default='metrics_step.jsonl', type=str)
    parser.add_argument('--step_histogram_every', default=100, type=int)
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
    parser.add_argument('--model_type', default='prob', type=str, choices=['prob', 'uod', 'bg'])

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

    parser.add_argument('--uod_known_unk_suppress_init', default=0.5, type=float, help='initial coefficient for suppressing known scores by unknownness')
    parser.add_argument('--uod_unknown_known_suppress_init', default=0.5, type=float, help='initial coefficient for suppressing unknown scores by max-known score')
    parser.add_argument('--uod_odqe_decay_min', default=0.1, type=float, help='minimum ODQE context contribution on the final decoder layer')
    parser.add_argument('--uod_odqe_decay_power', default=1.0, type=float, help='power factor for ODQE layer decay schedule')
    parser.add_argument('--uod_haux_low_obj_coef', default=0.35, type=float, help='aux weight multiplier for low-level objectness pseudo supervision')
    parser.add_argument('--uod_haux_mid_unknown_coef', default=0.45, type=float, help='aux weight multiplier for mid-level unknown supervision')
    parser.add_argument('--uod_haux_high_unknown_coef', default=0.7, type=float, help='aux weight multiplier for high-level unknown supervision')
    parser.add_argument('--uod_haux_high_decorr_coef', default=0.5, type=float, help='aux weight multiplier for high-level decorrelation supervision')

    parser.add_argument('--unk_loss_coef', default=8e-4, type=float, help='weight of matched-known negative unknownness loss')
    parser.add_argument('--uod_pseudo_unk_loss_coef', default=0.5, type=float, help='weight of pseudo-unknown unknownness loss')
    parser.add_argument('--uod_pseudo_obj_loss_coef', default=1, type=float, help='weight of pseudo-positive objectness loss')
    parser.add_argument('--uod_obj_neg_loss_coef', default=1.0, type=float, help='weight of reliable-background objectness negative loss')
    parser.add_argument('--uod_decorr_loss_coef', default=2, type=float, help='weight of prediction decorrelation loss')

    parser.add_argument('--uod_start_epoch', default=3, type=int, help='epoch to start pseudo supervision')
    parser.add_argument('--uod_neg_warmup_epochs', default=2, type=int, help='delay reliable-background objectness negatives after pseudo start')
    
    parser.add_argument('--uod_min_pos_thresh', default=0.08, type=float, help='minimum pseudo-positive energy threshold')
    parser.add_argument('--uod_known_reject_thresh', default=0.15, type=float, help='maximum known score for pseudo-unknown candidates')
    parser.add_argument('--uod_neg_margin', default=0.12, type=float, help='margin for reliable-background objectness negatives in normalized energy space')
    parser.add_argument('--uod_pos_per_img_cap', default=1, type=int, help='max pseudo positives per image, 小于等于0表示不限制')
    parser.add_argument('--uod_neg_per_img', default=2, type=int, help='max reliable-background negatives per image')
    parser.add_argument('--uod_neg_known_max', default=0.12, type=float, help='maximum max-known probability allowed for reliable-background negatives')
    parser.add_argument('--uod_neg_unk_max', default=0.10, type=float, help='maximum unknownness probability allowed for reliable-background negatives')
    parser.add_argument('--uod_neg_max_pseudo_iou', default=0.25, type=float, help='max IoU to selected pseudo positives for reliable-background negatives')
    parser.add_argument('--uod_batch_topk_max', default=8, type=int, help='max pseudo positives selected per batch')
    parser.add_argument('--uod_batch_topk_ratio', default=0.25, type=float, help='dynamic ratio for batch-level pseudo selection')
    parser.add_argument('--uod_max_iou', default=0.2, type=float, help='max IoU with GT for pseudo candidates')
    parser.add_argument('--uod_max_iof', default=0.4, type=float, help='max IoF with GT for pseudo candidates')
    parser.add_argument('--uod_min_area', default=0.002, type=float, help='min normalized area for pseudo/negative candidates')
    parser.add_argument('--uod_max_aspect_ratio', default=5.0, type=float, help='max aspect ratio for pseudo/negative candidates')
    parser.add_argument('--uod_candidate_nms_iou', default=0.6, type=float, help='IoU threshold for deduplicating pseudo-positive candidates per image before batch/global allocation')
    parser.add_argument('--uod_cls_soft_attn_alpha', default=0.8, type=float, help='strength of pseudo-positive classification attenuation')
    parser.add_argument('--uod_cls_soft_attn_min', default=0.1, type=float, help='minimum query weight under classification attenuation')
    parser.add_argument('--uod_pos_unk_min', default=0.05, type=float, help='minimum raw unknownness probability required for pseudo-positive candidates')
    parser.add_argument('--uod_postprocess_unknown_ratio', default=0.95, type=float, help='query-wise routing ratio for choosing unknown over best-known in postprocess')
    parser.add_argument('--uod_postprocess_unknown_scale', default=15.0, type=float, help='multiplicative scale for final unknown score in postprocess and visualization')
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

def _sum_optional_floats(*vals):
    xs = []
    for v in vals:
        v = _safe_float(v)
        if v is not None:
            xs.append(v)
    return None if len(xs) == 0 else float(sum(xs))

def _safe_div(num, den):
    num = _safe_float(num)
    den = _safe_float(den)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num / den)


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

    logging.info('git:\n  %s\n', utils.get_sha())
    if args.frozen_weights is not None:
        assert args.masks, 'Frozen training is meant for segmentation only'

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, exemplar_selection = build_model(args, mode=args.model_type)
    model.to(device)
    # 打印指定模型指定行数的参数信息，方便调试    
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     if 303 < idx <= 307:  
    #         logging.info(f"[{idx}]: {name}")

    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     if 318 < idx <= 335:  
    #         logging.info(f"[{idx}]: {name}")

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
                
        if hasattr(model_without_ddp, 'known_energy_head'):
            if isinstance(model_without_ddp.known_energy_head, torch.nn.ModuleList):
                for known_head in model_without_ddp.known_energy_head:
                    if hasattr(known_head, 'freeze_prob_model'):
                        known_head.freeze_prob_model()
            else:
                if hasattr(model_without_ddp.known_energy_head, 'freeze_prob_model'):
                    model_without_ddp.known_energy_head.freeze_prob_model()

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


        if args.output_dir and utils.is_main_process():
            metric_dict = test_stats.get('metrics', {}) if isinstance(test_stats, dict) else {}

            record = {
                'epoch': epoch,
                'n_parameters': n_parameters,

                # ---------- train basic ----------
                'train_loss': _safe_float(train_stats.get('loss')) if isinstance(train_stats, dict) else None,
                'train_lr': _safe_float(train_stats.get('lr')) if isinstance(train_stats, dict) else None,
                'train_grad_norm': _safe_float(train_stats.get('grad_norm')) if isinstance(train_stats, dict) else None,
                'train_class_error': _safe_float(train_stats.get('class_error')) if isinstance(train_stats, dict) else None,

                # ---------- base losses ----------
                'train_loss_ce': _safe_float(train_stats.get('loss_ce')) if isinstance(train_stats, dict) else None,
                'train_loss_ce_unscaled': _safe_float(train_stats.get('loss_ce_unscaled')) if isinstance(train_stats, dict) else None,
                'train_loss_bbox': _safe_float(train_stats.get('loss_bbox')) if isinstance(train_stats, dict) else None,
                'train_loss_bbox_unscaled': _safe_float(train_stats.get('loss_bbox_unscaled')) if isinstance(train_stats, dict) else None,
                'train_loss_giou': _safe_float(train_stats.get('loss_giou')) if isinstance(train_stats, dict) else None,
                'train_loss_giou_unscaled': _safe_float(train_stats.get('loss_giou_unscaled')) if isinstance(train_stats, dict) else None,
                'train_loss_obj_ll': _safe_float(train_stats.get('loss_obj_ll')) if isinstance(train_stats, dict) else None,
                'train_loss_obj_ll_unscaled': _safe_float(train_stats.get('loss_obj_ll_unscaled')) if isinstance(train_stats, dict) else None,

                # ---------- unknownness ----------
                'train_loss_unk_known': _safe_float(train_stats.get('loss_unk_known')) if isinstance(train_stats, dict) else None,
                'train_loss_unk_known_unscaled': _safe_float(train_stats.get('loss_unk_known_unscaled')) if isinstance(train_stats, dict) else None,
                'train_loss_unk_pseudo': _safe_float(train_stats.get('loss_unk_pseudo')) if isinstance(train_stats, dict) else None,
                'train_loss_unk_pseudo_unscaled': _safe_float(train_stats.get('loss_unk_pseudo_unscaled')) if isinstance(train_stats, dict) else None,

                'train_loss_unk': _sum_optional_floats(
                    train_stats.get('loss_unk_known'),
                    train_stats.get('loss_unk_pseudo'),
                ) if isinstance(train_stats, dict) else None,

                'train_loss_unk_unscaled': _sum_optional_floats(
                    train_stats.get('loss_unk_known_unscaled'),
                    train_stats.get('loss_unk_pseudo_unscaled'),
                ) if isinstance(train_stats, dict) else None,

                # ---------- pseudo objectness ----------
                'train_loss_obj_pseudo': _safe_float(train_stats.get('loss_obj_pseudo')) if isinstance(train_stats, dict) else None,
                'train_loss_obj_pseudo_unscaled': _safe_float(train_stats.get('loss_obj_pseudo_unscaled')) if isinstance(train_stats, dict) else None,
                'train_loss_obj_neg': _safe_float(train_stats.get('loss_obj_neg')) if isinstance(train_stats, dict) else None,
                'train_loss_obj_neg_unscaled': _safe_float(train_stats.get('loss_obj_neg_unscaled')) if isinstance(train_stats, dict) else None,

                # ---------- chapter 4 ----------
                'train_loss_decorr': _safe_float(train_stats.get('loss_decorr')) if isinstance(train_stats, dict) else None,
                'train_loss_decorr_unscaled': _safe_float(train_stats.get('loss_decorr_unscaled')) if isinstance(train_stats, dict) else None,
                'train_known_unk_suppress_coeff': _safe_float(train_stats.get('known_unk_suppress_coeff')) if isinstance(train_stats, dict) else None,
                'train_unknown_known_suppress_coeff': _safe_float(train_stats.get('unknown_known_suppress_coeff')) if isinstance(train_stats, dict) else None,
                'train_gate_mean': _safe_float(train_stats.get('gate_mean')) if isinstance(train_stats, dict) else None,

                # ---------- pseudo mining stats ----------
                'train_stat_num_dummy_pos': _safe_float(train_stats.get('stat_num_dummy_pos')) if isinstance(train_stats, dict) else None,
                'train_stat_num_dummy_neg': _safe_float(train_stats.get('stat_num_dummy_neg')) if isinstance(train_stats, dict) else None,
                'train_stat_num_valid_unmatched': _safe_float(train_stats.get('stat_num_valid_unmatched')) if isinstance(train_stats, dict) else None,
                'train_stat_num_pos_candidates': _safe_float(train_stats.get('stat_num_pos_candidates')) if isinstance(train_stats, dict) else None,
                'train_stat_num_neg_candidates': _safe_float(train_stats.get('stat_num_neg_candidates')) if isinstance(train_stats, dict) else None,
                'train_stat_num_batch_selected_pos': _safe_float(train_stats.get('stat_num_batch_selected_pos')) if isinstance(train_stats, dict) else None,
                'train_stat_pos_thresh_mean': _safe_float(train_stats.get('stat_pos_thresh_mean')) if isinstance(train_stats, dict) else None,

                # ---------- pseudo efficiency ----------
                'train_pseudo_selection_ratio': _safe_div(
                    train_stats.get('stat_num_batch_selected_pos'),
                    train_stats.get('stat_num_valid_unmatched')
                ) if isinstance(train_stats, dict) else None,

                'train_pseudo_accept_ratio': _safe_div(
                    train_stats.get('stat_num_batch_selected_pos'),
                    train_stats.get('stat_num_pos_candidates')
                ) if isinstance(train_stats, dict) else None,

                # ---------- eval metrics ----------
                'test_metrics': metric_dict,
            }

            append_json_record(output_dir / METRICS_JSONL, record)

            if args.viz:
                try:
                    refresh_metric_plots(output_dir, metrics_filename=METRICS_JSONL, step_metrics_filename=args.step_metrics_jsonl)
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
    try:
        main(args)
    except Exception as e:
        logging.error('An error occurred during execution: %s', e, exc_info=True)
        raise
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            logging.info("Distributed process group destroyed successfully.")
    