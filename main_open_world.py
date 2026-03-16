# ------------------------------------------------------------------------
# PROB: Probabilistic Objectness for Open World Object Detection 
# Orr Zohar, Jackson Wang, Serena Yeung
# -----------------------------------------------------------------------
# Modified from OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import OWDetection
from engine import evaluate, train_one_epoch, get_exemplar_replay
from models import build_model
import wandb
import logging
from util.log import setup_logging


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
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--lr_drop', default=35, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true') # 默认未开启
    parser.add_argument('--two_stage', default=True, action='store_true') # 默认没有开启
    parser.add_argument('--masks', default=False, action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone', default='dino_resnet50', type=str, help="Name of the convolutional backbone to use")

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # dataset parameters
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--eval_every', default=5, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
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
    # model config
    parser.add_argument('--model_type', default='prob', type=str)
    
    # logging
    parser.add_argument('--wandb_name', default='', type=str)
    parser.add_argument('--wandb_project', default='PROB_OWOD', type=str)
    
    # model hyperparameters
    parser.add_argument('--obj_loss_coef', default=100, type=float)
    parser.add_argument('--obj_temp', default=1, type=float)
    parser.add_argument('--freeze_prob_model', default=False, action='store_true', help='freeze model probabistic estimation')

    
    # Exemplar replay selection
    parser.add_argument('--num_inst_per_class', default=50, type=int, help="number of instances per class")
    parser.add_argument('--exemplar_replay_selection', default=False, action='store_true', help='use learned exemplar selection')
    parser.add_argument('--exemplar_replay_max_length', default=1e10, type=int, help="max number of images that can be saves")
    parser.add_argument('--exemplar_replay_dir', default='', type=str, help="directory of exemplar replay txt files")
    parser.add_argument('--exemplar_replay_prev_file', default='', type=str, help="path to previous ft file")
    parser.add_argument('--exemplar_replay_cur_file', default='', type=str, help="path to current ft file")
    parser.add_argument('--exemplar_replay_random', default=False, action='store_true', help='make selection random')
    
    # 伪样本挖掘
    parser.add_argument('--enable_unk_label_obj', default=False, action='store_true', help='使用基于物体性分数的自适应伪标签筛选')
    parser.add_argument('--use_valid_mask', default=False, action='store_true', help='使用有效掩码')
    parser.add_argument('--unk_label_obj_score_thresh', default=0.8, type=float, help='自适应筛选阈值, 基于匹配上的query的obj的平均值乘以该系数')
    parser.add_argument('--unk_label_start_epoch', default=2, type=int, help='从哪个epoch开始使用基于物体性分数的自适应伪标签筛选')
    ## 目标性预测的提前终止 (ETOP, Early Termination of Objectness Prediction)
    parser.add_argument('--etop', default=False, action='store_true', help='启用目标性预测的提前终止')
    parser.add_argument('--etop_layer', default=1, type=int, help='目标性预测的提前终止层')
    ## 任务解耦查询初始化 (TDQI, Task-Decoupled Query Initialization)
    parser.add_argument('--tdqi', default=False, action='store_true', help='使用任务解耦查询初始化')
    parser.add_argument('--tdqi_query_num', default=20, type=int, help='负责已知目标的查询数量, 默认值参考decoupled PROB论文')
    # CLIP特征融合
    parser.add_argument('--use_feature_align', default=False, action='store_true', help='使用CLIP特征对齐')
    parser.add_argument('--use_vlm_distill', default=False, action='store_true', help='使用基于视觉语言模型的蒸馏')
    parser.add_argument('--vlm_tau', default=0.1, type=float, help='多模态置信度（$\omega$）计算时的温度系数（Temperature）。')
    parser.add_argument('--clip_text_features', default=None, type=str, help='CLIP文本特征的路径')
    parser.add_argument('--clip_dim', default=512, type=int, help='CLIP 特征的维度')
    parser.add_argument('--align_loss_coef', default=1.0, type=float, help='对齐损失权重系数')
    parser.add_argument('--pred_per_im', default=100, type=int, help='每张图片预测的框数')
    
    
    
    return parser

def main(args):
    if len(args.wandb_project)>0:
        if len(args.wandb_name)>0:
            wandb.init(mode="offline", project=args.wandb_project, entity="kakoyo-chengdu-university-of-technology", group=args.wandb_name)
        else:
            wandb.init(mode="offline", project=args.wandb_project, entity="kakoyo-chengdu-university-of-technology")
        wandb.config = args
    #else:
    #    wandb=None

    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    setup_logging(output=args.output_dir, distributed_rank=utils.get_rank())
    logging.info("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # print(args)
    logging.info("args:\n  {}\n".format(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dataset_train, dataset_val, class_names = get_datasets(args)
    args.class_names = class_names
    model, criterion, postprocessors, exemplar_selection = build_model(args, mode = args.model_type)
    model.to(device)

    model_without_ddp = model
    logging.info("model_without_ddp: {}".format(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('number of params: {}'.format(n_parameters))


    
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
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = dataset_val

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.pretrain:
        # print('Initialized from the pre-training model')
        logging.info('Initialized from the pre-training model')
        checkpoint = torch.load(args.pretrain, map_location='cpu', weights_only=False) # 需要使用自定义的epoch对象, 需要关闭安全模式。不然加载不了epoch信息。
        state_dict = checkpoint['model']
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logging.info("InCompatible keys: {}".format(msg))
        args.start_epoch = checkpoint['epoch'] + 1
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
            return
        
        
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) # resume需要加载自定义的对象, 需要关闭安全模式。不然加载不了lr、optimizer等信息。
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        # 只在主进程打印
        if utils.is_main_process(): 
            if len(missing_keys) > 0:
                logging.info('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                logging.info('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # 只在主进程打印优化器信息
            # if utils.is_main_process():
            #     logging.info('Optimizer param groups: {}'.format(optimizer.param_groups))
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                 # 警告也只在主进程打印
                if utils.is_main_process():
                    logging.info('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if (not args.eval and not args.viz and args.dataset in ['coco', 'voc']):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return
        
    if args.freeze_prob_model:           
        if isinstance(model_without_ddp.prob_obj_head, torch.nn.ModuleList):
            for obj_head in model_without_ddp.prob_obj_head:
                obj_head.freeze_prob_model()
        else:
            model_without_ddp.prob_obj_head.freeze_prob_model()
            
        obj_bn_mean_before=model_without_ddp.prob_obj_head[0].objectness_bn.running_mean
    
    logging.info(f'Start training from epoch {args.start_epoch} to {args.epochs}')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.nc_epoch, args.clip_max_norm, wandb, args)
            
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch % args.eval_every == 0 or epoch == 0 or epoch == 1 or (args.epochs-epoch)<1):
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args)
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                if wandb is not None:
                    test_stats["metrics"]['epoch']=epoch
                    wandb.log({str(key): val for key, val in test_stats["metrics"].items()})
            elif epoch > args.epochs-6:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                
            else:
                 test_stats = {}
                    
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if args.dataset in ['owod', 'owdetr'] and epoch % args.eval_every == 0 and epoch > 0:
                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
                            
            
    if args.exemplar_replay_selection:
        image_sorted_scores = get_exemplar_replay(model,exemplar_selection, device, data_loader_train)
        create_ft_dataset(args, image_sorted_scores)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))
    logging.info('Training time {}'.format(total_time_str))
    return

def get_datasets(args):
    # print(args.dataset)
    logging.info(f"Dataset: {args.dataset}")
    
    train_set = args.train_set
    test_set = args.test_set
    dataset_train = OWDetection(args, args.data_root, image_set=args.train_set, transforms=make_coco_transforms(args.train_set), dataset = args.dataset)
    dataset_val = OWDetection(args, args.data_root, image_set=args.test_set, dataset = args.dataset, transforms=make_coco_transforms(args.test_set))

    # print(args.train_set)
    # print(args.test_set)
    # print(dataset_train)
    # print(dataset_val)
    logging.info(f"Train dataset: {dataset_train}, Test dataset: {dataset_val}")
    logging.info(f"dataset_train: {dataset_train}")
    logging.info(f"dataset_val: {dataset_val}")

    return dataset_train, dataset_val, dataset_train.CLASS_NAMES


def create_ft_dataset(args, image_sorted_scores):
    logging.info(f'found a total of {len(image_sorted_scores.keys())} images')
    tmp_dir=args.data_root +'/ImageSets/'+args.dataset+"/"+args.exemplar_replay_dir+"/"
    

    class_sorted_scores={}
    imgs_per_class={}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS):
        class_sorted_scores[str(i)]=[]
        imgs_per_class[str(i)]=[]

    for k,v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            class_sorted_scores[str(v['labels'][j])].append(v['scores'][j])


    class_threshold={}
    for i in range(args.PREV_INTRODUCED_CLS, args.CUR_INTRODUCED_CLS+args.PREV_INTRODUCED_CLS):
        tmp=np.array(class_sorted_scores[str(i)])
        tmp.sort()
        tmp = torch.Tensor(tmp)
        if len(tmp)>args.num_inst_per_class and not args.exemplar_replay_random:
            max_val = tmp[-args.num_inst_per_class//2]
            min_val = tmp[args.num_inst_per_class//2]
        else:
            if args.exemplar_replay_random:
                logging.info('using random exemplar selection')
            else:
                logging.info(f'only found {len(tmp)} imgs in class {i}')
            max_val = tmp.min()
            min_val = tmp.max()
            
        class_threshold[str(i)]=(min_val, max_val)

    save_imgs = []    
    for k,v in image_sorted_scores.items():
        for j in range(len(v['labels'])):
            label = str(v['labels'][j])
            if (v['scores'][j] <= class_threshold[label][0].numpy() or v['scores'][j] >= class_threshold[label][1].numpy()) and (len(imgs_per_class[label])<=args.num_inst_per_class+2):
                save_imgs.append(k)
                imgs_per_class[label].append(k)
                        
    logging.info(f'found {len(np.unique(save_imgs))} images in run')
    if len(args.exemplar_replay_prev_file)>0:
        previous_ft = open(tmp_dir+args.exemplar_replay_prev_file,'r').read().splitlines()
        save_imgs+=previous_ft
        
    save_imgs=np.unique(save_imgs)
    np.random.shuffle(save_imgs)
    if len(save_imgs)> args.exemplar_replay_max_length:
        save_imgs=save_imgs[:args.exemplar_replay_max_length]
    
    os.makedirs(tmp_dir, exist_ok=True)
    with open(tmp_dir+args.exemplar_replay_cur_file, 'w') as f:
        for line in save_imgs:
            f.write(line)
            f.write('\n')
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)