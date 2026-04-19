import logging

import torch

from util.visual.training import (
    append_json_record,
    build_eval_epoch_record,
    build_train_epoch_record,
    write_eval_scalars_to_tensorboard,
)
from visual.metrics_plotter import refresh_metric_plots


def _refresh_metric_plots(viz_ctx):
    if viz_ctx is None or not viz_ctx.visualization_enabled or viz_ctx.output_dir is None:
        return
    try:
        refresh_metric_plots(
            viz_ctx.output_dir,
            train_epoch_metrics_file=viz_ctx.train_epoch_metrics_file,
            eval_epoch_metrics_file=viz_ctx.eval_epoch_metrics_file,
            train_step_metrics_file=viz_ctx.train_step_metrics_file,
        )
    except Exception as error:
        logging.error('Failed to refresh metric plots: %s', error)


def _write_bbox_eval_artifacts(viz_ctx, eval_evaluator, epoch, args):
    if viz_ctx is None or eval_evaluator is None or args is None:
        return
    if epoch <= 0 or epoch % int(args.eval_every) != 0:
        return
    if 'bbox' not in eval_evaluator.coco_eval:
        return
    bbox_eval_dir = viz_ctx.bbox_eval_dir
    if bbox_eval_dir is None:
        return
    bbox_eval_dir.mkdir(parents=True, exist_ok=True)
    torch.save(eval_evaluator.coco_eval['bbox'].eval, bbox_eval_dir / 'latest.pth')
    if epoch % 50 == 0:
        torch.save(eval_evaluator.coco_eval['bbox'].eval, bbox_eval_dir / f'epoch_{int(epoch):04d}.pth')


def write_epoch_reports(viz_ctx, epoch, train_stats, eval_stats, num_trainable_parameters, eval_evaluator=None, args=None):
    if viz_ctx is None or not viz_ctx.should_write_artifacts:
        return

    train_epoch_record = build_train_epoch_record(
        epoch=epoch,
        train_stats=train_stats,
        num_trainable_parameters=num_trainable_parameters,
    )
    eval_epoch_record = build_eval_epoch_record(
        epoch=epoch,
        eval_stats=eval_stats,
        num_trainable_parameters=num_trainable_parameters,
    )

    append_json_record(viz_ctx.train_epoch_metrics_path, train_epoch_record)
    if eval_epoch_record.get('open_world_metrics'):
        append_json_record(viz_ctx.eval_epoch_metrics_path, eval_epoch_record)

    write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch)
    _refresh_metric_plots(viz_ctx)
    _write_bbox_eval_artifacts(viz_ctx, eval_evaluator, epoch, args)
