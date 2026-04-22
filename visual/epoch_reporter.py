import logging

import torch

from util.visual.training import write_eval_scalars_to_tensorboard


def write_epoch_reports(viz_ctx, epoch, train_stats, eval_stats, num_trainable_parameters, eval_evaluator=None, args=None):
    if viz_ctx is None or not viz_ctx.should_write_artifacts:
        return
    write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch)
