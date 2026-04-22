import json
from pathlib import Path
import torch

from util.visual.evaluation import compute_train_query_score_statistics
from util.visual.helper import save_svg_figure

PALETTE = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'cyan': '#33BBEE',
    'red': '#CC3311',
    'green': '#009988',
    'magenta': '#EE3377',
    'yellow': '#EEDD44',
    'purple': '#7A52A5',
    'gray': '#6C757D',
}

MODEL_STAT_KEYS = {
    'stat_num_dummy_pos': 'train/model_stats/num_dummy_pos',
    'stat_num_dummy_neg': 'train/model_stats/num_dummy_neg',
    'stat_num_ignore_queries': 'train/model_stats/num_ignore_queries',
    'stat_num_valid_unmatched': 'train/model_stats/num_valid_unmatched',
    'stat_num_pos_candidates': 'train/model_stats/num_pos_candidates',
    'stat_num_neg_candidates': 'train/model_stats/num_neg_candidates',
    'stat_num_batch_selected_pos': 'train/model_stats/num_batch_selected_pos',
    'stat_pos_thresh_mean': 'train/model_stats/pos_thresh_mean',
    'stat_cls_attn_mean': 'train/model_stats/cls_attn_mean',
    'stat_num_cls_soft': 'train/model_stats/num_cls_soft',
    'gate_mean': 'train/model_stats/gate_mean',
}


def safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        try:
            return float(value.detach().cpu().item())
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def safe_div(numerator, denominator):
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def write_train_step_artifacts(tb_writer, global_step, epoch, local_step, optimizer, grad_total_norm, total_loss, reduced_loss_dict, reduced_weighted_loss_dict, reduced_model_stat_dict=None):
    tb_writer.add_scalar('train/loss/total', total_loss, global_step)
    tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    tb_writer.add_scalar('train/grad_norm', grad_total_norm, global_step)

    for key, value in reduced_loss_dict.items():
        tb_writer.add_scalar(f'train/loss_raw/{key}', value, global_step)

    for key, value in reduced_weighted_loss_dict.items():
        tb_writer.add_scalar(f'train/loss_weighted/{key}', value, global_step)

    if reduced_model_stat_dict:
        for key, value in reduced_model_stat_dict.items():
            if key.startswith('stat_') or key.startswith('num_') or key == 'gate_mean':
                tb_writer.add_scalar(f'train/model_stats/{key}', value, global_step)


def write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch):
    if viz_ctx is None or viz_ctx.tb_writer is None or not isinstance(eval_stats, dict):
        return
    for key, value in eval_stats.items():
        if isinstance(value, (int, float)):
            viz_ctx.tb_writer.add_scalar(f'eval/{key}', value, epoch)
