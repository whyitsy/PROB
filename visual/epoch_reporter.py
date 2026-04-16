import logging

import torch

from visual.metrics_plotter import append_json_record, refresh_metric_plots


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _safe_div(numerator, denominator):
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def _sum_optional_floats(*values):
    valid_values = []
    for value in values:
        value = _safe_float(value)
        if value is not None:
            valid_values.append(value)
    return None if not valid_values else float(sum(valid_values))


def build_train_epoch_record(epoch, train_stats, num_trainable_parameters):
    train_stats = train_stats or {}
    return {
        'epoch': int(epoch),
        'num_trainable_parameters': num_trainable_parameters,
        'train_total_loss': _safe_float(train_stats.get('loss')),
        'train_lr': _safe_float(train_stats.get('lr')),
        'train_grad_norm': _safe_float(train_stats.get('grad_norm')),
        'train_class_error': _safe_float(train_stats.get('class_error')),
        'train_weighted_loss_ce': _safe_float(train_stats.get('loss_ce')),
        'train_raw_loss_ce': _safe_float(train_stats.get('loss_ce')),
        'train_weighted_loss_bbox': _safe_float(train_stats.get('loss_bbox')),
        'train_raw_loss_bbox': _safe_float(train_stats.get('loss_bbox')),
        'train_weighted_loss_giou': _safe_float(train_stats.get('loss_giou')),
        'train_raw_loss_giou': _safe_float(train_stats.get('loss_giou')),
        'train_weighted_loss_obj_ll': _safe_float(train_stats.get('loss_obj_ll')),
        'train_raw_loss_obj_ll': _safe_float(train_stats.get('loss_obj_ll')),
        'train_weighted_loss_unk_known': _safe_float(train_stats.get('loss_unk_known')),
        'train_raw_loss_unk_known': _safe_float(train_stats.get('loss_unk_known')),
        'train_weighted_loss_obj_pseudo': _safe_float(train_stats.get('loss_obj_pseudo')),
        'train_raw_loss_obj_pseudo': _safe_float(train_stats.get('loss_obj_pseudo')),
        'train_weighted_loss_obj_neg': _safe_float(train_stats.get('loss_obj_neg')),
        'train_raw_loss_obj_neg': _safe_float(train_stats.get('loss_obj_neg')),
        'train_weighted_loss_unk_pseudo': _safe_float(train_stats.get('loss_unk_pseudo')),
        'train_raw_loss_unk_pseudo': _safe_float(train_stats.get('loss_unk_pseudo')),
        'train_weighted_loss_decorr': _safe_float(train_stats.get('loss_decorr')),
        'train_raw_loss_decorr': _safe_float(train_stats.get('loss_decorr')),
        'train_weighted_loss_bbox_pseudo_cons': _safe_float(train_stats.get('loss_bbox_pseudo_cons')),
        'train_weighted_loss_giou_pseudo_cons': _safe_float(train_stats.get('loss_giou_pseudo_cons')),
        'num_selected_pseudo_positive_queries': _safe_float(train_stats.get('num_selected_pseudo_positive_queries', train_stats.get('stat_num_batch_selected_pos'))),
        'num_selected_reliable_background_queries': _safe_float(train_stats.get('num_selected_reliable_background_queries', train_stats.get('stat_num_dummy_neg'))),
        'num_pseudo_positive_candidates': _safe_float(train_stats.get('num_pseudo_positive_candidates', train_stats.get('stat_num_pos_candidates'))),
        'num_classification_ignored_queries': _safe_float(train_stats.get('num_classification_ignored_queries', train_stats.get('stat_num_ignore_queries'))),
        'pseudo_positive_selection_ratio': _safe_div(
            train_stats.get('num_selected_pseudo_positive_queries', train_stats.get('stat_num_batch_selected_pos')),
            train_stats.get('num_unmatched_queries_after_filter', train_stats.get('stat_num_valid_unmatched')),
        ),
        'pseudo_positive_accept_ratio': _safe_div(
            train_stats.get('num_selected_pseudo_positive_queries', train_stats.get('stat_num_batch_selected_pos')),
            train_stats.get('num_pseudo_positive_candidates', train_stats.get('stat_num_pos_candidates')),
        ),
        'train_total_knownness_loss': _sum_optional_floats(train_stats.get('loss_unk_known'), train_stats.get('loss_unk_pseudo')),
    }


def build_eval_epoch_record(epoch, eval_stats, num_trainable_parameters):
    open_world_metrics = eval_stats.get('open_world_metrics', {}) if isinstance(eval_stats, dict) else {}
    return {
        'epoch': int(epoch),
        'num_trainable_parameters': num_trainable_parameters,
        'open_world_metrics': open_world_metrics,
    }


def write_eval_scalars_to_tensorboard(viz_ctx, eval_stats, epoch):
    if viz_ctx is None or viz_ctx.tb_writer is None or not isinstance(eval_stats, dict):
        return
    for key, value in eval_stats.items():
        if key == 'open_world_metrics' and isinstance(value, dict):
            for metric_name, metric_value in value.items():
                if isinstance(metric_value, (int, float)):
                    tag = 'A-OSE' if metric_name == 'AOSA' else metric_name
                    viz_ctx.tb_writer.add_scalar(f'eval/metrics/{tag}', metric_value, epoch)
        elif isinstance(value, (int, float)):
            viz_ctx.tb_writer.add_scalar(f'eval/{key}', value, epoch)


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

    _refresh_metric_plots(viz_ctx)
    _write_bbox_eval_artifacts(viz_ctx, eval_evaluator, epoch, args)
