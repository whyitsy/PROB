
import util.misc as utils
from util.visual.evaluation import compute_branch_correlation_metrics
from visual.eval_visualizer import (
    compute_branch_correlation_metrics,
    finalize_eval_visualizations,
    init_eval_visual_state as _init_eval_visual_state,
    save_eval_qualitative_cases,
)
from visual.postprocess_aligned_stats import collect_eval_visual_stats_aligned
from visual.train_writer import write_train_step_artifacts


def log_train_step_artifacts(
    viz_ctx,
    global_step,
    epoch,
    local_step,
    optimizer,
    grad_total_norm,
    outputs,
    targets,
    criterion,
    total_loss,
    reduced_loss_dict,
    reduced_weighted_loss_dict,
    args=None,
):
    if viz_ctx is None or not viz_ctx.should_write_artifacts:
        return
    write_train_step_artifacts(
        tb_writer=viz_ctx.tb_writer,
        step_jsonl_path=viz_ctx.train_step_metrics_path,
        global_step=global_step,
        epoch=epoch,
        local_step=local_step,
        optimizer=optimizer,
        grad_total_norm=grad_total_norm,
        outputs=outputs,
        targets=targets,
        criterion=criterion,
        total_loss=total_loss,
        reduced_loss_dict=reduced_loss_dict,
        reduced_weighted_loss_dict=reduced_weighted_loss_dict,
        viz_cfg=viz_ctx.viz_cfg,
        args=args,
    )


def init_eval_visual_state(viz_ctx):
    if viz_ctx is None or not viz_ctx.visualization_enabled or not utils.is_main_process():
        return None
    return _init_eval_visual_state(viz_ctx.viz_cfg)


def collect_eval_visuals(viz_ctx, visual_state, samples, targets, visual_results, outputs, criterion, args, epoch=0):
    if viz_ctx is None or visual_state is None:
        return
    collect_eval_visual_stats_aligned(visual_state, outputs, targets, criterion, args)
    visual_output_dir = viz_ctx.eval_visualization_dir(epoch)
    save_eval_qualitative_cases(
        visual_state,
        samples,
        targets,
        visual_results,
        outputs,
        criterion,
        args,
        str(visual_output_dir),
        viz_ctx.viz_cfg,
        tb_writer=viz_ctx.tb_writer,
        global_step=epoch,
        epoch=epoch,
    )


def finalize_eval_visuals(viz_ctx, visual_state, epoch=0):
    if viz_ctx is None or visual_state is None:
        return {}

    extra_metrics = {}
    for key, value in compute_branch_correlation_metrics(visual_state).items():
        if value is not None:
            extra_metrics[key] = value

    finalize_eval_visualizations(
        visual_state,
        str(viz_ctx.output_dir),
        epoch,
        viz_ctx.viz_cfg,
        tb_writer=viz_ctx.tb_writer,
    )
    return extra_metrics
