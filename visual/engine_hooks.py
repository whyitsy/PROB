import util.misc as utils
from util.visual.training import write_train_step_artifacts


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
    reduced_model_stat_dict=None,
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
        reduced_model_stat_dict=reduced_model_stat_dict,
        viz_cfg=viz_ctx.viz_cfg,
        args=args,
    )


def init_eval_visual_state(viz_ctx):
    return None


def collect_eval_visuals(viz_ctx, visual_state, samples, targets, visual_results, outputs, criterion, args, epoch=0):
    return


def finalize_eval_visuals(viz_ctx, visual_state, epoch=0):
    return {}
