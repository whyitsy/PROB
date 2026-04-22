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
    del outputs, targets, criterion, args, epoch, local_step
    if viz_ctx is None or viz_ctx.tb_writer is None:
        return
    write_train_step_artifacts(
        tb_writer=viz_ctx.tb_writer,
        global_step=global_step,
        optimizer=optimizer,
        grad_total_norm=grad_total_norm,
        total_loss=total_loss,
        reduced_loss_dict=reduced_loss_dict,
        reduced_weighted_loss_dict=reduced_weighted_loss_dict,
        reduced_model_stat_dict=reduced_model_stat_dict,
    )


def init_eval_visual_state(viz_ctx):
    del viz_ctx
    return None


def collect_eval_visuals(viz_ctx, visual_state, samples, targets, visual_results, outputs, criterion, args, epoch=0):
    del viz_ctx, visual_state, samples, targets, visual_results, outputs, criterion, args, epoch
    return


def finalize_eval_visuals(viz_ctx, visual_state, epoch=0):
    del viz_ctx, visual_state, epoch
    return {}
