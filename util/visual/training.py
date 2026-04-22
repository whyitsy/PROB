import torch


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


def write_train_step_artifacts(tb_writer, global_step, optimizer, grad_total_norm, total_loss, reduced_loss_dict, reduced_weighted_loss_dict, reduced_model_stat_dict=None):
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
