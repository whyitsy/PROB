#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chapter 3 参数敏感性消融（建议在 C3-3 方向成立后再跑）
#
# P1: uod_pos_quantile   -> 验证伪未知阈值激进度
# P2: uod_batch_topk_ratio -> 验证 batch 动态分配强度
# P3: uod_start_epoch    -> 验证伪监督启用时机
#
# 固定基础：Chapter 3 best = unknownness + pseudo + batch dynamic
# -----------------------------------------------------------------------------

set -x
set -euo pipefail

BASE_EXP_DIR="${1:-exps/MOWODB/UOD_ABL_T1_CH3_PARAMS}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

BASE_ARGS="\
  --dataset TOWOD \
  --PREV_INTRODUCED_CLS 0 \
  --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --epochs 41 \
  --batch_size 5 \
  --eval_every 5 \
  --num_workers 16 \
  --obj_loss_coef 8e-4 \
  --obj_temp 1.3 \
  --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
  --unk_loss_coef 0.30 --uod_pseudo_unk_loss_coef 0.40 --uod_bg_unk_loss_coef 0.20 \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef 0.20 \
  --uod_neg_warmup_epochs 3 \
  --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
  --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
  --uod_pos_per_img_cap 1 --uod_neg_per_img 1 \
  --uod_batch_topk_max 8 \
  --viz --viz_num_samples 12 --viz_tb_images 4"

# P1: pos_quantile
for Q in 0.15 0.25 0.35; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P1_pos_quantile_${Q}" \
  ${BASE_ARGS} --uod_pos_quantile ${Q} --uod_batch_topk_ratio 0.25 --uod_start_epoch 8 \
  ${PY_ARGS}
done

# P2: batch_topk_ratio
for R in 0.15 0.25 0.35; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P2_batch_topk_ratio_${R}" \
  ${BASE_ARGS} --uod_pos_quantile 0.25 --uod_batch_topk_ratio ${R} --uod_start_epoch 8 \
  ${PY_ARGS}
done

# P3: start_epoch
for E in 4 8 12; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P3_start_epoch_${E}" \
  ${BASE_ARGS} --uod_pos_quantile 0.25 --uod_batch_topk_ratio 0.25 --uod_start_epoch ${E} \
  ${PY_ARGS}
done
