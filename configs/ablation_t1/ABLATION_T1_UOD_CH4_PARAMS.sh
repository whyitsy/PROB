#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chapter 4 参数敏感性消融（建议在 C4-3 方向成立后再跑）
#
# P1: uod_orth_loss_coef   -> 验证表示层正交强度
# P2: uod_decorr_loss_coef -> 验证预测去相关强度
# P3: 两者比例            -> 验证两类解耦谁更主导
#
# 固定基础：Chapter 4 best = CH3 best + orth + decorr
# -----------------------------------------------------------------------------

set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_PARAMS}"
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
  --num_workers 8 \
  --obj_loss_coef 8e-4 \
  --obj_temp 1.3 \
  --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_decorr \
  --unk_loss_coef 0.30 --uod_pseudo_unk_loss_coef 0.40 --uod_bg_unk_loss_coef 0.20 \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef 0.20 \
  --uod_start_epoch 8 --uod_neg_warmup_epochs 3 \
  --uod_pos_quantile 0.25 --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
  --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
  --uod_pos_per_img_cap 1 --uod_neg_per_img 1 \
  --uod_batch_topk_max 8 --uod_batch_topk_ratio 0.25 \
  --viz --viz_num_samples 12 --viz_tb_images 4"

# P1: orth_loss_coef
for O in 0.02 0.05 0.10; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P1_orth_coef_${O}" \
  ${BASE_ARGS} --uod_orth_loss_coef ${O} --uod_decorr_loss_coef 0.05 \
  ${PY_ARGS}
done

# P2: decorr_loss_coef
for D in 0.02 0.05 0.10; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P2_decorr_coef_${D}" \
  ${BASE_ARGS} --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef ${D} \
  ${PY_ARGS}
done

# P3: relative weighting
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P3_ratio_orth_only" \
  ${BASE_ARGS} --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.0 \
  ${PY_ARGS}

python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P3_ratio_decorr_only" \
  ${BASE_ARGS} --uod_orth_loss_coef 0.0 --uod_decorr_loss_coef 0.05 \
  ${PY_ARGS}

python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P3_ratio_balanced" \
  ${BASE_ARGS} --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.05 \
  ${PY_ARGS}
