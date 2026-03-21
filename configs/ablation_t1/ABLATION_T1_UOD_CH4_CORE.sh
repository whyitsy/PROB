#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chapter 4 主消融：objectness–unknownness–classification 解耦去相关优化
#
# C4-0: Chapter 3 best
# C4-1: + Orth only
# C4-2: + Decorr only
# C4-3: + Orth + Decorr
#
# 核心验证逻辑：
# 1) C4-1 vs C4-0：表示层正交是否单独有效
# 2) C4-2 vs C4-0：预测层去相关是否单独有效
# 3) C4-3 vs C4-1/C4-2：两类解耦是否互补
# -----------------------------------------------------------------------------

set -x
set -euo pipefail

BASE_EXP_DIR="${1:-exps/MOWODB/UOD_ABL_T1_CH4_CORE}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

CH3_BEST_ARGS="\
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
  --uod_start_epoch 8 --uod_neg_warmup_epochs 3 \
  --uod_pos_quantile 0.25 --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
  --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
  --uod_pos_per_img_cap 1 --uod_neg_per_img 1 \
  --uod_batch_topk_max 8 --uod_batch_topk_ratio 0.25 \
  --viz --viz_num_samples 12 --viz_tb_images 4"

# C4-0: Chapter 3 best
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_0_CH3_Best" \
  ${CH3_BEST_ARGS} \
  ${PY_ARGS}

# C4-1: Orth only
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_1_OrthOnly" \
  ${CH3_BEST_ARGS} \
  --uod_enable_decorr --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.0 \
  ${PY_ARGS}

# C4-2: Decorr only
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_2_DecorrOnly" \
  ${CH3_BEST_ARGS} \
  --uod_enable_decorr --uod_orth_loss_coef 0.0 --uod_decorr_loss_coef 0.05 \
  ${PY_ARGS}

# C4-3: Orth + Decorr
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_3_OrthPlusDecorr" \
  ${CH3_BEST_ARGS} \
  --uod_enable_decorr --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.05 \
  ${PY_ARGS}
