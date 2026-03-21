#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chapter 3 主消融：显式未知性建模与稀疏伪监督协同学习
#
# C3-0: PROB baseline
# C3-1: + Explicit Unknownness Branch
# C3-2: + Sparse Pseudo Supervision (不启用 batch dynamic)
# C3-3: + Batch Dynamic Allocation
#
# 核心验证逻辑：
# 1) C3-1 vs C3-0：显式 unknownness 分支是否必要
# 2) C3-2 vs C3-1：稀疏伪监督是否提供额外有效弱监督
# 3) C3-3 vs C3-2：batch 动态分配是否缓解伪正样本稀少/监督浪费
# -----------------------------------------------------------------------------

set -x
set -euo pipefail

BASE_EXP_DIR="${1:-exps/MOWODB/UOD_ABL_T1_CH3_CORE}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

COMMON_ARGS="\
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
  --viz --viz_num_samples 12 --viz_tb_images 4"

# C3-0: baseline
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_0_PROB_Baseline" \
  ${COMMON_ARGS} \
  --model_type prob \
  ${PY_ARGS}

# C3-1: explicit unknownness only
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_1_UnknownOnly" \
  ${COMMON_ARGS} \
  --model_type uod --uod_enable_unknown \
  --unk_loss_coef 0.30 \
  ${PY_ARGS}

# C3-2: unknownness + sparse pseudo supervision（静态逐图上限，不启用 batch dynamic）
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_2_UnknownPlusPseudo" \
  ${COMMON_ARGS} \
  --model_type uod --uod_enable_unknown --uod_enable_pseudo \
  --unk_loss_coef 0.30 --uod_pseudo_unk_loss_coef 0.40 --uod_bg_unk_loss_coef 0.20 \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef 0.20 \
  --uod_start_epoch 8 --uod_neg_warmup_epochs 3 \
  --uod_pos_quantile 0.25 --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
  --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
  --uod_pos_per_img_cap 1 --uod_neg_per_img 1 \
  ${PY_ARGS}

# C3-3: + batch dynamic allocation
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_3_UnknownPseudoBatchDynamic" \
  ${COMMON_ARGS} \
  --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
  --unk_loss_coef 0.30 --uod_pseudo_unk_loss_coef 0.40 --uod_bg_unk_loss_coef 0.20 \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef 0.20 \
  --uod_start_epoch 8 --uod_neg_warmup_epochs 3 \
  --uod_pos_quantile 0.25 --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
  --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
  --uod_pos_per_img_cap 1 --uod_neg_per_img 1 \
  --uod_batch_topk_max 8 --uod_batch_topk_ratio 0.25 \
  ${PY_ARGS}
