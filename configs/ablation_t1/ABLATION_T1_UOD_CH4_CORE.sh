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

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_CORE_3_25}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

CH3_BEST_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --obj_loss_coef 4e-3 \
  --model_type uod \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  --uod_pseudo_obj_loss_coef 0.50 --uod_obj_neg_loss_coef 0.40 --uod_cls_soft_attn_alpha 0.7  \
  --uod_pseudo_unk_loss_coef 0.8 \
  "

# # C4-0: Chapter 3 best
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_0_CH3_Best" \
#   ${CH3_BEST_ARGS} \
#   ${PY_ARGS}
# sleep 5

# C4-4: ODQE + Orth + Decorr
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_4_OrthPlusDecorr" \
  ${CH3_BEST_ARGS} \
  --uod_enable_odqe \
  --uod_enable_decorr --uod_orth_loss_coef 2.0 --uod_decorr_loss_coef 2.0 \
  ${PY_ARGS}
sleep 10

# C4_1: ODQE only
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_1_ODQEOnly" \
#   ${CH3_BEST_ARGS} \
#   --uod_enable_odqe \
#   ${PY_ARGS}
# sleep 5

# # C4-2: ODQE + Orth 
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_2_OrthOnly" \
#   ${CH3_BEST_ARGS} \
#   --uod_enable_orth \
#   --uod_enable_decorr --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.0 \
#   ${PY_ARGS}
# sleep 5

# C4-3: ODQE + Decorr
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_3_DecorrOnly" \
#   ${CH3_BEST_ARGS} \
#   --uod_enable_decorr \
#   --uod_enable_decorr --uod_orth_loss_coef 0.0 --uod_decorr_loss_coef 0.05 \
#   ${PY_ARGS}

