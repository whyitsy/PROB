#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chapter 3 主消融：显式未知性建模与稀疏伪监督协同学习
# 严谨消融路径重构版：
# C3-0: 统一架构基线 (Aligned Baseline, model_type=uod, 全部特性关闭)
# C3-1: 显式未知性 + 静态逐图伪监督 (开启 unknown & pseudo, 静态上限)
# C3-2: + 动态分配 (开启 batch_dynamic, 解决背景图挖掘噪声)
# C3-3: + 分类软掩码 (开启 cls_soft_attn, 缓解分类器排斥)
# -----------------------------------------------------------------------------

set -x
set -euo pipefail

# BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE_SOFT_ATTEN}"
BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE_3_25_second}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

COMMON_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --obj_loss_coef 4e-3 \
  --model_type uod \
  "

# -----------------------------------------------------------------------------
# C3-3: + Classification Soft Attention
# 目的：证明减弱分类器对伪正样本的排斥力，能保护高疑似未知物体的置信度。
# -----------------------------------------------------------------------------
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_3_SoftAttention" \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
  ${COMMON_ARGS} \
  --uod_pseudo_unk_loss_coef 0.8 --uod_pseudo_obj_loss_coef 0.50 --uod_obj_neg_loss_coef 0.40 --uod_cls_soft_attn_alpha 0.3 --obj_loss_coef 4e-3 \
  --uod_neg_per_img 20 --uod_pos_per_img_cap 2 --uod_batch_topk_max 16 --uod_batch_topk_ratio 0.5 \
  --uod_known_reject_thresh 0.2 --uod_pos_scale 1.0 --uod_min_pos_thresh 0.06 \
  ${PY_ARGS}
sleep 10

# -----------------------------------------------------------------------------
# C3-0: baseline
# -----------------------------------------------------------------------------
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_0_PROB_Baseline" \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 10

# -----------------------------------------------------------------------------
# C3-1: Unknownness + Static Pseudo
# 引入显式未知分支，并利用基础的静态挖掘给予正向监督
# -----------------------------------------------------------------------------
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_1_Unknown_StaticPseudo" \
#   ${COMMON_ARGS} \
#   --uod_enable_unknown --uod_enable_pseudo \
#   --uod_pseudo_obj_loss_coef 0.50 --uod_obj_neg_loss_coef 0.40 \
#   ${PY_ARGS}
# sleep 10


# -----------------------------------------------------------------------------
# C3-2: + Batch Dynamic Allocation
# 目的：证明跨图动态分配能够避免强制在纯背景图中挖掘伪标签，降低 FP，提升 A-OSE。
# -----------------------------------------------------------------------------
# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_2_BatchDynamic" \
#   ${COMMON_ARGS} \
#   --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
#   --uod_pseudo_obj_loss_coef 0.50 --uod_obj_neg_loss_coef 0.40 \
#   ${PY_ARGS}
# sleep 10

