#!/usr/bin/env bash


set -x
set -euo pipefail

# -----------------------------------------------------------------------------
# UOD ABLATION T1: UOD CH3 CORE PARAMLESS
# -----------------------------------------------------------------------------
BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE_PARAMLESS_ALL}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

COMMON_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --obj_loss_coef 8e-3 \
  --model_type uod_paramless \
  --uod_pseudo_unk_loss_coef 1.5 --uod_neg_per_img 50 \
  --uod_pseudo_obj_loss_coef 1.0 --uod_obj_neg_loss_coef 0.40 \
  --uod_pos_per_img_cap 2 --uod_batch_topk_max 16 --uod_batch_topk_ratio 0.5 \
  --uod_known_reject_thresh 0.2  --uod_min_pos_thresh 0.05 \
  "

# -----------------------------------------------------------------------------
# C3-3: + Classification Soft Attention
# 目的：证明减弱分类器对伪正样本的排斥力，能保护高疑似未知物体的置信度。
# -----------------------------------------------------------------------------
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_3_SoftAttention" \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  ${COMMON_ARGS} \
  ${PY_ARGS}
sleep 10

# -----------------------------------------------------------------------------
# C3-0: baseline
# -----------------------------------------------------------------------------
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_0_PROB_Baseline" \
  ${COMMON_ARGS} \
  ${PY_ARGS}
sleep 10

# -----------------------------------------------------------------------------
# C3-1: Unknownness + Static Pseudo
# 引入显式未知分支，并利用基础的静态挖掘给予正向监督
# -----------------------------------------------------------------------------
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_1_Unknown_StaticPseudo" \
  ${COMMON_ARGS} \
  --uod_enable_unknown --uod_enable_pseudo \
  ${PY_ARGS}
sleep 10


# -----------------------------------------------------------------------------
# C3-2: + Batch Dynamic Allocation
# 目的：证明跨图动态分配能够避免强制在纯背景图中挖掘伪标签，降低 FP，提升 A-OSE。
# -----------------------------------------------------------------------------
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_2_BatchDynamic" \
  ${COMMON_ARGS} \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
  ${PY_ARGS}
sleep 10

