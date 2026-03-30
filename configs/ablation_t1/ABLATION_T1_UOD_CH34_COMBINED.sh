#!/usr/bin/env bash
set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH34_COMBINED}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

COMMON_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --obj_loss_coef 8e-3 \
  --model_type uod \
  --uod_pos_per_img_cap 2 --uod_batch_topk_max 16 --uod_batch_topk_ratio 0.5 \
  --uod_known_reject_thresh 0.2 --uod_min_pos_thresh 0.05 \
  --uod_cls_soft_attn_alpha 1.0 --uod_cls_soft_attn_min 0 \
  --uod_odqe_decay_min 0.1 --uod_odqe_decay_power 1.0 \
  "

python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/CH3_Best" \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  ${COMMON_ARGS} \
  ${PY_ARGS}
sleep 30

python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/CH3_CH4_Combined" \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  --uod_enable_odqe --uod_enable_decorr \
  ${COMMON_ARGS} \
  ${PY_ARGS}
