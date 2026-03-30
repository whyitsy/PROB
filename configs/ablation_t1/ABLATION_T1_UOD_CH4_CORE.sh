#!/usr/bin/env bash
set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_INDEPENDENT}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

COMMON_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --obj_loss_coef 8e-3 \
  --model_type uod \
  --uod_enable_unknown \
  --uod_known_reject_thresh 0.2 --uod_min_pos_thresh 0.05 \
  --uod_odqe_decay_min 0.1 --uod_odqe_decay_power 1.0 \
  "

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_0_MinimalUOD" \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 30

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_1_ODQEOnly" \
#   --uod_enable_odqe \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 30

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_2_DecorrOnly" \
#   --uod_enable_decorr \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 30

torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
  main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_3_ODQE_Decorr" \
  --uod_enable_odqe --uod_enable_decorr \
  ${COMMON_ARGS} \
  ${PY_ARGS}
