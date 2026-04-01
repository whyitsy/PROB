#!/usr/bin/env bash
set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_ON_CH3}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

CH3_BEST_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --model_type uod \
  --uod_pseudo_obj_loss_coef 1.5  --uod_pseudo_unk_loss_coef 1 \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  "

# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#     main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_0_CH3Best" \
#   ${CH3_BEST_ARGS} \
#   ${PY_ARGS}
# sleep 30

# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#     main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_1_CH3Best_ODQE" \
#   --uod_enable_odqe \
#   ${CH3_BEST_ARGS} \
#   ${PY_ARGS}
# sleep 30

# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#     main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_2_CH3Best_Decorr" \
#   --uod_enable_decorr \
#   ${CH3_BEST_ARGS} \
#   ${PY_ARGS}
# sleep 30

torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
  main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_3_CH3Best_ODQE_Decorr" \
  --uod_enable_odqe --uod_enable_decorr \
  --resume '/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_ON_CH3/C4_3_CH3Best_ODQE_Decorr/checkpoint0005.pth' \
  ${CH3_BEST_ARGS} \
  ${PY_ARGS}
