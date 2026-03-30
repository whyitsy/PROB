#!/usr/bin/env bash
set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_INDEPENDENT}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

COMMON_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --model_type uod \
  "

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_0_PROB_Baseline" \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 30

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_1_UnknownOnly" \
#   --uod_enable_unknown \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 30

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_2_Unknown_StaticPseudo" \
#   --uod_enable_unknown --uod_enable_pseudo \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}
# sleep 30

# python -u main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_3_BatchDynamic" \
#   --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}


# 默认参数版本
# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#   main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_5_ClsSoftAttn" \
#   --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}


torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
  main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C3_5_ClsSoftAttn_PARAMS_1" \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  --unk_loss_coef 0.5 --uod_pseudo_obj_loss_coef 1 \
  --uod_pseudo_unk_loss_coef 50 --uod_start_epoch 7\
  ${COMMON_ARGS} \
  ${PY_ARGS}