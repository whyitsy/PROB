#!/usr/bin/env bash

set -x
set -euo pipefail

BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_ON_CH3"

COMMON_ARGS=(
  --dataset TOWOD
  --test_set owod_all_task_test
  --model_type uod
  --with_box_refine
  --viz
)

CH3_ARGS=(
  --unk_loss_coef 4e-4
  --uod_pseudo_obj_loss_coef 1
  --uod_cls_soft_attn_alpha 0.5
  --uod_cls_soft_attn_min 0.25
  --uod_haux_low_obj_coef 0
  --uod_haux_mid_unknown_coef 0
  --uod_haux_high_unknown_coef 0
  --uod_pos_unk_min 0.08
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
)


run_stage() {
  local out_dir="$1"
  shift
  torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    main_open_world.py \
    --output_dir "${out_dir}" \
    "${COMMON_ARGS[@]}" \
    "${CH3_ARGS[@]}" \
    "$@"
}

run_stage "${BASE_EXP_DIR}/C4_1_ODQE" \
  --uod_enable_odqe

sleep 5

run_stage "${BASE_EXP_DIR}/C4_2_Decorr" \
  --uod_enable_decorr

sleep 5

run_stage "${BASE_EXP_DIR}/C4_3_ODQE_Decorr" \
  --uod_enable_odqe \
  --uod_enable_decorr

sleep 5


# 最好指标：61.02、22.1、7278.0、0.0600
# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#   main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_3_CH3Best_ODQE_Decorr_04_02" \
#   --uod_enable_odqe --uod_enable_decorr \
#   ${CH3_BEST_ARGS} \
#   ${PY_ARGS}

