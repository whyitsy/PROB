#!/usr/bin/env bash

set -x
set -euo pipefail

BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3"

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

run_stage "${BASE_EXP_DIR}/C3_1_UnknownOnly" \
  --uod_enable_unknown

sleep 5

run_stage "${BASE_EXP_DIR}/C3_2_UnknownStaticPseudo" \
  --uod_enable_unknown \
  --uod_enable_pseudo

sleep 5

run_stage "${BASE_EXP_DIR}/C3_3_BatchDynamic" \
  --uod_enable_unknown \
  --uod_enable_pseudo \
  --uod_enable_batch_dynamic

sleep 5

run_stage "${BASE_EXP_DIR}/C3_4_ClsSoftAttn" \
  --uod_enable_unknown \
  --uod_enable_pseudo \
  --uod_enable_batch_dynamic \
  --uod_enable_cls_soft_attn


# 最好的指标：59.54、21.41、7031.0、0.0657
# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#   main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C3_5_ClsSoftAttn_04_02" \
#   --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
#   --uod_pseudo_obj_loss_coef 1.5  --uod_pseudo_unk_loss_coef 0 --unk_loss_coef 8e-4 \
#   --uod_pos_per_img_cap 2 --uod_batch_topk_max 16 \
#   --uod_cls_soft_attn_alpha 0.5 --uod_cls_soft_attn_min 0.25 --uod_start_epoch 12 \
#   --uod_haux_low_obj_coef 0 --uod_haux_mid_unknown_coef 0 --uod_haux_high_unknown_coef 0 \
#   ${COMMON_ARGS} \
#   ${PY_ARGS}