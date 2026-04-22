#!/usr/bin/env bash

set -euo pipefail
set -x

BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/SOWODB/UOD_ABL_T1_CH3"

COMMON_ARGS=(
  --dataset OWDETR
  --test_set owdetr_test
  --model_type uod
  --with_box_refine
  --viz
)

CH3_ARGS=(
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
  --uod_pseudo_bbox_loss_coef 3
  --uod_pseudo_giou_loss_coef 1
  --uod_pseudo_obj_loss_coef 1.5
  --uod_pseudo_unk_loss_coef 0
  --uod_haux_low_obj_coef 0
  --uod_haux_mid_unknown_coef 0
  --uod_haux_high_unknown_coef 0
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

# run_stage "${BASE_EXP_DIR}/C3_1_UnknownOnly" \
#   --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
#   --train_set owdetr_t1_train \
#   --lr_drop 31 \
#   --resume '/mnt/data/kky/output/PROB/exps/SOWODB/UOD_ABL_T1_CH3/C3_1_UnknownOnly/train/checkpoints/checkpoint_latest.pth' \
#   --uod_enable_unknown

# sleep 5

# run_stage "${BASE_EXP_DIR}/C3_2_Unknown_StaticPseudo" \
#   --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
#   --train_set owdetr_t1_train \
#   --lr_drop 31 \
#   --uod_enable_unknown \
#   --uod_enable_pseudo 

# sleep 5

# run_stage "${BASE_EXP_DIR}/C3_3_BatchDynamic" \
#   --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
#   --train_set owdetr_t1_train \
#   --lr_drop 31 \
#   --uod_enable_unknown \
#   --uod_enable_pseudo \
#   --uod_enable_batch_dynamic 

# sleep 5

run_stage "${BASE_EXP_DIR}/C3_4_ClsSoftAttn" \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
  --train_set owdetr_t1_train \
  --lr_drop 31 \
  --uod_enable_unknown \
  --uod_enable_pseudo \
  --uod_enable_batch_dynamic \
  --uod_enable_cls_soft_attn

sleep 5
