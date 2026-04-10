#!/usr/bin/env bash

set -x
set -euo pipefail


BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL"
stage="t1"

COMMON_ARGS=(
  --dataset TOWOD
  --train_set=owod_t1_train
  --test_set owod_all_task_test
  --model_type uod
  --with_box_refine
  --obj_loss_coef 8e-4
)

CH3_ARGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
  --unk_loss_coef 8e-4
  --uod_pseudo_obj_loss_coef 1.5
  --uod_pseudo_unk_loss_coef 0
  --uod_pos_per_img_cap 0
  --uod_batch_topk_max 16
  --uod_cls_soft_attn_alpha 0.5
  --uod_cls_soft_attn_min 0.25
  --uod_haux_low_obj_coef 0
  --uod_haux_mid_unknown_coef 0
  --uod_haux_high_unknown_coef 0
)


torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    main_open_world.py \
    --output_dir "${BASE_EXP_DIR}/${stage}/eval" \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 20 \
    --resume "${BASE_EXP_DIR}/$stage/checkpoint.pth" \
    --uod_postprocess_unknown_scale "20" \
    --num_workers 12 \
    --eval \
    "${COMMON_ARGS[@]}" \
    "${CH3_ARGS[@]}"