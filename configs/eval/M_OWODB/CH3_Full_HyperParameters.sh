#!/usr/bin/env bash

set -x
set -euo pipefail


BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/"


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

run_eval() {
  local uod_postprocess_unknown_scale="$1"
  local obj_temp="$2"
  local stage="$3"

  if [ "$stage" == "t1" ]; then
    PREV_INTRODUCED_CLS=0
    CUR_INTRODUCED_CLS=20
    
  elif [ "$stage" == "t2_ft" ]; then
    PREV_INTRODUCED_CLS=20
    CUR_INTRODUCED_CLS=20
  elif [ "$stage" == "t3_ft" ]; then
    PREV_INTRODUCED_CLS=40
    CUR_INTRODUCED_CLS=20
  elif [ "$stage" == "t4_ft" ]; then
    PREV_INTRODUCED_CLS=60
    CUR_INTRODUCED_CLS=20
  else
    echo "Unknown stage: $stage"
    exit 1
  fi


  if [ ! -f "${BASE_EXP_DIR}/$stage/checkpoint.pth" ]; then
    echo "[WARN] checkpoint not found, skip: ${BASE_EXP_DIR}/$stage/checkpoint.pth"
    return 0
  fi

  torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    main_open_world.py \
    --output_dir "${BASE_EXP_DIR}/${stage}/${uod_postprocess_unknown_scale}/${obj_temp}" \
    --PREV_INTRODUCED_CLS "$PREV_INTRODUCED_CLS" \
    --CUR_INTRODUCED_CLS "$CUR_INTRODUCED_CLS" \
    --pretrain "${BASE_EXP_DIR}/$stage/checkpoint.pth" \
    --uod_postprocess_unknown_scale "$uod_postprocess_unknown_scale" \
    --obj_temp "$obj_temp" \
    --num_workers 12 \
    --eval \
    "${COMMON_ARGS[@]}" \
    "${CH3_ARGS[@]}"

  echo "Done: ${BASE_EXP_DIR}/${stage}/${uod_postprocess_unknown_scale}/${obj_temp}"

}

for uod_postprocess_unknown_scale in 5 10 15 20 25 30; do
  for obj_temp in 0.8 1.0 1.3 1.5 1.8; do
    for stage in t1 t2_ft t3_ft t4_ft; do
      run_eval "$uod_postprocess_unknown_scale" "$obj_temp" "$stage" 
    done
  done
done
