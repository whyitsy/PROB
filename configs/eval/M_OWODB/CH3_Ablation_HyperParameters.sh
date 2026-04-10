#!/usr/bin/env bash

set -x
set -euo pipefail


BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE_O4_03"


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



  if [ ! -f "${BASE_EXP_DIR}/$stage/checkpoint.pth" ]; then
    echo "[WARN] checkpoint not found, skip: ${BASE_EXP_DIR}/$stage/checkpoint.pth"
    return 0
  fi

  torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    main_open_world.py \
    --output_dir "${BASE_EXP_DIR}/${stage}/${uod_postprocess_unknown_scale}/${obj_temp}" \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 20 \
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
    for stage in C3_1_UnknownOnly,C3_2_Unknown_StaticPseudo,C3_3_BatchDynamic,C3_5_ClsSoftAttn; do
      run_eval "$uod_postprocess_unknown_scale" "$obj_temp" "$stage" 
    done
  done
done
