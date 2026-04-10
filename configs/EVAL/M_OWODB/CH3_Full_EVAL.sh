#!/usr/bin/env bash

set -x
set -euo pipefail


BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL"

COMMON_ARGS=(
  --model_type uod
  --with_box_refine
  --viz
  --eval
)

CH3_ARGS=(
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
    --resume "${out_dir}/checkpoint.pth" \
    "${COMMON_ARGS[@]}" \
    "${CH3_ARGS[@]}" 
}


# run_stage "${BASE_EXP_DIR}/t1" 

run_stage "${BASE_EXP_DIR}/t2" 

run_stage "${BASE_EXP_DIR}/t2_ft" 

run_stage "${BASE_EXP_DIR}/t3" 

run_stage "${BASE_EXP_DIR}/t3_ft" 

run_stage "${BASE_EXP_DIR}/t4" 

run_stage "${BASE_EXP_DIR}/t4_ft" 
