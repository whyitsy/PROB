#!/usr/bin/env bash

set -euo pipefail
set -x

BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS/PROB_EVAL/SOWODB"


COMMON_ARGS=(
  --model_type prob
  --viz
  --eval
  --dataset OWDETR
  --train_set owdetr_t1_train
  --test_set owdetr_test
  --eval_batch_size 20
)

run_stage() {
  local out_dir="$1"
  shift
  torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    main_open_world.py \
    --output_dir "${out_dir}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# ----------------
# Task 1
# ----------------
run_stage "${BASE_EXP_DIR}/t1" \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
  --eval_checkpoint "/mnt/data/kky/output/PROB/exps/PROB/SOWODB/t1.pth" 

run_stage "${BASE_EXP_DIR}/t2_ft" \
  --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
  --eval_checkpoint "/mnt/data/kky/output/PROB/exps/PROB/SOWODB/t2.pth"

run_stage "${BASE_EXP_DIR}/t3_ft" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --eval_checkpoint "/mnt/data/kky/output/PROB/exps/PROB/SOWODB/t3.pth"

run_stage "${BASE_EXP_DIR}/t4_ft" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --eval_checkpoint "/mnt/data/kky/output/PROB/exps/PROB/SOWODB/t4.pth"
