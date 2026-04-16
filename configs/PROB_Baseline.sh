#!/usr/bin/env bash

set -euo pipefail
set -x

BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/PROB"

REPLAY_DIR="prob"

COMMON_ARGS=(
  --model_type prob
  --with_box_refine
  --exemplar_replay_dir "${REPLAY_DIR}"
  --viz
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
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t1_train \
  --epochs 41 \
  --uod_start_epoch 12 \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 850 \
  --exemplar_replay_cur_file learned_owod_t1_ft.txt

# ----------------
# Task 2
# ----------------
run_stage "${BASE_EXP_DIR}/t2" \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t2_train \
  --epochs 51 \
  --uod_start_epoch 46 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 1743 \
  --exemplar_replay_prev_file learned_owod_t1_ft.txt \
  --exemplar_replay_cur_file learned_owod_t2_ft.txt \
  --pretrain "${BASE_EXP_DIR}/t1/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

run_stage "${BASE_EXP_DIR}/t2_ft" \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owod_t2_ft" \
  --epochs 111 \
  --lr_drop 40 \
  --pretrain "${BASE_EXP_DIR}/t2/train/checkpoints/checkpoint_latest.pth"

# ----------------
# Task 3
# ----------------
run_stage "${BASE_EXP_DIR}/t3" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t3_train \
  --epochs 121 \
  --uod_start_epoch 116 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2361 \
  --exemplar_replay_prev_file learned_owod_t2_ft.txt \
  --exemplar_replay_cur_file learned_owod_t3_ft.txt \
  --pretrain "${BASE_EXP_DIR}/t2_ft/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

run_stage "${BASE_EXP_DIR}/t3_ft" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owod_t3_ft" \
  --epochs 181 \
  --lr_drop 35 \
  --pretrain "${BASE_EXP_DIR}/t3/train/checkpoints/checkpoint_latest.pth"

# ----------------
# Task 4
# ----------------
run_stage "${BASE_EXP_DIR}/t4" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t4_train \
  --epochs 191 \
  --uod_start_epoch 186 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2749 \
  --exemplar_replay_prev_file learned_owod_t3_ft.txt \
  --exemplar_replay_cur_file learned_owod_t4_ft.txt \
  --num_inst_per_class 40 \
  --pretrain "${BASE_EXP_DIR}/t3_ft/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

run_stage "${BASE_EXP_DIR}/t4_ft" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owod_t4_ft" \
  --epochs 261 \
  --lr_drop 50 \
  --pretrain "${BASE_EXP_DIR}/t4/train/checkpoints/checkpoint_latest.pth"
