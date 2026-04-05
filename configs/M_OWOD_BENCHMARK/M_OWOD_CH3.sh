#!/usr/bin/env bash
set -euo pipefail
set -x

# Full 4-task incremental training script for TOWOD (M_OWODB split)
# CH3 configuration 
#
# Usage:
#   bash FULL_TOWOD_CH3_4STAGE_REVISED.sh [BASE_EXP_DIR] [extra args...]
#
# Optional env vars:
#   REPLAY_DIR=UOD_CH3_V1
#   GPUS=gpu
#   LAUNCH=1           # kept for future extension; script uses torchrun directly

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS=("$@")

REPLAY_DIR="${REPLAY_DIR:-UOD_CH3}"
GPUS="${GPUS:-gpu}"

COMMON_ARGS=(
  --dataset TOWOD
  --test_set owod_all_task_test
  --num_workers 8
  --model_type uod
  --with_box_refine
  --obj_loss_coef 8e-4
  --obj_temp 1.3
)

CH3_ARGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
  --unk_loss_coef 8e-4
  --uod_pseudo_obj_loss_coef 1.5
  --uod_pseudo_unk_loss_coef 0
  --uod_pos_per_img_cap 2
  --uod_batch_topk_max 16
  --uod_cls_soft_attn_alpha 0.5
  --uod_cls_soft_attn_min 0.25
  --uod_start_epoch 12
  --uod_haux_low_obj_coef 0
  --uod_haux_mid_unknown_coef 0
  --uod_haux_high_unknown_coef 0
)

run_stage() {
  local out_dir="$1"
  shift
  torchrun --standalone --nnodes=1 --nproc-per-node="${GPUS}" \
    main_open_world.py \
    --output_dir "${out_dir}" \
    "$@" \
    "${COMMON_ARGS[@]}" \
    "${CH3_ARGS[@]}" \
    "${PY_ARGS[@]}"
}

# ----------------
# Task 1
# ----------------
run_stage "${BASE_EXP_DIR}/t1" \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t1_train \
  --epochs 41 \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 850 \
  --exemplar_replay_dir "${REPLAY_DIR}" \
  --exemplar_replay_cur_file learned_owod_t1_ft.txt

# ----------------
# Task 2
# ----------------
run_stage "${BASE_EXP_DIR}/t2" \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t2_train \
  --epochs 51 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 1743 \
  --exemplar_replay_dir "${REPLAY_DIR}" \
  --exemplar_replay_prev_file learned_owod_t1_ft.txt \
  --exemplar_replay_cur_file learned_owod_t2_ft.txt \
  --pretrain "${BASE_EXP_DIR}/t1/checkpoint0040.pth" \
  --lr 2e-5

run_stage "${BASE_EXP_DIR}/t2_ft" \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owod_t2_ft" \
  --epochs 111 \
  --lr_drop 40 \
  --pretrain "${BASE_EXP_DIR}/t2/checkpoint0050.pth"

# ----------------
# Task 3
# ----------------
run_stage "${BASE_EXP_DIR}/t3" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t3_train \
  --epochs 121 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2361 \
  --exemplar_replay_dir "${REPLAY_DIR}" \
  --exemplar_replay_prev_file learned_owod_t2_ft.txt \
  --exemplar_replay_cur_file learned_owod_t3_ft.txt \
  --pretrain "${BASE_EXP_DIR}/t2_ft/checkpoint0110.pth" \
  --lr 2e-5

run_stage "${BASE_EXP_DIR}/t3_ft" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owod_t3_ft" \
  --epochs 181 \
  --lr_drop 35 \
  --pretrain "${BASE_EXP_DIR}/t3/checkpoint0120.pth"

# ----------------
# Task 4
# ----------------
run_stage "${BASE_EXP_DIR}/t4" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t4_train \
  --epochs 191 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2749 \
  --exemplar_replay_dir "${REPLAY_DIR}" \
  --exemplar_replay_prev_file learned_owod_t3_ft.txt \
  --exemplar_replay_cur_file learned_owod_t4_ft.txt \
  --num_inst_per_class 40 \
  --pretrain "${BASE_EXP_DIR}/t3_ft/checkpoint0180.pth" \
  --lr 2e-5

run_stage "${BASE_EXP_DIR}/t4_ft" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owod_t4_ft" \
  --epochs 261 \
  --lr_drop 50 \
  --pretrain "${BASE_EXP_DIR}/t4/checkpoint0190.pth"
