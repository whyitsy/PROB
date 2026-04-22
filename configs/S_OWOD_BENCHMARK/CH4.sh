#!/usr/bin/env bash
set -euo pipefail
set -x

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH4_FULL}"

REPLAY_DIR="UOD_CH4"

COMMON_ARGS=(
  --dataset OWDETR
  --test_set owdetr_test
  --model_type uod
  --with_box_refine
  --exemplar_replay_dir ${REPLAY_DIR}
  --viz
)

CH4_ARGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
  --uod_enable_odqe
  --uod_enable_decorr
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
    "${CH4_ARGS[@]}" \
    "$@" 
}

# ----------------
# Task 1
# ----------------
run_stage "${BASE_EXP_DIR}/t1" \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 \
  --train_set owdetr_t1_train \
  --epochs 41 \
  --uod_start_epoch 12 \
  --lr_drop 31 \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 850 \
  --exemplar_replay_cur_file learned_owdetr_t1_ft.txt

sleep 5

# ----------------
# Task 2
# ----------------
run_stage "${BASE_EXP_DIR}/t2" \
  --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
  --train_set owdetr_t2_train \
  --epochs 51 \
  --uod_start_epoch 46 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 1679 \
  --exemplar_replay_prev_file learned_owdetr_t1_ft.txt \
  --exemplar_replay_cur_file learned_owdetr_t2_ft.txt \
  --resume "${BASE_EXP_DIR}/t1/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

sleep 5

run_stage "${BASE_EXP_DIR}/t2_ft" \
  --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
  --train_set "${REPLAY_DIR}/learned_owdetr_t2_ft" \
  --epochs 121 \
  --lr_drop 50 \
  --resume "${BASE_EXP_DIR}/t2/train/checkpoints/checkpoint_latest.pth"

sleep 5

# ----------------
# Task 3
# ----------------
run_stage "${BASE_EXP_DIR}/t3" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set owdetr_t3_train \
  --epochs 131 \
  --uod_start_epoch 126 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2345 \
  --exemplar_replay_prev_file learned_owdetr_t2_ft.txt \
  --exemplar_replay_cur_file learned_owdetr_t3_ft.txt \
  --resume "${BASE_EXP_DIR}/t2_ft/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

sleep 5

run_stage "${BASE_EXP_DIR}/t3_ft" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owdetr_t3_ft" \
  --epochs 201 \
  --lr_drop 50 \
  --resume "${BASE_EXP_DIR}/t3/train/checkpoints/checkpoint_latest.pth"

sleep 5

# ----------------
# Task 4
# ----------------
run_stage "${BASE_EXP_DIR}/t4" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set owdetr_t4_train \
  --epochs 211 \
  --uod_start_epoch 206 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2664 \
  --exemplar_replay_prev_file learned_owdetr_t3_ft.txt \
  --exemplar_replay_cur_file learned_owdetr_t4_ft.txt \
  --num_inst_per_class 40 \
  --resume "${BASE_EXP_DIR}/t3_ft/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

sleep 5

run_stage "${BASE_EXP_DIR}/t4_ft" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set "${REPLAY_DIR}/learned_owdetr_t4_ft" \
  --epochs 301 \
  --lr_drop 50 \
  --resume "${BASE_EXP_DIR}/t4/train/checkpoints/checkpoint_latest.pth"

sleep 5