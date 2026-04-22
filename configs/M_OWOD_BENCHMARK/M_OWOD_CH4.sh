#!/usr/bin/env bash

set -euo pipefail
set -x


BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH4_FULL"

GPUS="${GPUS:-gpu}"

COMMON_ARGS=(
  --dataset TOWOD
  --test_set owod_all_task_test
  --model_type uod
  --with_box_refine
  --exemplar_replay_dir "UOD_CH4"
)

CH4_ARGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
  --uod_enable_odqe
  --uod_enable_decorr
  --unk_loss_coef 8e-4
  --uod_pseudo_obj_loss_coef 1.5
  --uod_pseudo_unk_loss_coef 0
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
    "${CH4_ARGS[@]}" \
    "${PY_ARGS[@]}"
}

run_stage "${BASE_EXP_DIR}/t1" \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t1_train \
  --epochs 41 \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 850 \
  --exemplar_replay_cur_file learned_owod_t1_ft.txt

sleep 5

run_stage "${BASE_EXP_DIR}/t2" \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t2_train \
  --epochs 51 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 1743 \
  --exemplar_replay_prev_file learned_owod_t1_ft.txt \
  --exemplar_replay_cur_file learned_owod_t2_ft.txt \
  --resume "${BASE_EXP_DIR}/t1/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

sleep 5

run_stage "${BASE_EXP_DIR}/t2_ft" \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --train_set "UOD_CH4/learned_owod_t2_ft" \
  --epochs 111 \
  --lr_drop 40 \
  --resume "${BASE_EXP_DIR}/t2/train/checkpoints/checkpoint_latest.pth"

sleep 5

run_stage "${BASE_EXP_DIR}/t3" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t3_train \
  --epochs 121 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2361 \
  --exemplar_replay_prev_file learned_owod_t2_ft.txt \
  --exemplar_replay_cur_file learned_owod_t3_ft.txt \
  --lr 2e-5 \
  --resume "${BASE_EXP_DIR}/t2_ft/train/checkpoints/checkpoint_latest.pth" \

sleep 5

run_stage "${BASE_EXP_DIR}/t3_ft" \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set "UOD_CH4/learned_owod_t3_ft" \
  --epochs 181 \
  --lr_drop 35 \
  --resume "${BASE_EXP_DIR}/t3/train/checkpoints/checkpoint_latest.pth"

sleep 5

run_stage "${BASE_EXP_DIR}/t4" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t4_train \
  --epochs 191 \
  --freeze_prob_model \
  --exemplar_replay_selection \
  --exemplar_replay_max_length 2749 \
  --exemplar_replay_prev_file learned_owod_t3_ft.txt \
  --exemplar_replay_cur_file learned_owod_t4_ft.txt \
  --num_inst_per_class 40 \
  --resume "${BASE_EXP_DIR}/t3_ft/train/checkpoints/checkpoint_latest.pth" \
  --lr 2e-5

sleep 5

run_stage "${BASE_EXP_DIR}/t4_ft" \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set "UOD_CH4/learned_owod_t4_ft" \
  --epochs 261 \
  --lr_drop 50 \
  --resume "${BASE_EXP_DIR}/t4/train/checkpoints/checkpoint_latest.pth"
