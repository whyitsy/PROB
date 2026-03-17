#!/usr/bin/env bash

echo "Running full M-OWODB training pipeline: INNOV1"

set -x
set -e

if [ $# -ge 1 ]; then
    EXP_DIR="$1"
    shift
else
    EXP_DIR="/gemini/output/INNOV_1_FULL"
fi

PY_ARGS=${@:1}
RUN_NAME=INNOV_1_FULL

COMMON_ARGS="\
    --dataset TOWOD \
    --test_set owod_all_task_test \
    --obj_temp 1.3 \
    --model_type innov_1 \
    --enable_unk_label_obj \
    --use_valid_mask \
    --etop \
    --tdqi \
    --use_feature_align \
    --align_loss_coef 2.0 \
    --use_vlm_distill \
    --batch_size 2"

# -------------------------
# T1
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t1_train \
    --epochs 41 \
    --lr 1e-4 \
    --lr_drop 35 \
    --exemplar_replay_selection \
    --exemplar_replay_max_length 850 \
    --exemplar_replay_dir ${RUN_NAME} \
    --exemplar_replay_cur_file learned_owod_t1_ft.txt \
    ${PY_ARGS}

# -------------------------
# T2
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 20 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t2_train \
    --epochs 51 \
    --freeze_prob_model \
    --lr 2e-5 \
    --obj_loss_coef 10 \
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" \
    --exemplar_replay_selection \
    --exemplar_replay_max_length 1743 \
    --exemplar_replay_dir ${RUN_NAME} \
    --exemplar_replay_prev_file learned_owod_t1_ft.txt \
    --exemplar_replay_cur_file learned_owod_t2_ft.txt \
    ${PY_ARGS}

# -------------------------
# T2 FT
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 20 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set "${RUN_NAME}/learned_owod_t2_ft" \
    --epochs 111 \
    --lr 1e-4 \
    --lr_drop 40 \
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth" \
    --obj_loss_coef 10 \
    ${PY_ARGS}

# -------------------------
# T3
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 40 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t3_train \
    --epochs 121 \
    --freeze_prob_model \
    --lr 1e-5 \
    --obj_loss_coef 10 \
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" \
    --exemplar_replay_selection \
    --exemplar_replay_max_length 2361 \
    --exemplar_replay_dir ${RUN_NAME} \
    --exemplar_replay_prev_file learned_owod_t2_ft.txt \
    --exemplar_replay_cur_file learned_owod_t3_ft.txt \
    ${PY_ARGS}

# -------------------------
# T3 FT
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 40 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set "${RUN_NAME}/learned_owod_t3_ft" \
    --epochs 181 \
    --lr 1e-4 \
    --lr_drop 35 \
    --pretrain "${EXP_DIR}/t3/checkpoint0120.pth" \
    --obj_loss_coef 10 \
    ${PY_ARGS}

# -------------------------
# T4
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 60 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t4_train \
    --epochs 191 \
    --freeze_prob_model \
    --lr 1e-5 \
    --obj_loss_coef 10 \
    --num_inst_per_class 40 \
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" \
    --exemplar_replay_selection \
    --exemplar_replay_max_length 2749 \
    --exemplar_replay_dir ${RUN_NAME} \
    --exemplar_replay_prev_file learned_owod_t3_ft.txt \
    --exemplar_replay_cur_file learned_owod_t4_ft.txt \
    ${PY_ARGS}

# -------------------------
# T4 FT
# -------------------------
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" \
    ${COMMON_ARGS} \
    --PREV_INTRODUCED_CLS 60 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set "${RUN_NAME}/learned_owod_t4_ft" \
    --epochs 261 \
    --lr 1e-4 \
    --lr_drop 50 \
    --pretrain "${EXP_DIR}/t4/checkpoint0190.pth" \
    --obj_loss_coef 10 \
    ${PY_ARGS}