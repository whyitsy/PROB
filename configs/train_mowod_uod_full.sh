#!/usr/bin/env bash

echo "Running full M-OWODB training pipeline: merged UOD"
set -x
set -e

if [ $# -ge 1 ]; then
    EXP_DIR="$1"
    shift
else
    EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_FULL"
fi

PY_ARGS=${@:1}
RUN_NAME=UOD_FULL

COMMON_ARGS="\
    --model_type uod \
    --dataset TOWOD \
    --test_set owod_all_task_test \
    --enable_unk_label_obj \
    --use_valid_mask --soft_valid_mask \
    --enable_unk_head --train_unk_head --infer_with_unk_head \
    --unk_loss_use_known_neg --unk_loss_use_dummy_pos \
    --unk_pos_per_img 1 \
    --unk_neg_per_img 0 \
    --unk_label_start_epoch 8 \
    --unk_label_obj_warmup_epochs 2 \
    --unk_label_pos_quantile 0.25 \
    --unk_label_obj_score_thresh 0.8 \
    --obj_neg_margin 1.0 \
    --bg_neg_score_margin 0.5 \
    --unk_cls_reject_thresh 0.2 \
    --unk_min_area 0.0015 \
    --unk_min_side 0.04 \
    --unk_max_aspect_ratio 6.0 \
    --unk_border_max_aspect_ratio 3.5 \
    --unk_max_iou 0.3 \
    --unk_max_iof 0.6 \
    --image_gate_min_valid_ratio 0.05 \
    --image_gate_min_low_energy_ratio 0.02 \
    --image_gate_min_pos_candidates 1 \
    --image_gate_known_mean_max 0.25 \
    --dummy_pos_cls_weight 0.25 \
    --postproc_known_thresh 0.05 \
    --postproc_unknown_thresh 0.10 \
    --enable_train_uod_vis \
    --train_uod_vis_freq 200 \
    --train_uod_vis_max_images 4 \
    --enable_eval_uod_vis \
    --eval_uod_vis_num_batches 1 \
    --eval_uod_vis_max_images 6 \
    --batch_size 5"

python -u main_open_world_uod.py \
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

python -u main_open_world_uod.py \
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

python -u main_open_world_uod.py \
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

python -u main_open_world_uod.py \
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

python -u main_open_world_uod.py \
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

python -u main_open_world_uod.py \
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

python -u main_open_world_uod.py \
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
