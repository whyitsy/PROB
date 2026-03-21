#!/usr/bin/env bash

# 第四章方法（创新点2）的完整四阶段主实验脚本.
# 在第三章基础上，再打开 uod_enable_decorr，同时加入 uod_orth_loss_coef 和 uod_decorr_loss_coef。它的作用是给你第四章整章主结果，并验证“第四章 = 第三章 + 解耦优化”。
echo running chapter-4 UOD training, M-OWODB dataset
set -x

EXP_DIR=/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH4
TAG=UOD_CH4
PY_ARGS=${@:1}

COMMON_UOD_ARGS="--model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_decorr \
 --unk_loss_coef 0.3 --uod_pseudo_unk_loss_coef 0.4 --uod_bg_unk_loss_coef 0.2 \
 --uod_pseudo_obj_loss_coef 0.3 --uod_obj_neg_loss_coef 0.2 \
 --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.05 \
 --uod_start_epoch 8 --uod_neg_warmup_epochs 3 \
 --uod_pos_quantile 0.25 --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
 --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
 --uod_pos_per_img_cap 1 --uod_neg_per_img 1 --uod_batch_topk_max 8 --uod_batch_topk_ratio 0.25"

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 41 \
    --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --exemplar_replay_selection --exemplar_replay_max_length 850 \
    --exemplar_replay_dir ${TAG} --exemplar_replay_cur_file 'learned_owod_t1_ft.txt' \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set 'owod_t2_train' --test_set 'owod_all_task_test' --epochs 51 \
    --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --exemplar_replay_selection --exemplar_replay_max_length 1743 --exemplar_replay_dir ${TAG} \
    --exemplar_replay_prev_file 'learned_owod_t1_ft.txt' --exemplar_replay_cur_file 'learned_owod_t2_ft.txt' \
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --lr 2e-5 \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set "${TAG}/learned_owod_t2_ft" --test_set 'owod_all_task_test' --epochs 111 --lr_drop 40 \
    --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth" \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set 'owod_t3_train' --test_set 'owod_all_task_test' --epochs 121 \
    --obj_loss_coef 8e-4 --freeze_prob_model --obj_temp 1.3 \
    --exemplar_replay_selection --exemplar_replay_max_length 2361 --exemplar_replay_dir ${TAG} \
    --exemplar_replay_prev_file 'learned_owod_t2_ft.txt' --exemplar_replay_cur_file 'learned_owod_t3_ft.txt' \
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" --lr 2e-5 \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "${TAG}/learned_owod_t3_ft" --test_set 'owod_all_task_test' --epochs 181 --lr_drop 35 \
    --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --pretrain "${EXP_DIR}/t3/checkpoint0120.pth" \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set 'owod_t4_train' --test_set 'owod_all_task_test' --epochs 191 \
    --obj_loss_coef 8e-4 --freeze_prob_model --obj_temp 1.3 \
    --exemplar_replay_selection --exemplar_replay_max_length 2749 --exemplar_replay_dir ${TAG} \
    --exemplar_replay_prev_file 'learned_owod_t3_ft.txt' --exemplar_replay_cur_file 'learned_owod_t4_ft.txt' \
    --num_inst_per_class 40 \
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" --lr 2e-5 \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set "${TAG}/learned_owod_t4_ft" --test_set 'owod_all_task_test' --epochs 261 --lr_drop 50 \
    --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --pretrain "${EXP_DIR}/t4/checkpoint0190.pth" \
    ${COMMON_UOD_ARGS} --viz --viz_num_samples 12 --viz_tb_images 4 ${PY_ARGS}
