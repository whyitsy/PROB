#!/usr/bin/env bash

echo running evaluation for M_OWOD_BENCHMARK with Innov1

set -x
set -e



PY_ARGS=${@:1}
RUN_NAME=PROB_V1
EXP_DIR=/mnt/data/kky/output/PROB/exps/MOWODB/Innov1/A3_Innov1_Mining_ValidMask
# A3_Innov1_Mining_ValidMask的配置
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 41\
    --model_type 'innov_1'  --obj_temp 1.3\
    --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${RUN_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt"\
    --enable_unk_label_obj --use_valid_mask --unk_label_start_epoch 8 --unk_label_obj_warmup_epochs 2\
    --pretrain "/mnt/data/kky/output/PROB/exps/MOWODB/ABLATION_T1_INNOV1/A3_Innov1_Mining_ValidMask/t1/checkpoint0040.pth" \
    --eval \
    ${PY_ARGS}
    

# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t2" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20\
#     --train_set 'owod_t2_train' --test_set 'owod_all_task_test' --epochs 51\
#     --model_type 'innov_1'  --obj_temp 1.3 --freeze_innov_1_model\
#     --exemplar_replay_selection --exemplar_replay_max_length 1743 --exemplar_replay_dir ${RUN_NAME}\
#     --exemplar_replay_prev_file "learned_owod_t1_ft.txt" --exemplar_replay_cur_file "learned_owod_t2_ft.txt"\
#     --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --lr 2e-5\
#     --enable_unk_label_obj
#     ${PY_ARGS}
    

# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t2_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
#     --train_set "${RUN_NAME}/learned_owod_t2_ft" --test_set 'owod_all_task_test' --epochs 111 --lr_drop 40\
#     --model_type 'prob'  --obj_temp 1.3\
#     --pretrain "${EXP_DIR}/t2/checkpoint0050.pth"\
#     --enable_unk_label_obj
#     ${PY_ARGS}
    
    
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t3" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20\
#     --train_set 'owod_t3_train' --test_set 'owod_all_task_test' --epochs 121\
#     --model_type 'prob'  --freeze_prob_model --obj_temp 1.3\
#     --exemplar_replay_selection --exemplar_replay_max_length 2361 --exemplar_replay_dir ${RUN_NAME}\
#     --exemplar_replay_prev_file "learned_owod_t2_ft.txt" --exemplar_replay_cur_file "learned_owod_t3_ft.txt"\
#     --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" --lr 2e-5 \
#     --enable_unk_label_obj
#     ${PY_ARGS}
    
    
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t3_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
#     --train_set "${RUN_NAME}/learned_owod_t3_ft" --test_set 'owod_all_task_test' --epochs 181 --lr_drop 35\
#     --model_type 'prob'  --obj_temp 1.3\
#     --pretrain "${EXP_DIR}/t3/checkpoint0120.pth"\
#     --enable_unk_label_obj
#     ${PY_ARGS}
    
    
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t4" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
#     --train_set 'owod_t4_train' --test_set 'owod_all_task_test' --epochs 191 \
#     --model_type 'prob'  --freeze_prob_model --obj_temp 1.3\
#     --exemplar_replay_selection --exemplar_replay_max_length 2749 --exemplar_replay_dir ${RUN_NAME}\
#     --exemplar_replay_prev_file "learned_owod_t3_ft.txt" --exemplar_replay_cur_file "learned_owod_t4_ft.txt"\
#     --num_inst_per_class 40\
#     --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" --lr 2e-5\
#     --enable_unk_label_obj
#     ${PY_ARGS}
    
    
# PY_ARGS=${@:1}
# python -u main_open_world.py \
#     --output_dir "${EXP_DIR}/t4_ft" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
#     --train_set "${RUN_NAME}/learned_owod_t4_ft" --test_set 'owod_all_task_test' --epochs 261 --lr_drop 50\
#     --model_type 'prob'  --obj_temp 1.3\
#     --pretrain "${EXP_DIR}/t4/checkpoint0190.pth" \
#     --enable_unk_label_obj
#     ${PY_ARGS}