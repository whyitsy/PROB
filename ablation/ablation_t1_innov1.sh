#!/usr/bin/env bash

echo "Starting T1 ablation study for Innov1"

set -x
set -e

if [ $# -ge 1 ]; then
    BASE_EXP_DIR="$1"
    shift
else
    BASE_EXP_DIR="/gemini/output/ABLATION_T1_INNOV"
fi

PY_ARGS=${@:1}

COMMON_T1_ARGS="\
    --dataset TOWOD \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t1_train \
    --test_set owod_all_task_test \
    --epochs 41 \
    --obj_temp 1.3 \
    --lr 2e-4 \
    --lr_drop 35 \
    --batch_size 5 \
    --exemplar_replay_selection \
    --exemplar_replay_max_length 850"

CONFIGS=(
    "A1_Innov1|--model_type innov_1 --enable_unk_label_obj --use_valid_mask --use_feature_align --align_loss_coef 2.0 --use_vlm_distill"
    
    "A0_Baseline_PROB|--model_type prob"

    "A2_Innov1_MiningOnly|--model_type innov_1 --enable_unk_label_obj"

    "A3_Innov1_Mining_ValidMask|--model_type innov_1 --enable_unk_label_obj --use_valid_mask"

    "A3_Innov1_Mining_ValidMask_ETOP|--model_type innov_1 --enable_unk_label_obj --use_valid_mask --etop"

    "A4_Innov1_enhance|--model_type innov_1 --enable_unk_label_obj --use_valid_mask --use_feature_align --align_loss_coef 2.0 --use_vlm_distill --etop --tdqi"
)

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="${CONFIG%%|*}" # 第一项
    EXTRA_ARGS="${CONFIG#*|}" # 第二部分

    CURRENT_OUT_DIR="${BASE_EXP_DIR}/${EXP_NAME}/t1"

    echo "=========================================================="
    echo "Running Ablation: ${EXP_NAME}"
    echo "Args: ${EXTRA_ARGS}"
    echo "Output: ${CURRENT_OUT_DIR}"
    echo "=========================================================="

    python -u main_open_world.py \
        --output_dir "${CURRENT_OUT_DIR}" \
        ${COMMON_T1_ARGS} \
        --exemplar_replay_dir "${EXP_NAME}" \
        --exemplar_replay_cur_file learned_owod_t1_ft.txt \
        ${EXTRA_ARGS} \
        ${PY_ARGS}

    echo "Finished ${EXP_NAME}"
    echo "----------------------------------------------------------"
done

echo "All Innov1 T1 ablation experiments completed."