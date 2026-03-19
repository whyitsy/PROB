#!/usr/bin/env bash

echo "Starting T1 ablation study for Innov1 and Innov2"

set -x
set -e

if [ $# -ge 1 ]; then
    BASE_EXP_DIR="$1"
    shift
else
    BASE_EXP_DIR="/gemini/output/ABLATION_T1_INNOV2"
fi

PY_ARGS=${@:1}

COMMON_T1_ARGS="\
    --model_type innov_2 \
    --enable_unk_label_obj --use_valid_mask --use_feature_align --use_vlm_distill \
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
    --exemplar_replay_max_length 850 \
    " 

    

CONFIGS=(
    # "B1_Innov2_NoUnkHead|"

    "B2_Innov2_Full|--enable_unk_head --train_unk_head --infer_with_unk_head --unk_loss_use_known_neg --unk_loss_use_dummy_neg --unk_loss_use_dummy_pos"
    
    "B3_Innov2_TrainOnly|--enable_unk_head --train_unk_head --unk_loss_use_known_neg --unk_loss_use_dummy_neg --unk_loss_use_dummy_pos" # 增加unk head保持原推理的效果
    
    "B4_Innov2_InferOnly|--enable_unk_head --infer_with_unk_head" # 单推理改进无unk head的效果
    
    "B5_Innov2_NO_KnownNeg|--enable_unk_head --train_unk_head --infer_with_unk_head --unk_loss_use_dummy_neg --unk_loss_use_dummy_pos" # 分析将匹配上的正样本作为unk head的负样本的作用

    "B6_Innov2_NO_DummyNeg|--enable_unk_head --train_unk_head --infer_with_unk_head --unk_loss_use_known_neg --unk_loss_use_dummy_pos" # 分析dummy neg的作用
)

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="${CONFIG%%|*}"
    EXTRA_ARGS="${CONFIG#*|}"

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

echo "All Innov2 T1 ablation experiments completed."