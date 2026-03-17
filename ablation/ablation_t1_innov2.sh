#!/usr/bin/env bash

echo "Starting T1 ablation study for Innov1 and Innov2"

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
    --batch_size 2 \
    --exemplar_replay_selection \
    --exemplar_replay_max_length 850"

CONFIGS=(
    "B1_Innov2_NoUnkHead|ABLT_B1_I2_NOHEAD|--model_type innov_2 --enable_unk_label_obj --use_valid_mask --etop --tdqi --use_feature_align --align_loss_coef 2.0 --use_vlm_distill --unk_loss_coef 0.0"

    "B2_Innov2_Full|ABLT_B2_I2_FULL|--model_type innov_2 --enable_unk_label_obj --use_valid_mask --etop --tdqi --use_feature_align --align_loss_coef 2.0 --use_vlm_distill --enable_unk_head --unk_loss_coef 1.0 --unk_pos_per_img 1 --unk_neg_per_img 2 --unk_cls_reject_thresh 0.25 --postproc_known_thresh 0.05 --postproc_unknown_thresh 0.05"
)

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="${CONFIG%%|*}"
    REST="${CONFIG#*|}"
    RUN_NAME="${REST%%|*}"
    EXTRA_ARGS="${REST#*|}"

    CURRENT_OUT_DIR="${BASE_EXP_DIR}/${EXP_NAME}/t1"

    echo "=========================================================="
    echo "Running Ablation: ${EXP_NAME}"
    echo "Run name: ${RUN_NAME}"
    echo "Args: ${EXTRA_ARGS}"
    echo "Output: ${CURRENT_OUT_DIR}"
    echo "=========================================================="

    python -u main_open_world.py \
        --output_dir "${CURRENT_OUT_DIR}" \
        ${COMMON_T1_ARGS} \
        --exemplar_replay_dir "${RUN_NAME}" \
        --exemplar_replay_cur_file learned_owod_t1_ft.txt \
        ${EXTRA_ARGS} \
        ${PY_ARGS}

    echo "Finished ${EXP_NAME}"
    echo "----------------------------------------------------------"
done

echo "All Innov2 T1 ablation experiments completed."