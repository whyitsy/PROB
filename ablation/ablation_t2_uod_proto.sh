#!/usr/bin/env bash

set -e
set -x

echo "Starting T2 prototype-memory ablation for merged UOD"

if [ $# -ge 2 ]; then
    BASE_EXP_DIR="$1"
    T1_CKPT="$2"
    shift 2
else
    echo "Usage: bash ablation_t2_uod_proto.sh <base_exp_dir> <t1_checkpoint> [extra args]"
    exit 1
fi

PY_ARGS=${@:1}
COMMON_T2_ARGS="\
    --dataset TOWOD \
    --PREV_INTRODUCED_CLS 20 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t2_train \
    --test_set owod_all_task_test \
    --epochs 51 \
    --obj_temp 1.3 \
    --lr 2e-5 \
    --lr_drop 35 \
    --batch_size 5 \
    --freeze_prob_model \
    --model_type uod \
    --enable_unk_label_obj \
    --use_valid_mask \
    --soft_valid_mask \
    --enable_unk_head \
    --train_unk_head \
    --infer_with_unk_head \
    --unk_loss_use_known_neg \
    --unk_loss_use_dummy_pos \
    --unk_neg_per_img 0 \
    --pretrain ${T1_CKPT}"

CONFIGS=(
    "P0_NoProto|"
    "P1_TransitionOnly|--enable_proto_memory --proto_transition_on --build_proto_memory"
    "P2_PreserveOnly|--enable_proto_memory --proto_consistency_on --build_proto_memory"
    "P3_FullProto|--enable_proto_memory --proto_transition_on --proto_consistency_on --build_proto_memory"
)

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="${CONFIG%%|*}"
    EXTRA_ARGS="${CONFIG#*|}"
    CURRENT_OUT_DIR="${BASE_EXP_DIR}/${EXP_NAME}/t2"
    python -u main_open_world_uod.py \
        --output_dir "${CURRENT_OUT_DIR}" \
        ${COMMON_T2_ARGS} \
        ${EXTRA_ARGS} \
        ${PY_ARGS}
done
