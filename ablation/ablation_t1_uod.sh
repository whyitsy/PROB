#!/usr/bin/env bash

echo "Starting T1 ablation study for merged UOD"
set -x
set -e

if [ $# -ge 1 ]; then
    BASE_EXP_DIR="$1"
    shift
else
    BASE_EXP_DIR="/mnt/data/kky/output/PROB/exps/MOWODB/ABLATION_T1_UOD"
fi

PY_ARGS=${@:1}

COMMON_T1_ARGS="\
    --model_type uod \
    --dataset TOWOD \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t1_train \
    --test_set owod_all_task_test \
    --epochs 41 \
    --batch_size 5 \
    --enable_unk_label_obj \
    --use_valid_mask --soft_valid_mask \
    --enable_unk_head --train_unk_head --infer_with_unk_head \
    --unk_loss_use_known_neg --unk_loss_use_dummy_pos \
    --unk_pos_per_img 1 \
    --unk_neg_per_img 0 \
    --unk_label_start_epoch 5 \
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
    --exemplar_replay_selection \
    --exemplar_replay_max_length 850"

CONFIGS=(
    "U0_KnownOnly|--enable_unknown_output --unk_neg_per_img 0 --train_unk_head --infer_with_unk_head"
    "U1_Mining_NoSoftMask|--dummy_pos_cls_weight 0.0"
    "U2_Mining_SoftMask_Mainline|"
    "U3_AddReliableBgNeg|--unk_neg_per_img 1 --unk_loss_use_dummy_neg"
    "U4_NoImageGate|--image_gate_min_valid_ratio 0.0 --image_gate_min_low_energy_ratio 0.0 --image_gate_known_mean_max 1.0"
)

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME="${CONFIG%%|*}"
    EXTRA_ARGS="${CONFIG#*|}"
    CURRENT_OUT_DIR="${BASE_EXP_DIR}/${EXP_NAME}/t1"

    python -u main_open_world_uod.py \
        --output_dir "${CURRENT_OUT_DIR}" \
        ${COMMON_T1_ARGS} \
        --exemplar_replay_dir "${EXP_NAME}" \
        --exemplar_replay_cur_file learned_owod_t1_ft.txt \
        ${EXTRA_ARGS} \
        ${PY_ARGS}
done

