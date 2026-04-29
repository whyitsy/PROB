#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS"

CHECKPOINT="/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH3_FULL/t4_ft/train/checkpoints/checkpoint_latest.pth"
OUTPUT_DIR="${ROOT_DIR}/infer_output"
DEVICE="cuda"
MODEL_TYPE="uod"

USE_TEST_SET=1
INPUT_PATH="photos"
DATA_ROOT="/mnt/data/kky/datasets/owdetr/data/OWOD"
DATASET="OWDETR"
TEST_SET="owdetr_test"
EVERY_N=100
START_INDEX=0
MAX_IMAGES=50

KNOWN_SCORE_THRESHOLD=0.4
UNKNOWN_SCORE_THRESHOLD=0.50
NMS_IOU_THRESHOLD=0.30
MIN_BOX_AREA_RATIO=0.02
MIN_BOX_SIDE_RATIO=0.03
MAX_BOX_ASPECT_RATIO=5.0
UNKNOWN_SCORE_SCALE=2

if [ "${USE_TEST_SET}" -eq 1 ]; then
  python infer.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --model_type "${MODEL_TYPE}" \
    --known_score_threshold "${KNOWN_SCORE_THRESHOLD}" \
    --unknown_score_threshold "${UNKNOWN_SCORE_THRESHOLD}" \
    --nms_iou_threshold "${NMS_IOU_THRESHOLD}" \
    --min_box_area_ratio "${MIN_BOX_AREA_RATIO}" \
    --min_box_side_ratio "${MIN_BOX_SIDE_RATIO}" \
    --max_box_aspect_ratio "${MAX_BOX_ASPECT_RATIO}" \
    --unknown_score_scale "${UNKNOWN_SCORE_SCALE}" \
    --use_test_set \
    --data_root "${DATA_ROOT}" \
    --dataset "${DATASET}" \
    --test_set "${TEST_SET}" \
    --every_n "${EVERY_N}" \
    --start_index "${START_INDEX}" \
    --max_images "${MAX_IMAGES}"
else
  python infer.py \
    --checkpoint "${CHECKPOINT}" \
    --input "${INPUT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --model_type "${MODEL_TYPE}" \
    --known_score_threshold "${KNOWN_SCORE_THRESHOLD}" \
    --unknown_score_threshold "${UNKNOWN_SCORE_THRESHOLD}" \
    --nms_iou_threshold "${NMS_IOU_THRESHOLD}" \
    --min_box_area_ratio "${MIN_BOX_AREA_RATIO}" \
    --min_box_side_ratio "${MIN_BOX_SIDE_RATIO}" \
    --max_box_aspect_ratio "${MAX_BOX_ASPECT_RATIO}" \
    --unknown_score_scale "${UNKNOWN_SCORE_SCALE}"
fi
