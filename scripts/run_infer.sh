#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS"

CHECKPOINT="/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH3_FULL/t1/train/checkpoints/checkpoint_latest.pth"
INPUT_PATH="photos"
OUTPUT_DIR="${ROOT_DIR}/infer_output"
DEVICE="cuda"
MODEL_TYPE="uod"

KNOWN_SCORE_THRESHOLD=0.30
UNKNOWN_SCORE_THRESHOLD=0.50
NMS_IOU_THRESHOLD=0.10
MIN_BOX_AREA_RATIO=0.02
MIN_BOX_SIDE_RATIO=0.03
MAX_BOX_ASPECT_RATIO=5.0
UNKNOWN_SCORE_SCALE=10

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
