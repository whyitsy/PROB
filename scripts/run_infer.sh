#!/usr/bin/env bash
set -e

ROOT_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS/Infer_OUTPUT"

# ====== 直接改这里 ======
CHECKPOINT="/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH4_FULL/t1/train/checkpoints/checkpoint_latest.pth"
INPUT_PATH="photos"
OUTPUT_DIR="${ROOT_DIR}/infer_output"
DEVICE="cuda"
MODEL_TYPE="uod"

KNOWN_SCORE_THRESH=0.35
UNKNOWN_SCORE_THRESH=0.1
NMS_IOU=0.50
MIN_AREA_RATIO=0.002
MIN_SIDE_RATIO=0.03
MAX_ASPECT_RATIO=5.0
# =======================

python infer.py \
  --checkpoint "${CHECKPOINT}" \
  --input "${INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --model_type "${MODEL_TYPE}" \
  --known_score_thresh "${KNOWN_SCORE_THRESH}" \
  --unknown_score_thresh "${UNKNOWN_SCORE_THRESH}" \
  --nms_iou "${NMS_IOU}" \
  --min_area_ratio "${MIN_AREA_RATIO}" \
  --min_side_ratio "${MIN_SIDE_RATIO}" \
  --max_aspect_ratio "${MAX_ASPECT_RATIO}" \
  --uod_postprocess_unknown_scale 1 \
  --save_layer_debug