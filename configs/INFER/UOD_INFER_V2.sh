#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT="${CHECKPOINT:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t1/train/checkpoints/checkpoint_latest.pth}"
INPUT_PATH="${INPUT_PATH:-photos}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/infer_output_v2}"
DEVICE="${DEVICE:-cuda}"

KNOWN_SCORE_THRESH="${KNOWN_SCORE_THRESH:-0.35}"
UNKNOWN_SCORE_THRESH="${UNKNOWN_SCORE_THRESH:-0.20}"
NMS_IOU="${NMS_IOU:-0.50}"
MIN_AREA_RATIO="${MIN_AREA_RATIO:-0.002}"
MIN_SIDE_RATIO="${MIN_SIDE_RATIO:-0.03}"
MAX_ASPECT_RATIO="${MAX_ASPECT_RATIO:-5.0}"
SAVE_LAYER_DEBUG="${SAVE_LAYER_DEBUG:-1}"

run_infer_v2() {
  local extra_args=()
  if [[ "${SAVE_LAYER_DEBUG}" == "1" ]]; then
    extra_args+=(--save_layer_debug)
  fi

  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/infer_uod_v2.py" \
    --checkpoint "${CHECKPOINT}" \
    --input "${INPUT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --known_score_thresh "${KNOWN_SCORE_THRESH}" \
    --unknown_score_thresh "${UNKNOWN_SCORE_THRESH}" \
    --nms_iou "${NMS_IOU}" \
    --min_area_ratio "${MIN_AREA_RATIO}" \
    --min_side_ratio "${MIN_SIDE_RATIO}" \
    --max_aspect_ratio "${MAX_ASPECT_RATIO}" \
    "${extra_args[@]}"
}
