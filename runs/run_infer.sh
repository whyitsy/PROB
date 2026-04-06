#!/usr/bin/env bash
set -e

CKPT="$1"
IMAGE="$2"
OUTDIR="$3"

shift 3

common_args="
    --dataset TOWOD \
    --model_type uod \
    --with_box_refine \
    --uod_enable_unknown \
    --uod_enable_cls_soft_attn \
    --uod_enable_odqe \
    --uod_enable_decorr
  "

python tools/inference_visualize.py \
  --resume "${CKPT}" \
  --image_path "${IMAGE}" \
  --output_dir "${OUTDIR}" \
  "$@"