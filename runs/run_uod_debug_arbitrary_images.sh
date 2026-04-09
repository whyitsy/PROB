#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_uod_debug_arbitrary_images.sh <ckpt> <image_dir_or_csv_paths> <outdir> [extra args...]"
  echo "Directory mode example:"
  echo "  bash run_uod_debug_arbitrary_images.sh outputs/t2/checkpoint.pth /data/my_imgs debug/arbitrary --image_dir /data/my_imgs --xml_dir /data/my_xml --model_type uod --dataset OWDETR --data_root /path/to/OWOD --split t2_test"
  echo "Path-list mode example:"
  echo "  bash run_uod_debug_arbitrary_images.sh outputs/t2/checkpoint.pth a.jpg,b.jpg debug/arbitrary --image_paths a.jpg,b.jpg --xml_paths a.xml,b.xml --model_type uod --dataset OWDETR --data_root /path/to/OWOD --split t2_test"
  exit 1
fi

CKPT="$1"
IMG_SPEC="$2"
OUTDIR="$3"
shift 3

EXTRA=("$@")
HAS_IMAGE_FLAG=0
for arg in "${EXTRA[@]}"; do
  if [ "$arg" = "--image_dir" ] || [ "$arg" = "--image_paths" ]; then
    HAS_IMAGE_FLAG=1
    break
  fi
done

CMD=(python tools/uod_debug_arbitrary_images.py --resume "$CKPT" --output_dir_debug "$OUTDIR" --render_overlay --render_hist)
if [ $HAS_IMAGE_FLAG -eq 0 ]; then
  if [ -d "$IMG_SPEC" ]; then
    CMD+=(--image_dir "$IMG_SPEC")
  else
    CMD+=(--image_paths "$IMG_SPEC")
  fi
fi
CMD+=("${EXTRA[@]}")
"${CMD[@]}"
