#!/usr/bin/env bash
set -euo pipefail



CKPT="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t4_ft/checkpoint0260.pth"
SPLIT="owod_all_task_test"
OUTDIR="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t4_ft/debug"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTDIR}"

python "${REPO_ROOT}/tools/uod_collect_mining_debug.py" \
  --resume "${CKPT}" \
  --split "${SPLIT}" \
  --output_dir_debug "${OUTDIR}/raw" \
  "$@"

python "${REPO_ROOT}/tools/uod_render_mining_debug.py" \
  --input_dir "${OUTDIR}/raw" \
  --output_dir "${OUTDIR}/viz"
