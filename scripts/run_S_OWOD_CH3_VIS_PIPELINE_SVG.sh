#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/EVAL/S_OWODB/CH3_Full_VIS_PIPELINE_SVG.sh"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

RERUN_EVAL=1 run_visual_pipeline "$@"
