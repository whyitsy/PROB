#!/usr/bin/env bash

set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

BASE_EXP_DIR="${BASE_EXP_DIR:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/mnt/data/kky/output/PROB/exps/OUTPUTS/MOWODB/UOD_CH3_EVAL}"

DEFAULT_STAGES=(t1)

COMMON_EVAL_ARGS=(
  --model_type uod
  --with_box_refine
  --eval
  --eval_batch_size 15
)

CH3_ARGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
)

resolve_checkpoint() {
  local stage_src_dir="$1"
  local latest_epoch_ckpt

  if [[ -f "${stage_src_dir}/train/checkpoints/checkpoint_latest.pth" ]]; then
    printf '%s\n' "${stage_src_dir}/train/checkpoints/checkpoint_latest.pth"
    return 0
  fi

  latest_epoch_ckpt="$(find "${stage_src_dir}/train/checkpoints" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pth' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${latest_epoch_ckpt}" ]]; then
    printf '%s\n' "${latest_epoch_ckpt}"
    return 0
  fi

  if [[ -f "${stage_src_dir}/checkpoint.pth" ]]; then
    printf '%s\n' "${stage_src_dir}/checkpoint.pth"
    return 0
  fi

  echo "Failed to resolve checkpoint under ${stage_src_dir}" >&2
  return 1
}

run_stage_eval() {
  local stage_name="$1"
  local stage_src_dir="${BASE_EXP_DIR}/${stage_name}"
  local stage_out_dir="${EVAL_OUTPUT_DIR}/${stage_name}"
  local checkpoint_path

  checkpoint_path="$(resolve_checkpoint "${stage_src_dir}")"
  mkdir -p "${stage_out_dir}"

  torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
    "${ROOT_DIR}/main_open_world.py" \
    --output_dir "${stage_out_dir}" \
    --eval_checkpoint "${checkpoint_path}" \
    "${COMMON_EVAL_ARGS[@]}" \
    "${CH3_ARGS[@]}"
}

run_eval_pipeline() {
  local stages=("$@")

  if [[ ${#stages[@]} -eq 0 ]]; then
    stages=("${DEFAULT_STAGES[@]}")
  fi

  for stage_name in "${stages[@]}"; do
    run_stage_eval "${stage_name}"
  done
}
