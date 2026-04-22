#!/usr/bin/env bash


set -euo pipefail
# set -x

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_EXP_DIR="${BASE_EXP_DIR:-/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH3_FULL}"
OUTPUTS_VIS_DIR="${OUTPUTS_VIS_DIR:-/mnt/data/kky/output/PROB/exps/OUTPUTS/SOWODB/UOD_CH3_FULL_VIS}"

DEFAULT_STAGES=(
  t1
)

COMMON_EVAL_ARGS=(
  --model_type uod
  --with_box_refine
  --viz
  --eval
  --dataset OWDETR
  --train_set owdetr_t1_train
  --test_set owdetr_test
  --eval_batch_size 20
)

CH3_ARGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_batch_dynamic
  --uod_enable_cls_soft_attn
)

MINE_MAX_SAMPLES="${MINE_MAX_SAMPLES:-300}"
MINE_TOP_K="${MINE_TOP_K:-9}"
RENDER_PER_CATEGORY_LIMIT="${RENDER_PER_CATEGORY_LIMIT:-3}"
RENDER_CATEGORIES="${RENDER_CATEGORIES:-known,unknown,odqe_salient}"
RENDER_MODES="${RENDER_MODES:-sampling,gate,joint,trajectory}"
ATLAS_PER_GROUP_LIMIT="${ATLAS_PER_GROUP_LIMIT:-3}"
RERUN_EVAL="${RERUN_EVAL:-0}"

run_python_module() {
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m "$@"
}

resolve_checkpoint() {
  local stage_src_dir="$1"
  local latest_epoch_ckpt

  if [[ -f "${stage_src_dir}/train/checkpoints/checkpoint_latest.pth" ]]; then
    printf '%s\n' "${stage_src_dir}/train/checkpoints/checkpoint_latest.pth"
    return 0
  fi

  latest_epoch_ckpt="${stage_src_dir}/train/checkpoints/checkpoint_latest.pth"
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

latest_eval_epoch_dir() {
  local stage_out_dir="$1"
  local eval_root="${stage_out_dir}/eval/visualizations"

  if [[ ! -d "${eval_root}" ]]; then
    return 1
  fi

  find "${eval_root}" -maxdepth 1 -type d -name 'epoch_*' | sort | tail -n 1
}

run_stage_eval() {
  local stage_name="$1"
  local stage_src_dir="${BASE_EXP_DIR}/${stage_name}"
  local stage_out_dir="${OUTPUTS_VIS_DIR}/${stage_name}"
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

run_stage_offline_visualization() {
  local stage_name="$1"
  local stage_src_dir="${BASE_EXP_DIR}/${stage_name}"
  local stage_out_dir="${OUTPUTS_VIS_DIR}/${stage_name}"
  local checkpoint_path
  local latest_eval_dir
  local stats_dir
  local representative_manifest
  local render_manifest

  checkpoint_path="$(resolve_checkpoint "${stage_src_dir}")"
  latest_eval_dir="$(latest_eval_epoch_dir "${stage_out_dir}")"
  stats_dir="${latest_eval_dir}/stats"


  run_python_module tools.mine_representative_cases_svg \
    --checkpoint "${checkpoint_path}" \
    --split eval \
    --start_index 0 \
    --max_samples "${MINE_MAX_SAMPLES}" \
    --top_k "${MINE_TOP_K}" \
    --output_dir "${stage_out_dir}" \
    "${CH3_ARGS[@]}" \
    --model_type uod \
    --with_box_refine

  representative_manifest="${stage_out_dir}/infer/representative_cases/representative_case_manifest.json"

  run_python_module tools.render_mined_cases_svg \
    --checkpoint "${checkpoint_path}" \
    --manifest "${representative_manifest}" \
    --categories "${RENDER_CATEGORIES}" \
    --per_category_limit "${RENDER_PER_CATEGORY_LIMIT}" \
    --render_modes "${RENDER_MODES}" \
    --output_dir "${stage_out_dir}" \
    "${CH3_ARGS[@]}" \
    --model_type uod \
    --with_box_refine

  render_manifest="${stage_out_dir}/infer/rendered_cases/render_manifest.json"

  run_python_module tools.organize_rendered_cases_svg \
    --render_manifest "${render_manifest}" \
    --representative_manifest "${representative_manifest}" \
    --output_dir "${stage_out_dir}/infer/figure_atlas" \
    --per_group_limit "${ATLAS_PER_GROUP_LIMIT}"
}

run_stage_visual_pipeline() {
  local stage_name="$1"

  if [[ "${RERUN_EVAL}" == "1" ]]; then
    run_stage_eval "${stage_name}"
  fi

  run_stage_offline_visualization "${stage_name}"
}

run_visual_pipeline() {
  local stages=("$@")

  if [[ ${#stages[@]} -eq 0 ]]; then
    stages=("${DEFAULT_STAGES[@]}")
  fi

  for stage_name in "${stages[@]}"; do
    run_stage_visual_pipeline "${stage_name}"
  done
}
