#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Compatible evaluation script for current PROB/UOD checkpoints.
# Usage examples:
#   OBJ_TEMPS="0.7 1.0 1.3 1.6 2.0" MODEL_TYPE=prob \
#   bash EVAL_M_OWOD_BENCHMARK_COMPAT.sh /path/to/PROB
#
#   OBJ_TEMPS="1.0 1.3 1.6" MODEL_TYPE=uod \
#   MODEL_FLAGS="--uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
#                --uod_enable_cls_soft_attn --uod_cls_soft_attn_alpha 0.5 --uod_cls_soft_attn_min 0.25" \
#   bash EVAL_M_OWOD_BENCHMARK_COMPAT.sh /path/to/UOD_CH3
# -----------------------------------------------------------------------------
set -x
set -euo pipefail

EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/PROB}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

MODEL_TYPE="${MODEL_TYPE:-prob}"
OBJ_TEMPS="${OBJ_TEMPS:-1.3}"
OBJ_LOSS_COEF="${OBJ_LOSS_COEF:-8e-4}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-eval_objtemp}"
MODEL_FLAGS="${MODEL_FLAGS:-}"
COMMON_FLAGS="${COMMON_FLAGS:-}"

T1_CKPT="${T1_CKPT:-${EXP_DIR}/t1/checkpoint0040.pth}"
# T2_CKPT="${T2_CKPT:-${EXP_DIR}/t2_ft/checkpoint0110.pth}"
# T3_CKPT="${T3_CKPT:-${EXP_DIR}/t3_ft/checkpoint0180.pth}"
# T4_CKPT="${T4_CKPT:-${EXP_DIR}/t4_ft/checkpoint0260.pth}"

run_eval() {
  local tag="$1"
  local prev="$2"
  local cur="$3"
  local train_set="$4"
  local ckpt="$5"
  local temp="$6"

  if [ ! -f "$ckpt" ]; then
    echo "[WARN] checkpoint not found, skip: $ckpt"
    return 0
  fi

  python -u main_open_world.py \
    --output_dir "${EXP_DIR}/${OUTPUT_PREFIX}_${temp//./p}/${tag}" \
    --dataset TOWOD \
    --PREV_INTRODUCED_CLS "$prev" \
    --CUR_INTRODUCED_CLS "$cur" \
    --train_set "$train_set" \
    --test_set 'owod_all_task_test' \
    --epochs 191 \
    --lr_drop 35 \
    --model_type "$MODEL_TYPE" \
    --obj_loss_coef "$OBJ_LOSS_COEF" \
    --obj_temp "$temp" \
    --pretrain "$ckpt" \
    --eval \
    ${MODEL_FLAGS} \
    ${COMMON_FLAGS} \
    ${PY_ARGS}
}

for TEMP in ${OBJ_TEMPS}; do
  run_eval t1 0 20 owod_t1_train "$T1_CKPT" "$TEMP"
  # run_eval t2 20 20 owod_t2_train "$T2_CKPT" "$TEMP"
  # run_eval t3 40 20 owod_t3_train "$T3_CKPT" "$TEMP"
  # run_eval t4 60 20 owod_t4_train "$T4_CKPT" "$TEMP"
done
