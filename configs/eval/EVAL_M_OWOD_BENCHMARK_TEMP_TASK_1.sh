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

# C3_0_PROB_baseline 
# task_dir='/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE/C3_0_PROB_Baseline'
# C3_1
# task_dir='/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE/C3_1_UnknownOnly'
# C3_2
# task_dir='/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE/C3_2_Pseudo'
# C3_3
task_dir='/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH3_CORE_SOFT_ATTEN/C3_3_UnknownPseudoBatchDynamic'

# C4_0
# C4_3


EXP_DIR="${1:-${task_dir}}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

# MODEL_TYPE="prob"
# OBJ_TEMPS="1.3"
MODEL_TYPE="uod"
OBJ_TEMPS="0.7 1.0 1.3 1.6 2.0"
OBJ_LOSS_COEF="${OBJ_LOSS_COEF:-8e-4}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-eval_objtemp}"
MODEL_FLAGS="${MODEL_FLAGS:-}"
COMMON_FLAGS="${COMMON_FLAGS:-}"



run_eval() {
  local temp="$1"

  if [ ! -f "${EXP_DIR}/checkpoint.pth" ]; then
    echo "[WARN] checkpoint not found, skip: ${EXP_DIR}/checkpoint.pth"
    return 0
  fi

  python -u main_open_world.py \
    --output_dir "${EXP_DIR}/${OUTPUT_PREFIX}_${temp//./p}" \
    --dataset TOWOD \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 20 \
    --train_set 'owod_t1_train' \
    --test_set 'owod_all_task_test' \
    --epochs 41 \
    --lr_drop 35 \
    --model_type "$MODEL_TYPE" \
    --obj_loss_coef "$OBJ_LOSS_COEF" \
    --obj_temp "$temp" \
    --pretrain "${EXP_DIR}/checkpoint.pth" \
    --uod_enable_cls_soft_attn \
    --eval \
    ${MODEL_FLAGS} \
    ${COMMON_FLAGS} \
    ${PY_ARGS}
}

for TEMP in ${OBJ_TEMPS}; do
  run_eval "$TEMP"
done
