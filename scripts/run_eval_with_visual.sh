#!/usr/bin/env bash
set -euo pipefail


EXP_ROOT="/mnt/data/kky/output/PROB/exps/OUTPUTS"
EXP_NAME="MOWODB_CH3_EVAL"
CKPT="/mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t1/checkpoint.pth"

DATA_ROOT="/mnt/data/kky/datasets/owdetr/data/OWOD"
DATASET="TOWOD"
TRAIN_SET="owod_t1_train"
TEST_SET="owod_all_task_test"

MODEL_TYPE="uod"
DEVICE="cuda"

PREV_INTRODUCED_CLS=0
CUR_INTRODUCED_CLS=20
NUM_CLASSES=81

NUM_WORKERS=12
BATCH_SIZE=5

MANUAL_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${EXP_ROOT}/${EXP_NAME}__${MANUAL_TAG}"

# For ODQE-related tools
UOD_FLAGS=(
  --uod_enable_unknown
  --uod_enable_pseudo
  --uod_enable_odqe
)

COMMON_ARGS=(
  --output_dir "${OUT_DIR}"
  --device "${DEVICE}"
  --data_root "${DATA_ROOT}"
  --dataset "${DATASET}"
  --train_set "${TRAIN_SET}"
  --test_set "${TEST_SET}"
  --model_type "${MODEL_TYPE}"
  --PREV_INTRODUCED_CLS "${PREV_INTRODUCED_CLS}"
  --CUR_INTRODUCED_CLS "${CUR_INTRODUCED_CLS}"
  --num_classes "${NUM_CLASSES}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  "${UOD_FLAGS[@]}"
)

echo "[1/5] Manual eval with official visualization..."
python main_open_world.py \
  --resume "${CKPT}" \
  --eval \
  --viz \
  "${COMMON_ARGS[@]}"

LATEST_EVAL_DIR=$(find "${OUT_DIR}/eval/visualizations" -maxdepth 1 -type d -name "epoch_*" | sort | tail -n 1)
if [[ -z "${LATEST_EVAL_DIR}" ]]; then
  echo "ERROR: No eval visualization directory found."
  exit 1
fi
STATS_DIR="${LATEST_EVAL_DIR}/stats"

echo "[2/5] Plot 3D manifold / score-space figures..."
python tools/plot_uod_manifold_3d.py \
  --stats_dir "${STATS_DIR}"

echo "[3/5] Mine representative cases..."
python tools/mine_representative_cases.py \
  --checkpoint "${CKPT}" \
  --split eval \
  --start_index 0 \
  --max_samples 300 \
  --top_k 9 \
  "${COMMON_ARGS[@]}"

REP_MANIFEST="${OUT_DIR}/infer/representative_cases/representative_case_manifest.json"
if [[ ! -f "${REP_MANIFEST}" ]]; then
  echo "ERROR: representative_case_manifest.json not found."
  exit 1
fi

echo "[4/5] Render mined cases..."
python tools/render_mined_cases.py \
  --checkpoint "${CKPT}" \
  --manifest "${REP_MANIFEST}" \
  --categories known,unknown,odqe_salient \
  --per_category_limit 3 \
  --render_modes sampling,gate,joint,trajectory \
  "${COMMON_ARGS[@]}"

RENDER_MANIFEST="${OUT_DIR}/infer/rendered_cases/render_manifest.json"
if [[ ! -f "${RENDER_MANIFEST}" ]]; then
  echo "ERROR: render_manifest.json not found."
  exit 1
fi

echo "[5/5] Organize figure atlas..."
python tools/organize_rendered_cases.py \
  --render_manifest "${RENDER_MANIFEST}" \
  --representative_manifest "${REP_MANIFEST}" \
  --output_dir "${OUT_DIR}/infer/figure_atlas"

echo
echo "Done."
echo "Official eval outputs: ${OUT_DIR}/eval"
echo "Offline mechanism outputs: ${OUT_DIR}/infer"
echo "Figure atlas: ${OUT_DIR}/infer/figure_atlas"