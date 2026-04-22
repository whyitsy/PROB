#!/usr/bin/env bash
set -euo pipefail
set -x

# 在仓库根目录执行这个脚本
# 例如：bash run_vis_ch3.sh

CHECKPOINT="/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH4_FULL/t4_ft/train/checkpoints/checkpoint_latest.pth"
DATA_ROOT="/mnt/data/kky/datasets/owdetr/data/OWOD"

DUMP_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS/SOWODB/ch3_t4_ft"
FIG_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS/SOWODB/ch3_t4_ft"

mkdir -p "${DUMP_DIR}"
mkdir -p "${FIG_DIR}"

python tools/extract_vis_uod.py \
  --eval_checkpoint "${CHECKPOINT}" \
  --save_dir "${DUMP_DIR}" \
  --model_type uod \
  --with_box_refine \
  --dataset OWDETR \
  --data_root "${DATA_ROOT}" \
  --train_set owdetr_t1_train \
  --test_set owdetr_test \
  --num_classes 81 \
  --device cuda \
  --eval_batch_size 5 \
  --num_workers 4 \
  --PREV_INTRODUCED_CLS 60 \
  --CUR_INTRODUCED_CLS 20 \
  --obj_temp 1.3 \
  --uod_known_temp 1.3 \
  --uod_enable_unknown \
  --uod_enable_pseudo \
  --uod_enable_batch_dynamic \
  --uod_enable_cls_soft_attn \
  --dump_every_n 20 \
  --dump_max_images 80

python tools/render_vis_uod.py overlay \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/overlay" \
  --max_images 8 \
  --score_thr 0.15 \
  --max_det 30 \
  --iou_thr 0.5

python tools/render_vis_uod.py mining \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/mining" \
  --max_images 8

python tools/render_vis_uod.py histograms \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/histograms" \
  --bins 40

python tools/render_vis_uod.py box_evolution \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/box_evolution" \
  --max_images 8