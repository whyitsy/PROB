#!/usr/bin/env bash
set -euo pipefail
set -x

# 在仓库根目录执行这个脚本
# 例如：bash run_vis_ch4.sh

CHECKPOINT="/mnt/data/kky/output/PROB/exps/SOWODB/UOD_CH4_FULL/t4_ft/train/checkpoints/checkpoint_latest.pth"
DATA_ROOT="/mnt/data/kky/datasets/owdetr/data/OWOD"

DUMP_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS/SOWODB/ch4_t4_ft"
FIG_DIR="/mnt/data/kky/output/PROB/exps/OUTPUTS/SOWODB/ch4_t4_ft"

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
  --uod_enable_odqe \
  --uod_enable_decorr \
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

python tools/render_vis_uod.py odqe_sampling \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/odqe_sampling" \
  --max_images 8 \
  --layer_index -1 \
  --top_points 24

python tools/render_vis_uod.py gate_gain \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/gate_gain" \
  --max_images 8 \
  --layer_index -1 \
  --top_queries 12

python tools/render_vis_uod.py decorr \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/decorr" \
  --max_points 15000

python tools/render_vis_uod.py manifold \
  --dump_dir "${DUMP_DIR}" \
  --output_dir "${FIG_DIR}/manifold" \
  --max_points 15000 \
  --ellipsoid_quantile 0.95 \
  --kde_grid_size 30 \
  --kde_density_quantile 0.05 \
  --kde_band_ratio 0.1