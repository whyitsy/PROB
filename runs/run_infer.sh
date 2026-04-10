#!/usr/bin/env bash
set -euo pipefail

python tools/visualize_infer.py \
    --model_type uod \
    --resume /mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t1/checkpoint.pth \
    --output_dir ./ch3/t1/infer \
    --input_dir ./photos \
    --glob "*.*" \
    --save_debug \
    --with_box_refine \
    --uod_enable_unknown \
    --uod_enable_pseudo \
    --uod_enable_batch_dynamic \
    --uod_enable_cls_soft_attn \
