#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 单张图
# python infer.py \
#     --checkpoint /mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t1/checkpoint.pth \
#     --input /path/to/image.jpg \
#     --output_dir /path/to/infer_output \
#     --score_thresh 0.05 \
#     --save_layer_debug

# 批量图
python infer.py \
    --checkpoint /mnt/data/kky/output/PROB/exps/MOWODB/UOD_CH3_FULL/t1/checkpoint.pth \
    --input photos \
    --mode_type uod \
    --output_dir ./infer_output \
    --score_thresh 0.05 \
    --save_layer_debug