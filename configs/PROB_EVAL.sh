#!/usr/bin/env bash

set -euo pipefail
set -x

torchrun --standalone --nnodes=1 --nproc-per-node=gpu main_open_world.py \
  --model_type prob \
  --eval \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --eval_checkpoint /mnt/data/kky/output/PROB/exps/PROB/MOWODB/t4.pth \
  --output_dir /mnt/data/kky/output/PROB/exps/OUTPUTS/PROB_BASELINE_EVAL/MOWODB

torchrun --standalone --nnodes=1 --nproc-per-node=gpu main_open_world.py \
  --model_type prob \
  --dataset OWDETR \
  --test_set owdetr_test \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --eval \
  --eval_checkpoint /mnt/data/kky/output/PROB/exps/PROB/SOWODB/t4.pth \
  --output_dir /mnt/data/kky/output/PROB/exps/OUTPUTS/PROB_BASELINE_EVAL/SOWODB