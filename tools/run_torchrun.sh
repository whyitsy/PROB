#!/usr/bin/env bash
set -euo pipefail


CONFIG_SH=$1

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=gpu \
  "${CONFIG_SH}"