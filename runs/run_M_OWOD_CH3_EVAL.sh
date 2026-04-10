#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

bash configs/eval/M_OWODB/CH3_Full_EVAL.sh