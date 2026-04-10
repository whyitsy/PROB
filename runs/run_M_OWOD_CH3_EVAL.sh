#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash configs/EVAL/M_OWODB/CH3_Full_EVAL.sh