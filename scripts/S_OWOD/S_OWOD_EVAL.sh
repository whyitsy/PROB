#!/usr/bin/env bash

set -euo pipefail

# eval 需要显存：31234MB - 23366MB = 7858MB, 5 batchsize
export CUDA_VISIBLE_DEVICES=0,1,2,3

CH3_CONFIG_PATH="configs/EVAL/S_OWODB/CH3_EVAL.sh"
source "${CH3_CONFIG_PATH}"

run_eval_pipeline t4_ft
