#!/usr/bin/env bash

set -euo pipefail

# eval 需要显存：31234MB - 23366MB = 7858MB, 5 batchsize
export CUDA_VISIBLE_DEVICES=0,1


CH3_CONFIG_PATH="configs/EVAL/S_OWODB/CH3_EVAL.sh"
source "${CH3_CONFIG_PATH}"

RERUN_EVAL=1 run_visual_pipeline



# CH4_CONFIG_PATH="configs/EVAL/S_OWODB/CH4_EVAL.sh"
# source "${CH4_CONFIG_PATH}"

# RERUN_EVAL=1 run_visual_pipeline