#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3

CH3_CONFIG_PATH="configs/EVAL/M_OWODB/CH3_EVAL.sh"
source "${CH3_CONFIG_PATH}"

RERUN_EVAL=1 run_visual_pipeline 



# CH4_CONFIG_PATH="configs/EVAL/M_OWODB/CH4_EVAL.sh"
# source "${CH4_CONFIG_PATH}"

# RERUN_EVAL=1 run_visual_pipeline 