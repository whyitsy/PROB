#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=1,2,3
# bash configs/ablation_t1/ABLATION_T1_UOD_CH3_CORE.sh


sleep 10
export CUDA_VISIBLE_DEVICES=1,2,3
bash configs/ablation_t1/ABLATION_T1_UOD_CH4_ON_CH3.sh




