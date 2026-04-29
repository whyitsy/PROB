#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# bash configs/S_OWOD_BENCHMARK/ABLATION_CH3_T1.sh

# sleep 5

bash configs/S_OWOD_BENCHMARK/ABLATION_CH4_T1.sh
