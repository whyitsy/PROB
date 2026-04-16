#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash configs/S_OWOD_BENCHMARK/albation_t1/ABLATION_T1_CH3.sh

bash configs/S_OWOD_BENCHMARK/albation_t1/ABLATION_T1_CH4_ON_CH3.sh
