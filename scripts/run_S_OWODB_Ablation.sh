#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3

bash configs/S_OWOD_BENCHMARK/albation_t1/ABLATION_T1_CH3.sh

sleep 5

bash configs/S_OWOD_BENCHMARK/albation_t1/ABLATION_T1_CH4_ON_CH3.sh
