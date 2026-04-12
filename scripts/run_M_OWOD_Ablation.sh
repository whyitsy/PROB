#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash configs/M_OWOD_BENCHMARK/ablation_t1/ABLATION_T1_UOD_CH3_CORE.sh


bash configs/M_OWOD_BENCHMARK/ablation_t1/ABLATION_T1_UOD_CH4_ON_CH3.sh
