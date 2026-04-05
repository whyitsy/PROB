#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash configs/M_OWOD_BENCHMARK/M_OWOD_CH3.sh

bash configs/M_OWOD_BENCHMARK/M_OWOD_CH4.sh
