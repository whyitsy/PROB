#!/bin/bash

# GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/EVAL_M_OWOD_BENCHMARK.sh
CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 127.0.0.1 33068   configs/M_OWOD_BENCHMARK_innov_1_eval.sh
