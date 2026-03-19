#!/bin/bash

# GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/EVAL_M_OWOD_BENCHMARK.sh
CUDA_VISIBLE_DEVICES=0 ./tools/run_dist_launch.sh 1 33069  configs/M_OWOD_BENCHMARK_innov_2_eval.sh
