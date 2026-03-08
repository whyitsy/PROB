#!/bin/bash



# 这里的CUDA_VISIBLE_DEVICES的数量需要与run_dist_launch.sh的第一个参数保持一致
# CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 configs/M_OWOD_BENCHMARK.sh

CUDA_VISIBLE_DEVICES=1,2,3 ./tools/run_dist_launch.sh 3 configs/M_OWOD_BENCHMARK_dummy_label.sh
