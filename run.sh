#!/bin/bash



# 这里的CUDA_VISIBLE_DEVICES的数量需要与run_dist_launch.sh的第一个参数保持一致
# CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 33064 configs/M_OWOD_BENCHMARK.sh

# CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 33065 configs/M_OWOD_BENCHMARK_dummy_label.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 33066 configs/M_OWOD_BENCHMARK_dummy_label_innov_1.sh

# CUDA_VISIBLE_DEVICES=0,3 ./tools/run_dist_launch.sh 2 33067 configs/M_OWOD_BENCHMARK_innov_2.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 127.0.0.1 33068 configs/M_OWOD_BENCHMARK.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 127.0.0.1 33069 configs/M_OWOD_BENCHMARK_UOD_CH3.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 127.0.0.1 33060 configs/M_OWOD_BENCHMARK_UOD_CH4.sh
