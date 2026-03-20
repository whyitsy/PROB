#!/bin/bash



# 这里的CUDA_VISIBLE_DEVICES的数量需要与run_dist_launch.sh的第一个参数保持一致
# CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 33064 configs/M_OWOD_BENCHMARK.sh

# CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 33065 configs/M_OWOD_BENCHMARK_dummy_label.sh

# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 33066 configs/M_OWOD_BENCHMARK_dummy_label_innov_1.sh

# CUDA_VISIBLE_DEVICES=0,3 ./tools/run_dist_launch.sh 2 33067 configs/M_OWOD_BENCHMARK_innov_2.sh

CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 127.0.0.1 33068 ablation/ablation_t1_uod.sh

# ./tools/run_dist_launch.sh 4 33068 configs/innov/train_mowod_innov2_full.sh

# chmod +x run.sh  ./tools/run_dist_launch.sh ./configs/innov/train_mowod_innov2_full.sh