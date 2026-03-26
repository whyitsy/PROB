#!/bin/bash



# CUDA_VISIBLE_DEVICES=1,2 ./tools/run_dist_launch.sh 2 127.0.0.1 33068   configs/M_OWOD_BENCHMARK_innov_1_eval.sh


# 测试不同的obj_temp
# CUDA_VISIBLE_DEVICES=1 ./tools/run_dist_launch.sh 1 127.0.0.1 33069 configs/eval/EVAL_M_OWOD_BENCHMARK_TEMP_TASK_1.sh

CUDA_VISIBLE_DEVICES=1,2,3 ./tools/run_dist_launch.sh 3 127.0.0.1 33069 configs/eval/EVAL_M_OWOD_BENCHMARK_TEMP_TASK_1.sh


