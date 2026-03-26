#!/bin/bash


# CH3 CORE 消融
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 127.0.0.1 33063 configs/ablation_t1/ABLATION_T1_UOD_CH3_CORE_paramless.sh
sleep 10
# CH4 CORE 消融
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/run_dist_launch.sh 4 127.0.0.1 33060 configs/ablation_t1/ABLATION_T1_UOD_CH4_CORE_paramless.sh