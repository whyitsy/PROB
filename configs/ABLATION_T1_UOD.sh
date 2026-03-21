#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# T1 总递进消融（U0 -> U3）
# 作用：最快判断整条方法链是否成立。
#
# U0: baseline PROB
# U1: + explicit unknownness
# U2: + sparse pseudo supervision + batch dynamic allocation
# U3: + orth + decorrelation
#
# 重点验证：
# 1) U1 vs U0：显式 unknownness 分支是否有效
# 2) U2 vs U1：第三章完整机制是否继续带来增益
# 3) U3 vs U2：第四章解耦优化是否主要改善 WI / A-OSE
# -----------------------------------------------------------------------------


echo running T1 ablations for UOD
set -x
set -euo pipefail

EXP_ROOT="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"
COMMON="--dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --obj_loss_coef 8e-4 --obj_temp 1.3 --viz --viz_num_samples 12 --viz_tb_images 4"

# Baseline PROB
# python -u main_open_world.py --output_dir "${EXP_ROOT}/P0_prob" ${COMMON} --model_type prob ${PY_ARGS}

# Chapter 3 core: explicit unknownness only
python -u main_open_world.py --output_dir "${EXP_ROOT}/U1_unknown" ${COMMON} --model_type uod --uod_enable_unknown ${PY_ARGS}

# Chapter 3 full: unknownness + sparse pseudo + batch dynamic
python -u main_open_world.py --output_dir "${EXP_ROOT}/U2_ch3" ${COMMON} --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic ${PY_ARGS}

# Chapter 4: add decoupled optimization
python -u main_open_world.py --output_dir "${EXP_ROOT}/U3_ch4" ${COMMON} --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_decorr ${PY_ARGS}
