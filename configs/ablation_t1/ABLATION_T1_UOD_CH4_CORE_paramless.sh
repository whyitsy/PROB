#!/usr/bin/env bash


set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_CORE_PARAMLESS_ALL}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

CH3_BEST_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --obj_loss_coef 8e-3 \
  --model_type uod_paramless \
  --uod_pseudo_unk_loss_coef 1.5 --uod_neg_per_img 50 \
  --uod_pseudo_obj_loss_coef 1.0 --uod_obj_neg_loss_coef 0.40 \
  --uod_pos_per_img_cap 2 --uod_batch_topk_max 16 --uod_batch_topk_ratio 0.5 \
  --uod_known_reject_thresh 0.2  --uod_min_pos_thresh 0.05 \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  "


# C4-6: ODQE + Orth + Decorr
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_6_ODQEPlusOrthPlusDecorr" \
  ${CH3_BEST_ARGS} \
  --uod_enable_odqe --uod_enable_decorr --uod_orth_loss_coef 2.0 --uod_decorr_loss_coef 2.0 \
  ${PY_ARGS}
sleep 10

# C4-0: ODQE only
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_0_CH3_Best" \
  --uod_enable_odqe \
  ${CH3_BEST_ARGS} \
  ${PY_ARGS}
sleep 10

# C4-1: Orth only
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_1_OrthOnly" \
  ${CH3_BEST_ARGS} \
  --uod_enable_decorr --uod_orth_loss_coef 2.0 --uod_decorr_loss_coef 0.0 \
  ${PY_ARGS}
sleep 10

# C4-2: Decorr only
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_2_DecorrOnly" \
  ${CH3_BEST_ARGS} \
  --uod_enable_decorr --uod_orth_loss_coef 0.0 --uod_decorr_loss_coef 2.0 \
  ${PY_ARGS}
sleep 10

# C4-3: ODQE + Decorr
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_3_DecorrOnly" \
  ${CH3_BEST_ARGS} \
  --uod_enable_odqe --uod_enable_decorr --uod_orth_loss_coef 0.0 --uod_decorr_loss_coef 2.0 \
  ${PY_ARGS}
sleep 10

# C4-4: ODQE + Orth
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_4_ODQEPlusOrth" \
  ${CH3_BEST_ARGS} \
  --uod_enable_odqe --uod_enable_decorr --uod_orth_loss_coef 2.0 --uod_decorr_loss_coef 0.0 \
  ${PY_ARGS}
sleep 10

# C4-5: Orth + Decorr
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_5_OrthPlusDecorr" \
  ${CH3_BEST_ARGS} \
  --uod_enable_decorr --uod_orth_loss_coef 2.0 --uod_decorr_loss_coef 2.0 \
  ${PY_ARGS}

