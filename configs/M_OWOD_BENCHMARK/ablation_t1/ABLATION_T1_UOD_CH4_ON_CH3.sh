#!/usr/bin/env bash
set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_ON_CH3_O4_03}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

CH3_BEST_ARGS="\
  --dataset TOWOD \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --num_workers 8 \
  --model_type uod \
  --with_box_refine \
  --unk_loss_coef 4e-4 \
  --uod_pseudo_obj_loss_coef 1 \
  --uod_pos_per_img_cap 1 --uod_batch_topk_max 12 \
  --uod_cls_soft_attn_alpha 0.5 --uod_cls_soft_attn_min 0.25 \
  --uod_haux_low_obj_coef 0 --uod_haux_mid_unknown_coef 0 --uod_haux_high_unknown_coef 0 \
  --uod_pos_unk_min 0.08 --uod_known_reject_thresh 0.10 \
  --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \
  "
  # --uod_pseudo_obj_loss_coef 1.5  --uod_pseudo_unk_loss_coef 0 --unk_loss_coef 8e-4 \
  # --uod_pos_per_img_cap 2 --uod_batch_topk_max 16 \
  # --uod_cls_soft_attn_alpha 0.5 --uod_cls_soft_attn_min 0.25 --uod_start_epoch 12 \
  # --uod_haux_low_obj_coef 0 --uod_haux_mid_unknown_coef 0 --uod_haux_high_unknown_coef 0 \
  # --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic --uod_enable_cls_soft_attn \



# C4_1: CH3-best + ODQE
torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
  main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_1_CH3Best_ODQE" \
  --uod_enable_odqe \
  ${CH3_BEST_ARGS} \
  ${PY_ARGS}


# C4_2: CH3-best + Decorr
torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
  main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_2_CH3Best_Decorr" \
  --uod_enable_decorr \
  ${CH3_BEST_ARGS} \
  ${PY_ARGS}


# C4_3: CH3-best + ODQE + Decorr
torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
  main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/C4_3_CH3Best_ODQE_Decorr" \
  --uod_enable_odqe \
  --uod_enable_decorr \
  ${CH3_BEST_ARGS} \
  ${PY_ARGS}

# 最好指标：61.02、22.1、7278.0、0.0600
# torchrun --standalone --nnodes=1 --nproc-per-node=gpu \
#   main_open_world.py \
#   --output_dir "${BASE_EXP_DIR}/C4_3_CH3Best_ODQE_Decorr_04_02" \
#   --uod_enable_odqe --uod_enable_decorr \
#   ${CH3_BEST_ARGS} \
#   ${PY_ARGS}

