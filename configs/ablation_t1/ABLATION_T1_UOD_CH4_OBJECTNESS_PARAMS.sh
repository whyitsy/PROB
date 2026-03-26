#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Chapter 4 objectness-focused sweeps on top of C4 best + cls soft attenuation.
# Orth/Decorr fixed; objectness-side terms are swept.
# -----------------------------------------------------------------------------
set -x
set -euo pipefail

BASE_EXP_DIR="${1:-/mnt/data/kky/output/PROB/exps/MOWODB/UOD_ABL_T1_CH4_OBJECTNESS_PARAMS}"
shift $(( $# > 0 ? 1 : 0 )) || true
PY_ARGS="${@:1}"

BASE_ARGS="\
  --dataset TOWOD \
  --PREV_INTRODUCED_CLS 0 \
  --CUR_INTRODUCED_CLS 20 \
  --train_set owod_t1_train \
  --test_set owod_all_task_test \
  --epochs 41 \
  --batch_size 5 \
  --eval_every 5 \
  --num_workers 8 \
  --obj_temp 1.3 \
  --model_type uod --uod_enable_unknown --uod_enable_pseudo --uod_enable_batch_dynamic \
  --uod_enable_decorr --uod_orth_loss_coef 0.05 --uod_decorr_loss_coef 0.05 \
  --uod_enable_cls_soft_attn --uod_cls_soft_attn_min 0.25 \
  --unk_loss_coef 0.30 --uod_pseudo_unk_loss_coef 0.40 --uod_bg_unk_loss_coef 0.20 \
  --uod_start_epoch 8 --uod_neg_warmup_epochs 3 \
  --uod_pos_quantile 0.25 --uod_pos_scale 1.2 --uod_min_pos_thresh 0.08 \
  --uod_known_reject_thresh 0.15 --uod_neg_margin 0.8 \
  --uod_pos_per_img_cap 1 --uod_neg_per_img 1 \
  --uod_batch_topk_max 8 --uod_batch_topk_ratio 0.25 \
  --viz --viz_num_samples 12 --viz_tb_images 4"

for V in 8e-4 2e-3 5e-3; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P1_obj_loss_coef_${V}" \
  ${BASE_ARGS} \
  --obj_loss_coef ${V} \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef 0.20 --uod_cls_soft_attn_alpha 0.50 \
  ${PY_ARGS}
done

for V in 0.20 0.30 0.50; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P2_pseudo_obj_coef_${V}" \
  ${BASE_ARGS} \
  --obj_loss_coef 2e-3 \
  --uod_pseudo_obj_loss_coef ${V} --uod_obj_neg_loss_coef 0.20 --uod_cls_soft_attn_alpha 0.50 \
  ${PY_ARGS}
done

for V in 0.10 0.20 0.40; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P3_obj_neg_coef_${V}" \
  ${BASE_ARGS} \
  --obj_loss_coef 2e-3 \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef ${V} --uod_cls_soft_attn_alpha 0.50 \
  ${PY_ARGS}
done

for V in 0.00 0.35 0.50 0.70; do
python -u main_open_world.py \
  --output_dir "${BASE_EXP_DIR}/P4_cls_soft_attn_alpha_${V}" \
  ${BASE_ARGS} \
  --obj_loss_coef 2e-3 \
  --uod_pseudo_obj_loss_coef 0.30 --uod_obj_neg_loss_coef 0.20 --uod_cls_soft_attn_alpha ${V} \
  ${PY_ARGS}
done
