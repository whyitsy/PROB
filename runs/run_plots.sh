#!/usr/bin/env bash
set -e

CKPT="$1"
OUTDIR="$2"

shift 2

python tools/collect_eval_debug.py \
  --resume "${CKPT}" \
  --output_dir "${OUTDIR}" \
  "$@"

python tools/plot_score_distributions.py \
  --records "${OUTDIR}/score_records.jsonl" \
  --output_dir "${OUTDIR}/plots"

python tools/plot_attention_maps.py \
  --npz_dir "${OUTDIR}/debug_npz" \
  --output_dir "${OUTDIR}/attention_plots"