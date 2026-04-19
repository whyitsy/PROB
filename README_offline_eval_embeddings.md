# Offline eval embedding workflow

## Goal

During training-time eval, keep the usual qualitative panels and lightweight statistics,
but do **not** run PCA / t-SNE / UMAP embedding rendering online.

Instead:

- online eval only saves the embedding input data
- embedding figures are rendered later with a standalone script

This avoids slowing down multi-GPU training because the online embedding stage can be much slower than the other eval visualizations.

---

## Current behavior

The default visualization config now uses:

- `save_query_stats_csv = True`
- `save_feature_npz = True`
- `save_feature_embedding_plots = False`

That means training-time eval still saves:

- `eval/visualizations/epoch_xxxx/stats/query_statistics.csv`
- `eval/visualizations/epoch_xxxx/stats/feature_samples.npz`

but it does **not** render embedding figures online.

Other eval visualizations remain enabled as usual.

---

## Offline rendering script

Use this script after training:

```bash
python tools/render_eval_embeddings_offline.py \
  --stats_dir /path/to/output/eval/visualizations/epoch_0004/stats
```

This will read the saved CSV / NPZ files and render the embedding figures under:

- `stats/embeddings/feature/...`
- `stats/embeddings/score_space/...`

---

## Common examples

Render all default methods and dimensions:

```bash
python tools/render_eval_embeddings_offline.py \
  --stats_dir /path/to/output/eval/visualizations/epoch_0004/stats
```

Render only PCA 2D:

```bash
python tools/render_eval_embeddings_offline.py \
  --stats_dir /path/to/output/eval/visualizations/epoch_0004/stats \
  --methods pca \
  --dims 2
```

Render PCA and UMAP for both 2D and 3D:

```bash
python tools/render_eval_embeddings_offline.py \
  --stats_dir /path/to/output/eval/visualizations/epoch_0004/stats \
  --methods pca,umap \
  --dims 2,3
```

Render PNG instead of SVG:

```bash
python tools/render_eval_embeddings_offline.py \
  --stats_dir /path/to/output/eval/visualizations/epoch_0004/stats \
  --figure_format png
```

---

## Script arguments

- `--stats_dir`: required, path to one epoch's `stats` directory
- `--methods`: comma-separated methods, for example `pca,tsne,umap`
- `--dims`: comma-separated dimensions, for example `2,3`
- `--figure_format`: output figure format, usually `svg` or `png`

---

## Recommended usage

For training-time stability:

- keep online embedding rendering disabled
- keep `query_statistics.csv` and `feature_samples.npz` enabled
- run the offline script only for epochs you really want to inspect

A practical workflow is:

1. train normally with `--viz`
2. wait for `eval/visualizations/epoch_xxxx/stats` to be written
3. choose one or a few epochs of interest
4. run the offline script on those epochs only

---

## Notes

The offline script reuses the same embedding plotting logic as the online path.
It only changes **when** the embedding figures are rendered, not **how** they are rendered.
