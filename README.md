# PROB Visualization and Inference Guide

This README documents the **current canonical usage** for the PROB repository's
manual visualization pipeline and standalone image inference workflow.

The goal is to make the repository easier to use after the visualization
entrypoints were consolidated back into their formal names.

---

## 1. Canonical entrypoints

After consolidating the temporary versioned scripts, the workflows should be
invoked through the following **formal entrypoints**.

### Visualization pipeline

- `tools/mine_representative_cases_svg.py`
- `tools/render_mined_cases_svg.py`
- `tools/organize_rendered_cases_svg.py`
- `tools/plot_uod_manifold_3d_svg.py`
- `configs/EVAL/M_OWODB/CH3_Full_VIS_PIPELINE_SVG.sh`
- `scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh`

### Standalone inference

- `infer.py`
- `configs/INFER/UOD_INFER.sh`
- `scripts/run_infer.sh`

These are now the only entrypoints you should use. Any temporary `v2` / `v3`
variants should be considered obsolete once the cleanup PR is merged.

---

## 2. What the visualization pipeline does

The visualization pipeline is intended for **post-training analysis** and thesis
figure generation. It combines:

1. official evaluation visualization outputs,
2. standalone 3D statistics plots,
3. representative case mining,
4. batch rendering of single-case mechanism figures,
5. gallery/atlas generation for browsing and figure selection.

### Main capabilities

- qualitative detection visualization,
- unknown / known representative case mining,
- GT-aware future-unknown filtering for unknown representative cases,
- deformable query sampling figures,
- ODQE gate figures,
- joint query mechanism figures,
- query trajectory figures,
- SVG atlas boards for browsing.

---

## 3. Recommended usage: visualization pipeline

### 3.1 Most common command

Run a single stage, for example `t1`:

```bash
bash scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh t1
```

Run multiple stages:

```bash
bash scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh t1 t2 t2_ft
```

If no stage argument is given, the default stage list in the config file is
used.

---

## 4. Visualization configuration

Main config file:

```bash
configs/EVAL/M_OWODB/CH3_Full_VIS_PIPELINE_SVG.sh
```

### Key variables

#### Path variables

- `BASE_EXP_DIR`
  - root directory of trained experiments/checkpoints.
- `OUTPUTS_VIS_DIR`
  - output directory for manual visualization results.

#### Mining / rendering variables

- `MINE_MAX_SAMPLES`
  - maximum number of evaluation samples scanned for representative mining.
- `MINE_TOP_K`
  - number of representative candidates retained per category before rendering.
- `UNKNOWN_GT_MIN_IOU`
  - minimum overlap between an unknown representative case and a future-unknown
    GT box.
- `RENDER_PER_CATEGORY_LIMIT`
  - number of final rendered cases per category.
- `ATLAS_PER_GROUP_LIMIT`
  - number of cases shown per group in the SVG atlas boards.

#### Execution control

- `RERUN_EVAL`
  - `0`: reuse existing evaluation outputs and only regenerate offline figures.
  - `1`: rerun official eval before offline visualization.

---

## 5. Practical visualization examples

### 5.1 Reuse existing eval results and regenerate offline figures only

```bash
RERUN_EVAL=0 bash scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh t1
```

### 5.2 Increase the number of representative cases

```bash
MINE_TOP_K=40 RENDER_PER_CATEGORY_LIMIT=15 ATLAS_PER_GROUP_LIMIT=15 \
bash scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh t1
```

### 5.3 Make unknown representative cases more strict

```bash
UNKNOWN_GT_MIN_IOU=0.20 bash scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh t1
```

---

## 6. Output structure of the visualization pipeline

For a stage like `t1`, outputs are typically written under:

```bash
${OUTPUTS_VIS_DIR}/t1/
```

### 6.1 Official eval outputs

```bash
eval/visualizations/epoch_xxxx/
```

This includes:

- official qualitative panels,
- debug panels,
- statistics CSV/NPZ,
- official SVG distribution plots.

### 6.2 Offline rendered mechanism figures

```bash
infer/rendered_cases/
```

This includes per-case figures such as:

- `*_sampling.svg`
- `*_gate_curve.svg`
- `*_gate_heatmap.svg`
- `*_joint_mechanism.svg`
- `*_trajectory.svg`

### 6.3 Representative case manifests

```bash
infer/representative_cases/
```

This includes:

- `representative_case_manifest.json`
- `known_top_cases.csv`
- `unknown_top_cases.csv`
- `odqe_salient_top_cases.csv`
- representative contact sheets.

### 6.4 Figure atlas

```bash
infer/figure_atlas/
```

This includes:

- `boards/*.svg`
- `INDEX.md`
- `atlas_manifest.json`

Use this directory for browsing and selecting figures for papers or reports.

---

## 7. Interpretation of representative mining categories

### `known`

These are GT-aware known representative cases, ranked from model predictions with
geometry validity and overlap preference.

### `unknown`

These are **future-unknown GT-aware** representative cases.

This is important:

- they are **not** simply the highest final unknown-score queries,
- they are mined using the training pseudo-mining logic,
- they must also satisfy overlap constraints with future-unknown GT boxes.

### `odqe_salient`

These are unknown candidates that are additionally salient with respect to the
ODQE gate behavior.

---

## 8. Standalone inference for external images

The standalone inference path is intended for running a trained checkpoint on
external images and exporting:

- JSON results,
- all-box visualization,
- known-only visualization,
- unknown-only visualization,
- optional layer summary debug plots.

### Main command

```bash
bash scripts/run_infer.sh
```

---

## 9. Inference configuration

Main config file:

```bash
configs/INFER/UOD_INFER.sh
```

### Key variables

#### Paths

- `CHECKPOINT`
  - model checkpoint used for inference.
- `INPUT_PATH`
  - either a single image path or a directory of images.
- `OUTPUT_DIR`
  - output directory for inference results.

#### Thresholds and filtering

- `KNOWN_SCORE_THRESH`
- `UNKNOWN_SCORE_THRESH`
- `NMS_IOU`
- `MIN_AREA_RATIO`
- `MIN_SIDE_RATIO`
- `MAX_ASPECT_RATIO`
- `SAVE_LAYER_DEBUG`

These control known/unknown filtering and geometric cleanup.

---

## 10. Practical inference examples

### 10.1 Standard run

```bash
bash scripts/run_infer.sh
```

### 10.2 Tighter unknown filtering

```bash
UNKNOWN_SCORE_THRESH=0.30 bash scripts/run_infer.sh
```

### 10.3 Reduce tiny or strange boxes

```bash
MIN_AREA_RATIO=0.004 MIN_SIDE_RATIO=0.05 bash scripts/run_infer.sh
```

### 10.4 Reduce duplicated boxes

```bash
NMS_IOU=0.40 bash scripts/run_infer.sh
```

---

## 11. Output structure of standalone inference

Under:

```bash
${OUTPUT_DIR}/
```

you will typically find:

### 11.1 JSON results

```bash
json/
```

Per image:

- full detection list,
- label index,
- score,
- known/unknown flag.

### 11.2 Visualizations

```bash
vis/
```

Per image:

- `*_all.svg`
- `*_known.svg`
- `*_unknown.svg`

### 11.3 Optional layer-level summary

```bash
debug/
```

If enabled, this contains per-image layer summary figures.

---

## 12. Suggested workflow for thesis figures

### For evaluation-based figures

1. Run the visualization pipeline.
2. Browse `infer/rendered_cases/` for single-case mechanism figures.
3. Browse `infer/figure_atlas/boards/` for atlas-level selection.
4. Use `eval/visualizations/epoch_xxxx/` when you need official qualitative or
   debug panels.

### For external image demonstration

1. Edit `configs/INFER/UOD_INFER.sh`.
2. Run `bash scripts/run_infer.sh`.
3. Use `vis/*_known.svg` and `vis/*_unknown.svg` as presentation outputs.

---

## 13. Notes on figure semantics

### Sampling figures

Show where a query samples multi-scale deformable features across decoder layers.

### Trajectory figures

Show how a query evolves through decoder layers in terms of:

- box geometry,
- score changes,
- class tendency,
- ODQE behavior.

### Atlas boards

Atlas boards are for **browsing and selecting figures**. They are not always the
final figures you will place directly in a paper.

### Unknown representative figures

These should now be interpreted as:

- future-unknown GT-aware representative cases,
- not merely top-ranked raw unknown-score queries.

---

## 14. Troubleshooting

### Problem: too few representative figures

Increase:

```bash
MINE_TOP_K
RENDER_PER_CATEGORY_LIMIT
ATLAS_PER_GROUP_LIMIT
```

### Problem: unknown figures still look noisy

Increase:

```bash
UNKNOWN_GT_MIN_IOU
```

and/or tighten inference-side geometry thresholds.

### Problem: too many tiny boxes in inference

Increase:

```bash
MIN_AREA_RATIO
MIN_SIDE_RATIO
```

### Problem: duplicated predictions

Decrease:

```bash
NMS_IOU
```

---

## 15. Minimal quick start

### Visualization

```bash
RERUN_EVAL=0 bash scripts/run_M_OWOD_CH3_VIS_PIPELINE_SVG.sh t1
```

### Inference

```bash
bash scripts/run_infer.sh
```

---

## 16. Maintenance note

If you later add new visualization utilities, keep them routed through the
canonical formal entrypoints rather than introducing temporary `v2` / `v3`
variants again. This keeps the repository easier to maintain and easier to use.
