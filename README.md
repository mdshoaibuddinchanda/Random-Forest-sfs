# Feature-Selection Stability in High-Dimensional Gene Expression Data

## Overview

This project investigates the **stability of feature selection** when
Random Forest classifiers are applied to high-dimensional microarray data.
The central question: *does enforcing selection stability across resampled
folds improve predictive performance on a clinically meaningful endpoint?*

## Dataset

**GSE39582** — Affymetrix Human Genome U133 Plus 2.0.  Colon cancer, 585
samples, ~54 000 probes (RMA-normalised, distributed as a GEO Series
Matrix file).

Source: `data/Raw/GSE39582_series_matrix.txt`

For other datasets, pass the file with `--data` and choose a label column
with `--label-column`.

## Labels

| Column      | Values |                           Meaning                        |
|-------------|--------|----------------------------------------------------------|
| `rfs.event` | 1 / 0  | Relapse-free survival event.  1 = relapse; 0 = censored. |

**Justification.**  Relapse-free survival is a direct clinical outcome
that captures disease recurrence.  It is the standard prognostic endpoint
in colon-cancer studies and avoids proxy variables (e.g. tumour location)
whose relationship to prognosis is indirect.

Samples with missing `rfs.event` values are excluded.

## Methods

### Pre-processing

1. **Parsing** — the Series Matrix is parsed into an expression matrix
   (samples × probes) and a clinical metadata table.
2. **Variance filtering** — probes in the bottom 10th percentile by
   variance are removed.  This is an unsupervised step (does not use
   labels) that reduces dimensionality and noise.

### Experimental conditions

| # |       Condition       |                                                       Description                                                       |
|---|-----------------------|-------------------------------------------------------------------------------------------------------------------------|
| 1 | **Baseline (full)**   | All features after variance filtering.  No feature selection.                                                           |
| 2 | **Single-shot top-K** | One Random Forest trained on the full dataset; top-K features by importance are kept.                                   |
| 3 | **Stability-aware**   | Repeated stratified K-fold; top-K features extracted per fold; only features selected in ≥ 60 % of folds are retained.  |

### Classifier

Random Forest with 500 trees, balanced class weights, all other
parameters at scikit-learn defaults.

### Evaluation

Each condition is evaluated using **5-fold stratified cross-validation
repeated with 5 different random seeds**.  Metrics are reported as
**mean ± standard deviation** to capture performance uncertainty.

Metrics: accuracy, balanced accuracy, ROC-AUC.

### Stability metrics

- **Pairwise Jaccard similarity** — computed for every pair of
  fold-level feature sets.  Reported as mean ± std with a distribution
  histogram.
- **Selection-frequency distribution** — per-feature count normalised by
  the total number of folds.

## Outputs

|              File                                   |                    Description                           |
|-----------------------------------------------------|----------------------------------------------------------|
| `outputs/<dataset_name>/results.csv`                | Comparison table (3 conditions × 6 metrics + stability). |
| `outputs/<dataset_name>/run_summary.json`           | Run metadata (sample count, class balance, parameters).  |
| `outputs/<dataset_name>/stable_features.txt`        | Probe IDs retained by the stability-aware method.        |
| `outputs/<dataset_name>/feature_frequencies.csv`    | Per-probe selection frequency.                           |
| `outputs/<dataset_name>/feature_stability_bar.png`  | Top-30 features by selection frequency.                  |
| `outputs/<dataset_name>/frequency_distribution.png` | Histogram of all selection frequencies.                  |
| `outputs/<dataset_name>/jaccard_distribution.png`   | Histogram of pairwise Jaccard scores.                    |
| `outputs/<dataset_name>/model_comparison.png`       | Grouped bar chart comparing 3 conditions.                |

## Usage

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run (first time — full parse + compute)
python src/pipeline.py --data data/Raw/GSE39582_series_matrix.txt

# Example: GSE17536 (uses dfs_event with textual labels)
python src/pipeline.py --data data/Raw/GSE17536_series_matrix.txt --label-column "dfs_event (disease free survival; cancer recurrence)" --label-positive "recurrence" --label-negative "no recurrence"

# Resume from checkpoints (if interrupted)
python src/pipeline.py --data data/Raw/GSE39582_series_matrix.txt --resume
```

### Parameters

|        Flag            | Default   |                 Meaning                      |
|------------------------|-----------|----------------------------------------------|
| `--seed`               | 42        | Global random seed                           |
| `--n-splits`           | 5         | Folds per CV                                 |
| `--n-repeats`          | 5         | Repeats for stability resampling             |
| `--top-k`              | 100       | Features selected per fold                   |
| `--freq-threshold`     | 0.6       | Minimum selection frequency for retention    |
| `--min-features`       | 50        | Minimum features if threshold is too strict  |
| `--variance-percentile`| 10.0      | Bottom variance percentile to remove         |
| `--n-eval-seeds`       | 5         | Seeds for performance uncertainty estimation |
| `--checkpoint-every`   | 10        | Checkpoint interval (stability folds)        |
| `--label-column`       | rfs.event | Metadata column to use as the label          |
| `--label-positive`     | 1         | Value treated as positive class              |
| `--label-negative`     | 0         | Value treated as negative class              |

## Reproducibility

- All randomness is seeded.
- Checkpointing allows interruption and resumption.
- The pipeline reads from a single file (`GSE39582_series_matrix.txt`)
  and produces all outputs deterministically.

## Requirements

- Python ≥ 3.10
- numpy, pandas, scikit-learn, matplotlib

See `requirements.txt`.
