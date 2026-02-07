"""
GSE39582 Feature-Selection Stability Pipeline
==============================================

Research question
-----------------
How stable are Random-Forest-derived feature rankings when applied to
high-dimensional gene-expression data, and does enforcing selection
stability improve predictive performance on a clinical outcome?

Dataset
-------
GSE39582 — Affymetrix Human Genome U133 Plus 2.0, colon cancer,
585 samples, ~54 000 probes (RMA-normalised in the Series Matrix).

Label
-----
**rfs.event** (relapse-free survival event):
  1 = relapse occurred, 0 = no relapse (censored).
Justification: RFS is a direct clinical endpoint reflecting disease
recurrence.  It is widely used in colon-cancer prognostic studies and
avoids proxy variables (e.g. tumour location) that conflate anatomy
with prognosis.

Experimental conditions
-----------------------
1. **Baseline (full)** — all features after variance filtering.
2. **Single-shot top-K** — one RF trained once; top-K features kept.
3. **Stability-aware** — resampling-based selection; features retained
   only if selected in ≥ freq_threshold fraction of folds.

Metrics
-------
- Accuracy, balanced accuracy, ROC-AUC reported as mean ± std across
  multiple evaluation seeds (not a single point estimate).
- Pairwise Jaccard distribution for stability quantification.
- Per-feature selection-frequency distribution.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data_path: Path
    output_dir: Path
    checkpoint_dir: Path
    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 5
    top_k: int = 100
    freq_threshold: float = 0.6
    min_features: int = 50
    checkpoint_every: int = 10
    variance_percentile: float = 10.0   # drop bottom 10 % by variance
    n_eval_seeds: int = 5               # seeds for uncertainty estimation
    # Label — relapse-free survival event (binary clinical outcome)
    label_column: str = "rfs.event"
    label_positive: str = "1"           # relapse occurred
    label_negative: str = "0"           # no relapse


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"')


def _rf_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )


def _derive_run_dirs(
    data_path: Path, output_root: Path, checkpoint_root: Path
) -> Tuple[Path, Path]:
    """
    Create per-dataset subfolders under output/checkpoint roots.

    Example
    -------
    data_path = data/Raw/GSE39582_series_matrix.txt
    output_root = outputs
    => outputs/GSE39582_series_matrix
    """
    dataset_name = data_path.stem
    return output_root / dataset_name, checkpoint_root / dataset_name


# ──────────────────────────────────────────────────────────────────────
# STEP 1 — Parsing & validation
# ──────────────────────────────────────────────────────────────────────

def _parse_characteristics(values: List[str]) -> List[Dict[str, str]]:
    """Parse one !Sample_characteristics_ch1 line into per-sample dicts."""
    parsed: List[Dict[str, str]] = [dict() for _ in values]
    for i, raw in enumerate(values):
        if ":" not in raw:
            continue
        key, val = raw.split(":", 1)
        parsed[i][key.strip()] = val.strip()
    return parsed


def _merge_characteristics(
    all_lines: List[List[str]], sample_ids: List[str]
) -> pd.DataFrame:
    merged = [dict() for _ in sample_ids]
    for line_values in all_lines:
        parsed = _parse_characteristics(line_values)
        for i, item in enumerate(parsed):
            merged[i].update(item)
    return pd.DataFrame(merged, index=sample_ids)


def parse_series_matrix(
    path: Path, temp_dir: Path
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO Series Matrix file.

    Returns
    -------
    series_meta : dict   — series-level metadata
    meta_df     : DataFrame — sample-level metadata (index = GSM IDs)
    x_df        : DataFrame — expression matrix (rows = samples, cols = probes)
    """
    series_meta: Dict[str, str] = {}
    sample_meta: Dict[str, List[str]] = {}
    sample_characteristics: List[List[str]] = []
    sample_ids: List[str] | None = None

    _ensure_dir(temp_dir)
    matrix_temp = temp_dir / "matrix.tsv"

    in_matrix = False
    with (
        path.open("r", encoding="utf-8", errors="replace") as f_in,
        matrix_temp.open("w", encoding="utf-8") as f_matrix,
    ):
        for line in f_in:
            line = line.rstrip("\n")
            if line.startswith("!Series_"):
                key, val = line.split("\t", 1)
                series_meta[key[1:]] = _strip_quotes(val)
                continue
            if line.startswith("!Sample_"):
                key, rest = line.split("\t", 1)
                key = key[1:]
                values = [_strip_quotes(v) for v in rest.split("\t")]
                if key == "Sample_geo_accession":
                    sample_ids = values
                elif key == "Sample_characteristics_ch1":
                    sample_characteristics.append(values)
                else:
                    sample_meta[key] = values
                continue
            if line.startswith("!series_matrix_table_begin"):
                in_matrix = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_matrix = False
                break
            if in_matrix:
                f_matrix.write(line + "\n")

    if sample_ids is None:
        raise ValueError("Missing !Sample_geo_accession line")
    if not matrix_temp.exists() or matrix_temp.stat().st_size == 0:
        raise ValueError("Matrix section not found or empty")

    for key, values in sample_meta.items():
        if len(values) != len(sample_ids):
            raise ValueError(f"Metadata length mismatch: {key}")

    matrix_df = pd.read_csv(matrix_temp, sep="\t")
    id_col = matrix_df.columns[0]
    matrix_sample_ids = list(matrix_df.columns[1:])
    if matrix_sample_ids != sample_ids:
        raise ValueError("Sample ID order mismatch between metadata and matrix")

    matrix_df = matrix_df.set_index(id_col)
    matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")
    if matrix_df.isna().any().any():
        raise ValueError("Non-numeric values in expression matrix")

    x_df = matrix_df.T
    x_df.index.name = "sample_id"

    meta_df = pd.DataFrame(sample_meta, index=sample_ids)
    char_df = _merge_characteristics(sample_characteristics, sample_ids)
    meta_df = pd.concat([meta_df, char_df], axis=1)

    if list(x_df.index) != sample_ids:
        raise ValueError("Sample alignment failed after transpose")

    print(f"Parsed: {x_df.shape[0]} samples × {x_df.shape[1]} probes")
    print(f"Clinical metadata columns: {list(char_df.columns)}")
    return series_meta, meta_df, x_df


# ──────────────────────────────────────────────────────────────────────
# STEP 2 — Label definition
# ──────────────────────────────────────────────────────────────────────

def define_labels(meta_df: pd.DataFrame, config: Config) -> pd.Series:
    """
    Extract binary labels from metadata.

    Label column : rfs.event (relapse-free survival event)
    Positive (1) : relapse occurred
    Negative (0) : no relapse (censored)

    Samples with missing or ambiguous values (e.g. "N/A") are excluded
    downstream by ``filter_labeled()``.
    """
    if config.label_column not in meta_df.columns:
        candidates = [c for c in meta_df.columns
                      if "event" in c.lower() or "rfs" in c.lower()]
        raise ValueError(
            f"Label column '{config.label_column}' not found. "
            f"Candidates: {candidates}. All columns: {list(meta_df.columns)}"
        )

    raw = meta_df[config.label_column].astype(str).str.strip().str.lower()
    label_map = {
        config.label_positive.lower(): 1,
        config.label_negative.lower(): 0,
    }
    y = raw.map(label_map)

    if y.notna().sum() == 0:
        unique_vals = sorted(raw.unique().tolist())
        raise ValueError(
            "No valid labels found. Check --label-column, --label-positive, "
            f"--label-negative. Unique values: {unique_vals}"
        )

    n_valid = y.notna().sum()
    n_missing = y.isna().sum()
    print(f"Label '{config.label_column}': {n_valid} valid, {n_missing} excluded")
    if n_valid == 0:
        raise ValueError("No valid labels found")
    return y


def filter_labeled(
    x_df: pd.DataFrame, meta_df: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Drop samples with missing labels; fail on NaN in expression data."""
    keep = y.notna()
    x_df = x_df.loc[keep]
    meta_df = meta_df.loc[keep]
    y = y.loc[keep].astype(int)

    if x_df.isna().any().any():
        raise ValueError("NaN in expression matrix after label filtering")

    counts = y.value_counts().to_dict()
    print(f"After filtering: {x_df.shape[0]} samples  |  class balance: {counts}")
    return x_df, meta_df, y


# ──────────────────────────────────────────────────────────────────────
# STEP 3 — Variance filtering
# ──────────────────────────────────────────────────────────────────────

def variance_filter(
    x_df: pd.DataFrame, percentile: float
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Remove probes with variance below the given percentile.

    Rationale
    ---------
    Near-zero-variance genes contribute noise without discriminative signal.
    Removing the bottom *percentile* % is standard practice in microarray
    analysis and reduces dimensionality without using label information
    (unsupervised, so no data leakage).

    Returns
    -------
    x_filtered : DataFrame — filtered expression matrix
    kept_cols  : Index     — retained probe IDs
    """
    variances = x_df.var(axis=0)
    threshold = np.percentile(variances, percentile)
    keep = variances > threshold
    x_filtered = x_df.loc[:, keep]

    n_removed = (~keep).sum()
    print(
        f"Variance filter: removed {n_removed}/{x_df.shape[1]} probes "
        f"(bottom {percentile}%, threshold={threshold:.4f})"
    )
    return x_filtered, x_filtered.columns


# ──────────────────────────────────────────────────────────────────────
# STEP 4 & 6 — Evaluation with uncertainty
# ──────────────────────────────────────────────────────────────────────

def evaluate_cv_with_uncertainty(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seeds: List[int],
) -> Dict[str, float]:
    """
    Evaluate RF across *multiple random seeds*, each producing a full
    stratified K-fold run.  Report mean ± std for every metric.

    Why multiple seeds?
    -------------------
    A single CV gives a point estimate that hides variability from the
    random forest and fold assignment.  By repeating with different seeds,
    we obtain an uncertainty band that is essential for honest comparison.
    """
    all_acc, all_bal, all_auc = [], [], []

    for seed in seeds:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=1, random_state=seed
        )
        preds, probas, targets = [], [], []

        for train_idx, test_idx in cv.split(x, y):
            model = _rf_model(seed)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            preds.extend(model.predict(x.iloc[test_idx]).tolist())
            probas.extend(model.predict_proba(x.iloc[test_idx])[:, 1].tolist())
            targets.extend(y.iloc[test_idx].tolist())

        all_acc.append(accuracy_score(targets, preds))
        all_bal.append(balanced_accuracy_score(targets, preds))
        all_auc.append(roc_auc_score(targets, probas))

    return {
        "accuracy_mean": float(np.mean(all_acc)),
        "accuracy_std":  float(np.std(all_acc)),
        "balanced_accuracy_mean": float(np.mean(all_bal)),
        "balanced_accuracy_std":  float(np.std(all_bal)),
        "roc_auc_mean": float(np.mean(all_auc)),
        "roc_auc_std":  float(np.std(all_auc)),
    }


# ──────────────────────────────────────────────────────────────────────
# STEP 4b — Single-shot feature selection (Baseline 2)
# ──────────────────────────────────────────────────────────────────────

def single_shot_top_k(
    x: pd.DataFrame, y: pd.Series, top_k: int, seed: int
) -> List[str]:
    """
    Train one RF on all data, extract importances, return top-K features.
    This is the naïve (unstable) baseline — no resampling, no stability.
    """
    model = _rf_model(seed)
    model.fit(x, y)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_k]
    return list(x.columns[top_idx])


# ──────────────────────────────────────────────────────────────────────
# STEP 5 — Stability-aware feature selection
# ──────────────────────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairwise_jaccard_scores(sets: List[set]) -> List[float]:
    """All pairwise Jaccard similarities."""
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            scores.append(jaccard(sets[i], sets[j]))
    return scores


def run_stability_selection(
    x: pd.DataFrame,
    y: pd.Series,
    config: Config,
    resume: bool = True,
) -> Tuple[List[str], Dict[str, float], pd.Series]:
    """
    Resampling-based stability-aware feature selection.

    Procedure
    ---------
    For each fold of repeated stratified K-fold:
      1. Train RF on training fold.
      2. Extract top-K features by importance.
      3. Accumulate selection counts.

    Stability metrics (reported as a *distribution*, not a single scalar):
      - Mean pairwise Jaccard ± std
      - Median pairwise Jaccard
      - Selection-frequency distribution (saved separately)

    Feature retention
    -----------------
    Keep features selected in ≥ freq_threshold fraction of folds.
    If too few survive, fall back to the top min_features by frequency.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.seed,
    )
    splits = list(cv.split(x, y))
    total_splits = len(splits)

    ckpt_path = config.checkpoint_dir / "stability.pkl"
    start_idx = 0
    selected_sets: List[set] = []
    counts = np.zeros(x.shape[1], dtype=int)

    if resume and ckpt_path.exists():
        with ckpt_path.open("rb") as f:
            state = pickle.load(f)
        start_idx = state["start_idx"]
        selected_sets = state["selected_sets"]
        counts = state["counts"]
        print(f"  Resuming stability selection from split {start_idx}/{total_splits}")

    for idx in range(start_idx, total_splits):
        train_idx, _ = splits[idx]
        model = _rf_model(config.seed + idx)      # vary seed per fold
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        importances = model.feature_importances_
        top_idx = set(np.argsort(importances)[::-1][: config.top_k].tolist())
        selected_sets.append(top_idx)
        for ti in top_idx:
            counts[ti] += 1

        if (idx + 1) % config.checkpoint_every == 0:
            _save_stability_checkpoint(ckpt_path, idx + 1, selected_sets, counts)
            print(f"  Checkpoint at split {idx + 1}/{total_splits}")

    _save_stability_checkpoint(ckpt_path, total_splits, selected_sets, counts)

    # Frequency series
    freqs = counts / max(1, total_splits)
    freq_series = pd.Series(freqs, index=x.columns).sort_values(ascending=False)

    # Stable feature set
    stable = freq_series[freq_series >= config.freq_threshold].index.tolist()
    if len(stable) < config.min_features:
        stable = freq_series.head(config.min_features).index.tolist()

    # Stability metrics — distribution, not a single scalar
    jscores = pairwise_jaccard_scores(selected_sets)
    stability_metrics = {
        "mean_jaccard":   float(np.mean(jscores)) if jscores else 1.0,
        "std_jaccard":    float(np.std(jscores))  if jscores else 0.0,
        "median_jaccard": float(np.median(jscores)) if jscores else 1.0,
        "total_splits":   float(total_splits),
        "n_stable_features": float(len(stable)),
        "freq_threshold_used": config.freq_threshold,
    }

    print(
        f"Stability: Jaccard = {stability_metrics['mean_jaccard']:.3f} "
        f"± {stability_metrics['std_jaccard']:.3f}  |  "
        f"stable features: {len(stable)} (threshold ≥ {config.freq_threshold})"
    )
    return stable, stability_metrics, freq_series


def _save_stability_checkpoint(
    path: Path, start_idx: int, selected_sets: List[set], counts: np.ndarray
) -> None:
    with path.open("wb") as f:
        pickle.dump(
            {"start_idx": start_idx, "selected_sets": selected_sets,
             "counts": counts}, f,
        )


# ──────────────────────────────────────────────────────────────────────
# STEP 7 — Checkpointing (parsed stage)
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint_parsed(
    config: Config,
    series_meta: Dict[str, str],
    meta_df: pd.DataFrame,
    x_df: pd.DataFrame,
    y: pd.Series,
    x_filtered: pd.DataFrame,
) -> None:
    ckpt = config.checkpoint_dir / "parsed.pkl"
    with ckpt.open("wb") as f:
        pickle.dump({
            "series_meta": series_meta,
            "meta_df": meta_df,
            "x_df": x_df,
            "y": y,
            "x_filtered": x_filtered,
        }, f)


def load_checkpoint_parsed(config: Config):
    ckpt = config.checkpoint_dir / "parsed.pkl"
    if not ckpt.exists():
        return None
    with ckpt.open("rb") as f:
        state = pickle.load(f)
    return (
        state["series_meta"],
        state["meta_df"],
        state["x_df"],
        state["y"],
        state["x_filtered"],
    )


# ──────────────────────────────────────────────────────────────────────
# STEP 8 — Visualisations
# ──────────────────────────────────────────────────────────────────────

def _configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_feature_stability_bar(
    freqs: pd.Series, output_dir: Path, top_n: int = 30
) -> Path:
    """Horizontal bar chart — top-N features by selection frequency."""
    plt = _configure_matplotlib()

    top = freqs.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(top.index.astype(str), top.values, color="#4C72B0")
    ax.set_xlabel("Selection frequency across resampling folds")
    ax.set_title(f"Top-{top_n} Features by Selection Frequency")
    ax.axvline(x=0.6, color="red", ls="--", lw=0.8, label="threshold = 0.6")
    ax.legend()
    fig.tight_layout()

    out = output_dir / "feature_stability_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_frequency_distribution(
    freqs: pd.Series, output_dir: Path
) -> Path:
    """Histogram of selection frequencies across all features."""
    plt = _configure_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(freqs.values, bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of Feature Selection Frequencies")
    ax.axvline(x=0.6, color="red", ls="--", lw=0.8, label="threshold = 0.6")
    ax.legend()
    fig.tight_layout()

    out = output_dir / "frequency_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_jaccard_distribution(
    selected_sets: List[set], output_dir: Path
) -> Path:
    """Histogram of pairwise Jaccard similarities."""
    plt = _configure_matplotlib()

    scores = pairwise_jaccard_scores(selected_sets)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, color="#DD8452", edgecolor="white")
    ax.set_xlabel("Pairwise Jaccard similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Pairwise Jaccard Similarities Across Folds")
    fig.tight_layout()

    out = output_dir / "jaccard_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_model_comparison(
    results: pd.DataFrame, output_dir: Path
) -> Path:
    """Grouped bar chart comparing 3 conditions on AUC, bal-acc, acc."""
    plt = _configure_matplotlib()

    metrics = ["roc_auc", "balanced_accuracy", "accuracy"]
    labels = ["ROC-AUC", "Balanced Accuracy", "Accuracy"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, row) in enumerate(results.iterrows()):
        means = [row[f"{m}_mean"] for m in metrics]
        stds  = [row[f"{m}_std"]  for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=row["model"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (mean ± std across evaluation seeds)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out = output_dir / "model_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────
# STEP 8b — Results table
# ──────────────────────────────────────────────────────────────────────

def build_results_table(
    baseline_full: Dict[str, float],
    baseline_singleshot: Dict[str, float],
    stability_aware: Dict[str, float],
    stability_metrics: Dict[str, float],
    n_features_full: int,
    n_features_singleshot: int,
    n_features_stable: int,
) -> pd.DataFrame:
    """Comparison table across the three experimental conditions."""
    rows = []
    for name, perf, n_feat in [
        ("baseline_full",     baseline_full,       n_features_full),
        ("single_shot_topK",  baseline_singleshot, n_features_singleshot),
        ("stability_aware",   stability_aware,     n_features_stable),
    ]:
        row = {"model": name, "n_features": n_feat}
        row.update(perf)
        rows.append(row)

    # Attach stability metrics only to the stability-aware row
    rows[2].update(stability_metrics)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSE39582 Feature-Selection Stability Pipeline"
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to GSE39582 series matrix file")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--freq-threshold", type=float, default=0.6)
    parser.add_argument("--min-features", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--variance-percentile", type=float, default=10.0)
    parser.add_argument("--n-eval-seeds", type=int, default=5)
    parser.add_argument(
        "--label-column",
        type=str,
        default="rfs.event",
        help="Metadata column to use as the binary label",
    )
    parser.add_argument(
        "--label-positive",
        type=str,
        default="1",
        help="Value in label column treated as positive (mapped to 1)",
    )
    parser.add_argument(
        "--label-negative",
        type=str,
        default="0",
        help="Value in label column treated as negative (mapped to 0)",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints if available")
    args = parser.parse_args()

    run_output_dir, run_checkpoint_dir = _derive_run_dirs(
        args.data, args.output, args.checkpoint
    )

    config = Config(
        data_path=args.data,
        output_dir=run_output_dir,
        checkpoint_dir=run_checkpoint_dir,
        seed=args.seed,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        top_k=args.top_k,
        freq_threshold=args.freq_threshold,
        min_features=args.min_features,
        checkpoint_every=args.checkpoint_every,
        variance_percentile=args.variance_percentile,
        n_eval_seeds=args.n_eval_seeds,
        label_column=args.label_column,
        label_positive=args.label_positive,
        label_negative=args.label_negative,
    )

    _ensure_dir(config.output_dir)
    _ensure_dir(config.checkpoint_dir)

    eval_seeds = [config.seed + i for i in range(config.n_eval_seeds)]

    # ── Parse / label / variance-filter (or resume) ──────────────────
    parsed = load_checkpoint_parsed(config) if args.resume else None
    if parsed is not None:
        series_meta, meta_df, x_df, y, x_filtered = parsed
        print(f"Loaded checkpoint: {x_filtered.shape[0]} samples × "
              f"{x_filtered.shape[1]} features")
    else:
        series_meta, meta_df, x_df = parse_series_matrix(
            config.data_path, config.checkpoint_dir
        )
        y = define_labels(meta_df, config)
        x_df, meta_df, y = filter_labeled(x_df, meta_df, y)
        x_filtered, _ = variance_filter(x_df, config.variance_percentile)
        save_checkpoint_parsed(config, series_meta, meta_df, x_df, y, x_filtered)

    # ── Condition 1: Baseline — full features after variance filter ──
    print("\n=== Condition 1: Baseline (full features) ===")
    perf_full = evaluate_cv_with_uncertainty(
        x_filtered, y, config.n_splits, eval_seeds
    )
    print(f"  AUC = {perf_full['roc_auc_mean']:.3f} "
          f"± {perf_full['roc_auc_std']:.3f}")

    # ── Condition 2: Single-shot top-K ───────────────────────────────
    print("\n=== Condition 2: Single-shot top-K ===")
    ss_features = single_shot_top_k(
        x_filtered, y, config.top_k, config.seed
    )
    x_ss = x_filtered[ss_features]
    perf_ss = evaluate_cv_with_uncertainty(
        x_ss, y, config.n_splits, eval_seeds
    )
    print(f"  AUC = {perf_ss['roc_auc_mean']:.3f} "
          f"± {perf_ss['roc_auc_std']:.3f}")

    # ── Condition 3: Stability-aware selection ───────────────────────
    print("\n=== Condition 3: Stability-aware selection ===")
    stable_features, stability_metrics, freqs = run_stability_selection(
        x_filtered, y, config, resume=args.resume
    )
    x_stable = x_filtered[stable_features]
    perf_stable = evaluate_cv_with_uncertainty(
        x_stable, y, config.n_splits, eval_seeds
    )
    print(f"  AUC = {perf_stable['roc_auc_mean']:.3f} "
          f"± {perf_stable['roc_auc_std']:.3f}")

    # ── Results table ────────────────────────────────────────────────
    results = build_results_table(
        baseline_full=perf_full,
        baseline_singleshot=perf_ss,
        stability_aware=perf_stable,
        stability_metrics=stability_metrics,
        n_features_full=x_filtered.shape[1],
        n_features_singleshot=len(ss_features),
        n_features_stable=len(stable_features),
    )
    results_path = config.output_dir / "results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved → {results_path}")
    print(results.to_string(index=False))

    # ── Visualisations ───────────────────────────────────────────────
    plot_feature_stability_bar(freqs, config.output_dir)
    plot_frequency_distribution(freqs, config.output_dir)
    plot_model_comparison(results, config.output_dir)

    ckpt_stab = config.checkpoint_dir / "stability.pkl"
    if ckpt_stab.exists():
        with ckpt_stab.open("rb") as f:
            stab_state = pickle.load(f)
        plot_jaccard_distribution(stab_state["selected_sets"], config.output_dir)

    # ── Artefacts ────────────────────────────────────────────────────
    summary = {
        "series_title": series_meta.get("Series_title", ""),
        "label_column": config.label_column,
        "label_positive": config.label_positive,
        "label_negative": config.label_negative,
        "n_samples": int(x_filtered.shape[0]),
        "n_features_original": int(x_df.shape[1]),
        "n_features_after_variance_filter": int(x_filtered.shape[1]),
        "variance_percentile": config.variance_percentile,
        "top_k": config.top_k,
        "freq_threshold": config.freq_threshold,
        "n_eval_seeds": config.n_eval_seeds,
        "class_balance": y.value_counts().to_dict(),
    }
    with (config.output_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (config.output_dir / "stable_features.txt").open("w") as f:
        for feat in stable_features:
            f.write(feat + "\n")

    freqs.to_csv(config.output_dir / "feature_frequencies.csv",
                 header=["frequency"])

    print(f"\nDone. All outputs in {config.output_dir}/")


if __name__ == "__main__":
    main()
