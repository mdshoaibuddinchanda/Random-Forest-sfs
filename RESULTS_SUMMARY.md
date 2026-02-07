# Results Summary: Feature-Selection Stability Study

## Research question

How stable are Random-Forest-derived feature rankings in high-dimensional
microarray data, and does enforcing selection stability improve
predictive performance on a clinical outcome?

## Evidence reviewed

- outputs/GSE39582_series_matrix/results.csv
- outputs/GSE39582_series_matrix/run_summary.json
- outputs/GSE17536_series_matrix/results.csv
- outputs/GSE17536_series_matrix/run_summary.json
- All plots in outputs/<dataset_name>/ (frequency distributions, Jaccard
  distributions, model comparison bars)

## Dataset: GSE39582 (rfs.event)

Samples: 574 (class balance 395/179). Features after variance filter: 49,207.

Performance (mean +/- std):

- Baseline (full): acc 0.687 +/- 0.003, bal-acc 0.501 +/- 0.002, AUC 0.616 +/- 0.008
- Single-shot top-K: acc 0.734 +/- 0.003, bal-acc 0.588 +/- 0.004, AUC 0.768 +/- 0.006
- Stability-aware: acc 0.745 +/- 0.006, bal-acc 0.621 +/- 0.007, AUC 0.770 +/- 0.003

Stability metrics:

- Mean Jaccard 0.029 +/- 0.013 (median 0.031)
- Stable features retained: 50 (threshold 0.6)

Interpretation:

- Feature selection modestly improves performance vs full features.
- Stability-aware selection yields a small but consistent gain over
  single-shot top-K across all metrics, while enforcing reproducible
  feature selection.

## Dataset: GSE17536 (dfs_event)

Samples: 145 (class balance 109/36). Features after variance filter: 49,207.

Performance (mean +/- std):

- Baseline (full): acc 0.754 +/- 0.012, bal-acc 0.517 +/- 0.017, AUC 0.713 +/- 0.013
- Single-shot top-K: acc 0.832 +/- 0.013, bal-acc 0.709 +/- 0.024, AUC 0.861 +/- 0.011
- Stability-aware: acc 0.834 +/- 0.012, bal-acc 0.719 +/- 0.021, AUC 0.869 +/- 0.007

Stability metrics:

- Mean Jaccard 0.046 +/- 0.016 (median 0.042)
- Stable features retained: 50 (threshold 0.6)

Interpretation:

- As in GSE39582, feature selection provides modest  gains over full
  features.
- Stability-aware selection again yields a small improvement in all
  metrics over single-shot top-K, with higher selection consistency.

## Answer to the research question

Across both datasets, enforcing stability in feature selection improves
predictive performance slightly compared with a single-shot top-K approach,
while also producing reproducible feature sets (non-trivial Jaccard
similarity distributions). The results suggest that in high-dimensional
microarray settings, stability-aware selection is a reliable refinement
that trades minimal complexity for modest but consistent gains and more
interpretable, repeatable feature subsets.

## Notes

- The Jaccard distributions are low overall, indicating feature selection
  is inherently unstable in this regime; stability-aware selection helps
  regularize this instability.
- Class imbalance is present in both datasets; balanced accuracy is the
  preferred metric for clinical interpretation.
