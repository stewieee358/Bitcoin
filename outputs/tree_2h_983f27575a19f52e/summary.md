# BTCUSDT 2h tree-model and feature-selection study

UTC evaluation: [2025-01-01T00:00:00+00:00, 2026-01-01T00:00:00+00:00).
4,379 identical prediction rows; 365 daily refits.
Target: next completed 2h close-to-close return > 0; probability threshold strictly > 0.5.
Final prediction models use the previous 90 calendar days, refitting every 24 hours.
All-feature comparison: all models use the same 21 inputs, valid rows and time splits.

## Feature selection

At each daily cutoff, fit the screening RF on the first 60 days of the historical window; inspect it on the last 30 days.
Labels used by the screening fit must be known at the inner cutoff. Validation labels must be known at the daily cutoff.
Rank repeated permutation importance using balanced accuracy; fixed top_k=5.
The same daily subset is shared by DT, AdaBoost and GBM. Refit each on the entire available 90-day window.
Top-k days: 356; all-feature fallback days: 9.
Fallback applies to insufficient history, single-class inner partitions, or no positive permutation importance.
Frequency charts exclude fallback days: fallback does not count as evidence that all features are important.
A positive rank alone is not proof of predictive signal. Check validation and final baseline performance.

## Classification results

| Model | Accuracy | Balanced accuracy | F1 | ROC AUC | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 52.52% | 52.12% | 59.23% | 0.5264 | 0.7001 | 0.2530 |
| knn | 49.21% | 49.15% | 50.99% | 0.4946 | 1.9084 | 0.3013 |
| rf | 50.92% | 50.89% | 52.26% | 0.5094 | 0.7200 | 0.2621 |
| logistic_pca | 53.23% | 52.81% | 60.05% | 0.5318 | 0.6961 | 0.2512 |
| knn_pca | 49.39% | 49.34% | 51.12% | 0.4947 | 1.9890 | 0.3031 |
| dt | 50.17% | 49.89% | 55.36% | 0.5038 | 0.7676 | 0.2701 |
| ada | 51.15% | 50.42% | 61.97% | 0.5135 | 0.7203 | 0.2620 |
| gbm | 51.66% | 51.41% | 56.29% | 0.5201 | 0.7041 | 0.2550 |
| dt_selected | 50.97% | 50.62% | 57.12% | 0.5011 | 0.7368 | 0.2675 |
| ada_selected | 52.34% | 51.67% | 62.19% | 0.5115 | 0.7307 | 0.2664 |
| gbm_selected | 50.38% | 50.10% | 55.55% | 0.5127 | 0.7030 | 0.2546 |
| zero | 48.64% | 50.00% | 0.00% | 0.5000 | 18.5116 | 0.5136 |
| majority | 51.15% | 50.28% | 63.43% | 0.4965 | 0.6930 | 0.2499 |

## Paired feature-selection effects

Changes below are selected minus all-feature. Positive balanced accuracy/AUC is better; negative log loss is better.

| Model | Balanced accuracy change | AUC change | Log loss change |
| --- | ---: | ---: | ---: |
| dt | +0.0073 | -0.0027 | -0.0308 |
| ada | +0.0125 | -0.0020 | +0.0104 |
| gbm | -0.0131 | -0.0074 | -0.0010 |

## Interpretation and limitations

This is an exploratory comparison on a historical period already inspected by the group, not a new independent holdout.
Feature importance is model- and period-dependent. It is not a fraction of BTC price variance explained, nor causal attribution.
Correlated OHLC and volume features can share or mask individual importance. Correlation is shown from the initial historical window only.
The default permutation uses individual-row shuffling; temporal dependence and correlated features limit interpretation.
Overlapping daily training windows make daily importances dependent. Across-day SD and selection frequency are descriptive, not confidence intervals.
A small score difference is not evidence of statistically significant improvement. No significance or profitability claim is made.
The zero-probability baseline has extreme log loss; compare with the training-majority probability baseline too.
No trading backtest is run by this extension; classification quality does not establish trading profit.
