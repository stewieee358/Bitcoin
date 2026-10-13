"""Small exportable tables and figures, with explicit exploratory interpretation."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_tree_reports(out, config, pred, metrics, selection, permutation, mdi):
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    feature_order = config['feature_columns']
    baseline_names = list(config['models'])
    primary = metrics.loc[metrics.model.isin(baseline_names + ['majority'])].copy()
    primary.to_csv(out/'all_feature_metrics.csv', index=False)
    lookup = metrics.set_index('model')
    differences = []
    for name in ('dt', 'ada', 'gbm'):
        row = dict(model=name)
        for metric in ('accuracy', 'balanced_accuracy', 'roc_auc', 'f1', 'log_loss', 'brier_score'):
            row[metric+'_all'] = lookup.loc[name, metric]
            row[metric+'_selected'] = lookup.loc[name+'_selected', metric]
            row[metric+'_change'] = row[metric+'_selected'] - row[metric+'_all']
        differences.append(row)
    effect = pd.DataFrame(differences)
    effect.to_csv(out/'feature_selection_effect.csv', index=False)

    valid = permutation.dropna(subset=['permutation_mean']).copy()
    importance = valid.groupby('feature').agg(mean_decrease=('permutation_mean', 'mean'),
                std_across_days=('permutation_mean', 'std'), measured_days=('permutation_mean', 'count'),
                mean_shuffle_std=('permutation_std', 'mean'),
                positive_fraction=('permutation_mean', lambda values: (values > 0).mean()))
    importance = importance.reindex(feature_order).reset_index()
    top_days = selection.loc[selection.selection_reason == 'top_k']
    membership = {name: 0 for name in feature_order}
    for encoded in top_days.selected_features:
        for name in json.loads(encoded):
            membership[name] += 1
    importance['top_k_days'] = importance.feature.map(membership)
    importance['top_k_frequency'] = (importance.top_k_days/len(top_days) if len(top_days) else np.nan)
    importance['eligible_top_k_days'] = len(top_days)
    importance = importance.sort_values('mean_decrease', ascending=False, na_position='last')
    importance.to_csv(out/'feature_importance_summary.csv', index=False)

    mdi_summary = mdi.groupby(['model', 'feature']).agg(mean_mdi=('mdi', 'mean'),
                std_mdi=('mdi', 'std'), active_fraction=('active', 'mean')).reset_index()
    mdi_summary.to_csv(out/'tree_importance_summary.csv', index=False)
    monthly = valid.assign(month=pd.to_datetime(valid.train_cutoff, utc=True).dt.strftime('%Y-%m'))
    monthly = monthly.pivot_table(index='feature', columns='month', values='permutation_mean', aggfunc='mean')
    monthly = monthly.reindex(feature_order)
    monthly.to_csv(out/'feature_importance_by_month.csv')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ('balanced_accuracy', 'roc_auc'), ('Balanced accuracy', 'ROC AUC')):
        data = primary.sort_values(metric)
        ax.barh(data.model, data[metric], color='#356c97')
        ax.axvline(.5, color='#bc5c45', linestyle='--', linewidth=1)
        ax.set(xlim=(0, 1), title=title, xlabel='Out-of-sample score')
        for index, value in enumerate(data[metric]):
            if np.isfinite(value):
                ax.text(value+.012, index, f'{value:.4f}', va='center', fontsize=8)
    fig.savefig(figures/'model_comparison.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    positions = np.arange(3)
    for ax, metric, title in zip(axes, ('balanced_accuracy', 'roc_auc'), ('Balanced accuracy', 'ROC AUC')):
        ax.bar(positions-.18, effect[metric+'_all'], .36, label='All 21 features', color='#356c97')
        ax.bar(positions+.18, effect[metric+'_selected'], .36, label='Historical selection + fallback', color='#d09354')
        ax.axhline(.5, color='grey', linestyle='--', linewidth=1)
        ax.set(xticks=positions, xticklabels=effect.model, ylim=(0, 1), title=title)
        ax.legend(fontsize=8)
    fig.savefig(figures/'selection_comparison.png', dpi=160); plt.close(fig)

    plotted = importance.dropna(subset=['mean_decrease']).sort_values('mean_decrease')
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(plotted):
        ax.barh(plotted.feature, plotted.mean_decrease, xerr=plotted.std_across_days.fillna(0),
                color=np.where(plotted.mean_decrease > 0, '#356c97', '#bd7569'), capsize=2)
    else:
        ax.text(.5, .5, 'No measurable historical validation importance', ha='center', transform=ax.transAxes)
    ax.axvline(0, color='black', linewidth=.8)
    ax.set(title='Historical RF permutation importance',
           xlabel='Mean decrease in balanced accuracy (error bars: SD across days)')
    fig.savefig(figures/'permutation_importance.png', dpi=160); plt.close(fig)

    frequency = importance.sort_values('top_k_frequency', na_position='first')
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(top_days):
        ax.barh(frequency.feature, frequency.top_k_frequency, color='#356c97')
    else:
        ax.text(.5, .5, 'All days used the all-feature fallback', ha='center', transform=ax.transAxes)
    ax.set(xlim=(0, 1), title=f'Top-{config["tree_study"]["top_k"]} selection frequency',
           xlabel=f'Fraction of {len(top_days)} actual top-k days (fallback days excluded)')
    fig.savefig(figures/'selection_frequency.png', dpi=160); plt.close(fig)

    heatmap = mdi_summary.loc[mdi_summary.model.isin(['rf', 'dt', 'ada', 'gbm'])].pivot(
                              index='feature', columns='model', values='mean_mdi')
    heatmap = heatmap.reindex(feature_order)[['dt', 'rf', 'ada', 'gbm']]
    _heatmap(heatmap, figures/'tree_importance_heatmap.png', 'Training MDI: descriptive, not causal', 'Mean normalized MDI', 'viridis')
    if len(monthly.columns):
        _heatmap(monthly, figures/'importance_over_time.png', 'Historical validation importance by month', 'Mean decrease in balanced accuracy', 'RdBu_r')
    correlation_path = out/'initial_window_correlation.csv'
    if correlation_path.exists():
        correlation = pd.read_csv(correlation_path, index_col=0)
        _heatmap(correlation, figures/'feature_correlation.png', 'Feature correlation in the initial training window', 'Spearman correlation', 'RdBu_r', (-1, 1))

    fallback_days = len(selection)-len(top_days)
    lines = ['# BTCUSDT 2h tree-model and feature-selection study', '',
             f'UTC evaluation: [{config["evaluation_start"]}, {config["evaluation_end"]}).',
             f'{len(pred):,} identical prediction rows; {len(selection)} daily refits.',
             'Target: next completed 2h close-to-close return > 0; probability threshold strictly > 0.5.',
             'Final prediction models use the previous 90 calendar days, refitting every 24 hours.',
             'All-feature comparison: all models use the same 21 inputs, valid rows and time splits.', '',
             '## Feature selection', '',
             f'At each daily cutoff, fit the screening RF on the first {90-config["tree_study"]["validation_days"]} days '
             f'of the historical window; inspect it on the last {config["tree_study"]["validation_days"]} days.',
             'Labels used by the screening fit must be known at the inner cutoff. Validation labels must be known at the daily cutoff.',
             f'Rank repeated permutation importance using balanced accuracy; fixed top_k={config["tree_study"]["top_k"]}.',
             'The same daily subset is shared by DT, AdaBoost and GBM. Refit each on the entire available 90-day window.',
             f'Top-k days: {len(top_days)}; all-feature fallback days: {fallback_days}.',
             'Fallback applies to insufficient history, single-class inner partitions, or no positive permutation importance.',
             'Frequency charts exclude fallback days: fallback does not count as evidence that all features are important.',
             'A positive rank alone is not proof of predictive signal. Check validation and final baseline performance.', '',
             '## Classification results', '',
             '| Model | Accuracy | Balanced accuracy | F1 | ROC AUC | Log loss | Brier |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics.itertuples():
        lines.append(f'| {row.model} | {row.accuracy:.2%} | {row.balanced_accuracy:.2%} | {row.f1:.2%} | '
                     f'{row.roc_auc:.4f} | {row.log_loss:.4f} | {row.brier_score:.4f} |')
    lines += ['', '## Paired feature-selection effects', '',
              'Changes below are selected minus all-feature. Positive balanced accuracy/AUC is better; negative log loss is better.', '',
              '| Model | Balanced accuracy change | AUC change | Log loss change |',
              '| --- | ---: | ---: | ---: |']
    for row in effect.itertuples():
        lines.append(f'| {row.model} | {row.balanced_accuracy_change:+.4f} | {row.roc_auc_change:+.4f} | {row.log_loss_change:+.4f} |')
    lines += ['', '## Interpretation and limitations', '',
              'This is an exploratory comparison on a historical period already inspected by the group, not a new independent holdout.',
              'Feature importance is model- and period-dependent. It is not a fraction of BTC price variance explained, nor causal attribution.',
              'Correlated OHLC and volume features can share or mask individual importance. Correlation is shown from the initial historical window only.',
              'The default permutation uses individual-row shuffling; temporal dependence and correlated features limit interpretation.',
              'Overlapping daily training windows make daily importances dependent. Across-day SD and selection frequency are descriptive, not confidence intervals.',
              'A small score difference is not evidence of statistically significant improvement. No significance or profitability claim is made.',
              'The zero-probability baseline has extreme log loss; compare with the training-majority probability baseline too.',
              'No trading backtest is run by this extension; classification quality does not establish trading profit.', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def _heatmap(data, path, title, label, cmap, limits=None):
    fig, ax = plt.subplots(figsize=(max(7, len(data.columns)*.6), 9), constrained_layout=True)
    values = data.to_numpy(dtype=float)
    kwargs = dict(cmap=cmap, aspect='auto')
    if limits:
        kwargs.update(vmin=limits[0], vmax=limits[1])
    elif cmap == 'RdBu_r':
        finite = np.abs(values[np.isfinite(values)])
        bound = max(float(finite.max()) if len(finite) else 0., .001)
        kwargs.update(vmin=-bound, vmax=bound)
    visual = ax.imshow(values, **kwargs)
    ax.set(xticks=np.arange(len(data.columns)), xticklabels=list(data.columns),
           yticks=np.arange(len(data)), yticklabels=list(data.index), title=title)
    ax.tick_params(axis='both', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=60 if len(data.columns)>6 else 0, ha='right' if len(data.columns)>6 else 'center')
    fig.colorbar(visual, ax=ax, shrink=.8, label=label)
    fig.savefig(path, dpi=160)
    plt.close(fig)
