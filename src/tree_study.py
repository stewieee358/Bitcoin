"""Causal daily selection and six tree experiments on the upstream shared rows.

No upstream module is changed. A separate study directory owns its single
run.json, verified model caches, and additional feature-inspection tables.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

from src.data import load_data
from src.evaluation import write_reports
from src.experiment import (normalize_config, experiment_identity, digest, file_digest,
                            load_cached, save_cached, check_predictions)
from src.features import build_features
from src.metadata import read_run, update_run
from src.models import build_model, model_signature
from src.prepare_data import INTERVALS
from src.training import prediction_base, train_model
from src.tree_models import register_tree_models
from src.validation import make_splits

TREE_NAMES = ('dt', 'ada', 'gbm')
INSPECT_NAMES = ('rf',) + TREE_NAMES


def validate_study(config):
    register_tree_models()
    config = normalize_config(copy.deepcopy(config))
    if config['interval'] != '2h' or config['source_interval'] != '2h':
        raise ValueError('This extension requires native 2h archives')
    if config['window_days'] != 90 or config['update_hours'] != 24:
        raise ValueError('Keep the group protocol: 90 calendar days, 24-hour refits')
    if len(config['feature_columns']) != 21:
        raise ValueError('Use the group baseline of 21 shared features')
    if set(('rf',) + TREE_NAMES) - config['models'].keys():
        raise ValueError('Configure rf, dt, ada, and gbm')
    settings = config.setdefault('tree_study', {})
    for key, value in dict(top_k=5, validation_days=30, permutation_repeats=10,
                           scoring='balanced_accuracy', random_state=42,
                           min_fit_samples=100, min_validation_samples=100,
                           nonpositive_policy='all_features').items():
        settings.setdefault(key, value)
    if not 1 <= settings['top_k'] < len(config['feature_columns']):
        raise ValueError('top_k must be a positive integer smaller than 21')
    if not isinstance(settings['top_k'], int):
        raise ValueError('top_k must be an integer')
    if not 1 <= settings['validation_days'] < 90:
        raise ValueError('validation_days must be between 1 and 89')
    for key in ('permutation_repeats', 'min_fit_samples', 'min_validation_samples', 'validation_days'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings['scoring'] != 'balanced_accuracy' or settings['nonpositive_policy'] != 'all_features':
        raise ValueError('This study fixes balanced_accuracy and an all-feature fallback')
    return config


def internal_selection_rows(frame, split, validation_days):
    """Calendar boundaries plus explicit label availability at the inner cutoff."""
    history = frame.loc[split.train]
    boundary = split.cutoff - pd.Timedelta(days=validation_days)
    fit = history.loc[(history.feature_time < boundary) & (history.target_time <= boundary)]
    validation = history.loc[(history.feature_time >= boundary) &
                             (history.feature_time < split.cutoff) & (history.target_time <= split.cutoff)]
    if len(fit):
        assert (fit.target_time <= boundary).all()
    if len(validation):
        assert (validation.feature_time >= boundary).all()
        assert (validation.target_time <= split.cutoff).all()
    assert fit.index.intersection(validation.index).empty
    return fit, validation, boundary


def select_for_day(frame, columns, split, settings, rf_params):
    fit, validation, boundary = internal_selection_rows(frame, split, settings['validation_days'])
    audit = dict(train_cutoff=split.cutoff, window_start=split.metadata['window_start'],
                 inner_cutoff=boundary, n_inner_fit=len(fit), n_inner_validation=len(validation),
                 inner_fit_target_end=fit.target_time.max() if len(fit) else pd.NaT,
                 inner_validation_target_end=validation.target_time.max() if len(validation) else pd.NaT,
                 validation_balanced_accuracy=np.nan, validation_majority_balanced_accuracy=np.nan,
                 validation_edge_over_majority=False, positive_importance_count=0)
    importance = np.full(len(columns), np.nan)
    spread = importance.copy()
    reason = 'top_k'
    if len(fit) < settings['min_fit_samples'] or len(validation) < settings['min_validation_samples']:
        chosen, reason = list(columns), 'fallback_insufficient_history'
    elif (fit.target > 0).nunique() != 2 or (validation.target > 0).nunique() != 2:
        chosen, reason = list(columns), 'fallback_single_class'
    else:
        model = build_model('rf', rf_params).fit(fit[columns], (fit.target > 0).astype(int))
        y_validation = (validation.target > 0).astype(int)
        # Explicit >0.5 is the same direction rule as the upstream experiments.
        class_one = list(model.classes_).index(1)
        predicted = model.predict_proba(validation[columns])[:, class_one] > .5
        baseline = balanced_accuracy_score(y_validation, predicted)
        majority = int((fit.target > 0).mean() > .5)
        audit['validation_balanced_accuracy'] = float(baseline)
        audit['validation_majority_balanced_accuracy'] = float(
            balanced_accuracy_score(y_validation, np.full(len(validation), majority)))
        audit['validation_edge_over_majority'] = bool(
            baseline > audit['validation_majority_balanced_accuracy'])
        # Use the project's strict threshold also inside the permutation scorer.
        def scorer(estimator, X, y):
            probability = estimator.predict_proba(X)[:, list(estimator.classes_).index(1)]
            return balanced_accuracy_score(y, probability > .5)
        result = permutation_importance(model, validation[columns], y_validation, scoring=scorer,
                        n_repeats=settings['permutation_repeats'], random_state=settings['random_state'], n_jobs=1)
        importance, spread = result.importances_mean, result.importances_std
        audit['positive_importance_count'] = int((importance > 0).sum())
        if not np.any(importance > 0):
            chosen, reason = list(columns), 'fallback_no_positive_importance'
        else:
            # Stable original-order tie break; retain original feature order in X.
            ranked = np.argsort(-importance, kind='stable')[:settings['top_k']]
            membership = {columns[index] for index in ranked}
            chosen = [column for column in columns if column in membership]
    audit.update(selected_features=json.dumps(chosen), selected_count=len(chosen), selection_reason=reason)
    detail = pd.DataFrame(dict(train_cutoff=split.cutoff, feature=list(columns),
                              permutation_mean=importance, permutation_std=spread,
                              selected=[column in chosen for column in columns]))
    return audit, detail


def calculate_selections(frame, columns, splits, config):
    daily, details = [], []
    # Permutation inspection has much Python overhead. Separate processes avoid
    # GIL contention; the entry point's __main__ guard also supports Windows.
    with parallel_config(backend='loky', inner_max_num_threads=1):
        jobs = Parallel(n_jobs=config['parallel_days'], return_as='generator')(
                   delayed(select_for_day)(frame, columns, split, config['tree_study'],
                                           config['models']['rf']) for split in splits)
        for index, (audit, detail) in enumerate(jobs):
            daily.append(audit)
            details.append(detail)
            if index % 30 == 0:
                print(f'Selection: {index+1}/{len(splits)} days', flush=True)
    return pd.DataFrame(daily), pd.concat(details, ignore_index=True)


def get_importances(model):
    estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
    return np.asarray(estimator.feature_importances_, dtype=float)


def fit_tree_day(frame, columns, split, name, params, selected):
    train, test = frame.loc[split.train], frame.loc[split.test]
    active = selected if selected is not None else columns
    y = (train.target > 0).astype(int)
    if y.nunique() == 1:
        probability = np.full(len(test), float(y.iloc[0]))
        mdi = np.zeros(len(active))
    else:
        model = build_model(name, params).fit(train[active], y)
        probability = (model.predict_proba(test[active])[:, list(model.classes_).index(1)]
                       if len(test) else np.empty(0))
        mdi = get_importances(model)
    values = dict(zip(active, mdi))
    importance = pd.DataFrame(dict(train_cutoff=split.cutoff, feature=columns,
                         mdi=[values.get(column, 0.) for column in columns],
                         active=[column in active for column in columns], n_features=len(active)))
    return probability, importance


def train_tree(frame, columns, splits, base, name, params, selections, workers):
    probabilities, details = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = pool.map(lambda pair: fit_tree_day(frame, columns, pair[1], name, params,
                   selections[pair[0]] if selections is not None else None), enumerate(splits))
        for index, (probability, detail) in enumerate(jobs):
            probabilities.append(probability)
            details.append(detail)
            if index % 30 == 0:
                print(f'{name}{"_selected" if selections is not None else ""}: '
                      f'{index+1}/{len(splits)} updates', flush=True)
    pred = base[['feature_time', 'target_time', 'train_cutoff', 'actual_return', 'actual_class']].copy()
    pred['probability'] = np.concatenate(probabilities)
    pred['predicted_class'] = (pred.probability > .5).astype(int)
    return pred, pd.concat(details, ignore_index=True)


def read_selection_cache(out, identity, splits, columns):
    try:
        section = read_run(out)['selection_cache']
        if section['identity'] != identity or section['status'] != 'success':
            return None
        for name in ('selection_daily.csv', 'permutation_importance_daily.csv'):
            if file_digest(out/name) != section['sha256'][name]:
                return None
        daily = pd.read_csv(out/'selection_daily.csv')
        detail = pd.read_csv(out/'permutation_importance_daily.csv')
        times = pd.to_datetime(daily.train_cutoff, utc=True)
        if list(times) != [split.cutoff for split in splits] or len(detail) != len(splits)*len(columns):
            return None
        for encoded in daily.selected_features:
            selected = json.loads(encoded)
            if not selected or len(selected) != len(set(selected)) or set(selected) - set(columns):
                return None
        return daily, detail
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_mdi_cache(out, folder, identity, base, label, splits, columns):
    pred = load_cached(folder, identity, base)
    if pred is None:
        return None
    try:
        cache = read_run(out)['model_cache'][folder.name]
        path = out/'importance'/f'{label}.csv'
        if file_digest(path) != cache['importance_sha256']:
            return None
        detail = pd.read_csv(path)
        if len(detail) != len(splits)*len(columns):
            return None
        expected_times = [split.cutoff for split in splits]
        if list(pd.to_datetime(detail.train_cutoff, utc=True).drop_duplicates()) != expected_times:
            return None
        if detail.groupby('train_cutoff', sort=False).feature.apply(list).tolist() != [columns]*len(splits):
            return None
        return pred, detail
    except (OSError, ValueError, KeyError):
        return None


def run_tree_study(config, root, output_dir=None, force=False):
    from src.tree_reports import write_tree_reports
    root = Path(root)
    config = validate_study(config)
    columns = config['feature_columns']
    versions = dict(python=platform.python_version(), **{package: importlib.metadata.version(package)
             for package in ('pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib',
                             'scipy', 'joblib', 'threadpoolctl')})
    # Upstream cache identity uses its original dependency list. Our additional
    # dependencies and source are independently included in the study identity.
    core_versions = {key: versions[key] for key in
                    ('python', 'pandas', 'numpy', 'scikit-learn', 'requests', 'matplotlib')}
    frame, audit, manifest = load_data(config, root/'data/raw')
    frame = build_features(frame, columns, INTERVALS[config['interval']])
    audit['valid_samples'] = int(frame.valid_sample.sum())
    splits = make_splits(frame, config)
    base = prediction_base(frame, splits)
    core_id = experiment_identity(frame, config, manifest, core_versions)
    module_hash = file_digest(Path(__file__))
    tree_hash = file_digest(Path(__file__).with_name('tree_models.py'))
    # Models have independent cache identities. Changing selection settings
    # invalidates selected variants, but not any all-feature model identity.
    selection_id = digest(dict(core=core_id, settings=config['tree_study'],
                              rf=model_signature('rf', config['models']['rf']),
                              implementation=module_hash, dependencies=versions))
    study_id = digest(dict(core=core_id, selection=selection_id,
                          implementation=module_hash, tree_models=tree_hash, dependencies=versions))
    out = Path(output_dir) if output_dir else root/'outputs'/f'tree_2h_{study_id[:16]}'
    out = out.resolve()
    if (out/'run.json').exists() and read_run(out)['config'].get('study_id') != study_id:
        raise ValueError('Output directory belongs to another study; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    (out/'importance').mkdir(exist_ok=True)
    # Inspection only: the first 90-day window precedes the evaluation period.
    frame.loc[splits[0].train, columns].corr(method='spearman').to_csv(out/'initial_window_correlation.csv')
    update_run(out, config={**config, 'experiment_id': core_id, 'study_id': study_id,
                           'dependency_versions': versions},
               download_manifest=manifest, data_audit=audit,
               status=dict(status='running', stage='selection', n=len(base)))
    print(f'Tree study: {len(base)} shared predictions, {len(splits)} daily updates; {out}', flush=True)
    try:
        cached = None if force else read_selection_cache(out, selection_id, splits, columns)
        selection_action = 'reused' if cached is not None else 'trained'
        daily, permutation = cached if cached is not None else calculate_selections(frame, columns, splits, config)
        if cached is None:
            daily.to_csv(out/'selection_daily.csv', index=False)
            permutation.to_csv(out/'permutation_importance_daily.csv', index=False)
            update_run(out, selection_cache=dict(status='success', identity=selection_id,
                       sha256={name: file_digest(out/name) for name in
                               ('selection_daily.csv', 'permutation_importance_daily.csv')}))
        selected_columns = [json.loads(value) for value in daily.selected_features]
        result = base.copy()
        importance_frames, actions, cache_ids = [], {}, {}
        variants = [(name, name, False) for name in config['models']]
        variants += [(name+'_selected', name, True) for name in TREE_NAMES]
        for label, name, selected in variants:
            is_tree = name in INSPECT_NAMES
            identity = digest(dict(experiment=core_id, model=model_signature(name, config['models'][name]),
                          tree_trainer=module_hash if is_tree else None,
                          selection=selection_id if selected else None, dependencies=versions))
            folder = out/'models'/f'{label}_{identity[:16]}'
            pred, detail = None, None
            action = 'trained'
            if not force:
                if is_tree:
                    cached = read_mdi_cache(out, folder, identity, base, label, splits, columns)
                    if cached is not None:
                        pred, detail = cached
                else:
                    pred = load_cached(folder, identity, base)
                if pred is not None:
                    action = 'reused'
            # Valid upstream baseline predictions are reused only when all of
            # its identity/dependency and shared-row checks pass.
            if pred is None and not is_tree and not force:
                upstream_identity = digest(dict(experiment=core_id, model=model_signature(name, config['models'][name])))
                upstream_out = root/'outputs'/f'2h_{core_id[:16]}'
                candidate = upstream_out/'models'/f'{name}_{upstream_identity[:16]}'
                pred = load_cached(candidate, upstream_identity, base)
                if pred is not None:
                    action = 'reused_upstream'
            if pred is None:
                if is_tree:
                    pred, detail = train_tree(frame, columns, splits, base, name, config['models'][name],
                              selected_columns if selected else None, config['parallel_days'])
                else:
                    pred = train_model(frame, columns, splits, base, name,
                                       config['models'][name], config['parallel_days'])
            check_predictions(pred, base)
            if action != 'reused':
                save_cached(folder, identity, pred, action)
                if is_tree:
                    detail.to_csv(out/'importance'/f'{label}.csv', index=False)
                    document = read_run(out)
                    cache = document['model_cache']
                    cache[folder.name]['importance_sha256'] = file_digest(out/'importance'/f'{label}.csv')
                    update_run(out, model_cache=cache)
            result[label+'_probability'] = pred.probability
            result[label+'_class'] = pred.predicted_class
            if is_tree:
                importance_frames.append(detail.assign(model=label))
            actions[label] = action
            cache_ids[label] = folder.name
            update_run(out, status=dict(status='running', stage=label, n=len(base), models=actions))
            print(f'{label}: {action}', flush=True)
        result.to_csv(out/'predictions.csv', index=False)
        pd.DataFrame([split.metadata for split in splits]).to_csv(out/'training_updates.csv', index=False)
        names = [variant[0] for variant in variants] + ['zero', 'majority']
        metrics = write_reports(result, names, config, out)
        mdi = pd.concat(importance_frames, ignore_index=True)
        mdi.to_csv(out/'feature_importance_daily.csv', index=False)
        write_tree_reports(out, config, result, metrics, daily, permutation, mdi)
        update_run(out, active_model_cache=cache_ids)
        update_run(out, status=dict(status='success', stage='complete', n=len(base),
                    n_updates=len(splits), models=actions, selection=selection_action,
                    selected_days=int((daily.selection_reason == 'top_k').sum()),
                    fallback_days=int((daily.selection_reason != 'top_k').sum())))
        print(metrics.to_string(index=False), flush=True)
        return out
    except Exception as exc:
        update_run(out, status=dict(status='failed', error=str(exc)))
        raise
