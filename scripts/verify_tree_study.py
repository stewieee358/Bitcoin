"""Independent real-data timing, metric, cache and first-day refit verification."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data import load_data
from src.evaluation import scores
from src.experiment import read_predictions, read_updates, file_digest, check_predictions
from src.features import build_features
from src.metadata import read_run
from src.prepare_data import INTERVALS
from src.training import prediction_base, predict_split
from src.tree_study import validate_study, select_for_day, fit_tree_day, INSPECT_NAMES
from src.validation import make_splits

ROOT = Path(__file__).resolve().parents[1]


def verify(out):
    document = read_run(out)
    if document['status']['status'] != 'success':
        raise ValueError('The study is not complete')
    config = validate_study(document['config'])
    frame, _, _ = load_data(config, ROOT/'data/raw')
    frame = build_features(frame, config['feature_columns'], INTERVALS['2h'])
    splits = make_splits(frame, config)
    base = prediction_base(frame, splits)
    pred = read_predictions(out/'predictions.csv')
    pd.testing.assert_frame_equal(pred[list(base.columns)], base, check_exact=False, rtol=0, atol=1e-14)
    pd.testing.assert_frame_equal(read_updates(out/'training_updates.csv'), pd.DataFrame([s.metadata for s in splits]),
                                  check_exact=False, rtol=0, atol=1e-14)
    assert (pred.target_time == pred.feature_time + pd.Timedelta(hours=2)).all()
    assert (pred.train_cutoff <= pred.feature_time).all()
    assert not pred.feature_time.duplicated().any()
    selection = pd.read_csv(out/'selection_daily.csv')
    assert list(pd.to_datetime(selection.train_cutoff, utc=True)) == [split.cutoff for split in splits]
    for name, expected in document['selection_cache']['sha256'].items():
        assert file_digest(out/name) == expected, f'Selection cache changed: {name}'
    measured = selection.inner_fit_target_end.notna()
    assert (pd.to_datetime(selection.loc[measured, 'inner_fit_target_end'], utc=True) <=
            pd.to_datetime(selection.loc[measured, 'inner_cutoff'], utc=True)).all()
    measured = selection.inner_validation_target_end.notna()
    assert (pd.to_datetime(selection.loc[measured, 'inner_validation_target_end'], utc=True) <=
            pd.to_datetime(selection.loc[measured, 'train_cutoff'], utc=True)).all()
    first = next(split for split in splits if len(split.test))
    index = [split.cutoff for split in splits].index(first.cutoff)
    audit, _ = select_for_day(frame, config['feature_columns'], first, config['tree_study'], config['models']['rf'])
    selected = json.loads(selection.selected_features.iloc[index])
    assert selected == json.loads(audit['selected_features'])
    metrics = pd.read_csv(out/'overall_metrics.csv').set_index('model')
    for label in metrics.index:
        probability, direction = pred[label+'_probability'], pred[label+'_class']
        assert probability.between(0, 1).all()
        assert (direction == (probability > .5).astype(int)).all()
        for metric, value in scores(pred.actual_class, direction, probability).items():
            np.testing.assert_allclose(value, metrics.loc[label, metric], rtol=0, atol=1e-12, equal_nan=True)
        if label in ('zero', 'majority'):
            continue
        cache_name = document['active_model_cache'][label]
        cache = document['model_cache'][cache_name]
        cache_path = out/'models'/cache_name/'predictions.csv'
        assert cache['status'] == 'success'
        assert file_digest(cache_path) == cache['sha256']
        cached = read_predictions(cache_path)
        check_predictions(cached, base)
        np.testing.assert_allclose(cached.probability, probability, rtol=0, atol=1e-14)
        name = label.removesuffix('_selected')
        if name in INSPECT_NAMES:
            assert file_digest(out/'importance'/f'{label}.csv') == cache['importance_sha256']
            check, _ = fit_tree_day(frame, config['feature_columns'], first, name, config['models'][name],
                                   selected if label.endswith('_selected') else None)
        else:
            check = predict_split(frame, config['feature_columns'], first, name, config['models'][name])
        np.testing.assert_allclose(check, probability.iloc[:len(first.test)], rtol=0, atol=1e-12)
    print(f'Verified: {len(pred)} shared rows, {len(splits)} daily updates, '
          f'{len(metrics)-2} models, known-label cutoffs, file hashes, metrics and independent first-day refits.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.output_dir.resolve())
