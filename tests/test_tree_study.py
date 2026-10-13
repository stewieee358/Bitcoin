"""Small artificial tests only; these observations never enter real experiments."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.features import build_features
from src.prepare_data import add_features, INTERVALS
from src.models import build_model, MODEL_REGISTRY
from src.tree_models import register_tree_models
from src.tree_study import (validate_study, internal_selection_rows, select_for_day,
                            fit_tree_day, read_selection_cache, run_tree_study)
from src.validation import make_splits
from src.experiment import file_digest
from src.metadata import update_run, read_run

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    config = json.loads((ROOT/'configs/config_tree_study_2h_2025.json').read_text())
    config.update(months=['2024-09', '2024-10', '2024-11', '2024-12', '2025-01'],
                  evaluation_end='2025-01-03T00:00:00+00:00', parallel_days=1)
    config['tree_study']['permutation_repeats'] = 2
    config['models']['rf']['n_estimators'] = 5
    for name in ('ada', 'gbm'):
        config['models'][name]['n_estimators'] = 5
    config['models'] = {key: value for key, value in config['models'].items()
                        if key in ('logistic', 'rf', 'dt', 'ada', 'gbm')}
    config = validate_study(config)
    times = pd.date_range('2024-09-01', '2025-01-04', freq='2h', inclusive='left', tz='UTC')
    rng = np.random.default_rng(91)
    close = 100*np.exp(np.cumsum(rng.normal(0, .002, len(times))))
    opens = np.r_[close[0], close[:-1]]
    volume = rng.uniform(1, 5, len(times))
    raw = pd.DataFrame(dict(open_time=times, open=opens, high=np.maximum(opens, close)*1.001,
             low=np.minimum(opens, close)*.999, close=close, volume=volume,
             quote_asset_volume=volume*close, number_of_trades=rng.integers(50, 200, len(times)),
             taker_buy_base_asset_volume=volume*.6, taker_buy_quote_asset_volume=volume*close*.6,
             close_time=times+pd.Timedelta(hours=2)-pd.Timedelta(microseconds=1), ignore=0))
    frame = build_features(add_features(raw, INTERVALS['2h']), config['feature_columns'], INTERVALS['2h'])
    return config, frame, make_splits(frame, config)


class TreeStudyChecks(unittest.TestCase):
    def test_inner_boundaries_and_known_labels(self):
        config, frame, splits = fixture()
        fit, validation, boundary = internal_selection_rows(frame, splits[0], 30)
        self.assertTrue((fit.target_time <= boundary).all())
        self.assertTrue((fit.feature_time < boundary).all())
        self.assertTrue((validation.feature_time >= boundary).all())
        self.assertTrue((validation.target_time <= splits[0].cutoff).all())
        self.assertEqual(splits[0].cutoff - splits[0].metadata['window_start'], pd.Timedelta(days=90))
        self.assertEqual(splits[1].cutoff-splits[0].cutoff, pd.Timedelta(hours=24))
        # Even a malformed row with a delayed label is purged from the inner fit.
        modified = frame.copy()
        index = fit.index[-1]
        modified.loc[index, 'target_time'] = boundary+pd.Timedelta(hours=2)
        clean, _, _ = internal_selection_rows(modified, splits[0], 30)
        self.assertNotIn(index, clean.index)

    def test_future_changes_cannot_change_selection(self):
        config, frame, splits = fixture()
        columns = config['feature_columns']
        with threadpool_limits(limits=1):
            original = select_for_day(frame, columns, splits[0], config['tree_study'], config['models']['rf'])
            changed = frame.copy()
            future = changed.feature_time >= splits[0].cutoff
            changed.loc[future, columns] *= 1000
            changed.loc[future, 'target'] *= -1
            again = select_for_day(changed, columns, splits[0], config['tree_study'], config['models']['rf'])
        self.assertEqual(original[0]['selected_features'], again[0]['selected_features'])
        pd.testing.assert_frame_equal(original[1], again[1])

    def test_nonpositive_fallback_and_tie_break(self):
        config, frame, splits = fixture()
        columns = config['feature_columns']
        negative = SimpleNamespace(importances_mean=np.full(21, -.01), importances_std=np.zeros(21))
        positive = SimpleNamespace(importances_mean=np.full(21, .01), importances_std=np.zeros(21))
        with patch('src.tree_study.permutation_importance', return_value=negative), threadpool_limits(limits=1):
            audit, _ = select_for_day(frame, columns, splits[0], config['tree_study'], config['models']['rf'])
        self.assertEqual(audit['selection_reason'], 'fallback_no_positive_importance')
        self.assertEqual(json.loads(audit['selected_features']), columns)
        with patch('src.tree_study.permutation_importance', return_value=positive), threadpool_limits(limits=1):
            audit, _ = select_for_day(frame, columns, splits[0], config['tree_study'], config['models']['rf'])
        self.assertEqual(json.loads(audit['selected_features']),
                         columns[:config['tree_study']['top_k']])

    def test_new_probabilities_and_shared_subset(self):
        config, frame, splits = fixture()
        columns = config['feature_columns']
        upstream_rf = MODEL_REGISTRY['rf']
        register_tree_models()
        self.assertIs(MODEL_REGISTRY['rf'], upstream_rf)
        with threadpool_limits(limits=1):
            for name in ('dt', 'ada', 'gbm'):
                probability, detail = fit_tree_day(frame, columns, splits[0], name,
                                                  config['models'][name], columns[:5])
                self.assertEqual(len(probability), len(splits[0].test))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                self.assertEqual(detail.loc[detail.active, 'feature'].tolist(), columns[:5])
                self.assertEqual(len(detail), 21)
                self.assertTrue((detail.loc[~detail.active, 'mdi'] == 0).all())
            with self.assertRaisesRegex(ValueError, 'early-stopping'):
                build_model('gbm', {'n_iter_no_change': 5})

    def test_single_class_prediction_and_selection_fallback(self):
        config, frame, splits = fixture()
        frame.loc[splits[0].train, 'target'] = .01
        audit, _ = select_for_day(frame, config['feature_columns'], splits[0],
                                  config['tree_study'], config['models']['rf'])
        self.assertEqual(audit['selection_reason'], 'fallback_single_class')
        probability, _ = fit_tree_day(frame, config['feature_columns'], splits[0], 'dt',
                                     config['models']['dt'], None)
        np.testing.assert_array_equal(probability, np.ones(len(splits[0].test)))

    def test_selection_cache_rejects_corruption(self):
        config, frame, splits = fixture()
        with threadpool_limits(limits=1):
            rows = [select_for_day(frame, config['feature_columns'], split,
                    config['tree_study'], config['models']['rf']) for split in splits]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            pd.DataFrame([row[0] for row in rows]).to_csv(out/'selection_daily.csv', index=False)
            pd.concat([row[1] for row in rows]).to_csv(out/'permutation_importance_daily.csv', index=False)
            update_run(out, selection_cache=dict(status='success', identity='unit-test',
                   sha256={name: file_digest(out/name) for name in
                           ('selection_daily.csv', 'permutation_importance_daily.csv')}))
            self.assertIsNotNone(read_selection_cache(out, 'unit-test', splits, config['feature_columns']))
            path = out/'selection_daily.csv'
            path.write_text(path.read_text()+'\n')
            self.assertIsNone(read_selection_cache(out, 'unit-test', splits, config['feature_columns']))

    def test_end_to_end_cache_and_shared_rows(self):
        config, frame, splits = fixture()
        manifest = [{'month': 'unit-test', 'url': 'artificial-unit-test-only', 'sha256': 'not-market-data'}]
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1), \
             patch('src.tree_study.load_data', return_value=(frame, {}, manifest)), \
             patch('src.tree_reports.write_tree_reports'), \
             patch('src.tree_study.write_reports', return_value=pd.DataFrame()):
            out = run_tree_study(config, Path(temp))
            first = pd.read_csv(out/'predictions.csv')
            for name in ['dt', 'ada', 'gbm', 'dt_selected', 'ada_selected', 'gbm_selected']:
                self.assertTrue(first[name+'_probability'].between(0, 1).all())
            with patch('src.tree_study.train_tree', side_effect=AssertionError('cache should reuse')), \
                 patch('src.tree_study.calculate_selections', side_effect=AssertionError('selection should reuse')):
                again = run_tree_study(config, Path(temp))
            self.assertEqual(again, out)
            pd.testing.assert_frame_equal(first, pd.read_csv(out/'predictions.csv'))
            self.assertTrue(all(value == 'reused' for value in read_run(out)['status']['models'].values()))
            self.assertEqual(len(list(out.rglob('*.json'))), 1)
            with self.assertRaisesRegex(ValueError, 'another study'):
                run_tree_study({**config, 'evaluation_end': '2025-01-02T00:00:00+00:00'}, Path(temp), out)


if __name__ == '__main__':
    unittest.main()
