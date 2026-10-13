"""Isolated artificial checks for aggregation, causal splits and cache behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.data import aggregate_bars, load_data
from src.download_data import archive_url
from src.experiment import normalize_config, run_experiment
from src.evaluation import scores, write_reports
from src.features import build_features, FEATURE_REGISTRY, feature_signature, continuous_history
from src.models import build_model, logistic, ModelDefinition, MODEL_REGISTRY, model_signature
from src.prepare_data import add_features, INTERVALS
from src.training import train_model
from src.validation import make_splits
from src.metadata import read_run

ROOT = Path(__file__).resolve().parents[1]


def bars(times):
    close = 100+np.sin(np.arange(len(times)))
    step = times[1]-times[0]
    frame = pd.DataFrame(dict(open_time=times, open=close, high=close+.1, low=close-.1, close=close,
                        volume=1., quote_asset_volume=100., number_of_trades=2,
                        taker_buy_base_asset_volume=.5, taker_buy_quote_asset_volume=50.,
                        close_time=times+step-pd.Timedelta(microseconds=1), ignore=0))
    return add_features(frame, step)


class FrameworkChecks(unittest.TestCase):
    def test_documented_feature_formulas_and_periods(self):
        names = list(FEATURE_REGISTRY)
        for interval in ('15m', '2h', '1d'):
            step = INTERVALS[interval]
            n = int(pd.Timedelta(days=1) / step)
            frame = bars(pd.date_range('2025-01-01', periods=7*n+10, freq=step, tz='UTC'))
            frame['volume'] = np.arange(len(frame)) + 1.
            frame['taker_buy_base_asset_volume'] = frame.volume * .6
            result = build_features(frame, names, step)
            r = frame.close.pct_change(fill_method=None)
            width = (frame.high-frame.low).replace(0, np.nan)
            expected = {
                'return': r, 'return_lag1': r.shift(1),
                'return_7d': frame.close/frame.close.shift(7*n)-1,
                'ma_deviation': frame.close/frame.close.rolling(n).mean()-1,
                'volatility': r.rolling(max(2, n)).std(ddof=1),
                'price_range': (frame.high-frame.low)/frame.close,
                'body_ratio': (frame.close-frame.open).abs()/width,
                'upper_shadow_ratio': (frame.high-frame[['open', 'close']].max(axis=1))/width,
                'lower_shadow_ratio': (frame[['open', 'close']].min(axis=1)-frame.low)/width,
                'volume_change': frame.volume/frame.volume.shift(1)-1,
                'relative_volume': frame.volume/frame.volume.shift(1).rolling(n).mean()-1,
                'taker_buy_ratio': frame.taker_buy_base_asset_volume/frame.volume,
            }
            for name, values in expected.items():
                np.testing.assert_allclose(result[name], values, equal_nan=True, err_msg=f'{interval}: {name}')
            self.assertFalse(result.valid_features.iloc[7*n-1])
            self.assertTrue(result.valid_features.iloc[7*n])
        # Nontrivial candle bodies/shadows, independent hand calculation.
        frame.loc[0, ['open', 'high', 'low', 'close']] = [10., 14., 8., 12.]
        result = build_features(frame, names, step)
        self.assertAlmostEqual(result.price_range.iloc[0], .5)
        for name in ('body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio'):
            self.assertAlmostEqual(result[name].iloc[0], 1/3)

    def test_documented_features_gaps_zero_divisors_and_causality(self):
        step = INTERVALS['2h']
        names = list(FEATURE_REGISTRY)
        frame = bars(pd.date_range('2025-01-01', periods=130, freq=step, tz='UTC'))
        original = build_features(frame.iloc[:110], names, step)
        changed = frame.copy()
        changed.loc[110:, ['open', 'high', 'low', 'close', 'volume']] *= 2
        full = build_features(changed, names, step)
        pd.testing.assert_frame_equal(original[names], full.loc[original.index, names])
        # Label contents cannot affect input features.
        changed['target'] = 999.
        pd.testing.assert_frame_equal(full[names], build_features(changed, names, step)[names])
        gap = build_features(frame.drop(index=20), names, step)
        for name in ('return', 'return_lag1', 'return_7d', 'ma_deviation',
                     'volatility', 'volume_change', 'relative_volume'):
            self.assertTrue(pd.isna(gap.loc[21, name]), name)
        self.assertTrue(pd.isna(gap.loc[22, 'return_lag1']))
        self.assertTrue(pd.notna(gap.loc[23, 'return_lag1']))
        self.assertTrue(pd.isna(gap.loc[103, 'return_7d']))
        self.assertTrue(pd.notna(gap.loc[105, 'return_7d']))
        zero = frame.copy()
        zero.loc[0:12, 'volume'] = 0.
        zero.loc[12, ['open', 'high', 'low', 'close']] = 100.
        result = build_features(zero, names, step)
        for name in ('body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
                     'taker_buy_ratio', 'volume_change', 'relative_volume'):
            self.assertTrue(pd.isna(result.loc[12, name]), name)
        self.assertFalse(result.valid_features.loc[12])

    def test_feature_helper_implementation_is_in_cache_identity(self):
        signature = feature_signature(['return_7d', 'volume_change', 'return'])
        import inspect
        self.assertIn(inspect.getsource(continuous_history), signature['return_7d'])
        self.assertIn(inspect.getsource(continuous_history), signature['volume_change'])
        self.assertEqual(signature['return'], inspect.getsource(FEATURE_REGISTRY['return']))

    def test_roc_uses_probability_ranking_and_reduced_outputs(self):
        y = [0, 0, 1, 1]
        self.assertEqual(scores(y, [0, 0, 1, 1], [.1, .2, .8, .9])['roc_auc'], 1.)
        self.assertEqual(scores(y, [0]*4, [.5]*4)['roc_auc'], .5)
        self.assertTrue(np.isnan(scores([0, 0], [0, 0], [.1, .2])['roc_auc']))
        pred = pd.DataFrame(dict(feature_time=pd.date_range('2025-01-01', periods=4, freq='h', tz='UTC'),
                                 actual_class=y, logistic_class=[0, 0, 1, 1], logistic_probability=[.1, .2, .8, .9]))
        config = dict(interval='15m', window_days=90, update_hours=24, feature_columns=['return'],
                      evaluation_start='2025-01-01', evaluation_end='2025-01-02')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            out = Path(directory)
            write_reports(pred, ['logistic'], config, out)
            self.assertEqual({p.name for p in out.iterdir()}, {'overall_metrics.csv', 'summary.md', 'roc_curve.png'})

    def test_native_interval_default_and_explicit_aggregation(self):
        config = normalize_config(dict(interval='2h', months=['2025-01'], window_days=10,
                                  models={'logistic': {}}))
        self.assertEqual(config['source_interval'], '2h')
        self.assertIn('/2h/BTCUSDT-2h-2025-01.zip', archive_url('2025-01', '2h'))
        native = bars(pd.date_range('2025-01-01', periods=10, freq='2h', tz='UTC'))
        with patch('src.data.download', return_value=([], [])) as download, \
             patch('src.data.prepare', return_value=(native, {})) as prepare:
            result, _, _ = load_data(config, ROOT/'data/raw')
            self.assertEqual(download.call_args.args[0]['interval'], '2h')
            self.assertEqual(prepare.call_args.args[1], '2h')
            self.assertIs(result, native)
        raw = bars(pd.date_range('2025-01-01', periods=16, freq='15min', tz='UTC'))
        audit = dict(requested_start='2025-01-01T00:00Z', requested_end_exclusive='2025-02-01T00:00Z', actual_rows=16)
        with patch('src.data.download', return_value=([], [])) as download, \
             patch('src.data.prepare', return_value=(raw, audit)):
            result, _, _ = load_data({**config, 'source_interval': '15m'}, ROOT/'data/raw')
            self.assertEqual(download.call_args.args[0]['interval'], '15m')
            self.assertEqual(len(result), 2)

    def test_complete_utc_aggregation_and_gaps(self):
        times = pd.date_range('2025-01-01', periods=192, freq='15min', tz='UTC')
        raw = bars(times)
        result = aggregate_bars(raw, '2h')
        self.assertEqual(len(result), 24)
        self.assertEqual(result.volume.iloc[0], 8)
        self.assertEqual(result.number_of_trades.iloc[0], 16)
        self.assertEqual(result.open.iloc[0], raw.open.iloc[0])
        self.assertEqual(result.close.iloc[0], raw.close.iloc[7])
        self.assertEqual(result.target_time.iloc[0], pd.Timestamp('2025-01-01T04:00Z'))
        gap = aggregate_bars(raw.drop(index=9), '2h')
        self.assertEqual(len(gap), 23)
        self.assertTrue(pd.isna(gap.target.iloc[0]))
        self.assertTrue(pd.isna(gap['return'].iloc[1]))
        daily = aggregate_bars(raw, '1d')
        self.assertEqual(len(daily), 2)
        self.assertEqual(daily.volume.iloc[0], 96)

    def test_shared_feature_mask_and_validation_interface(self):
        frame = bars(pd.date_range('2025-01-01', periods=31, freq='D', tz='UTC'))
        def extra(data, step):
            return data.close.shift(2)
        with patch.dict(FEATURE_REGISTRY, {'lagged_close': extra}):
            result = build_features(frame, ['return', 'lagged_close'], INTERVALS['1d'])
        self.assertFalse(result.valid_sample.iloc[1])
        config = normalize_config(dict(interval='1d', months=['2025-01'], window_days=10,
                                  evaluation_start='2025-01-15', evaluation_end='2025-01-18',
                                  feature_columns=['return'], models={'logistic': {}}))
        splits = make_splits(result, config)
        for split in splits:
            self.assertTrue((result.loc[split.train].target_time <= split.cutoff).all())
            self.assertTrue((result.loc[split.test].feature_time >= split.cutoff).all())
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            make_splits(result, {**config, 'validation': 'expanding'})

    def test_model_probability_interfaces(self):
        X = pd.DataFrame({'x': np.arange(40), 'z': np.sin(np.arange(40))})
        y = (X.z>0).astype(int)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            for name, params in [('svm', {'C': 1}), ('fnn', {'hidden_layer_sizes': [4], 'max_iter': 20})]:
                model = build_model(name, params).fit(X, y)
                probability = model.predict_proba(X)
                self.assertEqual(probability.shape, (40, 2))
                np.testing.assert_allclose(probability.sum(axis=1), 1)
        with self.assertRaises(ValueError):
            build_model('fnn', {'early_stopping': True})

    def test_pca_variants_training_only_and_identity(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=60)
        X = pd.DataFrame({'x': x, 'duplicate': x, 'z': rng.normal(size=60)})
        y = (x > 0).astype(int)
        for name in ('logistic_pca', 'knn_pca'):
            params = {'pca_n_components': 0.95}
            with threadpool_limits(limits=1):
                model = build_model(name, params).fit(X.iloc[:40], y[:40])
                scaler = model.named_steps['standardscaler']
                pca = model.named_steps['pca']
                np.testing.assert_allclose(scaler.mean_, X.iloc[:40].mean())
                self.assertLess(pca.n_components_, X.shape[1])
                before = pca.components_.copy()
                probability = model.predict_proba(X.iloc[40:] * 1000)
                np.testing.assert_array_equal(before, pca.components_)
                self.assertEqual(probability.shape, (20, 2))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue(((probability >= 0) & (probability <= 1)).all())
                np.testing.assert_allclose(probability.sum(axis=1), 1)
            self.assertEqual(params, {'pca_n_components': 0.95})
            self.assertNotEqual(model_signature(name, params),
                                model_signature(name, {'pca_n_components': 0.9}))

    def test_independent_cache_reuse_and_invalidation(self):
        frame = bars(pd.date_range('2025-01-01', periods=31, freq='D', tz='UTC'))
        manifest = [{'month': '2025-01', 'url': 'isolated-test', 'sha256': 'test-only'}]
        config = dict(interval='1d', months=['2025-01'], window_days=10,
                      evaluation_start='2025-01-15', evaluation_end='2025-01-18',
                      feature_columns=['return'], models={'logistic': {'max_iter': 100}})
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, threadpool_limits(limits=1), \
             patch('src.experiment.load_data', return_value=(frame, {}, manifest)), \
             patch('src.experiment.write_reports', return_value=pd.DataFrame()), \
             patch('src.experiment.train_model', wraps=train_model) as fit:
            root = Path(directory)
            out = run_experiment(config, root)
            self.assertEqual(fit.call_count, 1)
            fit.reset_mock()
            self.assertEqual(run_experiment(config, root), out)
            self.assertEqual(fit.call_count, 0)
            pca_config = {**config, 'models': {**config['models'],
                          'logistic_pca': {'pca_n_components': 0.95}}}
            self.assertEqual(run_experiment(pca_config, root), out)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.args[4], 'logistic_pca')
            fit.reset_mock()
            with patch.dict(MODEL_REGISTRY, {'tiny': ModelDefinition(logistic)}):
                extended = {**config, 'models': {**config['models'], 'tiny': {'C': .5}}}
                self.assertEqual(run_experiment(extended, root), out)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(fit.call_args.args[4], 'tiny')
                fit.reset_mock()
                changed = {**extended, 'models': {**extended['models'], 'tiny': {'C': .7}}}
                self.assertEqual(run_experiment(changed, root), out)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(fit.call_args.args[4], 'tiny')
                fit.reset_mock()
                new_features = {**extended, 'feature_columns': ['return', 'volume']}
                new_out = run_experiment(new_features, root)
                self.assertNotEqual(new_out, out)
                self.assertEqual(fit.call_count, 2)
                fit.reset_mock()
                with self.assertRaisesRegex(ValueError, 'different experiment'):
                    run_experiment(new_features, root, output_dir=out)
                # logistic_* also matches logistic_pca_*; corrupt the actual
                # LogisticRegression cache, not an unrelated PCA variant.
                path = next(p for p in (out/'models').glob('logistic_*/predictions.csv')
                            if p.parent.name.rsplit('_', 1)[0] == 'logistic')
                path.write_text(path.read_text()+'\n')
                run_experiment(extended, root)
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(fit.call_args.args[4], 'logistic')
            status = read_run(out)['status']
            self.assertEqual(status['status'], 'success')
            self.assertIn('dependency_versions', read_run(out)['config'])
            self.assertEqual(len(list(out.rglob('*.json'))), 1)
            self.assertFalse((out/'dependency_versions.json').exists())


if __name__ == '__main__':
    unittest.main()
