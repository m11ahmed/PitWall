"""Real frozen-model replay, filtered errors and actionable empty states."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import model_ui
from pace_model import score_predictions

ROOT = Path(__file__).resolve().parents[1]


class ModelDashboardTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        at.radio(key='view').set_value('ML pace').run()
        self.assert_ok(at)
        return at

    def assert_ok(self, at):
        self.assertFalse(at.exception, [str(error.value) for error in at.exception])

    def metric(self, at, label):
        return next(metric.value for metric in at.metric if metric.label == label)

    def test_default_frozen_test_scores_and_driver_negative_result(self):
        at = self.app()
        self.assertEqual(self.metric(at, 'Training pairs'), '52')
        self.assertEqual(self.metric(at, 'Validation pairs'), '17')
        self.assertEqual(self.metric(at, 'Test pairs'), '22')
        self.assertEqual(self.metric(at, 'Boundary pairs omitted'), '2')
        self.assertEqual(self.metric(at, 'Scored targets'), '22')
        self.assertEqual(self.metric(at, 'Ridge MAE'), '0.144 s')
        self.assertEqual(self.metric(at, 'Last-lap MAE'), '0.156 s')
        self.assertEqual(self.metric(at, 'Recent-median MAE'), '0.181 s')
        self.assertTrue(any('LEC: the last-lap baseline outperforms Ridge' in caption.value for caption in at.caption))
        self.assertEqual(len(at.get('plotly_chart')), 1)

    def test_driver_filter_recomputes_only_scores_and_resets_replay(self):
        at = self.app()
        at.selectbox(key='ml_replay_driver').set_value('SAI').run()
        at.multiselect(key='drivers').set_value(['LEC']).run()
        self.assert_ok(at)
        self.assertEqual(at.selectbox(key='ml_replay_driver').value, 'LEC')
        self.assertEqual(self.metric(at, 'Scored targets'), '11')
        self.assertEqual(self.metric(at, 'Ridge MAE'), '0.169 s')
        self.assertEqual(self.metric(at, 'Last-lap MAE'), '0.160 s')
        self.assertTrue(any('baseline performs better' in warning.value for warning in at.warning))
        self.assertEqual(self.metric(at, 'Training pairs'), '52')
        at.multiselect(key='drivers').set_value(['SAI']).run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, 'Ridge MAE'), '0.119 s')
        self.assertEqual(self.metric(at, 'Last-lap MAE'), '0.152 s')

    def test_validation_window_and_single_target_inspection(self):
        at = self.app()
        at.radio(key='ml_phase').set_value('Validation (used to choose alpha)').run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, 'Scored targets'), '17')
        self.assertEqual(self.metric(at, 'Ridge MAE'), '0.203 s')
        at.selectbox(key='ml_target_lap').set_value(40).run()
        self.assert_ok(at)
        self.assertTrue(any('after lap 39, predict lap 40' in md.value for md in at.markdown))
        at.slider(key='lap_range').set_value((40, 40)).run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, 'Scored targets'), '2')
        self.assertEqual(at.selectbox(key='ml_target_lap').value, 40)

    def test_training_window_has_no_held_out_targets(self):
        at = self.app()
        at.slider(key='lap_range').set_value((2, 5)).run()
        self.assert_ok(at)
        self.assertTrue(any('No held-out targets' in info.value for info in at.info))
        self.assertEqual(len(at.get('plotly_chart')), 0)

    def test_missing_artifacts_and_changed_source_are_rejected(self):
        with patch('model_ui.ML_REPORTS', ROOT / 'reports' / 'missing_model_test'):
            at = self.app()
            self.assertTrue(any('Model artifacts are missing' in info.value for info in at.info))
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            changed = Path(directory) / 'changed_source.csv'
            changed.write_bytes(model_ui.INPUT.read_bytes() + b'\n')
            model_ui.load_model_reports.clear()
            with patch('model_ui.INPUT', changed):
                at = self.app()
                self.assertTrue(any('Source timing changed' in error.value for error in at.error))
        model_ui.load_model_reports.clear()

    def test_saved_prediction_metrics_match_recomputation_and_hashes(self):
        directory = ROOT / 'reports' / 'ml'
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, artifact in manifest['artifacts'].items():
            self.assertEqual(sha256((directory / name).read_bytes()).hexdigest(), artifact['sha256'])
        predictions = pd.read_csv(directory / 'predictions.csv')
        recomputed = score_predictions(predictions)
        saved = pd.read_csv(directory / 'metrics.csv')
        pd.testing.assert_frame_equal(recomputed, saved, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertTrue(manifest['model']['roundtrip_prediction_verified'])
        self.assertEqual(manifest['protocol']['selected_alpha'], 10)


if __name__ == '__main__':
    unittest.main()
