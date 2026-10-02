"""Real whole-weekend replay, independent filters and provenance protection."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import weekend_ui
from weekend_model import score_weekend_predictions

ROOT = Path(__file__).resolve().parents[1]


class WeekendDashboardTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        at.radio(key='study').set_value('Across weekends').run()
        self.assert_ok(at)
        return at

    def assert_ok(self, at):
        self.assertFalse(at.exception, [str(error.value) for error in at.exception])

    def metric(self, at, label):
        return next(metric.value for metric in at.metric if metric.label == label)

    def test_default_fixed_split_and_event_failure_visible(self):
        at = self.app()
        for label, value in [('Race weekends', '6'), ('Training targets', '708'), ('Validation targets', '232'), ('Test targets', '596'), ('Scored weekend targets', '596')]:
            self.assertEqual(self.metric(at, label), value)
        self.assertEqual(self.metric(at, 'Weekend Ridge MAE'), '0.281 s')
        self.assertEqual(self.metric(at, 'Weekend last-lap MAE'), '0.288 s')
        self.assertEqual(self.metric(at, 'Weekend median MAE'), '0.287 s')
        self.assertEqual(len(at.dataframe[0].value), 6)
        self.assertEqual(at.dataframe[1].value['Targets'].tolist(), [262, 334])
        self.assertTrue(any('Miami Grand Prix: Ridge loses' in caption.value for caption in at.caption))
        self.assertEqual(len(at.get('plotly_chart')), 1)

    def test_event_filter_preserves_fixed_fit_and_resets_replay(self):
        at = self.app()
        at.selectbox(key='weekend_replay_event').set_value('2024-07').run()
        at.multiselect(key='weekend_events').set_value(['2024-06']).run()
        self.assert_ok(at)
        self.assertEqual(at.selectbox(key='weekend_replay_event').value, '2024-06')
        self.assertEqual(self.metric(at, 'Scored weekend targets'), '262')
        self.assertEqual(self.metric(at, 'Weekend last-lap MAE'), '0.280 s')
        self.assertEqual(self.metric(at, 'Training targets'), '708')
        at.multiselect(key='weekend_events').set_value(['2024-07']).run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, 'Scored weekend targets'), '334')
        self.assertEqual(self.metric(at, 'Weekend last-lap MAE'), '0.295 s')

    def test_phase_driver_and_empty_filters(self):
        at = self.app()
        at.radio(key='weekend_phase').set_value('Validation weekend').run()
        self.assert_ok(at)
        self.assertEqual(at.multiselect(key='weekend_events').value, ['2024-05'])
        self.assertEqual(self.metric(at, 'Scored weekend targets'), '232')
        self.assertEqual(self.metric(at, 'Weekend Ridge MAE'), '0.225 s')
        at.multiselect(key='weekend_drivers').set_value(['LEC']).run()
        self.assert_ok(at)
        self.assertEqual(at.selectbox(key='weekend_replay_driver').value, 'LEC')
        at.radio(key='weekend_phase').set_value('Test weekends').run()
        self.assert_ok(at)
        self.assertEqual(at.multiselect(key='weekend_events').value, ['2024-06', '2024-07'])
        self.assertEqual(at.multiselect(key='weekend_drivers').value, ['LEC'])
        at.multiselect(key='weekend_drivers').set_value([]).run()
        self.assert_ok(at)
        self.assertTrue(any('Select at least one held-out weekend' in info.value for info in at.info))
        self.assertEqual(len(at.get('plotly_chart')), 0)

    def test_bahrain_filters_do_not_limit_weekend_study(self):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        at.multiselect(key='drivers').set_value(['LEC']).run()
        at.slider(key='lap_range').set_value((1, 1)).run()
        at.radio(key='study').set_value('Across weekends').run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, 'Scored weekend targets'), '596')
        self.assertEqual(len(at.multiselect(key='weekend_drivers').value), 6)
        at.radio(key='study').set_value('Bahrain case study').run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, 'Selected lap records'), '114')

    def test_missing_report_and_altered_source_are_actionable(self):
        with patch('weekend_ui.REPORTS', ROOT / 'reports' / 'missing_weekend_test'):
            at = self.app()
            self.assertTrue(any('Weekend evaluation is missing' in info.value for info in at.info))
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            altered = Path(directory)
            for path in (ROOT / 'data' / 'weekends').iterdir():
                if path.is_file():
                    shutil.copy2(path, altered / path.name)
            changed = altered / '2024-06_laps.csv'
            changed.write_bytes(changed.read_bytes() + b'\n')
            weekend_ui.load_weekend_reports.clear()
            with patch('weekend_ui.SOURCE', altered):
                at = self.app()
                self.assertTrue(any('Source timing changed for Miami' in error.value for error in at.error))
        weekend_ui.load_weekend_reports.clear()

    def test_saved_metrics_source_hashes_and_pilot_preservation(self):
        directory = ROOT / 'reports' / 'weekends'
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, artifact in manifest['artifacts'].items():
            self.assertEqual(sha256((directory / name).read_bytes()).hexdigest(), artifact['sha256'])
        for name, digest in manifest['original_pilot_sha256'].items():
            self.assertEqual(sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        for name, digest in manifest['code_sha256'].items():
            self.assertEqual(sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        predictions = pd.read_csv(directory / 'predictions.csv')
        saved = pd.read_csv(directory / 'metrics.csv')
        pd.testing.assert_frame_equal(score_weekend_predictions(predictions), saved, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertEqual(predictions.loc[predictions.Split.eq('test'), 'EventKey'].nunique(), 2)
        self.assertTrue(manifest['original_pilot_unchanged'])
        self.assertTrue(manifest['model']['roundtrip_prediction_verified'])


if __name__ == '__main__':
    unittest.main()
