"""Real-export and dashboard checks for selected-lap public telemetry."""
import base64
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class TelemetryDashboardTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        at.radio(key='view').set_value('Telemetry').run()
        self.assert_ok(at)
        return at

    def assert_ok(self, at):
        self.assertFalse(at.exception, [str(error.value) for error in at.exception])

    def test_export_matches_provenance_and_native_sources(self):
        manifest = json.loads((ROOT / 'data' / 'telemetry_manifest.json').read_text())
        raw = ROOT / 'data' / manifest['telemetry_csv']
        self.assertEqual(sha256(raw.read_bytes()).hexdigest(), manifest['telemetry_sha256'])
        self.assertEqual(sha256((ROOT / 'data' / manifest['source_lap_csv']).read_bytes()).hexdigest(), manifest['source_lap_sha256'])
        samples = pd.read_csv(raw)
        self.assertEqual(len(samples), 41286)
        self.assertEqual(len(samples[['Driver', 'LapNumber']].drop_duplicates()), 114)
        self.assertEqual(set(samples['SampleSource']), {'car'})
        self.assertEqual(sum(q['analysis_eligible'] and q['alignment_eligible'] for q in manifest['lap_quality']), 64)

    def test_default_pair_and_aligned_comparison(self):
        at = self.app()
        self.assertEqual(at.selectbox(key='telemetry_shared_lap').value, 21)
        context, quality = at.dataframe[0].value, at.dataframe[1].value
        self.assertEqual(context['Race lap'].tolist(), [21, 21])
        self.assertEqual(context['Compound'].tolist(), ['HARD', 'HARD'])
        self.assertEqual(quality['Samples'].tolist(), [369, 364])
        self.assertTrue(quality['Distance supported'].all())
        at.checkbox(key='telemetry_delta').check().run()
        self.assert_ok(at)
        self.assertEqual(len(at.get('plotly_chart')), 2)
        self.assertFalse(any('withheld' in info.value for info in at.info))

    def test_gap_lap_withholds_alignment_and_breaks_native_time_lines(self):
        at = self.app()
        at.selectbox(key='telemetry_shared_lap').set_value(20).run()
        at.radio(key='telemetry_axis').set_value('Elapsed time').run()
        at.checkbox(key='telemetry_delta').check().run()
        self.assert_ok(at)
        self.assertTrue(any('withheld' in info.value for info in at.info))
        self.assertEqual(len(at.get('plotly_chart')), 1)
        spec = json.loads(at.get('plotly_chart')[0].proto.spec)
        # Every line explicitly has at least one null separator for the >1 s gap.
        for trace in spec['data']:
            x = trace['x']
            if isinstance(x, dict):
                x = np.frombuffer(base64.b64decode(x['bdata']), dtype=x['dtype'])
                self.assertTrue(np.isnan(x).any())
            else:
                self.assertIn(None, x)
            self.assertFalse(trace['connectgaps'])

    def test_window_reset_single_driver_and_no_eligible_laps(self):
        at = self.app()
        at.slider(key='lap_range').set_value((2, 3)).run()
        self.assert_ok(at)
        self.assertIn(at.selectbox(key='telemetry_shared_lap').value, [2, 3])
        at.multiselect(key='drivers').set_value(['LEC']).run()
        self.assert_ok(at)
        self.assertEqual(at.dataframe[0].value['Driver'].tolist(), ['LEC'])
        self.assertEqual(len(at.get('plotly_chart')), 1)
        at.slider(key='lap_range').set_value((1, 1)).run()
        self.assert_ok(at)
        self.assertTrue(any('No analysis-eligible telemetry' in info.value for info in at.info))

    def test_separate_selection_and_reference_order(self):
        at = self.app()
        at.radio(key='telemetry_mode').set_value('Choose laps separately').run()
        at.selectbox(key='telemetry_lap_SAI').set_value(22).run()
        self.assert_ok(at)
        self.assertEqual(at.dataframe[0].value['Race lap'].tolist(), [21, 22])
        self.assertTrue(any('Separately selected' in warning.value for warning in at.warning))
        at.selectbox(key='reference').set_value('SAI').run()
        self.assert_ok(at)
        self.assertEqual(at.dataframe[0].value['Driver'].tolist(), ['SAI', 'LEC'])

    def test_missing_export_is_actionable_empty_state(self):
        with patch('telemetry_ui.TELEMETRY', ROOT / 'data' / 'missing_test_export.csv'):
            at = self.app()
        self.assertTrue(any('export is missing' in info.value for info in at.info))
        self.assertEqual(len(at.get('plotly_chart')), 0)


if __name__ == '__main__':
    unittest.main()
