"""Integration checks for the dashboard's real data, controls and empty states."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class DashboardTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(at.exception, [str(error.value) for error in at.exception])
        return at

    def assert_ok(self, at):
        self.assertFalse(at.exception, [str(error.value) for error in at.exception])

    def metric(self, at, label):
        return next(metric.value for metric in at.metric if metric.label == label)

    def test_default_real_cohort_counts(self):
        at = self.app()
        self.assertEqual(self.metric(at, "Selected lap records"), "114")
        self.assertEqual(self.metric(at, "Analysis-eligible laps"), "101")
        self.assertEqual(self.metric(at, "Excluded laps"), "13")

    def test_single_driver_and_empty_selection(self):
        at = self.app()
        at.multiselect(key="drivers").set_value(["LEC"]).run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, "Selected lap records"), "57")
        self.assertEqual(self.metric(at, "Analysis-eligible laps"), "50")
        at.radio(key="view").set_value("Tyre stints").run()
        self.assert_ok(at)
        self.assertTrue(any("both drivers" in info.value for info in at.info))
        at.multiselect(key="drivers").set_value([]).run()
        self.assert_ok(at)
        self.assertTrue(any("at least one driver" in info.value for info in at.info))

    def test_matched_counts_and_reference_direction(self):
        at = self.app()
        at.radio(key="view").set_value("Tyre stints").run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, "Matched race laps"), "46")
        self.assertEqual(self.metric(at, "Median paired delta"), "-0.368 s")
        at.selectbox(key="reference").set_value("SAI").run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, "Median paired delta"), "+0.368 s")
        at.checkbox(key="same_compound").uncheck().run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, "Matched race laps"), "47")
        self.assertTrue(any("Different compounds" in warning.value for warning in at.warning))

    def test_lap_window_changes_pair_denominator(self):
        at = self.app()
        at.radio(key="view").set_value("Tyre stints").run()
        at.slider(key="lap_range").set_value((2, 3)).run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, "Selected lap records"), "4")
        self.assertEqual(self.metric(at, "Matched race laps"), "2")

    def test_opening_lap_has_no_eligible_data(self):
        at = self.app()
        at.slider(key="lap_range").set_value((1, 1)).run()
        self.assert_ok(at)
        self.assertEqual(self.metric(at, "Analysis-eligible laps"), "0")
        self.assertTrue(any("No analysis-eligible laps" in info.value for info in at.info))
        at.selectbox(key="cohort").set_value("All recorded laps").run()
        self.assert_ok(at)
        self.assertFalse(any("No analysis-eligible laps" in info.value for info in at.info))
        at.radio(key="view").set_value("Tyre stints").run()
        self.assert_ok(at)
        self.assertTrue(any("No shared eligible laps" in info.value for info in at.info))

    def test_audit_preserves_excluded_and_all_records(self):
        at = self.app()
        at.radio(key="view").set_value("Lap audit").run()
        self.assert_ok(at)
        self.assertEqual(len(at.dataframe[1].value), 13)
        at.checkbox(key="excluded_only").uncheck().run()
        self.assert_ok(at)
        self.assertEqual(len(at.dataframe[1].value), 114)


if __name__ == "__main__":
    unittest.main()
