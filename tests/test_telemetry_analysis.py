"""Telemetry mathematics/quality checks with known and adversarial samples."""
import json
import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from telemetry_analysis import align_laps_by_distance, prepare_lap_telemetry


def telemetry(times=(0, 0.5, 1, 1.5, 2), *, speed=72, throttle=100, brake=False, driver="LEC", lap=20):
    count = len(times)
    def expand(value):
        return value if isinstance(value, (list, tuple)) else [value] * count
    return pd.DataFrame({
        "Driver": [driver] * count, "LapNumber": [lap] * count,
        "ElapsedSeconds": list(times), "SpeedKmh": expand(speed),
        "ThrottlePct": expand(throttle), "BrakeOn": expand(brake),
        "RPM": [10000] * count, "Gear": [4] * count,
        "DRS": [0] * count, "SampleSource": ["car"] * count,
    })


def prepared(data=None, *, duration=2, **kwargs):
    if data is None:
        data = telemetry()
    return prepare_lap_telemetry(data, driver="LEC", lap_number=20, lap_time_seconds=duration, **kwargs)


class TelemetryAnalysisTests(unittest.TestCase):
    def test_known_constant_speed_integral_and_cadence(self):
        lap = prepared()
        np.testing.assert_allclose(lap.samples["EstimatedDistanceM"], [0, 10, 20, 30, 40])
        self.assertEqual(lap.quality.median_sample_interval_seconds, 0.5)
        self.assertEqual(lap.quality.max_sample_interval_seconds, 0.5)
        self.assertEqual(lap.quality.estimated_observed_distance_m, 40)
        self.assertTrue(lap.quality.alignment_eligible)
        self.assertEqual(lap.quality.observed_time_fraction, 1)
        self.assertFalse(lap.samples["GapBefore"].any())
        json.dumps(asdict(lap.quality), allow_nan=False)

    def test_trapezoid_integral_uses_both_endpoints(self):
        lap = prepared(telemetry([0, 0.5, 1], speed=[0, 36, 72]), duration=1)
        np.testing.assert_allclose(lap.samples["EstimatedDistanceM"], [0, 2.5, 10])

    def test_sample_origin_is_not_extrapolated_to_lap_boundary(self):
        lap = prepared(telemetry([0.25, 1.25, 2.25]), duration=2.5)
        np.testing.assert_allclose(lap.samples["EstimatedDistanceM"], [0, 20, 40])
        self.assertEqual(lap.quality.start_missing_seconds, 0.25)
        self.assertEqual(lap.quality.end_missing_seconds, 0.25)
        self.assertEqual(lap.quality.observed_time_fraction, 0.8)
        self.assertTrue(lap.quality.boundary_coverage_within_tolerance)
        self.assertTrue(any("not extrapolated" in warning for warning in lap.quality.warnings))

    def test_temporal_order_is_sorted_and_disclosed_without_mutation(self):
        data = telemetry().iloc[[4, 0, 2, 1, 3]].reset_index(drop=True)
        before = data.copy(deep=True)
        lap = prepared(data)
        self.assertTrue(lap.quality.reordered)
        self.assertEqual(lap.samples["ElapsedSeconds"].tolist(), [0, 0.5, 1, 1.5, 2])
        assert_frame_equal(data, before)

    def test_duplicate_valid_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepared(telemetry([0, 0.5, 0.5, 2]))

    def test_invalid_timestamps_retain_rows_and_disable_distance(self):
        for bad_time in (None, -0.1, float("inf"), 2.1, "unknown"):
            with self.subTest(bad_time=bad_time):
                lap = prepared(telemetry([0, bad_time, 2]))
                self.assertEqual(len(lap.samples), 3)
                self.assertEqual(lap.quality.invalid_elapsed_samples, 1)
                self.assertTrue(lap.samples["EstimatedDistanceM"].isna().all())
                self.assertFalse(lap.quality.alignment_eligible)
                self.assertIn("RawElapsedSeconds", lap.samples)

    def test_speed_failure_withholds_all_later_distance(self):
        for bad_speed in (None, -1, float("inf"), "unknown"):
            with self.subTest(bad_speed=bad_speed):
                lap = prepared(telemetry(speed=[72, 72, bad_speed, 72, 72]))
                self.assertEqual(lap.quality.invalid_speed_samples, 1)
                self.assertEqual(lap.quality.distance_supported_samples, 2)
                self.assertEqual(lap.samples["EstimatedDistanceM"].iloc[:2].tolist(), [0, 10])
                self.assertTrue(lap.samples["EstimatedDistanceM"].iloc[2:].isna().all())
                self.assertFalse(lap.quality.alignment_eligible)

    def test_large_gap_is_not_integrated_or_stitched(self):
        lap = prepared(telemetry([0, 0.2, 2, 2.2]), duration=2.2)
        self.assertEqual(lap.quality.gap_count, 1)
        self.assertEqual(lap.samples["GapBefore"].tolist(), [False, False, True, False])
        self.assertAlmostEqual(lap.quality.max_sample_interval_seconds, 1.8)
        np.testing.assert_allclose(lap.samples["EstimatedDistanceM"].iloc[:2], [0, 4])
        self.assertTrue(lap.samples["EstimatedDistanceM"].iloc[2:].isna().all())
        with self.assertRaisesRegex(ValueError, "continuous distance"):
            align_laps_by_distance(lap, prepared())

    def test_invalid_throttle_and_brake_stay_missing(self):
        lap = prepared(telemetry(throttle=[0, 101, -1, "unknown", 100],
                                 brake=["False", "True", "unknown", 2, 0]))
        self.assertEqual(lap.quality.invalid_throttle_samples, 3)
        self.assertEqual(lap.quality.invalid_brake_samples, 2)
        self.assertTrue(lap.samples["ThrottlePct"].iloc[1:4].isna().all())
        self.assertEqual(lap.samples["BrakeOn"].iloc[:2].tolist(), [0, 1])
        self.assertTrue(lap.quality.alignment_eligible)

    def test_common_grid_never_extrapolates_beyond_shorter_lap(self):
        left = prepared()
        right = prepared(telemetry(speed=36))
        aligned = align_laps_by_distance(left, right, spacing_m=7)
        self.assertEqual(aligned["EstimatedDistanceM"].tolist(), [0, 7, 14, 20])
        self.assertTrue(aligned["LeftSpeedKmh"].eq(72).all())
        self.assertTrue(aligned["RightSpeedKmh"].eq(36).all())
        self.assertTrue(aligned["SpeedDifferenceKmh"].eq(-36).all())

    def test_alignment_swapping_laps_reverses_speed_delta(self):
        left = prepared()
        right = prepared(telemetry(speed=36))
        forward = align_laps_by_distance(left, right)
        reverse = align_laps_by_distance(right, left)
        np.testing.assert_allclose(forward["SpeedDifferenceKmh"], -reverse["SpeedDifferenceKmh"])

    def test_channel_gaps_are_not_interpolated_across(self):
        left = prepared(telemetry(throttle=[0, None, 50, 100, 100],
                                 brake=[False, None, True, False, True]))
        aligned = align_laps_by_distance(left, prepared(), spacing_m=5)
        gap_rows = aligned["EstimatedDistanceM"].isin([5, 10, 15])
        self.assertTrue(aligned.loc[gap_rows, "LeftThrottlePct"].isna().all())
        self.assertTrue(aligned.loc[gap_rows, "LeftBrakeOn"].isna().all())
        self.assertEqual(aligned.loc[aligned["EstimatedDistanceM"].eq(20), "LeftThrottlePct"].iloc[0], 50)

    def test_brake_is_step_binary_and_never_pressure(self):
        left = prepared(telemetry(brake=[False, True, False, True, False]))
        aligned = align_laps_by_distance(left, prepared(), spacing_m=5)
        self.assertEqual(aligned["LeftBrakeOn"].tolist(), [0, 0, 1, 1, 0, 0, 1, 1, 0])
        self.assertTrue(aligned["LeftBrakeOn"].isin([0, 1]).all())

    def test_insufficient_boundary_coverage_refuses_alignment(self):
        lap = prepared(telemetry([2, 3, 4]), duration=4)
        self.assertTrue(lap.quality.distance_is_continuous)
        self.assertFalse(lap.quality.boundary_coverage_within_tolerance)
        with self.assertRaisesRegex(ValueError, "boundary coverage"):
            align_laps_by_distance(lap, prepared())

    def test_stationary_intervals_and_single_samples_refuse_alignment(self):
        for data in (telemetry(speed=0), telemetry([0], speed=72)):
            with self.subTest(count=len(data)):
                lap = prepared(data)
                self.assertFalse(lap.quality.alignment_eligible)
                with self.assertRaises(ValueError):
                    align_laps_by_distance(lap, prepared())

    def test_selection_and_invalid_configuration(self):
        mixed = pd.concat([telemetry(driver="SAI"), telemetry()])
        lap = prepared(mixed)
        self.assertEqual(lap.quality.sample_count, 5)
        self.assertEqual(lap.samples["Driver"].unique().tolist(), ["LEC"])
        with self.assertRaisesRegex(ValueError, "No telemetry"):
            prepare_lap_telemetry(mixed, driver="NOR", lap_number=20, lap_time_seconds=2)
        for spacing in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                align_laps_by_distance(prepared(), prepared(), spacing_m=spacing)
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepared(duration=duration)
        with self.assertRaisesRegex(ValueError, "Missing telemetry"):
            prepared(telemetry().drop(columns="SpeedKmh"))


if __name__ == "__main__":
    unittest.main()
