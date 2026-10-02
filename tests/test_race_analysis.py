"""Small adversarial checks for temporal matching and summary denominators."""
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from race_analysis import matched_laps, matched_summary, segment_summary, stint_summary


def lap(driver, number, seconds=90.0, *, included=True, compound="SOFT", stint=1, age=5):
    return {
        "Driver": driver, "LapNumber": number, "LapTimeSeconds": seconds,
        "Included": included, "Compound": compound, "Stint": stint, "TyreLife": age,
    }


def frame(*rows):
    return pd.DataFrame(rows, columns=[
        "Driver", "LapNumber", "LapTimeSeconds", "Included", "Compound", "Stint", "TyreLife",
    ])


class RaceAnalysisTests(unittest.TestCase):
    def test_disjoint_windows_never_match_equal_tyre_age(self):
        data = frame(lap("LEC", 2, age=5), lap("LEC", 3, age=6),
                     lap("SAI", 20, age=5), lap("SAI", 21, age=6))
        self.assertTrue(matched_laps(data).empty)
        self.assertTrue(segment_summary(matched_laps(data)).empty)

    def test_exclusions_apply_to_both_sides_including_csv_false(self):
        data = frame(lap("LEC", 2), lap("LEC", 3, included="False"),
                     lap("LEC", 4), lap("SAI", 2), lap("SAI", 3),
                     lap("SAI", 4, included=False))
        self.assertEqual(matched_laps(data)["LapNumber"].tolist(), [2])

    def test_compound_mismatch_can_only_be_explicitly_requested(self):
        data = frame(lap("LEC", 2, compound="SOFT"), lap("SAI", 2, compound="HARD"))
        self.assertTrue(matched_laps(data).empty)
        pairs = matched_laps(data, same_compound=False)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs.iloc[0]["SameCompound"])
        self.assertEqual(pairs.iloc[0]["ReferenceCompound"], "SOFT")
        self.assertEqual(pairs.iloc[0]["ComparisonCompound"], "HARD")

    def test_unknown_compounds_never_satisfy_same_compound(self):
        for unknown in [None, pd.NA, "", "UNKNOWN", "unknown", "N/A", "UNRECOGNIZED"]:
            with self.subTest(compound=unknown):
                data = frame(lap("LEC", 2, compound=unknown), lap("SAI", 2, compound=unknown))
                self.assertTrue(matched_laps(data).empty)
                self.assertFalse(matched_laps(data, same_compound=False).iloc[0]["SameCompound"])

    def test_compound_case_and_whitespace_are_normalized(self):
        data = frame(lap("LEC", 2, compound=" soft "), lap("SAI", 2, compound="SOFT"))
        self.assertEqual(len(matched_laps(data)), 1)

    def test_delta_direction_and_driver_swap(self):
        data = frame(lap("LEC", 2, 91), lap("SAI", 2, 90.5))
        self.assertEqual(matched_laps(data).iloc[0]["DeltaSeconds"], -0.5)
        self.assertEqual(matched_laps(data, reference="SAI", comparison="LEC").iloc[0]["DeltaSeconds"], 0.5)

    def test_single_lap_segment_is_retained_with_count_one(self):
        data = frame(lap("LEC", 20, 90), lap("SAI", 20, 91))
        segments = segment_summary(matched_laps(data))
        row = segments.iloc[0]
        self.assertEqual(row["LapStart"], 20)
        self.assertEqual(row["LapEnd"], 20)
        self.assertEqual(row["MatchedLaps"], 1)
        self.assertEqual(row["MedianDeltaSeconds"], 1)
        self.assertEqual(row["IQRDeltaSeconds"], 0)

    def test_gaps_and_stint_changes_split_windows(self):
        data = frame(
            lap("LEC", 2), lap("SAI", 2),
            lap("LEC", 3), lap("SAI", 3),
            lap("LEC", 5), lap("SAI", 5),
            lap("LEC", 6, stint=2), lap("SAI", 6),
        )
        pairs = matched_laps(data)
        segments = segment_summary(pairs)
        self.assertEqual(segments["MatchedLaps"].tolist(), [2, 1, 1])
        self.assertEqual(segments["LapStart"].tolist(), [2, 5, 6])
        self.assertEqual(int(segments["MatchedLaps"].sum()), len(pairs))

    def test_compound_changes_split_even_when_both_are_same(self):
        data = frame(
            lap("LEC", 2), lap("SAI", 2),
            lap("LEC", 3, compound="HARD"), lap("SAI", 3, compound="HARD"),
        )
        self.assertEqual(segment_summary(matched_laps(data))["MatchedLaps"].tolist(), [1, 1])

    def test_segment_median_is_median_of_paired_differences(self):
        data = frame(
            lap("LEC", 2, 90), lap("SAI", 2, 91),
            lap("LEC", 3, 100), lap("SAI", 3, 110),
            lap("LEC", 4, 110), lap("SAI", 4, 101),
        )
        row = segment_summary(matched_laps(data)).iloc[0]
        self.assertEqual(row["MedianDeltaSeconds"], 1)
        self.assertEqual(row["ComparisonMedianLapTimeSeconds"] - row["ReferenceMedianLapTimeSeconds"], 1)
        # Change the middle reference so separate medians diverge from paired median.
        data.loc[(data["Driver"] == "LEC") & (data["LapNumber"] == 3), "LapTimeSeconds"] = 99
        row = segment_summary(matched_laps(data)).iloc[0]
        self.assertEqual(row["MedianDeltaSeconds"], 1)
        self.assertEqual(row["ComparisonMedianLapTimeSeconds"] - row["ReferenceMedianLapTimeSeconds"], 2)

    def test_stint_denominator_includes_excluded_and_missing_times(self):
        data = frame(
            lap("LEC", 2, 90), lap("LEC", 3, 92),
            lap("LEC", 4, float("nan"), included=False),
            lap("LEC", 5, 200, included=False),
            lap("SAI", 2, 95, included=False),
        )
        summary = stint_summary(data).set_index("Driver")
        row = summary.loc["LEC"]
        self.assertEqual(row["RecordedLaps"], 4)
        self.assertEqual(row["EligibleLaps"], 2)
        self.assertEqual(row["ExcludedLaps"], 2)
        self.assertEqual(row["EligibilityRate"], 0.5)
        self.assertEqual(row["MedianLapTimeSeconds"], 91)
        self.assertEqual(row["IQRSeconds"], 1)
        self.assertEqual(row["LapEnd"], 5)
        self.assertEqual(row["EligibleLapEnd"], 3)
        self.assertTrue(pd.isna(summary.loc["SAI", "MedianLapTimeSeconds"]))

    def test_reported_tyre_usage_is_preserved_without_age_matching(self):
        data = frame(lap("LEC", 2, age=12, stint=1), lap("SAI", 2, age=4, stint=2))
        pairs = matched_laps(data)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.iloc[0]["ReferenceTyreLife"], 12)
        self.assertEqual(pairs.iloc[0]["ComparisonTyreLife"], 4)
        self.assertEqual(pairs.iloc[0]["ComparisonStint"], 2)

    def test_duplicate_driver_laps_fail_instead_of_multiplying_rows(self):
        data = frame(lap("LEC", 2), lap("LEC", 2), lap("SAI", 2))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            matched_laps(data)

    def test_invalid_eligible_records_fail(self):
        for number, seconds in [(2, 0), (2, float("inf")), (2.5, 90), (2, float("nan"))]:
            with self.subTest(number=number, seconds=seconds):
                with self.assertRaisesRegex(ValueError, "Eligible"):
                    matched_laps(frame(lap("LEC", number, seconds), lap("SAI", 2)))

    def test_inputs_are_not_modified(self):
        data = frame(lap("LEC", 2, compound=" soft "), lap("SAI", 2))
        original = data.copy(deep=True)
        stint_summary(data)
        pairs = matched_laps(data)
        pairs_original = pairs.copy(deep=True)
        segment_summary(pairs)
        assert_frame_equal(data, original)
        assert_frame_equal(pairs, pairs_original)

    def test_empty_outputs_have_stable_columns_and_alias(self):
        empty = frame()
        self.assertTrue(stint_summary(empty).empty)
        self.assertIn("RecordedLaps", stint_summary(empty).columns)
        pairs = matched_laps(empty)
        self.assertIn("DeltaSeconds", pairs.columns)
        self.assertIn("MatchedLaps", segment_summary(pairs).columns)
        assert_frame_equal(matched_summary(pairs), segment_summary(pairs))

    def test_same_driver_comparison_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different"):
            matched_laps(frame(lap("LEC", 2)), reference="LEC", comparison="LEC")


if __name__ == "__main__":
    unittest.main()
