"""Adversarial tests for fixed event holdouts, cohort audit and future leakage."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from analyze_data import RULES, audit_laps
from pace_model import MODEL_FEATURES, NUMERIC_FEATURES
from weekend_model import (
    DEFAULT_DRIVERS, EVENT_ROLES, audit_single_event, build_weekend_examples,
    run_weekend_experiment, score_weekend_predictions, validate_event_metadata,
)


EVENTS = {
    "2024-01": ("Bahrain Grand Prix", "2024-03-02"),
    "2024-03": ("Australian Grand Prix", "2024-03-24"),
    "2024-04": ("Japanese Grand Prix", "2024-04-07"),
    "2024-05": ("Chinese Grand Prix", "2024-04-21"),
    "2024-06": ("Miami Grand Prix", "2024-05-05"),
    "2024-07": ("Emilia Romagna Grand Prix", "2024-05-19"),
}


def model_fixture(*, constant=False):
    rows = []
    for key, role in EVENT_ROLES.items():
        event_name, date = EVENTS[key]
        round_number = int(key[-2:])
        for driver_number, driver in enumerate(DEFAULT_DRIVERS):
            offset = float(driver_number * 10)
            for number in range(1, 13):
                seconds = 100.0 if constant else 100 + 0.07 * number + 0.12 * np.sin(number + round_number) + driver_number * 0.02
                rows.append({
                    "EventKey": key, "EventName": event_name, "EventDate": date,
                    "Year": 2024, "RoundNumber": round_number, "Role": role,
                    "Driver": driver, "LapNumber": number,
                    "LapTimeSeconds": seconds, "Included": True,
                    "Compound": "HARD", "Stint": 1, "TyreLife": number,
                    "LapStartTime": pd.to_timedelta(float(offset + (number - 1) * 100), unit="s"),
                    "Time": pd.to_timedelta(float(offset + number * 100), unit="s"),
                })
    return pd.DataFrame(rows)


def raw_lap(number=2, *, driver="LEC", **changes):
    result = {
        "Driver": driver, "LapNumber": number,
        "LapTime": "0 days 00:01:40", "Stint": 1, "Compound": "HARD",
        "TyreLife": number, "PitInTime": None, "PitOutTime": None,
        "TrackStatus": "1", "IsAccurate": "True", "Deleted": "False",
        "FastF1Generated": "False",
    }
    result.update(changes)
    return result


class WeekendModelTests(unittest.TestCase):
    def test_audit_has_exact_original_rules_and_preserves_records(self):
        raw = pd.DataFrame([
            raw_lap(1), raw_lap(2), raw_lap(3, TrackStatus="12", Deleted="True"),
            raw_lap(4, PitInTime="0 days 00:05:00", IsAccurate=None),
            raw_lap(5, LapTime=None, FastF1Generated="True"),
            raw_lap(6, TrackStatus="1.0", Deleted=None),
            raw_lap(7, LapTime="0 days 00:00:00", PitOutTime="0 days 00:05:00"),
            raw_lap(2, driver="SAI"),
        ])
        original, original_checks = audit_laps(raw)
        generalized, checks = audit_single_event(raw, ("LEC", "SAI"))
        assert_frame_equal(original, generalized, check_dtype=False)
        assert_frame_equal(original_checks, checks)
        self.assertEqual(tuple(checks.columns), tuple(RULES))
        self.assertEqual(len(generalized), len(raw))
        self.assertEqual(generalized.loc[generalized["Included"], "ExclusionReasons"].tolist(), ["", ""])

    def test_audit_arbitrary_cohort_missing_duplicate_and_mixed_events(self):
        raw = pd.DataFrame([raw_lap(driver="VER"), raw_lap(driver="HAM")])
        audited, _ = audit_single_event(raw, ("VER", "HAM"))
        self.assertTrue(audited["Included"].all())
        with self.assertRaisesRegex(ValueError, "requested drivers"):
            audit_single_event(raw, ("VER", "HAM", "LEC"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_single_event(pd.concat([raw, raw.iloc[[0]]]), ("VER", "HAM"))
        with self.assertRaisesRegex(ValueError, "separately"):
            audit_single_event(raw.assign(EventKey=["2024-01", "2024-03"]), ("VER", "HAM"))
        for drivers in [(), ("",), ("LEC", "LEC")]:
            with self.subTest(drivers=drivers), self.assertRaisesRegex(ValueError, "Allowed drivers"):
                audit_single_event(raw, drivers)

    def test_csv_boolean_unknowns_and_mixed_track_status_are_excluded(self):
        changes = [
            {"IsAccurate": "False"}, {"IsAccurate": "unknown"},
            {"Deleted": None}, {"FastF1Generated": None},
            {"TrackStatus": "12"}, {"TrackStatus": ""},
            {"LapNumber": 2.5}, {"LapTime": "-1 days 23:59:59"},
        ]
        for change in changes:
            with self.subTest(change=change):
                audited, _ = audit_single_event(pd.DataFrame([raw_lap(**change)]), ("LEC",))
                self.assertFalse(audited.iloc[0]["Included"])
                self.assertTrue(audited.iloc[0]["ExclusionReasons"])

    def test_event_grouping_resets_history_and_session_elapsed_time(self):
        examples = build_weekend_examples(model_fixture())
        self.assertEqual(set(examples["EventKey"]), set(EVENT_ROLES))
        self.assertEqual(len(examples), 6 * 6 * 11)
        for (key, driver), laps in examples.groupby(["EventKey", "Driver"]):
            with self.subTest(event=key, driver=driver):
                self.assertEqual(laps["OriginLapNumber"].tolist(), list(range(1, 12)))
                self.assertEqual(laps["TargetLapNumber"].tolist(), list(range(2, 13)))
                self.assertEqual(laps.iloc[0]["HistoryCount"], 1)
                self.assertEqual(laps.iloc[0]["Split"], EVENT_ROLES[key])
                self.assertEqual(laps.iloc[-1]["HistoryCount"], 3)

    def test_no_cross_event_continuation_when_lap_numbers_could_join(self):
        source = model_fixture()
        source = source.loc[~(source["EventKey"].eq("2024-01") & source["LapNumber"].gt(6))].copy()
        source = source.loc[~(source["EventKey"].eq("2024-03") & source["LapNumber"].lt(7))].copy()
        examples = build_weekend_examples(source)
        self.assertFalse((examples["OriginLapNumber"].eq(6) & examples["EventKey"].eq("2024-01")).any())
        australia = examples.loc[examples["EventKey"].eq("2024-03")]
        self.assertTrue(australia.loc[australia["OriginLapNumber"].eq(7), "HistoryCount"].eq(1).all())

    def test_missing_or_replaced_weekend_does_not_silently_adapt_protocol(self):
        source = model_fixture()
        with self.assertRaisesRegex(ValueError, "Fixed six-event"):
            run_weekend_experiment(source.loc[~source["EventKey"].eq("2024-07")])
        source.loc[source["EventKey"].eq("2024-07"), "EventKey"] = "2024-08"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            build_weekend_examples(source)

    def test_event_identity_fold_date_and_round_inconsistency_fail(self):
        source = model_fixture()
        mutations = [
            (0, "Role", "test", "one identity"),
            (0, "EventName", "different name", "one identity"),
            (None, "EventDate", "not a date", "valid EventDate"),
        ]
        for index, column, value, message in mutations:
            with self.subTest(column=column, value=value):
                altered = source.copy()
                if index is None:
                    altered[column] = value
                else:
                    altered.loc[index, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_event_metadata(altered)
        for column, value, message in [("Role", "test", "Fixed role"),
                                       ("Year", 2025, "Year/round"),
                                       ("RoundNumber", 2, "Year/round"),
                                       ("EventDate", "2023-03-02", "year")]:
            altered = source.copy()
            altered.loc[altered["EventKey"].eq("2024-01"), column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                validate_event_metadata(altered)

    def test_chronology_uses_event_dates_and_rounds_not_session_timestamps(self):
        source = model_fixture()
        normalized = validate_event_metadata(source)
        self.assertEqual(normalized["EventDate"].nunique(), 6)
        source.loc[source["EventKey"].eq("2024-03"), "EventDate"] = "2024-03-01"
        with self.assertRaisesRegex(ValueError, "increase"):
            validate_event_metadata(source)

    def test_missing_driver_and_duplicate_lap_within_event_raise(self):
        source = model_fixture()
        partial = source.loc[~(source["EventKey"].eq("2024-07") & source["Driver"].eq("PER"))]
        with self.assertRaisesRegex(ValueError, "six-driver"):
            build_weekend_examples(partial)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_weekend_examples(pd.concat([source, source.iloc[[0]]]))

    def test_unknown_compound_or_excluded_lap_resets_only_its_event_history(self):
        source = model_fixture()
        mask = source["EventKey"].eq("2024-06") & source["Driver"].eq("LEC")
        source.loc[mask & source["LapNumber"].eq(4), "Compound"] = "UNKNOWN"
        source.loc[mask & source["LapNumber"].eq(8), "Included"] = False
        examples = build_weekend_examples(source)
        changed = examples.loc[examples["EventKey"].eq("2024-06") & examples["Driver"].eq("LEC")]
        self.assertNotIn(3, changed["OriginLapNumber"].tolist())
        self.assertNotIn(4, changed["OriginLapNumber"].tolist())
        self.assertNotIn(7, changed["OriginLapNumber"].tolist())
        self.assertNotIn(8, changed["OriginLapNumber"].tolist())
        self.assertEqual(changed.loc[changed["OriginLapNumber"].eq(5), "HistoryCount"].iloc[0], 1)
        self.assertEqual(changed.loc[changed["OriginLapNumber"].eq(9), "HistoryCount"].iloc[0], 1)

    def test_features_and_preprocessing_use_only_training_examples(self):
        source = model_fixture()
        # MEDIUM is a real category but deliberately absent from training here.
        source.loc[source["Role"].ne("train"), "Compound"] = "MEDIUM"
        result = run_weekend_experiment(source)
        train = result.examples.loc[result.examples["Split"].eq("train")]
        preprocess = result.pipeline.named_steps["preprocessing"]
        scaler = preprocess.named_transformers_["numeric"]
        encoder = preprocess.named_transformers_["categorical"]
        self.assertEqual(scaler.n_samples_seen_, len(train))
        np.testing.assert_allclose(scaler.mean_, train.loc[:, NUMERIC_FEATURES].mean().to_numpy())
        self.assertEqual(preprocess.feature_names_in_.tolist(), list(MODEL_FEATURES))
        self.assertEqual(encoder.categories_[1].tolist(), ["HARD"])
        self.assertEqual(encoder.handle_unknown, "ignore")
        self.assertTrue(np.isfinite(result.predictions["RidgeSeconds"]).all())
        self.assertEqual(set(result.predictions["Split"]), {"validation", "test"})
        support = result.protocol["category_support_by_split"]["test"]["Compound"]
        self.assertEqual(support["unknown_categories"], ["MEDIUM"])
        self.assertEqual(support["unknown_example_count"], 132)

    def test_target_changes_never_enter_origin_features_and_test_labels_do_not_tune(self):
        source = model_fixture()
        first = run_weekend_experiment(source)
        altered = source.copy()
        final_test = altered["Role"].eq("test") & altered["LapNumber"].eq(12)
        altered.loc[final_test, "LapTimeSeconds"] += 50
        altered.loc[final_test, "TyreLife"] += 500
        second = run_weekend_experiment(altered)
        self.assertEqual(first.protocol["selected_alpha"], second.protocol["selected_alpha"])
        assert_frame_equal(first.tuning, second.tuning)
        assert_frame_equal(first.predictions.loc[:, MODEL_FEATURES], second.predictions.loc[:, MODEL_FEATURES])
        np.testing.assert_allclose(first.predictions["RidgeSeconds"], second.predictions["RidgeSeconds"])
        self.assertFalse(np.allclose(first.predictions["ActualSeconds"], second.predictions["ActualSeconds"]))

    def test_validation_ties_choose_lowest_fixed_alpha_without_refitting(self):
        result = run_weekend_experiment(model_fixture(constant=True))
        self.assertEqual(result.protocol["selected_alpha"], 0.1)
        self.assertEqual(result.tuning["Alpha"].tolist(), [0.1, 1.0, 10.0, 100.0])
        self.assertTrue(result.tuning["ValidationMAESeconds"].eq(0).all())
        self.assertEqual(result.protocol["counts"], {"train": 198, "validation": 66, "test": 132})
        self.assertEqual(int(result.tuning["Selected"].sum()), 1)
        self.assertEqual(result.pipeline.named_steps["preprocessing"].named_transformers_["numeric"].n_samples_seen_, 198)

    def test_metrics_retain_worse_driver_event_and_correct_denominators(self):
        predictions = pd.DataFrame({
            "Split": ["test"] * 3, "EventKey": ["2024-06", "2024-06", "2024-07"],
            "EventName": ["Miami", "Miami", "Imola"], "Driver": ["LEC", "SAI", "LEC"],
            "ActualSeconds": [100, 100, 100], "RidgeSeconds": [98, 104, 110],
            "PersistenceSeconds": [100, 100, 100], "TrailingMedianSeconds": [101, 99, 100],
        })
        metrics = score_weekend_predictions(predictions)
        pooled = metrics.query("EventKey == 'ALL' and Driver == 'ALL' and Model == 'Ridge'").iloc[0]
        self.assertEqual(pooled["Count"], 3)
        self.assertAlmostEqual(pooled["MAESeconds"], 16 / 3)
        self.assertEqual(pooled["BiasSeconds"], 4)
        event = metrics.query("EventKey == '2024-06' and Driver == 'ALL' and Model == 'Ridge'").iloc[0]
        self.assertEqual(event["Count"], 2)
        self.assertEqual(event["MAESeconds"], 3)
        self.assertAlmostEqual(event["RMSESeconds"], np.sqrt(10))
        self.assertEqual(event["BiasSeconds"], 1)
        self.assertEqual(metrics.query("EventKey == '2024-07' and Driver == 'LEC' and Model == 'Ridge'").iloc[0]["MAESeconds"], 10)
        self.assertTrue(metrics.loc[metrics["Model"].eq("Persistence"), "MAESeconds"].eq(0).all())
        self.assertEqual(len(metrics), 24)
        broken = predictions.copy()
        broken["RidgeSeconds"] = broken["RidgeSeconds"].astype(float)
        broken.loc[0, "RidgeSeconds"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            score_weekend_predictions(broken)

    def test_empty_event_examples_cannot_be_replaced_by_other_test_event(self):
        source = model_fixture()
        source.loc[source["EventKey"].eq("2024-07"), "Included"] = False
        with self.assertRaisesRegex(ValueError, "Every declared event"):
            run_weekend_experiment(source)


if __name__ == "__main__":
    unittest.main()
