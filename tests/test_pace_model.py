"""Adversarial checks for future leakage, real adjacency and held-out scoring."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pace_model import (
    MODEL_FEATURES, NUMERIC_FEATURES, build_examples, run_experiment,
    score_predictions, split_examples,
)


def lap(number, seconds=100.0, *, driver="LEC", offset=0.0, **changes):
    row = {
        "Driver": driver, "LapNumber": number, "LapTimeSeconds": seconds,
        "Included": True, "Compound": "HARD", "Stint": 1, "TyreLife": number,
        "LapStartTime": pd.to_timedelta(float(offset + (number - 1) * 100), unit="s"),
        "Time": pd.to_timedelta(float(offset + number * 100), unit="s"),
    }
    row.update(changes)
    return row


def fixture():
    return pd.DataFrame([
        lap(number, 100 + 0.04 * number + 0.15 * np.sin(number), driver=driver, offset=offset)
        for driver, offset in [("LEC", 0), ("SAI", 10)]
        for number in range(1, 21)
    ])


class PaceModelTests(unittest.TestCase):
    def test_never_jumps_exclusions_missing_laps_or_false_csv_strings(self):
        data = pd.DataFrame([lap(1), lap(2), lap(3, Included="False"),
                             lap(4), lap(5), lap(7), lap(8)])
        examples = build_examples(data)
        self.assertEqual(examples["OriginLapNumber"].tolist(), [1, 4, 7])
        self.assertEqual(examples["TargetLapNumber"].tolist(), [2, 5, 8])
        self.assertEqual(examples["HistoryCount"].tolist(), [1, 1, 1])

    def test_history_uses_at_most_three_past_laps_and_resets_on_stint(self):
        data = pd.DataFrame([lap(1, 90), lap(2, 100), lap(3, 110), lap(4, 120),
                             lap(5, 500, Stint=2), lap(6, 600, Stint=2)])
        examples = build_examples(data).set_index("OriginLapNumber")
        self.assertEqual(examples.loc[3, "TrailingMedianSeconds"], 100)
        self.assertEqual(examples.loc[3, "HistoryCount"], 3)
        self.assertEqual(examples.loc[5, "TrailingMedianSeconds"], 500)
        self.assertEqual(examples.loc[5, "HistoryCount"], 1)
        self.assertNotIn(4, examples.index)

    def test_invalid_metadata_or_unknown_compound_resets_history(self):
        for bad in [{"Stint": None}, {"Stint": 1.5}, {"Compound": "UNKNOWN"},
                    {"Compound": None}, {"TyreLife": np.nan}, {"LapTimeSeconds": np.inf},
                    {"Included": None}]:
            with self.subTest(bad=bad):
                data = pd.DataFrame([lap(1), lap(2, **bad), lap(3), lap(4)])
                examples = build_examples(data)
                self.assertEqual(examples["OriginLapNumber"].tolist(), [3])
                self.assertEqual(examples["HistoryCount"].tolist(), [1])

    def test_target_and_future_values_never_enter_origin_features(self):
        data = pd.DataFrame([lap(1, 90), lap(2, 100), lap(3, 110), lap(4, 120)])
        first = build_examples(data).query("OriginLapNumber == 2")
        altered = data.copy()
        altered.loc[altered["LapNumber"].eq(3), ["LapTimeSeconds", "TyreLife"]] = [500, 999]
        altered.loc[altered["LapNumber"].eq(4), "LapTimeSeconds"] = 1000
        second = build_examples(altered).query("OriginLapNumber == 2")
        assert_frame_equal(first.loc[:, MODEL_FEATURES], second.loc[:, MODEL_FEATURES])
        self.assertEqual(second.iloc[0]["TargetSeconds"], 500)
        self.assertEqual(first.iloc[0]["TyreLife"], 2)
        self.assertTrue(first.iloc[0]["OriginEndSeconds"] <= first.iloc[0]["TargetStartSeconds"])

    def test_compound_changes_reset_even_when_stint_does_not_change(self):
        data = pd.DataFrame([lap(1, Compound=" soft "), lap(2, Compound="SOFT"),
                             lap(3, Compound="HARD"), lap(4, Compound="HARD")])
        examples = build_examples(data)
        self.assertEqual(examples["OriginLapNumber"].tolist(), [1, 3])
        self.assertEqual(examples["Compound"].tolist(), ["SOFT", "HARD"])
        self.assertEqual(examples["HistoryCount"].tolist(), [1, 1])

    def test_nonmonotonic_completion_times_cannot_form_pair(self):
        data = pd.DataFrame([lap(1), lap(2, LapStartTime=pd.to_timedelta(50.0, unit="s")),
                             lap(3), lap(4)])
        examples = build_examples(data)
        self.assertNotIn(1, examples["OriginLapNumber"].tolist())
        row = examples.query("OriginLapNumber == 2").iloc[0]
        self.assertEqual(row["HistoryCount"], 1)

    def test_duplicate_driver_laps_raise_instead_of_selecting_arbitrarily(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_examples(pd.DataFrame([lap(1), lap(1)]))

    def test_global_boundaries_include_excluded_laps_and_purge_crossing_pairs(self):
        raw = pd.DataFrame([lap(1, Included=False), lap(10, Included=False)])
        examples = pd.DataFrame({
            "OriginEndSeconds": [450, 550, 600, 750, 800],
            "TargetEndSeconds": [550, 650, 650, 850, 850],
        }, index=[9, 3, 7, 2, 5])
        split = split_examples(examples, raw)
        self.assertEqual(split.session_start_seconds, 0)
        self.assertEqual(split.session_end_seconds, 1000)
        self.assertEqual(split.train_cutoff_seconds, 600)
        self.assertEqual(split.validation_cutoff_seconds, 800)
        self.assertEqual(split.assignments.index.tolist(), [9, 3, 7, 2, 5])
        self.assertEqual(split.assignments.tolist(), ["train", "purged", "validation", "purged", "test"])

    def test_both_drivers_share_cutoffs_and_fit_observed_before_validation(self):
        raw = fixture()
        examples = build_examples(raw)
        split = split_examples(examples, raw)
        train = examples.loc[split.assignments.eq("train")]
        validation = examples.loc[split.assignments.eq("validation")]
        test = examples.loc[split.assignments.eq("test")]
        self.assertEqual(set(train["Driver"]), {"LEC", "SAI"})
        self.assertEqual(set(test["Driver"]), {"LEC", "SAI"})
        self.assertTrue(train["TargetEndSeconds"].le(split.train_cutoff_seconds).all())
        self.assertTrue(validation["OriginEndSeconds"].ge(split.train_cutoff_seconds).all())
        self.assertTrue(validation["TargetEndSeconds"].le(split.validation_cutoff_seconds).all())
        self.assertTrue(test["OriginEndSeconds"].ge(split.validation_cutoff_seconds).all())

    def test_preprocessing_fitted_only_on_training_rows(self):
        result = run_experiment(fixture())
        train = result.examples.loc[result.splits.assignments.eq("train")]
        scaler = result.pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
        np.testing.assert_allclose(scaler.mean_, train.loc[:, NUMERIC_FEATURES].mean().to_numpy())
        self.assertEqual(scaler.n_samples_seen_, len(train))
        self.assertEqual(len(result.predictions), len(result.examples) - len(train) - result.protocol["counts"]["purged"])

    def test_test_label_changes_do_not_select_alpha_or_change_predictions(self):
        raw = fixture()
        first = run_experiment(raw)
        altered = raw.copy()
        # Final labels have no later origin features; timestamp cutoffs stay fixed.
        altered.loc[altered["LapNumber"].eq(20), "LapTimeSeconds"] += 50
        second = run_experiment(altered)
        self.assertEqual(first.protocol["selected_alpha"], second.protocol["selected_alpha"])
        assert_frame_equal(first.tuning, second.tuning)
        np.testing.assert_allclose(first.predictions["RidgeSeconds"], second.predictions["RidgeSeconds"])
        self.assertFalse(np.allclose(first.predictions["ActualSeconds"], second.predictions["ActualSeconds"]))

    def test_residual_pipeline_roundtrip_and_prediction_error_sign(self):
        result = run_experiment(fixture())
        expected = result.predictions["LastLapSeconds"].to_numpy() + result.pipeline.predict(result.predictions.loc[:, MODEL_FEATURES])
        np.testing.assert_allclose(expected, result.predictions["RidgeSeconds"])
        np.testing.assert_allclose(result.predictions["RidgeErrorSeconds"], result.predictions["RidgeSeconds"] - result.predictions["ActualSeconds"])
        self.assertEqual(int(result.tuning["Selected"].sum()), 1)
        selected = result.tuning.loc[result.tuning["Selected"]].iloc[0]
        self.assertEqual(selected["ValidationMAESeconds"], result.tuning["ValidationMAESeconds"].min())

    def test_metrics_include_losing_model_and_signed_bias_without_trimming(self):
        predictions = pd.DataFrame({
            "Split": ["test", "test"], "Driver": ["LEC", "SAI"],
            "ActualSeconds": [100, 100], "RidgeSeconds": [98, 104],
            "PersistenceSeconds": [100, 100], "TrailingMedianSeconds": [101, 99],
        })
        metrics = score_predictions(predictions)
        row = metrics.query("Driver == 'ALL' and Model == 'Ridge'").iloc[0]
        self.assertEqual(row["Count"], 2)
        self.assertEqual(row["MAESeconds"], 3)
        self.assertAlmostEqual(row["RMSESeconds"], np.sqrt(10))
        self.assertEqual(row["BiasSeconds"], 1)
        self.assertEqual(metrics.query("Driver == 'ALL' and Model == 'Persistence'").iloc[0]["MAESeconds"], 0)
        self.assertEqual(len(metrics), 9)

    def test_empty_or_invalid_split_and_invalid_alphas_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "every chronological split"):
            run_experiment(pd.DataFrame([lap(1), lap(2)]))
        with self.assertRaisesRegex(ValueError, "valid raw race timestamps"):
            split_examples(pd.DataFrame(), pd.DataFrame([lap(1, Time=None)]))
        for alphas in [(), (0,), (-1,), (np.nan,)]:
            with self.subTest(alphas=alphas), self.assertRaisesRegex(ValueError, "alphas"):
                run_experiment(fixture(), alphas=alphas)


if __name__ == "__main__":
    unittest.main()

