"""A small, chronologically held-out next-lap pace experiment.

Predictions are conditional on the next lap remaining analysis-eligible in the
same reported stint and compound. Eligibility is established retrospectively;
this experiment does not predict pit stops, incidents or eligibility itself.
Only public timing and reported metadata available through lap t are features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ("OriginLapNumber", "TyreLife", "LastLapSeconds", "TrailingDeltaSeconds")
CATEGORICAL_FEATURES = ("Driver", "Compound")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0)
KNOWN_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"})
EXAMPLE_COLUMNS = [
    "Driver", "Compound", "Stint", "OriginLapNumber", "TargetLapNumber",
    "OriginStartSeconds", "OriginEndSeconds", "TargetStartSeconds", "TargetEndSeconds",
    "TyreLife", "LastLapSeconds", "TrailingMedianSeconds", "TrailingDeltaSeconds",
    "HistoryCount", "TargetSeconds", "ResidualTargetSeconds",
]


@dataclass(frozen=True)
class TemporalSplit:
    """Assignments preserve the examples' index, including purged pairs."""
    assignments: pd.Series
    session_start_seconds: float
    session_end_seconds: float
    train_cutoff_seconds: float
    validation_cutoff_seconds: float


@dataclass
class ExperimentResult:
    examples: pd.DataFrame
    splits: TemporalSplit
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    tuning: pd.DataFrame
    pipeline: Any
    protocol: dict[str, Any]


def _true(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().eq("true").fillna(False)


def _elapsed_seconds(values: pd.Series) -> pd.Series:
    """FastF1 Time/LapStartTime are elapsed timedeltas, not datetimes."""
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def _normalized_laps(audited: pd.DataFrame) -> pd.DataFrame:
    required = {"Driver", "LapNumber", "LapTimeSeconds", "Included", "Stint", "Compound",
                "TyreLife", "Time", "LapStartTime"}
    missing = required.difference(audited.columns)
    if missing:
        raise ValueError(f"Missing required model columns: {sorted(missing)}")
    data = audited.copy()
    for name in ("LapNumber", "LapTimeSeconds", "Stint", "TyreLife"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data["Driver"] = data["Driver"].astype("string").str.strip().str.upper()
    data["Compound"] = data["Compound"].astype("string").str.strip().str.upper()
    data["StartSeconds"] = _elapsed_seconds(data["LapStartTime"])
    data["EndSeconds"] = _elapsed_seconds(data["Time"])
    valid_number = np.isfinite(data["LapNumber"]) & data["LapNumber"].gt(0) & data["LapNumber"].mod(1).eq(0)
    if data.loc[valid_number].duplicated(["Driver", "LapNumber"]).any():
        raise ValueError("Duplicate driver/lap records cannot define temporal examples.")
    data["ValidForModel"] = (
        _true(data["Included"]) & valid_number
        & data["Driver"].notna() & data["Driver"].ne("")
        & data["Compound"].isin(KNOWN_COMPOUNDS)
        & np.isfinite(data["LapTimeSeconds"]) & data["LapTimeSeconds"].gt(0)
        & np.isfinite(data["Stint"]) & data["Stint"].gt(0) & data["Stint"].mod(1).eq(0)
        & np.isfinite(data["TyreLife"]) & data["TyreLife"].ge(0)
        & np.isfinite(data["StartSeconds"]) & data["StartSeconds"].ge(0)
        & np.isfinite(data["EndSeconds"]) & data["EndSeconds"].gt(data["StartSeconds"])
    ).fillna(False)
    return data.sort_values(["Driver", "LapNumber"], na_position="last").reset_index(drop=True)


def _continues(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    return bool(
        current["LapNumber"] == previous["LapNumber"] + 1
        and current["Stint"] == previous["Stint"]
        and current["Compound"] == previous["Compound"]
        # A one-millisecond tolerance accommodates serialized timing resolution.
        and current["StartSeconds"] >= previous["EndSeconds"] - 0.001
        and current["StartSeconds"] > previous["StartSeconds"]
        and current["EndSeconds"] > previous["EndSeconds"]
    )


def build_examples(audited: pd.DataFrame) -> pd.DataFrame:
    """True t -> t+1 pairs; never connect the next retained lap across a gap.

    The rolling history contains at most three eligible, consecutive laps through
    the origin. Exclusions, missing laps and stint/compound changes reset it.
    Target metadata controls the retrospective cohort and is never a feature.
    """
    data = _normalized_laps(audited)
    examples: list[dict[str, Any]] = []
    for _, driver_laps in data.groupby("Driver", dropna=False, sort=True):
        history: list[dict[str, Any]] = []
        for current in driver_laps.to_dict("records"):
            if not current["ValidForModel"]:
                history = []
                continue
            if history and _continues(history[-1], current):
                origin = history[-1]
                recent_median = float(np.median([row["LapTimeSeconds"] for row in history]))
                last_seconds = float(origin["LapTimeSeconds"])
                target_seconds = float(current["LapTimeSeconds"])
                examples.append({
                    "Driver": str(origin["Driver"]), "Compound": str(origin["Compound"]),
                    "Stint": int(origin["Stint"]), "OriginLapNumber": int(origin["LapNumber"]),
                    "TargetLapNumber": int(current["LapNumber"]),
                    "OriginStartSeconds": float(origin["StartSeconds"]),
                    "OriginEndSeconds": float(origin["EndSeconds"]),
                    "TargetStartSeconds": float(current["StartSeconds"]),
                    "TargetEndSeconds": float(current["EndSeconds"]),
                    "TyreLife": float(origin["TyreLife"]), "LastLapSeconds": last_seconds,
                    "TrailingMedianSeconds": recent_median,
                    "TrailingDeltaSeconds": recent_median - last_seconds,
                    "HistoryCount": len(history), "TargetSeconds": target_seconds,
                    "ResidualTargetSeconds": target_seconds - last_seconds,
                })
                history = (history + [current])[-3:]
            else:
                history = [current]
    return pd.DataFrame(examples, columns=EXAMPLE_COLUMNS).sort_values(
        ["OriginEndSeconds", "Driver"], ignore_index=True
    )


def split_examples(examples: pd.DataFrame, audited: pd.DataFrame) -> TemporalSplit:
    """One shared elapsed-time 60/20/20 split over all supplied raw lap timings.

    Labels crossing either cutoff are purged. Validation/test origins must be at
    or after the preceding cutoff, so training labels are already observable.
    Excluded laps still define the race's wall-clock extent.
    """
    data = _normalized_laps(audited)
    valid_times = (
        np.isfinite(data["StartSeconds"]) & np.isfinite(data["EndSeconds"])
        & data["StartSeconds"].ge(0) & data["EndSeconds"].gt(data["StartSeconds"])
    )
    if not valid_times.any():
        raise ValueError("No valid raw race timestamps exist for chronological cutoffs.")
    start = float(data.loc[valid_times, "StartSeconds"].min())
    end = float(data.loc[valid_times, "EndSeconds"].max())
    train_cutoff = start + 0.6 * (end - start)
    validation_cutoff = start + 0.8 * (end - start)
    assignments = pd.Series("purged", index=examples.index, name="Split", dtype="string")
    if not examples.empty:
        required = {"OriginEndSeconds", "TargetEndSeconds"}
        if not required.issubset(examples):
            raise ValueError("Examples require origin and target end timestamps.")
        origin, target = examples["OriginEndSeconds"], examples["TargetEndSeconds"]
        valid = np.isfinite(origin) & np.isfinite(target) & target.gt(origin)
        assignments.loc[valid & target.le(train_cutoff)] = "train"
        assignments.loc[valid & origin.ge(train_cutoff) & target.le(validation_cutoff)] = "validation"
        assignments.loc[valid & origin.ge(validation_cutoff) & target.le(end)] = "test"
    return TemporalSplit(assignments, start, end, train_cutoff, validation_cutoff)


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Positive signed bias means predicted lap times are slower than observed."""
    columns = ["Split", "Model", "Driver", "Count", "MAESeconds", "RMSESeconds", "BiasSeconds"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    model_columns = {"Ridge": "RidgeSeconds", "Persistence": "PersistenceSeconds",
                     "Trailing median": "TrailingMedianSeconds"}
    rows: list[dict[str, Any]] = []
    for split, subset in predictions.groupby("Split", sort=False):
        cohorts = [("ALL", subset)] + [(str(driver), laps) for driver, laps in subset.groupby("Driver", sort=True)]
        for driver, cohort in cohorts:
            for model, prediction_column in model_columns.items():
                error = pd.to_numeric(cohort[prediction_column], errors="coerce") - pd.to_numeric(cohort["ActualSeconds"], errors="coerce")
                if not np.isfinite(error).all():
                    raise ValueError("Predictions and actual lap times must be finite for every scored example.")
                rows.append({"Split": str(split), "Model": model, "Driver": driver,
                             "Count": int(len(error)), "MAESeconds": float(np.mean(np.abs(error))),
                             "RMSESeconds": float(np.sqrt(np.mean(np.square(error)))),
                             "BiasSeconds": float(np.mean(error))})
    return pd.DataFrame(rows, columns=columns)


def _pipeline(alpha: float) -> Any:
    # Imports stay lazy: preparing examples and inspecting protocol needs no sklearn.
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([("preprocessing", preprocessing), ("ridge", Ridge(alpha=alpha))])


def run_experiment(audited: pd.DataFrame, alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> ExperimentResult:
    """Tune on validation MAE, freeze the train fit, evaluate test once.

    Each alpha uses its own preprocessing fitted solely on training examples.
    Deterministic ties choose the lowest alpha. No train+validation refit follows.
    """
    alpha_values = sorted(set(float(alpha) for alpha in alphas))
    if not alpha_values or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alpha_values):
        raise ValueError("Ridge alphas must be a non-empty set of finite positive values.")
    examples = build_examples(audited)
    splits = split_examples(examples, audited)
    cohorts = {name: examples.loc[splits.assignments.eq(name)] for name in ("train", "validation", "test")}
    if any(cohorts[name].empty for name in cohorts):
        counts = {name: len(frame) for name, frame in cohorts.items()}
        raise ValueError(f"Need examples in every chronological split; observed {counts}.")
    train, validation = cohorts["train"], cohorts["validation"]
    candidates: list[tuple[float, float, Any]] = []
    for alpha in alpha_values:
        pipeline = _pipeline(alpha)
        pipeline.fit(train.loc[:, MODEL_FEATURES], train["ResidualTargetSeconds"])
        predicted = validation["LastLapSeconds"].to_numpy() + pipeline.predict(validation.loc[:, MODEL_FEATURES])
        mae = float(np.mean(np.abs(predicted - validation["TargetSeconds"].to_numpy())))
        candidates.append((mae, alpha, pipeline))
    _, selected_alpha, selected_pipeline = min(candidates, key=lambda entry: (entry[0], entry[1]))
    tuning = pd.DataFrame([
        {"Alpha": alpha, "ValidationMAESeconds": mae, "Selected": alpha == selected_alpha}
        for mae, alpha, _ in candidates
    ]).sort_values("Alpha", ignore_index=True)
    predictions = examples.loc[splits.assignments.isin(["validation", "test"])].copy()
    predictions["Split"] = splits.assignments.loc[predictions.index].to_numpy()
    predictions["ActualSeconds"] = predictions["TargetSeconds"]
    predictions["PersistenceSeconds"] = predictions["LastLapSeconds"]
    predictions["RidgeSeconds"] = predictions["LastLapSeconds"].to_numpy() + selected_pipeline.predict(predictions.loc[:, MODEL_FEATURES])
    for model, column in [("Ridge", "RidgeSeconds"), ("Persistence", "PersistenceSeconds"), ("TrailingMedian", "TrailingMedianSeconds")]:
        predictions[f"{model}ErrorSeconds"] = predictions[column] - predictions["ActualSeconds"]
    metrics = score_predictions(predictions)
    counts = {name: int(splits.assignments.eq(name).sum()) for name in ("train", "validation", "test", "purged")}
    protocol = {
        "experiment": "Conditional next-lap pace, Ferrari Bahrain 2024 pilot",
        "target": "LapTimeSeconds of actual lap t+1, eligible and in the same reported stint and compound",
        "prediction_origin": "Nominal completion of lap t (public Time elapsed seconds); feed arrival latency is not replayed",
        "eligibility": "Established retrospectively using final archived flags for both laps; not predicted",
        "features": list(MODEL_FEATURES), "residual_target": "TargetSeconds - LastLapSeconds",
        "history": "At most three consecutive eligible laps through t; reset on exclusions, missing laps, invalid metadata, stint or compound changes",
        "split": "Global race wall-clock 60% training, 20% validation, 20% test; crossing pairs purged",
        "session_start_seconds": splits.session_start_seconds,
        "session_end_seconds": splits.session_end_seconds,
        "train_cutoff_seconds": splits.train_cutoff_seconds,
        "validation_cutoff_seconds": splits.validation_cutoff_seconds,
        "counts": counts, "alphas": alpha_values, "selected_alpha": selected_alpha,
        "selection": "Lowest validation MAE; exact ties choose lowest alpha",
        "fit": "Preprocessing and Ridge fitted only on training examples; selected training fit frozen; no validation refit",
        "baselines": ["Last lap persistence", "Trailing median of at most three laps through t"],
        "signed_error": "Prediction minus actual seconds; positive means predicted slower",
        "limitations": [
            "One event and two drivers, with small chronological evaluation cohorts",
            "No claim of generalization to other races, drivers, circuits or weather",
            "No private fuel, tyre wear, setup, brake pressure or team telemetry measurements",
            "Validation/test origins may use preceding observed laps, matching sequential one-step forecasting",
            "Target eligibility is retrospective; model does not predict pit stops, incidents or eligibility",
        ],
    }
    return ExperimentResult(examples, splits, predictions.reset_index(drop=True), metrics, tuning, selected_pipeline, protocol)

