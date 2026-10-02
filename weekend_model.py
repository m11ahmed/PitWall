"""Fixed, whole-weekend evaluation of the public-timing next-lap model.

The Bahrain pilot stays unchanged. This experiment uses a separate, declared
six-event cohort and never fits preprocessing, chooses alpha, or refits Ridge on
validation/test laps. Eligibility remains retrospective and conditional.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analyze_data import RULES, parse_bool
from pace_model import (
    CATEGORICAL_FEATURES, DEFAULT_ALPHAS, EXAMPLE_COLUMNS, MODEL_FEATURES,
    _pipeline, build_examples,
)

DEFAULT_DRIVERS = ("LEC", "SAI", "HAM", "RUS", "VER", "PER")
EVENT_ROLES = {
    "2024-01": "train", "2024-03": "train", "2024-04": "train",
    "2024-05": "validation", "2024-06": "test", "2024-07": "test",
}
EVENT_METADATA = ("EventKey", "EventName", "Year", "RoundNumber", "EventDate", "Role")
METRIC_COLUMNS = [
    "Split", "EventKey", "EventName", "Model", "Driver", "Count",
    "MAESeconds", "RMSESeconds", "BiasSeconds",
]


@dataclass
class WeekendExperimentResult:
    examples: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    tuning: pd.DataFrame
    pipeline: Any
    protocol: dict[str, Any]


def audit_single_event(
    raw: pd.DataFrame, allowed_drivers: Iterable[str] = DEFAULT_DRIVERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exactly the original nine audit rules to a declared driver cohort.

No raw records are removed. The requested drivers must all be present and extra
drivers are rejected, so a partial acquisition cannot silently change the cohort.
Use a separate call per event because lap numbers restart at each race.
"""
    required = {
        "Driver", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
        "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate", "Deleted",
        "FastF1Generated",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    requested = tuple(str(driver).strip().upper() for driver in allowed_drivers)
    if not requested or any(not driver for driver in requested) or len(set(requested)) != len(requested):
        raise ValueError("Allowed drivers must be non-empty, unique driver codes.")
    data = raw.copy()
    # Driver normalization concerns serialized identifiers, not audit eligibility.
    data["Driver"] = data["Driver"].astype("string").str.strip().str.upper()
    actual = set(data["Driver"].dropna())
    if data["Driver"].isna().any() or actual != set(requested):
        raise ValueError(f"Expected all and only requested drivers {sorted(requested)}; observed {sorted(actual)}.")
    if "EventKey" in data and data["EventKey"].nunique(dropna=False) != 1:
        raise ValueError("Audit each event separately; multiple EventKey values were supplied.")
    data["LapNumber"] = pd.to_numeric(data["LapNumber"], errors="coerce")
    if data.duplicated(["Driver", "LapNumber"]).any():
        raise ValueError("Duplicate driver/lap records require investigation.")
    data = data.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    seconds = pd.to_timedelta(data["LapTime"], errors="coerce").dt.total_seconds()
    data["LapTimeSeconds"] = seconds
    status = data["TrackStatus"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    lap_number = data["LapNumber"]
    checks = pd.DataFrame({
        "invalid_lap_time": ~np.isfinite(seconds) | seconds.le(0),
        "invalid_lap_number": (
            lap_number.isna() | ~np.isfinite(lap_number)
            | lap_number.le(0) | lap_number.mod(1).ne(0)
        ),
        "opening_lap": lap_number.eq(1),
        "pit_in": data["PitInTime"].astype("string").str.strip().replace("", pd.NA).notna(),
        "pit_out": data["PitOutTime"].astype("string").str.strip().replace("", pd.NA).notna(),
        "not_green_only": status.ne("1").fillna(True),
        "timing_not_accurate": parse_bool(data["IsAccurate"]).ne(True).fillna(True),
        "deleted_or_unknown": parse_bool(data["Deleted"]).ne(False).fillna(True),
        "generated_or_unknown": parse_bool(data["FastF1Generated"]).ne(False).fillna(True),
    }, index=data.index).astype(bool)
    if tuple(checks.columns) != tuple(RULES):
        raise RuntimeError("The weekend audit must preserve the original nine rules.")
    data["Included"] = ~checks.any(axis=1)
    data["ExclusionReasons"] = checks.apply(lambda row: ";".join(row.index[row]), axis=1)
    return data, checks


def validate_event_metadata(
    combined: pd.DataFrame, *, require_all_events: bool = True,
) -> pd.DataFrame:
    """Normalize metadata and reject split changes or inconsistent race identity."""
    missing = set(EVENT_METADATA).difference(combined.columns)
    if missing:
        raise ValueError(f"Missing event metadata: {sorted(missing)}")
    if combined.empty:
        raise ValueError("No event records were supplied.")
    data = combined.copy()
    for column in ("EventKey", "EventName", "Role"):
        data[column] = data[column].astype("string").str.strip()
    data["Role"] = data["Role"].str.lower()
    for column in ("Year", "RoundNumber"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    dates = pd.to_datetime(data["EventDate"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("Every event requires a valid EventDate.")
    data["EventDate"] = dates.dt.strftime("%Y-%m-%d")
    if data[list(EVENT_METADATA)].isna().any().any() or data["EventName"].eq("").any():
        raise ValueError("Event metadata cannot be missing or blank.")
    keys = set(data["EventKey"])
    unknown = keys.difference(EVENT_ROLES)
    absent = set(EVENT_ROLES).difference(keys)
    if unknown or (require_all_events and absent):
        raise ValueError(f"Fixed six-event protocol mismatch; missing {sorted(absent)}, unexpected {sorted(unknown)}.")
    metadata = data.loc[:, EVENT_METADATA].drop_duplicates()
    if metadata["EventKey"].duplicated().any():
        raise ValueError("Each event must have one identity and one role; an event cannot cross folds.")
    for event in metadata.to_dict("records"):
        key = str(event["EventKey"])
        if event["Year"] != 2024 or event["RoundNumber"] != int(key.split("-")[1]):
            raise ValueError(f"Year/round metadata does not match EventKey {key}.")
        if pd.Timestamp(event["EventDate"]).year != 2024:
            raise ValueError(f"EventDate year does not match EventKey {key}.")
        if event["Role"] != EVENT_ROLES[key]:
            raise ValueError(f"Fixed role for {key} is {EVENT_ROLES[key]}, received {event['Role']}.")
    ordered = metadata.sort_values("RoundNumber")
    ordered_dates = pd.to_datetime(ordered["EventDate"])
    if not ordered_dates.is_monotonic_increasing or ordered_dates.duplicated().any():
        raise ValueError("Event dates must increase with race rounds; chronological folds are required.")
    data["Year"] = data["Year"].astype(int)
    data["RoundNumber"] = data["RoundNumber"].astype(int)
    return data


def build_weekend_examples(combined_audited: pd.DataFrame) -> pd.DataFrame:
    """Build the unchanged pilot's examples separately for each whole race.

    Session timestamps are local to each event. History and t -> t+1 adjacency
    cannot cross an event boundary even when driver, stint, and lap numbers match.
    """
    data = validate_event_metadata(combined_audited)
    frames: list[pd.DataFrame] = []
    for key in EVENT_ROLES:
        source = data.loc[data["EventKey"].eq(key)]
        if "Driver" not in source:
            raise ValueError("Missing required model column: Driver")
        actual_drivers = set(source["Driver"].dropna())
        if source["Driver"].isna().any() or actual_drivers != set(DEFAULT_DRIVERS):
            raise ValueError(f"Fixed six-driver cohort mismatch in {key}; observed {sorted(actual_drivers)}.")
        examples = build_examples(source)
        metadata = source.iloc[0]
        for column in EVENT_METADATA:
            examples[column] = metadata[column]
        examples["Split"] = metadata["Role"]
        if not examples.empty:
            frames.append(examples)
    columns = EXAMPLE_COLUMNS + list(EVENT_METADATA) + ["Split"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, columns].sort_values(
        ["RoundNumber", "OriginEndSeconds", "Driver"], ignore_index=True,
    )


def score_weekend_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score every supplied example, retaining weak races and driver results.

    EventKey ALL denotes a pooled split. Driver ALL pools drivers; event/driver
    rows keep denominators explicit. These are lap-weighted point estimates.
    """
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    required = {"Split", "EventKey", "EventName", "Driver", "ActualSeconds",
                "RidgeSeconds", "PersistenceSeconds", "TrailingMedianSeconds"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing scoring columns: {sorted(missing)}")
    for key, subset in predictions.groupby("EventKey", sort=False):
        if subset["EventName"].nunique(dropna=False) != 1 or subset["Split"].nunique(dropna=False) != 1:
            raise ValueError(f"Scored event {key} must have one name and one split.")
    models = {"Ridge": "RidgeSeconds", "Persistence": "PersistenceSeconds",
              "Trailing median": "TrailingMedianSeconds"}
    rows: list[dict[str, Any]] = []
    for split, split_laps in predictions.groupby("Split", sort=False):
        event_cohorts = [("ALL", "ALL", split_laps)] + [
            (str(key), str(laps.iloc[0]["EventName"]), laps)
            for key, laps in split_laps.groupby("EventKey", sort=True)
        ]
        for key, name, event_laps in event_cohorts:
            cohorts = [("ALL", event_laps)] + [
                (str(driver), laps) for driver, laps in event_laps.groupby("Driver", sort=True)
            ]
            for driver, cohort in cohorts:
                for model, column in models.items():
                    error = pd.to_numeric(cohort[column], errors="coerce") - pd.to_numeric(cohort["ActualSeconds"], errors="coerce")
                    if not np.isfinite(error).all():
                        raise ValueError("Every scored prediction and actual must be finite; do not trim failed predictions.")
                    rows.append({
                        "Split": str(split), "EventKey": key, "EventName": name,
                        "Model": model, "Driver": driver, "Count": int(len(error)),
                        "MAESeconds": float(np.mean(np.abs(error))),
                        "RMSESeconds": float(np.sqrt(np.mean(np.square(error)))),
                        "BiasSeconds": float(np.mean(error)),
                    })
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def run_weekend_experiment(combined_audited: pd.DataFrame) -> WeekendExperimentResult:
    """Train on rounds 1/3/4, choose alpha on 5, and score untouched rounds 6/7.

    The candidate list is fixed. Exact validation-MAE ties choose the lowest
    alpha. The chosen model remains fitted only to training races; validation
    is not folded into a later refit. Prior observed test laps may provide the
    current origin's history, as in sequential one-step replay.
    """
    examples = build_weekend_examples(combined_audited)
    event_counts = examples.groupby("EventKey").size().to_dict()
    empty_events = [key for key in EVENT_ROLES if event_counts.get(key, 0) == 0]
    if empty_events:
        raise ValueError(f"Every declared event needs eligible adjacent examples; none in {empty_events}.")
    cohorts = {name: examples.loc[examples["Split"].eq(name)] for name in ("train", "validation", "test")}
    train, validation = cohorts["train"], cohorts["validation"]
    candidates: list[tuple[float, float, Any]] = []
    for alpha in DEFAULT_ALPHAS:
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
    predictions = examples.loc[examples["Split"].isin(["validation", "test"])].copy()
    predictions["ActualSeconds"] = predictions["TargetSeconds"]
    predictions["PersistenceSeconds"] = predictions["LastLapSeconds"]
    predictions["RidgeSeconds"] = predictions["LastLapSeconds"].to_numpy() + selected_pipeline.predict(predictions.loc[:, MODEL_FEATURES])
    for model, column in [("Ridge", "RidgeSeconds"), ("Persistence", "PersistenceSeconds"), ("TrailingMedian", "TrailingMedianSeconds")]:
        predictions[f"{model}ErrorSeconds"] = predictions[column] - predictions["ActualSeconds"]
    metrics = score_weekend_predictions(predictions)
    normalized = validate_event_metadata(combined_audited)
    events = normalized.loc[:, EVENT_METADATA].drop_duplicates().sort_values("RoundNumber")
    counts = {name: int(examples["Split"].eq(name).sum()) for name in ("train", "validation", "test")}
    category_support = {}
    for name, cohort in cohorts.items():
        category_support[name] = {}
        for column in CATEGORICAL_FEATURES:
            training_categories = set(train[column].dropna())
            observed_categories = set(cohort[column].dropna())
            unknown = observed_categories.difference(training_categories)
            category_support[name][column] = {
                "observed_categories": sorted(observed_categories),
                "unknown_categories": sorted(unknown),
                "unknown_example_count": int(cohort[column].isin(unknown).sum()),
            }
    event_protocol = []
    for event in events.to_dict("records"):
        source = normalized.loc[normalized["EventKey"].eq(event["EventKey"])]
        included = source["Included"].astype("string").str.strip().str.lower().eq("true").fillna(False)
        event_protocol.append({
            **event, "recorded_laps": int(len(source)), "eligible_laps": int(included.sum()),
            "examples": int(event_counts[event["EventKey"]]),
        })
    protocol = {
        "experiment": "Conditional next-lap pace, fixed six-weekend 2024 evaluation",
        "events": event_protocol, "drivers": list(DEFAULT_DRIVERS),
        "features": list(MODEL_FEATURES), "residual_target": "TargetSeconds - LastLapSeconds",
        "target": "Actual lap t+1, retrospectively eligible and in the same reported stint and compound",
        "prediction_origin": "Nominal completion of lap t; public feed arrival latency and revisions are not replayed",
        "eligibility": "Original nine-rule lap audit, applied identically to each event; final archived flags, not predicted",
        "history": "At most three consecutive eligible laps through t; reset on exclusions, missing laps, invalid metadata, stint/compound changes, and every event boundary",
        "split": "Whole events, chronologically fixed: train rounds 1/3/4; validation round 5; test rounds 6/7",
        "counts": counts, "alphas": list(DEFAULT_ALPHAS), "selected_alpha": selected_alpha,
        "selection": "Lowest pooled validation lap MAE; exact ties choose lowest alpha",
        "fit": "Scaler, categorical encoder and Ridge fitted only on training races; selected fit frozen with no validation refit",
        "baselines": ["Last lap persistence", "Trailing median of at most three laps through t"],
        "signed_error": "Prediction minus actual seconds; positive means predicted slower",
        "metrics": "Lap-weighted MAE/RMSE/bias, pooled by split and separately per event/driver; every valid example retained",
        "metadata_features": "EventKey, EventName, Year, RoundNumber, EventDate and Role are bookkeeping only, never model inputs",
        "category_policy": "One-hot categories fitted only on training races; known valid categories absent from training produce zero indicators with handle_unknown=ignore, not a learned category effect",
        "category_support_by_split": category_support,
        "limitations": [
            "Six selected drivers, six 2024 races and two held-out weekends; broader generalization is untested",
            "Same drivers recur across folds; this tests event transfer rather than unseen-driver transfer",
            "Retrospective conditional cohort; no prediction of target eligibility, pit stops or incidents",
            "Earlier observed validation/test laps may supply origin history without retraining",
            "Adjacent lap errors are correlated; no independent-sample significance or calibrated interval claims",
            "Public timing and reported metadata; no private fuel, tyre wear, setup or brake-pressure measurements",
            "Fixed test weekends may expose failure; no test-score retuning or event replacement",
        ],
    }
    return WeekendExperimentResult(examples, predictions.reset_index(drop=True), metrics, tuning, selected_pipeline, protocol)
