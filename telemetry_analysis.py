from __future__ import annotations

"""Conservative distance estimates from native public car telemetry.

Distance is trapezoidal speed integration, zeroed at the first recorded sample.
It is neither circuit position nor a measured lap-distance channel. No speed is
inferred before/after recorded samples. An unknown speed, timestamp or excessive
time gap makes the later absolute offset unknowable; distance is then withheld.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Driver", "LapNumber", "ElapsedSeconds", "SpeedKmh", "ThrottlePct",
    "BrakeOn", "RPM", "Gear", "DRS", "SampleSource",
}
DISTANCE_LABEL = "Estimated distance since first recorded sample (m)"
ALIGNMENT_COLUMNS = [
    "EstimatedDistanceM", "LeftSpeedKmh", "RightSpeedKmh", "SpeedDifferenceKmh",
    "LeftThrottlePct", "RightThrottlePct", "LeftBrakeOn", "RightBrakeOn",
]


@dataclass(frozen=True)
class TelemetryQuality:
    driver: str
    lap_number: int
    lap_time_seconds: float
    sample_count: int
    reordered: bool
    invalid_elapsed_samples: int
    invalid_speed_samples: int
    invalid_throttle_samples: int
    invalid_brake_samples: int
    first_elapsed_seconds: float | None
    last_elapsed_seconds: float | None
    start_missing_seconds: float | None
    end_missing_seconds: float | None
    observed_duration_seconds: float
    observed_time_fraction: float
    median_sample_interval_seconds: float | None
    max_sample_interval_seconds: float | None
    gap_count: int
    max_gap_seconds: float
    distance_supported_samples: int
    estimated_observed_distance_m: float | None
    distance_is_continuous: bool
    distance_is_strictly_increasing: bool
    boundary_coverage_within_tolerance: bool
    alignment_eligible: bool
    sample_sources: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PreparedLap:
    samples: pd.DataFrame
    quality: TelemetryQuality


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def _brake_flag(values: pd.Series) -> pd.Series:
    # Do not use bool("False"). Invalid values remain missing, never pressure.
    flags = values.astype("string").str.strip().str.lower()
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
               "1.0": 1.0, "0.0": 0.0}
    return flags.map(mapping).astype(float)


def prepare_lap_telemetry(
    raw: pd.DataFrame, *, driver: str, lap_number: int,
    lap_time_seconds: float, max_gap_seconds: float = 1.0,
) -> PreparedLap:
    """Select/validate a lap, retaining invalid channel samples as missing.

    Valid elapsed timestamps are sorted; duplicate timestamps are rejected.
    Out-of-range or unknown timestamps retain their rows but disable distance
    for the whole lap because their position within the sequence is unknown.
    BrakeOn is encoded 0/1/NaN so plots and CSV never imply brake pressure.
    Original channel values remain in Raw* columns for inspection.
    GapBefore marks excessive time gaps/invalid timestamps: elapsed-time plots
    must insert a line separator before those rows to avoid bridging gaps.

    max_gap_seconds is an explicit analysis threshold, not a provider guarantee.
    Boundary coverage uses the same tolerance, and never fills missing boundaries.
    """
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {sorted(missing)}")
    if not isinstance(driver, str) or not driver.strip():
        raise ValueError("driver must be a nonempty driver code.")
    if not np.isfinite(lap_number) or lap_number <= 0 or int(lap_number) != lap_number:
        raise ValueError("lap_number must be a positive integer.")
    if not np.isfinite(lap_time_seconds) or lap_time_seconds <= 0:
        raise ValueError("lap_time_seconds must be positive and finite.")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive and finite.")
    driver = driver.strip().upper()
    lap_number = int(lap_number)
    chosen = raw.loc[
        raw["Driver"].astype("string").str.strip().str.upper().eq(driver)
        & pd.to_numeric(raw["LapNumber"], errors="coerce").eq(lap_number)
    ].copy().reset_index(drop=True)
    if chosen.empty:
        raise ValueError(f"No telemetry samples for {driver} lap {lap_number}.")
    chosen["Driver"] = driver
    chosen["LapNumber"] = lap_number
    for column in ("ElapsedSeconds", "SpeedKmh", "ThrottlePct", "BrakeOn", "RPM", "Gear", "DRS"):
        chosen[f"Raw{column}"] = chosen[column]
    chosen["ElapsedSeconds"] = _numeric(chosen["ElapsedSeconds"])
    valid_elapsed = (
        np.isfinite(chosen["ElapsedSeconds"])
        & chosen["ElapsedSeconds"].between(0, lap_time_seconds)
    )
    invalid_elapsed_count = int((~valid_elapsed).sum())
    chosen.loc[~valid_elapsed, "ElapsedSeconds"] = np.nan
    if chosen.loc[valid_elapsed, "ElapsedSeconds"].duplicated().any():
        raise ValueError("Duplicate elapsed timestamps make telemetry alignment ambiguous.")
    chosen["_InputOrder"] = np.arange(len(chosen))
    chosen = chosen.sort_values("ElapsedSeconds", kind="stable", na_position="last").reset_index(drop=True)
    reordered = not np.array_equal(chosen["_InputOrder"], np.arange(len(chosen)))
    chosen = chosen.drop(columns="_InputOrder")
    for column in ("SpeedKmh", "ThrottlePct", "RPM", "Gear", "DRS"):
        chosen[column] = _numeric(chosen[column])
        valid = np.isfinite(chosen[column]) & chosen[column].ge(0)
        if column == "ThrottlePct":
            valid &= chosen[column].le(100)
        elif column in ("Gear", "DRS"):
            valid &= chosen[column].mod(1).eq(0)
        chosen.loc[~valid, column] = np.nan
    chosen["BrakeOn"] = _brake_flag(chosen["BrakeOn"])
    times = chosen["ElapsedSeconds"].to_numpy(dtype=float)
    speeds = chosen["SpeedKmh"].to_numpy(dtype=float)
    valid_times = times[np.isfinite(times)]
    intervals = np.diff(valid_times)
    first = float(valid_times[0]) if len(valid_times) else None
    last = float(valid_times[-1]) if len(valid_times) else None
    distance = np.full(len(chosen), np.nan)
    if invalid_elapsed_count == 0 and np.isfinite(speeds[0]):
        distance[0] = 0.0
        for index in range(1, len(chosen)):
            interval = times[index] - times[index - 1]
            if interval > max_gap_seconds or not np.isfinite(speeds[index]):
                break
            distance[index] = distance[index - 1] + (
                (speeds[index - 1] + speeds[index]) / 2 / 3.6 * interval
            )
    chosen["EstimatedDistanceM"] = distance
    gap_before = np.zeros(len(chosen), dtype=bool)
    if len(chosen) > 1:
        gap_before[1:] = np.diff(times) > max_gap_seconds
    gap_before |= ~np.isfinite(times)
    chosen["GapBefore"] = gap_before
    supported = distance[np.isfinite(distance)]
    continuous = len(chosen) >= 2 and len(supported) == len(chosen)
    strictly_increasing = continuous and bool(np.all(np.diff(distance) > 0))
    boundary_coverage = (
        first is not None and last is not None
        and first <= max_gap_seconds and lap_time_seconds - last <= max_gap_seconds
    )
    observed_duration = 0.0 if first is None or last is None else last - first
    invalid_speed = int(chosen["SpeedKmh"].isna().sum())
    invalid_throttle = int(chosen["ThrottlePct"].isna().sum())
    invalid_brake = int(chosen["BrakeOn"].isna().sum())
    gaps = int((intervals > max_gap_seconds).sum())
    warnings = [
        "Distance is estimated from speed and starts at the first recorded sample; lap-start offsets can differ."
    ]
    if first is not None and (first > 0 or lap_time_seconds - last > 0):
        warnings.append("Missing lap-boundary time is disclosed, not extrapolated.")
    if invalid_elapsed_count:
        warnings.append("Invalid timestamps disable all distance estimates for this lap.")
    if invalid_speed or gaps:
        warnings.append("Distance is withheld from the first invalid speed or excessive sampling gap onward.")
    if invalid_throttle or invalid_brake:
        warnings.append("Invalid throttle/brake values remain gaps in their channel traces.")
    if not boundary_coverage:
        warnings.append("Recorded samples do not cover lap boundaries within the stated tolerance.")
    if continuous and not strictly_increasing:
        warnings.append("Stationary intervals make a unique distance alignment ambiguous.")
    quality = TelemetryQuality(
        driver=driver, lap_number=lap_number, lap_time_seconds=float(lap_time_seconds),
        sample_count=len(chosen), reordered=reordered,
        invalid_elapsed_samples=invalid_elapsed_count, invalid_speed_samples=invalid_speed,
        invalid_throttle_samples=invalid_throttle, invalid_brake_samples=invalid_brake,
        first_elapsed_seconds=first, last_elapsed_seconds=last,
        start_missing_seconds=first,
        end_missing_seconds=None if last is None else float(lap_time_seconds - last),
        observed_duration_seconds=observed_duration,
        observed_time_fraction=observed_duration / lap_time_seconds,
        median_sample_interval_seconds=float(np.median(intervals)) if len(intervals) else None,
        max_sample_interval_seconds=float(np.max(intervals)) if len(intervals) else None,
        gap_count=gaps, max_gap_seconds=float(max_gap_seconds),
        distance_supported_samples=len(supported),
        estimated_observed_distance_m=float(supported[-1]) if len(supported) >= 2 else None,
        distance_is_continuous=continuous,
        distance_is_strictly_increasing=strictly_increasing,
        boundary_coverage_within_tolerance=bool(boundary_coverage),
        alignment_eligible=bool(strictly_increasing and boundary_coverage),
        sample_sources=tuple(sorted(chosen["SampleSource"].dropna().astype(str).unique())),
        warnings=tuple(warnings),
    )
    return PreparedLap(samples=chosen, quality=quality)


def _interpolate_channel(samples: pd.DataFrame, channel: str, grid: np.ndarray, *, step: bool = False) -> np.ndarray:
    """Interpolate only contiguous valid-channel runs; missing rows break support."""
    distance = samples["EstimatedDistanceM"].to_numpy(dtype=float)
    values = samples[channel].to_numpy(dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(distance) & np.isfinite(values))
    output = np.full(len(grid), np.nan)
    if not len(valid_indices):
        return output
    runs = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1)
    for run in runs:
        x, y = distance[run], values[run]
        inside = (grid >= x[0]) & (grid <= x[-1])
        if len(run) == 1:
            exact = grid == x[0]
            output[exact] = y[0]
        elif step:
            previous = np.searchsorted(x, grid[inside], side="right") - 1
            output[inside] = y[previous]
        else:
            output[inside] = np.interp(grid[inside], x, y)
    return output


def align_laps_by_distance(left: PreparedLap, right: PreparedLap, *, spacing_m: float = 10.0) -> pd.DataFrame:
    """Create a shared sample-origin distance grid without extrapolation.

    Requires continuous speed support and boundary coverage within each lap's
    declared gap tolerance. This still does not prove identical circuit position:
    starting sample times and driven paths can differ. Speed/throttle use linear
    interpolation; brake uses preceding-sample step values within valid runs.
    Missing channel rows are not bridged. No lap-time delta is inferred.
    """
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be positive and finite.")
    for side, lap in (("left", left), ("right", right)):
        if not lap.quality.alignment_eligible:
            raise ValueError(f"The {side} lap lacks continuous distance support or adequate boundary coverage.")
    end = min(float(left.samples["EstimatedDistanceM"].iloc[-1]), float(right.samples["EstimatedDistanceM"].iloc[-1]))
    grid = np.arange(0.0, end, spacing_m)
    grid = np.append(grid, end)
    aligned = pd.DataFrame({"EstimatedDistanceM": grid})
    for side, lap in (("Left", left), ("Right", right)):
        for channel in ("SpeedKmh", "ThrottlePct", "BrakeOn"):
            aligned[f"{side}{channel}"] = _interpolate_channel(lap.samples, channel, grid, step=channel == "BrakeOn")
    aligned["SpeedDifferenceKmh"] = aligned["RightSpeedKmh"] - aligned["LeftSpeedKmh"]
    return aligned[ALIGNMENT_COLUMNS]
