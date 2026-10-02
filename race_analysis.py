from __future__ import annotations

"""Descriptive public lap timing comparisons, matched by race lap.

These functions accept the output of analyze_data.audit_laps. Eligibility comes
from that audit; matching never proves clear air or isolates fuel, tyre wear,
setup or driver effects. TyreLife is reported usage in laps, not measured wear.
"""

import pandas as pd

KNOWN_COMPOUNDS = frozenset({
    "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET",
    "SUPERSOFT", "ULTRASOFT", "HYPERSOFT", "SUPERHARD",
})
REQUIRED_COLUMNS = {
    "Driver", "LapNumber", "Stint", "Compound", "TyreLife",
    "LapTimeSeconds", "Included",
}
STINT_COLUMNS = [
    "Driver", "Stint", "Compound", "RecordedLaps", "EligibleLaps",
    "ExcludedLaps", "EligibilityRate", "LapStart", "LapEnd",
    "EligibleLapStart", "EligibleLapEnd", "TyreLifeMin", "TyreLifeMax",
    "MedianLapTimeSeconds", "IQRSeconds",
]
MATCHED_COLUMNS = [
    "LapNumber", "ReferenceDriver", "ComparisonDriver", "ReferenceStint",
    "ComparisonStint", "ReferenceCompound", "ComparisonCompound",
    "ReferenceTyreLife", "ComparisonTyreLife", "ReferenceLapTimeSeconds",
    "ComparisonLapTimeSeconds", "SameCompound", "DeltaSeconds",
]
SEGMENT_COLUMNS = [
    "ReferenceDriver", "ComparisonDriver", "ReferenceStint", "ComparisonStint",
    "ReferenceCompound", "ComparisonCompound", "SameCompound", "LapStart",
    "LapEnd", "MatchedLaps", "MedianDeltaSeconds", "IQRDeltaSeconds",
    "ReferenceMedianLapTimeSeconds", "ComparisonMedianLapTimeSeconds",
    "ReferenceTyreLifeMin", "ReferenceTyreLifeMax",
    "ComparisonTyreLifeMin", "ComparisonTyreLifeMax",
]


def _prepare(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing audited lap columns: {sorted(missing)}")
    result = data.copy()
    for column in ("LapNumber", "LapTimeSeconds", "TyreLife"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    # CSV strings must never acquire eligibility through Python truthiness.
    included = result["Included"].astype("string").str.strip().str.lower()
    invalid_flags = included.notna() & ~included.isin(["true", "false"])
    if invalid_flags.any():
        raise ValueError("Included must contain explicit true/false audit flags.")
    result["Included"] = included.eq("true").fillna(False).astype(bool)
    eligible = result["Included"]
    valid_number = (
        result["LapNumber"].gt(0)
        & result["LapNumber"].lt(float("inf"))
        & result["LapNumber"].mod(1).eq(0)
    )
    valid_time = result["LapTimeSeconds"].gt(0) & result["LapTimeSeconds"].lt(float("inf"))
    if (eligible & ~(valid_number & valid_time)).any():
        raise ValueError("Eligible laps require positive finite times and integer lap numbers.")
    if result.duplicated(["Driver", "LapNumber"]).any():
        raise ValueError("Duplicate driver/lap records cannot be matched safely.")
    compound = result["Compound"].astype("string").str.strip().str.upper()
    result["Compound"] = compound.mask(
        compound.isin(["", "UNKNOWN", "NONE", "NAN", "N/A", "NA", "NULL"])
    )
    return result.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def stint_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize each reported driver/stint/compound group descriptively.

    RecordedLaps is the denominator, including excluded laps. Timing and tyre
    usage summaries use only eligible laps. Windows are race-lap bounds and
    may contain gaps; EligibleLaps records the actual sample count. Whole-stint
    medians need not cover the same race laps and cannot establish an advantage.
    A single eligible lap has IQR zero, which is not evidence of consistency.
    """
    prepared = _prepare(data)
    rows = []
    for (driver, stint, compound), group in prepared.groupby(
        ["Driver", "Stint", "Compound"], dropna=False, sort=False
    ):
        eligible = group.loc[group["Included"]]
        recorded_count = len(group)
        eligible_count = len(eligible)
        rows.append({
            "Driver": driver, "Stint": stint, "Compound": compound,
            "RecordedLaps": recorded_count, "EligibleLaps": eligible_count,
            "ExcludedLaps": recorded_count - eligible_count,
            "EligibilityRate": eligible_count / recorded_count,
            "LapStart": group["LapNumber"].min(),
            "LapEnd": group["LapNumber"].max(),
            "EligibleLapStart": eligible["LapNumber"].min(),
            "EligibleLapEnd": eligible["LapNumber"].max(),
            "TyreLifeMin": eligible["TyreLife"].min(),
            "TyreLifeMax": eligible["TyreLife"].max(),
            "MedianLapTimeSeconds": eligible["LapTimeSeconds"].median(),
            "IQRSeconds": _iqr(eligible["LapTimeSeconds"]) if eligible_count else float("nan"),
        })
    return pd.DataFrame(rows, columns=STINT_COLUMNS)


def matched_laps(
    data: pd.DataFrame,
    reference: str = "LEC",
    comparison: str = "SAI",
    same_compound: bool = True,
) -> pd.DataFrame:
    """Inner-join eligible laps on race LapNumber, optionally by known compound.

    DeltaSeconds = comparison time minus reference time; negative means the
    comparison driver was faster on the matched lap. Tyre age and stint IDs
    are preserved, never used to match laps from different race windows.
    Unknown or unrecognized compound labels cannot satisfy SameCompound.
    Empty/missing driver selections yield no pairs; same-driver pairs are invalid.
    """
    if reference == comparison:
        raise ValueError("Choose two different drivers.")
    prepared = _prepare(data)
    source_columns = ["LapNumber", "Driver", "Stint", "Compound", "TyreLife", "LapTimeSeconds"]
    eligible = prepared.loc[prepared["Included"], source_columns]
    left = eligible.loc[eligible["Driver"].eq(reference)].rename(
        columns={column: f"Reference{column}" for column in source_columns if column != "LapNumber"}
    )
    right = eligible.loc[eligible["Driver"].eq(comparison)].rename(
        columns={column: f"Comparison{column}" for column in source_columns if column != "LapNumber"}
    )
    pairs = left.merge(right, on="LapNumber", how="inner", validate="one_to_one")
    known_left = pairs["ReferenceCompound"].isin(KNOWN_COMPOUNDS)
    known_right = pairs["ComparisonCompound"].isin(KNOWN_COMPOUNDS)
    pairs["SameCompound"] = (
        known_left & known_right
        & pairs["ReferenceCompound"].eq(pairs["ComparisonCompound"]).fillna(False)
    ).astype(bool)
    if same_compound:
        pairs = pairs.loc[pairs["SameCompound"]].copy()
    pairs["DeltaSeconds"] = pairs["ComparisonLapTimeSeconds"] - pairs["ReferenceLapTimeSeconds"]
    return pairs.reindex(columns=MATCHED_COLUMNS).sort_values("LapNumber").reset_index(drop=True)


def segment_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize contiguous matched windows with unchanged stint/compound pairs.

    Missing eligible race laps, stint changes, or compound changes split the
    window. MedianDeltaSeconds is the median of paired lap differences, not
    the difference between separate medians. MatchedLaps is its denominator.
    Single-lap segments are retained and explicitly identifiable by their count.
    """
    missing = set(MATCHED_COLUMNS).difference(pairs.columns)
    if missing:
        raise ValueError(f"Missing matched lap columns: {sorted(missing)}")
    if pairs.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    ordered = pairs.sort_values("LapNumber").reset_index(drop=True).copy()
    if ordered["LapNumber"].duplicated().any():
        raise ValueError("Each race lap must occur once in a matched comparison.")
    for column in ("ReferenceDriver", "ComparisonDriver"):
        if ordered[column].nunique(dropna=False) != 1:
            raise ValueError("Segments must describe a single driver comparison.")
    keys = [
        "ReferenceStint", "ComparisonStint",
        "ReferenceCompound", "ComparisonCompound", "SameCompound",
    ]
    unchanged = pd.Series(True, index=ordered.index)
    for key in keys:
        current = ordered[key]
        previous = current.shift()
        unchanged &= (current.eq(previous).fillna(False) | (current.isna() & previous.isna()))
    contiguous = ordered["LapNumber"].diff().eq(1)
    groups = (~(unchanged & contiguous)).cumsum()
    rows = []
    for _, group in ordered.groupby(groups, sort=False):
        first = group.iloc[0]
        rows.append({
            "ReferenceDriver": first["ReferenceDriver"],
            "ComparisonDriver": first["ComparisonDriver"],
            "ReferenceStint": first["ReferenceStint"],
            "ComparisonStint": first["ComparisonStint"],
            "ReferenceCompound": first["ReferenceCompound"],
            "ComparisonCompound": first["ComparisonCompound"],
            "SameCompound": bool(first["SameCompound"]),
            "LapStart": group["LapNumber"].min(),
            "LapEnd": group["LapNumber"].max(),
            "MatchedLaps": len(group),
            "MedianDeltaSeconds": group["DeltaSeconds"].median(),
            "IQRDeltaSeconds": _iqr(group["DeltaSeconds"]),
            "ReferenceMedianLapTimeSeconds": group["ReferenceLapTimeSeconds"].median(),
            "ComparisonMedianLapTimeSeconds": group["ComparisonLapTimeSeconds"].median(),
            "ReferenceTyreLifeMin": group["ReferenceTyreLife"].min(),
            "ReferenceTyreLifeMax": group["ReferenceTyreLife"].max(),
            "ComparisonTyreLifeMin": group["ComparisonTyreLife"].min(),
            "ComparisonTyreLifeMax": group["ComparisonTyreLife"].max(),
        })
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def matched_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Alias for segment_summary; each output row is a contiguous window."""
    return segment_summary(pairs)
