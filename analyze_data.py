"""Audit the downloaded public lap data and render the first PitWall chart."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "bahrain_2024_ferrari_laps.csv"
REPORTS = ROOT / "reports"
DRIVERS = ("LEC", "SAI")
RULES = {
    "invalid_lap_time": "Lap time is missing, unparseable, non-finite or non-positive.",
    "invalid_lap_number": "Lap number is missing, non-positive or not an integer.",
    "opening_lap": "Opening lap includes the standing start.",
    "pit_in": "Reported pit-in timestamp is present.",
    "pit_out": "Reported pit-out timestamp is present.",
    "not_green_only": "Track status is not exclusively 1 (green), including unknown statuses.",
    "timing_not_accurate": "IsAccurate is not explicitly true, including unknown values.",
    "deleted_or_unknown": "Deleted is not explicitly false, including unknown values.",
    "generated_or_unknown": "FastF1Generated is not explicitly false, including unknown values.",
}


def parse_bool(values: pd.Series) -> pd.Series:
    """Do not confuse a CSV string 'False' with Python truthiness."""
    return values.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False}
    ).astype("boolean")


def audit_laps(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "Driver", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
        "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate",
        "Deleted", "FastF1Generated",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(raw["Driver"].dropna()) != set(DRIVERS):
        raise ValueError("Expected only LEC and SAI, with both drivers present.")

    data = raw.copy()
    data["LapNumber"] = pd.to_numeric(data["LapNumber"], errors="coerce")
    if data.duplicated(["Driver", "LapNumber"]).any():
        raise ValueError("Duplicate driver/lap records require investigation.")
    data = data.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    seconds = pd.to_timedelta(data["LapTime"], errors="coerce").dt.total_seconds()
    data["LapTimeSeconds"] = seconds
    status = data["TrackStatus"].astype("string").str.strip()
    # Also accept the CSV numeric representation 1.0, but never a mixed status 12.
    status = status.str.replace(r"\.0$", "", regex=True)
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
    data["Included"] = ~checks.any(axis=1)
    data["ExclusionReasons"] = checks.apply(
        lambda row: ";".join(row.index[row]), axis=1
    )
    assert data.loc[data["Included"], "ExclusionReasons"].eq("").all()
    assert data.loc[~data["Included"], "ExclusionReasons"].ne("").all()
    return data, checks


def render_chart(data: pd.DataFrame, output: Path) -> None:
    os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LEC": "#1473b8", "SAI": "#db5816"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("PitWall | Bahrain 2024 race pace", fontsize=18, fontweight="bold")
    for driver in DRIVERS:
        laps = data[data["Driver"].eq(driver)]
        axes[0].plot(
            laps["LapNumber"], laps["LapTimeSeconds"],
            color=colors[driver], linewidth=1.1, marker="o", markersize=3,
            label=f"{driver}: {len(laps)} recorded laps",
        )
        count = int(laps["Included"].sum())
        # NaNs break the line at every omitted lap and at every stint boundary.
        first = True
        for _, stint in laps.groupby("Stint", dropna=False):
            axes[1].plot(
                stint["LapNumber"],
                stint["LapTimeSeconds"].where(stint["Included"]),
                color=colors[driver], linewidth=1.2, marker="o", markersize=3,
                label=f"{driver}: {count} eligible laps" if first else None,
            )
            first = False
    excluded = data[~data["Included"]]
    axes[0].scatter(
        excluded["LapNumber"], excluded["LapTimeSeconds"],
        facecolors="none", edgecolors="#202020", s=65, linewidths=1.2,
        zorder=5, label=f"Excluded from pace analysis: {len(excluded)}",
    )
    axes[0].set_title("All recorded lap times; outlined points are excluded", fontsize=11)
    axes[1].set_title("Analysis-eligible laps; gaps preserve exclusions and stint changes", fontsize=11)
    for axis in axes:
        axis.set_ylabel("Lap time (seconds)")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=9, loc="upper right")
    axes[1].set_xlabel("Race lap")
    axes[1].set_xlim(0, data["LapNumber"].max() + 1)
    fig.text(
        0.08, 0.02,
        "Public timing via FastF1. Eligible = green-only, non-pit, non-opening, valid timing and flags.\n"
        "Eligibility does not establish clear air; trends combine traffic, fuel, tyres and driving.",
        fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    raw = pd.read_csv(INPUT)
    data, checks = audit_laps(raw)
    REPORTS.mkdir(exist_ok=True)
    data.to_csv(REPORTS / "lap_audit.csv", index=False)
    data[data["Included"]].to_csv(REPORTS / "eligible_laps.csv", index=False)
    counts = data.groupby("Driver")["Included"].agg(["count", "sum"])
    summary = {
        "case_study": "2024 Bahrain Grand Prix race, LEC and SAI",
        "source": "Public timing exported with FastF1",
        "source_file": INPUT.name,
        "source_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "input_rows": len(data),
        "eligible_rows": int(data["Included"].sum()),
        "excluded_rows": int((~data["Included"]).sum()),
        "drivers": {
            driver: {
                "recorded": int(row["count"]),
                "eligible": int(row["sum"]),
                "excluded": int(row["count"] - row["sum"]),
            } for driver, row in counts.iterrows()
        },
        "exclusion_counts": {key: int(value) for key, value in checks.sum().items()},
        "note": "Exclusion reasons overlap; their counts must not be added.",
    }
    (REPORTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# PitWall first data audit", "",
        f"Input: {len(data)} records. Eligible: {summary['eligible_rows']}. "
        f"Excluded: {summary['excluded_rows']}.", "",
        "| Driver | Recorded | Eligible | Excluded |",
        "|---|---:|---:|---:|",
    ]
    for driver, counts in summary["drivers"].items():
        lines.append(
            f"| {driver} | {counts['recorded']} | {counts['eligible']} | {counts['excluded']} |"
        )
    lines += [
        "", "## Exclusion rules", "",
        "A lap is retained only if it passes every rule. Reasons overlap.", "",
    ]
    for key, description in RULES.items():
        lines.append(f"- **{key}: {summary['exclusion_counts'][key]}**. {description}")
    lines += [
        "", "## Interpretation", "",
        "The figure retains all timed laps in its upper panel and eligible laps in its lower panel.",
        "Lines break across omitted laps and stint boundaries. Lower lap time means faster.",
        "Different scales in the two panels show pit-stop effects and racing pace separately.", "",
        "These are analysis-eligible laps, not verified traffic-free laps.",
        "IsAccurate concerns timing integrity; it does not guarantee an error-free driving lap.",
        "TyreLife is reported usage in laps, not measured tyre wear.",
        "A stint slope cannot separate tyre effects from fuel burn, traffic, pace management or track evolution.",
        "Unmatched whole-race medians cannot establish a driver or car's intrinsic advantage.", "",
        "Next: compare overlapping race-lap windows and compounds, with sample counts.",
        "This milestone includes no prediction model and no causal tyre-degradation estimate.", "",
        "Source documentation: https://github.com/theOehrly/Fast-F1",
        f"Input SHA-256: {summary['source_sha256']}", "",
    ]
    (REPORTS / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    render_chart(data, REPORTS / "race_pace.png")
    print(json.dumps(summary, indent=2))
    print(f"Chart: {REPORTS / 'race_pace.png'}")


if __name__ == "__main__":
    main()
