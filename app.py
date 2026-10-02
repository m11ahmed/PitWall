"""PitWall: public-data race pace and tyre-stint dashboard."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyze_data import INPUT, RULES, audit_laps
from race_analysis import matched_laps, segment_summary, stint_summary
from telemetry_ui import show_telemetry
from model_ui import show_model
from weekend_ui import show_weekends

ROOT = Path(__file__).resolve().parent
COLORS = {"LEC": "#69a9ff", "SAI": "#ffa96b"}
NAMES = {"LEC": "Charles Leclerc", "SAI": "Carlos Sainz"}
COMPOUNDS = {"SOFT": "#f36562", "MEDIUM": "#e8c65a", "HARD": "#d9e2ec"}
REASON_LABELS = {
    "invalid_lap_time": "Invalid lap time",
    "invalid_lap_number": "Invalid lap number",
    "opening_lap": "Opening lap",
    "pit_in": "Pit-in lap",
    "pit_out": "Pit-out lap",
    "not_green_only": "Non-green or unknown track status",
    "timing_not_accurate": "Timing quality not confirmed",
    "deleted_or_unknown": "Deleted lap or unknown deletion flag",
    "generated_or_unknown": "Generated lap or unknown generation flag",
}

st.set_page_config(page_title="PitWall | Race Pace Lab", page_icon="🏁", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(path: str, modified_ns: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Include file modification time in the cache key when the CSV is replaced.
    return audit_laps(pd.read_csv(path))


def style_chart(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=15, r=20, t=35, b=30),
        font=dict(family="Arial, sans-serif", size=13),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(220,230,245,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(220,230,245,0.08)", zeroline=False)
    return fig


def pace_chart(scope: pd.DataFrame, eligible_only: bool) -> go.Figure:
    fig = go.Figure()
    for driver, laps in scope.groupby("Driver", sort=True):
        first = True
        for _, stint in laps.groupby("Stint", dropna=False, sort=True):
            plotted = stint["LapTimeSeconds"].where(stint["Included"]) if eligible_only else stint["LapTimeSeconds"]
            custom = pd.DataFrame({
                "Compound": stint["Compound"].fillna("Unknown"),
                "TyreLife": stint["TyreLife"],
                "Stint": stint["Stint"],
                "Eligibility": stint["Included"].map({True: "Eligible", False: "Excluded"}),
                "Reasons": stint["ExclusionReasons"].replace("", "None"),
            })
            fig.add_trace(go.Scatter(
                x=stint["LapNumber"], y=plotted, mode="lines+markers",
                name=f"{driver} · {NAMES[driver]}", legendgroup=driver,
                showlegend=first, connectgaps=False,
                line=dict(color=COLORS[driver], width=2),
                marker=dict(size=6), customdata=custom.to_numpy(),
                hovertemplate=(
                    f"<b>{driver}</b> · lap %{{x:.0f}}<br>Lap time: %{{y:.3f}} s"
                    "<br>Compound: %{customdata[0]} · stint %{customdata[2]:.0f}"
                    "<br>Reported tyre age: %{customdata[1]:.0f} laps"
                    "<br>%{customdata[3]}<br>Exclusions: %{customdata[4]}<extra></extra>"
                ),
            ))
            first = False
    if not eligible_only:
        excluded = scope[~scope["Included"]]
        fig.add_trace(go.Scatter(
            x=excluded["LapNumber"], y=excluded["LapTimeSeconds"], mode="markers",
            name="Excluded laps", marker=dict(symbol="circle-open", size=13, color="#f3f5f7", line=dict(width=2)),
            customdata=excluded["ExclusionReasons"].to_numpy(),
            hovertemplate="Lap %{x:.0f}: %{y:.3f} s<br>%{customdata}<extra>Excluded</extra>",
        ))
    fig.update_xaxes(title="Race lap", dtick=5)
    fig.update_yaxes(title="Lap time (seconds)")
    return style_chart(fig)


def timeline_chart(scope: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    seen = set()
    for (driver, stint_id), laps in scope.groupby(["Driver", "Stint"], dropna=False, sort=True):
        compounds = laps["Compound"].dropna().unique()
        compound = str(compounds[0]) if len(compounds) == 1 else "UNKNOWN"
        start, end = int(laps["LapNumber"].min()), int(laps["LapNumber"].max())
        label = f"S{int(stint_id)} · {compound}" if pd.notna(stint_id) else compound
        fig.add_trace(go.Bar(
            y=[driver], x=[end - start + 1], base=[start - 1], orientation="h",
            name=compound, legendgroup=compound, showlegend=compound not in seen,
            marker=dict(color=COMPOUNDS.get(compound, "#8291a5"), line=dict(color="#10151f", width=2)),
            text=[label], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#10151f", size=12),
            customdata=[[start, end, int(laps["Included"].sum()), len(laps)]],
            hovertemplate=(
                f"<b>{driver} · {label}</b><br>Race laps %{{customdata[0]}}–%{{customdata[1]}}"
                "<br>Eligible: %{customdata[2]} of %{customdata[3]}<extra></extra>"
            ),
        ))
        seen.add(compound)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Completed race laps", range=[0, scope["LapNumber"].max() + 1])
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=["SAI", "LEC"])
    return style_chart(fig, height=230)


def paired_chart(pairs: pd.DataFrame, reference: str, comparison: str) -> go.Figure:
    fig = go.Figure()
    if not pairs.empty:
        # Do not draw lines through excluded or unmatched race laps.
        dense = pairs.set_index("LapNumber").reindex(
            range(int(pairs["LapNumber"].min()), int(pairs["LapNumber"].max()) + 1)
        )
        fig.add_trace(go.Scatter(
            x=dense.index, y=dense["DeltaSeconds"], mode="lines+markers",
            connectgaps=False, name=f"{comparison} − {reference}",
            line=dict(color=COLORS[comparison], width=2),
            customdata=dense[["ReferenceCompound", "ComparisonCompound",
                              "ReferenceTyreLife", "ComparisonTyreLife"]].to_numpy(),
            hovertemplate=(
                "Lap %{x:.0f}<br>Paired delta: %{y:+.3f} s"
                "<br>Reference: %{customdata[0]}, tyre age %{customdata[2]:.0f}"
                "<br>Comparison: %{customdata[1]}, tyre age %{customdata[3]:.0f}<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_color="#aeb9c8", line_dash="dash")
    fig.update_xaxes(title="Same race lap", dtick=5)
    fig.update_yaxes(title=f"{comparison} − {reference} lap time (seconds)")
    return style_chart(fig, height=350)


def show_pace(scope: pd.DataFrame, cohort: str) -> None:
    st.subheader("Race pace")
    eligible_only = cohort == "Analysis-eligible laps"
    st.caption(
        "Hover to inspect the lap, compound, tyre age and exclusions. Lines break at omitted laps and stint changes."
    )
    if eligible_only and not scope["Included"].any():
        st.info("No analysis-eligible laps in this selection. Widen the lap window or choose all recorded laps.")
    else:
        st.plotly_chart(pace_chart(scope, eligible_only), width="stretch", key="pace_chart")
    st.caption("Lower lap time means faster. Traffic, fuel burn, tyre effects and driver choices all affect these observations.")
    eligible = scope[scope["Included"]]
    if not eligible.empty:
        stats = eligible.groupby("Driver")["LapTimeSeconds"].agg(
            EligibleLaps="count", MedianSeconds="median",
            IQRSeconds=lambda s: s.quantile(0.75) - s.quantile(0.25),
        ).reset_index()
        st.markdown("**Descriptive pace in the selected window**")
        st.dataframe(stats.rename(columns={"EligibleLaps": "Eligible laps", "MedianSeconds": "Median (s)", "IQRSeconds": "IQR (s)"}), hide_index=True, width="stretch")
        st.caption("These medians use each driver's own eligible laps. Use Tyre stints for a comparison on shared race laps.")


def show_stints(scope: pd.DataFrame, reference: str | None) -> None:
    st.subheader("Tyre stints")
    st.caption("Reported tyre compounds and stint boundaries within the selected race-lap window.")
    st.plotly_chart(timeline_chart(scope), width="stretch", key="stint_timeline")
    summary = stint_summary(scope)
    display = pd.DataFrame({
        "Driver": summary["Driver"], "Stint": summary["Stint"],
        "Compound": summary["Compound"],
        "Race laps": summary["LapStart"].map(lambda x: f"{x:.0f}") + "–" + summary["LapEnd"].map(lambda x: f"{x:.0f}"),
        "Eligible / recorded": summary["EligibleLaps"].astype(str) + " / " + summary["RecordedLaps"].astype(str),
        "Median (s)": summary["MedianLapTimeSeconds"], "IQR (s)": summary["IQRSeconds"],
        "Tyre age (laps)": summary["TyreLifeMin"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "?") + "–" + summary["TyreLifeMax"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "?"),
    })
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("Stint summaries are descriptive. The drivers' stint lengths and sampled race laps differ.")
    drivers = sorted(scope["Driver"].unique())
    if len(drivers) != 2:
        st.info("Select both drivers to compare their pace on shared race laps.")
        return
    comparison = next(driver for driver in drivers if driver != reference)
    st.markdown("**Compare pace on the same race laps**")
    same_compound = st.checkbox("Require the same tyre compound", value=True, key="same_compound")
    pairs = matched_laps(scope, reference=reference, comparison=comparison, same_compound=same_compound)
    sign_text = f"Delta = {comparison} − {reference}. Negative values mean {comparison} was faster on that lap."
    st.caption(sign_text)
    if pairs.empty:
        st.info("No shared eligible laps satisfy this selection. Widen the window or inspect the compound requirement.")
        return
    cols = st.columns(3)
    cols[0].metric("Matched race laps", len(pairs))
    cols[1].metric("Median paired delta", f"{pairs['DeltaSeconds'].median():+.3f} s")
    cols[2].metric("Delta IQR", f"{pairs['DeltaSeconds'].quantile(.75) - pairs['DeltaSeconds'].quantile(.25):.3f} s")
    if not same_compound:
        st.warning("Different compounds are allowed in this comparison; tyre context differs.")
    st.plotly_chart(paired_chart(pairs, reference, comparison), width="stretch", key="paired_pace")
    st.caption("Matching controls race lap and, when enabled, compound. Tyre age, fuel, traffic and pace management may still differ. IQR describes variation, not statistical confidence.")
    st.markdown("**Contiguous comparison windows**")
    st.caption("Each window uses one stint pair and compound pair. Gaps in matched laps start a new window.")
    segments = segment_summary(pairs)
    windows = pd.DataFrame({
        "Race laps": segments["LapStart"].map(lambda x: f"{x:.0f}") + "–" + segments["LapEnd"].map(lambda x: f"{x:.0f}"),
        f"{reference} stint": segments["ReferenceStint"],
        f"{comparison} stint": segments["ComparisonStint"],
        "Compounds": segments["ReferenceCompound"].fillna("?") + " / " + segments["ComparisonCompound"].fillna("?"),
        "Matched laps": segments["MatchedLaps"],
        "Median delta (s)": segments["MedianDeltaSeconds"],
        "Delta IQR (s)": segments["IQRDeltaSeconds"],
    })
    st.dataframe(windows, hide_index=True, width="stretch")
    st.download_button(
        "Download paired laps", data=pairs.to_csv(index=False),
        file_name="pitwall_paired_laps.csv", mime="text/csv", key="download_pairs",
    )


def show_audit(scope: pd.DataFrame, checks: pd.DataFrame) -> None:
    st.subheader("Lap audit")
    st.caption("Every record is retained. A lap enters pace analysis only when it passes every eligibility rule.")
    totals = checks.loc[scope.index].sum()
    reasons = pd.DataFrame({
        "Exclusion rule": [REASON_LABELS[key] for key in totals.index],
        "Flagged laps": totals.to_numpy(),
    })
    st.dataframe(reasons, hide_index=True, width="stretch")
    st.caption("One lap can fail several rules, so the counts above overlap.")
    excluded_only = st.checkbox("Show excluded laps only", value=True, key="excluded_only")
    rows = scope[~scope["Included"]] if excluded_only else scope
    st.dataframe(
        rows[["Driver", "LapNumber", "LapTimeSeconds", "Compound", "Stint",
              "TyreLife", "TrackStatus", "Included", "ExclusionReasons"]],
        hide_index=True, width="stretch",
    )
    with st.expander("Read the eligibility rules"):
        for key, text in RULES.items():
            st.markdown(f"**{REASON_LABELS[key]}:** {text}")
    st.caption("Timing quality and green flags do not establish clear air or an error-free driving lap.")


def main() -> None:
    st.caption("PITWALL / PUBLIC DATA LAB")
    st.title("Race pace, with evidence.")
    study = st.radio("Study", ["Bahrain case study", "Across weekends"], horizontal=True, key="study")
    if study == "Across weekends":
        show_weekends()
        return
    st.markdown("**2024 Bahrain Grand Prix** · Charles Leclerc & Carlos Sainz · historical race analysis")
    if not INPUT.exists():
        st.error("The lap CSV is missing. Run download_data.py in the project folder first.")
        st.stop()
    try:
        data, checks = load_data(str(INPUT), INPUT.stat().st_mtime_ns)
    except (ValueError, OSError) as error:
        st.error(f"Unable to load the lap export: {error}")
        st.stop()

    with st.sidebar:
        st.subheader("Analysis window")
        st.caption("Bahrain · 2024 · Race")
        drivers = st.multiselect(
            "Drivers", options=["LEC", "SAI"], default=["LEC", "SAI"],
            format_func=lambda driver: f"{driver} · {NAMES[driver]}", key="drivers",
        )
        lap_range = st.slider(
            "Race-lap window", min_value=1, max_value=int(data["LapNumber"].max()),
            value=(1, int(data["LapNumber"].max())), key="lap_range",
        )
        cohort = st.selectbox(
            "Race-pace cohort", ["Analysis-eligible laps", "All recorded laps"], key="cohort",
        )
        reference = st.selectbox("Reference driver", options=drivers, key="reference") if len(drivers) == 2 else None
        st.divider()
        st.caption("SOURCE · Public timing exported with FastF1")
        st.caption("The dashboard reads the local CSV. No live feed or team-private measurements.")
        st.caption("Reported tyre age is lap usage; it does not measure wear.")
    if not drivers:
        st.info("Select at least one driver in the sidebar.")
        st.stop()
    scope = data[
        data["Driver"].isin(drivers) & data["LapNumber"].between(*lap_range)
    ]
    selected = len(scope)
    eligible_count = int(scope["Included"].sum())
    cols = st.columns(3)
    cols[0].metric("Selected lap records", selected)
    cols[1].metric("Analysis-eligible laps", eligible_count)
    cols[2].metric("Excluded laps", selected - eligible_count)
    st.caption(f"Selection: laps {lap_range[0]}–{lap_range[1]} · {', '.join(drivers)}")
    view = st.radio(
        "Analysis view", ["Race pace", "Tyre stints", "Lap audit", "Telemetry", "ML pace"],
        horizontal=True, key="view",
    )
    if view == "Race pace":
        show_pace(scope, cohort)
    elif view == "Tyre stints":
        show_stints(scope, reference)
    elif view == "Lap audit":
        show_audit(scope, checks)
    elif view == "Telemetry":
        show_telemetry(scope, reference)
    else:
        show_model(scope)
    st.divider()
    st.download_button(
        "Download selected lap audit", data=scope.to_csv(index=False),
        file_name="pitwall_selected_laps.csv", mime="text/csv", key="download_audit",
    )
    with st.expander("What this analysis can establish"):
        st.write(
            "Public lap timing supports observed pace comparisons. Filtering makes the "
            "selection auditable; matching race lap and compound reduces some comparison differences."
        )
        st.write(
            "Actual fuel mass, tyre wear and temperatures, brake pressure and engineering setup "
            "are unavailable here. A pace trend cannot isolate their individual effects."
        )
        st.write("The ML pace view contains a conditional historical forecast pilot. Causal tyre-degradation estimates and team affiliation are not claimed.")


if __name__ == "__main__":
    main()
