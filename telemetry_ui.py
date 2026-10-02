"""Selected native telemetry, provenance checks and cautious distance alignment."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analyze_data import INPUT
from race_analysis import matched_laps
from telemetry_analysis import DISTANCE_LABEL, align_laps_by_distance, prepare_lap_telemetry

ROOT = Path(__file__).resolve().parent
TELEMETRY = ROOT / 'data' / 'bahrain_2024_ferrari_telemetry.csv'
MANIFEST = ROOT / 'data' / 'telemetry_manifest.json'
SHORT_DISTANCE_LABEL = 'Estimated sample-origin distance (m)'
COLORS = {'LEC': '#69a9ff', 'SAI': '#ffa96b'}


@st.cache_data(show_spinner=False)
def load_telemetry(path, modified_ns, manifest_modified_ns, lap_modified_ns):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if sha256(INPUT.read_bytes()).hexdigest() != manifest['source_lap_sha256']:
        raise ValueError('Lap CSV changed after the telemetry export. Regenerate export_telemetry.py.')
    if sha256(Path(path).read_bytes()).hexdigest() != manifest['telemetry_sha256']:
        raise ValueError('Telemetry CSV does not match its provenance manifest. Regenerate export_telemetry.py.')
    return pd.read_csv(path), manifest


def lap_select(label, options, key):
    options = [int(value) for value in options]
    default = 21 if 21 in options else options[len(options) // 2]
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return st.selectbox(label, options, index=options.index(default), key=key)


def native_chart(prepared, axis):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.075)
    x_column = 'EstimatedDistanceM' if axis == 'Estimated distance' else 'ElapsedSeconds'
    for lap in prepared:
        data = lap.samples
        # A separator prevents a visual line from crossing a sampling outage.
        rows = []
        for index, row in data.iterrows():
            if index and row['GapBefore']:
                rows.append({x_column: None, 'SpeedKmh': None, 'ThrottlePct': None, 'BrakeOn': None})
            rows.append(row.to_dict())
        plotted = pd.DataFrame(rows)
        label = f'{lap.quality.driver} · lap {lap.quality.lap_number}'
        for position, channel in enumerate(['SpeedKmh', 'ThrottlePct', 'BrakeOn'], start=1):
            fig.add_trace(go.Scatter(
                x=plotted[x_column], y=plotted[channel], mode='lines',
                name=label, legendgroup=label, showlegend=position == 1,
                connectgaps=False,
                line=dict(color=COLORS[lap.quality.driver], width=2,
                          shape='hv' if channel == 'BrakeOn' else 'linear'),
                hovertemplate=f'{label}<br>%{{x:.2f}}<br>{channel}: %{{y:.2f}}<extra></extra>',
            ), row=position, col=1)
    fig.update_yaxes(title_text='Speed (km/h)', row=1, col=1)
    fig.update_yaxes(title_text='Throttle (%)', range=[-3, 103], row=2, col=1)
    fig.update_yaxes(title_text='Brake', tickvals=[0, 1], ticktext=['Off', 'On'], range=[-.1, 1.1], row=3, col=1)
    fig.update_xaxes(title_text=SHORT_DISTANCE_LABEL if axis == 'Estimated distance' else 'Elapsed time since lap start (s)', row=3, col=1)
    fig.update_layout(template='plotly_dark', height=690, margin=dict(l=15, r=20, t=45, b=40),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Arial, sans-serif', size=12),
                      legend=dict(orientation='h', y=1.08, x=0), hovermode='x unified')
    fig.update_xaxes(gridcolor='rgba(220,230,245,0.08)')
    fig.update_yaxes(gridcolor='rgba(220,230,245,0.08)')
    return fig


def show_telemetry(scope, reference):
    st.subheader('Selected-lap telemetry')
    st.caption('Native public car samples: speed, throttle and brake on/off. Telemetry selection uses analysis-eligible laps in the sidebar window.')
    if not TELEMETRY.exists() or not MANIFEST.exists():
        st.info('Telemetry export is missing. Run export_telemetry.py after downloading the race session.')
        return
    try:
        samples, manifest = load_telemetry(str(TELEMETRY), TELEMETRY.stat().st_mtime_ns,
                                          MANIFEST.stat().st_mtime_ns, INPUT.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Unable to load the telemetry export: {error}')
        return
    available = samples[['Driver', 'LapNumber']].drop_duplicates()
    eligible = scope[scope['Included']].merge(available, on=['Driver', 'LapNumber'], how='inner')
    drivers = sorted(scope['Driver'].unique())
    if eligible.empty or any(not eligible['Driver'].eq(driver).any() for driver in drivers):
        st.info('No analysis-eligible telemetry laps for every selected driver. Widen the race-lap window or select one driver.')
        return
    selected = {}
    if len(drivers) == 2:
        comparison = next(driver for driver in drivers if driver != reference)
        ordered = [reference, comparison]
        mode = st.radio('Telemetry lap selection', ['Same race lap', 'Choose laps separately'], horizontal=True, key='telemetry_mode')
        if mode == 'Same race lap':
            pairs = matched_laps(eligible, reference=reference, comparison=comparison, same_compound=True)
            if pairs.empty:
                st.info('No shared eligible telemetry laps with matching compounds. Choose laps separately or widen the window.')
                return
            number = lap_select('Shared race lap (same compound)', pairs['LapNumber'], 'telemetry_shared_lap')
            selected = {driver: number for driver in ordered}
            st.caption('Shared race lap and compound reduce context differences. Initial lap 21 was chosen for continuous speed support under the 1 s threshold; all shared eligible laps remain selectable. Reported tyre age, traffic and pace management can still differ.')
        else:
            for driver in ordered:
                selected[driver] = lap_select(f'{driver} race lap', eligible.loc[eligible['Driver'].eq(driver), 'LapNumber'], f'telemetry_lap_{driver}')
            st.warning('Separately selected laps can differ in race phase, compound and tyre age. Inspect the context below before comparing traces.')
    else:
        ordered = drivers
        driver = drivers[0]
        selected[driver] = lap_select(f'{driver} race lap', eligible['LapNumber'], f'telemetry_lap_{driver}')
    prepared, context = [], []
    try:
        for driver in ordered:
            record = eligible.loc[eligible['Driver'].eq(driver) & eligible['LapNumber'].eq(selected[driver])].iloc[0]
            lap = prepare_lap_telemetry(samples.loc[samples['Driver'].eq(driver) & samples['LapNumber'].eq(selected[driver])],
                                       driver=driver, lap_number=selected[driver], lap_time_seconds=record['LapTimeSeconds'])
            prepared.append(lap)
            context.append({'Driver': driver, 'Race lap': selected[driver], 'Lap time (s)': record['LapTimeSeconds'],
                            'Compound': record['Compound'], 'Stint': record['Stint'], 'Tyre age (laps)': record['TyreLife']})
    except ValueError as error:
        st.error(f'Cannot prepare selected telemetry: {error}')
        return
    st.dataframe(pd.DataFrame(context), hide_index=True, width='stretch')
    if len(prepared) == 2:
        delta = context[1]['Lap time (s)'] - context[0]['Lap time (s)']
        st.caption(f'Observed lap-time difference: {ordered[1]} − {ordered[0]} = {delta:+.3f} s. This comes from lap timing, not integrated telemetry.')
    axis = st.radio('Telemetry horizontal axis', ['Estimated distance', 'Elapsed time'], horizontal=True, key='telemetry_axis')
    if axis == 'Estimated distance':
        st.caption('Distance is a trapezoidal speed integral, zero at each lap\'s first recorded sample. Missing boundary samples are not filled. The origins and driven paths can differ; this is an approximate trace overlay, not exact circuit-position alignment.')
    else:
        st.caption('Each trace uses seconds since its reported lap start. Drivers reach corners at different times, so equal elapsed time does not mean equal track position.')
    if axis == 'Estimated distance' and any(not lap.quality.alignment_eligible for lap in prepared):
        st.warning('Distance traces are incomplete for this selection. Switch to elapsed time to inspect samples after gaps; aligned speed comparison will be withheld.')
    st.plotly_chart(native_chart(prepared, axis), width='stretch', config={'displayModeBar': False}, key='telemetry_native')
    st.caption('Lines connect irregular native samples; brake is drawn as a step. Sparse sampling and sensor resolution limit exact braking-point or control-input claims.')
    st.markdown('**Sampling quality**')
    quality = pd.DataFrame([{
        'Driver': lap.quality.driver, 'Samples': lap.quality.sample_count,
        'Median interval (s)': lap.quality.median_sample_interval_seconds,
        'Largest interval (s)': lap.quality.max_sample_interval_seconds,
        'Start missing (s)': lap.quality.start_missing_seconds,
        'End missing (s)': lap.quality.end_missing_seconds,
        'Gaps > 1 s': lap.quality.gap_count,
        'Invalid throttle samples': lap.quality.invalid_throttle_samples,
        'Distance supported': lap.quality.alignment_eligible,
    } for lap in prepared])
    st.dataframe(quality, hide_index=True, width='stretch')
    st.caption('A 1.0 s gap and boundary tolerance is an explicit analysis choice. Missing or invalid channel values remain gaps; speed gaps disable later distance estimates.')
    if len(prepared) == 2 and st.checkbox('Show aligned speed difference', value=False, key='telemetry_delta'):
        try:
            aligned = align_laps_by_distance(prepared[0], prepared[1], spacing_m=10.0)
        except ValueError as error:
            st.info(f'Distance comparison withheld: {error}')
        else:
            fig = go.Figure(go.Scatter(x=aligned['EstimatedDistanceM'], y=aligned['SpeedDifferenceKmh'],
                                      mode='lines', connectgaps=False, name=f'{ordered[1]} − {ordered[0]}',
                                      line=dict(color=COLORS[ordered[1]], width=2)))
            fig.add_hline(y=0, line_dash='dash')
            fig.update_layout(template='plotly_dark', height=330, margin=dict(l=15, r=20, t=30, b=35),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis_title=SHORT_DISTANCE_LABEL, yaxis_title=f'{ordered[1]} − {ordered[0]} speed (km/h)')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False}, key='telemetry_speed_delta')
            st.caption('Derived 10 m grid over common observed distance only, with a final endpoint. Speed/throttle are linearly interpolated; brake uses preceding samples. This is not a time gain/loss or corner-performance measure.')
            export = aligned.rename(columns={column: column.replace('Left', ordered[0]).replace('Right', ordered[1]) for column in aligned.columns})
            st.download_button('Download aligned telemetry', export.to_csv(index=False), 'pitwall_aligned_telemetry.csv', 'text/csv', key='download_aligned_telemetry')
    st.download_button('Download selected native samples and estimates', pd.concat([lap.samples for lap in prepared]).to_csv(index=False),
                       'pitwall_selected_telemetry.csv', 'text/csv', key='download_telemetry')
    with st.expander('Telemetry provenance and full quality diagnostics'):
        st.write(f'FastF1 {manifest["fastf1_version"]} · cached Bahrain 2024 race · {manifest["sample_count"]:,} native samples across {manifest["lap_count"]} exported laps.')
        st.write('Native car channels were sliced without interpolated boundary samples or merged position data. CSV and source-lap hashes are checked when loaded.')
        st.json([asdict(lap.quality) for lap in prepared])
        st.caption('Observed time fraction is first-to-last sample span divided by lap time; internal gaps can fall inside that span. No private brake pressure, fuel mass, setup or tyre-temperature data is available.')
