"""Independent dashboard for the fixed, whole-weekend model evaluation."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_ui import prediction_chart
from weekend_model import DEFAULT_DRIVERS, score_weekend_predictions

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports' / 'weekends'
SOURCE = ROOT / 'data' / 'weekends'
FILES = ('lap_audit.csv', 'examples.csv', 'predictions.csv', 'metrics.csv', 'coverage.csv',
         'rule_counts.csv', 'alpha_validation.csv', 'domain_checks.csv', 'equal_weekend_metrics.csv')
NAMES = {'LEC': 'Charles Leclerc', 'SAI': 'Carlos Sainz', 'HAM': 'Lewis Hamilton',
         'RUS': 'George Russell', 'VER': 'Max Verstappen', 'PER': 'Sergio Perez'}
LABELS = {'Ridge': 'Ridge model', 'Persistence': 'Last lap', 'Trailing median': 'Recent median'}


@st.cache_data(show_spinner=False)
def load_weekend_reports(path, modified_ns):
    directory = Path(path)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if sha256((SOURCE / 'source_manifest.json').read_bytes()).hexdigest() != manifest['source_manifest_sha256']:
        raise ValueError('Acquisition manifest changed after evaluation. Regenerate train_weekends.py.')
    for event in manifest['sources']:
        if sha256((SOURCE / f'{event["EventKey"]}_laps.csv').read_bytes()).hexdigest() != event['sha256']:
            raise ValueError(f'Source timing changed for {event["EventName"]}. Regenerate the fixed study.')
    if sha256((ROOT / 'WEEKEND_PROTOCOL.md').read_bytes()).hexdigest() != manifest['code_sha256']['WEEKEND_PROTOCOL.md']:
        raise ValueError('Fixed weekend protocol changed after evaluation.')
    frames = {}
    for name in FILES:
        file = directory / name
        if sha256(file.read_bytes()).hexdigest() != manifest['artifacts'][name]['sha256']:
            raise ValueError(f'{name} differs from the saved evaluation; regenerate train_weekends.py.')
        frames[name] = pd.read_csv(file)
    if sha256((ROOT / 'models' / 'weekend_ridge.joblib').read_bytes()).hexdigest() != manifest['model']['sha256']:
        raise ValueError('Saved weekend model differs from its evaluation.')
    return manifest, frames


def select_valid(label, options, key, format_func=None):
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return st.selectbox(label, options, key=key, format_func=format_func or str)


def show_weekends():
    st.subheader('Whole-weekend model evaluation')
    st.caption('2024 races · Ferrari, Mercedes and Red Bull drivers · separate experiment with complete race weekends held out.')
    paths = [REPORTS / 'manifest.json'] + [REPORTS / name for name in FILES] + [SOURCE / 'source_manifest.json', ROOT / 'WEEKEND_PROTOCOL.md', ROOT / 'models' / 'weekend_ridge.joblib']
    if any(not path.exists() for path in paths):
        st.info('Weekend evaluation is missing. Run download_weekends.py, then train_weekends.py in G:\\PitWall.')
        return
    try:
        source = json.loads((SOURCE / 'source_manifest.json').read_text(encoding='utf-8'))
        paths += [SOURCE / f'{event["EventKey"]}_laps.csv' for event in source['events']]
        manifest, frames = load_weekend_reports(str(REPORTS), tuple(path.stat().st_mtime_ns for path in paths))
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Unable to load weekend results: {error}')
        return
    protocol, predictions = manifest['protocol'], frames['predictions.csv']
    cols = st.columns(4)
    for column, label, value in zip(cols, ['Race weekends', 'Training targets', 'Validation targets', 'Test targets'],
                                   [len(protocol['events']), protocol['counts']['train'], protocol['counts']['validation'], protocol['counts']['test']]):
        column.metric(label, value)
    event_table = pd.DataFrame(protocol['events'])
    st.dataframe(event_table[['EventName', 'Role', 'recorded_laps', 'eligible_laps', 'examples']].rename(columns={
        'EventName': 'Weekend', 'Role': 'Role', 'recorded_laps': 'Recorded laps', 'eligible_laps': 'Eligible laps', 'examples': 'Next-lap targets'}), hide_index=True, width='stretch')
    st.caption(f'Alpha {protocol["selected_alpha"]:g} was selected on China only. The model and preprocessing were fitted on Bahrain, Australia and Japan, then frozen for China, Miami and Imola replay. The original Bahrain pilot is preserved.')
    st.warning('This retrospective evaluation scores actual consecutive laps that passed the final audit and kept the same stint/compound. It does not predict future eligibility, pit stops or incidents. Earlier observed race laps update history while model parameters stay frozen.')
    phase = st.radio('Weekend evaluation window', ['Test weekends', 'Validation weekend'], horizontal=True, key='weekend_phase')
    split = 'test' if phase == 'Test weekends' else 'validation'
    choices = event_table.loc[event_table['Role'].eq(split)].sort_values('RoundNumber')
    event_names = choices.set_index('EventKey')['EventName'].to_dict()
    keys = choices['EventKey'].tolist()
    with st.sidebar:
        st.subheader('Weekend evaluation filters')
        st.caption('These controls are independent of the Bahrain case-study filters.')
        drivers = st.multiselect('Drivers (weekend study)', list(DEFAULT_DRIVERS), default=list(DEFAULT_DRIVERS),
                                 format_func=lambda driver: f'{driver} · {NAMES[driver]}', key='weekend_drivers')
        # Reset weekend selection on phase changes; do not retain invisible test IDs.
        if st.session_state.get('_weekend_filter_phase') != split:
            st.session_state['weekend_events'] = keys
            st.session_state['_weekend_filter_phase'] = split
        events = st.multiselect('Held-out weekends', keys, format_func=lambda key: event_names[key], key='weekend_events')
        st.caption('Changing filters recomputes scores on saved predictions; it does not retrain the model.')
    selected = predictions.loc[predictions['Split'].eq(split) & predictions['Driver'].isin(drivers) & predictions['EventKey'].isin(events)]
    if selected.empty:
        st.info('Select at least one held-out weekend and a driver with scored targets.')
        return
    metrics = score_weekend_predictions(selected)
    pooled = metrics.loc[metrics['EventKey'].eq('ALL') & metrics['Driver'].eq('ALL')].set_index('Model')
    cols = st.columns(4)
    cols[0].metric('Scored weekend targets', len(selected))
    cols[1].metric('Weekend Ridge MAE', f'{pooled.loc["Ridge", "MAESeconds"]:.3f} s')
    cols[2].metric('Weekend last-lap MAE', f'{pooled.loc["Persistence", "MAESeconds"]:.3f} s')
    cols[3].metric('Weekend median MAE', f'{pooled.loc["Trailing median", "MAESeconds"]:.3f} s')
    st.caption('These pooled scores weight every target equally. Weekends and drivers with more eligible targets contribute more.')
    st.markdown('**Each held-out weekend**')
    event_scores = metrics.loc[metrics['EventKey'].ne('ALL') & metrics['Driver'].eq('ALL')]
    table = event_scores.pivot(index=['EventKey', 'EventName', 'Count'], columns='Model', values='MAESeconds').reset_index()
    table['Better baseline'] = table[['Persistence', 'Trailing median']].min(axis=1)
    table['Ridge minus baseline (s)'] = table['Ridge'] - table['Better baseline']
    st.dataframe(table[['EventName', 'Count', 'Ridge', 'Persistence', 'Trailing median', 'Ridge minus baseline (s)']].rename(columns={
        'EventName': 'Weekend', 'Count': 'Targets', 'Ridge': 'Ridge MAE (s)', 'Persistence': 'Last-lap MAE (s)', 'Trailing median': 'Recent-median MAE (s)'}), hide_index=True, width='stretch')
    for _, row in table.iterrows():
        if row['Ridge minus baseline (s)'] > 0:
            st.caption(f'{row["EventName"]}: Ridge loses to the better baseline by {row["Ridge minus baseline (s)"]:.3f} s MAE in this selection.')
    macro = event_scores.groupby('Model')['MAESeconds'].mean()
    st.caption(f'Equal-weekend mean MAE over {len(table)} selected weekend(s): Ridge {macro["Ridge"]:.3f} s, last lap {macro["Persistence"]:.3f} s, recent median {macro["Trailing median"]:.3f} s. Each weekend receives equal weight here.')
    st.markdown('**Driver results in the selected weekends**')
    driver_scores = metrics.loc[metrics['EventKey'].eq('ALL') & metrics['Driver'].ne('ALL')]
    st.dataframe(driver_scores[['Driver', 'Model', 'Count', 'MAESeconds', 'RMSESeconds', 'BiasSeconds']].rename(columns={
        'Model': 'Method', 'Count': 'Targets', 'MAESeconds': 'MAE (s)', 'RMSESeconds': 'RMSE (s)', 'BiasSeconds': 'Bias (s)'}), hide_index=True, width='stretch')
    losing_drivers = []
    for code, group in driver_scores.groupby('Driver'):
        scores = group.set_index('Model')['MAESeconds']
        if scores['Ridge'] > scores.loc[['Persistence', 'Trailing median']].min():
            losing_drivers.append(code)
    if losing_drivers:
        st.caption('Ridge loses to a baseline for ' + ', '.join(losing_drivers) + ' in this selection. Pooled results can hide these failures.')
    st.caption('Lower MAE/RMSE is better. Bias is prediction minus observed; positive means predicted slower. Adjacent laps are correlated. Two test weekends with recurring drivers are limited evidence of race transfer.')
    event_key = select_valid('Replay weekend', table['EventKey'].tolist(), 'weekend_replay_event', event_names.get)
    available_drivers = sorted(selected.loc[selected['EventKey'].eq(event_key), 'Driver'].unique())
    driver = select_valid('Replay driver (weekend study)', available_drivers, 'weekend_replay_driver')
    replay = selected.loc[selected['EventKey'].eq(event_key) & selected['Driver'].eq(driver)].sort_values('TargetLapNumber')
    st.plotly_chart(prediction_chart(replay), width='stretch', config={'displayModeBar': False}, key='weekend_predictions_chart')
    target = select_valid('Inspect target lap (weekend study)', replay['TargetLapNumber'].astype(int).tolist(), 'weekend_target_lap')
    row = replay.loc[replay['TargetLapNumber'].eq(target)].iloc[0]
    st.markdown(f'**{event_names[event_key]} · {driver}: after lap {int(row["OriginLapNumber"])}, predict lap {target}**')
    methods = [('Ridge model', row['RidgeSeconds']), ('Last-lap baseline', row['PersistenceSeconds']), ('Recent-median baseline', row['TrailingMedianSeconds'])]
    st.dataframe(pd.DataFrame([{'Method': name, 'Prediction (s)': value, 'Observed (s)': row['ActualSeconds'], 'Error (s)': value - row['ActualSeconds']} for name, value in methods]), hide_index=True, width='stretch')
    st.download_button('Download weekend predictions', selected.to_csv(index=False), 'pitwall_weekend_predictions.csv', 'text/csv', key='download_weekend_predictions')
    with st.expander('Data coverage, feature checks and provenance'):
        st.write(f'{manifest["recorded_laps"]:,} recorded laps, {manifest["eligible_laps"]:,} timing-eligible. Exclusions remain in the audit. Event dates establish chronology; absolute per-lap UTC dates were not loaded by this timing-only acquisition.')
        st.dataframe(frames['coverage.csv'], hide_index=True, width='stretch')
        st.caption('Driver retirements produce fewer recorded laps and usable targets. Timing eligibility does not establish traffic-free driving.')
        st.dataframe(frames['domain_checks.csv'], hide_index=True, width='stretch')
        st.caption('Unseen categories use the training-fitted encoder\'s all-zero fallback; numeric inputs outside training ranges may extrapolate. No feature checks change the frozen test cohort.')
        st.dataframe(frames['alpha_validation.csv'], hide_index=True, width='stretch')
        st.write('Source timing, audit/evaluation exports, fixed protocol and saved model hashes are verified. The dashboard reads historical predictions and never deserializes the model or trains on interaction.')
        st.caption('Public tyre age and race lap do not measure tyre wear or fuel. Post-race flags do not establish live availability. No causal strategy, calibrated interval, unseen-driver or complete-season claim is made.')
        st.json({'versions': manifest['versions'], 'original_pilot_unchanged': manifest['original_pilot_unchanged'], 'model': manifest['model']})
        st.download_button('Download full weekend lap audit', frames['lap_audit.csv'].to_csv(index=False), 'pitwall_weekend_lap_audit.csv', 'text/csv', key='download_weekend_audit')
