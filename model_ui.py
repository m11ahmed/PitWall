"""Read-only replay of the frozen, chronologically evaluated pace pilot."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyze_data import INPUT
from pace_model import score_predictions

ROOT = Path(__file__).resolve().parent
ML_REPORTS = ROOT / 'reports' / 'ml'
FILES = ('examples.csv', 'predictions.csv', 'metrics.csv', 'alpha_validation.csv')
MODEL_NAMES = {'Ridge': 'Ridge model', 'Persistence': 'Last-lap baseline', 'Trailing median': 'Recent-median baseline'}


@st.cache_data(show_spinner=False)
def load_model_reports(path, source_modified_ns, modified_ns):
    directory = Path(path)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(INPUT.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('Source timing changed after training. Regenerate train_model.py before replaying results.')
    frames = {}
    for name in FILES:
        file = directory / name
        if sha256(file.read_bytes()).hexdigest() != manifest['artifacts'][name]['sha256']:
            raise ValueError(f'{name} does not match the saved experiment manifest. Regenerate train_model.py.')
        frames[name] = pd.read_csv(file)
    model_file = ROOT / 'models' / 'pace_ridge.joblib'
    if sha256(model_file.read_bytes()).hexdigest() != manifest['model']['sha256']:
        raise ValueError('Saved model does not match this evaluation. Regenerate train_model.py.')
    return manifest, frames


def prediction_chart(rows):
    dense = rows.set_index('TargetLapNumber').reindex(range(int(rows['TargetLapNumber'].min()), int(rows['TargetLapNumber'].max()) + 1))
    fig = go.Figure()
    for column, name, color, dash in (
        ('ActualSeconds', 'Observed', '#e7edf5', 'solid'),
        ('RidgeSeconds', 'Ridge model', '#ff9f69', 'solid'),
        ('PersistenceSeconds', 'Last lap', '#68adff', 'dot'),
        ('TrailingMedianSeconds', 'Recent median', '#9dcc88', 'dash'),
    ):
        fig.add_trace(go.Scatter(x=dense.index, y=dense[column], mode='lines+markers', name=name,
                                connectgaps=False, line=dict(color=color, dash=dash, width=2),
                                marker=dict(size=5), hovertemplate=f'{name}<br>Target lap %{{x:.0f}}<br>%{{y:.3f}} s<extra></extra>'))
    fig.update_layout(template='plotly_dark', height=380, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=15, r=15, t=75, b=40),
                      legend=dict(orientation='h', y=1.22, x=0), hovermode='x unified',
                      xaxis_title='Target race lap t+1', yaxis_title='Lap time (s)')
    return fig


def show_model(scope):
    st.subheader('Next-lap model: historical replay')
    st.caption('A frozen Ridge model predicts the actual next lap from timing and metadata through the completed lap. This is a retrospective, conditional experiment using one race and two drivers.')
    paths = [ML_REPORTS / 'manifest.json'] + [ML_REPORTS / name for name in FILES] + [ROOT / 'models' / 'pace_ridge.joblib']
    if any(not path.exists() for path in paths):
        st.info('Model artifacts are missing. Run train_model.py in G:\\PitWall to generate the offline experiment.')
        return
    try:
        manifest, frames = load_model_reports(str(ML_REPORTS), INPUT.stat().st_mtime_ns, tuple(path.stat().st_mtime_ns for path in paths))
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Unable to load ML results: {error}')
        return
    examples, predictions = frames['examples.csv'], frames['predictions.csv']
    counts = manifest['split_counts']
    st.markdown('**One shared chronological split for both drivers**')
    cols = st.columns(4)
    for column, label, key in zip(cols, ['Training pairs', 'Validation pairs', 'Test pairs', 'Boundary pairs omitted'], ['train', 'validation', 'test', 'purged']):
        column.metric(label, counts.get(key, 0))
    alpha = manifest['protocol']['selected_alpha']
    st.caption(f'Ridge alpha {alpha:g} was chosen using validation MAE only. Scaling, category encoding and model fitting use the training window; the selected fit stays frozen. Both held-out windows contain HARD tyres.')
    st.warning('Scoring includes only consecutive laps that both passed the final audit and stayed in the same stint and compound. Future eligibility is unknown at prediction time. Pit stops, incidents and flagged laps are outside this scored cohort.')
    phase = st.radio('Evaluation window', ['Final test', 'Validation (used to choose alpha)'], horizontal=True, key='ml_phase')
    split = 'test' if phase == 'Final test' else 'validation'
    drivers = sorted(scope['Driver'].unique())
    low, high = int(scope['LapNumber'].min()), int(scope['LapNumber'].max())
    selected = predictions.loc[predictions['Split'].eq(split) & predictions['Driver'].isin(drivers) & predictions['TargetLapNumber'].between(low, high)]
    st.caption('Sidebar filters apply to target laps t+1 and drivers. They do not retrain the model. Earlier observed laps can supply history for a target inside the selected window.')
    if selected.empty:
        st.info('No held-out targets in this selection. Widen the race-lap window or switch evaluation window.')
        return
    metrics = score_predictions(selected)
    overall = metrics.loc[metrics['Driver'].eq('ALL')].set_index('Model')
    cols = st.columns(4)
    cols[0].metric('Scored targets', len(selected))
    cols[1].metric('Ridge MAE', f'{overall.loc["Ridge", "MAESeconds"]:.3f} s')
    cols[2].metric('Last-lap MAE', f'{overall.loc["Persistence", "MAESeconds"]:.3f} s')
    cols[3].metric('Recent-median MAE', f'{overall.loc["Trailing median", "MAESeconds"]:.3f} s')
    baseline = overall.loc[['Persistence', 'Trailing median'], 'MAESeconds'].min()
    difference = baseline - overall.loc['Ridge', 'MAESeconds']
    if difference > 1e-9:
        st.info(f'In this selected {split} cohort, Ridge MAE is {difference:.3f} s lower than the better baseline. This small within-race result does not establish performance on another weekend.')
    elif difference < -1e-9:
        st.warning(f'In this selected {split} cohort, Ridge MAE is {-difference:.3f} s higher than the better baseline. The baseline performs better here.')
    else:
        st.info('Ridge and the better baseline have the same MAE in this selected cohort.')
    display = metrics.rename(columns={'Model': 'Method', 'Driver': 'Cohort', 'Count': 'Targets', 'MAESeconds': 'MAE (s)', 'RMSESeconds': 'RMSE (s)', 'BiasSeconds': 'Bias (s)'})
    display['Method'] = display['Method'].map(MODEL_NAMES)
    st.dataframe(display[['Cohort', 'Method', 'Targets', 'MAE (s)', 'RMSE (s)', 'Bias (s)']], hide_index=True, width='stretch')
    st.caption('MAE is average absolute prediction error; RMSE emphasizes larger misses. Bias = prediction minus observed time; positive means predicted slower. Targets are temporally correlated, so counts are not independent trials.')
    for driver in drivers:
        per_driver = metrics.loc[metrics['Driver'].eq(driver)].set_index('Model')
        if not per_driver.empty and per_driver.loc['Ridge', 'MAESeconds'] > per_driver.loc['Persistence', 'MAESeconds']:
            st.caption(f'{driver}: the last-lap baseline outperforms Ridge in this selection. The pooled result can hide driver differences.')
    available_drivers = sorted(selected['Driver'].unique())
    if 'ml_replay_driver' in st.session_state and st.session_state['ml_replay_driver'] not in available_drivers:
        del st.session_state['ml_replay_driver']
    driver = st.selectbox('Replay driver', available_drivers, key='ml_replay_driver')
    replay = selected.loc[selected['Driver'].eq(driver)].sort_values('TargetLapNumber')
    st.plotly_chart(prediction_chart(replay), width='stretch', config={'displayModeBar': False}, key='ml_predictions_chart')
    targets = replay['TargetLapNumber'].astype(int).tolist()
    if 'ml_target_lap' in st.session_state and st.session_state['ml_target_lap'] not in targets:
        del st.session_state['ml_target_lap']
    target = st.selectbox('Inspect target lap', targets, key='ml_target_lap')
    row = replay.loc[replay['TargetLapNumber'].eq(target)].iloc[0]
    st.markdown(f'**{driver}: after lap {int(row["OriginLapNumber"])}, predict lap {target}**')
    methods = [('Ridge model', row['RidgeSeconds']), ('Last-lap baseline', row['PersistenceSeconds']), ('Recent-median baseline', row['TrailingMedianSeconds'])]
    st.dataframe(pd.DataFrame([{'Method': name, 'Predicted (s)': value, 'Observed (s)': row['ActualSeconds'], 'Error (s)': value - row['ActualSeconds']} for name, value in methods]), hide_index=True, width='stretch')
    with st.expander('Information available at this prediction point'):
        st.dataframe(pd.DataFrame([{'Completed lap': row['OriginLapNumber'], 'Compound': row['Compound'], 'Reported tyre age (laps)': row['TyreLife'], 'Last lap (s)': row['LastLapSeconds'], 'Recent median (s)': row['TrailingMedianSeconds'], 'Observed history laps': row['HistoryCount']}]), hide_index=True, width='stretch')
        st.caption('Driver and compound are categories; reported tyre age and race lap are public metadata. No target timing or private fuel/tyre measurements enter the model. Availability is reconstructed from post-race records; live feed delay is untested.')
    st.download_button('Download selected ML predictions', selected.to_csv(index=False), 'pitwall_ml_predictions.csv', 'text/csv', key='download_ml_predictions')
    with st.expander('Training protocol and saved-model provenance'):
        windows = []
        for key in ['train', 'validation', 'test', 'purged']:
            rows = examples.loc[examples['Split'].eq(key)]
            windows.append({'Window': key, 'Pairs': len(rows), 'Target race laps': f'{int(rows.TargetLapNumber.min())}-{int(rows.TargetLapNumber.max())}' if not rows.empty else '-'})
        st.dataframe(pd.DataFrame(windows), hide_index=True, width='stretch')
        st.dataframe(frames['alpha_validation.csv'].rename(columns={'ValidationMAESeconds': 'Validation MAE (s)'}), hide_index=True, width='stretch')
        st.write('Cutoffs use 60% and 80% of the two-driver source timing span. Pairs crossing a cutoff are omitted. Recent history updates from observed earlier laps; model coefficients never update during validation/test replay.')
        st.write('Saved model: G:\\PitWall\\models\\pace_ridge.joblib. Saved-model prediction roundtrip was verified. The dashboard reads exported predictions; it does not train on interaction or deserialize the model.')
        st.caption('Source timing, evaluation CSV and model hashes are checked. No accuracy percentage, calibrated uncertainty interval, causal degradation estimate or across-race validation is claimed.')
        st.json({'source_sha256': manifest['source_sha256'], 'versions': manifest['versions'], 'split': manifest['split'], 'model': manifest['model']})
