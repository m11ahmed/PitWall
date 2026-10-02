"""Train and evaluate the fixed whole-weekend study entirely from local CSVs."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from pace_model import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from weekend_model import DEFAULT_DRIVERS, EVENT_METADATA, EVENT_ROLES, audit_single_event, run_weekend_experiment

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data' / 'weekends'
REPORTS = ROOT / 'reports' / 'weekends'
MODEL = ROOT / 'models' / 'weekend_ridge.joblib'
PILOT_FILES = ['data/bahrain_2024_ferrari_laps.csv', 'models/pace_ridge.joblib',
               'reports/ml/manifest.json', 'reports/ml/predictions.csv', 'reports/ml/metrics.csv']


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load_sources():
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('complete') is not True or manifest.get('year') != 2024 or manifest.get('session_type') != 'Race':
        raise ValueError('The fixed acquisition is incomplete or has the wrong year/session.')
    if set(manifest['drivers']) != set(DEFAULT_DRIVERS):
        raise ValueError('Acquisition driver cohort differs from the fixed protocol.')
    if manifest['protocol_sha256'] != digest(ROOT / 'WEEKEND_PROTOCOL.md'):
        raise ValueError('Protocol changed after acquisition; keep the fixed study intact.')
    events = manifest['events']
    if len(events) != len(EVENT_ROLES) or {event['EventKey']: event['Role'] for event in events} != EVENT_ROLES:
        raise ValueError('Source manifest does not declare the six fixed event/role assignments.')
    audited, rule_counts = [], []
    for event in events:
        key = event['EventKey']
        if event['file'] != f'{key}_laps.csv':
            raise ValueError(f'Unexpected source filename for {key}.')
        path = SOURCE / event['file']
        if digest(path) != event['sha256']:
            raise ValueError(f'Source hash changed: {path.name}.')
        raw = pd.read_csv(path)
        if len(raw) != event['lap_records']:
            raise ValueError(f'Source record count changed: {key}.')
        for column in EVENT_METADATA:
            if column not in raw or not raw[column].astype(str).eq(str(event[column])).all():
                raise ValueError(f'Inconsistent source event metadata: {key} {column}.')
        data, checks = audit_single_event(raw)
        audited.append(data)
        rule_counts.extend({'EventKey': key, 'EventName': event['EventName'], 'Role': event['Role'],
                            'Rule': rule, 'FlaggedLaps': int(count)} for rule, count in checks.sum().items())
    return manifest, pd.concat(audited, ignore_index=True), pd.DataFrame(rule_counts)


def domain_checks(examples):
    train = examples.loc[examples['Split'].eq('train')]
    rows = []
    held = examples.loc[~examples['Split'].eq('train')]
    for key, event in held.groupby('EventKey'):
        for feature in NUMERIC_FEATURES:
            low, high = float(train[feature].min()), float(train[feature].max())
            outside = event[feature].lt(low) | event[feature].gt(high)
            rows.append({'EventKey': key, 'EventName': event.iloc[0]['EventName'], 'Role': event.iloc[0]['Role'],
                         'Feature': feature, 'Check': 'Outside training range', 'AffectedTargets': int(outside.sum()),
                         'Targets': len(event), 'TrainingMin': low, 'TrainingMax': high})
        for feature in CATEGORICAL_FEATURES:
            unseen = ~event[feature].isin(train[feature].unique())
            rows.append({'EventKey': key, 'EventName': event.iloc[0]['EventName'], 'Role': event.iloc[0]['Role'],
                         'Feature': feature, 'Check': 'Unseen training category', 'AffectedTargets': int(unseen.sum()),
                         'Targets': len(event), 'TrainingMin': None, 'TrainingMax': None})
    return pd.DataFrame(rows)


def main():
    pilot_before = {name: digest(ROOT / name) for name in PILOT_FILES if (ROOT / name).exists()}
    source, audited, rules = load_sources()
    result = run_weekend_experiment(audited)
    coverage = audited.groupby(list(EVENT_METADATA) + ['Driver'], sort=False).agg(
        RecordedLaps=('LapNumber', 'size'), EligibleLaps=('Included', 'sum'),
    ).reset_index()
    pair_counts = result.examples.groupby(['EventKey', 'Driver']).size().rename('Examples').reset_index()
    coverage = coverage.merge(pair_counts, on=['EventKey', 'Driver'], how='left')
    coverage['Examples'] = coverage['Examples'].fillna(0).astype(int)
    coverage['ExcludedLaps'] = coverage['RecordedLaps'] - coverage['EligibleLaps']
    event_metrics = result.metrics.loc[result.metrics['Split'].eq('test') & result.metrics['EventKey'].ne('ALL') & result.metrics['Driver'].eq('ALL')]
    macro = event_metrics.groupby('Model').agg(TestWeekends=('EventKey', 'size'), EqualWeekendMAESeconds=('MAESeconds', 'mean')).reset_index()
    outputs = {'lap_audit.csv': audited, 'examples.csv': result.examples, 'predictions.csv': result.predictions,
               'metrics.csv': result.metrics, 'coverage.csv': coverage, 'rule_counts.csv': rules,
               'alpha_validation.csv': result.tuning, 'domain_checks.csv': domain_checks(result.examples),
               'equal_weekend_metrics.csv': macro}
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(exist_ok=True)
    for name, data in outputs.items():
        data.to_csv(REPORTS / name, index=False)
    joblib.dump(result.pipeline, MODEL)
    restored = joblib.load(MODEL)
    replay = result.predictions['LastLapSeconds'].to_numpy() + restored.predict(result.predictions[list(MODEL_FEATURES)])
    np.testing.assert_allclose(replay, result.predictions['RidgeSeconds'], atol=1e-10, rtol=0)
    pilot_after = {name: digest(ROOT / name) for name in pilot_before}
    if pilot_before != pilot_after:
        raise RuntimeError('Original pilot changed during the weekend experiment.')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': result.protocol,
                'source_manifest_sha256': digest(SOURCE / 'source_manifest.json'),
                'sources': source['events'], 'recorded_laps': len(audited), 'eligible_laps': int(audited['Included'].sum()),
                'code_sha256': {name: digest(ROOT / name) for name in ('weekend_model.py', 'pace_model.py', 'train_weekends.py', 'WEEKEND_PROTOCOL.md')},
                'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                             'fastf1': source['fastf1_version'], 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
                'model': {'file': 'models/weekend_ridge.joblib', 'sha256': digest(MODEL),
                          'features': list(MODEL_FEATURES), 'roundtrip_prediction_verified': True},
                'original_pilot_sha256': pilot_before, 'original_pilot_unchanged': True,
                'artifacts': {name: {'rows': len(data), 'sha256': digest(REPORTS / name)} for name, data in outputs.items()}}
    (REPORTS / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_value, allow_nan=False) + '\n', encoding='utf-8')
    print('Raw/eligible laps:', manifest['recorded_laps'], manifest['eligible_laps'])
    print('Example counts:', result.protocol['counts'])
    print(result.tuning.to_string(index=False))
    print(result.metrics.loc[result.metrics['Driver'].eq('ALL')].to_string(index=False))
    print('Equal-weekend test metrics:')
    print(macro.to_string(index=False))
    print('Saved model roundtrip verified; original pilot unchanged.')


if __name__ == '__main__':
    main()
