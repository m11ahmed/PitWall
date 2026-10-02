"""Run the fixed offline pilot, save the fitted model and its audit artifacts."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from analyze_data import INPUT, audit_laps
from pace_model import MODEL_FEATURES, run_experiment

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports' / 'ml'
MODELS = ROOT / 'models'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unserializable result: {type(value).__name__}')


def main():
    data, _ = audit_laps(pd.read_csv(INPUT))
    result = run_experiment(data)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    examples = result.examples.assign(Split=result.splits.assignments)
    outputs = {
        'examples.csv': examples,
        'predictions.csv': result.predictions,
        'metrics.csv': result.metrics,
        'alpha_validation.csv': result.tuning,
    }
    for name, frame in outputs.items():
        frame.to_csv(REPORTS / name, index=False)
    model_path = MODELS / 'pace_ridge.joblib'
    joblib.dump(result.pipeline, model_path)
    # Only reload the model this command created, to verify the saved artifact.
    restored = joblib.load(model_path)
    replay = result.predictions['LastLapSeconds'].to_numpy() + restored.predict(result.predictions[list(MODEL_FEATURES)])
    np.testing.assert_allclose(replay, result.predictions['RidgeSeconds'].to_numpy(), rtol=0, atol=1e-10)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'case_study': 'Bahrain 2024 race, LEC and SAI',
        'source_file': INPUT.name, 'source_sha256': digest(INPUT),
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'protocol': result.protocol,
        'code_sha256': {name: digest(ROOT / name) for name in ('pace_model.py', 'train_model.py', 'analyze_data.py', 'MODEL_PROTOCOL.md')},
        'split': {name: getattr(result.splits, name) for name in (
            'session_start_seconds', 'session_end_seconds', 'train_cutoff_seconds', 'validation_cutoff_seconds')},
        'split_counts': {key: int(value) for key, value in examples['Split'].value_counts().items()},
        'model': {'file': 'models/pace_ridge.joblib', 'sha256': digest(model_path),
                  'roundtrip_prediction_verified': True, 'features': list(MODEL_FEATURES),
                  'training': 'Training window only, frozen after validation selection.'},
        'artifacts': {name: {'sha256': digest(REPORTS / name), 'rows': len(frame)} for name, frame in outputs.items()},
        'limits': [
            'Retrospective conditional scoring: future lap eligibility is not predicted.',
            'Post-race timing/flags; real-time availability and revision delays are untested.',
            'One race, two drivers; held-out compounds are HARD; no across-race validation.',
            'Rolling one-step replay observes earlier held-out laps without retraining.',
            'Adjacent lap errors are correlated; no calibrated intervals or causal claims.',
        ],
    }
    (REPORTS / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_value, allow_nan=False) + '\n', encoding='utf-8')
    print('Split counts:', manifest['split_counts'])
    print(result.tuning.to_string(index=False))
    print(result.metrics.loc[result.metrics['Driver'].eq('ALL')].to_string(index=False))
    print('Saved model roundtrip verified:', model_path)


if __name__ == '__main__':
    main()
