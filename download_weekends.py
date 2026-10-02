"""Acquire the fixed six-weekend timing study without telemetry/weather."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'weekends'
DRIVERS = ('LEC', 'SAI', 'HAM', 'RUS', 'VER', 'PER')
EVENTS = [(1, 'Bahrain', 'train'), (3, 'Australia', 'train'), (4, 'Japan', 'train'),
          (5, 'China', 'validation'), (6, 'Miami', 'test'), (7, 'Emilia Romagna', 'test')]
MANIFEST = DATA / 'source_manifest.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    (ROOT / 'cache').mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(ROOT / 'cache'))
    fastf1.Cache.offline_mode(False)
    source = {'created_utc': datetime.now(timezone.utc).isoformat(), 'year': 2024,
              'session_type': 'Race', 'drivers': list(DRIVERS), 'fastf1_version': fastf1.__version__,
              'load_options': {'laps': True, 'telemetry': False, 'weather': False, 'messages': True},
              'protocol_sha256': digest(ROOT / 'WEEKEND_PROTOCOL.md'), 'complete': False, 'events': []}
    for round_number, label, role in EVENTS:
        key = f'2024-{round_number:02d}'
        path = DATA / f'{key}_laps.csv'
        print(f'Acquiring {key} {label} ({role})', flush=True)
        session = fastf1.get_session(2024, round_number, 'R')
        if int(session.event['RoundNumber']) != round_number:
            raise ValueError('Returned event round differs from fixed protocol.')
        session.load(laps=True, telemetry=False, weather=False, messages=True)
        messages = session.race_control_messages
        if messages is None or messages.empty:
            raise ValueError(f'{key}: race-control messages unavailable; deletion flags cannot be established.')
        laps = session.laps.loc[session.laps['Driver'].isin(DRIVERS)].copy()
        present = set(laps['Driver'].dropna())
        if present != set(DRIVERS):
            raise ValueError(f'{key}: expected all six entrants; absent {set(DRIVERS) - present}. No fallback event is used.')
        if laps.duplicated(['Driver', 'LapNumber']).any():
            raise ValueError(f'{key}: duplicate driver/lap records need investigation.')
        metadata = {'EventKey': key, 'EventName': str(session.event['EventName']),
                    'Year': 2024, 'RoundNumber': round_number,
                    'EventDate': session.date.date().isoformat(), 'Role': role}
        for column, value in metadata.items():
            laps[column] = value
        laps.to_csv(path, index=False)
        event = {**metadata, 'file': path.name, 'sha256': digest(path), 'lap_records': len(laps),
                 'driver_records': {str(d): int(n) for d, n in laps.groupby('Driver').size().items()},
                 'race_control_message_records': len(messages),
                 'missing_flags': {name: int(laps[name].isna().sum()) for name in ['Deleted', 'IsAccurate', 'FastF1Generated']}}
        source['events'].append(event)
        MANIFEST.write_text(json.dumps(source, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Saved {len(laps)} laps, {len(messages)} race-control messages: {path.name}', flush=True)
    source['complete'] = True
    source['lap_records'] = sum(event['lap_records'] for event in source['events'])
    MANIFEST.write_text(json.dumps(source, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Complete: {len(source["events"])} weekends, {source["lap_records"]} lap records.', flush=True)


if __name__ == '__main__':
    main()
