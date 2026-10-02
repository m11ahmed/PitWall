"""Export native car samples from the existing Bahrain FastF1 cache, offline."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform

import fastf1
import pandas as pd

from analyze_data import INPUT, audit_laps
from telemetry_analysis import prepare_lap_telemetry

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data' / 'bahrain_2024_ferrari_telemetry.csv'
MANIFEST = ROOT / 'data' / 'telemetry_manifest.json'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    audit, _ = audit_laps(pd.read_csv(INPUT))
    fastf1.Cache.enable_cache(str(ROOT / 'cache'))
    fastf1.Cache.offline_mode(True)
    session = fastf1.get_session(2024, 'Bahrain', 'R')
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    frames = []
    for _, record in audit.iterrows():
        driver, lap_number = record['Driver'], int(record['LapNumber'])
        selected = session.laps.loc[
            session.laps['Driver'].eq(driver) & session.laps['LapNumber'].eq(lap_number)
        ]
        if len(selected) != 1:
            raise ValueError(f'Expected one cached timing record for {driver} lap {lap_number}.')
        lap = selected.iloc[0]
        seconds = lap['LapTime'].total_seconds()
        if abs(seconds - record['LapTimeSeconds']) > 0.001:
            raise ValueError(f'Timing export and cache disagree for {driver} lap {lap_number}.')
        car = lap.get_car_data(interpolate_edges=False)
        if car.empty:
            raise ValueError(f'No native car samples for {driver} lap {lap_number}.')
        if not car['Source'].eq('car').all():
            raise ValueError('Expected only native car samples, with no interpolated edges.')
        frames.append(pd.DataFrame({
            'Driver': driver, 'LapNumber': lap_number,
            'ElapsedSeconds': (car['SessionTime'] - lap['LapStartTime']).dt.total_seconds(),
            'SpeedKmh': car['Speed'], 'ThrottlePct': car['Throttle'],
            'BrakeOn': car['Brake'], 'RPM': car['RPM'], 'Gear': car['nGear'],
            'DRS': car['DRS'], 'SampleSource': car['Source'],
            'SampleDate': car['Date'].astype(str),
            'SessionTimeSeconds': car['SessionTime'].dt.total_seconds(),
            'LapStartTimeSeconds': lap['LapStartTime'].total_seconds(),
        }))
    samples = pd.concat(frames, ignore_index=True)
    quality = []
    for _, record in audit.iterrows():
        prepared = prepare_lap_telemetry(
            samples, driver=record['Driver'], lap_number=int(record['LapNumber']),
            lap_time_seconds=record['LapTimeSeconds'],
        )
        quality.append({**asdict(prepared.quality), 'analysis_eligible': bool(record['Included'])})
    # Validate everything before replacing the derived export.
    samples.to_csv(OUTPUT, index=False)
    manifest = {
        'session': {'year': 2024, 'event': 'Bahrain', 'type': 'Race'},
        'drivers': ['LEC', 'SAI'], 'fastf1_version': fastf1.__version__,
        'python_version': platform.python_version(),
        'source_lap_csv': INPUT.name, 'source_lap_sha256': digest(INPUT),
        'telemetry_csv': OUTPUT.name, 'telemetry_sha256': digest(OUTPUT),
        'acquisition': 'Existing FastF1 cache only; offline mode enabled.',
        'slicing': 'Lap.get_car_data(interpolate_edges=False); Source=car required.',
        'sample_count': len(samples), 'lap_count': len(quality),
        'channels': {
            'ElapsedSeconds': 'seconds since reported LapStartTime',
            'SpeedKmh': 'public speed channel, km/h',
            'ThrottlePct': 'public throttle channel, percent; invalid values remain in raw export',
            'BrakeOn': 'public boolean brake channel, not pressure',
            'RPM': 'public engine-speed channel, rpm', 'Gear': 'public gear channel',
            'DRS': 'public provider status code, retained without activation inference',
            'EstimatedDistanceM': 'not a raw channel; computed only by analysis from native speed samples',
        },
        'quality_threshold': '1.0 s maximum gap and boundary-coverage tolerance; an analysis choice.',
        'lap_quality': quality,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Exported {len(samples):,} native samples across {len(quality)} laps.')
    print(f'Eligible laps with distance support: {sum(q["analysis_eligible"] and q["alignment_eligible"] for q in quality)}')
    for q in quality:
        if q['lap_number'] == 20:
            print(json.dumps(q, indent=2))
    print(OUTPUT)


if __name__ == '__main__':
    main()
