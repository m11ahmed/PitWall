# PitWall first data audit

Input: 114 records. Eligible: 101. Excluded: 13.

| Driver | Recorded | Eligible | Excluded |
|---|---:|---:|---:|
| LEC | 57 | 50 | 7 |
| SAI | 57 | 51 | 6 |

## Exclusion rules

A lap is retained only if it passes every rule. Reasons overlap.

- **invalid_lap_time: 0**. Lap time is missing, unparseable, non-finite or non-positive.
- **invalid_lap_number: 0**. Lap number is missing, non-positive or not an integer.
- **opening_lap: 2**. Opening lap includes the standing start.
- **pit_in: 4**. Reported pit-in timestamp is present.
- **pit_out: 4**. Reported pit-out timestamp is present.
- **not_green_only: 4**. Track status is not exclusively 1 (green), including unknown statuses.
- **timing_not_accurate: 10**. IsAccurate is not explicitly true, including unknown values.
- **deleted_or_unknown: 1**. Deleted is not explicitly false, including unknown values.
- **generated_or_unknown: 0**. FastF1Generated is not explicitly false, including unknown values.

## Interpretation

The figure retains all timed laps in its upper panel and eligible laps in its lower panel.
Lines break across omitted laps and stint boundaries. Lower lap time means faster.
Different scales in the two panels show pit-stop effects and racing pace separately.

These are analysis-eligible laps, not verified traffic-free laps.
IsAccurate concerns timing integrity; it does not guarantee an error-free driving lap.
TyreLife is reported usage in laps, not measured tyre wear.
A stint slope cannot separate tyre effects from fuel burn, traffic, pace management or track evolution.
Unmatched whole-race medians cannot establish a driver or car's intrinsic advantage.

Next: compare overlapping race-lap windows and compounds, with sample counts.
This milestone includes no prediction model and no causal tyre-degradation estimate.

Source documentation: https://github.com/theOehrly/Fast-F1
Input SHA-256: c0a921d8a638a8d788728ce9a6f908a0292c45269676936c63c1a88ac0c797b5
