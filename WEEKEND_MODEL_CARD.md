# Model card: conditional next-lap pace across six weekends

## Purpose and preserved pilot

This is PitWall's separate whole-weekend evaluation of the next-lap Ridge model.
It estimates the recorded time of actual lap t+1 after completed lap t, conditional
on both laps being analysis-eligible and in the same reported stint and compound.
The model and baselines are evaluated on the same historical targets.

The original two-driver Bahrain study remains in [MODEL_CARD.md](MODEL_CARD.md).
Its source CSV, fitted model, predictions and metrics were hash-checked before and
after this expansion and remained unchanged. The expanded fitted pipeline is
models/weekend_ridge.joblib, with evidence under reports/weekends/.

## Public data and coverage

Source: archived FastF1 Race sessions from six selected 2024 weekends. The driver
cohort is LEC/SAI (Ferrari), HAM/RUS (Mercedes) and VER/PER (Red Bull Racing), as
reported in the exports. This is a compact selected cohort, not a season-wide
sample or a study of the entire grid.

| Weekend | Date | Role | Recorded laps | Eligible laps | Consecutive examples |
|---|---|---|---:|---:|---:|
| Bahrain, round 1 | 2024-03-02 | Training | 342 | 300 | 273 |
| Australia, round 3 | 2024-03-24 | Training | 251 | 211 | 192 |
| Japan, round 4 | 2024-04-07 | Training | 318 | 267 | 243 |
| China, round 5 | 2024-04-21 | Validation | 336 | 249 | 232 |
| Miami, round 6 | 2024-05-05 | Test | 342 | 285 | 262 |
| Emilia Romagna / Imola, round 7 | 2024-05-19 | Test | 378 | 352 | 334 |
| Total | | | 1,967 | 1,664 | 1,536 |

Retirements remain in the raw exports; drivers need not finish a race. For example,
Australia contains 16 HAM and 4 VER lap records. Coverage and exclusions are
reported per event and driver rather than replacing incomplete races.

Acquisition requests lap timing and race-control messages, with telemetry and
weather disabled. Race-control messages establish archived deletion flags. Every
export has a source hash, event identity, date, fixed role and driver counts.
LapStartDate is missing in all 1,967 timing-only records; EventDate is available.
The study uses session-relative Time/LapStartTime for adjacency and event dates
for weekend ordering. It makes no claim about absolute UTC timestamps for each lap.

## Target, features and information boundary

The same nine audit rules from the first study apply: invalid lap time/number,
opening lap, pit-in, pit-out, non-green status, inaccurate timing, deleted/unknown
flags, and generated/unknown flags. Reasons may overlap. Analysis-eligible does
not establish clear air or comparable traffic.

Pairs must be actual consecutive t -> t+1 laps within one event and driver, with
valid timing/metadata and matching known stint/compound. They never jump across
excluded or missing laps. Target eligibility is established retrospectively from
final archived flags; the model does not predict eligibility, incidents or stops.

| Predictor | Information through completed lap t |
|---|---|
| OriginLapNumber | Reported race-lap number |
| TyreLife | Reported tyre usage in laps, not measured wear |
| LastLapSeconds | Observed lap time at t |
| TrailingDeltaSeconds | Median of up to 3 consecutive eligible times through t, minus last time |
| Driver | Reported driver identifier |
| Compound | Reported compound at t |

History resets after excluded/missing laps, invalid metadata, stint/compound
changes and every event boundary. Event identity, event date and fold role are
bookkeeping only. No target-lap time, future tyre age, target flags, telemetry,
weather, private fuel or setup data enter the predictors.

## Fixed evaluation

[WEEKEND_PROTOCOL.md](WEEKEND_PROTOCOL.md) fixes the races and roles. Training
uses 708 examples from Bahrain/Australia/Japan; 232 China examples select alpha;
596 Miami/Imola targets evaluate the frozen selected model. Whole events belong
to one role, so histories cannot cross a fold through an event boundary.

Ridge predicts TargetSeconds - LastLapSeconds; adding that residual to the last
observed time produces the estimated next-lap time. Numeric StandardScaler,
one-hot encoding and Ridge are fitted solely on training weekends. The fixed
alpha grid is 0.1, 1, 10 and 100; China selects 10 by lowest MAE (0.225279 s).
There is no refit on China and no test-based alpha or event replacement.

Persistence predicts the last observed time. The other baseline predicts the
same available trailing median. Earlier observed held-out laps may update later
one-step histories; model parameters remain fixed. Feed arrival latency and
later timing/flag revisions are not replayed.

## Held-out results

Lower MAE/RMSE is better. Bias is prediction minus observation; positive means
predicted slower. Values are seconds, not an accuracy percentage.

| Pooled test method | Targets | MAE (s) | RMSE (s) | Bias (s) |
|---|---:|---:|---:|---:|
| Ridge | 596 | 0.280751 | 0.401706 | +0.042865 |
| Last-lap persistence | 596 | 0.288183 | 0.419431 | +0.014213 |
| Trailing median | 596 | 0.287437 | 0.417424 | +0.008283 |

| Test weekend | Targets | Ridge MAE (s) | Persistence MAE (s) | Median MAE (s) |
|---|---:|---:|---:|---:|
| Miami | 262 | 0.281009 | 0.280095 | 0.283742 |
| Imola | 334 | 0.280548 | 0.294527 | 0.290335 |
| Equal-weekend mean | 2 weekends | 0.280779 | 0.287311 | 0.287039 |

The pooled result weights each lap equally. The equal-weekend result averages the
two weekend MAEs, giving each weekend equal weight. Ridge's pooled gain over
persistence is only 0.007432 s MAE. Ridge is slightly worse than persistence in
Miami; the overall improvement is not consistent across weekends.

| Pooled test driver | Targets | Ridge MAE (s) | Persistence MAE (s) | Median MAE (s) |
|---|---:|---:|---:|---:|
| HAM | 102 | 0.3568 | 0.3462 | 0.3760 |
| LEC | 100 | 0.2446 | 0.2613 | 0.2551 |
| PER | 99 | 0.3679 | 0.3881 | 0.3555 |
| RUS | 97 | 0.3004 | 0.3028 | 0.3117 |
| SAI | 102 | 0.2381 | 0.2547 | 0.2440 |
| VER | 96 | 0.1732 | 0.1723 | 0.1785 |

Ridge loses to persistence for HAM and VER, and to the trailing median for PER.
Further event/driver failures remain in metrics.csv and the dashboard, including
Miami PER/RUS versus persistence and Miami SAI versus the trailing median. No
errors are removed to improve the comparison.

## Domain checks and practical limits

All held-out Driver/Compound categories occur in training. The encoder learns
categories only from training and ignores unseen categories; that behavior alone
would not validate predictions for an unseen driver or compound.

Numeric min/max checks are retained rather than filtering difficult targets.
China has 125/232 last-lap times and 3/232 tyre-age values outside training ranges.
Miami has 2/262 tyre-age values outside. Imola has 36/334 origin lap numbers,
5/334 tyre-age values, 8/334 last-lap times and 1/334 trailing-delta values outside.
Counts overlap and must not be added. Range checks inspect each feature
separately; they do not measure all forms of distribution shift.

This evaluates transfer to two specific later weekends with the same six drivers.
It does not establish unseen-driver, full-season, all-circuit, weather-condition
or live-service performance. Nearby lap errors are correlated; 596 targets do not
constitute 596 independent trials. No calibrated intervals or statistical
significance are claimed for the small pooled gain.

Lap times are public observations; model predictions are estimates. Reported
tyre usage and race lap do not measure wear or fuel mass. Private tyre
measurements, brake pressure, fuel load and engineering setup are unavailable.
No causal degradation, optimal strategy, driver-skill or team-affiliation claims
are made.

## Reproduce and inspect

Existing exports train entirely offline:

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe train_weekends.py
```

For first acquisition, run download_weekends.py before training; that step uses
the public network and existing FastF1 cache. The fixed protocol hash is checked,
and missing required sources stop evaluation rather than trigger a substitute.

The trainer saves models/weekend_ridge.joblib, reloads the trusted local artifact
it just wrote and verifies matching residual-plus-last-lap predictions. The
dashboard uses exported tables rather than deserializing the fitted model.

reports/weekends/ includes lap_audit.csv, examples.csv, predictions.csv,
metrics.csv, alpha_validation.csv, coverage.csv, rule_counts.csv,
domain_checks.csv, equal_weekend_metrics.csv and manifest.json. The manifest
records source/artifact hashes, versions, feature names, roles, counts, saved
model verification and preserved pilot hashes. Versions are Python 3.14.0,
FastF1 3.8.3, scikit-learn 1.9.1, NumPy 2.5.3, pandas 2.3.3 and joblib 1.6.0.

Select **Across weekends** in the dashboard's Study control to inspect the
expanded results. Its weekend/driver controls are independent of the Bahrain
sidebar. [VALIDATION.md](VALIDATION.md) records executed tests, browser checks
and their limits.

Method references: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[pipelines](https://scikit-learn.org/stable/modules/compose.html),
[model persistence](https://scikit-learn.org/stable/model_persistence.html),
[FastF1](https://github.com/theOehrly/Fast-F1).
