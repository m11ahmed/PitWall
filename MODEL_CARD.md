# Model card: conditional next-lap pace, first pilot

## Purpose and status

A small local regression experiment for PitWall's Bahrain 2024 LEC/SAI case study.
At the completion of lap t, estimate the recorded time of the actual next lap t+1,
conditional on both laps being analysis-eligible and in the same known stint and
compound. It is a historical one-step replay, not a tested live race service.

The fitted model is models/pace_ridge.joblib. The dashboard reads held-out CSV
predictions rather than loading the model. train_model.py reloads only the trusted
local artifact it just created and verifies identical predictions after saving.
Use matching dependency versions when reproducing that artifact.

## Data and target selection

Source: public FastF1 Bahrain 2024 race timing; 114 laps, two drivers. Original CSV
SHA256: c0a921d8a638a8d788728ce9a6f908a0292c45269676936c63c1a88ac0c797b5.
The timing audit retains 101 eligible laps. It excludes opening, pit-in/pit-out,
non-green, inaccurate, deleted, generated and invalid records under explicit rules.

The model uses 93 true consecutive t -> t+1 pairs with valid timing/metadata and
matching known stint/compound. It never jumps over an excluded or missing lap.
Target eligibility is known retrospectively for scoring, not predicted. Results
do not cover excluded targets, pit-stop decisions, incidents or future eligibility.

Features describe lap t and earlier, but timing and selection flags come from a
post-race export. Live delivery delays and flag revisions have not been evaluated.

## Features and estimator

| Feature | Information used at completed lap t |
|---|---|
| OriginLapNumber | Reported race-lap number |
| TyreLife | Reported tyre usage in laps, not wear |
| LastLapSeconds | Recorded lap time at t |
| TrailingDeltaSeconds | Median of up to 3 consecutive eligible times through t, minus last time |
| Driver | LEC or SAI |
| Compound | Reported compound at t |

History resets after missing/excluded laps, invalid metadata, stint changes or
compound changes. There are no target-lap measurements, target tyre age, target
flags, telemetry, weather, position, private fuel or setup features.

Ridge predicts the residual TargetSeconds - LastLapSeconds. The forecast adds
that correction to LastLapSeconds. StandardScaler numeric preprocessing and
one-hot category encoding are fitted solely on training examples. Baselines are
last-lap persistence and the same available trailing median; all methods score
identical targets.

## Frozen evaluation

Shared cutoffs use 60% and 80% of the full two-driver export's elapsed-time span,
including excluded timing records when defining that span. Pairs crossing either
boundary are purged. These are time proportions, not promised row proportions.

| Window | Example count | Use |
|---|---:|---|
| Training | 52 | Fit preprocessing and each Ridge candidate |
| Validation | 17 | Choose alpha only |
| Final test | 22 | Score the selected frozen training fit |
| Purged | 2 | No fitting or scoring |

Session elapsed-time cutoffs are 6926.734 s and 8035.675 s. Alpha candidates were
0.1, 1, 10 and 100; validation selected 10 (MAE 0.2032 s). The model was not refitted
on validation. Features and protocol were fixed for this first pilot; later
experiments should report separately rather than rewrite its test result.

Earlier observed held-out lap times may enter later one-step histories. This
matches sequential prediction after completed laps; the estimator remains frozen.
Full rules are in [MODEL_PROTOCOL.md](MODEL_PROTOCOL.md).

## Results

Lower MAE/RMSE is better. Signed bias = prediction minus observed time, so positive
bias means predicted slower. These are errors in seconds, not accuracy percentages.

| Final test method | n | MAE (s) | RMSE (s) | Bias (s) |
|---|---:|---:|---:|---:|
| Ridge | 22 | 0.1439 | 0.2076 | -0.0839 |
| Last-lap persistence | 22 | 0.1559 | 0.2120 | -0.0791 |
| Trailing median | 22 | 0.1806 | 0.2402 | -0.1481 |

| Final test MAE (s) | LEC, n=11 | SAI, n=11 |
|---|---:|---:|
| Ridge | 0.1689 | 0.1188 |
| Last-lap persistence | 0.1598 | 0.1520 |
| Trailing median | 0.2100 | 0.1512 |

Ridge's pooled improvement over persistence is 0.0120 s MAE. It is worse for LEC
on final test. On validation, SAI Ridge MAE is 0.2488 s versus persistence 0.2177 s.
These failures remain part of the result. The pilot does not establish that Ridge
will beat either baseline in a new race.

## Limits and next evidence

- One weekend, two drivers, small evaluation windows; nearby errors are correlated.
- All validation/test targets use HARD tyres; SOFT generalisation is untested.
- No held-out race, circuit, weather condition, other driver or team validation.
- Retrospective target selection and timing flags do not establish real-time use.
- Public reported tyre age and race lap are not measured tyre wear or fuel mass.
- No causal tyre degradation, driver-skill, optimal strategy or private team claims.
- No calibrated prediction interval or independent-trial confidence interval.
- Model estimates and measured lap timings remain explicitly distinguishable.

Add public races/drivers and hold out whole weekends from training and tuning.
Retain the original result and report event/driver failures alongside pooled errors.
Computer vision is a separate future study with its own footage and annotation.

## Reproduce and inspect

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe train_model.py
```

No network or paid provider is required for training from the existing lap CSV.
Installed versions recorded by the run: Python 3.14.0, scikit-learn 1.9.1,
NumPy 2.5.3, pandas 2.3.3 and joblib 1.6.0.

reports/ml/ contains examples.csv (including purged rows), predictions.csv (39
validation/test targets), metrics.csv (18 overall/per-driver rows),
alpha_validation.csv and manifest.json. The manifest records source/artifact
hashes, dependencies, split cutoffs, feature names and saved-model verification.
[VALIDATION.md](VALIDATION.md) records executed tests/browser checks and their limits.

## Method references

[Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[training pipelines](https://scikit-learn.org/stable/modules/compose.html) and
[model persistence](https://scikit-learn.org/stable/model_persistence.html).
