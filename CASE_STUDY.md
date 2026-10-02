# Bahrain 2024: auditable race pace, telemetry and first ML pilot

## Question and selection

How does Leclerc's and Sainz's observed pace compare over shared,
analysis-eligible race laps using the same reported tyre compound? Can a small
model predict the next eligible lap more accurately than recent-lap baselines?

FastF1 timing contains 114 records: 57 per driver. The audit preserves every
record and adds eligibility/exclusion reasons. Eligible laps require positive
finite timing, valid lap numbers, green-only track status, confirmed timing
accuracy and known non-deleted/non-generated flags. Opening, pit-in and pit-out
laps are excluded.

- LEC: 50 eligible, 7 excluded.
- SAI: 51 eligible, 6 excluded.
- Total: 101 eligible, 13 excluded.

Exclusion reasons overlap; adding their counts does not give the excluded total.
Original lap CSV SHA256:
c0a921d8a638a8d788728ce9a6f908a0292c45269676936c63c1a88ac0c797b5.

## Matched pace

Joining on race-lap number and requiring the same known compound yields 46 pairs.
Disabling compound matching yields 47 pairs, including one different-compound pair.

Delta is SAI lap time minus LEC lap time. The median paired delta is -0.368 s:
SAI's recorded time was lower in this selected comparison. This uses the median
of paired differences, not the difference between unrelated stint medians.

| Race laps | Compound | Matched pairs | Median SAI - LEC (s) |
|---|---|---:|---:|
| 2-3 | SOFT | 2 | -0.304 |
| 5-9 | SOFT | 5 | -0.041 |
| 16-33 | HARD | 18 | -0.587 |
| 37-57 | HARD | 21 | -0.241 |

Windows split at missing matched laps or changes to either driver's stint or
compound. Counts sum to 46. The two-lap window provides little evidence about
repeatability. The earlier HARD window has a more negative median than the later
one; this describes sampled race windows and does not identify its cause.

## Telemetry coverage and example

The offline cache export retains 41,286 native car samples across all 114 laps.
It uses Lap.get_car_data(interpolate_edges=False), requires Source=car, and avoids
merged position data. The manifest records hashes and per-lap diagnostics.

Timing eligibility and telemetry support are separate checks. Of the 101
timing-eligible laps, 64 support continuous, strictly increasing estimated distance
and boundary coverage under the stated 1.0 s tolerance. Across all 114 laps,
39 contain at least one sampling gap over 1.0 s. Two invalid timestamps on
already-excluded opening laps are retained in the export; their distance is withheld.

The default shared example is lap 21, selected for usable telemetry quality rather
than fastest performance. Both drivers are on HARD tyres, but reported ages differ.

| Lap 21 context | LEC | SAI |
|---|---:|---:|
| Recorded lap time (s) | 96.977 | 96.104 |
| Reported tyre age (laps) | 10 | 7 |
| Native samples | 369 | 364 |
| Median sample interval (s) | 0.240 | 0.240 |
| Largest sample interval (s) | 0.879 | 0.879 |
| Time before first sample (s) | 0.162 | 0.217 |
| Time after final sample (s) | 0.096 | 0.286 |

The observed lap-time difference is -0.873 s, taken from timing. The telemetry
overlay shows public speed, throttle and brake on/off; it does not explain how
much each control input caused that difference.

Distance is estimated by trapezoidal speed integration from each first recorded
sample. Missing boundary time is disclosed without extrapolation. Invalid speed
or a gap over the threshold withholds every later absolute distance estimate;
elapsed-time traces remain available with gap separators. Optional interpolation
uses a 10 m grid over common supported distance, plus its final endpoint.

## Conditional next-lap model

The prediction point is the completion of lap t; the target is the actual lap t+1.
Both laps must be eligible, truly consecutive and in the same known stint and
compound. Future eligibility is established retrospectively for scoring. The model
does not predict pit stops, incidents or whether the next lap will be eligible.

Inputs are completed lap number, reported tyre age, last lap time, trailing median
minus last lap time, driver and compound. The trailing history contains at most
three consecutive eligible laps through t and resets after gaps or exclusions.
No target-lap measurement or telemetry enters the model.

Ridge predicts a correction to the last observed time. Baselines repeat that time
or use the trailing median. All methods score the same targets. Numeric scaling
and category encoding are fitted only on training examples.

The two drivers share global elapsed-time cutoffs at 60% and 80% of the exported
race span. Pairs crossing cutoffs are purged, avoiding random adjacent-lap splits.
Of 93 examples, 52 are training, 17 validation, 22 final test and 2 purged.
Validation chose alpha 10 from 0.1, 1, 10 and 100. The selected training fit remains
frozen; it was not refitted on validation.

| Final test MAE (s) | Both drivers, n=22 | LEC, n=11 | SAI, n=11 |
|---|---:|---:|---:|
| Ridge | 0.1439 | 0.1689 | 0.1188 |
| Last-lap persistence | 0.1559 | 0.1598 | 0.1520 |
| Trailing median | 0.1806 | 0.2100 | 0.1512 |

Pooled Ridge MAE is 0.0120 s lower than persistence. The result is mixed: Ridge
performs worse for LEC and better for SAI in this small final window. On validation,
Ridge is worse than persistence for SAI. The experiment supports reporting a modest
within-race result with subgroup failures retained; it does not establish general
predictive superiority.

These are historical one-step predictions: an earlier observed validation/test
lap can enter the history of the next prediction without refitting. All held-out
targets use HARD tyres. Adjacent errors are correlated; 22 rows are not 22
independent trials. No confidence intervals, accuracy percentage or causal tyre
wear/fuel interpretation is claimed. Full MAE, RMSE, signed bias and provenance
are in reports/ml and [MODEL_CARD.md](MODEL_CARD.md).

## Interpretation limits

- Equal race lap/compound do not establish equal tyre age, fuel load or clear air.
- IsAccurate is a timing-quality flag, not a guarantee of ideal driving.
- Reported tyre usage is not measured wear; actual wear and temperatures are absent.
- Sample-origin distance is not exact circuit position. Origins and paths differ.
- Brake is boolean, not pressure. Sparse samples limit exact braking-point claims.
- A speed difference is not a corner time gain/loss or causal performance measure.
- IQR describes dispersion, not a confidence interval.
- Missing/excluded samples can change the cohort.
- Model outputs are estimates; the lap timings and public channels are observations.
- Post-race timing/flags do not establish tested real-time feature availability.
- One weekend and two drivers do not validate predictions across races.

## Reproduce

Run analyze_data.py, export_telemetry.py and train_model.py, then launch app.py
through Streamlit. See [MODEL_PROTOCOL.md](MODEL_PROTOCOL.md) for the fixed model
procedure and [VALIDATION.md](VALIDATION.md) for executed checks.

For the pace table, open Tyre stints with both drivers, laps 1-57, LEC as reference,
and matching compounds enabled. Download the pairs for inspection.

For the telemetry example, open Telemetry, choose Same race lap and select lap 21.
Inspect lap context, sampling diagnostics and the selected-sample CSV. Switch to
elapsed time to see a trace without a derived distance axis.

For the ML result, open ML pace and select the final test phase with both drivers
and the full lap window. Inspect baseline and per-driver errors, then download the
predictions. Filtering changes the displayed cohort, not the trained model or
original evaluation. More races and whole-weekend holdouts are the next step.
