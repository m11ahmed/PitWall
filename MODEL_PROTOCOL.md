# Fixed first ML experiment

Prediction point: after the completed race lap t. Target: the actual next lap t+1,
for the same driver. This is historical one-step replay using post-race timing;
live feed delays and retrospective flag revisions have not been tested.

## Cohort

Retain a pair only if t and t+1 are both analysis-eligible, truly consecutive,
and use the same known stint and compound. Current metadata and timing must be
valid. Eligibility at t+1 is determined retrospectively for scoring, not predicted.
No claim is made for pit, yellow-flag, opening, deleted or generated target laps.

History contains at most the current lap and the preceding two consecutive
eligible laps in the same stint/compound. Exclusions and gaps reset it. A prior
observed validation/test lap can enter the next prediction's history; the fitted
model remains frozen throughout that rolling replay.

## Features and methods

Fixed numeric features: completed lap number, reported tyre age at t, lap time at
t, and trailing median minus last-lap time. Categories: driver and compound at t.
No target-lap telemetry, time, tyre age, flags, position or weather enters features.
Tyre age measures laps used, not tyre wear. Race lap is not measured fuel mass.

Baselines: repeat the last observed lap time; median of the available last <=3
contiguous eligible lap times. Both use exactly the same scored targets as Ridge.

Ridge predicts a correction to the last-lap baseline. Numeric scaling and category
encoding are fitted on training examples only. Alpha candidates are fixed at
0.1, 1, 10 and 100. Choose the lowest validation MAE, with the smaller alpha breaking
a tie. Keep that training-fitted pipeline frozen for final test scoring. Do not
refit on validation or modify the features after viewing test performance.

## Global chronology

Use minimum reported lap-start and maximum reported lap-end timestamps from the
complete two-driver timing export to define 60% and 80% elapsed-time cutoffs.
These define windows, not exact row proportions.

- Training: target lap finishes at or before the 60% cutoff.
- Validation: origin lap finishes at/after 60%, target finishes at/before 80%.
- Test: origin lap finishes at/after 80%.
- Any pair crossing a window boundary is purged.

The same time cutoffs apply to both drivers. All training targets have completed
before validation prediction points; likewise validation targets precede test
prediction points. Split boundaries use session timing, never random row shuffle.

## Reporting

Record validation alpha scores, per-lap predictions, all-method MAE/RMSE/bias and
counts, overall and by driver. Error = prediction minus observed; positive means
the prediction was slower. Lower MAE/RMSE is better. Negative model results remain
visible. No probability interval or accuracy percentage is claimed.

This pilot uses one race and two Ferrari drivers. Validation/test laps are HARD;
future race, driver, compound and team generalisation require additional data and
whole-weekend holdouts. Nearby lap errors are correlated, so row count is not a
count of independent trials. Results do not establish causal tyre degradation,
fuel effects or driver skill. All data and artifacts stay in G:\PitWall.
