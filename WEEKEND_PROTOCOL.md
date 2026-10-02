# Whole-weekend evaluation protocol

This is a separate expansion of the original Bahrain Ferrari pilot. Keep the
pilot CSV, model, predictions and scores unchanged.

## Fixed data scope and split

Race sessions only, 2024; drivers LEC, SAI, HAM, RUS, VER and PER.

| Round | Weekend | Role |
|---|---|---|
| 1 | Bahrain | Training |
| 3 | Australia | Training |
| 4 | Japan | Training |
| 5 | China | Validation |
| 6 | Miami | Test |
| 7 | Emilia Romagna / Imola | Test |

These roles are fixed before Ridge test scores are generated. Round 2 is outside
this deliberately compact same-driver scope; Sainz did not race that weekend.
This is not a complete-season or representative all-circuit sample. Races are
ordered by round and event date, not comparable session-elapsed timestamps.

Use public FastF1 archived timing and race-control messages. Do not request
telemetry or weather for this expansion. Preserve every exported selected-driver
lap, including retirements and excluded records. A driver need not finish a race.
Record event/driver counts, exclusion reasons, source hashes and missing fields.
Missing required weekends stop evaluation; do not silently swap an event or its
role based on availability or scores.

## Same conditional target, separate event histories

Apply the original timing eligibility rules to every selected driver. Build
actual consecutive t -> t+1 pairs inside each event and driver. Both laps must be
eligible, have valid metadata and use the same known stint and compound. History
contains up to three eligible consecutive laps through t and resets after any
excluded/missing lap or stint/compound change. Never carry history between races.

Target eligibility is retrospective and does not predict pit stops or incidents.
Archived post-race timing/flags reconstruct the prediction point; live delivery
latency and later flag corrections have not been evaluated.

## Model and baselines

Keep the first pilot's features and residual target: OriginLapNumber, TyreLife,
LastLapSeconds, TrailingDeltaSeconds, Driver and Compound. Predict a correction
to the last observed lap time. Event identifiers, role, event date, target time,
future tyre age, target flags and target telemetry are not predictors.

Fit numeric scaling, category encoding and Ridge only on training weekends.
Fixed alpha grid: 0.1, 1, 10, 100. Choose the lowest pooled validation MAE, breaking
exact ties with the smaller alpha. Freeze that training-fitted pipeline; do not
refit on China or tune after looking at Miami/Imola results.

Score last-lap persistence, trailing median and Ridge on exactly the same targets.
During held-out one-step replay, prior actual laps may update rolling features,
but model parameters stay frozen. Retain all errors and failed comparisons.

## Report and interpretation

Report MAE, RMSE, signed bias and target counts: pooled, by event and by driver.
Show each test weekend separately and provide an equal-weekend mean of test MAEs
alongside the lap-weighted pooled result. Show compound/driver coverage; unseen
categories and out-of-training-range numeric inputs limit interpretation.

This tests transfer to two specific later weekends using the same six drivers;
it does not establish performance on unseen drivers, all tracks or an entire
season. Adjacent laps within an event are correlated. No accuracy percentage,
calibrated uncertainty interval, causal tyre/fuel estimate or team affiliation is
claimed. Baseline wins and retirements remain visible.

[FastF1 2024 event schedule](https://raw.githubusercontent.com/theOehrly/f1schedule/master/schedule_2024.json).
