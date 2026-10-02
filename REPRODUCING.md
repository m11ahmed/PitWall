# Running and reproducing PitWall

This guide retains the full acquisition, setup, dashboard and evaluation commands.
For the questions and conclusions, start with [ENGINEERING_OVERVIEW.md](ENGINEERING_OVERVIEW.md).

## Open the dashboard

From PowerShell:

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [the local dashboard](http://127.0.0.1:8501). Keep PowerShell open while it
runs; press Ctrl+C in that window to stop it. Interactions read local exports and
do not download race data or refit a model.

## Explore it

The **Study** control switches between two independent views:

- **Bahrain case study**: LEC/SAI race pace, tyre stints, telemetry, the first
  chronological ML pilot and the lap audit. Its sidebar controls drivers,
  race-lap window, cohort and comparison reference.
- **Across weekends**: the separate whole-weekend evaluation for
  LEC/SAI/HAM/RUS/VER/PER. Select test or validation weekends, weekends and drivers
  using that view's own controls. Inspect overall/event/driver errors, baseline
  wins, coverage and numeric training-range checks. The Bahrain sidebar does not
  filter this study.

In the Bahrain case study:

- Race pace shows eligible/all recorded laps with timing, compound and exclusions.
- Tyre stints compares shared eligible race laps, with matching compounds by default.
- Telemetry shows speed, throttle and brake on/off for selected laps. Compare the
  same race lap/compound or choose laps separately with context visible. Switch
  between elapsed time and estimated distance; optional common-distance alignment
  exposes its sampling limits.
- ML pace replays held-out estimates from the frozen first Ridge model, alongside
  both baselines. Validation and final test results remain distinct.
- Lap audit exposes each selection rule and individual excluded records.

Downloads include audits, matched pairs, selected native/aligned telemetry and
held-out predictions. Changing filters never trains a model.

Lap-time delta = comparison time minus reference time; negative means the
comparison driver's recorded time was lower. Speed difference has the same
comparison direction in km/h and does not measure time gain. ML signed error =
prediction minus observed time; positive means predicted slower.

## Set up a new environment on G:

Fresh installation was verified on Windows with Python 3.14.0, including all
86 tests and desktop/mobile dashboard checks. Use Python 3.14; the measured
environment uses scikit-learn 1.9.1. The complete version-pinned environment is requirements-lock.txt;
requirements.txt lists the main project dependencies.

The repository includes the study exports and frozen results. Opening the
existing dashboard needs no FastF1 cache, race download, model training or API key.
Installing dependencies requires internet access; running the exported studies
is local. Extraction of native telemetry again requires acquisition of the
FastF1 session cache.

From PowerShell after putting the repository on G:

```powershell
Set-Location "G:\PitWall"
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If python is not on PATH, supply your installed Python 3.14 executable explicitly:

```powershell
.\scripts\setup.ps1 -PythonExecutable "C:\path\to\Python314\python.exe"
```

The C: path above is an example for an already installed interpreter. The script
places the virtual environment and temporary installation files inside the
project on G: and disables pip's download cache. It checks Python's version,
installs the lock file and runs pip check. No new race acquisition or fitting is
performed. The Windows py launcher is optional and is not required by the script.

For another checkout location, change Set-Location to that folder. Runtime paths
are resolved relative to the project, not to the original G:\PitWall directory.
The script is for Windows PowerShell; other OS installations have not been checked.

## Repository contents

Code, tests, documentation, small public-data exports, frozen model/results and
selected screenshots are included. Local environments, FastF1 cache, tmp/ and
local secret files are ignored. .gitattributes preserves exact file bytes,
including line endings, because the experiment manifests verify SHA256 hashes.
Do not normalize line endings in frozen protocol/code/data files without declaring
and regenerating a new experiment.

Raw measurements remain distinct from estimates. Public data accessed through
FastF1 are not claimed as PitWall-owned or privately sourced team data. This local
repository preparation has not published or hosted the project.

## Acquire and reproduce the Bahrain study

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe analyze_data.py
.\.venv\Scripts\python.exe export_telemetry.py
.\.venv\Scripts\python.exe train_model.py
```

The downloader uses the public network and creates the FastF1 cache. The other
commands work offline. Telemetry export requires the existing session cache;
it does not fetch missing data. It checks cached lap timing against the original
CSV and slices native car samples without interpolated lap boundaries or merged
position data.

train_model.py writes models/pace_ridge.joblib and reports/ml/ tables plus a
provenance manifest. It reloads its locally generated fitted model and verifies
matching predictions. The dashboard uses exported tables rather than deserializing
the model and checks source/artifact hashes.

## Acquire and reproduce the whole-weekend study

```powershell
Set-Location "G:\PitWall"
New-Item -ItemType Directory -Path "G:\PitWall\cache" -Force
.\.venv\Scripts\python.exe download_weekends.py
.\.venv\Scripts\python.exe train_weekends.py
```

The first command acquires the six fixed 2024 Race sessions with lap timing and
race-control messages; it requests neither telemetry nor weather. After those
exports exist, train_weekends.py works offline. The protocol/source hashes and
six-driver/event identities are checked; missing required sources stop the run.
The original Bahrain source/model/predictions/metrics are preserved and
hash-checked around the expanded training run.

The expanded model is models/weekend_ridge.joblib. Audit, predictions, metrics,
coverage, domain checks and provenance are under reports/weekends/. The saved
model's prediction roundtrip is verified. See
[WEEKEND_PROTOCOL.md](WEEKEND_PROTOCOL.md) for fixed event roles and
[WEEKEND_MODEL_CARD.md](WEEKEND_MODEL_CARD.md) for the full result.

## Model evidence

Both studies predict actual next lap t+1 after completed t, conditional on both
laps being eligible in the same known stint/compound. Features use observed timing
and reported metadata through t. Telemetry is not a predictor in either model.

The first pilot contains 93 pairs: 52 train, 17 validation, 22 test and 2 purged.
Ridge alpha 10 has final-test MAE 0.1439 s versus last-lap persistence 0.1559 s and
trailing median 0.1806 s. Ridge loses to persistence for LEC. This remains the
original result in [MODEL_CARD.md](MODEL_CARD.md) and
[MODEL_PROTOCOL.md](MODEL_PROTOCOL.md).

The separate expansion contains 1,967 raw laps, 1,664 eligible laps and 1,536
consecutive examples from six drivers. Whole Bahrain/Australia/Japan weekends
provide 708 training examples; China provides 232 validation examples. Alpha 10
is selected on China; the training-fitted pipeline remains frozen for Miami/Imola.

| Whole-weekend test method | MAE (s), 596 targets |
|---|---:|
| Ridge correction to last lap | 0.280751 |
| Last-lap persistence | 0.288183 |
| Trailing median, up to 3 laps | 0.287437 |

The pooled gain over persistence is only 0.007432 s MAE. Ridge loses in Miami
(0.281009 versus 0.280095 s); pooled driver results also include baseline wins.
Two held-out weekends with the same six drivers provide limited event-transfer
evidence, not unseen-driver or full-season validation. The newer and pilot MAEs
use different targets and should not be compared as a before/after improvement.

## Validate

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

[VALIDATION.md](VALIDATION.md) records executed checks, browser evidence and limits.

## Source and artifact map

- download_data.py, analyze_data.py and race_analysis.py: original acquisition,
  timing audit, stint summaries and comparisons matched by race lap.
- export_telemetry.py and telemetry_analysis.py: offline native samples, quality
  diagnostics, estimated distance and supported alignment.
- pace_model.py and train_model.py: original consecutive-lap pilot and artifacts.
- download_weekends.py: fixed six-event timing/race-control acquisition.
- weekend_model.py and train_weekends.py: event-local examples, whole-weekend
  selection, baselines, metrics, domain checks and saved expanded model.
- app.py, telemetry_ui.py, model_ui.py and weekend_ui.py: dashboard views/downloads.
- data/bahrain_2024_ferrari_laps.csv: original 114-record timing export.
- data/bahrain_2024_ferrari_telemetry.csv: 41,286 native car samples.
- data/telemetry_manifest.json: acquisition hashes and per-lap quality.
- data/weekends/: six raw race exports and source_manifest.json.
- models/pace_ridge.joblib and models/weekend_ridge.joblib: separate fitted models.
- reports/ml/ and reports/weekends/: separate experiment tables/manifests.
- CASE_STUDY.md: original Bahrain findings and interpretation limits.
- MODEL_CARD.md/MODEL_PROTOCOL.md and WEEKEND_MODEL_CARD.md/WEEKEND_PROTOCOL.md:
  separate evidence and fixed evaluation procedures.
- PROJECT_STATUS.md and VALIDATION.md: milestones and executed verification.

## Evidence boundary

Analysis-eligible does not mean traffic-free. Equal race lap/compound do not
establish equal tyre age, fuel load or pace management. Reported tyre usage counts
laps, not measured wear.

Speed, throttle and boolean brake come from public Bahrain data. Distance is a
trapezoidal speed integral from the first recorded sample, not measured circuit
position. Missing boundaries are not filled. Excessive sampling gaps/invalid
speed withhold later distance; gaps also break elapsed-time chart lines. The
explicit 1.0 s gap/boundary tolerance is an analysis choice.

Optional 10 m alignment interpolates over common supported distance only.
Speed/throttle use linear interpolation; brake uses preceding samples within
valid runs. Different sample origins and driven paths limit spatial comparison.
Sparse sampling cannot establish exact braking points or corner time gains.

ML predictions are estimates. Target eligibility is established retrospectively;
real-time delivery and flag revisions are untested. One-step replay observes
preceding held-out lap times without retraining. Nearby errors are correlated;
no accuracy percentage, calibrated interval or causal degradation estimate is
provided. The timing-only expansion has no per-lap LapStartDate timestamps.

Private fuel mass, tyre wear/temperatures, brake pressure and engineering setup
are unavailable. No team affiliation is claimed.

## Further experiments

Keep the declared test results fixed. Additional circuits, conditions or drivers
need a separately declared study; model changes need fresh held-out evidence.
Computer vision for annotated pit-stop video and drifting remain separately
scoped later work. Public deployment and FastAPI/React integration are also future
options once the analytical scope is stable.

Data tooling: [FastF1](https://github.com/theOehrly/Fast-F1).
