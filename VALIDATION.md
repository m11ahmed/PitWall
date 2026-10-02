# Validation - October 2, 2026

## Verified

- All 86 focused tests passed in the final full suite: the original 65
  race-comparison, telemetry, ML pilot and dashboard checks, plus 15
  whole-weekend protocol tests and 6 expanded dashboard/export checks.
  Command: python -B -m unittest discover -s tests -v; 86 tests in 69.563 s.
- pip check found no broken dependency requirements after installing free
  scikit-learn 1.9.1. requirements.txt and requirements-lock.txt were updated.
- Original lap CSV SHA256 is unchanged:
  c0a921d8a638a8d788728ce9a6f908a0292c45269676936c63c1a88ac0c797b5.
- Timing retains 114 records, with 101 eligible and 13 excluded.
- Same-compound race comparison yields 46 pairs, or 47 when relaxed; swapping
  reference reverses the sign without changing the denominator.
- Offline telemetry extraction produced 41,286 native Source=car samples across
  all 114 laps, without interpolated edges or merged position data.
- Cached timing agrees with the original CSV. The manifest records source and
  telemetry hashes; the dashboard checks them when loading.
- Under the stated 1.0 s threshold, 64 of 101 timing-eligible laps qualify for
  continuous distance alignment. Quality failures remain explicit.
- Synthetic fixtures check known speed integrals, no extrapolation, missing
  boundaries/channels, binary brake stepping, gap withholding and delta sign.
- Real-data dashboard checks cover lap 21 context, lap 20 sampling gaps,
  withheld alignment, explicit line separators, window resets, single-driver
  selection, separate laps, reference order and missing-export empty states.
- Headless Chrome rendered the telemetry charts at 1440 px and 390 px, with no
  page errors or page-level horizontal overflow.
- Browser controls exercised shared/separate lap selection, elapsed-time traces
  and alignment withheld for the gap-containing lap 20 pair.
- Native CSV download contains 733 selected lap-21 samples and distinguishes raw
  channels from normalized/derived fields. The aligned CSV contains 535 grid
  points with driver names, over common support only.
- Telemetry desktop, aligned-comparison and mobile screenshots were inspected.
  Chart label encoding and mobile title/toolbar readability were corrected.
- Earlier browser verification exercised the race-pace, stint and audit views,
  compound restriction, and a 114-row audit download.

## ML pilot verification

- 93 true consecutive eligible pairs produce 52 training, 17 validation,
  22 test and 2 purged pairs under shared session-time cutoffs.
- Independent code/results review reproduced the metrics and confirmed that
  scaling uses only 52 training rows. Target/future perturbations do not enter
  earlier prediction features; final-target perturbation does not change alpha
  selection or any model predictions.
- Fixed alpha candidates 0.1/1/10/100; validation selects 10. The training fit
  remains frozen. No validation/test refit or test-driven feature change occurs.
- Test MAE: Ridge 0.143864 s, persistence 0.155909 s, recent median 0.180591 s.
  Ridge is worse for LEC individually; this remains visible in reports and UI.
- The saved local model reload reproduces all 39 validation/test predictions
  within 1e-10 s. Source, code, dependency and artifact provenance is recorded
  in reports/ml/manifest.json; source/artifact mismatches are rejected by the UI.
- Dashboard checks verify driver filtering, positive/negative model results,
  evaluation-window changes, individual replay, empty windows and stale data.
- Headless Chrome verifies test/validation controls, SAI replay and a 22-target
  prediction download retaining origin timestamps, predictions and baselines.
- Desktop 1440 px and mobile 390 px screenshots were visually inspected.
  No page errors, app exceptions or page-level horizontal overflow were found.
  Wide data tables scroll within their container on mobile.
- New screenshots: reports/dashboard_ml.png, dashboard_ml_predictions.png and
  dashboard_ml_mobile.png. Harness: tmp/ml_browser_check.cjs.

## Whole-weekend evaluation verification

- Six fixed 2024 race sources contain 1,967 recorded laps and 1,664 eligible
  laps for LEC, SAI, HAM, RUS, VER and PER. Excluded laps and retirement-shortened
  records are retained. Sources include timing and race-control messages;
  telemetry and weather were disabled for this expansion.
- Event-local consecutive examples total 1,536: 708 training targets from
  Bahrain/Australia/Japan, 232 validation targets from China and 596 test targets
  from Miami/Imola. Histories never continue across sessions or audit gaps.
- The declared protocol preceded acquisition/scoring. Alpha 10 was selected
  from the same four candidates using China only. Preprocessing and regression
  use only 708 training examples; validation/test never refit the model.
- Independent review reproduced counts, alpha, metrics and training scaler
  means. Adversarial tests exercise identity/chronology errors, missing events,
  driver coverage, duplicates, future-target changes and event boundaries.
- Pooled test MAE: Ridge 0.280751 s, last lap 0.288183 s, recent median
  0.287437 s. Ridge loses in Miami and to a baseline for HAM, PER and VER in
  pooled driver results. All failures remain visible; no difficult targets
  are trimmed. Equal-weekend MAE is also retained, giving both test events
  equal weight: Ridge 0.280779 s versus last lap 0.287311 s.
- Numeric training-range checks and categorical support are saved separately.
  Out-of-range inputs are reported rather than removed. These checks do not
  establish equivalent distributions or calibrated predictive uncertainty.
- models/weekend_ridge.joblib was reloaded immediately after local training;
  all 828 saved validation/test predictions match within 1e-10 s. The manifest
  hashes raw sources, all nine exported CSV reports, code, protocol and model.
  Report loading rejects missing/stale artifacts. The original Bahrain source,
  pilot model, manifest, predictions and metrics hashes remain unchanged.
- Headless Chrome exercised both Study views, 596 test targets, 232 validation
  targets, Imola/VER replay, the 596-row prediction download and 1,967-row lap
  audit download. Prediction downloads retain event identity and baselines;
  audit downloads retain Included and ExclusionReasons.
- Expanded desktop (1440 x 1100) and mobile (390 x 844) screenshots were visually
  inspected. Legends, axes, gap breaks and lap inspection are readable. No page
  errors, app exceptions or page-level horizontal overflow were found; wide
  mobile data tables scroll inside their own containers.
- Screenshots: reports/dashboard_weekends.png,
  reports/dashboard_weekends_predictions.png and
  reports/dashboard_weekends_mobile.png. Harness: tmp/weekend_browser_check.cjs.
- This milestone introduced no new package dependency. The previous successful
  pip check remains applicable to the unchanged environment.

Screenshots: reports/dashboard_telemetry.png,
reports/dashboard_telemetry_aligned.png and
reports/dashboard_telemetry_mobile.png. The reproducible browser harness is
in tmp/telemetry_browser_check.cjs; it uses the locally bundled Playwright and
existing Chrome. AppTest verifies app state and does not replace rendering QA.

## Fresh repository and environment verification

- A Git-index export contained 82 project files (7,850,436 bytes), with every
  index/export byte checked against the working source. It contained no FastF1
  cache or pre-existing project tmp directory. The environment was newly created
  inside the G-drive verification folder, without shared system site-packages.
- Installing requirements-lock.txt from the public package index succeeded with
  Python 3.14.0. pip check found no broken requirements. All 86 tests passed in
  this exported checkout in 103.612 s. The two missing-tmp assumptions in tests
  were fixed; they now create their own project-local temporary directory.
- scripts/setup.ps1 passed PowerShell parsing and ran successfully against this
  newly installed environment. It checks Python 3.14, installs the lock file,
  places temporary files in the project and runs pip check. Venv creation and
  first dependency installation were also executed independently during this
  verification; the script run confirmed the installed state.
- In the fresh environment, the byte-verified saved models reproduced all 39
  pilot and 828 weekend validation/test predictions. Maximum absolute difference
  was 1.4210854715202004e-14 s, below the declared 1e-10 s tolerance. Neither model
  was retrained and no original source/result/model bytes changed.
- A separate fresh dashboard on localhost port 8502 passed the pilot and weekend
  desktop/mobile browser checks. Downloads contained 22 pilot predictions,
  596 weekend predictions and 1,967 weekend audit rows. Validation/test switches
  and Imola/VER replay worked, with no page errors or page-level mobile overflow.
- The first audit-download check timed out during a phase change. An isolated
  audit download succeeded, and the complete weekend check passed after the
  harness explicitly waited for the Streamlit rerun to finish. No app change was
  required. Fresh mobile screenshots for both studies were visually inspected.
- .gitattributes disables line-ending conversion and accepts the original CRLF
  and EOF blank lines. Exact bytes are required by the provenance manifests.
  .gitignore excludes environments, cache, scratch files and local secrets.
- joblib 1.6.0 is now explicit in requirements.txt; its previously locked and
  installed version is unchanged. The original lock file and experiments remain
  intact. Machine-readable evidence: reports/release/verification.json.
- Historical browser harnesses and fresh-check screenshots in tmp/ are local
  verification aids excluded from Git. The earlier inspected screenshots under
  reports/ are included. AppTest and saved artifacts remain reproducible through
  the standard test command without installing browser automation dependencies.

## Limits

Fresh installation is verified on Windows with Python 3.14.0. Other operating
systems, physical-phone interaction and public deployment have not been verified. The original pilot covers one historical
race and two drivers. The separate expansion trains on three selected races,
selects alpha on one and tests on two later weekends with the same six drivers.
This supports only those stated cohorts and conditions, not a full season,
unseen drivers, all circuits or live forecasting. Neither study validates
held-out SOFT laps. Scoring conditions on actual future eligibility using final
archived flags; it does not predict that eligibility. Timing-only exports lack
per-lap LapStartDate, and archived data do not establish live feed latency or
revision behavior. Adjacent errors are correlated; no significance claim,
calibrated uncertainty or causal tyre-degradation/corner time-gain estimate
is made.

Native public channels have sparse, irregular samples. Estimated distance starts
at the first recorded sample, with missing boundary time disclosed. Similar
estimated distance does not prove identical circuit position or driven path.
The 1.0 s threshold is an analysis choice, not a provider quality guarantee.

## Run checks

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```
