# PitWall continuation checkpoint

Updated October 2, 2026. Engineering presentation and GitHub publication are complete.

## Project and scope

PitWall is a separate motorsport data-science portfolio at G:\PitWall, aimed at
engineers and senior data scientists working in F1. Keep it separate from the
completed banking project. Favor free tooling, public accessible data and a
focused runnable scope. User knows Python, FastAPI and React; F1, drifting and
pit-stop computer vision are interests. Current implementation is Streamlit.

## Completed and verified

- Bahrain 2024 LEC/SAI analysis: 114 laps, 101 eligible, 13 excluded; 46 matched
  same-compound pairs. Audit preserves excluded records and reasons.
- Native public Bahrain telemetry: 41,286 car samples; 64/101 eligible laps support
  conservative distance alignment. Distance/interpolation are estimates.
- Original Ridge next-lap pilot: 52 train, 17 validation, 22 test, 2 purged.
  Test MAE 0.143864 s versus persistence 0.155909 s; Ridge loses for LEC.
- Separate six-weekend/six-driver 2024 study: 1,967 raw laps, 1,664 eligible;
  708 train targets from Bahrain/Australia/Japan, 232 validation from China,
  596 test from Miami/Imola. Drivers LEC/SAI/HAM/RUS/VER/PER.
- Expanded test MAE: Ridge 0.280751 s, last lap 0.288183 s, median 0.287437 s.
  Gain is small; Ridge loses in Miami and to a baseline for HAM/PER/VER.
- Both studies use alpha 10 selected on validation; preprocessing/model fit
  training only. Future eligibility is retrospectively established, not predicted.
- Dashboard has Study controls for Bahrain case study and Across weekends.
  Changing filters reads/re-scores saved forecasts; it does not retrain.
- Fresh Windows/Python 3.14.0 installation from requirements-lock.txt passed
  pip check and all 86 tests. Both saved models reproduced all 39/828 forecasts
  within 1e-10 s. Desktop/mobile and CSV download checks passed.
- Git repository on main. Initial implementation commit: 090299d. Exact clone
  and ZIP bytes were verified. .gitattributes prevents line-ending conversion;
  hash-protected source/protocol/data/model/result files must retain their bytes.
- scripts/setup.ps1 supports an explicit Python 3.14 executable and keeps venv
  and temporary installation files on G:. Windows py launcher did not work here.
- Working environment remains .venv. Temporary fresh verification environment
  was removed after checks. Local cache and scratch work are ignored by Git.

## Resume the dashboard

```powershell
Set-Location "G:\PitWall"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://127.0.0.1:8501/. The server may need restarting after closing the app
or terminal; all study files/results are saved on disk. No download or retraining
is needed to explore the current dashboard.

## Current release and next options

The engineering overview, reviewer-oriented README and full reproduction guide
are complete. The user explicitly authorized creation/publication under m11ahmed.
Public repository: https://github.com/m11ahmed/PitWall, default branch main.
Origin is configured; pushes were verified against local commits. No public
Streamlit demo has been deployed.

Read ENGINEERING_OVERVIEW.md for questions, screenshots, baseline comparisons,
negative findings and observation/estimate/private-data distinctions. README.md
is the GitHub entry point; REPRODUCING.md carries full commands. GITHUB_RELEASE.md
records repository metadata and publication procedure. The frozen analytical
results and model bytes are unchanged.

Optional next work: select and verify free public dashboard hosting. Computer
vision and new model experiments remain separate milestones. Preserve declared
test results; changes to models need fresh held-out evidence. Do not conflate
public GitHub publication with a deployed interactive dashboard.

For exact evidence read CASE_STUDY.md, WEEKEND_MODEL_CARD.md, VALIDATION.md and
reports/release/verification.json. The compact project archive is
tmp/PitWall-portfolio.zip; .venv/cache/tmp and local secrets are excluded.
Inspect git status/log and origin/main for the latest release checkpoint.
