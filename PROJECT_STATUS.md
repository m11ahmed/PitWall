# PitWall project status

As of October 2, 2026, the local analysis/telemetry foundation, first ML pilot and
separate whole-weekend evaluation are implemented. The detailed Bahrain case
study covers LEC/SAI; the expanded study covers six drivers from Ferrari, Mercedes
and Red Bull across six selected 2024 races. Everything lives in G:\PitWall,
independently of the banking project.

Completed:

1. Separate Python environment, public data acquisition and local FastF1 cache.
2. Bahrain timing audit: 114 records, 101 eligible laps and 13 excluded.
3. Pace charts, tyre stints and comparisons on shared eligible race laps.
4. Offline export of 41,286 native Bahrain car samples with provenance/quality.
5. Speed/throttle/brake traces, elapsed time, estimated distance and optional
   common-distance interpolation with explicit sampling-gap handling.
6. First Ridge next-lap pilot with consecutive examples, chronological selection,
   persistence/median baselines, saved model and held-out per-driver results.
7. Separate six-event timing-only acquisition: 1,967 raw records, 1,664 eligible
   laps and 1,536 event-local consecutive examples for six selected drivers.
8. Whole-weekend training on Bahrain/Australia/Japan, alpha selection on China,
   and frozen evaluation on Miami/Imola, with original pilot hashes preserved.
9. Separate expanded model, verified saved-prediction roundtrip, event/driver
   metrics, coverage, exclusion counts, domain checks and artifact manifests.
10. Dashboard Study selector: Bahrain case study and Across weekends, with
    independent controls and downloads. Documentation keeps each study distinct.
11. Separate local Git repository, byte-preserving checkout rules and G-drive
    setup script. A clean repository export and new isolated environment passed
    all 86 tests, saved-model replay and desktop/mobile browser/download checks.
12. Engineering overview with selected screenshots and retained negative results;
    concise GitHub README and separate full reproduction guide.
13. Public repository: [m11ahmed/PitWall](https://github.com/m11ahmed/PitWall),
    with main pushed and the remote commit verified against the local repository.

Telemetry coverage is narrower than timing eligibility: 64 of 101 eligible laps
meet conservative distance-support checks. Distance/alignment are estimates.
The expanded acquisition contains lap timing and race-control messages, not
multi-event telemetry or weather. Its per-lap LapStartDate values are missing.

The first model has 52 train, 17 validation and 22 test targets plus 2 purged
pairs. Test Ridge MAE is 0.1439 s versus persistence 0.1559 s; Ridge loses for LEC.
The separate weekend model has 708 train, 232 validation and 596 test examples.
Alpha 10 is selected on China, and the training fit remains frozen. Pooled test
MAE is 0.280751 s versus persistence 0.288183 s and median 0.287437 s. Ridge loses
in Miami and for some drivers. Pilot and expanded errors use different targets.

This provides limited evidence for transfer to two later weekends with the same
six drivers. It does not validate all circuits, unseen drivers, the entire season
or live forecasting. Eligibility remains retrospective and conditional; no model
predicts future eligibility, pit stops or incidents. Baseline wins remain visible.

Executed tests and browser checks are recorded in [VALIDATION.md](VALIDATION.md).
The preserved pilot is documented in [MODEL_CARD.md](MODEL_CARD.md) and
[MODEL_PROTOCOL.md](MODEL_PROTOCOL.md). The expansion is documented in
[WEEKEND_MODEL_CARD.md](WEEKEND_MODEL_CARD.md) and
[WEEKEND_PROTOCOL.md](WEEKEND_PROTOCOL.md).

Next work, as separately scoped milestones:

- Preserve both declared results; use fresh held-out evidence for model changes.
- Broaden circuit/condition coverage through a separately declared experiment.
- Evaluate real-time feed availability if adding a live forecasting flow.
- Build pit-stop video timing with accessible footage and annotated evaluation.
- Develop a separately scoped drifting computer-vision study.
- Consider FastAPI/React integration and public deployment.

Current delivery is a runnable local portfolio with two trained, scored studies
and verified repository setup on Windows/Python 3.14.0. The code and portfolio
documentation are published on GitHub; dashboard hosting remains future work.
Public deployment and computer vision remain future work. No private team data,
causal performance conclusions or team affiliation is claimed.
