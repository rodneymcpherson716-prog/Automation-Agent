The new-shot recorder preserves the evidence needed to check future shots.
Historical learning rows are not repaired, relabelled or promoted by this work.

During a normal controller run, each new shot now saves these files in its
session directory under `wgt_debug/runs/`. Nine-hole runs retain their existing
`hole_01_`, `hole_02_`, etc. prefixes, preventing one hole's records from
overwriting another's:

- `shot_01_before_capture.json`: the final state frozen after the last turn
  check and before input. Includes selected club/range/mode/units, target
  distance, lie and displayed percentage, wind, elevation, raw OCR evidence,
  ownership/turn context and the pre-shot statistics baseline.
- `shot_01_swing_capture.json`: observed yellow-meter power and the current
  shot's DING trace, saved after the existing accuracy click has finished and
  before waiting for the result. This remains available if result verification
  subsequently times out. No recording calls were inserted between the drag
  release and the accuracy click.
- `shot_01_capture.json`: a combined before/after evidence bundle, requested,
  observed-meter and attributed game power kept separate, DING target/edge and
  samples, carry/roll, attributed result owner, remaining hole distance/lie,
  verification checks, and an explicit list of missing fields.
- `shot_01_result.json`: existing result log, now also containing the archived
  DING trace and combined capture. The same capture is included in the learning
  row's evidence when the result is saved to the shadow-learning database.

Before-shot predictions also retain a deep copy of the source state and stats
baseline. Later condition reads and post-shot attribution cannot rewrite the
archived pre-shot state. New result-log imports recheck their capture against
the source before/after observations. A saved `complete=true` flag cannot make
missing evidence valid. Existing IDs remain idempotent and existing rows are
skipped; this does not repair old records.

Completeness and model-training eligibility answer different questions. The
capture is complete only when the requested before/swing/after evidence is
verified, including the current shot's DING trace and remaining hole position.
The existing distance-model eligibility still requires verified conditions,
attributed changed statistics, actual power and the verified stroke/motion
transition. It does not require a measured remaining hole distance to fit a
carry/total-distance response. Neither status enables live learned control.
No sample requirements, freshness checks or fitting thresholds were lowered.
An explicit foreign statistics owner now vetoes a solo header in both the
learning adapter and the existing live power-calibration recording path.

Unknown fields remain unknown. Current final-position evidence is a remaining
distance explicitly linked to the hole plus an observed lie; there are no
verified absolute course coordinates. A hole-out/scorecard is not recorded as
a measured zero distance. Completion screens or unreadable HUD/stat fields
can therefore produce an incomplete capture. Missing before statistics, lie,
wind, elevation, ownership, stale results and missing DING evidence are listed
individually. This recorder exposes such collection gaps rather than filling
them from assumptions or relaxing training checks.

DING evidence describes the green edge when the existing tracker triggered a
click. It is not a verified exact mouse-down position, ball-impact accuracy or
wind-offset efficacy. The recorder does not change the click target or timing.

Offline validation uses synthetic shots only in temporary sessions/databases.
The tests exercise the actual controller recording path, archive ordering,
mutation isolation, persistence, reimport, missing facts, foreign ownership,
stale stats, and unknown final positions. Existing saved-image reader tests
separately exercise OCR. These establish recorder behavior offline. Confirming
that a new live shot yields every field still requires controlled gameplay,
which was not launched or performed for this work.

Relevant files: `shot_capture.py`, `execution_engine.py`, `automation_learning.py`,
`learning_data.py`, and `test_shot_capture.py`. The existing swing,
release, accuracy, F9, transition, 60-second clock and nine-hole protections
remain in place. Unvalidated automatic bunker swings and learned live
accuracy changes remain blocked.

Validation completed on 2026-11-14: the final COMPLETE offline suite passed
356 tests, with 0 failures, 0 errors and 0 skips in 363.971 seconds. This
includes 11 new recorder tests. The runner reported 0 native-input or real
capture attempts. WGT was not launched. Full results are saved in
`wgt_debug/new_shot_recorder_final_full_20261114.log`. SHA-256 checks confirm
the existing learning database/report, automatic calibration data and manual
bunker reference were not changed. Actual new-shot HUD capture still needs
controlled gameplay validation; no such live result is claimed here.
