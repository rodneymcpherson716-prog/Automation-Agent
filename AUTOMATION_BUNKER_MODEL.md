# Bunker recovery

The normal round controller stops in bunkers. For a user-requested live
calibration, `--one-shot --solo-confirmed --bunker-calibration HOLE STROKE`
enables a single full lob-wedge trial from a verified 40–50% sand lie. It uses
the geometric midpoint of the lowest and highest lie-hypothesis powers, with
live distance, wind and elevation. The controller checks the solo owner and
rereads the conditions before input. The trial saves the meter power, next
lie and any attributable Shot Stats. It does not fit a reusable sand curve.

`--bunker-reference PATH` uses a recorded successful recovery with the same
club, mode, range and displayed lie. It caps the layup at the measured travel
and scales the recorded meter power for current elevation and wind. The
reference is one observed shot, so the layup focuses on clearing the bunker.

## Evidence and data flow

`context_manager.verified_bunker_readings` verifies the bunker label and displayed
percentage using two OCR variants at at least 85% label/range confidence.
The existing 93% OCR result concerns text recognition, not power accuracy.
`game_state.parse_context` stores `BUNKER | 40–50% LIE` as `lie='BUNKER'` and
`lie_percent=[40,50]`. Only an explicit `LIE LOSS:` label sets
`lie_loss_percent`. Conflicting percentages abstain and changing percentages
reset the consecutive-frame context count.

`GameCoordinator.inspect` copies both fields independently to the shot state,
clearing absent values from previous shots. Previously `PowerCalibration.plan`
only used `lie_loss_percent`, so the displayed percentage had no effect and a
bunker could receive the generic distance/selected-range estimate. Renaming
the display to loss would introduce an unsupported physical interpretation.

`PowerCalibration.plan` now routes sand to `terrain_handler.bunker_power_estimate`.
`automation_core.plan_power` rejects sand instead of providing a fallback. The main
controller saves the diagnostic estimate and stops before selecting a club.
The club selector and final `GameCoordinator.arm_stroke` also block sand,
including raw bunker/sand text with unverified percentages. The controller
also rejects a blocked plan if sand appears on its later reread.
Already observed sand is blocked before that reread can erase its raw evidence.
The final stroke guard snapshots percentage lists and strategy values so even
in-place changes require replanning.

## Measurements versus estimates

The unchanged `wgt_calibration/bunker_reference_manual.json` contains the
player-reported LW hole-out: 16 yards to the target, selected range 52 yards,
26% power, +4 feet elevation, 13–15 mph wind, and displayed 40–50% lie.
Carry is null. Wind direction, carry, roll and impact accuracy are unknown.
The reference remains `use_for_automatic_power=false`; it is displayed as
context and never fitted, scaled or treated as a carry measurement.

There are zero verified observations labelled BUNKER in automatic calibration.
An older LW record with approximately 26% power and 0.83-yard recorded carry
has no lie label. It cannot establish bunker calibration or the carry of the
manual hole-out. Existing observation files are unchanged. Newly recorded
observations preserve the lie display, separate explicit loss, full wind
vector/range, selected mode/range, target distance and actual statistics for
future analysis. The generic empirical fitter refuses all bunker states.

## Provisional sensitivity model

For a verified wedge and airborne shot in yards:

    effective_yards = (strategy_target_yards + elevation_feet / 3)
                      * (1 + 0.003 * wind_screen_dy * mean_wind_mph)
    estimated_power = effective_yards / selected_club_range_yards

Distance/range linearity, elevation feet/3 and 0.3% per mph are assumptions.
They have not been measured for this bunker mode. Screen-top wind is treated
as tailwind as in the existing planner; its relationship to current aim is
also unverified. Crosswind is reported with no inferred carry correction or
aim movement. Gust endpoints produce a sensitivity range. No coefficient is
fitted to make the 26% reference match.

The displayed lie stays uninterpreted. The output includes separate scenarios:

- No conversion of the displayed percentage (the nominal estimate).
- If the display means retained effectiveness: divide by the displayed range.
- If the display means loss: divide by one minus the displayed range.

These scenarios illustrate uncertainty; none establishes WGT semantics or a
validated confidence interval. A separately supplied explicit loss label gets
its own linear hypothesis; it does not change the nominal estimate either.
Even an explicit loss label does not establish that recovery distance scales
linearly with that percentage. No loss adjustment is selected or combined.
A 100% hypothetical loss has an unbounded endpoint,
represented as null. Estimates outside the controlled meter range are flagged,
not silently clipped. Missing, nonfinite, inconsistent or invalid inputs fail.

For example, 16/52 is 30.77% before conditions. Adding +4 feet gives 33.33%.
Under the assumed wind formula, 14 mph directly tail/head gives approximately
31.93%/34.73%, before any lie hypothesis. These are diagnostic calculations,
not suggested replacements for the successful 26% shot. Lip clearance, flight
shape, sand extraction, landing surface and roll remain unresolved.

Diagnostic bunker plans return `power=None`, a separate `estimated_power`,
`automatic_swing_allowed=false`, and zero fitted calibration points.
`observed_inputs` records the distance, strategy target, club/mode/range,
elevation, lie labels and wind components used. These are caller-supplied
readings, not independently verified measurements by the model. `assumptions`
identifies the unvalidated coefficients and unknown carry/roll split. Output
copies cannot mutate the source shot state or the loaded manual reference.

Offline regressions include a 240-case combination matrix: 4/8/16/32 yards,
20-30/40-50/70-80/100-100% displayed lies, -4/0/+4 feet elevation, and calm,
headwind, tailwind, crosswind and diagonal strong-wind cases. They check the
diagnostic calculation and block together, not simulated success in WGT.

The continuation review ran all 341 offline tests: 340 passed and one historical
whole-file snapshot assertion rejected the intentional bunker edits. That check
now preserves all unaffected control definitions and module statements while
allowing the revised bunker estimator and final guard. All 88 tests in the
affected learning, bunker-power and context suites passed on rerun. Full-run
and rerun logs are `automation_debug/bunker_review_full_offline.log` and
`automation_debug/bunker_review_final_regressions.log`. Both runs reported zero native
input or real capture attempts. The manual reference, automatic calibration
data and nine-hole sequencer were checked unchanged by SHA-256.

## Validation needed before resuming through bunkers

Collect independently verified manual recovery measurements across several
distances, powers and displayed lies for the same club, range and shot mode.
Record carry and roll separately, wind speed and direction, elevation, actual
meter power and impact accuracy, and any lip/landing constraints. Start with
repeatable calm/level cases so lie effects can be separated from other effects.
Then check wind directions and elevation changes independently. Resolve the
displayed percentage semantics with evidence, fit a contextual recovery model,
and assess it on held-out shots using an agreed acceptable landing-error limit.
Define a validated operating range and continue blocking outside it. Enabling
automatic bunker play requires a later explicit implementation and review;
passing algebra and mock-controller tests does not remove the block.

The existing drag/release, accuracy click, F9 watcher and nine-hole sequencing
are preserved. `run_offline_tests.py` blocks native input APIs and real MSS
capture while running the regression suites on saved images and mocks.

Manual measurements can now be collected with `Record_Manual_Bunker.cmd`.
See [manual bunker recording](AUTOMATION_MANUAL_BUNKER_RECORDING.md) for the BEFORE/AFTER procedure and the
precise field/attribution checks. These records stay in a separate manual-data
folder, explicitly excluded from automatic power. No bunker validation or
automatic enabling is implied by collecting them.
