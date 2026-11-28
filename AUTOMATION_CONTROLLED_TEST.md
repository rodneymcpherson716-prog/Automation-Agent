# Controlled WGT test — November 11, 2026

The controller remains `execution_engine.py`. The November 10 backup and its
native drag, release, accuracy, emergency-stop and motion code are preserved.
The backup SHA-256 is
`6439dec84234c6a4b73d55bddb6d35ded649242fd9150df920c5cbfd8abdc626`.

## Current calibration

`wgt_calibration/game_context_profile.json` now loads reviewed local MSS
references for the four-player Green/Gold team HUD at 1288x728:

- Green: Tigerwould52 and clarencelogan.
- Gold: BrianColley02 and PTSUN.

Eight authentic active/compact name glyphs, their hashes, highlighted panel,
team colours, turn clock and ready controls verify the actor. WATCHING TEAMMATE
and WATCHING OPPONENT block our swings. A highlighted Tiger row without ready
Swing/meter controls also blocks input, including ball flight and loading.
Unknown or covered names fail verification. This profile is specific to the
reviewed team layout and roster; other rosters and solo layouts still require
actual local calibration. Policy unit tests for those formats are not proof
that their live screen readers are calibrated.

WGT's AS badge means All Square, a match score. It is never used alone as an
Alternate Shot format detector. Score changes to 1 UP/1 DN are supported by
reading the player names, teams and highlighted actor independently.
Official rules: https://www.wgt.com/help-flash.aspx

Real saved tee and second-shot screens are replayed offline. The opening shot
requires explicit double dashes on all four stroke counters and ready controls;
the selected lie must verify TEE for the 100% tee-power policy. A reviewed full
2ND SHOT label template handles the tiny ordinal that OCR otherwise misreads.
Other labels require confident agreement. Missing readings stop input.

## One controlled live command (PowerShell)

Load the actual hole and keep WGT visible at the recorded 1288x728 size. The
current screen observed during this review was the main lobby; the controller
does not navigate that lobby or start another match.

```powershell
& '.\start_agent.cmd' --controlled-test --activate --countdown 15 --max-shots 15 --max-seconds 900 --concede-feet 2.0
```

This raises the existing WGT window after 15 seconds. Windows may require one
verified native-title-bar click to focus it. The controller waits for a verified
own turn, with a readable clock and sufficient remaining time for the whole
stroke. It is bounded to 15 strokes and 900 seconds. F9 immediately latches the
emergency stop and releases a held button; Ctrl+C stops from the console.
Focus loss or a moved/resized/minimized window stops input. Do not launch two
live controllers; an OS-owned lock prevents that.

The controller does not drag the aiming flag, change the camera, or resize WGT.
It reuses the recorded meter anchor, automatic feedback drag/release and
accuracy crossing. Club selection uses the existing bottom-left wheel control.
Windows ClickLock is temporarily disabled only for live control and restored
on normal exit/F9/errors. Observe/check modes leave it unchanged.

## Local evidence

For passive observation with no game inputs:

```powershell
& '.\start_agent.cmd' --observe --observe-seconds 60
```

MSS raw/annotated captures, OCR words/confidence, actor/team reports, clocks,
turn history, motion and decisions are saved under `automation_debug/runs/`.
The latest-run pointer is in `automation_debug/latest_run.json`. No screenshot upload
is needed. Cold shot OCR can take several seconds. Before live input, an
additional capture may renew turn evidence only if all measured HUD/target
pixels are byte-identical and actor, roster, label and ready controls agree.
Changed measurement pixels retain the old timestamp and cannot authorize input.
The turn-evidence age limit remains four seconds.

## Concessions and learning

`wgt_calibration/game_policy.json` stores the configurable two-foot threshold.
`--concede-feet` overrides it for one run; zero disables it. Policy tests cover
2.0 feet, 2.01 feet, prohibited formats, teammates and opposing shared balls.

Automatic concessions require verified opposing ownership, green lie, distance,
permitted format and an enabled, hash-calibrated actual CONCEDE PUTT control.
That control has not yet been captured, so real concession clicks remain
blocked. The controller never substitutes a hole/match concession. Before an
eligible press it atomically journals the attempt; ambiguous results and restarts
cannot retry the same logical putt. Two fresh owner-bound results must verify
that WGT recorded it.

Power estimates use only verified own-shot outcomes. Opponent statistics and
externally played shots are not treated as bot successes or training evidence.
Hazard strategy requires explicit verified current-aim distances; this review
does not establish automatic hazard mapping, under-par performance, all-format
live support or complete-hole success.

## Verification

The full suite output is `automation_debug/offline_team_hud_verified_20261111.txt`.
All 177 tests passed in 554.275 seconds. It includes the original regression suites, preserved-control AST comparison,
real tee/second-shot/flight/loading/opponent frames and fresh-capture guards.
`--check` passes and reports the layout calibrated. The passive run
`20261111_072900_e7394742` recorded the lobby, rejected the missing roster and
sent no input. No live swing, concession or hole completion was verified.

Earlier diagnostic notes are retained in
`repair_backups/alternate_shot_hud_20261111/WGT_CONTROLLED_TEST.before.md`.


## Solo profile selection repair, November 11 evening

An installed team calibration previously suppressed the fixed solo player and
stroke OCR even when --solo-confirmed was explicitly selected. The fallback
now requires the actual calibrated navy solo panel and an invalid team reading.
Exact player OCR, ready Swing/meter, two stable frames, absence of multiplayer
conflicts and fresh unchanged measurement pixels remain mandatory. The stored
team calibration is preserved and is not claimed to calibrate the solo frame.
Default runs and actual team frames retain the multiplayer reader and guards.
F9, the swing/release/accuracy routines, aim and camera controls are unchanged.

The solo controlled command (only for the user's single-player session) is:

```powershell
& '.\start_agent.cmd' --controlled-test --solo-confirmed --activate --countdown 15 --max-shots 0 --max-seconds 0 --concede-feet 0
```

Leave WGT active and avoid mouse/keyboard input during play. F9 stops input.
Diagnostic evidence is saved automatically in automation_debug/runs. Offline and live
outcomes of this repair are recorded separately; no completion is implied here.

Full installed suite: 183 tests passed in 488.994 seconds. Output: `automation_debug/offline_solo_profile_20261111.txt`. Current MSS probe captured the Windows lock screen, so no live input was sent and no new swing/hole success is claimed. Unlock Windows before the controlled command. Review: `automation_debug/solo_profile_review_20261111.json`.

## Solo OCR freshness repair

The November 11 23:30 run verified the solo account and controls but background context OCR took 6–7 seconds, exceeding the unchanged four-second evidence budget. In an explicitly confirmed solo session with the independently verified solo header and player, context scans now read the lie and watching banner. Actual team geometry still selects the team reader. Optional hazards/concessions/stat ownership are left unverified instead of scanning scenery. Read-only live probes reached permission without input; swing/release/F9 code is unchanged.

Installed suite: 190 tests passed in 258.808 seconds (`automation_debug/offline_solo_freshness_20261111.txt`). Actual controlled attempt `20261111_234244_b2c76fb0` verified the solo turn and selected 9.0D/282 yd from 3W/205 yd, then stopped when Microsoft Edge became foreground. No backswing, accuracy click, movement or hole completion was verified. Keep WGT restored and foreground and avoid other input during the next test; F9 remains active. Review: `automation_debug/solo_freshness_review_20261111.json`.

## Club hover repair, November 12

The real run `20261112_015125_4dcc7340/context_00021.png` shows CLUB SELECTION open while the cursor remains over the club range, hiding Swing. The controller now parks the pointer inside the idle meter at the start of each shot and immediately after a club wheel event, before HUD verification. Parking never presses a button and refuses movement on F9, lost focus or a held left button. Required Swing/turn/meter checks remain mandatory. The manual drag, release, accuracy and F9 routines are unchanged. Five targeted hover regressions passed.

## First automatic green, November 12

Run `20261112_020415_9840572f` executed the tee, 121-yard second shot and 22-yard pitch with feedback-controlled power, verified release and accuracy input. It stopped at the first putting HUD because the course flag interrupted the badge border and selector arrows spoiled the putter FT crop. A final bright-border pass bridges that narrow gap while still requiring side/bottom borders and blue fill. The FT crop now excludes arrows. Two real frames independently read 17.5 FT target, 30 FT range and +0.6 IN. Five new tests pass, including hidden digit/missing unit rejection and actual pitch-motion/putter-HUD transition replay. No hole completion is claimed by this note.

The full installed 200-test suite passed in 299.674 seconds (`automation_debug/offline_putt_badge_20261112.txt`). A subsequent MSS capture placed the badge lower on screen. The search now includes that position without camera input. A visible IN glyph variant from the independently OCR-verified earlier +0.6 IN badge supports the blurred lower label, retaining correlation, competing-label margin and numeric/arrow checks. All seven putt layout tests pass, including hidden-unit rejection.

The additional IN template is restricted to elevation labels at its calibrated large badge scale. Affected-suite testing found two older-label ambiguities before that restriction; both failing checks plus all three old conditions checks now pass. All eight putt-layout checks pass, including scope and hidden-unit rejection. The control/F9 routines remain unchanged.

## Post-OCR putting freshness

The animated lower green view took longer than the unchanged four-second age budget to read. A fresh MSS frame now independently revalidates each measured putt glyph, decimal/unit, player/stroke, hole/par, range, wind-speed text, green speed, elevation arrow and badge borders. It also requires live Swing/meter and a newly detected wind vector within the existing tolerance. Grid/scenery pixels do not renew evidence. Missing or changed glyphs refuse renewal. The captured-time update is logged with its original OCR time and a fresh image, and it never increments the stable-frame count. Two consecutive individually fresh complete putt readings may span slow OCR, while other shot modes retain their existing time window. Input guards use the current player reading. Thirteen new offline tests pass, including F9/focus, decimal loss, hidden unit, changed stroke, stale proof and slow-OCR cases.

Final installed freshness suite: **216 tests passed**, 472.694 seconds, `automation_debug/offline_putt_freshness_20261112.txt`. Three actual automatic strokes are verified from the 02:04 run (the third by repaired-reader replay of actual video/HUD evidence). Current shot remains the 17.5 FT putt. Latest attempt `20261112_031047_0b66a193` stopped before putting when Microsoft Edge took focus. No putt or hole completion has been verified. Keep WGT active and avoid interacting with other apps throughout a controlled live run. Consolidated evidence: `automation_debug/one_hole_progress_20261112.json`.

## Normal 60-second shot-clock constraint

The user reports a normal 60-second shot clock. Its budget includes observation, OCR, club selection, power, release and accuracy input. A cold putt read has taken about 50 seconds; this version is NOT validated for a 60-second timed hole. Fresh HUD evidence does not mean a fresh clock. The timer is anchored to the original captured countdown, retained through lie/readiness changes and temporary missing identity, and reset only on a verified new hole/stroke/actor/ball identity. A known clock without a remaining-time reading or tracked deadline blocks the stroke. An already tracked deadline can still conservatively budget time when the current timer glyph is unreadable.

The current single-player screenshot has no verified countdown. Do not infer 60 seconds remaining when attaching to an existing turn. The untimed solo path is preserved; the user must clarify whether the current solo test is timed. No new live test was performed for this clock repair. Eight new clock tests passed.

Clock patch validation: 47 affected clock, role, putting freshness and preserved-control checks passed (8 new clock cases), `automation_debug/offline_shot_clock_20261112.txt`. The earlier full 216-test pass predates this small clock patch. No timed live play was performed or verified. Review: `automation_debug/shot_clock_review_20261112.json`.

## Nine-hole controller and putter-range correction

`round_manager.py` supplies sequencing through V48's `--holes 9` option. Multiple
holes require `--solo-confirmed`. Each completion must be observed before a
single verified scorecard Continue click; the next hole number and first stroke
must then agree across ready HUD reads. The ninth completion stops without a
tenth-hole click. An ambiguous transition stops without a repeated click.
Per-hole diagnostic prefixes and calibration shot IDs preserve separate records
when the stroke-attempt counter restarts on the next hole.

The putter reader separates its number and FT label. It excludes range arrows,
checks complete centred digit components, and requires unthresholded and binary
OCR passes to agree with the visible glyph count. FT requires two explicit reads
and both letter components. A missing zero, conflicting number/unit, clipped
boundary or unverified reading cannot authorize a swing. The fast target reader
also compares brightness masks with an independent grayscale read; an older
faded 5.9-foot target is handled by the established reader when those disagree.
Grid animation may only renew putting evidence after a fresh semantic badge
read agrees with every prior measurement and the other HUD fields still match.

The full-swing `refresh_turn_frame` method, native swing/release/accuracy code,
F9 watcher, manual recording and accuracy calibration remain unchanged. No
aiming flag or camera input was added. The user clarified that this solo test is
untimed; the 60-second limit applies to multiplayer.

The final correction report and individual module results are saved under
`automation_debug/putter_range_correction_20261112.md` and
`automation_debug/nine_hole_final_regressions/`. These distinguish offline fixture and
simulation checks from actual gameplay. Live nine-hole play is held at the
user's request. No new putt or nine-hole completion is claimed.

Prepared command for a later controlled live test, **not executed**:

```powershell
& '.\.venv\Scripts\python.exe' -B -u '.\execution_engine.py' --controlled-test --solo-confirmed --holes 9 --activate --countdown 10 --max-shots 0 --max-seconds 0 --concede-feet 0
```

It counts nine verified hole completions starting with the current hole, which
can already be in progress. Keep WGT foreground during play. F9 stops input.
Actual scorecard transitions and nine-hole completion still require a live test.

The user confirmed a 60-second shot clock for team alternate-shot play on
2026-11-13. `ALTERNATE_SHOT_TURN_SECONDS` records that session limit. The bot
still requires the visible remaining countdown; it never starts a fresh 60
seconds when OCR finishes or a club/lie changes. Existing timing accounts for
OCR elapsed time, the complete drag and accuracy phase, and a three-second
margin. Missing, expired or above-60 alternate-shot timer readings block a stroke.
Historical 90/83-second screenshots remain unchanged and readable, but cannot
authorize a stroke under this confirmed session limit. Solo timing and the
bunker block are unchanged. All 133 affected clock, policy, multiplayer HUD,
context, learning and bunker regressions passed with zero native input or real
capture attempts; see `automation_debug/alternate_shot_60_tests.log`.
