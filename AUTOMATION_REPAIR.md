# WGT repair and verification

The launcher now runs the repaired `execution_engine.py`, backed by the
testable readers in `automation_core.py`. Direct execution of `automation_compat.py`
also forwards to that controller. Historical helper definitions remain
available to older scripts. No older versions, calibration, recordings, or
shot logs were deleted. Backups of replaced files are in `repair_backups`.

## What was reviewed

All 95 original Python sources, including the copy under `.github`, were
parsed for syntax and inventoried for repeated top-level definitions. There
were no syntax errors. The runtime review followed the actual launcher into
`automation_compat.py`, then compared the V47/V48 one-hole controllers, V43/V44
putting controllers, V45/V46 calibration/replay, V39-V41 shot recorders,
the rebuilt swing, distance/putting readers, and saved debug evidence.
Archived versions were surveyed for usable fixes; this is not a claim that
every historical script was exercised or every line was audited.

The previous default explicitly refused the final accuracy click and automatic
putting, despite many later-looking scripts in the directory. Several functions
were overridden repeatedly. V48 was not connected to the launcher.

## Repairs

- Distance: reuse the putting reader's white sign-edge locator at both tee and
  putting sizes. A cyan-only mask joined the sign to the sky. Read the numeric
  field separately from its explicit YDS/FT label and require agreement across
  OCR variants and consecutive ready frames. Never substitute the top-left
  hole length for the floating target's distance. Preserve putting decimals.
- Club/range: read and verify the displayed unit. Scroll only over the existing
  bottom-left club/range control, verifying the target and mode after each
  step. Oversized short-putt ranges stop if a smaller range cannot be verified.
- Power: the saved `power_check_*` images show yellow growing LEFT from neutral
  toward the rectangle's left limit. Measure that actual span. The first shot
  is no longer automatically forced to 100%. Stop on missing feedback,
  overshoot, exhausted travel, or timeout, and release the button.
- Recorded stroke: validate and reuse `last_manual_stroke.json`'s actual meter
  press, upward travel, drag cadence, and accuracy interval. Its two complete
  cycles were verified. It contains no distance, club, slope, or measured power
  labels, so it cannot justify V48's assumed "17.5-foot calibration". It is
  never overwritten. The window must match its saved geometry (1288x728,
  allowing at most ten pixels of frame variation).
- Release: native SendInput, correct Windows signatures and virtual-desktop
  coordinates, independent mouse-up fallback, and finally blocks around both
  held-drag and accuracy-click phases. No PyAutoGUI failsafe can block release.
- Accuracy: track a growing green run anchored at the released yellow edge,
  as visible in `green_return_review.png`. Reject static scenery, backwards
  jumps, late first observations, and implausible speeds. Predict the crossing
  from observed forward speed and send one down/up cycle. No time-only replay
  of the recorded accuracy click and no guessed late click.
- One-hole flow: reread after each verified shot, including yards-to-feet
  transitions and putts. Require two stable, ready observations before another
  stroke. Require repeated explicit completion text or a stable new ready hole.
  A permanent SCORECARD button and FOR BOGEY are not completion. A failure
  stops the run rather than retrying a potentially consumed stroke.
- F9: dedicated watcher latches stop requests while OCR is busy and releases
  a held button. Focus loss, minimization, or moved/resized geometry also stops
  input. No flag movement, camera action, window activation, or resizing occurs.

## Run from PowerShell in Automation-Agent project root

```powershell
.\start_agent.cmd --check
.\.venv\Scripts\python.exe -B .\test_automation.py
.\start_agent.cmd --observe
.\start_agent.cmd --one-shot
.\start_agent.cmd
```

`--check` checks dependencies/recording without capture or mouse input.
`--observe` reads a screenshot without input; the game must be unobstructed.
The default attempts one hole with up to 15 strokes, stopping on ambiguity or
visible completion. Activate the existing WGT window during the eight-second
countdown. Press F9 to stop; Ctrl+C also stops from the console. There is no
automatic restart. Diagnostics go into `automation_debug/accuracy_trace.json`,
`last_accuracy_meter.png`, and `shot_*.json` after time-critical input ends.

## Verification and remaining limits

Offline regressions cover the real saved tee screenshot (282 YDS, selected
187 YDS club), real putting screenshot (5.9 FT, selected 300 FT range), saved
stroke transitions, real saved yellow/green traces, synthetic moving and static
meters, input cleanup, F9, focus/geometry loss, and a mocked tee-to-approach-to-
putt-to-completion controller flow. Mocked completion is not live evidence.

No live swing, accuracy hit, putt, or completed hole was observed during this
repair. The computer-use service was unavailable (native pipe connection
failure). No game inputs were sent by the offline checks.

The distance-to-power ratio is a baseline estimate using the selected club's
displayed range. Carry/roll, wind, lie, slope and green speed have not been
empirically fitted. The saved shot CSV repeats the same results and even
conflicts on roll distance, so it was not used as a validated calibration set.
Club wheel behavior and meter timing still require a real observed test.
With aiming and camera changes forbidden, the bot retains WGT's current aim;
it cannot guarantee hole-out on a breaking green or a target aimed elsewhere.
Unsupported HUD modes and ambiguous screens stop instead of guessing.
