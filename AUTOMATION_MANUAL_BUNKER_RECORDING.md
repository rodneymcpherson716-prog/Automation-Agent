# Recording one real manual bunker measurement

The recorder is separate from the bot. It finds an already-open WGT window
and reads two BEFORE images and two AFTER images. It never opens WGT, activates
a window, moves the pointer, clicks, changes aim, selects a club or swings.
It holds the existing controller lock while running, preventing a nine-hole
controller in this workspace from running alongside it. F9 is checked during
countdown and between capture/OCR stages; Ctrl+C stops from the console. Neither
sends a release/click or discards pending evidence. The bot's existing F9 watcher
is unchanged.

## First measurement — no screenshots to send and no commands to copy

1. When you choose to collect data, open WGT yourself in **SOLO stroke play**.
   Stop any automation. Get to a bunker with **LW** selected. Keep the usual
   **1288 × 728** window layout; the recorder does not resize it. Set your aim
   and shot mode manually. Make the floating distance/elevation badge, club
   range, bunker lie, wind and your player/stroke HUD visible.
2. Open **WGT Shot Stats yourself before the shot**, showing the previous
   shot's CARRY, ROLL and POWER. The recorder needs that baseline to distinguish
   a new result from an old panel. Leave the panel in its established position.
3. In the workspace, double-click **Record_Manual_Bunker.cmd**. Choose **1**,
   then type **SOLO** to confirm single-player manual play. During the eight-
   second countdown, switch to the existing WGT window yourself. Keep it
   unobstructed and do not take a shot yet.
4. Wait for the two local images to be read, then switch back to the recorder
   console yourself. Before swinging, confirm it says **BEFORE SAVED**,
   **Missing conditions: none**, and that baseline carry, roll and power are
   all numbers rather than `None`. OCR can take a while; wait for this message.
   If any condition/baseline is missing, use **3**, type **DISCARD**, make that
   field visible and repeat **1**. Do not supply a guessed value.
5. Return to WGT yourself and take **exactly one manual shot**, using your own
   chosen power. Keep aim, club and mode unchanged after the BEFORE capture.
   If you notice wind, lie or elevation changing before the shot, discard and
   recapture BEFORE instead. **26% is not a required power setting.**
6. After the ball stops, keep the fresh Shot Stats visible and wait for the
   next ready HUD showing the same hole and exactly one higher stroke count.
   Return to the recorder, choose **2**, and type **ONE** only if exactly one
   manual shot was taken with the confirmed setup. Switch back to WGT during
   the eight-second countdown. Keep its HUD and stats unobstructed.
7. Wait for reading to finish and check the console. **RESULT: VERIFIED** means
   every requested field passed its checks and the new panel was attributed to
   the manual shot. **RESULT: PARTIAL** preserves readable fields and lists
   what was missing or why attribution failed. Neither status enables swings.
   Do not repeat a possibly consumed shot just to repair its recording.

For the first full measurement, use a shot whose next ready stroke HUD remains
visible on the same hole. A hole-out/scorecard that hides that identity can
only produce a partial record. If the new carry/roll/power exactly match the
previous panel, the recorder conservatively refuses to call it fresh. A BEFORE
capture expires after ten minutes; never reconstruct an expired/missing setup
from memory or from AFTER conditions.

## What is saved and what “verified” means

The record contains selected LW range, floating target distance in yards,
shot mode, displayed bunker lie percentage, wind speed range and screen arrow
vector, signed elevation in feet, and actual Shot Stats carry, roll and POWER.
POWER is the game's reported percentage, not a requested setting or inferred
meter fraction. Total travel is marked as a derived carry-plus-roll sum only
when both components were read. No missing roll is replaced with zero and no
target distance is treated as measured carry.

Each field requires agreement across two independently timestamped captures.
Shot Stats also require multiple OCR reads, exact row labels, and explicit
units; carry/roll need decimal readings so `0.83` cannot become bare `83`.
Panel recognition uses a readable title or all three exact row labels; the
chosen basis and raw reads are retained. Unreadable spin/apex do not erase
verified carry, roll or power. Missing/contradictory values remain null.
A calm speed reading does not invent a missing arrow: speed is retained while
direction stays null, so that capture remains partial under the full-field rule.

Shot attribution requires the independently visible local SOLO ready HUD on
both phases, the same hole, exactly one stroke increment, a changed complete
carry/roll/power panel, and your explicit one-shot/setup attestation. It does
not claim to have tracked the ball or witnessed the swing. Conditions are
labelled **observed BEFORE the shot**, not measured at impact. A “verified”
record verifies those displayed numbers and this documented attribution;
it does not validate bunker physics or guarantee wind stayed fixed in flight.
If attribution fails, readable values are merely unattributed partial evidence.

- Results: `wgt_calibration/manual_bunker/records/<id>.json`.
- Pending setup: `wgt_calibration/manual_bunker/pending.json`.
- Discarded setups: `wgt_calibration/manual_bunker/discarded/`.
- Local PNG evidence, timestamps and SHA-256 links:
  `automation_debug/manual_bunker/<capture-id>/`.

Each capture folder also keeps `capture.json` with the raw reads, including
captures that could not be paired into an accepted measurement.

Images stay local. There is no upload or messaging step. Do not edit records
to mark them verified. If the process stops, pending state survives; the
recorder will not overwrite it without explicit discard. Duplicate AFTER
submission cannot create a second measurement. Result files are committed
before the pending setup is retired, allowing interrupted saves to recover.

All results explicitly set `use_for_automatic_power=false`. They are separate
from `power_observations.json` and are not loaded by the bunker model. The
successful 26% manual reference remains unchanged with unknown carry. The
existing automatic bunker block, swing, accuracy, F9 controller and nine-hole
functions are unchanged.
