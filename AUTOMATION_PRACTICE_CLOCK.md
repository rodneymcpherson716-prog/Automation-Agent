# Solo practice with the alternate-shot time budget

The user requested a simulated 60-second shot clock on November 14, 2026,
then changed this practice run to **90 seconds**. Use
`--solo-confirmed --practice-shot-clock 90` for the current solo practice.
The 60-second option remains available; real alternate-shot timing remains 60.
The clock starts at the original capture of each independently identified ready
shot, before OCR and measurement revalidation. Rereads, club changes and a
temporarily unreadable HUD retain that deadline. A verified new hole/stroke
starts the next practice clock. Attaching to an already active solo shot starts
a new practice attempt; it does not establish when that real turn began.

The existing stroke budget reserves the entire drag, return/accuracy phase and
three-second margin. Insufficient time blocks the swing. A practice timeout is
a failed timing attempt, not proof of readiness for alternate-shot multiplayer.
The simulated clock never supplies a missing multiplayer countdown. Real
alternate-shot games still require their visible remaining timer.

F9, foreground, geometry, ownership, freshness and bunker guards remain active.
Leave WGT selected and do not use the mouse/keyboard during live play.

Clock tests include capture-time accounting, both durations and expired-clock
rejection. The other 73 focused context, putt freshness, solo and
nine-hole regressions passed in the same validation run. These are offline
checks, not live round completion evidence.
