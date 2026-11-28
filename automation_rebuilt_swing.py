
from pathlib import Path
import ctypes
from ctypes import wintypes
import importlib.util
import time
from collections import deque

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "automation_one_shot_v20.py"

if not SOURCE.is_file():
    raise SystemExit("Original one-shot bot missing.")

spec = importlib.util.spec_from_file_location(
    "wgt_existing_one_shot", SOURCE
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

u = ctypes.windll.user32
u.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
u.SetCursorPos.restype = wintypes.BOOL
u.GetCursorPos.argtypes = [
    ctypes.POINTER(wintypes.POINT)
]
u.GetCursorPos.restype = wintypes.BOOL

LEFT_DOWN = 0x0002
LEFT_UP = 0x0004


def pointer():
    pt = wintypes.POINT()
    if not u.GetCursorPos(ctypes.byref(pt)):
        raise RuntimeError("Cannot read mouse position.")
    return int(pt.x), int(pt.y)


def move_to(x, y):
    if not u.SetCursorPos(int(x), int(y)):
        raise RuntimeError("Mouse movement refused.")

    actual = pointer()

    if abs(actual[0] - x) > 3 or abs(actual[1] - y) > 3:
        raise RuntimeError(
            f"Mouse clipped: {actual}"
        )


def down():
    u.mouse_event(LEFT_DOWN, 0, 0, 0, 0)


def up():
    u.mouse_event(LEFT_UP, 0, 0, 0, 0)


def one_swing(hwnd, rect, sct, template):
    old.guard(hwnd)

    print("WINDOW UNCHANGED - NO RESIZING")

    frame = old.capture(sct, rect)

    if not old.verify_player(frame):
        print("Player not verified. No swing.")
        return

    readings = []

    for _ in range(3):
        old.guard(hwnd)
        frame = old.capture(sct, rect)
        readings.append(
            old.swing_button(frame, template, rect)
        )
        time.sleep(.08)

    score, position = readings[-1]

    if (
        position is None
        or min(r[0] for r in readings)
        < old.MIN_SWING_SCORE
    ):
        print("Swing button not verified.")
        return

    x, y = position

    lower = next(
        (
            m for m in sct.monitors[1:]
            if m["left"] <= x < m["left"] + m["width"]
            and m["top"] <= y < m["top"] + m["height"]
        ),
        None
    )

    if lower is None:
        print("WGT monitor not found. No swing.")
        return

    upper = next(
        (
            m for m in sct.monitors[1:]
            if m is not lower
            and abs(
                m["top"] + m["height"] - lower["top"]
            ) <= 20
            and m["left"] + 20 <= x
            < m["left"] + m["width"] - 20
        ),
        None
    )

    if upper is None:
        print("Upper monitor not aligned. No swing.")
        return

    minimum_y = upper["top"] + 25
    available = y - minimum_y

    print(
        f"Available movement: {available} pixels"
    )

    if available < 680:
        print("Insufficient movement. No swing.")
        return

    # Test crossing monitors BEFORE pressing the mouse.
    old.guard(hwnd)
    move_to(x, lower["top"] - 30)
    old.guard(hwnd)
    move_to(x, y)

    print("CROSS-MONITOR CURSOR TEST PASSED")

    meter = old.meter_rect(rect)
    target = old.calibrated_target(rect, meter)

    print("PLAYER VERIFIED:", old.PLAYER)
    print("Swing confidence:", round(score, 3))
    print("Accuracy mark:", target)

    reached = False
    confirmed = 0
    best = 0.0
    crossed = False
    stuck = 0
    is_down = False
    started = time.perf_counter()

    try:
        old.guard(hwnd)
        down()
        is_down = True

        for step in range(500):
            old.guard(hwnd)

            if time.perf_counter() - started > 9:
                print("Power timeout.")
                break

            px, py = pointer()

            if py - 9 < minimum_y:
                print("Upper screen edge reached.")
                break

            move_to(x, py - 9)

            if pointer()[1] >= py - 3:
                stuck += 1
            else:
                stuck = 0

            if stuck >= 3:
                print("Mouse movement stopped.")
                break

            if not crossed and pointer()[1] < lower["top"]:
                crossed = True
                print(
                    "POINTER CROSSED INTO MONITOR 1",
                    flush=True
                )

            reading = old.estimate_yellow_power(
                old.capture(sct, meter)
            )

            if reading is not None:
                best = max(best, reading)

                if reading >= .985:
                    confirmed += 1

                    if confirmed >= 2:
                        reached = True
                        break
                else:
                    confirmed = 0

            if step % 20 == 0:
                print(
                    f"Power: {best:.0%}, "
                    f"mouse Y: {pointer()[1]}",
                    flush=True
                )

            time.sleep(.012)

    finally:
        if is_down:
            up()

    print(
        f"Highest measured power: {best:.0%}",
        flush=True
    )

    if not reached:
        print("100% POWER NOT VERIFIED")
        print("Accuracy click skipped.")
        print("Mouse release may count as a stroke.")
        return

    old.guard(hwnd)

    move_to(
        meter["left"] + target,
        meter["top"] + meter["height"] // 2
    )

    print("POWER VERIFIED - WATCHING GREEN RETURN")

    history = deque(maxlen=5)
    records = []
    clicked = False
    progress = 0
    last_save = -1.0
    start = time.perf_counter()

    while time.perf_counter() - start < old.RETURN_TIMEOUT:
        old.guard(hwnd)

        now = time.perf_counter()
        image = old.capture(sct, meter)
        edge = old.green_edge(image, rect, meter)
        elapsed = now - start

        if elapsed - last_save >= .10:
            records.append(
                (elapsed, image.copy(), edge)
            )
            last_save = elapsed

        if edge is not None:
            history.append((now, edge))

            if len(history) >= 3:
                recent = list(history)[-3:]

                movement = (
                    recent[-1][1] - recent[0][1]
                )

                if movement >= max(
                    4, int(rect["width"] * .003)
                ):
                    progress += 1
                else:
                    progress = max(0, progress - 1)

                if progress >= 3:
                    dt = max(
                        .001,
                        recent[-1][0] - recent[0][0]
                    )

                    speed = max(
                        0, movement / dt
                    )

                    lead = max(
                        6, min(24, int(speed * .020))
                    )

                    if (
                        target - lead <= edge
                        <= target + 5
                    ):
                        old.guard(hwnd)
                        down()
                        up()

                        clicked = True

                        print(
                            "AUTOMATIC ACCURACY CLICK",
                            flush=True
                        )
                        break

        time.sleep(.007)

    old.save_trace(records, target)

    if clicked:
        print("ONE AUTOMATIC SWING ATTEMPTED.")
        print("Check whether the ball moved.")
    else:
        print("Green return not verified.")
        print("No guessed accuracy click.")


old.one_swing = one_swing

if __name__ == "__main__":
    print("REBUILT WGT SWING CONTROLLER")
    print("No automatic window resizing")
    print("F8 = One swing")
    print("F9 = Emergency stop")
    old.main()
