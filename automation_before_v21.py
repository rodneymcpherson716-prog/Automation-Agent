
"""
WGT GOLF - TIGERWOULD52
GREEN RETURN SWING TEST

You open WGT Golf yourself.
The bot never launches WGT or Microsoft Edge.

F8 = One automatic swing
F9 = Emergency stop
Ctrl+C = Stop from terminal

This version follows the RIGHT edge of the
green bar as it grows toward the accuracy mark.

The bot makes only ONE swing attempt.
"""

import ctypes
import ctypes.wintypes
import json
import re
import time

from collections import deque
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


# ==================================================
# SETTINGS
# ==================================================

PLAYER = "Tigerwould52"
GAME_MODE = "SOLO"

ROOT = Path(__file__).resolve().parent

TEMPLATE = ROOT / "swing_button.png"
CALIBRATION = ROOT / "automation_accuracy.json"
DEBUG = ROOT / "automation_debug"

MIN_SWING_SCORE = 0.83

BACKSWING_PIXELS = 140
BACKSWING_SECONDS = 0.80

# Give the green return bar more time.
RETURN_TIMEOUT = 8.0

VK_F8 = 0x77
VK_F9 = 0x78

user32 = ctypes.windll.user32

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


# ==================================================
# KEYBOARD
# ==================================================

def pressed(code):
    return bool(
        user32.GetAsyncKeyState(code) & 0x8000
    )


# ==================================================
# WINDOWS SETUP
# ==================================================

def setup():
    try:
        user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)
        )
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]

    for location in paths:
        if Path(location).is_file():
            pytesseract.pytesseract.tesseract_cmd = (
                location
            )
            break


# ==================================================
# FIND EXISTING WGT WINDOW
# ==================================================

def find_wgt():
    results = []

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p
    )

    def visit(hwnd, unused):
        if (
            user32.IsWindowVisible(hwnd)
            and not user32.IsIconic(hwnd)
        ):
            length = user32.GetWindowTextLengthW(
                hwnd
            )

            if length:
                title = ctypes.create_unicode_buffer(
                    length + 1
                )

                user32.GetWindowTextW(
                    hwnd,
                    title,
                    length + 1
                )

                if "wgt golf" in title.value.lower():
                    results.append(hwnd)

        return True

    user32.EnumWindows(
        callback_type(visit),
        0
    )

    return results[0] if results else None


def get_rect(hwnd):
    r = ctypes.wintypes.RECT()

    if not user32.GetWindowRect(
        hwnd,
        ctypes.byref(r)
    ):
        return None

    width = r.right - r.left
    height = r.bottom - r.top

    if width < 600 or height < 400:
        return None

    return {
        "left": r.left,
        "top": r.top,
        "width": width,
        "height": height
    }


def guard(hwnd):
    if pressed(VK_F9):
        raise KeyboardInterrupt("F9 stop")

    if user32.GetForegroundWindow() != hwnd:
        raise RuntimeError(
            "WGT lost focus. No more clicks."
        )


# ==================================================
# SCREEN CAPTURE
# ==================================================

def capture(sct, region):
    return np.asarray(
        sct.grab(region)
    )[:, :, :3].copy()


# ==================================================
# VERIFY TIGERWOULD52
# ==================================================

def verify_player(frame):
    height, width = frame.shape[:2]

    wanted = re.sub(
        r"[^a-z0-9]",
        "",
        PLAYER.lower()
    )

    areas = [
        (.825, .056, .911, .106),
        (.785, .045, .965, .118),
        (.72, .02, 1.0, .22)
    ]

    for x1, y1, x2, y2 in areas:
        patch = frame[
            int(height * y1):int(height * y2),
            int(width * x1):int(width * x2)
        ]

        if not patch.size:
            continue

        gray = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2GRAY
        )

        enlarged = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        _, binary = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        for image in (
            enlarged,
            binary,
            255 - binary
        ):
            for mode in (7, 6, 11):
                try:
                    text = pytesseract.image_to_string(
                        image,
                        config=f"--psm {mode}"
                    )

                    clean = re.sub(
                        r"[^a-z0-9]",
                        "",
                        text.lower()
                    )

                    if wanted in clean:
                        return True

                except Exception:
                    pass

    return False


# ==================================================
# SWING BUTTON
# ==================================================

def swing_button(frame, template, rect):
    height, width = frame.shape[:2]

    ox = int(width * .55)
    oy = int(height * .45)

    area = frame[
        oy:int(height * .92),
        ox:
    ]

    gray = cv2.cvtColor(
        area,
        cv2.COLOR_BGR2GRAY
    )

    best_score = 0.0
    best_position = None

    for scale in (
        .65, .75, .85, .95,
        1.0, 1.05, 1.15,
        1.25, 1.35
    ):
        tw = round(
            template.shape[1] * scale
        )

        th = round(
            template.shape[0] * scale
        )

        if (
            min(tw, th) < 15
            or tw >= gray.shape[1]
            or th >= gray.shape[0]
        ):
            continue

        resized = cv2.resize(
            template,
            (tw, th),
            interpolation=cv2.INTER_AREA
        )

        if resized.std() < 5:
            continue

        result = cv2.matchTemplate(
            gray,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, location = (
            cv2.minMaxLoc(result)
        )

        if score > best_score:
            best_score = score

            best_position = (
                rect["left"] + ox
                + location[0] + tw // 2,

                rect["top"] + oy
                + location[1] + th // 2
            )

    return best_score, best_position


# ==================================================
# METER CAPTURE REGION
# ==================================================

def meter_rect(rect):
    width = rect["width"]
    height = rect["height"]

    return {
        "left": (
            rect["left"] + int(width * .12)
        ),
        "top": (
            rect["top"] + int(height * .940)
        ),
        "width": int(width * .82),
        "height": max(
            30,
            int(height * .055)
        )
    }


# ==================================================
# LOAD YOUR SAVED ACCURACY MARK
# ==================================================

def calibrated_target(rect, meter):
    if not CALIBRATION.is_file():
        raise RuntimeError(
            "Saved accuracy mark not found."
        )

    data = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    fraction = float(
        data["accuracy_x_fraction"]
    )

    if not .25 <= fraction <= .90:
        raise RuntimeError(
            "Accuracy mark outside meter area."
        )

    target = (
        round(fraction * rect["width"])
        -
        (meter["left"] - rect["left"])
    )

    if not 25 < target < meter["width"] - 25:
        raise RuntimeError(
            "Accuracy mark outside capture area."
        )

    return target


# ==================================================
# DETECT THE RIGHT EDGE OF THE GREEN BAR
# ==================================================

def green_edge(image, rect, meter):
    """
    Detect the fluorescent green bar growing
    from left to right during the return.

    Ignore the scenery and the yellow phase.
    """

    height = image.shape[0]

    patch = image[
        int(height * .35):int(height * .85),
        :
    ]

    if not patch.size:
        return None

    hsv = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        (35, 65, 65),
        (95, 255, 255)
    )

    column_counts = np.count_nonzero(
        mask,
        axis=0
    )

    minimum = max(
        3,
        int(patch.shape[0] * .38)
    )

    filled = (
        column_counts >= minimum
    ).astype(np.uint8)

    kernel_width = max(
        5,
        int(rect["width"] * .006)
    )

    kernel = np.ones(
        (1, kernel_width),
        np.uint8
    )

    closed = cv2.morphologyEx(
        filled[None, :],
        cv2.MORPH_CLOSE,
        kernel
    )[0]

    coordinates = np.flatnonzero(
        closed
    )

    if len(coordinates) == 0:
        return None

    breaks = np.flatnonzero(
        np.diff(coordinates) > 1
    ) + 1

    runs = np.split(
        coordinates,
        breaks
    )

    # Only consider a colored bar beginning
    # near the left origin of WGT's swing meter.

    offset = (
        meter["left"] - rect["left"]
    )

    anchor_min = (
        int(rect["width"] * .115) - offset
    )

    anchor_max = (
        int(rect["width"] * .185) - offset
    )

    minimum_length = max(
        10,
        int(rect["width"] * .010)
    )

    eligible = [
        run
        for run in runs
        if (
            anchor_min <= int(run[0]) <= anchor_max
            and len(run) >= minimum_length
        )
    ]

    if not eligible:
        return None

    longest = max(
        eligible,
        key=len
    )

    return int(longest[-1])


# ==================================================
# SAVE COLOR TRACKING RESULTS
# ==================================================

def save_trace(records, target):
    if not records:
        return

    DEBUG.mkdir(exist_ok=True)

    indices = np.linspace(
        0,
        len(records) - 1,
        min(15, len(records)),
        dtype=int
    )

    images = []

    for index in indices:
        elapsed, frame, edge = records[index]

        display = frame.copy()

        # White = saved accuracy mark.
        cv2.line(
            display,
            (target, 0),
            (target, display.shape[0] - 1),
            (255, 255, 255),
            2
        )

        # Red = detected green edge.
        if edge is not None:
            cv2.line(
                display,
                (edge, 0),
                (edge, display.shape[0] - 1),
                (0, 0, 255),
                2
            )

        cv2.putText(
            display,
            f"{elapsed:.2f}s edge={edge}",
            (5, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            .35,
            (0, 255, 255),
            1
        )

        images.append(
            cv2.resize(
                display,
                (850, 75)
            )
        )

    output = cv2.vconcat(images)

    cv2.imwrite(
        str(DEBUG / "green_return_trace.png"),
        output
    )


# ==================================================


# POWER AND CLICK REPAIR V20B
def estimate_yellow_power(frame):
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    band = frame[int(h*.25):int(h*.90)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(
        hsv, (85, 35, 55), (130, 255, 255)
    )

    threshold = max(3, int(band.shape[0]*.22))
    columns = np.arange(w)

    blue_x = np.flatnonzero(
        (np.count_nonzero(blue, axis=0) >= threshold)
        & (columns > w*.35)
        & (columns < w*.90)
    )

    if len(blue_x) < 15:
        return None

    blue_start = int(blue_x.min())

    b, g, r = [
        band[:, :, j].astype(np.float32)
        for j in range(3)
    ]

    yellow = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 43)
        & (hsv[:, :, 1] >= 65)
        & (r >= 34)
        & (r >= g*1.025)
        & (g >= b*1.45)
    )

    filled = (
        np.count_nonzero(yellow, axis=0)
        >= threshold
    )

    filled[:max(0, int(w*.02))] = False
    filled[blue_start:] = False

    x = np.flatnonzero(filled)

    if len(x) < max(12, int(w*.02)):
        return None

    runs = np.split(
        x,
        np.flatnonzero(np.diff(x) > 15) + 1
    )

    runs = [
        run for run in runs
        if len(run) >= max(12, int(w*.02))
    ]

    if not runs:
        return None

    bar = max(runs, key=len)
    left, right = int(bar[0]), int(bar[-1])

    if left > w*.30 or blue_start-left < w*.23:
        return None

    return float(np.clip(
        (right-left)/(blue_start-left),
        0, 1
    ))


# ONE AUTOMATIC SWING
# ==================================================

def one_swing(hwnd, rect, sct, template):
    guard(hwnd)

    frame = capture(sct, rect)

    if not verify_player(frame):
        print(
            "Tigerwould52 not verified. No click."
        )
        return

    detections = []

    for _ in range(3):
        guard(hwnd)

        frame = capture(sct, rect)

        detections.append(
            swing_button(
                frame,
                template,
                rect
            )
        )

        time.sleep(.10)

    score, position = detections[-1]

    if (
        position is None
        or min(
            result[0]
            for result in detections
        ) < MIN_SWING_SCORE
    ):
        print(
            "Swing button not confirmed:",
            round(score, 3)
        )
        return

    meter = meter_rect(rect)

    target = calibrated_target(
        rect,
        meter
    )

    print()
    print("PLAYER VERIFIED:", PLAYER)
    print("Swing confidence:", round(score, 3))
    print("Saved accuracy mark:", target)
    print()


    # AUTOMATIC FULL-POWER BACKSWING
    pyautogui.moveTo(*position, duration=.10)

    reached = False
    confirmed = 0
    best = 0.0
    started = time.perf_counter()

    try:
        guard(hwnd)
        pyautogui.mouseDown()

        for _ in range(220):
            guard(hwnd)

            if time.perf_counter() - started >= 8.0:
                break

            if pyautogui.position().y > (
                rect["top"] + int(rect["height"]*.03)
            ):
                pyautogui.moveRel(
                    0, -7, duration=.015
                )

            reading = estimate_yellow_power(
                capture(sct, meter)
            )

            if reading is not None:
                best = max(best, reading)

                if reading >= .985:
                    confirmed += 1
                else:
                    confirmed = 0

                if confirmed >= 2:
                    reached = True
                    break

            time.sleep(.01)

    finally:
        pyautogui.mouseUp()

    print(
        "Highest measured power: {:.0%}".format(best),
        flush=True
    )

    if not reached:
        print("Full power NOT verified.")
        print("No automatic accuracy click.")
        print("Mouse release may count as a stroke.")
        return

    guard(hwnd)

    pyautogui.moveTo(
        meter["left"] + target,
        meter["top"] + meter["height"]//2,
        duration=.05
    )

    pyautogui.moveTo(
        meter["left"] + target,
        meter["top"] + meter["height"]//2,
        duration=.05
    )
    print("Backswing completed.")
    print("Watching green bar grow RIGHT...")

    start = time.perf_counter()

    history = deque(maxlen=5)

    records = []
    last_recorded = -1.0

    valid_progress = 0
    clicked = False

    while (
        time.perf_counter() - start
        < RETURN_TIMEOUT
    ):
        guard(hwnd)

        now = time.perf_counter()

        image = capture(
            sct,
            meter
        )

        edge = green_edge(
            image,
            rect,
            meter
        )

        elapsed = now - start

        if elapsed - last_recorded >= .10:
            records.append(
                (
                    elapsed,
                    image.copy(),
                    edge
                )
            )

            last_recorded = elapsed

        if edge is not None:
            history.append(
                (now, edge)
            )

            if len(history) >= 3:
                recent = list(history)[-3:]

                movement = (
                    recent[-1][1]
                    - recent[0][1]
                )

                # The returning green edge
                # must be moving RIGHT.

                if movement >= max(
                    4,
                    int(rect["width"] * .003)
                ):
                    valid_progress += 1
                else:
                    valid_progress = max(
                        0,
                        valid_progress - 1
                    )

                if valid_progress >= 3:
                    elapsed_frames = max(
                        .001,
                        recent[-1][0]
                        - recent[0][0]
                    )

                    speed = max(
                        0,
                        movement / elapsed_frames
                    )

                    # Small lead for input latency.
                    lead = max(
                        6,
                        min(
                            24,
                            int(speed * .020)
                        )
                    )

                    if (
                        target - lead
                        <= edge
                        <= target + 5
                    ):
                        guard(hwnd)

                        # WGT accepts a screen click
                        # during the return phase.
                        # Do not waste time moving
                        # the pointer again.

                        pyautogui.click()

                        clicked = True

                        print()
                        print("ACCURACY CLICK PERFORMED")
                        print("Green edge:", edge)
                        print("Mark:", target)
                        print("Lead:", lead)

                        break

        time.sleep(.007)

    save_trace(
        records,
        target
    )

    print()

    if clicked:
        print(
            "ONE SWING ATTEMPTED."
        )
        print(
            "Check whether the ball traveled."
        )

    else:
        print(
            "Green return was not verified "
            "at the accuracy mark."
        )
        print(
            "No guessed accuracy click."
        )

    print(
        r"Trace: automation_debug\green_return_trace.png"
    )


# ==================================================
# MAIN
# ==================================================

def main():
    setup()

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("Tesseract OCR unavailable.")
        return

    template = cv2.imread(
        str(TEMPLATE),
        cv2.IMREAD_GRAYSCALE
    )

    if template is None:
        print("Missing swing_button.png")
        return

    print("=" * 52)
    print("WGT - TIGERWOULD52")
    print("GREEN RETURN SWING CONTROLLER")
    print("=" * 52)

    print("Game launching: DISABLED")
    print("Game mode:", GAME_MODE)
    print("Maximum swing attempts: ONE")
    print()
    print("F8 = One automatic swing")
    print("F9 = Emergency stop")
    print("CTRL+C = Stop from terminal")
    print()

    previous_f8 = False

    try:
        factory = getattr(
            mss,
            "MSS",
            mss.mss
        )

        with factory() as sct:
            while True:
                if pressed(VK_F9):
                    print("Stopped by F9.")
                    return

                hwnd = find_wgt()

                focused = (
                    hwnd is not None
                    and user32.GetForegroundWindow()
                    == hwnd
                )

                f8 = pressed(VK_F8)

                if (
                    focused
                    and f8
                    and not previous_f8
                ):
                    rect = get_rect(hwnd)

                    if rect is not None:
                        one_swing(
                            hwnd,
                            rect,
                            sct,
                            template
                        )

                    # Stop after one attempt.
                    return

                previous_f8 = (
                    f8 if focused else False
                )

                time.sleep(.05)

    except (
        KeyboardInterrupt,
        pyautogui.FailSafeException
    ):
        print("Emergency stop.")

    except Exception as error:
        print(
            "Stopped safely:",
            error
        )

    finally:
        pyautogui.mouseUp()
        print("WGT CONTROLLER CLOSED")


if __name__ == "__main__":
    main()

