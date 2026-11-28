
"""
WGT GOLF - TIGERWOULD52
AUTOMATIC ONE SHOT V25

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
RETURN_TIMEOUT = 3.0
# Account for screen-capture, processing and game-input delay. Tune after shot.
ACCURACY_LATENCY_SECONDS = 0.045
POWER_TARGET = .985  # Changed per short shot; tee shots remain full power

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
    """Track the growing green return bar from its real HUD origin.

    Calibrated against the user's 15-frame green_return_trace: the bar
    starts at ~23% of the capture width and moves toward ~73% target.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    patch = image[int(height * .18):int(height * .94), :]
    if not patch.size:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # Exclude blue accuracy zone; its hue is above the green range.
    mask = cv2.inRange(hsv, (35, 65, 55), (90, 255, 255))
    counts = np.count_nonzero(mask, axis=0)
    minimum = max(2, int(patch.shape[0] * .19))
    filled = (counts >= minimum).astype(np.uint8)
    # Bridge the narrow white target line without joining distant shapes.
    filled = cv2.morphologyEx(
        filled[None, :], cv2.MORPH_CLOSE,
        np.ones((1, 7), np.uint8)
    )[0]
    coords = np.flatnonzero(filled)
    if coords.size == 0:
        return None
    runs = np.split(coords, np.flatnonzero(np.diff(coords) > 1) + 1)
    # Earlier versions mistakenly expected the origin at 0-12%.
    # The provided trace establishes it at roughly 23%.
    anchor_lo = int(width * .19)
    anchor_hi = int(width * .29)
    valid = [r for r in runs
             if anchor_lo <= int(r[0]) <= anchor_hi and len(r) >= 9]
    if not valid:
        return None
    return int(max(valid, key=len)[-1])


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


# SWING CIRCLE AND TEXT CHECK
_previous_swing_button = swing_button

def swing_button(frame, template, rect):
    score, position = _previous_swing_button(
        frame, template, rect
    )

    if position is not None and score >= MIN_SWING_SCORE:
        return score, position

    h, w = frame.shape[:2]
    x0, y0 = int(w*.74), int(h*.45)
    area = frame[y0:int(h*.90), x0:]

    if area.size == 0:
        return score, position

    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        smooth, cv2.HOUGH_GRADIENT, 1.2,
        minDist=max(35, int(w*.035)),
        param1=115,
        param2=40,
        minRadius=int(w*.025),
        maxRadius=int(w*.055)
    )

    if circles is None:
        return score, position

    for cx, cy, radius in circles[0]:
        x, y = int(cx)+x0, int(cy)+y0
        r = int(radius)

        if not (
            .78*w < x < .97*w
            and .57*h < y < .85*h
        ):
            continue

        label = frame[
            max(0, y-int(.37*r)):min(h, y+int(.53*r)),
            max(0, x-int(.72*r)):min(w, x+int(.72*r))
        ]

        if not label.size:
            continue

        enlarged = cv2.resize(
            cv2.cvtColor(label, cv2.COLOR_BGR2GRAY),
            None, fx=5, fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        word = re.sub(
            r"[^a-z]", "",
            pytesseract.image_to_string(
                enlarged, config="--psm 7"
            ).lower()
        )

        if word == "swing":
            print(
                "Swing circle AND text verified",
                flush=True
            )
            return .90, (
                rect["left"]+x,
                rect["top"]+y
            )

    return score, position


def force_mouse_release():
    """Release the left button via both PyAutoGUI and Windows input."""
    try:
        pyautogui.mouseUp(button="left")
    finally:
        # MOUSEEVENTF_LEFTUP, independent of PyAutoGUI state.
        user32.mouse_event(0x0004, 0, 0, 0, 0)


def one_swing(hwnd, rect, sct, template):
    guard(hwnd)

    # V21 DESKTOP HEADROOM
    cx = rect["left"] + rect["width"] // 2
    cy = rect["top"] + rect["height"] // 2

    monitor = next(
        (m for m in sct.monitors[1:]
         if m["left"] <= cx < m["left"] + m["width"]
         and m["top"] <= cy < m["top"] + m["height"]),
        None
    )

    if monitor is None:
        print("WGT monitor not found. No swing.")
        return

    desktop_floor = monitor["top"]
    new_top = (
        desktop_floor + monitor["height"]
        - rect["height"] - 35
    )

    if new_top > rect["top"] + 65:
        move = user32.SetWindowPos
        move.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint
        ]
        move.restype = ctypes.c_bool

        success = move(
            ctypes.c_void_p(hwnd), None,
            rect["left"], int(new_top),
            0, 0, 0x0005
        )

        if not success:
            print("Could not reposition WGT. No swing.")
            return

        time.sleep(.35)
        guard(hwnd)
        rect = get_rect(hwnd)

        if rect is None:
            print("Window position unavailable. No swing.")
            return

        print("WGT moved lower for full-power backswing.")

    # V26: Resize is optional. V25 stopped on its second run because
    # the window was already 1024 x 576, the minimum size used here.
    # Keep the current window and proceed if there is no safe resize.
    goal_w = max(1024, int(rect["width"] * .80))
    goal_h = max(576, int(rect["height"] * .80))

    can_resize = (
        goal_w < rect["width"]
        and goal_h < rect["height"]
        and goal_w <= monitor["width"] - 20
        and goal_h <= monitor["height"] - 30
    )

    if can_resize:
        x = monitor["left"] + (monitor["width"] - goal_w) // 2
        y = monitor["top"] + monitor["height"] - goal_h - 24

        move = user32.SetWindowPos
        move.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint
        ]
        move.restype = ctypes.c_bool

        success = move(
            ctypes.c_void_p(hwnd), None,
            int(x), int(y), int(goal_w), int(goal_h), 0x0014
        )
        if success:
            time.sleep(.35)
            guard(hwnd)
            measured = get_rect(hwnd)
            if (measured is not None
                    and abs(measured["width"] - goal_w) <= 16
                    and abs(measured["height"] - goal_h) <= 16):
                rect = measured
                print("Window resized to", goal_w, "x", goal_h)
            else:
                rect = get_rect(hwnd) or rect
                print("Resize not confirmed; using current WGT window.")
        else:
            print("Resize unavailable; using current WGT window.")
    else:
        print("Window already small enough; using current size:",
              rect["width"], "x", rect["height"])

    # Refresh monitor for the window's final position.
    cx = rect["left"] + rect["width"] // 2
    cy = rect["top"] + rect["height"] // 2
    monitor = next(
        (m for m in sct.monitors[1:]
         if m["left"] <= cx < m["left"] + m["width"]
         and m["top"] <= cy < m["top"] + m["height"]),
        None
    )
    if monitor is None:
        print("WGT monitor not found after positioning. No swing.")
        return

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

    # V24: WGT on Monitor 1 (LEFT), Monitor 2 (RIGHT).
    # An upper monitor is NOT required. Keep the entire drag
    # inside the physical display containing the WGT window.
    desktop_floor = monitor["top"]
    travel_limit = desktop_floor + 18
    available = position[1] - travel_limit
    print("Using WGT display for mouse travel (side-by-side monitors).")
    print(f"Available upward mouse travel: {max(0, available)}px")

    if available < 45:
        print("Swing button too close to top edge. No swing attempted.")
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


    # Attempt full power, but RELEASE at the display edge and finish the shot.
    pyautogui.moveTo(*position, duration=.10)
    reached = False
    confirmed = 0
    best = 0.0
    started = time.perf_counter()
    stuck = 0

    try:
        guard(hwnd)
        pyautogui.mouseDown(button="left")
        for _ in range(220):
            guard(hwnd)
            if time.perf_counter() - started >= 3.0:
                break

            remaining = pyautogui.position().y - travel_limit
            if remaining >= 2:
                pyautogui.moveRel(0, -min(9, remaining), duration=.012)
                stuck = 0
            else:
                stuck += 1

            reading = estimate_yellow_power(capture(sct, meter))
            if reading is not None:
                best = max(best, reading)
                confirmed = confirmed + 1 if reading >= POWER_TARGET else 0
                if confirmed >= 2:
                    reached = True
                    break

            if stuck >= 2:
                print("Top of display reached; finishing shot.", flush=True)
                break
            time.sleep(.006)
    finally:
        force_mouse_release()
        print("LEFT BUTTON UP SENT (twice)", flush=True)

    print("Highest measured power: {:.0%}".format(best), flush=True)
    if not reached:
        print("Requested shot power not verified; continuing to accuracy shot.", flush=True)

    # The pointer must be at the accuracy mark before the click is due.
    # SetCursorPos is instantaneous compared with moving during the trigger.
    guard(hwnd)
    click_x = meter["left"] + target
    click_y = meter["top"] + meter["height"] // 2
    user32.SetCursorPos(int(click_x), int(click_y))
    print("Tracking green return with latency compensation...", flush=True)
    start = time.perf_counter()
    previous_edge = None
    previous_time = None
    last_speed = None
    records = []
    last_recorded = -1.0
    clicked = False
    last_print = -1.0

    while time.perf_counter() - start < RETURN_TIMEOUT:
        guard(hwnd)
        now = time.perf_counter()
        frame = capture(sct, meter)
        observed = time.perf_counter()
        edge = green_edge(frame, rect, meter)
        elapsed = observed - start
        if elapsed - last_recorded >= .06:
            records.append((elapsed, frame.copy(), edge))
            last_recorded = elapsed
        if edge is not None:
            if previous_edge is not None and previous_time is not None:
                dt = observed - previous_time
                delta = edge - previous_edge
                if .003 < dt < .25 and 1 <= delta <= 110:
                    instantaneous_speed = delta / dt
                    last_speed = (instantaneous_speed if last_speed is None
                                  else .55 * last_speed + .45 * instantaneous_speed)
            previous_edge, previous_time = edge, observed
            # Earlier than the line: predict where the growing bar will be
            # after processing, event delivery and display refresh.
            if last_speed is not None:
                lead = max(12, min(65, int(last_speed * ACCURACY_LATENCY_SECONDS)))
                predicted_edge = edge + lead
                if elapsed - last_print >= .12:
                    print(f"RETURN edge={edge} predicted={predicted_edge} mark={target} speed={last_speed:.0f}px/s", flush=True)
                    last_print = elapsed
                if predicted_edge >= target and edge <= target + 8:
                    guard(hwnd)
                    # Do not move the pointer again. A direct Windows click
                    # minimizes input overhead relative to pyautogui.click(x,y).
                    user32.mouse_event(0x0002, 0, 0, 0, 0)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)
                    clicked = True
                    print(f"PREDICTIVE ACCURACY CLICK: edge={edge}, lead={lead}, mark={target}", flush=True)
                    break
            elif edge >= target - 12 and edge <= target + 5:
                guard(hwnd)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                clicked = True
                print(f"ACCURACY CLICK (no speed estimate): edge={edge}, mark={target}", flush=True)
                break
        time.sleep(.002)

    save_trace(
        records,
        target
    )

    print()

    if clicked:
        print("ONE SWING ATTEMPTED. Check whether the ball traveled.", flush=True)
    else:
        print("RETURN METER NOT TRACKED. No late fallback click was sent.", flush=True)
        print("Inspect automation_debug\\green_return_trace.png for return tracking.", flush=True)

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
    print("AUTOMATIC SHOT CONTROLLER V26")
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
        force_mouse_release()
        print("WGT CONTROLLER CLOSED")




# ============================================================
# V34: ONE-RUN SHOT READER, NO AIM MOVEMENT, AUTOMATIC SWING
# Preserves current aiming and camera for the 12-yard, 6-inch-uphill chip.
# Requires existing swing_button.png, automation_accuracy.json, and
# Tesseract in the same setup as V26. No keyboard shortcuts needed.
# ============================================================

import math

DEBUG.mkdir(exist_ok=True)

# User's example: PW shows 16 yards, 12 yards to pin, 6 inches up.
# These are explicit fallback inputs ONLY for this exact shot.
PRESET_YARDS = 12
PRESET_UP_IN = 6
PRESET_CLUB_YARDS = 16
PRESET_WIND_LEFT_MPH = 11
COUNTDOWN = 8


def _ocr_region(frame, bounds, psm=6):
    h, w = frame.shape[:2]
    x0,y0,x1,y1 = bounds
    patch = frame[int(h*y0):int(h*y1), int(w*x0):int(w*x1)]
    if patch.size == 0:
        return ''
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4,
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    readings = []
    for im in (gray, cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]):
        try:
            readings.append(pytesseract.image_to_string(im, config=f'--psm {psm}'))
        except Exception:
            pass
    return '\n'.join(readings)


def read_shot(frame):
    # The floating distance sign is in the upper-middle of the WGT scene.
    text = _ocr_region(frame, (.38,.17,.65,.48), 11)
    text = text.upper().replace('|','1').replace('O','0')
    y = re.search(r'(\d{1,3})\s*(?:YDS?|YD5)', text)
    # WGT may OCR the large digits and unit on separate lines.
    if not y:
        y = re.search(r'(?:^|\n)\s*(\d{1,3})\s*(?:\n|\s+).*?YDS?', text, re.S)
    z = re.search(r'(\d{1,2})\s*(?:IN|1N)', text)
    yards = int(y.group(1)) if y else None
    up = int(z.group(1)) if z else None
    return yards, up, text


def find_flagpole(frame):
    # Search vertical bright strokes beneath the central target sign.
    h,w = frame.shape[:2]
    x0,x1 = int(.34*w),int(.68*w)
    y0,y1 = int(.31*h),int(.70*h)
    crop = frame[y0:y1,x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0,0,145), (179,95,255))
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=19,
                            minLineLength=max(21,int(.045*h)), maxLineGap=10)
    candidates=[]
    if lines is not None:
        # OpenCV may return (N,1,4), (N,4), or a single 4-element line.
        # Normalize to rows of four coordinates before iterating.
        segments = np.asarray(lines).reshape(-1, 4)
        for ax, ay, bx, by in segments:
            ax, ay, bx, by = int(ax), int(ay), int(bx), int(by)
            dx=abs(bx-ax); dy=abs(by-ay)
            cx=(ax+bx)/2+x0; cy=(ay+by)/2+y0
            if dy >= 22 and dx <= max(5,dy*.12):
                # Favor center field and taller poles.
                score = dy-abs(cx-w*.5)*.55
                candidates.append((score,int(cx),int(cy)))
    if not candidates:
        return None
    _,x,y=max(candidates)
    return x,y


def check_key_stop():
    if pressed(VK_F9):
        raise KeyboardInterrupt('F9 stop')


def v32_main():
    global POWER_TARGET
    setup()
    print('WGT V37 - FIXED GREEN BAR DETECTION | NO AIM DRAG')
    print('Monitor 1: WGT | Monitor 2: PowerShell')
    print('F9: emergency stop | Ctrl+C: terminal stop')
    print('One shot only. Existing game must be at address before running.')
    template=cv2.imread(str(TEMPLATE),cv2.IMREAD_GRAYSCALE)
    if template is None:
        print('Missing swing_button.png in WGT_Bot folder');return
    if not CALIBRATION.exists():
        print('Missing automation_accuracy.json in WGT_Bot folder');return
    input('Press Enter for countdown; then click WGT once and HANDS OFF... ')
    for n in range(COUNTDOWN,0,-1):
        check_key_stop();print(f'{n}...',flush=True);time.sleep(1)
    hwnd=find_wgt()
    if hwnd is None:
        print('WGT window not found. No shot.');return
    guard(hwnd)
    with mss.MSS() as sct:
        rect=get_rect(hwnd)
        if rect is None:
            print('WGT rect unavailable');return
        before=capture(sct,rect)
        cv2.imwrite(str(DEBUG/'v35_before.png'),before)
        if not verify_player(before):
            print('Player not verified; no shot.');return
        original_score,original_pos=swing_button(before,template,rect)
        if original_pos is None or original_score < MIN_SWING_SCORE:
            print(f'Swing button not visible BEFORE aiming ({original_score:.3f}).');return
        y,up,raw=read_shot(before)
        print('OCR text:',repr(raw[:120]))
        if y is None:
            y=PRESET_YARDS
            print('OCR could not confirm yardage. USING EXAMPLE PRESET: 12 yd.')
        if up is None:
            up=PRESET_UP_IN
            print('OCR could not confirm elevation. USING EXAMPLE PRESET: +6 in.')
        print(f'TARGET: {y} yards, +{up} inches uphill')
        if not (9 <= y <= 16 and 0 <= up <= 12):
            print('Shot differs from calibrated 12-yard chip. No automatic guessed shot.');return
        # Approximate uphill adjustment: 1 extra yard for each 36 in of rise.
        effective_yards=y+up/36.0
        # Short-chip power estimate; real WGT swing curve is not linear.
        # Do not claim this achieves exact carry.
        POWER_TARGET=min(.95,max(.45,effective_yards/PRESET_CLUB_YARDS))
        print(f'PW short-chip starting power: {POWER_TARGET:.0%} (estimate)')
        # V34: Preserve the existing flag location and camera view.
        # NO aim drag, NO flagpole click, and NO camera rotation.
        print('AIM LOCKED: leaving flag and camera exactly where they are.')
        print('Starting automatic swing from current view.', flush=True)
        guard(hwnd)
        one_swing(hwnd,rect,sct,template)
        print('Attempt complete. Check whether WGT ball traveled.',flush=True)


if __name__ == '__main__':
    try:
        v32_main()
    except (KeyboardInterrupt,pyautogui.FailSafeException):
        print('Stopped by emergency release.')
    except Exception as e:
        import traceback
        print('Stopped safely:',e)
        traceback.print_exc()
    finally:
        force_mouse_release()
        print('V35 CONTROLLER CLOSED')
