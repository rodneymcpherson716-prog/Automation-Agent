
"""
WGT GOLF - TIGERWOULD52
ONE-PUTT CONTROLLER V44 (POWER FEEDBACK TEST)

You open WGT Golf yourself.
The bot never launches WGT or Microsoft Edge.

Countdown = One experimental putt
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



# ------------------------------------------------------------
# One-shot 17.5-foot downhill putt test. No flag, camera or resizing.
# These parameters are estimates and are not calibrated against WGT.
# ------------------------------------------------------------
PUTT_FEET = 17.5
DOWNHILL_INCHES = 4.8
GREEN_SPEED = "VERY FAST"
# This low-power travel is an initial hypothesis, not a validated formula.
# Initial experimental target, NOT a validated distance-to-power formula.
# No 100% tee-shot behavior on putts.
PUTT_POWER_TARGET = 0.18
MAX_PUTT_PULL_PIXELS = 180
MAX_PULL_SECONDS = 3.5
RETURN_TIMEOUT = 3.5


def confirm_putt_display(frame):
    """Guard against accidentally taking a full swing on another screen."""
    h, w = frame.shape[:2]
    bottom = frame[int(h*.88):h, :int(w*.21)]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(enlarged, config="--psm 11").lower()
    if not re.search(r"\bputt\b|\bputter\b", txt):
        print("PUTT HUD not verified. Stopping without a shot. OCR:", repr(txt))
        return False
    return True


def execute_one_putt(hwnd, rect, sct, template):
    guard(hwnd)
    frame = capture(sct, rect)
    if not confirm_putt_display(frame):
        return
    if not verify_player(frame):
        print("Player not verified. No shot.")
        return

    hits = []
    for _ in range(3):
        guard(hwnd)
        hits.append(swing_button(capture(sct, rect), template, rect))
        time.sleep(.08)
    confidence, position = hits[-1]
    if position is None or min(x[0] for x in hits) < MIN_SWING_SCORE:
        print("Swing button not verified. No shot. Confidence:", round(confidence, 3))
        return

    meter = meter_rect(rect)
    target = calibrated_target(rect, meter)
    print("Putt: 17.5 ft, downhill 4.8 in, very fast green")
    print(f"Experimental putting target: {PUTT_POWER_TARGET:.0%} of indicated power")
    print("Aim/camera/window unchanged")
    print("Saved accuracy mark:", target)

    # WGT on Monitor 1. No window, aim, or camera changes.
    monitor = next((m for m in sct.monitors[1:]
                    if m['left'] <= position[0] < m['left']+m['width']
                    and m['top'] <= position[1] < m['top']+m['height']), None)
    if monitor is None:
        print('WGT monitor unknown; no putt.'); return

    floor = monitor['top'] + 22
    if position[1] - floor < 60:
        print('Not enough upward mouse travel; no putt.'); return

    pyautogui.moveTo(*position, duration=.10)
    saw_meter = False
    reached_target = False
    best_power = 0.0
    traveled = 0
    start_pull = time.perf_counter()
    try:
        guard(hwnd)
        pyautogui.mouseDown(button='left')
        while time.perf_counter()-start_pull < MAX_PULL_SECONDS:
            guard(hwnd)
            power = estimate_yellow_power(capture(sct, meter))
            if power is not None:
                saw_meter = True
                best_power = max(best_power, power)
                if power >= PUTT_POWER_TARGET:
                    reached_target = True
                    break
            if traveled >= MAX_PUTT_PULL_PIXELS:
                break
            available = pyautogui.position().y - floor
            if available < 6: break
            delta = min(8, available, MAX_PUTT_PULL_PIXELS-traveled)
            if delta <= 0: break
            pyautogui.moveRel(0, -delta, duration=.03)
            traveled += delta
            time.sleep(.012)
    finally:
        force_mouse_release()
        print('PUTT MOUSE RELEASE SENT',flush=True)

    print(f'Power feedback: best={best_power:.0%}; pulled {traveled}px; '
          f'target verified={reached_target}; meter readable={saw_meter}', flush=True)
    if not saw_meter:
        print('Power meter could not be read. Release may or may not have started a stroke.')
        print('No speculative accuracy click. Use the meter trace for calibration.')
        return
    if not reached_target:
        print('Power target not confirmed. Return tracking will be attempted if a stroke started.')

    click_x = meter['left'] + target
    click_y = meter['top'] + meter['height']//2
    guard(hwnd)
    user32.SetCursorPos(int(click_x), int(click_y))

    history = deque(maxlen=4)
    records = []
    last_record = -1.0
    t0 = time.perf_counter()
    clicked = False
    while time.perf_counter() - t0 < RETURN_TIMEOUT:
        guard(hwnd)
        now = time.perf_counter()
        frame = capture(sct, meter)
        edge = green_edge(frame, rect, meter)
        elapsed = now - t0
        if elapsed - last_record > .065:
            records.append((elapsed, frame.copy(), edge))
            last_record = elapsed
        if edge is not None:
            history.append((now, edge))
            if len(history) >= 2:
                old_t, old_edge = history[0]
                dt = max(.001, now - old_t)
                speed = (edge - old_edge) / dt
                if 20 <= speed <= 1500:
                    lead = min(55, max(7, speed * .040))
                    if edge + lead >= target and edge <= target + 7:
                        guard(hwnd)
                        user32.mouse_event(0x0002, 0, 0, 0, 0)
                        user32.mouse_event(0x0004, 0, 0, 0, 0)
                        print("PUTT ACCURACY CLICK: edge", edge, "target", target,
                              "speed", round(speed), flush=True)
                        clicked = True
                        break
        time.sleep(.003)
    save_trace(records, target)
    if not clicked:
        print("Return bar was not tracked; no guessed late click.")
    print("One putt attempt finished. Check actual ball movement and distance.")
    print(r"Trace: automation_debug\green_return_trace.png")


def main():
    setup()
    if not TEMPLATE.is_file() or not CALIBRATION.is_file():
        print('Required existing swing_button.png / automation_accuracy.json not found.')
        return
    template = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print('Cannot read swing button template.'); return
    print('WGT PUTT V44 — ONE PUTT — POWER FEEDBACK — NO AIM OR CAMERA CHANGES')
    print('17.5 ft; 4.8 in downhill; very fast green')
    print('Experimental 18% meter-power target, not calibrated for WGT.')
    print('F9 emergency stop; Ctrl+C terminal stop.')
    input('When the same putt is on screen, press Enter to start 8-second countdown...')
    for n in range(8,0,-1):
        if pressed(VK_F9):
            print('Stopped by F9'); return
        print(str(n)+'...',flush=True)
        time.sleep(1)
    try:
        with mss.MSS() as sct:
            hwnd = find_wgt()
            if hwnd is None:
                print('WGT window not found.'); return
            if user32.GetForegroundWindow() != hwnd:
                print('WGT is not the focused window. No putt.'); return
            rect = get_rect(hwnd)
            if rect is None: return
            execute_one_putt(hwnd, rect, sct, template)
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        print('Emergency stop.')
    except Exception as exc:
        print('Stopped safely:', exc)
    finally:
        force_mouse_release()
        print('PUTTING CONTROLLER CLOSED')


if __name__ == '__main__':
    main()
