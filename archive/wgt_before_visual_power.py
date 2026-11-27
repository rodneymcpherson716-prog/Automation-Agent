import json

"""
WGT GOLF - TIGERWOULD52
CONTINUOUS SINGLE PLAYER CONTROLLER

You open WGT Golf yourself.
The bot never launches the game or Microsoft Edge.

Features:
- Automatically starts when WGT becomes active
- Verifies Tigerwould52
- Reads hole and shot numbers
- Reads target distance and selected club distance
- Estimates swing power from those distances
- Tracks the returning green bar
- Attempts the final accuracy click
- Waits for WGT to confirm the next shot
- Continues without pressing F8 again

F9 = EMERGENCY STOP
Ctrl+C = Stop from the terminal

EXPERIMENTAL:
Club selection, wind aiming, elevation corrections,
and putting are not yet automated.
"""

import ctypes
from ctypes import wintypes
import re
import time
from pathlib import Path
from collections import deque

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


# ==================================================
# SETTINGS
# ==================================================

PLAYER = "tigerwould52"

ROOT = Path(__file__).resolve().parent

SWING_PNG = ROOT / "swing_button.png"
CALIBRATION = ROOT / "wgt_accuracy.json"

MIN_MATCH = 0.83

# Experimental power calibration.
# Previously, 140 pixels started a full-power swing.
REFERENCE_BACKSWING = 280

MAX_ATTEMPTS = 1

F9 = 0x78

user32 = ctypes.windll.user32

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


# ==================================================
# WINDOWS SETUP
# ==================================================

def configure():
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

    for path in paths:
        if Path(path).is_file():
            pytesseract.pytesseract.tesseract_cmd = path
            break


def stopped():
    return bool(
        user32.GetAsyncKeyState(F9) & 0x8000
    )


# ==================================================
# FIND WGT GOLF
# ==================================================

def game_window():
    results = []

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p
    )

    def inspect(hwnd, unused):
        if (
            user32.IsWindowVisible(hwnd)
            and not user32.IsIconic(hwnd)
        ):
            n = user32.GetWindowTextLengthW(hwnd)

            if n:
                buf = ctypes.create_unicode_buffer(n + 1)

                user32.GetWindowTextW(
                    hwnd,
                    buf,
                    n + 1
                )

                if "wgt golf" in buf.value.lower():
                    results.append(hwnd)

        return True

    user32.EnumWindows(
        callback_type(inspect),
        0
    )

    return results[0] if results else None


def rect_of(hwnd):
    rect = wintypes.RECT()

    if not user32.GetWindowRect(
        hwnd,
        ctypes.byref(rect)
    ):
        return None

    w = rect.right - rect.left
    h = rect.bottom - rect.top

    if w < 600 or h < 400:
        return None

    return {
        "left": rect.left,
        "top": rect.top,
        "width": w,
        "height": h
    }


def guard(hwnd):
    if stopped():
        raise KeyboardInterrupt("F9")

    if (
        user32.GetForegroundWindow() != hwnd
        or game_window() != hwnd
    ):
        raise RuntimeError(
            "WGT lost focus; stopped without further clicks"
        )


# ==================================================
# SCREEN CAPTURE
# ==================================================

def grab(sct, rect):
    return np.asarray(
        sct.grab(rect)
    )[:, :, :3].copy()


def crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]

    return frame[
        int(y1 * h):int(y2 * h),
        int(x1 * w):int(x2 * w)
    ]


# ==================================================
# OCR HELPERS
# ==================================================

def read_ocr(patch, psm=6, digits=False):
    if patch.size == 0:
        return ""

    gray = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2GRAY
    )

    big = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    config = f"--psm {psm}"

    if digits:
        config += (
            " -c tessedit_char_whitelist=0123456789"
        )

    return pytesseract.image_to_string(
        big,
        config=config
    ).strip()


# ==================================================
# VERIFY PLAYER
# ==================================================

def player_ok(frame):
    regions = (
        (.825, .056, .911, .106),
        (.785, .045, .965, .118),
        (.72, .02, 1, .22)
    )

    for region in regions:
        area = crop(frame, *region)

        text = read_ocr(area, 6).lower()

        clean = re.sub(
            r"[^a-z0-9]",
            "",
            text
        )

        if PLAYER in clean:
            return True

    return False


# ==================================================
# READ HOLE NUMBER
# ==================================================

def read_hole(frame):
    regions = (
        (.059, .102, .085, .147),
        (.055, .098, .09, .153)
    )

    for region in regions:
        text = read_ocr(
            crop(frame, *region),
            psm=10,
            digits=True
        )

        matches = re.findall(
            r"\d{1,2}",
            text
        )

        if len(matches) == 1:
            value = int(matches[0])

            if 1 <= value <= 18:
                return value

    return None


# ==================================================
# READ SHOT NUMBER
# ==================================================



def read_shot_digits(frame):
    h, w = frame.shape[:2]

    # Isolate the small shot label beneath the score.
    regions = [
        (.887, .079, .949, .117),
        (.895, .083, .946, .116),
        (.879, .076, .953, .124)
    ]

    votes = []

    for x1, y1, x2, y2 in regions:
        patch = frame[
            int(h*y1):int(h*y2),
            int(w*x1):int(w*x2)
        ]

        if patch.size == 0:
            continue

        gray = cv2.cvtColor(
            patch, cv2.COLOR_BGR2GRAY
        )

        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        for variant in (gray, binary):
            big = cv2.resize(
                variant, None,
                fx=5, fy=5,
                interpolation=cv2.INTER_CUBIC
            )

            for mode in (7, 6):
                text = pytesseract.image_to_string(
                    big, config=f"--psm {mode}"
                ).upper()

                clean = re.sub(
                    r"[^A-Z0-9]", "", text
                ).replace("SH0T", "SHOT")

                match = re.search(
                    r"([1-9I]|1[0-9])"
                    r"(?:ST|5T|ND|HD|MD|RD|RO|TH|TM|T)?"
                    r"SHOT",
                    clean
                )

                if match:
                    number = match.group(1).replace(
                        "I", "1"
                    )
                    votes.append(int(number))

    if not votes:
        return None

    best = max(set(votes), key=votes.count)

    if votes.count(best) >= 2:
        return best

    return None


def read_shot_previous(frame):
    number = read_shot_digits(frame)

    if number is not None:
        return number

    hole = read_hole(frame)

    pars = {
        1: 4, 2: 4, 3: 5,
        4: 4, 5: 3, 6: 5,
        7: 3, 8: 4, 9: 4
    }

    par = pars.get(hole)

    if par is None:
        return None

    h, w = frame.shape[:2]

    regions = [
        (.870, .025, .969, .085),
        (.870, .065, .973, .126),
        (.830, .020, .979, .145)
    ]

    labels = [
        ("FORTRIPLEBOGEY", 3),
        ("FORDOUBLEBOGEY", 2),
        ("FORBOGEY", 1),
        ("FORBIRDIE", -1),
        ("FOREAGLE", -2),
        ("FORPAR", 0)
    ]

    for x1, y1, x2, y2 in regions:
        patch = frame[
            int(h*y1):int(h*y2),
            int(w*x1):int(w*x2)
        ]

        if patch.size == 0:
            continue

        gray = cv2.cvtColor(
            patch, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None, fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        binary = cv2.threshold(
            big, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        for image in (big, binary):
            text = pytesseract.image_to_string(
                image, config="--psm 6"
            ).upper()

            clean = re.sub(
                r"[^A-Z]", "", text
            )

            for label, offset in labels:
                if label in clean:
                    shot = par + offset

                    if 1 <= shot <= 30:
                        print(
                            "SHOT LABEL:",
                            label,
                            "->",
                            shot
                        )
                        return shot

    return None



# ==================================================
# READ SELECTED CLUB DISTANCE
# ==================================================




def read_club_yards(frame):
    from collections import Counter

    h, w = frame.shape[:2]
    votes = []

    regions = [
        (.135, .935, .170, .985),
        (.130, .934, .174, .988),
        (.125, .923, .177, .992),
        (.110, .920, .170, .980)
    ]

    for x1, y1, x2, y2 in regions:
        patch = frame[
            int(h*y1):int(h*y2),
            int(w*x1):int(w*x2)
        ]

        if patch.size == 0:
            continue

        gray = cv2.cvtColor(
            patch, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None,
            fx=5, fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        _, binary = cv2.threshold(
            big, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        for variant in (big, binary, 255-binary):
            for mode in (6, 7, 13):
                text = pytesseract.image_to_string(
                    variant,
                    config=(
                        f"--psm {mode} "
                        "-c tessedit_char_whitelist=0123456789"
                    )
                ).strip()

                match = re.fullmatch(
                    r"\s*(\d{2,3})\s*",
                    text
                )

                if match:
                    yards = int(match.group(1))

                    if 25 <= yards <= 350:
                        votes.append(yards)

    if not votes:
        return None

    best, count = Counter(votes).most_common(1)[0]

    if count >= 2:
        return best

    return None


# ==================================================
# FIND BLUE TARGET DISTANCE SIGN
# ==================================================



def read_target(frame):
    from collections import Counter

    h, w = frame.shape[:2]
    x0 = int(w * .25)
    y0 = int(h * .07)

    roi = frame[
        y0:int(h * .45),
        x0:int(w * .80)
    ]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        (85, 30, 35),
        (125, 255, 255)
    )

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    candidates = []

    for x, y, bw, bh, pixels in stats[1:]:
        fill = pixels / max(1, bw * bh)

        if (
            35 <= bw <= 220
            and 30 <= bh <= 160
            and fill >= .65
        ):
            cx = (x0 + x + bw / 2) / w
            cy = (y0 + y + bh / 2) / h

            confidence = (
                fill
                - .15 * abs(cx - .52)
                - .08 * abs(cy - .20)
            )

            candidates.append(
                (confidence, x, y, bw, bh)
            )

    if not candidates:
        return None

    _, x, y, bw, bh = max(candidates)

    sign = roi[
        max(0, y - 4):min(roi.shape[0], y + bh + 4),
        max(0, x - 4):min(roi.shape[1], x + bw + 4)
    ]

    H, W = sign.shape[:2]

    upper = sign[
        int(H * .05):int(H * .52),
        int(W * .12):int(W * .72)
    ]

    gray = cv2.cvtColor(
        upper, cv2.COLOR_BGR2GRAY
    )

    big = cv2.resize(
        gray, None,
        fx=5, fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    _, binary = cv2.threshold(
        big, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    votes = []

    for variant in (big, binary, 255 - binary):
        for mode in (6, 7):
            text = pytesseract.image_to_string(
                variant,
                config=(
                    f"--psm {mode} "
                    "-c tessedit_char_whitelist=0123456789."
                )
            ).strip()

            if re.fullmatch(
                r"\d{1,3}(?:\.\d+)?",
                text
            ):
                votes.append(text)

    if not votes:
        return None

    value = Counter(votes).most_common(1)[0][0]

    # Never treat a decimal putting distance
    # in feet as regular golf yards.
    if "." in value:
        return None

    yards = int(value)

    if 5 <= yards <= 600:
        return yards, None

    return None


# ==================================================
# SWING BUTTON DETECTION
# ==================================================

def find_swing(frame, template, rect):
    h, w = frame.shape[:2]

    ox = int(w * .55)
    oy = int(h * .45)

    gray = cv2.cvtColor(
        frame[
            oy:int(h * .92),
            ox:
        ],
        cv2.COLOR_BGR2GRAY
    )

    best = (0.0, None)

    scales = (
        .65, .75, .85, .95,
        1.0, 1.05, 1.15,
        1.25, 1.35
    )

    for scale in scales:
        sw = round(
            template.shape[1] * scale
        )

        sh = round(
            template.shape[0] * scale
        )

        if (
            sw < 15
            or sh < 15
            or sw >= gray.shape[1]
            or sh >= gray.shape[0]
        ):
            continue

        small = cv2.resize(
            template,
            (sw, sh),
            interpolation=cv2.INTER_AREA
        )

        if small.std() < 5:
            continue

        result = cv2.matchTemplate(
            gray,
            small,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, p = cv2.minMaxLoc(
            result
        )

        if score > best[0]:
            position = (
                rect["left"] + ox + p[0] + sw // 2,
                rect["top"] + oy + p[1] + sh // 2
            )

            best = (score, position)

    return best


# ==================================================
# SWING METER LOCATION
# ==================================================

def meter_rect(rect):
    return {
        "left": (
            rect["left"]
            + int(rect["width"] * .12)
        ),
        "top": (
            rect["top"]
            + int(rect["height"] * .940)
        ),
        "width": (
            int(rect["width"] * .82)
        ),
        "height": max(
            30,
            int(rect["height"] * .055)
        )
    }


# ==================================================
# SAVED ACCURACY MARK
# ==================================================

def accuracy_target(rect, meter):
    data = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    fraction = float(
        data["accuracy_x_fraction"]
    )

    target = (
        round(fraction * rect["width"])
        - (
            meter["left"]
            - rect["left"]
        )
    )

    if not (
        25 < target < meter["width"] - 25
    ):
        raise RuntimeError(
            "Accuracy mark outside meter"
        )

    return target


# ==================================================
# GREEN RETURN BAR TRACKING
# ==================================================

def green_edge(image, rect, meter):
    hh = image.shape[0]

    sub = image[
        int(hh * .35):int(hh * .85),
        :
    ]

    if sub.size == 0:
        return None

    hsv = cv2.cvtColor(
        sub,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        (35, 65, 65),
        (95, 255, 255)
    )

    counts = np.count_nonzero(
        mask,
        axis=0
    )

    active = (
        counts >= max(
            3,
            int(sub.shape[0] * .38)
        )
    ).astype("uint8")

    k = max(
        5,
        int(rect["width"] * .006)
    )

    active = cv2.morphologyEx(
        active[None, :],
        cv2.MORPH_CLOSE,
        np.ones((1, k), np.uint8)
    )[0]

    coords = np.flatnonzero(
        active
    )

    if not len(coords):
        return None

    runs = np.split(
        coords,
        np.flatnonzero(
            np.diff(coords) > 1
        ) + 1
    )

    offset = (
        meter["left"] - rect["left"]
    )

    lo = (
        int(rect["width"] * .420)
        - offset
    )

    hi = (
        int(rect["width"] * .500)
        - offset
    )

    allowed = [
        run
        for run in runs
        if (
            lo <= int(run[0]) <= hi
            and len(run) >= max(
                10,
                int(rect["width"] * .01)
            )
        )
    ]

    if not allowed:
        return None

    return int(
        max(allowed, key=len)[-1]
    )


# ==================================================
# DISTANCE-BASED AUTOMATIC SHOT
# ==================================================

def do_shot(
    hwnd,
    rect,
    sct,
    template,
    shot_state
):
    guard(hwnd)

    img = grab(sct, rect)

    if not player_ok(img):
        raise RuntimeError(
            "Player could not be verified"
        )

    target = read_target(img)
    club = read_club_yards(img)

    if target is None or club is None:
        raise RuntimeError(
            "Distance or club yardage unreadable. "
            "No swing attempted."
        )

    yards, elevation = target

    # PGA National front nine par values,
    # confirmed from the current scorecard.
    # Update these for a different course.
    hole_par = {
        1: 4,
        2: 4,
        3: 5,
        4: 4,
        5: 3,
        6: 5,
        7: 3,
        8: 4,
        9: 4
    }

    par = hole_par.get(shot_state[0])
    tee_shot = shot_state[1] == 1

    if tee_shot and par in (4, 5):
        # Request maximum configured backswing.
        ratio = 1.0
        pixels = REFERENCE_BACKSWING

        print(
            f"PAR {par} TEE SHOT: FULL POWER REQUESTED"
        )

    else:
        if tee_shot and par is None:
            raise RuntimeError(
                "Unknown hole par. Tee power not guessed."
            )

        ratio = yards / club

        if ratio > 1.12 or ratio < .13:
            raise RuntimeError(
                f"Club {club} yd unsuitable "
                f"for {yards} yd target."
            )

        pixels = round(
            REFERENCE_BACKSWING
            * min(1, max(.16, ratio))
        )

        if tee_shot and par == 3:
            print(
                "PAR 3: DISTANCE-BASED TEE SHOT"
            )

    print()
    print(
        f"HOLE {shot_state[0]}, "
        f"SHOT {shot_state[1]}"
    )

    print(
        f"Target distance: {yards} yards"
    )

    print(
        f"Selected club distance: {club} yards"
    )

    print(
        f"Estimated power: {ratio * 100:.0f}%"
    )

    print(
        f"Backswing distance: {pixels} pixels"
    )

    print(
        "Elevation reading:",
        elevation if elevation is not None
        else "UNKNOWN"
    )

    print(
        "Power is experimental. "
        "Aim and club remain as selected."
    )

    detections = []

    for i in range(3):
        guard(hwnd)

        detections.append(
            find_swing(
                grab(sct, rect),
                template,
                rect
            )
        )

        time.sleep(.10)

    score, pos = detections[-1]

    if (
        pos is None
        or min(
            x[0] for x in detections
        ) < MIN_MATCH
    ):
        raise RuntimeError(
            f"Swing button not confirmed: "
            f"{score:.3f}"
        )

    meter = meter_rect(rect)

    mark = accuracy_target(
        rect,
        meter
    )

    # Start backswing at calculated power.
    pyautogui.moveTo(
        *pos,
        duration=.10
    )

    try:
        guard(hwnd)

        pyautogui.mouseDown()

        pyautogui.moveRel(
            0,
            -pixels,
            duration=.80
        )

    finally:
        pyautogui.mouseUp()

    print(
        "Backswing started. "
        "Tracking green return..."
    )

    t0 = time.perf_counter()

    history = deque(maxlen=5)

    advancing = 0
    last_edge = None

    while (
        time.perf_counter() - t0
        < 8.0
    ):
        guard(hwnd)

        edge = green_edge(
            grab(sct, meter),
            rect,
            meter
        )

        now = time.perf_counter()

        if edge is not None:
            if (
                last_edge is not None
                and edge > last_edge + 2
            ):
                advancing += 1

            elif (
                last_edge is not None
                and edge < last_edge - 10
            ):
                advancing = 0

            history.append(
                (now, edge)
            )

            last_edge = edge

            if (
                advancing >= 2
                and len(history) >= 3
            ):
                a, b, c = list(history)[-3:]

                move = c[1] - a[1]

                dt = max(
                    .001,
                    c[0] - a[0]
                )

                speed = max(
                    0,
                    move / dt
                )

                lead = max(
                    6,
                    min(
                        24,
                        int(speed * .020)
                    )
                )

                if (
                    move > 4
                    and mark - lead
                    <= edge <= mark + 5
                ):
                    guard(hwnd)

                    pyautogui.click()

                    print(
                        f"ACCURACY CLICK: "
                        f"edge={edge}, mark={mark}"
                    )

                    return True

        time.sleep(.007)

    raise RuntimeError(
        "No confirmed green-bar crossing. "
        "Controller stopped."
    )


# ==================================================
# WAIT UNTIL WGT ADVANCES TO NEXT SHOT
# ==================================================

def wait_next_state(
    hwnd,
    rect,
    sct,
    previous
):
    deadline = (
        time.monotonic() + 30
    )

    while time.monotonic() < deadline:
        guard(hwnd)

        time.sleep(1)

        img = grab(sct, rect)

        new = (
            read_hole(img),
            read_shot(img)
        )

        if (
            all(x is not None for x in new)
            and new != previous
        ):
            print(
                "Game advanced to:",
                new
            )

            return new

    raise RuntimeError(
        "Could not confirm the next shot. "
        "No repeat click."
    )


# ==================================================
# MAIN CONTINUOUS CONTROLLER
# ==================================================

def main():
    configure()

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print(
            "Tesseract OCR not available."
        )
        return

    template = cv2.imread(
        str(SWING_PNG),
        cv2.IMREAD_GRAYSCALE
    )

    if (
        template is None
        or not CALIBRATION.exists()
    ):
        print(
            "Missing swing_button.png "
            "or wgt_accuracy.json."
        )
        return

    print("=" * 55)
    print("WGT GOLF - TIGERWOULD52")
    print("CONTINUOUS SOLO CONTROLLER")
    print("=" * 55)

    print(
        "Distance-based power: EXPERIMENTAL"
    )

    print(
        "Automatic next-shot attempts: ENABLED"
    )

    print(
        "No F8 required"
    )

    print(
        "F9 = EMERGENCY STOP"
    )

    print(
        "You must open WGT Golf yourself."
    )

    print()
    print(
        "Waiting for WGT to become active..."
    )

    try:
        with mss.mss() as sct:
            active_since = None
            hwnd = None

            # Wait until user activates WGT.
            while True:
                if stopped():
                    return

                h = game_window()

                if (
                    h
                    and user32.GetForegroundWindow()
                    == h
                ):
                    if h != hwnd:
                        active_since = time.monotonic()
                        hwnd = h

                    if (
                        active_since
                        and time.monotonic()
                        - active_since >= 4
                    ):
                        break

                else:
                    active_since = None
                    hwnd = None

                time.sleep(.25)

            rect = rect_of(hwnd)

            first_hole = None

            for attempt in range(
                1,
                MAX_ATTEMPTS + 1
            ):
                guard(hwnd)

                img = grab(sct, rect)

                if not player_ok(img):
                    raise RuntimeError(
                        "Tigerwould52 not verified."
                    )

                print("Checking hole and shot...")

                state = (None, None)
                previous = None
                consecutive = 0
                deadline = time.monotonic() + 20

                while time.monotonic() < deadline:
                    guard(hwnd)

                    img = grab(sct, rect)

                    candidate = (
                        read_hole(img),
                        read_shot(img)
                    )

                    if None not in candidate:
                        if candidate == previous:
                            consecutive += 1
                        else:
                            consecutive = 1

                        previous = candidate

                        if consecutive >= 2:
                            state = candidate
                            break
                    else:
                        consecutive = 0
                        previous = None

                    time.sleep(0.5)

                if None in state:
                    raise RuntimeError(
                        "Shot information not confirmed "
                        "after 20 seconds. No swing attempted."
                    )

                print("SHOT VERIFIED:", state)

                if first_hole is None:
                    first_hole = state[0]

                if state[0] >= first_hole + 9:
                    print(
                        "Nine-hole span completed."
                    )
                    return

                print()
                print(
                    f"AUTOMATIC SHOT ATTEMPT {attempt}"
                )

                do_shot(
                    hwnd,
                    rect,
                    sct,
                    template,
                    state
                )

                next_state = wait_next_state(
                    hwnd,
                    rect,
                    sct,
                    state
                )

                if next_state[0] >= first_hole + 9:
                    print(
                        "Nine-hole range completed."
                    )
                    return

            print(
                "Shot-attempt limit reached."
            )

    except (
        KeyboardInterrupt,
        pyautogui.FailSafeException
    ):
        print(
            "Emergency stop."
        )

    except Exception as error:
        print(
            "STOPPED SAFELY:",
            error
        )

    finally:
        pyautogui.mouseUp()

        print()
        print(
            "WGT CONTROLLER CLOSED"
        )




def read_shot(frame):
    h, w = frame.shape[:2]

    regions = [
        (.82, .02, .98, .15),
        (.87, .055, .96, .125),
        (.88, .075, .96, .12)
    ]

    def extract_number(text):
        clean = re.sub(
            r"[^A-Z0-9]", "", text.upper()
        )

        clean = clean.replace("SH0T", "SHOT")
        clean = clean.replace("SHQT", "SHOT")

        match = re.search(
            r"(\d{1,2})(?:ST|5T|ND|RD|TH)?"
            r"(?:SHOT|SHT)",
            clean
        )

        if match:
            number = int(match.group(1))
            if 1 <= number <= 30:
                return number

        if re.search(r"I(?:ST|5T)SHOT", clean):
            return 1

        return None

    for region in regions:
        patch = crop(frame, *region)

        if patch.size == 0:
            continue

        # First try the method that previously
        # recognized the full player header.
        text = read_ocr(patch, 6)
        result = extract_number(text)

        if result is not None:
            return result

        gray = cv2.cvtColor(
            patch, cv2.COLOR_BGR2GRAY
        )

        enlarged = cv2.resize(
            gray, None, fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        binary = cv2.threshold(
            enlarged, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        for image in (enlarged, binary):
            for mode in (6, 7, 11):
                text = pytesseract.image_to_string(
                    image,
                    config=f"--psm {mode}"
                )

                result = extract_number(text)

                if result is not None:
                    return result

    # Preserve scoring labels such as FOR BIRDIE.
    if "read_shot_previous" in globals():
        return read_shot_previous(frame)

    return None


if __name__ == "__main__":
    main()


