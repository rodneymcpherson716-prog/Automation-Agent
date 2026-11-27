
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
REFERENCE_BACKSWING = 140

MAX_ATTEMPTS = 60

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


def read_shot(frame):
    h, w = frame.shape[:2]

    regions = [
        (.895, .080, .958, .112),
        (.880, .067, .970, .125),
        (.855, .055, .980, .140)
    ]

    for x1, y1, x2, y2 in regions:
        image = frame[
            int(h*y1):int(h*y2),
            int(w*x1):int(w*x2)
        ]

        if image.size == 0:
            continue

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None,
            fx=5, fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        binary = cv2.threshold(
            big, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        for variant in (big, binary):
            for mode in (7, 6, 11):
                text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {mode}"
                ).upper()

                clean = re.sub(
                    r"[^A-Z0-9]", "", text
                )

                match = re.search(
                    r"(\d{1,2})(?:ST|ND|RD|TH)?"
                    r"(?:SHOT|SH0T|SHT)",
                    clean
                )

                if match:
                    shot = int(match.group(1))

                    if 1 <= shot <= 30:
                        return shot

    return None


# ==================================================
# READ SELECTED CLUB DISTANCE
# ==================================================



def read_club_yards(frame):
    h, w = frame.shape[:2]

    regions = [
        (.140, .950, .169, .977),
        (.134, .947, .175, .981),
        (.143, .955, .168, .974)
    ]

    results = []

    for x1, y1, x2, y2 in regions:
        image = frame[
            int(h*y1):int(h*y2),
            int(w*x1):int(w*x2)
        ]

        if image.size == 0:
            continue

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None,
            fx=5, fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        binary = cv2.threshold(
            big, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        for variant in (big, binary):
            for mode in (6, 7):
                text = pytesseract.image_to_string(
                    variant,
                    config=(
                        f"--psm {mode} "
                        "-c tessedit_char_whitelist=0123456789"
                    )
                ).strip()

                if re.fullmatch(r"\d{2,3}", text):
                    value = int(text)

                    if 25 <= value <= 350:
                        results.append(value)

    if not results:
        return None

    best = max(
        set(results),
        key=results.count
    )

    if results.count(best) >= 2:
        return best

    return None


# ==================================================
# FIND BLUE TARGET DISTANCE SIGN
# ==================================================


def read_target(frame):
    # Read target yards from the top-left WGT display.
    # This display mirrors the blue target sign.
    regions = [
        (.136, .057, .173, .103),
        (.138, .058, .170, .102),
        (.135, .057, .178, .104)
    ]

    for region in regions:
        image = crop(frame, *region)

        if image.size == 0:
            continue

        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None,
            fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        for mode in (6, 7):
            text = pytesseract.image_to_string(
                big,
                config=f"--psm {mode}"
            )

            numbers = re.findall(
                r"\d{1,3}", text
            )

            if len(numbers) == 1:
                yards = int(numbers[0])

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

    ratio = yards / club

    if ratio > 1.12 or ratio < .13:
        raise RuntimeError(
            f"Club {club} yd unsuitable "
            f"for {yards} yd target."
        )

    # Experimental power calculation.
    # This assumes roughly proportional power.
    pixels = round(
        REFERENCE_BACKSWING
        * min(1, max(.16, ratio))
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

                state = (
                    read_hole(img),
                    read_shot(img)
                )

                if None in state:
                    raise RuntimeError(
                        "Hole or shot number unreadable."
                    )

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


if __name__ == "__main__":
    main()
