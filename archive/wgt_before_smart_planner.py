
"""WGT Golf / Tigerwould52 - one-shot supervised controller.

You open WGT yourself. This script never launches a browser or a round.
F9 = emergency stop; Ctrl+C also stops. Do not leave it unattended.

Club choice, putting, wind/elevation adjustment and nine-hole play
are NOT automated yet.
"""

import ctypes
from ctypes import wintypes
import json
import re
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


ROOT = Path(__file__).resolve().parent
PLAYER = "tigerwould52"

SWING_PNG = ROOT / "swing_button.png"
CALIBRATION = ROOT / "wgt_accuracy.json"
DEBUG = ROOT / "wgt_debug"
DEBUG.mkdir(exist_ok=True)

F9 = 0x78
MIN_MATCH = 0.83
MAX_ATTEMPTS = 1
BACKSWING_TIMEOUT = 9.0

PARS = {
    1: 4,
    2: 4,
    3: 5,
    4: 4,
    5: 3,
    6: 5,
    7: 3,
    8: 4,
    9: 4,
}

USER32 = ctypes.windll.user32

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def configure():
    try:
        USER32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)
        )
    except Exception:
        try:
            USER32.SetProcessDPIAware()
        except Exception:
            pass

    paths = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    )

    for name in paths:
        if Path(name).is_file():
            pytesseract.pytesseract.tesseract_cmd = name
            break


def stopping():
    return bool(USER32.GetAsyncKeyState(F9) & 0x8000)


def window():
    found = []

    PROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )

    def inspect(hwnd, unused):
        if (
            USER32.IsWindowVisible(hwnd)
            and not USER32.IsIconic(hwnd)
        ):
            n = USER32.GetWindowTextLengthW(hwnd)

            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                USER32.GetWindowTextW(hwnd, buf, n + 1)

                if "wgt golf" in buf.value.lower():
                    found.append(hwnd)

        return True

    USER32.EnumWindows(PROC(inspect), 0)

    return found[0] if found else None


def geometry(hwnd):
    r = wintypes.RECT()

    if not USER32.GetWindowRect(
        hwnd, ctypes.byref(r)
    ):
        raise RuntimeError(
            "Could not read WGT window."
        )

    area = {
        "left": r.left,
        "top": r.top,
        "width": r.right - r.left,
        "height": r.bottom - r.top,
    }

    if area["width"] < 600 or area["height"] < 400:
        raise RuntimeError(
            "WGT window is too small."
        )

    return area


def guard(hwnd):
    if stopping():
        raise KeyboardInterrupt("F9 emergency stop")

    if (
        USER32.GetForegroundWindow() != hwnd
        or window() != hwnd
    ):
        raise RuntimeError(
            "WGT lost focus; no more actions."
        )


def shot_frame(sct, area):
    return np.asarray(
        sct.grab(area)
    )[:, :, :3].copy()


def crop(frame, bounds):
    h, w = frame.shape[:2]
    a, b, c, d = bounds

    return frame[
        int(b * h):int(d * h),
        int(a * w):int(c * w),
    ]


def ocr_variants(patch, modes=(6, 7)):
    if patch.size == 0:
        return []

    gray = cv2.cvtColor(
        patch, cv2.COLOR_BGR2GRAY
    )

    big = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    binary = cv2.threshold(
        big,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    texts = []

    for image in (big, binary):
        for mode in modes:
            txt = pytesseract.image_to_string(
                image,
                config=f"--psm {mode}",
            ).upper()

            if txt.strip():
                texts.append(txt)

    return texts


def player_ok(frame):
    regions = (
        (.785, .045, .965, .118),
        (.72, .02, 1, .22),
    )

    for box in regions:
        for text in ocr_variants(
            crop(frame, box), (6,)
        ):
            clean = re.sub(
                "[^a-z0-9]", "", text.lower()
            )

            if PLAYER in clean:
                return True

    return False


def read_hole(frame):
    regions = (
        (.059, .102, .085, .147),
        (.055, .098, .09, .153),
    )

    for box in regions:
        patch = crop(frame, box)

        if patch.size == 0:
            continue

        gray = cv2.cvtColor(
            patch, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None, fx=5, fy=5
        )

        text = pytesseract.image_to_string(
            big,
            config=(
                "--psm 10 "
                "-c tessedit_char_whitelist=0123456789"
            ),
        )

        nums = re.findall(r"\d{1,2}", text)

        if len(nums) == 1:
            value = int(nums[0])

            if 1 <= value <= 18:
                return value

    return None


def read_shot(frame, hole=None):
    """Read numbered shots or scoring labels."""

    if hole is None:
        hole = read_hole(frame)

    par = PARS.get(hole)

    regions = (
        (.894, .080, .950, .108),
        (.898, .083, .943, .109),
        (.88, .045, .97, .117),
        (.87, .055, .96, .125),
        (.82, .02, .98, .15),
        (.87, .025, .97, .085),
    )

    all_text = []

    for box in regions:
        all_text.extend(
            ocr_variants(crop(frame, box))
        )

    labels = (
        ("TRIPLEBOGEY", 3),
        ("DOUBLEBOGEY", 2),
        ("BOGEY", 1),
        ("BIRDIE", -1),
        ("EAGLE", -2),
        ("PAR", 0),
    )

    # Recognize scoring labels before numbered shots.
    for raw in all_text:
        clean = re.sub(
            "[^A-Z0-9]", "", raw
        ).replace("0", "O")

        for word, offset in labels:
            if word in clean and (
                "FOR" in clean
                or word in ("BIRDIE", "EAGLE")
            ):
                if par is not None:
                    stroke = par + offset

                    if stroke >= 1:
                        print(
                            f"SHOT LABEL: FOR {word} "
                            f"-> shot {stroke}",
                            flush=True,
                        )
                        return stroke

    # Recognize numbered labels.
    for raw in all_text:
        clean = re.sub(
            "[^A-Z0-9]", "", raw
        )

        clean = clean.replace(
            "SH0T", "SHOT"
        ).replace(
            "SHQT", "SHOT"
        )

        clean = clean.replace(
            "24DSHOT", "2NDSHOT"
        ).replace(
            "2NOSHOT", "2NDSHOT"
        )

        match = re.search(
            r"([1-9]\d?)"
            r"(?:ST|5T|ND|RD|TH|D)?"
            r"(?:SHOT|SHT)",
            clean,
        )

        if match:
            return int(match.group(1))

        if re.search(
            r"I(?:ST|5T)SHOT", clean
        ):
            return 1

    return None


def read_club(frame):
    votes = []

    regions = (
        (.135, .935, .170, .985),
        (.130, .934, .174, .988),
        (.125, .923, .177, .992),
        (.110, .920, .170, .980),
    )

    for box in regions:
        patch = crop(frame, box)

        if patch.size == 0:
            continue

        gray = cv2.cvtColor(
            patch, cv2.COLOR_BGR2GRAY
        )

        big = cv2.resize(
            gray, None, fx=5, fy=5
        )

        binary = cv2.threshold(
            big,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        for image in (
            big,
            binary,
            255 - binary,
        ):
            for mode in (6, 7, 13):
                text = pytesseract.image_to_string(
                    image,
                    config=(
                        f"--psm {mode} "
                        "-c tessedit_char_whitelist="
                        "0123456789"
                    ),
                ).strip()

                if re.fullmatch(
                    r"\d{2,3}", text
                ):
                    value = int(text)

                    if 1 <= value <= 350:
                        votes.append(value)

    if votes:
        number, count = Counter(
            votes
        ).most_common(1)[0]

        if count >= 2:
            return number

    return None


def read_target(frame):
    """Read the blue target yardage sign."""

    h, w = frame.shape[:2]

    x0 = int(w * .25)
    y0 = int(h * .07)

    roi = frame[
        y0:int(h * .45),
        x0:int(w * .80),
    ]

    mask = cv2.inRange(
        cv2.cvtColor(
            roi, cv2.COLOR_BGR2HSV
        ),
        (85, 30, 35),
        (125, 255, 255),
    )

    _, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask, 8
        )
    )

    matches = []

    for x, y, bw, bh, pixels in stats[1:]:
        fill = pixels / max(1, bw * bh)

        if (
            35 <= bw <= 220
            and 30 <= bh <= 160
            and fill >= .65
        ):
            cx = (x0 + x + bw / 2) / w
            cy = (y0 + y + bh / 2) / h

            score = (
                fill
                - .15 * abs(cx - .52)
                - .08 * abs(cy - .20)
            )

            matches.append(
                (score, x, y, bw, bh)
            )

    if not matches:
        return None

    _, x, y, bw, bh = max(matches)

    sign = roi[
        max(0, y - 4):y + bh + 4,
        max(0, x - 4):x + bw + 4,
    ]

    H, W = sign.shape[:2]

    upper = sign[
        int(H * .05):int(H * .52),
        int(W * .12):int(W * .72),
    ]

    if upper.size == 0:
        return None

    gray = cv2.cvtColor(
        upper, cv2.COLOR_BGR2GRAY
    )

    big = cv2.resize(
        gray, None, fx=5, fy=5
    )

    binary = cv2.threshold(
        big,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    votes = []

    for image in (
        big,
        binary,
        255 - binary,
    ):
        for mode in (6, 7):
            text = pytesseract.image_to_string(
                image,
                config=(
                    f"--psm {mode} "
                    "-c tessedit_char_whitelist="
                    "0123456789."
                ),
            ).strip()

            if re.fullmatch(
                r"\d{1,3}(?:\.\d+)?",
                text,
            ):
                votes.append(text)

    if not votes:
        return None

    best, count = Counter(
        votes
    ).most_common(1)[0]

    if count < 2 or "." in best:
        return None

    value = int(best)

    if 5 <= value <= 600:
        return value

    return None


def read_top_yards(frame):
    """Fallback distance from the top-left YDS label."""

    votes = []

    regions = (
        (.14, .045, .19, .105),
        (.135, .043, .195, .11),
    )

    for bounds in regions:
        for raw in ocr_variants(
            crop(frame, bounds)
        ):
            clean = re.sub(
                "[^A-Z0-9]", "", raw
            )

            match = re.search(
                r"(\d{2,3})"
                r"(?:YDS|YOS|VOS|0S|OS|YS|S)",
                clean,
            )

            if match:
                value = int(match.group(1))

                if 5 <= value <= 600:
                    votes.append(value)

    if votes:
        value, count = Counter(
            votes
        ).most_common(1)[0]

        if count >= 2:
            return value

    return None


def swing_button(frame, template, rect):
    h, w = frame.shape[:2]

    ox = int(w * .55)
    oy = int(h * .45)

    area = cv2.cvtColor(
        frame[
            oy:int(h * .92),
            ox:,
        ],
        cv2.COLOR_BGR2GRAY,
    )

    best = (0, None)

    scales = (
        .65, .75, .85, .95,
        1.0, 1.05, 1.15,
        1.25, 1.35,
    )

    for scale in scales:
        sw = round(
            template.shape[1] * scale
        )

        sh = round(
            template.shape[0] * scale
        )

        if not (
            15 <= sw < area.shape[1]
            and 15 <= sh < area.shape[0]
        ):
            continue

        item = cv2.resize(
            template,
            (sw, sh),
            interpolation=cv2.INTER_AREA,
        )

        if item.std() < 5:
            continue

        result = cv2.matchTemplate(
            area,
            item,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, p = (
            cv2.minMaxLoc(result)
        )

        if score > best[0]:
            best = (
                score,
                (
                    rect["left"] + ox
                    + p[0] + sw // 2,

                    rect["top"] + oy
                    + p[1] + sh // 2,
                ),
            )

    return best


def meter_area(rect):
    return {
        "left": (
            rect["left"]
            + int(rect["width"] * .12)
        ),
        "top": (
            rect["top"]
            + int(rect["height"] * .94)
        ),
        "width": int(
            rect["width"] * .82
        ),
        "height": max(
            30,
            int(rect["height"] * .055),
        ),
    }


def target_mark(rect, meter):
    saved = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    x = (
        round(
            float(saved["accuracy_x_fraction"])
            * rect["width"]
        )
        - (
            meter["left"]
            - rect["left"]
        )
    )

    if not (
        25 < x < meter["width"] - 25
    ):
        raise RuntimeError(
            "Saved accuracy mark is outside the swing meter."
        )

    return x


def yellow_power(frame):
    """Estimate power from the yellow/blue meter."""

    h, w = frame.shape[:2]

    band = frame[
        int(h * .28):int(h * .88)
    ]

    if band.size == 0:
        return None

    hsv = cv2.cvtColor(
        band, cv2.COLOR_BGR2HSV
    )

    blue = cv2.inRange(
        hsv,
        (85, 30, 55),
        (130, 255, 255),
    )

    yellow = cv2.inRange(
        hsv,
        (14, 68, 60),
        (45, 255, 255),
    )

    threshold = max(
        3,
        int(band.shape[0] * .25),
    )

    xs = np.arange(w)

    bs = np.flatnonzero(
        (
            np.count_nonzero(
                blue, axis=0
            ) >= threshold
        )
        & (xs >= int(w * .40))
        & (xs <= int(w * .80))
    )

    if len(bs) < 15:
        return None

    blue_start = int(bs.min())

    ys = np.flatnonzero(
        (
            np.count_nonzero(
                yellow, axis=0
            ) >= threshold
        )
        & (xs >= int(w * .12))
        & (xs < blue_start - 5)
    )

    if len(ys) == 0:
        return 0.0

    runs = np.split(
        ys,
        np.flatnonzero(
            np.diff(ys) > 1
        ) + 1,
    )

    runs = [
        r for r in runs
        if len(r) >= max(
            5, int(w * .01)
        )
    ]

    if not runs:
        return 0.0

    left = int(
        max(runs, key=len)[0]
    )

    full = (
        blue_start
        - int(w * .17)
    )

    if full <= 20:
        return None

    estimate = (
        blue_start - left
    ) / full

    return min(
        1.0,
        max(0.0, estimate),
    )


def green_edge(frame, rect, meter):
    """GREEN EDGE DETECTOR V2"""
    h, w = frame.shape[:2]
    band = frame[int(h*.35):int(h*.85)]

    if band.size == 0:
        return None

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, (35, 65, 65), (95, 255, 255)
    )

    counts = np.count_nonzero(mask, axis=0)
    active = (
        counts >= max(3, int(band.shape[0]*.38))
    ).astype(np.uint8)

    gap = max(3, int(w*.003))
    active = cv2.morphologyEx(
        active[None, :],
        cv2.MORPH_CLOSE,
        np.ones((1, gap), np.uint8)
    )[0]

    pixels = np.flatnonzero(active)

    if len(pixels) == 0:
        return None

    runs = np.split(
        pixels,
        np.flatnonzero(np.diff(pixels) > 1) + 1
    )

    candidates = [
        run for run in runs
        if int(run[0]) < int(w*.40)
        and int(run[-1]) < int(w*.94)
        and len(run) >= max(25, int(w*.035))
    ]

    if not candidates:
        return None

    edge = int(max(candidates, key=len)[-1])

    last = getattr(green_edge, "_last_logged", -1000)

    if edge - last >= max(50, int(w*.07)):
        print(
            f"GREEN BAR TRACKED: {edge}",
            flush=True
        )
        green_edge._last_logged = edge

    return edge



def release_mouse():
    """Release the mouse through both input methods."""

    pyautogui.mouseUp(
        button="left"
    )

    USER32.mouse_event(
        0x0004, 0, 0, 0, 0
    )


def do_shot(hwnd, rect, sct, template, state):
    guard(hwnd)

    img = shot_frame(sct, rect)

    if not player_ok(img):
        raise RuntimeError(
            "Player verification lost."
        )

    yards = read_target(img)

    if yards is None:
        yards = read_top_yards(img)

    club = read_club(img)

    hole, stroke = state
    par = PARS.get(hole)

    if yards is None or club is None:
        raise RuntimeError(
            "Cannot verify target yards or selected club. "
            "No swing attempted."
        )

    if stroke == 1 and par in (4, 5):
        ratio = 1.0
    else:
        ratio = yards / club

        if ratio < .13 or ratio > 1.12:
            raise RuntimeError(
                f"Club {club} yd unsuitable "
                f"for target {yards} yd."
            )

    required = max(
        .16,
        min(1.0, ratio),
    )

    if required < .28:
        raise RuntimeError(
            "Very short approach/putt not automated. "
            "No swing."
        )

    print(
        f"HOLE {hole}, SHOT {stroke}, "
        f"target {yards} yd, club {club} yd"
    )

    print(
        f"ESTIMATED POWER TARGET: "
        f"{required:.0%} (experimental)"
    )

    detections = []

    for _ in range(3):
        guard(hwnd)

        detections.append(
            swing_button(
                shot_frame(sct, rect),
                template,
                rect,
            )
        )

        time.sleep(.1)

    score, pos = detections[-1]

    if (
        pos is None
        or any(
            s < MIN_MATCH
            for s, _ in detections
        )
    ):
        raise RuntimeError(
            f"Swing button not confirmed "
            f"({score:.2f}). No swing."
        )

    meter = meter_area(rect)
    mark = target_mark(rect, meter)

    pyautogui.moveTo(
        *pos, duration=.12
    )

    reached = False
    best = 0.0
    confirmed = 0
    tick = 0

    floor = (
        rect["top"]
        + max(
            42,
            int(rect["height"] * .06),
        )
    )

    print(
        "Building backswing; "
        "release is automatic..."
    )

    try:
        guard(hwnd)

        pyautogui.mouseDown(
            button="left"
        )

        pyautogui.moveRel(
            0, -90, duration=.2
        )

        t0 = time.perf_counter()

        while (
            time.perf_counter() - t0
            < BACKSWING_TIMEOUT
        ):
            guard(hwnd)

            tick += 1

            if (
                tick % 2 == 0
                and pyautogui.position().y
                > floor + 10
            ):
                dy = min(
                    15,
                    pyautogui.position().y
                    - floor,
                )

                pyautogui.moveRel(
                    0,
                    -dy,
                    duration=.025,
                )

            power = yellow_power(
                shot_frame(sct, meter)
            )

            if power is not None:
                best = max(best, power)

                if tick % 8 == 0:
                    print(
                        f"POWER {tick}: "
                        f"{power:.0%}",
                        flush=True,
                    )

                if power >= required - .025:
                    confirmed += 1
                else:
                    confirmed = 0

                if confirmed >= 2:
                    reached = True

                    print(
                        f"POWER VERIFIED: "
                        f"{power:.0%}",
                        flush=True,
                    )
                    break

            time.sleep(.015)

    finally:
        release_mouse()

        print(
            "LEFT MOUSE RELEASED",
            flush=True,
        )

    if not reached:
        raise RuntimeError(
            f"Power not reached; "
            f"maximum estimate {best:.0%}. "
            f"No accuracy click."
        )


    # MANUAL ACCURACY HANDOFF V1
    print(
        "YOUR TURN: Click the white accuracy mark!",
        flush=True
    )
    print(
        "Bot will NOT make the final accuracy click.",
        flush=True
    )

    deadline = time.monotonic() + 12.0
    check_stop = (
        globals().get("stopping")
        or globals().get("stopped")
    )

    while time.monotonic() < deadline:
        if check_stop and check_stop():
            break
        time.sleep(0.05)

    return True


def main():
    configure()

    template = cv2.imread(
        str(SWING_PNG),
        cv2.IMREAD_GRAYSCALE,
    )

    if (
        template is None
        or not CALIBRATION.exists()
    ):
        print(
            "Missing swing_button.png or "
            "wgt_accuracy.json in bot folder."
        )
        return

    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        print(
            "Tesseract unavailable:",
            error,
        )
        return

    print(
        "WGT TIGERWOULD52 - ONE SUPERVISED SHOT"
    )

    print(
        "No game/browser launch. "
        "F9 emergency stop. "
        "One-shot limit: 1."
    )

    print(
        "Open WGT on Monitor 2 and activate it."
    )

    try:
        with mss.mss() as sct:
            hwnd = None
            active_since = None

            while True:
                if stopping():
                    return

                h = window()

                if (
                    h
                    and USER32.GetForegroundWindow()
                    == h
                ):
                    if hwnd != h:
                        hwnd = h
                        active_since = (
                            time.monotonic()
                        )

                    if (
                        time.monotonic()
                        - active_since
                        >= 4
                    ):
                        break
                else:
                    hwnd = None
                    active_since = None

                time.sleep(.25)

            rect = geometry(hwnd)

            print(
                "WGT active. Verifying player, "
                "hole, and shot..."
            )

            state = None
            previous = None
            consecutive = 0

            end = time.monotonic() + 25
            last_debug = 0

            while time.monotonic() < end:
                guard(hwnd)

                img = shot_frame(
                    sct, rect
                )

                if not player_ok(img):
                    time.sleep(.35)
                    continue

                hole = read_hole(img)
                stroke = read_shot(
                    img, hole
                )

                candidate = (
                    hole, stroke
                )

                if (
                    candidate[0] is not None
                    and candidate[1] is not None
                ):
                    if candidate == previous:
                        consecutive += 1
                    else:
                        consecutive = 1

                    if consecutive >= 2:
                        state = candidate
                        break

                else:
                    consecutive = 0
                    now = time.monotonic()

                    if now - last_debug > 4:
                        print(
                            f"HOLE/SHOT READ: "
                            f"{candidate}",
                            flush=True,
                        )

                        cv2.imwrite(
                            str(
                                DEBUG
                                / "shot_header_failed.png"
                            ),
                            crop(
                                img,
                                (
                                    .80, .015,
                                    .985, .155,
                                ),
                            ),
                        )

                        last_debug = now

                previous = candidate
                time.sleep(.35)

            if state is None:
                raise RuntimeError(
                    "Cannot confirm shot label. "
                    "No swing attempted. "
                    "Saved shot_header_failed.png."
                )

            print(
                "SHOT VERIFIED:",
                state,
            )

            for _ in range(MAX_ATTEMPTS):
                do_shot(
                    hwnd,
                    rect,
                    sct,
                    template,
                    state,
                )

                print(
                    "One-shot safety limit reached; "
                    "controller ending."
                )

    except (
        KeyboardInterrupt,
        pyautogui.FailSafeException,
    ):
        print(
            "EMERGENCY STOP"
        )

    except Exception as error:
        print(
            "STOPPED SAFELY:",
            error,
        )

    finally:
        release_mouse()

        print(
            "WGT CONTROLLER CLOSED"
        )


if __name__ == "__main__":
    main()