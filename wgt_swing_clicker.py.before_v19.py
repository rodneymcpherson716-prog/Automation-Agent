
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
    # FOCUSED BOGEY OCR V1
    _hole = read_hole(frame)
    _par = globals().get("PARS", {}).get(_hole)

    if _par is not None:
        _h, _w = frame.shape[:2]
        _hits = 0

        for _x1, _y1, _x2, _y2 in (
            (.879, .080, .915, .110),
            (.879, .083, .913, .108),
        ):
            _patch = frame[
                int(_h*_y1):int(_h*_y2),
                int(_w*_x1):int(_w*_x2),
            ]

            if _patch.size == 0:
                continue

            _gray = cv2.cvtColor(
                _patch, cv2.COLOR_BGR2GRAY
            )

            _large = cv2.resize(
                _gray, None, fx=5, fy=5,
                interpolation=cv2.INTER_CUBIC
            )

            _binary = cv2.threshold(
                _large, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            for _image in (_large, _binary):
                _text = pytesseract.image_to_string(
                    _image, config="--psm 7"
                ).upper()

                _clean = re.sub(
                    r"[^A-Z]", "", _text
                )

                if "FORBOGEY" in _clean:
                    _hits += 1
                    break

        if _hits == 2:
            print(
                f"SHOT LABEL: FOR BOGEY -> shot {_par + 1}",
                flush=True
            )
            return _par + 1

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
    """CLUB OCR V5: isolate the yardage digits."""
    from collections import Counter, defaultdict

    h, w = frame.shape[:2]
    votes = Counter()
    regions_seen = defaultdict(set)

    regions = (
        (.126, .177, .934, .982),
        (.129, .173, .938, .980),
        (.122, .182, .930, .985),
    )

    for index, (x1, x2, y1, y2) in enumerate(regions):
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

        binary = cv2.threshold(
            big, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        for image in (big, binary):
            for mode in (6, 11):
                raw = pytesseract.image_to_string(
                    image,
                    config=(
                        f"--psm {mode} "
                        "-c tessedit_char_whitelist=0123456789"
                    )
                )

                numbers = re.findall(
                    r"(?<!\d)\d{2,3}(?!\d)",
                    raw
                )

                if len(numbers) == 1:
                    value = int(numbers[0])

                    if 10 <= value <= 350:
                        votes[value] += 1
                        regions_seen[value].add(index)

    if votes:
        ranked = votes.most_common()
        best, count = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0

        if (
            count >= 3
            and len(regions_seen[best]) >= 2
            and count > second
        ):
            return best

    folder = ROOT / "wgt_debug"
    folder.mkdir(exist_ok=True)

    cv2.imwrite(
        str(folder / "club_read_failed.png"),
        frame[
            int(h*.90):,
            :int(w*.22)
        ]
    )

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

    playing_yards = condition_distance(yards)
    print(
        f"EXPERIMENTAL PLAYING DISTANCE: {playing_yards} yd",
        flush=True
    )

    # TEE MAX POWER V18
    if stroke == 1:
        ratio = 1.0
    else:
        ratio = playing_yards / club

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

                if power >= (.99 if stroke == 1 else required - .025):
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


    # CURSOR HANDOFF V13B
    # The power-drag mouse button was already released.
    if stopping():
        raise KeyboardInterrupt("F9 emergency stop")

    aim_x = meter["left"] + mark
    aim_y = meter["top"] + meter["height"] // 2

    pyautogui.moveTo(
        aim_x, aim_y, duration=.06
    )

    actual_x, actual_y = pyautogui.position()

    if (
        abs(actual_x - aim_x) > 12
        or abs(actual_y - aim_y) > 12
    ):
        raise RuntimeError(
            "Pointer positioning failed. "
            "Final click remains manual."
        )

    print(
        "MOUSE RELEASED AND POSITIONED AT ACCURACY MARK",
        flush=True
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


def smart_shot_review(frame, state):
    """Read shot conditions without making extra clicks."""
    hole, stroke = state
    hh, ww = frame.shape[:2]

    target_fn = globals().get('read_target')
    club_fn = (
        globals().get('read_club')
        or globals().get('read_club_yards')
    )

    target = target_fn(frame) if target_fn else None
    yards = (
        target[0]
        if isinstance(target, (list, tuple))
        else target
    )
    club = club_fn(frame) if club_fn else None

    # Read wind speed from the WGT header.
    area = frame[
        int(hh*.035):int(hh*.113),
        int(ww*.055):int(ww*.145)
    ]

    wind = None

    if area.size:
        gray = cv2.cvtColor(
            area, cv2.COLOR_BGR2GRAY
        )
        big = cv2.resize(
            gray, None, fx=5, fy=5
        )

        for mode in (6, 7, 11):
            raw = pytesseract.image_to_string(
                big, config=f'--psm {mode}'
            ).upper()

            match = re.search(
                r'(\d{1,2})\s*[-?]\s*(\d{1,2})',
                raw
            )

            if match:
                a, b = map(int, match.groups())

                if 0 <= a <= b <= 60:
                    wind = (a, b)
                    break

    # Look for possible sand-coloured areas.
    roi = frame[
        int(hh*.22):int(hh*.57),
        int(ww*.24):int(ww*.79)
    ]

    possible_bunkers = 0

    if roi.size:
        hsv = cv2.cvtColor(
            roi, cv2.COLOR_BGR2HSV
        )

        sand = cv2.inRange(
            hsv,
            (8, 9, 125),
            (36, 92, 245)
        )

        sand = cv2.morphologyEx(
            sand,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        _, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                sand, 8
            )
        )

        for x, y, w, h, pixels in stats[1:]:
            center = int(ww*.24) + x + w/2

            if (
                100 <= pixels
                and w >= 22
                and h >= 8
                and 1.4 <= w/max(h, 1) <= 12
                and ww*.32 <= center <= ww*.67
            ):
                possible_bunkers += 1

    if yards is None:
        swing = 'Target unreadable'
    elif yards <= 20:
        swing = 'CHIP/PITCH; check bunker clearance'
    elif yards <= 65:
        swing = 'PITCH; consider loft if needed'
    else:
        swing = 'FULL; check landing area'

    print(
        '\n=== SMART SHOT / BUNKER REVIEW ===',
        flush=True
    )

    print(
        f'Hole {hole}, shot {stroke}, '
        f'target {yards} yd, '
        f'current club {club} yd'
    )

    print(
        'Wind speed:',
        f'{wind[0]}-{wind[1]} MPH'
        if wind else 'not readable'
    )

    print('Suggested swing style:', swing)

    if club and yards is not None:
        print(
            f'Unadjusted power: '
            f'{100*yards/club:.0f}%'
        )

    print(
        'Possible sand-coloured areas:',
        possible_bunkers
    )

    print(
        'Wind direction, lie, elevation, '
        'and bunker clearance NOT VERIFIED'
    )

    print(
        'Club and aim unchanged. '
        'Accuracy click remains manual.'
    )

    print(
        '==================================\n',
        flush=True
    )

    # Stop a potentially risky short approach.
    if (
        yards is not None
        and 0 < yards <= 60
        and possible_bunkers
    ):
        raise RuntimeError(
            'Possible bunker near short approach. '
            'Safe aim and carry not verified. '
            'No swing attempted.'
        )



def auto_prepare_club_and_swing(hwnd, rect, sct, state):
    """Choose and verify club and swing type."""
    guard(hwnd)
    image = shot_frame(sct, rect)

    # Retain the existing condition and bunker review.
    if (
        "smart_shot_review" in globals()
        and SHOT_CONDITIONS["lie"] != "bunker"
    ):
        smart_shot_review(image, state)

    target = read_target(image)

    if isinstance(target, (tuple, list)):
        target = target[0]

    if target is None:
        target = read_top_yards(image)

    if target is None or not 5 <= target <= 600:
        raise RuntimeError(
            "Target yardage not verified. No swing."
        )

    target = condition_distance(target)

    if SHOT_CONDITIONS["lie"] == "bunker":
        desired = "FULL"
    elif target <= 20:
        desired = "CHIP"
    elif target <= 65:
        desired = "PITCH"
    else:
        desired = "FULL"

    # FULL DEFAULT V12B
    if SHOT_CONDITIONS.get("lie") == "green":
        raise RuntimeError(
            "Putting automation not ready. No swing attempted."
        )

    desired = "FULL"

    print(
        f"AUTO SETUP: {target} yards, {desired}",
        flush=True
    )

    h, w = image.shape[:2]

    def sample():
        guard(hwnd)
        return shot_frame(sct, rect)

    def menu_anchor(frame):
        """Find the WGT Swing Selection menu."""
        y0 = int(h * .57)

        roi = frame[
            y0:int(h * .975),
            :int(w * .28)
        ]

        gray = cv2.cvtColor(
            roi, cv2.COLOR_BGR2GRAY
        )

        large = cv2.resize(
            gray, None,
            fx=3, fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        data = pytesseract.image_to_data(
            large,
            config="--psm 11",
            output_type=pytesseract.Output.DICT
        )

        seen = {}

        for i, word in enumerate(data["text"]):
            clean = re.sub(
                "[^A-Z]", "", word.upper()
            )

            if clean in ("SWING", "SELECTION"):
                seen[clean] = (
                    data["left"][i] / 3,
                    y0 + data["top"][i] / 3,
                    data["width"][i] / 3
                )

        if (
            "SWING" not in seen
            or "SELECTION" not in seen
        ):
            return None

        x, y, width1 = seen["SWING"]
        sx, sy, width2 = seen["SELECTION"]

        if (
            abs(sy - y) > 14
            or abs(sx - (x + width1)) > 25
        ):
            return None

        scale = min(
            1.5,
            max(.65, (sx + width2 - x) / 121)
        )

        return x, y, scale

    def button_pos(anchor, swing):
        positions = {
            "FULL": (0, 0),
            "CHIP": (1, 0),
            "PUNCH": (0, 1),
            "FLOP": (1, 1),
            "PITCH": (0, 2),
            "PUTT": (1, 2)
        }

        column, row = positions[swing]
        x, y, scale = anchor

        return (
            int(
                x + (
                    46 if column == 0 else 164
                ) * scale
            ),
            int(y + (39 + 38 * row) * scale)
        )

    # FULL SWING MENU SKIP V14
    # Verify FULL without opening the selection menu.
    def full_on_hud():
        for _ in range(2):
            guard(hwnd)
            frame = sample()
            fh, fw = frame.shape[:2]

            region = frame[
                int(fh * .78):int(fh * .995),
                :int(fw * .32)
            ]

            if region.size == 0:
                continue

            gray = cv2.cvtColor(
                region, cv2.COLOR_BGR2GRAY
            )

            enlarged = cv2.resize(
                gray, None, fx=3, fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            _, white = cv2.threshold(
                enlarged, 160, 255,
                cv2.THRESH_BINARY
            )

            for picture in (enlarged, white):
                words = pytesseract.image_to_string(
                    picture, config="--psm 11"
                )

                if re.search(r"\bFULL\b", words, re.I):
                    return True

            time.sleep(.12)

        return False

    if not full_on_hud():
        raise RuntimeError(
            "FULL not verified in WGT HUD. "
            "No swing attempted."
        )

    print(
        "FULL verified; selection menu skipped.",
        flush=True
    )

    def club_read():
        # CLUB OCR RETRY V3
        for attempt in range(4):
            guard(hwnd)
            time.sleep(0.25)
            value = read_club(sample())

            if (
                isinstance(value, int)
                and 1 <= value <= 350
            ):
                print(
                    f"CLUB READING VERIFIED: {value} yd",
                    flush=True
                )
                return value

        return None

    start = club_read()

    if start is None:
        raise RuntimeError(
            "Selected club distance unreadable."
        )

    print(
        f"CURRENT CLUB DISTANCE: {start} yd",
        flush=True
    )

    def suitable(yards):
        # BUNKER LW FULL DEFAULT V10
        if SHOT_CONDITIONS["lie"] == "bunker":
            return 49 <= yards <= 51
        ratio = target / yards
        return .60 <= ratio <= .98

    if suitable(start):
        print(
            f"CLUB VERIFIED: {start} yd, "
            f"power {target/start:.0%}",
            flush=True
        )
        return

    # Cycle through clubs and check the yardage.
    wheel_x = (
        rect["left"] + int(w * .115)
    )

    wheel_y = (
        rect["top"] + int(h * .960)
    )

    options = []

    def wheel(amount):
        guard(hwnd)

        pyautogui.moveTo(
            wheel_x, wheel_y,
            duration=.08
        )

        pyautogui.scroll(amount)
        time.sleep(.35)

    for direction in (-1, 1):
        steps = 0
        previous = start

        for step in range(1, 13):
            wheel(direction)
            steps += 1

            observed = club_read()

            if observed is None:
                print(
                    "CLUB READING UNCERTAIN - checking next",
                    flush=True
                )
                continue

            if observed == previous:
                print(
                    "Club yardage unchanged",
                    flush=True
                )
                break

            previous = observed

            print(
                f"CLUB CANDIDATE: {observed} yd",
                flush=True
            )

            if suitable(observed):
                score = abs(
                    target / observed - .82
                )

                options.append(
                    (
                        score,
                        direction,
                        step,
                        observed
                    )
                )
                # CLUB SELECTION EARLY RETURN V2
                verified = club_read()

                if verified != observed:
                    raise RuntimeError(
                        "Selected club not stable. No swing."
                    )

                print(
                    f"CLUB SELECTED AND VERIFIED: {verified} yd",
                    flush=True
                )

                print(
                    f"BASELINE POWER: {target/verified:.0%}",
                    flush=True
                )

                return

        if steps:
            wheel(-direction * steps)

        restored = club_read()

        if restored != start:
            raise RuntimeError(
                "Could not restore original club. "
                "No swing attempted."
            )

    if not options:
        raise RuntimeError(
            "No verified suitable club. "
            "No swing attempted."
        )

    _, direction, count, chosen = min(
        options
    )

    wheel(direction * count)

    verified = club_read()

    if verified != chosen:
        raise RuntimeError(
            "Selected club yardage not confirmed."
        )

    print(
        f"CLUB VERIFIED: {chosen} yd",
        flush=True
    )

    print(
        f"ESTIMATED POWER: "
        f"{target/chosen:.0%}",
        flush=True
    )

    print(
        "Wind-direction and bunker clearance "
        "are not yet fully automated.",
        flush=True
    )


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
                        >= 1.0
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
                auto_prepare_club_and_swing(hwnd, rect, sct, state)
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


# PLAYER SHOT AND LOGGING REPAIR V6
_old_player_ok = player_ok
_old_read_shot = read_shot
_old_main = main


def player_ok(frame):
    if _old_player_ok(frame):
        return True

    h, w = frame.shape[:2]

    for x1, y1, x2, y2 in (
        (.795, .030, .974, .126),
        (.810, .048, .955, .112),
        (.775, .035, .988, .150),
    ):
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
            gray, None, fx=4, fy=4
        )

        for mode in (6, 11):
            raw = pytesseract.image_to_string(
                big, config=f'--psm {mode}'
            )

            clean = re.sub(
                r'[^a-z0-9]', '', raw.lower()
            )

            if 'tigerwould52' in clean:
                print(
                    'PLAYER VERIFIED BY FALLBACK OCR',
                    flush=True
                )
                return True

    if not getattr(player_ok, '_saved', False):
        cv2.imwrite(
            str(DEBUG / 'player_read_failed.png'),
            frame[
                int(h*.02):int(h*.16),
                int(w*.77):
            ]
        )

        player_ok._saved = True

        print(
            'Player unreadable: diagnostic saved',
            flush=True
        )

    return False


def read_shot(frame, hole=None):
    result = _old_read_shot(frame, hole)

    if result is not None:
        return result

    if hole is None:
        hole = read_hole(frame)

    par = PARS.get(hole)

    if par is None:
        return None

    h, w = frame.shape[:2]

    regions = (
        (.866, .070, .943, .118),
        (.870, .077, .938, .115),
        (.856, .065, .957, .129),
    )

    labels = (
        ('TRIPLEBOGEY', 3),
        ('DOUBLEBOGEY', 2),
        ('BOGEY', 1),
        ('BIRDIE', -1),
        ('EAGLE', -2),
        ('PAR', 0),
    )

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

        large = cv2.resize(
            gray, None, fx=5, fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        binary = cv2.threshold(
            large, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        for image in (large, binary):
            for mode in (7, 11):
                raw = pytesseract.image_to_string(
                    image, config=f'--psm {mode}'
                )

                clean = re.sub(
                    r'[^A-Z]', '',
                    raw.upper().replace('0', 'O')
                )

                if 'FOR' not in clean:
                    continue

                tail = clean.split('FOR', 1)[1]

                for label, offset in labels:
                    shot = par + offset

                    if (
                        label in tail
                        and 1 <= shot <= 30
                    ):
                        print(
                            f'SHOT LABEL: FOR {label} '
                            f'-> shot {shot}',
                            flush=True
                        )
                        return shot

    return None


def main():
    import sys

    class Tee:
        def __init__(self, screen, log):
            self.screen = screen
            self.log = log

        def write(self, message):
            self.screen.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.screen.flush()
            self.log.flush()

    DEBUG.mkdir(exist_ok=True)

    path = DEBUG / 'latest_session.txt'

    with path.open('w', encoding='utf-8') as log:
        original = sys.stdout
        sys.stdout = Tee(original, log)

        try:
            return _old_main()
        finally:
            sys.stdout = original



# WIND AND LIE GATE V7
SHOT_CONDITIONS = {
    "wind": 0,
    "direction": "unknown",
    "lie": "unknown"
}


def condition_distance(yards):
    if yards is None:
        return None

    speed = SHOT_CONDITIONS["wind"]
    direction = SHOT_CONDITIONS["direction"]
    lie = SHOT_CONDITIONS["lie"]

    if direction not in ("head", "tail", "calm"):
        raise RuntimeError(
            "Wind direction unverified. No swing."
        )

    # Experimental estimates; not calibrated to WGT.
    rate = .003 if yards <= 65 else .005

    wind = (
        rate * speed if direction == "head"
        else -rate * speed if direction == "tail"
        else 0
    )

    sand = .25 if lie == "bunker" else 0

    result = round(
        yards * (1 + wind + sand), 1
    )

    print(
        f"CONDITIONS: {yards} yd -> {result} yd "
        f"({direction} {speed} mph, {lie})",
        flush=True
    )

    return result


def condition_prompt():
    print()
    print("WIND AND LIE CHECK - EVERY SHOT")

    direction = input(
        "Wind [head/tail/left/right/calm/unknown]: "
    ).strip().lower()

    if direction not in (
        "head", "tail", "left",
        "right", "calm", "unknown"
    ):
        print("STOPPED: Invalid wind direction.")
        return False

    try:
        speed = float(
            input("Wind speed MPH: ").strip()
        )
    except ValueError:
        print("STOPPED: Invalid wind speed.")
        return False

    if not 0 <= speed <= 60:
        print("STOPPED: Invalid wind speed.")
        return False

    lie = input(
        "Lie [fairway/rough/bunker/green/unknown]: "
    ).strip().lower()

    if lie not in (
        "fairway", "rough", "bunker",
        "green", "unknown"
    ):
        print("STOPPED: Invalid lie.")
        return False

    # PUTTING GATE V16A
    # Allow green for read-only putting analysis.
    if (
        lie == "unknown"
        or (
            lie != "green"
            and direction in ("left", "right", "unknown")
        )
    ):
        print(
            "STOPPED: Crosswind aim, putting, or "
            "lie not yet verified."
        )
        return False

    if lie == "bunker":
        lip = input(
            "Bunker lip [low/high/unknown]: "
        ).strip().lower()

        if lip != "low":
            print(
                "STOPPED: Bunker lip clearance "
                "not established."
            )
            return False

        print(
            "Bunker default: LW / FULL / 50 YDS. "
            "Club yardage will be checked."
        )

    SHOT_CONDITIONS.update(
        wind=speed,
        direction=direction,
        lie=lie
    )

    print(
        "CONDITIONS ACCEPTED - ONE SUPERVISED SHOT"
    )
    return True


_main_before_conditions = main


def main():
    if not condition_prompt():
        return

    return _main_before_conditions()

# TURN CLOCK SAFETY V8
# Internal safety budget; does NOT read WGT's HUD clock.
_CLOCK_SECONDS = 0
_CLOCK_STARTED = None
_prior_geometry = geometry
_prior_guard = guard
_prior_do_shot = do_shot
_prior_main_clock = main


def geometry(hwnd):
    global _CLOCK_STARTED
    rect = _prior_geometry(hwnd)
    if _CLOCK_SECONDS and _CLOCK_STARTED is None:
        _CLOCK_STARTED = time.monotonic()
        print(
            f"LOCAL {_CLOCK_SECONDS}-SECOND BUDGET STARTED",
            flush=True
        )
    return rect


def _clock_remaining():
    if not _CLOCK_SECONDS or _CLOCK_STARTED is None:
        return None
    return _CLOCK_SECONDS - (
        time.monotonic() - _CLOCK_STARTED
    )


def guard(hwnd):
    _prior_guard(hwnd)
    remaining = _clock_remaining()
    if remaining is not None and remaining < 8:
        raise RuntimeError(
            "Local shot deadline reached. "
            "No further bot clicks."
        )


def do_shot(hwnd, rect, sct, template, state):
    remaining = _clock_remaining()
    if remaining is not None and remaining < 18:
        raise RuntimeError(
            f"Insufficient time: {remaining:.0f} seconds. "
            "No new swing attempted."
        )
    return _prior_do_shot(
        hwnd, rect, sct, template, state
    )


def main():
    global _CLOCK_SECONDS, _CLOCK_STARTED
    _CLOCK_STARTED = None

    print(
        "SHOT CLOCK MODE (0=solo, 30, 60, 90):",
        flush=True
    )

    raw = input().strip() or "0"

    if raw not in ("0", "30", "60", "90"):
        print("Invalid clock setting. No game actions.")
        return

    _CLOCK_SECONDS = int(raw)

    if _CLOCK_SECONDS:
        print(
            "Internal timer only. Actual WGT countdown "
            "and active turn are not verified.",
            flush=True
        )

    return _prior_main_clock()



# FAST SHOT ENTRY V9
_previous_fast_read_shot = read_shot
_previous_fast_main = main
_FAST_SHOT_EXPECTED = None


def read_shot(frame, hole=None):
    if _FAST_SHOT_EXPECTED is None:
        return _previous_fast_read_shot(frame, hole)

    expected_hole, expected_shot = _FAST_SHOT_EXPECTED

    if hole == expected_hole:
        return expected_shot

    if hole is not None:
        raise RuntimeError(
            f"Hole mismatch: expected {expected_hole}, "
            f"WGT shows {hole}. No swing."
        )

    return None


def main():
    global _FAST_SHOT_EXPECTED

    raw = input(
        "FAST MODE - hole and shot/score "
        "(e.g. 5 5 or 5 double bogey), "
        "Enter for OCR: "
    ).strip()

    if raw:
        pieces = raw.split(maxsplit=1)

        if len(pieces) != 2 or not pieces[0].isdigit():
            print("Invalid entry; bot not started.")
            return

        hole = int(pieces[0])

        if hole not in PARS:
            print("Hole not supported; bot not started.")
            return

        label = re.sub(
            r"[^A-Z0-9]", "", pieces[1].upper()
        )

        if label.startswith("FOR"):
            label = label[3:]

        scores = {
            "EAGLE": -2,
            "BIRDIE": -1,
            "PAR": 0,
            "BOGEY": 1,
            "DOUBLEBOGEY": 2,
            "TRIPLEBOGEY": 3,
        }

        if label.isdigit():
            shot = int(label)
        elif label in scores:
            shot = PARS[hole] + scores[label]
        else:
            print("Shot/score not recognized. No changes.")
            return

        if not 1 <= shot <= 20:
            print("Invalid shot number; bot not started.")
            return

        _FAST_SHOT_EXPECTED = (hole, shot)

        print(
            f"FAST MODE: Hole {hole}, shot {shot}. "
            "WGT hole verification enabled."
        )

    return _previous_fast_main()


# CONFIRMED TARGET FALLBACK V15
_original_target_reader_v15 = read_target
_previous_main_v15 = main
_target_backup_v15 = None
_target_backup_logged_v15 = False


def read_target(frame):
    global _target_backup_logged_v15

    result = _original_target_reader_v15(frame)

    if _target_backup_v15 is None:
        return result

    if isinstance(result, (tuple, list)):
        value = result[0] if result else None
    else:
        value = result

    if value is not None:
        if abs(float(value) - _target_backup_v15) > 3:
            raise RuntimeError(
                f"Target mismatch: WGT {value} yd versus "
                f"entered {_target_backup_v15:g} yd. No swing."
            )
        return result

    if not _target_backup_logged_v15:
        print(
            f"WGT target unreadable; using confirmed "
            f"{_target_backup_v15:g} YARDS.",
            flush=True
        )
        _target_backup_logged_v15 = True

    return _target_backup_v15


def main():
    global _target_backup_v15, _target_backup_logged_v15

    _target_backup_v15 = None
    _target_backup_logged_v15 = False

    raw = input(
        "TARGET BACKUP in YARDS (Enter for automatic): "
    ).strip()

    if raw:
        try:
            yards = float(raw)
        except ValueError:
            print("Invalid distance. Bot not started.")
            return

        if not 5 <= yards <= 600:
            print("Distance outside 5-600 yards. Bot not started.")
            return

        answer = input(
            f"Confirm WGT displays {yards:g} YARDS, "
            "not feet [yes/no]: "
        ).strip().lower()

        if answer != "yes":
            print("Target not confirmed. Bot not started.")
            return

        _target_backup_v15 = yards

    return _previous_main_v15()


# PUTTING HUD ROUTER V16
_previous_prepare_v16 = auto_prepare_club_and_swing


def auto_prepare_club_and_swing(hwnd, rect, sct, state):
    guard(hwnd)
    frame = shot_frame(sct, rect)
    h, w = frame.shape[:2]

    area = frame[
        int(h * .92):int(h * .995),
        :int(w * .21)
    ]

    if area.size == 0:
        return _previous_prepare_v16(hwnd, rect, sct, state)

    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(
        gray, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    text = pytesseract.image_to_string(
        big, config="--psm 11"
    )

    if not re.search(r"\bPUTT\b", text, re.I):
        return _previous_prepare_v16(hwnd, rect, sct, state)

    print(
        "PUTTING MODE RECOGNIZED - DISTANCE IN FEET",
        flush=True
    )

    if globals().get("_target_backup_v15") is not None:
        raise RuntimeError(
            "Putting distance is in feet, not yards. "
            "Restart without a yardage backup."
        )

    reader = ROOT / "wgt_putt_read.py"

    if not reader.is_file():
        raise RuntimeError(
            "Putting reader missing. No swing attempted."
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wgt_putt_reader_v16", reader
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frames = [frame]

    for i in range(19):
        guard(hwnd)
        time.sleep(.10)
        frames.append(shot_frame(sct, rect))

    feet, inches = module.read_distance_and_elevation(
        frames[0]
    )

    print("=== PUTTING ANALYSIS ===", flush=True)

    if feet is not None:
        print(f"Distance: {feet:.1f} FT", flush=True)
    else:
        print("Distance: unreadable", flush=True)

    if inches is not None:
        print(
            f"Elevation displayed: {inches:.1f} IN "
            "(check uphill/downhill arrow)",
            flush=True
        )
    else:
        print("Elevation: unreadable", flush=True)

    print(
        "Break:", module.analyze_break(frames),
        flush=True
    )

    raise RuntimeError(
        "Putt recognized and analyzed. "
        "Automatic putting aim and power are not "
        "calibrated yet. No swing attempted."
    )

if __name__ == "__main__":
    main()