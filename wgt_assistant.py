import os
import re
import time
import queue
import threading
import tkinter as tk
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


# ============================================================
# WGT ASSISTANT — SINGLE FILE
# ============================================================
# What this file DOES:
#   - Finds the real "WGT Golf" window
#   - Handles secondary monitors / negative coordinates
#   - Captures WGT with MSS
#   - Reads player/mode/distance/elevation/wind/green speed/club
#   - Calculates putting and full-shot recommendations
#   - Shows a live overlay
#   - Saves self-diagnostic files when OCR is weak
#   - Recalculates only when the shot state changes
#
# What this file DOES NOT do:
#   - It does not click, hold, release, swing, or putt for you.
# ============================================================


# -----------------------------
# CONFIG
# -----------------------------

WINDOW_TITLE = "WGT Golf"

BASE_WIDTH = 1288
BASE_HEIGHT = 728

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT

SCAN_DELAY_SECONDS = 0.45
OVERLAY_REFRESH_MS = 200
DEBUG_INTERVAL_SECONDS = 10.0

DEBUG_DIR = Path("wgt_debug")

pyautogui.FAILSAFE = True


# -----------------------------
# BASE HUD REGIONS
# -----------------------------
# These are broad regions for the 1288 x 728 WGT window discovered
# in your diagnostic. They are scaled automatically if the window
# size changes slightly.

HUD_REGIONS = {
    "top_left": (0, 18, 420, 165),          # wind / hole info
    "center_top": (440, 55, 430, 210),      # pin distance / elevation
    "top_right": (900, 10, 370, 150),       # player/profile area
    "bottom_left": (0, 545, 520, 175),      # club / carry / putter info
}


# -----------------------------
# WINDOW + CAPTURE
# -----------------------------

def find_wgt_window():
    matches = []

    try:
        windows = pyautogui.getAllWindows()
    except Exception:
        return None

    for win in windows:
        try:
            title = (win.title or "").strip()

            if title.lower() != WINDOW_TITLE.lower():
                continue

            if int(win.width) < 700 or int(win.height) < 450:
                continue

            matches.append(win)

        except Exception:
            pass

    if not matches:
        return None

    return max(
        matches,
        key=lambda w: int(w.width) * int(w.height),
    )


def capture_wgt(window, sct):
    monitor = {
        "left": int(window.left),
        "top": int(window.top),
        "width": int(window.width),
        "height": int(window.height),
    }

    raw = np.array(
        sct.grab(monitor)
    )

    return cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2BGR,
    )


def scale_region(window_rect, base_region):
    bx, by, bw, bh = base_region

    sx = window_rect["width"] / BASE_WIDTH
    sy = window_rect["height"] / BASE_HEIGHT

    x = int(bx * sx)
    y = int(by * sy)
    w = int(bw * sx)
    h = int(bh * sy)

    return x, y, w, h


def crop_region(frame, window_rect, base_region):
    x, y, w, h = scale_region(
        window_rect,
        base_region,
    )

    fh, fw = frame.shape[:2]

    x1 = max(0, min(fw, x))
    y1 = max(0, min(fh, y))
    x2 = max(0, min(fw, x + w))
    y2 = max(0, min(fh, y + h))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[
        y1:y2,
        x1:x2,
    ]


# -----------------------------
# OCR
# -----------------------------

def preprocess_variants(frame):
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    _, inverse = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return [
        gray,
        binary,
        inverse,
    ]


def ocr_region(frame):
    texts = []

    for variant in preprocess_variants(frame):
        for psm in (6, 7, 11):
            try:
                text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {psm}",
                ).strip()
            except Exception:
                text = ""

            if text:
                texts.append(text)

    # Deduplicate while preserving order.
    seen = set()
    unique = []

    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)

    return unique


# -----------------------------
# PARSING HELPERS
# -----------------------------

def vote(values):
    if not values:
        return None

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return max(
        counts,
        key=counts.get,
    )


def parse_distance(texts, mode):
    yd_values = []
    ft_values = []

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*YDS?",
            upper,
        ):
            value = float(match.group(1))

            if 1 <= value <= 999:
                yd_values.append(value)

        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*FT",
            upper,
        ):
            value = float(match.group(1))

            if 0.1 <= value <= 500:
                ft_values.append(value)

    if mode == "PUTTING":
        if ft_values:
            return vote(ft_values), "FT"

        if yd_values:
            return vote(yd_values) * 3.0, "FT"

    else:
        if yd_values:
            return vote(yd_values), "YD"

        if ft_values:
            return vote(ft_values) / 3.0, "YD"

    return None, None


def parse_elevation(texts, mode):
    candidates = []

    for text in texts:
        upper = text.upper()

        matches = list(
            re.finditer(
                r"([+-]?\d+(?:\.\d+)?)\s*FT",
                upper,
            )
        )

        for match in matches:
            value = float(
                match.group(1)
            )

            if -300 <= value <= 300:
                candidates.append(
                    value
                )

    if not candidates:
        return 0.0

    # Elevation is usually the smaller absolute FT value when a
    # putting distance in feet is also visible.
    return min(
        candidates,
        key=lambda value: abs(value),
    )


def parse_wind(texts):
    range_votes = []
    single_votes = []

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*MPH",
            upper,
        ):
            a = int(
                match.group(1)
            )
            b = int(
                match.group(2)
            )

            if 0 <= a <= 50 and 0 <= b <= 50:
                range_votes.append(
                    (
                        min(a, b),
                        max(a, b),
                    )
                )

        for match in re.finditer(
            r"(\d{1,2})\s*MPH",
            upper,
        ):
            value = int(
                match.group(1)
            )

            if 0 <= value <= 50:
                single_votes.append(
                    (
                        value,
                        value,
                    )
                )

    if range_votes:
        return vote(
            range_votes
        )

    if single_votes:
        return vote(
            single_votes
        )

    return None


def parse_green_speed(texts):
    joined = "\n".join(
        texts
    ).upper()

    # Numeric value if WGT displays one.
    for pattern in (
        r"GREEN[^\d]{0,12}(\d+(?:\.\d+)?)",
        r"STIMP[^\d]{0,12}(\d+(?:\.\d+)?)",
    ):
        match = re.search(
            pattern,
            joined,
        )

        if match:
            value = float(
                match.group(1)
            )

            if 5.0 <= value <= 15.0:
                return value, f"{value:.1f}"

    if "VERY FAST" in joined:
        return 11.5, "VERY FAST"

    if "FAST" in joined:
        return 10.0, "FAST"

    if "SLOW" in joined:
        return 7.0, "SLOW"

    return 9.0, "NORMAL"


def detect_mode(texts):
    joined = "\n".join(
        texts
    ).upper()

    if (
        "PUTTER" in joined
        or "PUTT" in joined
    ):
        return "PUTTING"

    return "FULL_SHOT"


def parse_club(texts):
    joined = "\n".join(
        texts
    ).upper()

    clubs = (
        "PUTTER",
        "DRIVER",
        "3 WOOD",
        "5 WOOD",
        "7 WOOD",
        "3 HYBRID",
        "4 HYBRID",
        "5 HYBRID",
        "LOB WEDGE",
        "SAND WEDGE",
        "GAP WEDGE",
        "PITCHING WEDGE",
        "9 IRON",
        "8 IRON",
        "7 IRON",
        "6 IRON",
        "5 IRON",
        "4 IRON",
        "3 IRON",
    )

    for club in clubs:
        if club in joined:
            return club.title()

    return None


def parse_carry(texts):
    values = []

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*YDS?",
            upper,
        ):
            value = float(
                match.group(1)
            )

            if 20 <= value <= 450:
                values.append(
                    value
                )

    return vote(
        values
    )


def parse_player_name(texts):
    joined = "\n".join(
        texts
    )

    candidates = re.findall(
        r"\b[A-Za-z][A-Za-z0-9_]{2,23}\b",
        joined,
    )

    reject = {
        "lvl",
        "level",
        "shot",
        "score",
        "hole",
        "tee",
        "green",
        "speed",
        "club",
        "putter",
        "driver",
        "iron",
        "wedge",
        "yards",
        "feet",
    }

    for candidate in candidates:
        if candidate.lower() not in reject:
            return candidate

    return None


# -----------------------------
# CONFIDENCE + DIAGNOSTICS
# -----------------------------

def measurement_confidence(data):
    score = 0
    max_score = 6

    if data["mode"]:
        score += 1

    if data["distance"] is not None:
        score += 1

    if data["elevation"] is not None:
        score += 1

    if data["mode"] == "PUTTING":
        if data["green_speed"] is not None:
            score += 1
    else:
        if data["wind"] is not None:
            score += 1

    if data["club"]:
        score += 1

    if data["player"]:
        score += 1

    return round(
        score / max_score,
        2,
    )


def build_diagnostics(data, window_rect):
    diagnostics = []

    if (
        window_rect["width"] != BASE_WIDTH
        or window_rect["height"] != BASE_HEIGHT
    ):
        diagnostics.append(
            f"Resolution scaled from {BASE_WIDTH}x{BASE_HEIGHT} "
            f"to {window_rect['width']}x{window_rect['height']}"
        )

    if data["distance"] is None:
        diagnostics.append(
            "Distance unreadable — waiting for clear HUD."
        )

    if data["mode"] == "FULL_SHOT" and data["wind"] is None:
        diagnostics.append(
            "Wind unreadable — full-shot recommendation is partial."
        )

    if data["mode"] == "PUTTING" and data["green_speed"] is None:
        diagnostics.append(
            "Green speed unreadable — using no recommendation."
        )

    if data["club"] is None:
        diagnostics.append(
            "Club name unreadable."
        )

    if not diagnostics:
        diagnostics.append(
            "HUD read OK."
        )

    return diagnostics


# -----------------------------
# CALCULATIONS
# -----------------------------

def calculate_putt(data):
    distance_ft = data["distance"]

    if distance_ft is None:
        return None

    elevation_ft = (
        data["elevation"]
        if data["elevation"] is not None
        else 0.0
    )

    green_speed = data["green_speed"]

    if green_speed is None:
        return None

    playing_distance = float(
        distance_ft
    )

    # Starting model from the user's earlier formula.
    playing_distance += (
        elevation_ft * 1.8
    )

    if green_speed >= 11.0:
        playing_distance *= 0.88

    elif green_speed >= 10.0:
        playing_distance *= 0.92

    elif green_speed <= 7.0:
        playing_distance *= 1.05

    playing_distance = max(
        0.1,
        playing_distance,
    )

    return {
        "playing_distance": round(
            playing_distance,
            1,
        ),
        "unit": "FT",
        "summary": (
            f"PLAY PUTT {playing_distance:.1f} FT"
        ),
    }


def calculate_full_shot(data):
    distance_yd = data["distance"]

    if distance_yd is None:
        return None

    elevation_ft = (
        data["elevation"]
        if data["elevation"] is not None
        else 0.0
    )

    wind = data["wind"]

    adjusted = float(
        distance_yd
    )

    # Basic elevation compensation.
    adjusted += (
        elevation_ft / 3.0
    )

    wind_mid = None

    if wind is not None:
        wind_mid = (
            wind[0] + wind[1]
        ) / 2.0

    carry = data["carry"]

    suggested_power = None

    if carry is not None and carry > 0:
        suggested_power = (
            adjusted / carry
        ) * 100.0

        suggested_power = max(
            10.0,
            min(
                100.0,
                suggested_power,
            ),
        )

    summary = (
        f"PLAY SHOT {adjusted:.1f} YD"
    )

    if suggested_power is not None:
        summary += (
            f" | {suggested_power:.1f}%"
        )

    if wind_mid is not None:
        summary += (
            f" | WIND {wind_mid:.1f} MPH"
        )

    return {
        "playing_distance": round(
            adjusted,
            1,
        ),
        "unit": "YD",
        "suggested_power": (
            round(
                suggested_power,
                1,
            )
            if suggested_power is not None
            else None
        ),
        "wind_mid": (
            round(
                wind_mid,
                1,
            )
            if wind_mid is not None
            else None
        ),
        "summary": summary,
    }


def calculate(data):
    if data["mode"] == "PUTTING":
        return calculate_putt(
            data
        )

    return calculate_full_shot(
        data
    )


# -----------------------------
# HUD READER
# -----------------------------

def read_hud(frame, window_rect):
    regions = {}

    for name, base_region in HUD_REGIONS.items():
        regions[name] = crop_region(
            frame,
            window_rect,
            base_region,
        )

    texts = {
        name: ocr_region(
            region
        )
        for name, region in regions.items()
    }

    all_text = []

    for values in texts.values():
        all_text.extend(
            values
        )

    mode = detect_mode(
        texts["bottom_left"]
        + texts["center_top"]
    )

    distance, distance_unit = parse_distance(
        texts["center_top"],
        mode,
    )

    elevation = parse_elevation(
        texts["center_top"],
        mode,
    )

    wind = parse_wind(
        texts["top_left"]
    )

    green_speed, green_label = parse_green_speed(
        texts["top_left"]
        + texts["bottom_left"]
    )

    club = parse_club(
        texts["bottom_left"]
    )

    carry = parse_carry(
        texts["bottom_left"]
    )

    player = parse_player_name(
        texts["top_right"]
    )

    data = {
        "player": player,
        "mode": mode,
        "distance": distance,
        "distance_unit": distance_unit,
        "elevation": elevation,
        "wind": wind,
        "green_speed": green_speed,
        "green_speed_label": green_label,
        "club": club,
        "carry": carry,
        "texts": texts,
    }

    data["confidence"] = measurement_confidence(
        data
    )

    data["diagnostics"] = build_diagnostics(
        data,
        window_rect,
    )

    data["calculation"] = calculate(
        data
    )

    return data, regions


# -----------------------------
# DEBUG
# -----------------------------

def save_debug(frame, regions, data):
    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            DEBUG_DIR
            / "latest_wgt_capture.png"
        ),
        frame,
    )

    for name, region in regions.items():
        if (
            region is not None
            and region.size
        ):
            cv2.imwrite(
                str(
                    DEBUG_DIR
                    / f"{name}.png"
                ),
                region,
            )

    report = [
        f"player={data['player']}",
        f"mode={data['mode']}",
        f"distance={data['distance']}",
        f"distance_unit={data['distance_unit']}",
        f"elevation={data['elevation']}",
        f"wind={data['wind']}",
        f"green_speed={data['green_speed']}",
        f"green_speed_label={data['green_speed_label']}",
        f"club={data['club']}",
        f"carry={data['carry']}",
        f"confidence={data['confidence']}",
        f"calculation={data['calculation']}",
        "",
        "DIAGNOSTICS:",
        *data["diagnostics"],
        "",
        "RAW OCR:",
        repr(
            data["texts"]
        ),
    ]

    (
        DEBUG_DIR
        / "latest_read.txt"
    ).write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )


# -----------------------------
# BACKGROUND READER
# -----------------------------

class ReaderThread(threading.Thread):
    def __init__(
        self,
        result_queue,
        stop_event,
    ):
        super().__init__(
            daemon=True
        )

        self.result_queue = result_queue
        self.stop_event = stop_event
        self.last_signature = None
        self.last_debug = 0.0

    def run(self):
        try:
            with mss.MSS() as sct:
                while not self.stop_event.is_set():
                    window = find_wgt_window()

                    if window is None:
                        self.result_queue.put(
                            {
                                "status": "WAITING_FOR_WGT",
                                "message": "WGT Golf window not found.",
                            }
                        )

                        time.sleep(
                            1.0
                        )
                        continue

                    window_rect = {
                        "left": int(
                            window.left
                        ),
                        "top": int(
                            window.top
                        ),
                        "width": int(
                            window.width
                        ),
                        "height": int(
                            window.height
                        ),
                    }

                    try:
                        frame = capture_wgt(
                            window,
                            sct,
                        )
                    except Exception as exc:
                        self.result_queue.put(
                            {
                                "status": "CAPTURE_ERROR",
                                "message": str(
                                    exc
                                ),
                            }
                        )

                        time.sleep(
                            1.0
                        )
                        continue

                    data, regions = read_hud(
                        frame,
                        window_rect,
                    )

                    signature = (
                        data["mode"],
                        data["distance"],
                        data["elevation"],
                        data["wind"],
                        data["green_speed"],
                        data["club"],
                        data["carry"],
                    )

                    shot_changed = (
                        signature
                        != self.last_signature
                    )

                    self.last_signature = signature

                    data["shot_changed"] = shot_changed
                    data["window_rect"] = window_rect

                    self.result_queue.put(
                        {
                            "status": "LIVE",
                            "data": data,
                        }
                    )

                    now = time.time()

                    if (
                        data["confidence"] < 0.67
                        and now - self.last_debug
                        >= DEBUG_INTERVAL_SECONDS
                    ):
                        save_debug(
                            frame,
                            regions,
                            data,
                        )

                        self.last_debug = now

                    time.sleep(
                        SCAN_DELAY_SECONDS
                    )

        except Exception as exc:
            self.result_queue.put(
                {
                    "status": "THREAD_ERROR",
                    "message": str(
                        exc
                    ),
                }
            )


# -----------------------------
# OVERLAY
# -----------------------------

class WGTOverlay:
    def __init__(self):
        self.root = tk.Tk()

        self.root.title(
            "WGT Assistant"
        )

        self.root.attributes(
            "-topmost",
            True,
        )

        self.root.configure(
            bg="#121212"
        )

        self.root.resizable(
            False,
            False,
        )

        self.root.geometry(
            "460x470+40+40"
        )

        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.reader = ReaderThread(
            self.result_queue,
            self.stop_event,
        )

        self.reader.start()

        self.create_widgets()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

        self.root.after(
            OVERLAY_REFRESH_MS,
            self.process_queue,
        )

    def make_label(
        self,
        text,
        size=11,
        bold=False,
        background="#121212",
    ):
        label = tk.Label(
            self.root,
            text=text,
            font=(
                "Segoe UI",
                size,
                "bold"
                if bold
                else "normal",
            ),
            fg="white",
            bg=background,
            anchor="w",
            justify="left",
        )

        label.pack(
            fill="x",
            padx=16,
            pady=2,
        )

        return label

    def create_widgets(self):
        self.make_label(
            "WGT ASSISTANT",
            18,
            True,
        )

        self.player_label = self.make_label(
            "Player: waiting..."
        )

        self.mode_label = self.make_label(
            "Mode: waiting...",
            13,
            True,
        )

        self.distance_label = self.make_label(
            "Distance: --"
        )

        self.elevation_label = self.make_label(
            "Elevation: --"
        )

        self.wind_label = self.make_label(
            "Wind: --"
        )

        self.green_label = self.make_label(
            "Green speed: --"
        )

        self.club_label = self.make_label(
            "Club: --"
        )

        self.confidence_label = self.make_label(
            "Read confidence: --"
        )

        self.calc_label = tk.Label(
            self.root,
            text=(
                "CALCULATION: WAITING..."
            ),
            font=(
                "Segoe UI",
                16,
                "bold",
            ),
            fg="white",
            bg="#292929",
            anchor="center",
            justify="center",
            wraplength=410,
            padx=12,
            pady=14,
        )

        self.calc_label.pack(
            fill="x",
            padx=16,
            pady=(12, 8),
        )

        self.diag_label = tk.Label(
            self.root,
            text=(
                "Starting reader..."
            ),
            font=(
                "Segoe UI",
                9,
            ),
            fg="#d0d0d0",
            bg="#121212",
            anchor="w",
            justify="left",
            wraplength=420,
        )

        self.diag_label.pack(
            fill="x",
            padx=16,
            pady=(4, 8),
        )

        self.stop_button = tk.Button(
            self.root,
            text="Stop Assistant",
            command=self.close,
        )

        self.stop_button.pack(
            pady=(4, 12),
        )

    def move_near_wgt(
        self,
        window_rect,
    ):
        try:
            x = (
                window_rect["left"]
                + window_rect["width"]
                - 490
            )

            y = (
                window_rect["top"]
                + 80
            )

            self.root.geometry(
                f"460x470+{x}+{y}"
            )

        except Exception:
            pass

    def process_queue(self):
        try:
            while True:
                result = (
                    self.result_queue
                    .get_nowait()
                )

                self.apply_result(
                    result
                )

        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(
                OVERLAY_REFRESH_MS,
                self.process_queue,
            )

    def apply_result(
        self,
        result,
    ):
        status = result.get(
            "status"
        )

        if status != "LIVE":
            self.diag_label.config(
                text=result.get(
                    "message",
                    status,
                )
            )

            return

        data = result["data"]

        self.move_near_wgt(
            data["window_rect"]
        )

        self.player_label.config(
            text=(
                f"Player: "
                f"{data['player'] or 'reading...'}"
            )
        )

        self.mode_label.config(
            text=(
                f"Mode: "
                f"{data['mode'] or '--'}"
            )
        )

        if data["distance"] is None:
            distance_text = "--"
        else:
            distance_text = (
                f"{data['distance']:.1f} "
                f"{data['distance_unit']}"
            )

        self.distance_label.config(
            text=(
                f"Distance: "
                f"{distance_text}"
            )
        )

        self.elevation_label.config(
            text=(
                f"Elevation: "
                f"{data['elevation']:+.1f} FT"
                if data["elevation"]
                is not None
                else "Elevation: --"
            )
        )

        wind = data["wind"]

        if wind is None:
            wind_text = "--"

        elif wind[0] == wind[1]:
            wind_text = (
                f"{wind[0]} MPH"
            )

        else:
            wind_text = (
                f"{wind[0]}-"
                f"{wind[1]} MPH"
            )

        self.wind_label.config(
            text=(
                f"Wind: "
                f"{wind_text}"
            )
        )

        self.green_label.config(
            text=(
                f"Green speed: "
                f"{data['green_speed_label']}"
            )
        )

        club_text = (
            data["club"]
            or "reading..."
        )

        if data["carry"] is not None:
            club_text += (
                f" | "
                f"{data['carry']:.0f} YD"
            )

        self.club_label.config(
            text=(
                f"Club: "
                f"{club_text}"
            )
        )

        self.confidence_label.config(
            text=(
                f"Read confidence: "
                f"{data['confidence'] * 100:.0f}%"
            )
        )

        calculation = (
            data["calculation"]
        )

        if calculation is None:
            calc_text = (
                "CALCULATION: "
                "READING HUD..."
            )

        else:
            calc_text = calculation[
                "summary"
            ]

        self.calc_label.config(
            text=calc_text
        )

        diag_text = " | ".join(
            data["diagnostics"]
        )

        if data["shot_changed"]:
            diag_text = (
                "NEW SHOT DETECTED | "
                + diag_text
            )

        self.diag_label.config(
            text=diag_text
        )

    def close(self):
        self.stop_event.set()

        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


# -----------------------------
# MAIN
# -----------------------------

def main():
    if not os.path.exists(
        TESSERACT
    ):
        raise FileNotFoundError(
            "Tesseract not found at: "
            + TESSERACT
        )

    print()
    print(
        "============================================"
    )
    print(
        "WGT ASSISTANT — SINGLE FILE"
    )
    print(
        "============================================"
    )
    print(
        "Automatic reading + calculation enabled."
    )
    print(
        "Swing/putt execution is intentionally not included."
    )
    print()

    WGTOverlay().run()


if __name__ == "__main__":
    main()
