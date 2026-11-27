import os
import re
import time
import tkinter as tk
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


# ============================================================
# WGT LIVE OVERLAY ASSISTANT
# ------------------------------------------------------------
# - Captures the exact "WGT Golf" window, including negative
#   monitor coordinates.
# - Displays calculations ON TOP OF THE GAME instead of hiding
#   them in the VS Code terminal.
# - Does not click or execute the swing/putt.
# ============================================================

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT

SCAN_MS = 700
DEBUG_DIR = Path("wgt_debug")

pyautogui.FAILSAFE = True


def get_wgt_window():
    matches = []

    try:
        windows = pyautogui.getAllWindows()
    except Exception:
        windows = []

    for win in windows:
        try:
            title = (win.title or "").strip()

            if title.lower() != "wgt golf":
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


def capture_wgt(win):
    monitor = {
        "left": int(win.left),
        "top": int(win.top),
        "width": int(win.width),
        "height": int(win.height),
    }

    with mss.mss() as sct:
        raw = np.array(sct.grab(monitor))

    return cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2BGR,
    )


def crop_norm(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    a = max(0, min(w, int(w * x1)))
    b = max(0, min(h, int(h * y1)))
    c = max(0, min(w, int(w * x2)))
    d = max(0, min(h, int(h * y2)))

    if c <= a or d <= b:
        return None

    return img[b:d, a:c]


def preprocess(region):
    if region is None or region.size == 0:
        return []

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC,
    )

    _, white = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    _, black = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return [gray, white, black]


def read_text(region):
    texts = []

    for image in preprocess(region):
        for psm in (6, 7, 11, 12):
            try:
                text = pytesseract.image_to_string(
                    image,
                    config=f"--psm {psm}",
                ).strip()
            except Exception:
                text = ""

            if text:
                texts.append(text)

    seen = set()
    result = []

    for text in texts:
        if text not in seen:
            seen.add(text)
            result.append(text)

    return result


def parse_distance_ft(texts):
    votes = []

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*FT",
            upper,
        ):
            value = float(match.group(1))

            if 0.1 <= value <= 500:
                votes.append(value)

        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*YDS?",
            upper,
        ):
            value = float(match.group(1)) * 3.0

            if 0.1 <= value <= 1500:
                votes.append(value)

    if not votes:
        return None

    # Median is more robust against one bad OCR result.
    votes.sort()
    return votes[len(votes) // 2]


def parse_elevation_ft(texts):
    values = []

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(-?\d+(?:\.\d+)?)\s*FT",
            upper,
        ):
            value = float(match.group(1))

            if -300 <= value <= 300:
                values.append(value)

    if not values:
        return 0.0

    # Smaller FT value in the center HUD is usually elevation.
    return min(
        values,
        key=lambda x: abs(x),
    )


def parse_green_speed(texts):
    joined = "\n".join(texts).upper()

    # If WGT exposes a numeric green-stimp value, prefer it.
    match = re.search(
        r"(?:GREEN|STIMP)[^\d]{0,12}(\d+(?:\.\d+)?)",
        joined,
    )

    if match:
        value = float(match.group(1))

        if 5.0 <= value <= 15.0:
            return value

    if "VERY FAST" in joined:
        return 11.5

    if "FAST" in joined:
        return 10.0

    if "SLOW" in joined:
        return 7.0

    return 9.0


def parse_player(texts):
    joined = "\n".join(texts)

    # Known/typical usernames: letters/numbers/underscore, 3-24 chars.
    candidates = re.findall(
        r"\b[A-Za-z][A-Za-z0-9_]{2,23}\b",
        joined,
    )

    rejected = {
        "shot", "score", "hole", "green", "speed", "putter",
        "driver", "iron", "wedge", "yards", "feet", "level",
    }

    for value in candidates:
        if value.lower() not in rejected:
            return value

    return None


def detect_putting(texts):
    joined = "\n".join(texts).upper()
    return "PUTTER" in joined or "PUTT" in joined


def calculate_putt(distance_ft, elevation_ft, green_speed):
    if distance_ft is None:
        return None

    play = float(distance_ft)

    # Starting putting model:
    # 1 ft uphill adds roughly 1.8 ft of playing distance.
    play += elevation_ft * 1.8

    if green_speed >= 11:
        play *= 0.88
    elif green_speed >= 10:
        play *= 0.92
    elif green_speed <= 7:
        play *= 1.05

    return max(0.1, round(play, 1))


def calculate_full_shot(distance_ft, elevation_ft):
    if distance_ft is None:
        return None

    yards = distance_ft / 3.0

    # Starting full-shot elevation model.
    yards += elevation_ft / 3.0

    return round(max(1.0, yards), 1)


class Overlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WGT Assistant")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#111111")
        self.root.resizable(False, False)

        self.title = tk.Label(
            self.root,
            text="WGT ASSISTANT",
            font=("Segoe UI", 15, "bold"),
            fg="white",
            bg="#111111",
        )
        self.title.pack(
            padx=18,
            pady=(12, 2),
        )

        self.player = tk.Label(
            self.root,
            text="Player: waiting...",
            font=("Segoe UI", 10),
            fg="#dddddd",
            bg="#111111",
        )
        self.player.pack()

        self.mode = tk.Label(
            self.root,
            text="Mode: waiting...",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#111111",
        )
        self.mode.pack(
            pady=(6, 0)
        )

        self.distance = tk.Label(
            self.root,
            text="Distance: --",
            font=("Segoe UI", 12),
            fg="white",
            bg="#111111",
        )
        self.distance.pack()

        self.elevation = tk.Label(
            self.root,
            text="Elevation: --",
            font=("Segoe UI", 12),
            fg="white",
            bg="#111111",
        )
        self.elevation.pack()

        self.green = tk.Label(
            self.root,
            text="Green speed: --",
            font=("Segoe UI", 12),
            fg="white",
            bg="#111111",
        )
        self.green.pack()

        self.recommendation = tk.Label(
            self.root,
            text="CALCULATION: waiting...",
            font=("Segoe UI", 15, "bold"),
            fg="#ffffff",
            bg="#222222",
            padx=12,
            pady=9,
        )
        self.recommendation.pack(
            padx=12,
            pady=(10, 8),
            fill="x",
        )

        self.status = tk.Label(
            self.root,
            text="Looking for WGT Golf...",
            font=("Segoe UI", 9),
            fg="#bbbbbb",
            bg="#111111",
            wraplength=320,
        )
        self.status.pack(
            padx=12,
            pady=(0, 12),
        )

        self.last_debug = 0.0

        self.root.after(
            250,
            self.update,
        )

    def place_next_to_game(self, win):
        try:
            # Put overlay inside the top-right portion of the WGT monitor.
            x = int(win.left + win.width - 365)
            y = int(win.top + 85)

            self.root.geometry(
                f"340x285+{x}+{y}"
            )
        except Exception:
            pass

    def save_debug(self, img, texts):
        DEBUG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(DEBUG_DIR / "overlay_latest_capture.png"),
            img,
        )

        (DEBUG_DIR / "overlay_latest_ocr.txt").write_text(
            repr(texts),
            encoding="utf-8",
        )

    def update(self):
        try:
            win = get_wgt_window()

            if win is None:
                self.status.config(
                    text="WGT Golf window not found."
                )

                self.root.after(
                    SCAN_MS,
                    self.update,
                )
                return

            self.place_next_to_game(
                win
            )

            img = capture_wgt(
                win
            )

            # Broad reading areas based on actual WGT UI:
            top_right = crop_norm(
                img,
                0.72, 0.02,
                1.00, 0.18,
            )

            center_top = crop_norm(
                img,
                0.38, 0.07,
                0.70, 0.34,
            )

            top_left = crop_norm(
                img,
                0.00, 0.02,
                0.34, 0.24,
            )

            lower_left = crop_norm(
                img,
                0.00, 0.78,
                0.42, 1.00,
            )

            player_text = read_text(
                top_right
            )

            center_text = read_text(
                center_top
            )

            top_text = read_text(
                top_left
            )

            lower_text = read_text(
                lower_left
            )

            all_text = (
                player_text
                + center_text
                + top_text
                + lower_text
            )

            player = parse_player(
                player_text
            )

            putting = detect_putting(
                all_text
            )

            distance_ft = parse_distance_ft(
                center_text
            )

            elevation_ft = parse_elevation_ft(
                center_text
            )

            green_speed = parse_green_speed(
                top_text + lower_text
            )

            self.player.config(
                text=f"Player: {player or 'reading...'}"
            )

            self.mode.config(
                text=(
                    "Mode: PUTTING"
                    if putting
                    else "Mode: FULL SHOT"
                )
            )

            if distance_ft is None:
                self.distance.config(
                    text="Distance: reading..."
                )
            else:
                if putting:
                    self.distance.config(
                        text=f"Distance: {distance_ft:.1f} ft"
                    )
                else:
                    self.distance.config(
                        text=f"Distance: {distance_ft / 3.0:.1f} yd"
                    )

            self.elevation.config(
                text=f"Elevation: {elevation_ft:+.1f} ft"
            )

            self.green.config(
                text=f"Green speed model: {green_speed:.1f}"
            )

            if distance_ft is None:
                self.recommendation.config(
                    text="CALCULATION: READING HUD..."
                )
                self.status.config(
                    text=(
                        "The overlay is live. Waiting for a readable "
                        "distance value from the WGT HUD."
                    )
                )
            elif putting:
                play = calculate_putt(
                    distance_ft,
                    elevation_ft,
                    green_speed,
                )

                self.recommendation.config(
                    text=f"PLAY PUTT: {play:.1f} ft"
                )

                self.status.config(
                    text=(
                        "Live calculation updated automatically. "
                        "Make the putt in WGT when ready."
                    )
                )
            else:
                play_yards = calculate_full_shot(
                    distance_ft,
                    elevation_ft,
                )

                self.recommendation.config(
                    text=f"PLAY SHOT: {play_yards:.1f} yd"
                )

                self.status.config(
                    text=(
                        "Live full-shot distance calculation updated "
                        "automatically."
                    )
                )

            now = time.time()

            if (
                distance_ft is None
                and now - self.last_debug > 8.0
            ):
                self.save_debug(
                    img,
                    {
                        "player": player_text,
                        "center": center_text,
                        "top": top_text,
                        "bottom": lower_text,
                    },
                )

                self.last_debug = now

        except Exception as exc:
            self.status.config(
                text=f"Reader error: {exc}"
            )

        self.root.after(
            SCAN_MS,
            self.update,
        )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if not os.path.exists(TESSERACT):
        raise FileNotFoundError(
            f"Tesseract was not found here: {TESSERACT}"
        )

    Overlay().run()
