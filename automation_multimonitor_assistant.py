import ctypes
import os
import re
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


# ============================================================
# WGT ASSISTANT — MULTI-MONITOR CAPTURE VERSION
# Handles WGT windows positioned on monitors with negative X/Y.
# Reads the WGT HUD and gives shot information/recommendations.
# ============================================================

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEBUG_DIR = Path("automation_debug")

SCAN_DELAY = 0.50
DEBUG_INTERVAL = 15.0

pytesseract.pytesseract.tesseract_cmd = TESSERACT
pyautogui.FAILSAFE = True

user32 = ctypes.windll.user32


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
    """
    MSS correctly supports negative monitor coordinates.
    This is the important change from the previous pyautogui screenshot path.
    """
    monitor = {
        "left": int(win.left),
        "top": int(win.top),
        "width": int(win.width),
        "height": int(win.height),
    }

    with mss.mss() as sct:
        raw = np.array(sct.grab(monitor))

    # BGRA -> BGR
    return cv2.cvtColor(
        raw,
        cv2.COLOR_BGRA2BGR,
    )


def crop_norm(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    xx1 = max(0, min(w, int(w * x1)))
    yy1 = max(0, min(h, int(h * y1)))
    xx2 = max(0, min(w, int(w * x2)))
    yy2 = max(0, min(h, int(h * y2)))

    if xx2 <= xx1 or yy2 <= yy1:
        return None

    return img[yy1:yy2, xx1:xx2]


def preprocess(region, scale=5):
    if region is None or region.size == 0:
        return []

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
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

    return [gray, binary, inverse]


def ocr(region, psm_values=(6, 7, 11, 12)):
    texts = []

    for image in preprocess(region):
        for psm in psm_values:
            try:
                text = pytesseract.image_to_string(
                    image,
                    config=f"--psm {psm}",
                ).strip()
            except Exception:
                text = ""

            if text:
                texts.append(text)

    # preserve order, remove duplicates
    seen = set()
    output = []

    for text in texts:
        key = text.strip()

        if key and key not in seen:
            seen.add(key)
            output.append(key)

    return output


def parse_wind(texts):
    votes = {}

    for text in texts:
        upper = text.upper()

        # Typical WGT: 11-13 MPH
        for match in re.finditer(
            r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*MPH",
            upper,
        ):
            a = int(match.group(1))
            b = int(match.group(2))

            if 0 <= a <= 50 and 0 <= b <= 50:
                pair = (min(a, b), max(a, b))
                votes[pair] = votes.get(pair, 0) + 5

        # Single wind speed fallback
        for match in re.finditer(
            r"(\d{1,2})\s*MPH",
            upper,
        ):
            value = int(match.group(1))

            if 0 <= value <= 50:
                pair = (value, value)
                votes[pair] = votes.get(pair, 0) + 2

        # OCR may lose "MPH", so accept a plausible range as a last fallback.
        for match in re.finditer(
            r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\b",
            upper,
        ):
            a = int(match.group(1))
            b = int(match.group(2))

            if 0 <= a <= 50 and 0 <= b <= 50:
                pair = (min(a, b), max(a, b))
                votes[pair] = votes.get(pair, 0) + 1

    if not votes:
        return None

    return max(votes, key=votes.get)


def parse_yards(texts, minimum=1, maximum=999):
    votes = {}

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(\d{1,3})\s*(?:YD|YDS|YARD|YARDS)",
            upper,
        ):
            value = int(match.group(1))

            if minimum <= value <= maximum:
                votes[value] = votes.get(value, 0) + 4

    if not votes:
        return None

    return max(votes, key=votes.get)


def parse_number(texts, minimum=0, maximum=999):
    votes = {}

    for text in texts:
        for token in re.findall(r"\b\d{1,3}\b", text):
            value = int(token)

            if minimum <= value <= maximum:
                votes[value] = votes.get(value, 0) + 1

    if not votes:
        return None

    return max(votes, key=votes.get)


def parse_elevation(texts, region):
    magnitude = None

    # Prefer FT-associated value.
    votes = {}

    for text in texts:
        upper = text.upper()

        for match in re.finditer(
            r"(\d{1,3})\s*(?:FT|FEET)",
            upper,
        ):
            value = int(match.group(1))

            if 0 <= value <= 300:
                votes[value] = votes.get(value, 0) + 4

    if votes:
        magnitude = max(votes, key=votes.get)
    else:
        magnitude = parse_number(
            texts,
            minimum=0,
            maximum=300,
        )

    if magnitude is None:
        return None

    # WGT uses a green directional triangle. Determine up/down by its shape.
    try:
        hsv = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2HSV,
        )

        green = cv2.inRange(
            hsv,
            np.array([35, 70, 70]),
            np.array([100, 255, 255]),
        )

        ys, _ = np.where(green > 0)

        if len(ys) >= 8:
            midpoint = (
                float(ys.min()) + float(ys.max())
            ) / 2.0

            # Down triangle has its pixel mass slightly above the box center.
            sign = -1 if float(ys.mean()) < midpoint else 1

            return magnitude * sign
    except Exception:
        pass

    return magnitude


def parse_player_name(texts):
    for text in texts:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:
            # Reject obvious HUD-only lines.
            upper = line.upper()

            if any(
                token in upper
                for token in (
                    "LVL",
                    "SHOT",
                    "TEE",
                    "YDS",
                    "MPH",
                )
            ):
                continue

            cleaned = re.sub(
                r"[^A-Za-z0-9_.-]",
                "",
                line,
            )

            if 3 <= len(cleaned) <= 24:
                return cleaned

    return None


def parse_club_name(texts):
    known = (
        "PUTTER",
        "DRIVER",
        "3 WOOD",
        "5 WOOD",
        "HYBRID",
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

    joined = "\n".join(texts).upper()

    for club in known:
        if club in joined:
            return club.title()

    return None


def read_hud(img):
    # Broad HUD zones based on the WGT layout rather than tiny fixed boxes.
    top_left = crop_norm(
        img,
        0.00, 0.035,
        0.24, 0.19,
    )

    center_top = crop_norm(
        img,
        0.47, 0.10,
        0.66, 0.27,
    )

    top_right = crop_norm(
        img,
        0.79, 0.035,
        1.00, 0.15,
    )

    bottom_left = crop_norm(
        img,
        0.00, 0.86,
        0.34, 1.00,
    )

    top_left_text = ocr(
        top_left,
        psm_values=(6, 7, 11, 12),
    )

    center_text = ocr(
        center_top,
        psm_values=(6, 7, 11, 12),
    )

    top_right_text = ocr(
        top_right,
        psm_values=(6, 7, 11, 12),
    )

    bottom_text = ocr(
        bottom_left,
        psm_values=(6, 7, 11, 12),
    )

    wind = parse_wind(top_left_text)

    # Hole distance usually lives in upper-left.
    hole_distance = parse_yards(
        top_left_text,
        minimum=30,
        maximum=999,
    )

    # Target/pin distance lives center-top.
    pin_distance = parse_yards(
        center_text,
        minimum=1,
        maximum=999,
    )

    elevation = parse_elevation(
        center_text,
        center_top,
    )

    player_name = parse_player_name(
        top_right_text
    )

    club_name = parse_club_name(
        bottom_text
    )

    club_carry = parse_yards(
        bottom_text,
        minimum=20,
        maximum=450,
    )

    putting = (
        club_name == "Putter"
        or "PUTTER" in "\n".join(bottom_text).upper()
    )

    return {
        "wind": wind,
        "hole_distance": hole_distance,
        "pin_distance": pin_distance,
        "elevation": elevation,
        "player_name": player_name,
        "club_name": club_name,
        "club_carry": club_carry,
        "putting": putting,
        "raw": {
            "top_left": top_left_text,
            "center_top": center_text,
            "top_right": top_right_text,
            "bottom_left": bottom_text,
        },
    }


def calculate_shot(hud):
    pin = hud["pin_distance"]

    if pin is None:
        return None

    adjusted = float(pin)

    elevation = hud["elevation"]

    if elevation is not None:
        adjusted += float(elevation) / 3.0

    power = None

    carry = hud["club_carry"]

    if carry is not None and carry > 0:
        power = adjusted / float(carry) * 100.0
        power = max(
            10.0,
            min(100.0, power),
        )

    return {
        "adjusted_distance": adjusted,
        "power_percent": power,
    }


def format_wind(wind):
    if wind is None:
        return "?"

    if wind[0] == wind[1]:
        return f"{wind[0]} MPH"

    return f"{wind[0]}-{wind[1]} MPH"


def save_debug(img, hud):
    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(DEBUG_DIR / "latest_automation_capture.png"),
        img,
    )

    lines = [
        f"player_name={hud['player_name']}",
        f"pin_distance={hud['pin_distance']}",
        f"elevation={hud['elevation']}",
        f"wind={hud['wind']}",
        f"hole_distance={hud['hole_distance']}",
        f"club_name={hud['club_name']}",
        f"club_carry={hud['club_carry']}",
        f"putting={hud['putting']}",
        "",
        "RAW OCR:",
        repr(hud["raw"]),
    ]

    (DEBUG_DIR / "latest_hud.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    if not os.path.exists(TESSERACT):
        print(
            "ERROR: Tesseract was not found:",
            TESSERACT,
        )
        return

    print()
    print("=" * 70)
    print("WGT MULTI-MONITOR ASSISTANT")
    print("=" * 70)
    print(
        "Reads WGT on the secondary monitor, including negative coordinates."
    )
    print("CTRL+C stops the assistant.")
    print()

    last_output = None
    last_debug = 0.0

    try:
        while True:
            win = get_wgt_window()

            if win is None:
                message = "Waiting for WGT Golf..."

                if message != last_output:
                    print(message)
                    last_output = message

                time.sleep(1.0)
                continue

            try:
                img = capture_wgt(win)
            except Exception as exc:
                message = f"WGT capture error: {exc}"

                if message != last_output:
                    print(message)
                    last_output = message

                time.sleep(1.0)
                continue

            hud = read_hud(img)
            shot = calculate_shot(hud)

            player = (
                hud["player_name"]
                or "?"
            )

            mode = (
                "PUTTING"
                if hud["putting"]
                else "FULL SHOT"
            )

            output = (
                f"player={player} | "
                f"mode={mode} | "
                f"pin={hud['pin_distance']} yd | "
                f"elev={hud['elevation']} ft | "
                f"wind={format_wind(hud['wind'])} | "
                f"hole={hud['hole_distance']} yd | "
                f"club={hud['club_name']} | "
                f"carry={hud['club_carry']} yd"
            )

            if shot is not None:
                output += (
                    f" | adjusted={shot['adjusted_distance']:.1f} yd"
                )

                if shot["power_percent"] is not None:
                    output += (
                        f" | suggested_power={shot['power_percent']:.1f}%"
                    )

            if output != last_output:
                print(output)
                last_output = output

            now = time.time()

            if (
                hud["pin_distance"] is None
                or hud["wind"] is None
                or now - last_debug >= DEBUG_INTERVAL
            ):
                save_debug(
                    img,
                    hud,
                )
                last_debug = now

            time.sleep(SCAN_DELAY)

    except KeyboardInterrupt:
        print()
        print("WGT assistant stopped.")

    except pyautogui.FailSafeException:
        print()
        print("Emergency stop.")


if __name__ == "__main__":
    main()
