import re
import time

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT

SCAN_DELAY = 0.5


def get_wgt_window():
    windows = []

    for window in pyautogui.getAllWindows():
        try:
            if (
                (window.title or "").strip().lower() == "wgt golf"
                and window.width >= 700
                and window.height >= 450
            ):
                windows.append(window)
        except Exception:
            pass

    if not windows:
        return None

    return max(
        windows,
        key=lambda w: w.width * w.height
    )


def capture_wgt(window):
    monitor = {
        "left": int(window.left),
        "top": int(window.top),
        "width": int(window.width),
        "height": int(window.height),
    }

    with mss.mss() as sct:
        image = np.array(
            sct.grab(monitor)
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2BGR
    )


def crop_percent(
    image,
    x1,
    y1,
    x2,
    y2
):
    h, w = image.shape[:2]

    return image[
        int(h * y1):int(h * y2),
        int(w * x1):int(w * x2)
    ]


def read_text(region):
    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    return pytesseract.image_to_string(
        gray,
        config="--psm 7"
    ).strip()


def parse_distance(text):
    text = text.upper()

    feet = re.search(
        r"(\d+(?:\.\d+)?)\s*FT",
        text
    )

    yards = re.search(
        r"(\d+(?:\.\d+)?)\s*YDS?",
        text
    )

    if feet:
        return {
            "value": float(feet.group(1)),
            "unit": "ft"
        }

    if yards:
        return {
            "value": float(yards.group(1)),
            "unit": "yd"
        }

    return None


def parse_elevation(text):
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*FT",
        text.upper()
    )

    if match:
        return float(match.group(1))

    return 0.0


def parse_green_speed(text):
    text = text.upper()

    if "VERY FAST" in text:
        return 11.0

    if "FAST" in text:
        return 10.0

    if "SLOW" in text:
        return 7.0

    return 9.0


def calculate_putt(
    distance,
    elevation,
    green_speed
):
    if distance["unit"] == "yd":
        distance_ft = (
            distance["value"] * 3.0
        )
    else:
        distance_ft = distance["value"]

    adjusted = distance_ft

    # Starting approximation only.
    adjusted += elevation * 1.8

    if green_speed >= 11:
        adjusted *= 0.88

    elif green_speed >= 10:
        adjusted *= 0.92

    elif green_speed <= 7:
        adjusted *= 1.05

    return round(
        adjusted,
        1
    )


def read_putt():
    window = get_wgt_window()

    if window is None:
        return None

    image = capture_wgt(
        window
    )

    # Broad regions so we're not locked to one resolution.
    center = crop_percent(
        image,
        0.42,
        0.10,
        0.68,
        0.34
    )

    top_left = crop_percent(
        image,
        0.00,
        0.03,
        0.30,
        0.20
    )

    center_text = read_text(
        center
    )

    green_text = read_text(
        top_left
    )

    distance = parse_distance(
        center_text
    )

    elevation = parse_elevation(
        center_text
    )

    green_speed = parse_green_speed(
        green_text
    )

    return {
        "distance": distance,
        "elevation": elevation,
        "green_speed": green_speed,
        "center_ocr": center_text,
        "green_ocr": green_text,
    }


print()
print("=" * 60)
print("WGT PUTTING ASSISTANT")
print("=" * 60)
print("CTRL+C to stop.")
print()

try:

    while True:

        data = read_putt()

        if data is None:
            print(
                "Waiting for WGT Golf..."
            )

            time.sleep(1)
            continue

        distance = data["distance"]

        if distance is None:
            print(
                "Waiting for putting distance..."
            )

            time.sleep(
                SCAN_DELAY
            )
            continue

        recommendation = calculate_putt(
            distance,
            data["elevation"],
            data["green_speed"]
        )

        print(
            f"distance="
            f"{distance['value']} "
            f"{distance['unit']} | "
            f"elevation="
            f"{data['elevation']} ft | "
            f"green_speed="
            f"{data['green_speed']} | "
            f"recommended_playing_distance="
            f"{recommendation} ft"
        )

        time.sleep(
            SCAN_DELAY
        )

except KeyboardInterrupt:
    print()
    print(
        "WGT putting assistant stopped."
    )