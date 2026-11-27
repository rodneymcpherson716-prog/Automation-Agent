import re
import time

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract


TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT

SCAN_DELAY = 0.4


def find_wgt():
    matches = []

    for window in pyautogui.getAllWindows():
        try:
            if (
                (window.title or "").strip().lower() == "wgt golf"
                and window.width >= 700
                and window.height >= 450
            ):
                matches.append(window)
        except Exception:
            pass

    if not matches:
        return None

    return max(
        matches,
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


def crop(image, x1, y1, x2, y2):
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

    texts = []

    for psm in (6, 7, 11):
        text = pytesseract.image_to_string(
            gray,
            config=f"--psm {psm}"
        ).strip()

        if text:
            texts.append(text)

    return "\n".join(texts)


def parse_distance(text):
    upper = text.upper()

    ft = re.search(
        r"(\d+(?:\.\d+)?)\s*FT",
        upper
    )

    if ft:
        return float(ft.group(1))

    yd = re.search(
        r"(\d+(?:\.\d+)?)\s*YDS?",
        upper
    )

    if yd:
        return float(yd.group(1)) * 3.0

    return None


def parse_elevation(text):
    upper = text.upper()

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*FT",
        upper
    )

    if not match:
        return 0.0

    return float(match.group(1))


def detect_green_speed(text):
    upper = text.upper()

    if "VERY FAST" in upper:
        return 11.5

    if "FAST" in upper:
        return 10.0

    if "SLOW" in upper:
        return 7.0

    return 9.0


def is_putting(text):
    return "PUTTER" in text.upper()


def calculate_putt(
    distance_ft,
    elevation_ft,
    green_speed
):
    playing_distance = distance_ft

    # Elevation starting model
    playing_distance += elevation_ft * 1.8

    # Green-speed adjustment
    if green_speed >= 11:
        playing_distance *= 0.88

    elif green_speed >= 10:
        playing_distance *= 0.92

    elif green_speed <= 7:
        playing_distance *= 1.05

    playing_distance = max(
        0.1,
        playing_distance
    )

    return round(
        playing_distance,
        1
    )


def read_putt_state():
    window = find_wgt()

    if window is None:
        return None

    image = capture_wgt(
        window
    )

    center = crop(
        image,
        0.40,
        0.08,
        0.69,
        0.35
    )

    upper_left = crop(
        image,
        0.00,
        0.02,
        0.32,
        0.23
    )

    lower_left = crop(
        image,
        0.00,
        0.83,
        0.38,
        1.00
    )

    center_text = read_text(center)
    upper_text = read_text(upper_left)
    lower_text = read_text(lower_left)

    combined = (
        center_text
        + "\n"
        + upper_text
        + "\n"
        + lower_text
    )

    return {
        "putting": is_putting(combined),
        "distance": parse_distance(center_text),
        "elevation": parse_elevation(center_text),
        "green_speed": detect_green_speed(
            upper_text
        ),
        "raw": combined,
    }


def main():
    print()
    print("=" * 60)
    print("WGT PUTTING AGENT")
    print("=" * 60)
    print("CTRL+C to stop.")
    print()

    last_output = None

    try:
        while True:
            state = read_putt_state()

            if state is None:
                message = "Waiting for WGT Golf..."

            elif not state["putting"]:
                message = "WGT detected — waiting for putting position."

            elif state["distance"] is None:
                message = "Putting detected — reading distance..."

            else:
                recommendation = calculate_putt(
                    state["distance"],
                    state["elevation"],
                    state["green_speed"]
                )

                message = (
                    f"PUTT READY | "
                    f"distance={state['distance']:.1f} ft | "
                    f"elevation={state['elevation']:.1f} ft | "
                    f"green={state['green_speed']:.1f} | "
                    f"PLAY={recommendation:.1f} ft"
                )

            if message != last_output:
                print(message)
                last_output = message

            time.sleep(
                SCAN_DELAY
            )

    except KeyboardInterrupt:
        print()
        print("Putting agent stopped.")


if __name__ == "__main__":
    main()