import time
import re
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract

for path in [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]:
    if Path(path).exists():
        pytesseract.pytesseract.tesseract_cmd = path
        break


def find_wgt():
    for window in pyautogui.getWindowsWithTitle("WGT Golf"):
        if window.width >= 600 and window.height >= 400:
            if not window.isMinimized:
                return {
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height
                }
    return None


def find_distance_box(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue = cv2.inRange(
        hsv, (85, 70, 65), (114, 255, 255)
    )

    blue[int(h * .48):] = 0
    blue[:, :int(w * .16)] = 0
    blue[:, int(w * .84):] = 0

    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=3
    )

    contours, _ = cv2.findContours(
        blue, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        if (
            bw > .07 * w
            and bh > .06 * h
            and .65 < bw / bh < 2.5
        ):
            boxes.append(
                (cv2.contourArea(contour),
                 (x, y, bw, bh))
            )

    return max(boxes, default=(None, None))[1]


def read_distance_and_elevation(frame):
    box = find_distance_box(frame)

    if box is None:
        return None, None

    x, y, w, h = box

    gray = cv2.cvtColor(
        frame[y:y+h, x:x+w],
        cv2.COLOR_BGR2GRAY
    )

    sections = [
        gray[3:int(h * .49),
             int(w * .09):int(w * .97)],
        gray[int(h * .45):,
             int(w * .28):]
    ]

    results = []

    for section in sections:
        _, mask = cv2.threshold(
            section, 180, 255,
            cv2.THRESH_BINARY
        )

        large = cv2.resize(
            mask, None, fx=4, fy=4,
            interpolation=cv2.INTER_NEAREST
        )

        text = pytesseract.image_to_string(
            large,
            config="--psm 6 -c tessedit_char_whitelist=0123456789.+-FTIN "
        )

        match = re.search(
            r"[-+]?\d+(?:\.\d+)?", text
        )

        results.append(
            float(match.group()) if match else None
        )

    return results[0], results[1]


def find_beads(frame):
    h, w = frame.shape[:2]

    x0, x1 = int(w * .10), int(w * .92)
    y0, y1 = int(h * .32), int(h * .95)

    area = frame[y0:y1, x0:x1]

    highest = area.max(axis=2)
    lowest = area.min(axis=2)

    white = (
        (lowest > 190)
        & ((highest - lowest) < 48)
    ).astype(np.uint8)

    count, _, stats, centers = (
        cv2.connectedComponentsWithStats(
            white, 8
        )
    )

    points = []
    scale = max(1.0, w / 625.0)

    for i in range(1, count):
        x, y, bw, bh, size = stats[i]

        if (
            3 <= size <= 110 * scale * scale
            and 2 <= bw <= 13 * scale
            and 2 <= bh <= 13 * scale
            and .40 <= bw / bh <= 2.5
        ):
            cx, cy = centers[i]
            points.append((cx + x0, cy + y0))

    return np.array(
        points, dtype=np.float32
    ).reshape(-1, 2)


def analyze_break(frames):
    movements = []

    for first, second in zip(
        frames, frames[1:]
    ):
        a = find_beads(first)
        b = find_beads(second)

        if len(a) < 8 or len(b) < 8:
            continue

        distances = np.sum(
            (a[:, None, :] - b[None, :, :]) ** 2,
            axis=2
        )

        nearest = np.argmin(distances, axis=1)

        movement = b[nearest] - a

        lengths = np.linalg.norm(
            movement, axis=1
        )

        valid = (
            (lengths > .75)
            & (lengths < 12 * first.shape[1] / 625)
        )

        movements.extend(
            movement[valid].tolist()
        )

    if len(movements) < 12:
        return "UNCERTAIN - insufficient bead movement"

    dx = np.array(movements)[:, 0]

    left = np.sum(dx < -1)
    right = np.sum(dx > 1)

    if left >= 9 and left > 1.8 * right:
        return "BEADS MOVING LEFT - possible left break"

    if right >= 9 and right > 1.8 * left:
        return "BEADS MOVING RIGHT - possible right break"

    return "MIXED OR UNCERTAIN BREAK"


def main():
    print()
    print("TIGERWOULD52 PUTTING ANALYZER")
    print("READ ONLY - NO AUTOMATIC SWING")
    print()

    with mss.mss() as screen:
        region = find_wgt()

        if region is None:
            if len(screen.monitors) < 3:
                print("WGT window not found.")
                return

            region = screen.monitors[2]
            print("Checking Monitor 2.")
        else:
            print("WGT game window found.")

        frames = []

        print("Reading putting green...")

        for i in range(14):
            image = np.asarray(
                screen.grab(region)
            )[:, :, :3].copy()

            frames.append(image)

            time.sleep(.13)

    feet, inches = read_distance_and_elevation(
        frames[0]
    )

    print()
    print("PUTTING ANALYSIS")
    print("----------------")

    if feet is not None:
        print(f"Distance: {feet} feet")
    else:
        print("Distance: Unable to read")

    if inches is not None:
        print(f"Elevation displayed: {inches} inches")
        print("Verify uphill/downhill arrow.")
    else:
        print("Elevation: Unable to read")

    print("Break:", analyze_break(frames))
    print()
    print("No club, aim, power or swing changed.")


if __name__ == "__main__":
    main()
