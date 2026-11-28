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



# PUTT BADGE OCR V17B
_original_read_v17b = read_distance_and_elevation

def read_distance_and_elevation(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(
        hsv, (0, 0, 170), (179, 75, 255)
    )

    white[:int(h * .035)] = 0
    white[int(h * .30):] = 0
    white[:, :int(w * .27)] = 0
    white[:, int(w * .73):] = 0

    line = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(30, int(w * .05)), 2)
        )
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(line)
    candidates = []

    for x, y, width, _, area in stats[1:]:
        if .075*w < width < .27*w and .035*h < y < .25*h:
            score = width - abs(x + width/2 - w/2) * .35
            candidates.append((score, x, y, width))

    for _, x, y, width in sorted(candidates, reverse=True):
        inside = hsv[
            y + int(width*.15):y + int(width*.70),
            x + int(width*.10):x + int(width*.90)
        ]

        if not inside.size:
            continue

        blue = (
            (inside[:, :, 0] > 80)
            & (inside[:, :, 0] < 115)
            & (inside[:, :, 1] > 65)
        ).mean()

        if blue < .45:
            continue

        def number(left, right, top, bottom):
            section = frame[
                y + int(width*top):y + int(width*bottom),
                x + int(width*left):x + int(width*right)
            ]

            if not section.size:
                return None

            gray = cv2.cvtColor(
                section, cv2.COLOR_BGR2GRAY
            )

            enlarged = cv2.resize(
                gray, None,
                fx=4, fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            result = pytesseract.image_to_string(
                enlarged,
                config=(
                    "--psm 6 "
                    "-c tessedit_char_whitelist=0123456789."
                )
            )

            match = re.search(
                r"\d+(?:\.\d+)?", result
            )

            return float(match.group()) if match else None

        feet = number(.12, .77, .06, .38)
        inches = number(.30, .82, .40, .76)

        if (
            feet is not None
            and inches is not None
            and .5 <= feet <= 400
            and 0 <= inches <= 100
        ):
            return feet, inches

    return _original_read_v17b(frame)

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
    # PUTTING MOTION V2
    zones = ("near ball", "middle", "near hole")
    groups = {name: [] for name in zones}

    for before, after in zip(frames, frames[1:]):
        h, w = before.shape[:2]

        points = find_beads(before)
        future = find_beads(after)

        if len(points) < 12 or len(future) < 12:
            continue

        gray_a = cv2.cvtColor(
            before, cv2.COLOR_BGR2GRAY
        )
        gray_b = cv2.cvtColor(
            after, cv2.COLOR_BGR2GRAY
        )

        predicted, status, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray_a, gray_b,
                points.reshape(-1, 1, 2),
                None,
                winSize=(13, 13),
                maxLevel=2,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,
                    20, .02
                )
            )
        )

        if predicted is None:
            continue

        predicted = predicted.reshape(-1, 2)

        distances = (
            (predicted[:, None, :]
             - future[None, :, :]) ** 2
        ).sum(axis=2)

        nearest = distances.min(axis=1)
        movement = predicted - points
        lengths = np.linalg.norm(
            movement, axis=1
        )

        valid = (
            (status[:, 0] > 0)
            & (nearest < 16)
            & (lengths > .7)
            & (lengths < 12 * w / 625)
            & (np.abs(movement[:, 0]) > .5)
        )

        for point, motion in zip(
            points[valid], movement[valid]
        ):
            y = point[1] / h

            if y >= .76:
                zone = "near ball"
            elif y >= .63:
                zone = "middle"
            elif y >= .50:
                zone = "near hole"
            else:
                continue

            groups[zone].append(float(motion[0]))

    results = []
    reliable = []

    for zone in zones:
        values = np.asarray(groups[zone])

        if len(values) < 12:
            direction = "UNCERTAIN"
        else:
            right = int(np.sum(values > .5))
            left = int(np.sum(values < -.5))

            if right >= 12 and right > 1.6 * max(left, 1):
                direction = "BEADS MOVING RIGHT"
            elif left >= 12 and left > 1.6 * max(right, 1):
                direction = "BEADS MOVING LEFT"
            else:
                direction = "MIXED / UNCERTAIN"

        results.append(
            f"  {zone}: {direction}"
        )

        if direction.startswith("BEADS MOVING"):
            reliable.append(direction)

    if len(set(reliable)) > 1:
        overall = "CHANGING BREAK ACROSS GREEN"
    elif len(reliable) >= 2:
        overall = reliable[0] + " (provisional)"
    else:
        overall = "UNCERTAIN"

    return overall + "\n" + "\n".join(results)


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

        for i in range(30):
            image = np.asarray(
                screen.grab(region)
            )[:, :, :3].copy()

            frames.append(image)

            time.sleep(.10)

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
