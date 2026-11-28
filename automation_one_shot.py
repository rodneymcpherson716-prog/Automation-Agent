import ctypes
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pytesseract

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEBUG_DIR = Path("automation_debug")

pytesseract.pytesseract.tesseract_cmd = TESSERACT
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03

user32 = ctypes.windll.user32


def get_real_wgt_window():
    matches = []

    try:
        windows = pyautogui.getAllWindows()
    except Exception:
        windows = []

    for win in windows:
        try:
            title = (win.title or "").strip()
            if title.lower() == "wgt golf" and win.width >= 700 and win.height >= 450:
                matches.append(win)
        except Exception:
            pass

    if not matches:
        return None

    return max(matches, key=lambda w: int(w.width) * int(w.height))


def focus_window(win):
    try:
        if win.isMinimized:
            win.restore()
            time.sleep(0.3)
    except Exception:
        pass

    try:
        win.activate()
        time.sleep(0.4)
    except Exception:
        pass

    return int(user32.GetForegroundWindow()) == int(win._hWnd)


def capture_window(win):
    # We still capture by the known WGT rectangle, but only AFTER selecting
    # the exact window titled "WGT Golf".
    shot = pyautogui.screenshot(
        region=(
            int(win.left),
            int(win.top),
            int(win.width),
            int(win.height),
        )
    )

    return cv2.cvtColor(
        np.array(shot),
        cv2.COLOR_RGB2BGR,
    )


def ocr_region(img, x1, y1, x2, y2):
    h, w = img.shape[:2]

    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    region = img[y1:y2, x1:x2]

    if region.size == 0:
        return ""

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    return pytesseract.image_to_string(
        gray,
        config="--psm 7",
    ).strip()


def main():
    print()
    print("=" * 68)
    print("WGT ONE-SHOT DIAGNOSTIC")
    print("=" * 68)

    if not os.path.exists(TESSERACT):
        print("ERROR: Tesseract not found:", TESSERACT)
        return

    win = get_real_wgt_window()

    if win is None:
        print("ERROR: No window titled exactly 'WGT Golf' was found.")
        print()
        print("Visible window titles:")

        try:
            for w in pyautogui.getAllWindows():
                title = (w.title or "").strip()
                if title:
                    print(" -", repr(title))
        except Exception as exc:
            print("Could not enumerate windows:", exc)

        return

    print("FOUND:", repr(win.title))
    print(
        f"RECT: left={win.left}, top={win.top}, "
        f"width={win.width}, height={win.height}"
    )

    focused = focus_window(win)
    print("FOREGROUND:", focused)

    img = capture_window(win)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    capture_path = DEBUG_DIR / "one_shot_automation_capture.png"
    cv2.imwrite(str(capture_path), img)

    h, w = img.shape[:2]

    # Broad regions only. These are diagnostic, not final calibration.
    upper_left = ocr_region(
        img,
        0,
        int(h * 0.03),
        int(w * 0.32),
        int(h * 0.19),
    )

    center_top = ocr_region(
        img,
        int(w * 0.45),
        int(h * 0.10),
        int(w * 0.65),
        int(h * 0.25),
    )

    lower_left = ocr_region(
        img,
        0,
        int(h * 0.86),
        int(w * 0.35),
        h,
    )

    print()
    print("UPPER LEFT OCR:", repr(upper_left))
    print("CENTER TOP OCR:", repr(center_top))
    print("LOWER LEFT OCR:", repr(lower_left))
    print()
    print("SAVED:", capture_path.resolve())
    print()
    print("DIAGNOSTIC COMPLETE - PROGRAM EXITING")
    print("No loop. No repeated screenshots.")


if __name__ == "__main__":
    main()
