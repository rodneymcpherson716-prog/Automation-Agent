"""WGT V31: controlled aim drag, settle, verify swing button, then V26 swing.

Requires wgt_auto_one_shot_v26.py, swing_button.png, and wgt_accuracy.json
in the same WGT_Bot directory. This is an integration test, NOT a true
wind/club/trajectory planner. WGT must already be open on Monitor 1.
"""
import ctypes
import importlib.util
import re
import sys
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract

ROOT = Path(__file__).resolve().parent
V26 = ROOT / 'wgt_auto_one_shot_v26.py'
DEBUG = ROOT / 'wgt_debug'
AIM_DRAG_PIXELS = 80  # Observed once: 12 yards to 11 yards. Not universal.
COUNTDOWN_SECONDS = 10
VK_F9 = 0x78
user32 = ctypes.windll.user32
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def check_stop():
    if user32.GetAsyncKeyState(VK_F9) & 0x8000:
        raise KeyboardInterrupt('F9 emergency stop')


def release_mouse():
    try:
        pyautogui.mouseUp(button='left')
    finally:
        user32.mouse_event(0x0004, 0, 0, 0, 0)


def load_v26():
    if not V26.is_file():
        raise RuntimeError('Missing wgt_auto_one_shot_v26.py in WGT_Bot folder')
    spec = importlib.util.spec_from_file_location('wgt_v26_controller', V26)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(sct, rect):
    return np.asarray(sct.grab(rect))[:, :, :3].copy()


def save_image(name, image):
    DEBUG.mkdir(exist_ok=True)
    cv2.imwrite(str(DEBUG / name), image)


def read_approx_yards(frame):
    """Diagnostic OCR only. Not accurate enough to drive a shot planner."""
    h, w = frame.shape[:2]
    # The large 12 YDS overlay was near the upper-center of the example.
    patch = frame[int(h * .12):int(h * .40), int(w * .33):int(w * .70)]
    if not patch.size:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        txt = pytesseract.image_to_string(gray, config='--psm 11')
    except Exception:
        return None
    hits = re.findall(r'(?i)\b(\d{1,3})\s*(?:YDS?|YARDS?)\b', txt)
    return int(hits[0]) if hits else None


def main():
    controller = load_v26()
    controller.setup()
    print('WGT V31 — AIM, VERIFY, THEN SWING', flush=True)
    print('WGT on Monitor 1 LEFT; PowerShell on Monitor 2 RIGHT.', flush=True)
    print('One calibrated 80px rightward flag drag, then ONE automatic swing.', flush=True)
    print('Test only: wind/club/power calculations not yet calibrated.', flush=True)
    print('F9 = emergency stop; moving mouse to screen corner triggers failsafe.', flush=True)
    print('After countdown starts, activate WGT and hover LOWER WHITE FLAGPOLE.', flush=True)
    print('Do not click or drag yourself. Keep hands off after positioning.', flush=True)
    input('Press Enter to start the 10-second countdown... ')
    for sec in range(COUNTDOWN_SECONDS, 0, -1):
        check_stop()
        print(f'{sec}...', flush=True)
        time.sleep(1)

    hwnd = controller.find_wgt()
    if hwnd is None:
        raise RuntimeError('WGT Golf window not found')
    controller.guard(hwnd)
    rect = controller.get_rect(hwnd)
    if rect is None:
        raise RuntimeError('WGT window dimensions not available')
    point = pyautogui.position()
    local_x = point.x - rect['left']
    local_y = point.y - rect['top']
    if not (.12 * rect['width'] < local_x < .85 * rect['width']
            and .12 * rect['height'] < local_y < .80 * rect['height']):
        raise RuntimeError('Pointer outside aiming area. No aim or swing performed.')
    if point.x + AIM_DRAG_PIXELS >= rect['left'] + rect['width'] - 25:
        raise RuntimeError('Flag too close to right edge. No aim or swing performed.')

    template = cv2.imread(str(controller.TEMPLATE), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError('Missing swing_button.png')
    # Check required swing calibration before we move aim.
    controller.calibrated_target(rect, controller.meter_rect(rect))

    with mss.mss() as sct:
        controller.guard(hwnd)
        before = capture(sct, rect)
        save_image('v30_aim_before.png', before)
        before_yd = read_approx_yards(before)
        print(f'Flag point relative to WGT: ({local_x}, {local_y})')
        print(f'Before aim OCR: {before_yd if before_yd is not None else "unreadable"} yards')
        print('Dragging flag right 80px...', flush=True)
        try:
            controller.guard(hwnd)
            pyautogui.mouseDown(button='left')
            for step in range(1, 9):
                controller.guard(hwnd)
                pyautogui.moveTo(point.x + 10 * step, point.y, duration=.16)
        finally:
            release_mouse()
        time.sleep(.5)
        controller.guard(hwnd)
        after = capture(sct, controller.get_rect(hwnd))
        save_image('v30_aim_after.png', after)
        after_yd = read_approx_yards(after)
        print(f'After aim OCR: {after_yd if after_yd is not None else "unreadable"} yards')
        print('AIM DRAG SENT. Waiting for WGT to settle...', flush=True)
        # Aiming changes the screen; the original V30 immediately searched for
        # the swing button and got a 0.242 match. Move off the aiming target,
        # allow the shot UI to settle, and verify before handing off to V26.
        rect = controller.get_rect(hwnd)
        if rect is None:
            raise RuntimeError('WGT window disappeared after aiming')
        controller.guard(hwnd)
        pyautogui.moveTo(rect['left'] + int(rect['width'] * .72),
                           rect['top'] + int(rect['height'] * .88),
                           duration=.15)
        time.sleep(.8)
        strongest = 0.0
        confirmed = False
        for attempt in range(12):
            check_stop()
            controller.guard(hwnd)
            rect = controller.get_rect(hwnd)
            shot_frame = capture(sct, rect)
            score, button_point = controller.swing_button(shot_frame, template, rect)
            strongest = max(strongest, score)
            if button_point is not None and score >= controller.MIN_SWING_SCORE:
                confirmed = True
                print(f'Swing button visible after aiming: {score:.3f}', flush=True)
                break
            time.sleep(.25)
        if not confirmed:
            save_image('v31_swing_button_missing.png', shot_frame)
            print(f'SWING NOT STARTED: button confidence {strongest:.3f}.', flush=True)
            print('Saved wgt_debug\v31_swing_button_missing.png for diagnosis.', flush=True)
            print('Check whether the swing circle is visible in WGT after aiming.', flush=True)
            return
        print('Launching V26 swing with verified button.', flush=True)
        # V26 independently re-verifies the button after any window movement.
        controller.one_swing(hwnd, controller.get_rect(hwnd), sct, template)
        print('V31 handoff finished. Verify ball movement in WGT.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        print('Emergency stop.', flush=True)
    except Exception as error:
        print('Stopped safely:', error, flush=True)
    finally:
        release_mouse()
        print('V31 CONTROLLER CLOSED', flush=True)
