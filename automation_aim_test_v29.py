"""WGT AIM TEST V29: timed, one small aiming drag; no swing.
Run from PowerShell. During countdown, focus WGT and hover aiming stick.
F9 emergency stop. Does not launch game.
"""
import ctypes
import ctypes.wintypes
import json
import time
from pathlib import Path
import cv2
import mss
import numpy as np
import pyautogui

ROOT = Path(__file__).resolve().parent
CAL = ROOT / 'automation_aim_calibration.json'
DEBUG = ROOT / 'automation_debug'
user32 = ctypes.windll.user32
VK_F9 = 0x78
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def stopped():
    if user32.GetAsyncKeyState(VK_F9) & 0x8000:
        raise KeyboardInterrupt('F9 pressed')


def setup():
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def wgt_window():
    matches = []
    cbtype = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd, unused):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                title = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, title, n + 1)
                if 'wgt golf' in title.value.lower():
                    matches.append(hwnd)
        return True
    callback_ref = cbtype(callback)
    user32.EnumWindows(callback_ref, 0)
    return matches[0] if matches else None


def get_rect(hwnd):
    r = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError('Cannot locate WGT window')
    return dict(left=r.left, top=r.top, width=r.right-r.left, height=r.bottom-r.top)


def check_focus(hwnd):
    stopped()
    if user32.GetForegroundWindow() != hwnd:
        raise RuntimeError('WGT is not the active window; no drag performed')


def screenshot(sct, rect, name):
    bgr = np.asarray(sct.grab(rect))[:, :, :3].copy()
    cv2.imwrite(str(DEBUG / name), bgr)


def main():
    setup()
    DEBUG.mkdir(exist_ok=True)
    print('WGT AIM TEST V29 - COUNTDOWN / NO SHOT', flush=True)
    print('WGT on Monitor 1 (left); PowerShell on Monitor 2.', flush=True)
    print('After Enter, click WGT to focus, then move pointer to the lower WHITE FLAG POLE (not the distance sign).', flush=True)
    print('Keep pointer on the flag pole WITHOUT holding the mouse button.', flush=True)
    print('A 80-pixel rightward drag will be tested once. F9 = emergency stop.', flush=True)
    input('Press Enter here to start the 10-second countdown... ')
    for n in range(10, 0, -1):
        stopped()
        print(f'{n}...', flush=True)
        time.sleep(1)
    hwnd = wgt_window()
    if not hwnd:
        raise RuntimeError('WGT window not found')
    check_focus(hwnd)
    rect = get_rect(hwnd)
    point = pyautogui.position()
    local_x = point.x - rect['left']
    local_y = point.y - rect['top']
    if not (0.12*rect['width'] < local_x < 0.85*rect['width'] and
            0.12*rect['height'] < local_y < 0.80*rect['height']):
        raise RuntimeError('Pointer not in safe central aiming area; no drag')
    if point.x + 80 >= rect['left'] + rect['width'] - 25:
        raise RuntimeError('Pointer too close to right edge; no drag')
    CAL.write_text(json.dumps(dict(x_fraction=local_x/rect['width'],
                                    y_fraction=local_y/rect['height']), indent=2), encoding='utf-8')
    print(f'Aim point saved: ({local_x}, {local_y}) relative to WGT', flush=True)
    with mss.mss() as sct:
        check_focus(hwnd)
        screenshot(sct, rect, 'aim_before.png')
        print('Testing one 80-pixel rightward drag. HANDS OFF.', flush=True)
        try:
            check_focus(hwnd)
            pyautogui.mouseDown(button='left')
            for i in range(1, 9):
                check_focus(hwnd)
                pyautogui.moveTo(point.x + 10*i, point.y, duration=.16)
        finally:
            pyautogui.mouseUp(button='left')
            user32.mouse_event(0x0004, 0, 0, 0, 0)
        time.sleep(.5)
        check_focus(hwnd)
        screenshot(sct, rect, 'aim_after.png')
        print('TEST COMPLETE. Compare automation_debug\\aim_before.png and aim_after.png.', flush=True)
        print('No golf swing was attempted.', flush=True)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        print('Emergency stop.', flush=True)
    except Exception as exc:
        print('Stopped safely:', exc, flush=True)
    finally:
        try:
            pyautogui.mouseUp(button='left')
            user32.mouse_event(0x0004, 0, 0, 0, 0)
        except Exception:
            pass
        print('AIM CONTROLLER CLOSED', flush=True)
