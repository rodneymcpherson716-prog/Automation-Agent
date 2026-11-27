"""WGT aim-stick calibration, no swing.
Monitor 1 left: WGT. Monitor 2 right: terminal.
F6: Save current mouse pointer as aiming-stick grab point.
F7: Test one tiny rightward drag at saved point.
F9: Stop and release mouse.
This never launches WGT or hits a shot.
"""
import ctypes
import ctypes.wintypes
import json
import time
from pathlib import Path

import mss
import numpy as np
import cv2
import pyautogui

ROOT = Path(__file__).resolve().parent
CAL = ROOT / 'wgt_aim_calibration.json'
DEBUG = ROOT / 'wgt_debug'
VK_F6, VK_F7, VK_F9 = 0x75, 0x76, 0x78
user32 = ctypes.windll.user32
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def pressed(key):
    return bool(user32.GetAsyncKeyState(key) & 0x8000)


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
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                name = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, name, length + 1)
                if 'wgt golf' in name.value.lower():
                    matches.append(hwnd)
        return True
    cb = cbtype(callback)
    user32.EnumWindows(cb, 0)
    return matches[0] if matches else None


def window_rect(hwnd):
    r = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError('Cannot read WGT window position')
    return {'left': r.left, 'top': r.top,
            'width': r.right-r.left, 'height': r.bottom-r.top}


def check(hwnd):
    if pressed(VK_F9):
        raise KeyboardInterrupt
    if user32.GetForegroundWindow() != hwnd:
        raise RuntimeError('WGT lost focus; stopped without clicking')


def screenshot(sct, rect, path):
    frame = np.asarray(sct.grab(rect))[:, :, :3].copy()
    cv2.imwrite(str(path), frame)


def main():
    setup()
    DEBUG.mkdir(exist_ok=True)
    print('WGT AIM TEST V27 -- NO SWING')
    print('Keep WGT on Monitor 1. Leave PowerShell open on Monitor 2.')
    print('F6 = save aiming-stick position under your mouse cursor')
    print('F7 = test a single 20-pixel rightward aiming-stick drag')
    print('F9 = emergency stop / mouse release')
    print('Move pointer onto the aiming stick, without clicking, then press F6.')
    prev6 = prev7 = False
    try:
        with mss.mss() as sct:
            while True:
                if pressed(VK_F9):
                    print('F9 stop')
                    break
                hwnd = wgt_window()
                focus = hwnd is not None and user32.GetForegroundWindow() == hwnd
                f6, f7 = pressed(VK_F6), pressed(VK_F7)
                if focus and f6 and not prev6:
                    rect = window_rect(hwnd)
                    x, y = pyautogui.position()
                    local_x, local_y = x-rect['left'], y-rect['top']
                    # Reject title bar, HUD meter, and swing-button corner.
                    if not (.12*rect['width'] < local_x < .85*rect['width'] and
                            .12*rect['height'] < local_y < .80*rect['height']):
                        print('Pointer outside safe aim area. Aim not saved.')
                    else:
                        CAL.write_text(json.dumps({'x_fraction': local_x/rect['width'],
                            'y_fraction': local_y/rect['height']}, indent=2), encoding='utf-8')
                        screenshot(sct, rect, DEBUG/'aim_before.png')
                        print(f'AIM POINT SAVED: local ({local_x},{local_y})')
                        print('Press F7 for a tiny rightward test. No shot will be hit.')
                if focus and f7 and not prev7:
                    if not CAL.exists():
                        print('Press F6 over aiming stick first.')
                    else:
                        rect = window_rect(hwnd)
                        saved = json.loads(CAL.read_text(encoding='utf-8'))
                        x = round(rect['left'] + saved['x_fraction']*rect['width'])
                        y = round(rect['top'] + saved['y_fraction']*rect['height'])
                        if x+20 >= rect['left']+rect['width']-25:
                            print('Too close to right edge; no movement.')
                            continue
                        check(hwnd)
                        screenshot(sct, rect, DEBUG/'aim_before.png')
                        print('Testing 20-pixel rightward drag. HANDS OFF MOUSE.')
                        try:
                            pyautogui.moveTo(x, y, duration=.15)
                            check(hwnd)
                            pyautogui.mouseDown(button='left')
                            for n in range(1, 5):
                                check(hwnd)
                                pyautogui.moveTo(x+n*5, y, duration=.08)
                        finally:
                            pyautogui.mouseUp(button='left')
                            user32.mouse_event(0x0004,0,0,0,0)
                        time.sleep(.4)
                        check(hwnd)
                        screenshot(sct, rect, DEBUG/'aim_after.png')
                        print('AIM TEST FINISHED. No swing attempted.')
                        print('Compare wgt_debug\\aim_before.png and aim_after.png.')
                        print('Did the aim/yardage change?')
                        break
                prev6 = f6 if focus else False
                prev7 = f7 if focus else False
                time.sleep(.045)
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        print('Emergency stop')
    except Exception as error:
        print('Stopped:', error)
    finally:
        pyautogui.mouseUp(button='left')
        user32.mouse_event(0x0004,0,0,0,0)
        print('AIM CONTROLLER CLOSED')

if __name__ == '__main__':
    main()
