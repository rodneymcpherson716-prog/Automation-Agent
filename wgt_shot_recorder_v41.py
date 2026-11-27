"""WGT Shot Stats Recorder V41 (Windows)
Passive recorder: does not click, drag, resize, aim, or swing.
Requires: pip install mss opencv-python numpy pytesseract
Requires Tesseract OCR installed. Leave WGT Shot Data ON and panel visible.
Ctrl+C or F9 ends recording. Files saved beside this script in wgt_shot_logs.
"""
import csv
import ctypes
import ctypes.wintypes
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np
import pytesseract

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'wgt_shot_logs'
CSV = OUT / 'shots.csv'
FIELDNAMES = ['timestamp', 'player', 'carry_yd', 'roll_yd', 'total_yd',
              'power_pct', 'spin_pct', 'apex_ft', 'image_file']
# Coordinates measured from the user's screenshot (1257x718 WGT window).
# Scaled to the actual window size; if the HUD changes, recapture calibration.
BASE_W, BASE_H = 1257, 718
CROP = (955, 115, 1160, 236)
VK_F9 = 0x78
user32 = ctypes.windll.user32


def setup():
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try: user32.SetProcessDPIAware()
        except Exception: pass
    for path in (r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                 r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'):
        if Path(path).is_file():
            pytesseract.pytesseract.tesseract_cmd = path
            break
    pytesseract.get_tesseract_version()


def find_wgt():
    found = []
    cbtype = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, unused):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buff = ctypes.create_unicode_buffer(n+1)
                user32.GetWindowTextW(hwnd, buff, n+1)
                if 'wgt golf' in buff.value.lower():
                    rect = ctypes.wintypes.RECT()
                    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                        w, h = rect.right-rect.left, rect.bottom-rect.top
                        if w >= 600 and h >= 400:
                            found.append((hwnd, {'left':rect.left, 'top':rect.top,
                                                 'width':w, 'height':h}))
        return True
    user32.EnumWindows(cbtype(cb), 0)
    return found[0] if found else (None, None)


def crop_stats(sct, rect):
    # Keep the WGT outer window at a stable size, preferably 1366x768.
    x1,y1,x2,y2 = CROP
    x1 = rect['left'] + round(x1*rect['width']/BASE_W)
    y1 = rect['top'] + round(y1*rect['height']/BASE_H)
    x2 = rect['left'] + round(x2*rect['width']/BASE_W)
    y2 = rect['top'] + round(y2*rect['height']/BASE_H)
    region = {'left':x1, 'top':y1, 'width':max(25,x2-x1),'height':max(25,y2-y1)}
    img = np.array(sct.grab(region))[:,:,:3].copy()
    return img


def parse_stats(text):
    t = text.upper().replace(',', '.')
    def value(label, unit):
        pat = rf'\b{label}\s*[:;|]?\s*([0-9]{{1,4}}(?:\.[0-9]{{1,2}})?)\s*(?:{unit})?'
        match = re.search(pat, t)
        return float(match.group(1)) if match else None
    carry = value('CARRY', 'YDS?')
    roll = value('ROLL', 'YDS?')
    power = value('POWER', '%')
    spin = value('SPIN', '%')
    apex = value('APEX', 'FT')
    if None in (carry, roll, power, spin, apex):
        return None
    if not (0 <= carry <= 600 and 0 <= roll <= 200 and
            0 <= power <= 110 and 0 <= spin <= 100 and 0 <= apex <= 500):
        return None
    return {'carry_yd':round(carry,2), 'roll_yd':round(roll,2),
            'total_yd':round(carry+roll,2), 'power_pct':round(power,2),
            'spin_pct':round(spin,2), 'apex_ft':round(apex,2)}


def read_stats(img):
    """Read only right-hand numeric columns of the five Shot Stats rows.

    Calibrated using the user's actual 210x122 diagnostic crop, not
    the labels + values OCR that repeatedly failed in V39/V40.
    """
    h, w = img.shape[:2]
    # Fractions derived from the image of the real in-game panel.
    row_bands = ((.238, .369), (.361, .492), (.475, .615),
                 (.598, .738), (.721, .877))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    values = []
    raw = []
    for row_index, (lo, hi) in enumerate(row_bands):
        y0, y1 = round(h * lo), round(h * hi)
        # Avoid panel labels on the left and background to the right.
        x0, x1 = round(w * .405), round(w * .753)
        strip = gray[y0:y1, x0:x1]
        if strip.size == 0:
            return None, 'empty value region'
        enlarged = cv2.resize(strip, None, fx=8, fy=8,
                              interpolation=cv2.INTER_CUBIC)
        reading = pytesseract.image_to_string(
            enlarged, config='--psm 7 '
            '-c tessedit_char_whitelist=0123456789.%').strip()
        raw.append(reading)
        match = re.search(r'\d{1,4}(?:\.\d{1,2})?', reading)
        if not match:
            return None, 'row ' + str(row_index) + ' unreadable: ' + str(raw)
        values.append(float(match.group(0)))

    carry, roll, power, spin, apex = values
    if not (0 <= carry <= 600 and 0 <= roll <= 200
            and 0 <= power <= 110 and 0 <= spin <= 100
            and 0 <= apex <= 500):
        return None, 'numbers outside expected bounds: ' + str(values)
    result = {'carry_yd': round(carry, 2), 'roll_yd': round(roll, 2),
              'total_yd': round(carry + roll, 2),
              'power_pct': round(power, 2), 'spin_pct': round(spin, 2),
              'apex_ft': round(apex, 2)}
    return result, 'NUMERIC ROWS: ' + str(values)


def save(record, img):
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    imgfile = f'shot_{stamp}.png'
    cv2.imwrite(str(OUT/imgfile), img)
    row = {'timestamp':datetime.now().astimezone().isoformat(timespec='seconds'),
           'player':'Tigerwould52', **record, 'image_file':imgfile}
    new = not CSV.exists()
    with CSV.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new: writer.writeheader()
        writer.writerow(row)
    with (OUT/'latest_shot.json').open('w',encoding='utf-8') as handle:
        json.dump(row, handle, indent=2)
    print(f"SAVED | carry {record['carry_yd']:.2f} yd | roll {record['roll_yd']:.2f} yd | "
          f"power {record['power_pct']:.0f}% | spin {record['spin_pct']:.0f}% | "
          f"apex {record['apex_ft']:.2f} ft", flush=True)


def main():
    setup()
    OUT.mkdir(exist_ok=True)
    print('WGT SHOT RECORDER V41 — PASSIVE, NO MOUSE MOVEMENT')
    print('Keep WGT Shot Data ON with Shot Stats visible.')
    print('Saves to:', CSV)
    print('F9 or Ctrl+C = stop. Open WGT before starting.\n')
    previous = None
    candidate = None
    repeats = 0
    last_notice = 0
    unreadable_since = None
    with mss.MSS() as sct:
        while True:
            if user32.GetAsyncKeyState(VK_F9) & 0x8000:
                print('Stopped with F9.')
                break
            hwnd, rect = find_wgt()
            if rect is None:
                if time.monotonic()-last_notice > 8:
                    print('Waiting for visible WGT Golf window...',flush=True)
                    last_notice=time.monotonic()
                time.sleep(1)
                continue
            try:
                img = crop_stats(sct, rect)
                stats, raw = read_stats(img)
            except Exception as e:
                if time.monotonic()-last_notice > 8:
                    print('Capture/OCR failed:',e,flush=True)
                    last_notice=time.monotonic()
                time.sleep(1.5)
                continue
            if stats is None:
                if unreadable_since is None:
                    unreadable_since = time.monotonic()
                if time.monotonic()-last_notice > 25:
                    print('Waiting for readable stats; saved diagnostic crop. No guessed data.',flush=True)
                    cv2.imwrite(str(OUT/'latest_unreadable.png'), img)
                    last_notice=time.monotonic()
                candidate=None; repeats=0
            else:
                unreadable_since = None
                signature = tuple(stats[k] for k in ('carry_yd','roll_yd','power_pct','spin_pct','apex_ft'))
                if signature == candidate:
                    repeats += 1
                else:
                    candidate=signature; repeats=1
                if repeats >= 2 and signature != previous:
                    save(stats, img)
                    previous=signature
            time.sleep(1.25)
    print('Recorder closed.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Data saved to wgt_shot_logs.')
    except Exception as error:
        print('Recorder stopped:',error)
