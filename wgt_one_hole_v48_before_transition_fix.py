"""One WGT hole, including putts, with unchanged aim/camera and F9 stop.

Reuses the saved manual stroke's meter press and drag cadence, the saved
accuracy calibration, MSS capture, and the existing HUD/meter colour readers.
Distance-to-power is a baseline estimate, not an empirically fitted golf model.
--check and --observe never send mouse input. See WGT_CONTROLLED_TEST.md for limits.
"""
import argparse
from contextlib import contextmanager
from collections import deque, OrderedDict
import ctypes
import ctypes.wintypes as wt
from dataclasses import replace, asdict
import json
import hashlib
import math
from pathlib import Path
import re
import sys
import threading
import time
import traceback

import cv2
import mss
import numpy as np
import pytesseract

from wgt_core import (AccuracyTracker, Distance, StrokeReference, TransitionTracker,
                      complete_text, consensus, detect_meter, green_front,
                      meter_rect, parse_distance, parse_shot, plan_power,
                      state_signature, yellow_power, resolve_target_distance)
from wgt_observation import MotionEvidence, PowerCalibration, Session, read_conditions, read_stats_patch, unit_from_glyphs, visible_hole_lock

ROOT = Path(__file__).resolve().parent
DEBUG = ROOT/'wgt_debug'
RECORD = ROOT/'wgt_calibration/last_manual_stroke.json'
CALIBRATION = ROOT/'wgt_accuracy.json'
SWING_TEMPLATE = ROOT/'swing_button.png'
PLAYER = 'tigerwould52'
F9 = 0x78
DOWN, UP, WHEEL = 0x0002, 0x0004, 0x0800
MOVE, ABSOLUTE, VIRTUAL = 0x0001, 0x8000, 0x4000
user32 = ctypes.WinDLL('user32', use_last_error=True) if sys.platform == 'win32' else None
DIAGNOSTICS = None
POWER_MODEL = None
SUPPORTED_SHOT_MODES=('FULL','CHIP','PITCH','PUTT')
OCR_CACHE=OrderedDict()


def ocr_text(image,config):
    key=(image.shape,str(image.dtype),hashlib.sha256(image.tobytes()).digest(),config)
    if key in OCR_CACHE:
        OCR_CACHE.move_to_end(key)
        return OCR_CACHE[key]
    try:
        text=pytesseract.image_to_string(image,config=config,timeout=5).upper()
    except RuntimeError as exc:
        if 'timeout' not in str(exc).lower():
            raise
        if DIAGNOSTICS:
            DIAGNOSTICS.event('ocr_timeout',config=config)
        return ''  # Abstain; other variants/frames must still establish consensus.
    OCR_CACHE[key]=text
    while len(OCR_CACHE)>192:
        OCR_CACHE.popitem(last=False)
    return text


class ConsoleLog:
    def __init__(self,original,session):
        self.original,self.session=original,session
    def write(self,text):
        self.original.write(text)
        self.session.write(text)
    def flush(self):
        self.original.flush()


def diagnostic_state(name,state,frame=None):
    if DIAGNOSTICS:
        DIAGNOSTICS.state(name,state,frame)


def shot_stats(frame):
    # The working V41 recorder's 1257x718 crop, normalized to this window.
    return read_stats_patch(crop(frame,(955/1257,115/718,1160/1257,236/718)),ocr_variants)


class ShotObserver:
    """Continuous MSS video/motion observation while the main thread runs OCR.

    MSS owns thread-local handles, so this thread creates its own MSS context.
    JPEG encoding/writes are deferred until after the accuracy input phase.
    """
    def __init__(self,rect,before):
        self.rect=rect
        self.evidence=MotionEvidence(before)
        self.closed=threading.Event()
        self.armed=threading.Event()
        self.thread=None
        self.frames=deque(maxlen=90)
        self.early_frames=[]
        self.error=None
        self.last_report={}

    def start(self):
        self.thread=threading.Thread(target=self._watch,name='WGT-MSS-observer',daemon=True)
        self.thread.start()

    def arm(self):
        self.armed.set()

    def _watch(self):
        try:
            with mss.MSS() as sct:
                while not self.closed.wait(.065):
                    if not self.armed.is_set():
                        continue
                    frame=capture(sct,self.rect)
                    self.evidence.update(frame)
                    self.last_report=self.evidence.report()
                    sample=(time.perf_counter(),cv2.resize(frame,(644,364)))
                    self.frames.append(sample)
                    if len(self.early_frames)<60:
                        self.early_frames.append(sample)
        except Exception as exc:
            self.error=str(exc)

    def report(self):
        return dict(self.last_report or self.evidence.report(),capture_error=self.error)

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=2)

    def save(self,shot):
        if not DIAGNOSTICS:
            return
        report=self.report()
        DIAGNOSTICS.state(f'shot_{shot:02d}_motion',report)
        early_times={v[0] for v in self.early_frames}
        frames=self.early_frames+[v for v in self.frames if v[0] not in early_times]
        if frames:
            # Contact sheets provide automatic evidence without huge videos.
            chosen=np.linspace(0,len(frames)-1,min(16,len(frames)),dtype=int)
            tiles=[]
            for i in chosen:
                elapsed,frame=frames[i]
                tile=cv2.resize(frame,(322,182))
                cv2.putText(tile,f'+{elapsed-frames[0][0]:.2f}s',(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                tiles.append(tile)
            while len(tiles)%4:
                tiles.append(np.zeros_like(tiles[0]))
            DIAGNOSTICS.image(f'shot_{shot:02d}_motion_contact_sheet',
                              cv2.vconcat([cv2.hconcat(tiles[j:j+4]) for j in range(0,len(tiles),4)]))


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', wt.LONG), ('dy', wt.LONG), ('mouseData', wt.DWORD),
                ('dwFlags', wt.DWORD), ('time', wt.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', wt.WORD), ('wScan', wt.WORD), ('dwFlags', wt.DWORD),
                ('time', wt.DWORD), ('dwExtraInfo', ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg', wt.DWORD), ('wParamL', wt.WORD), ('wParamH', wt.WORD)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [('type', wt.DWORD), ('data', INPUT_UNION)]


def setup():
    if user32 is None:
        raise RuntimeError('Live WGT controller requires Windows')
    user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wt.UINT
    user32.GetForegroundWindow.restype = wt.HWND
    user32.SetForegroundWindow.argtypes=[wt.HWND]
    user32.SetForegroundWindow.restype=wt.BOOL
    user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
    user32.GetWindowRect.restype = wt.BOOL
    user32.GetWindowTextLengthW.argtypes = [wt.HWND]
    user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wt.HWND]
    user32.IsIconic.argtypes = [wt.HWND]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
    user32.GetCursorPos.restype = wt.BOOL
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = wt.SHORT
    user32.SystemParametersInfoW.argtypes=[wt.UINT,wt.UINT,ctypes.c_void_p,wt.UINT]
    user32.SystemParametersInfoW.restype=wt.BOOL
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()
    for path in (r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                 r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'):
        if Path(path).is_file():
            pytesseract.pytesseract.tesseract_cmd = path
            break


def key_down(key):
    return bool(user32.GetAsyncKeyState(key) & 0x8000)


def clicklock_state():
    enabled=wt.BOOL()
    if not user32.SystemParametersInfoW(0x101E,0,ctypes.byref(enabled),0):
        raise OSError(ctypes.get_last_error(),'Cannot verify Windows ClickLock setting')
    return bool(enabled.value)


def set_clicklock(enabled):
    # pvParam is the BOOL value for SET, a pointer to BOOL for GET. Flags 0
    # deliberately avoid writing this temporary change to the user profile.
    if not user32.SystemParametersInfoW(0x101F,0,ctypes.c_void_p(int(enabled)),0):
        raise OSError(ctypes.get_last_error(),'Cannot set temporary Windows ClickLock state')


@contextmanager
def clicklock_disabled():
    original=clicklock_state()
    if DIAGNOSTICS:
        DIAGNOSTICS.event('clicklock_original',enabled=original)
    try:
        if original:
            set_clicklock(False)
            if clicklock_state():
                raise RuntimeError('Windows ClickLock remained enabled; no bot stroke')
            print('ClickLock temporarily disabled for bot input; restored on exit.',flush=True)
        yield
    finally:
        if original:
            set_clicklock(True)
            if not clicklock_state():
                raise RuntimeError('Could not verify restoration of Windows ClickLock')
            print('Original Windows ClickLock setting restored.',flush=True)
            if DIAGNOSTICS:
                DIAGNOSTICS.event('clicklock_restored',enabled=True)


def send(flags, x=0, y=0, data=0):
    event = INPUT(type=0, data=INPUT_UNION(mi=MOUSEINPUT(x, y, data, flags, 0, 0)))
    if user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
        raise OSError(ctypes.get_last_error(), 'Windows rejected SendInput')


def mouse_up():
    # Reuse both release paths from the working legacy controller. SendInput
    # success alone did not clear the held state in the observed live attempt.
    try:
        send(UP)
    except Exception:
        pass
    finally:
        user32.mouse_event(UP, 0, 0, 0, 0)


def cursor():
    point = wt.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise RuntimeError('Cannot verify cursor position')
    return point.x, point.y


def move_to(x, y):
    vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
    vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    if min(vw, vh) < 2 or not vx <= x < vx+vw or not vy <= y < vy+vh:
        raise RuntimeError('Requested pointer position is outside the virtual desktop')
    ax = round((x-vx)*65535/(vw-1))
    ay = round((y-vy)*65535/(vh-1))
    send(MOVE | ABSOLUTE | VIRTUAL, ax, ay)
    actual = cursor()
    if abs(actual[0]-x) > 3 or abs(actual[1]-y) > 3:
        raise RuntimeError(f'Pointer was clipped: requested {(x,y)}, observed {actual}')


def find_wgt():
    found = []
    callback_type = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def visit(hwnd, unused):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(length+1)
            user32.GetWindowTextW(hwnd, title, length+1)
            if title.value.strip().lower() == 'wgt golf':
                found.append(hwnd)
        return True
    user32.EnumWindows(callback_type(visit), 0)
    if len(found) != 1:
        raise RuntimeError(f'Expected exactly one visible WGT Golf window; found {len(found)}')
    return found[0]


def rect_of(hwnd):
    r = wt.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError('Cannot read WGT window geometry')
    result = dict(left=r.left, top=r.top, width=r.right-r.left, height=r.bottom-r.top)
    if result['width'] < 650 or result['height'] < 440:
        raise RuntimeError('WGT window is too small for verified input')
    return result


def activate_wgt(hwnd):
    """Raise the existing game only; no click, key, aim or camera input."""
    if key_down(F9):
        raise KeyboardInterrupt('F9 emergency stop')
    if user32.IsIconic(hwnd):
        raise RuntimeError('WGT is minimized; foreground input cannot be verified')
    if user32.GetForegroundWindow()!=hwnd:
        user32.SetForegroundWindow(hwnd)
    end=time.perf_counter()+2.
    while time.perf_counter()<end:
        if key_down(F9):
            raise KeyboardInterrupt('F9 emergency stop')
        if user32.GetForegroundWindow()==hwnd:
            return
        time.sleep(.005)
    raise RuntimeError('Windows did not give WGT foreground focus; no game input sent')


class Safety:
    """F9 is latched across OCR calls; release continues even if OCR blocks."""
    def __init__(self, hwnd, rect):
        self.hwnd, self.rect = hwnd, rect
        self.stopped = threading.Event()
        self.closed = threading.Event()
        self.reason = ''
        self.lock = threading.RLock()
        self.pressed = False
        self.thread = None
        self.deadline = None

    def abort(self, reason):
        with self.lock:
            self.reason = reason
            self.stopped.set()
            if self.pressed:
                mouse_up()
                self.pressed = False

    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.abort('Controlled live-test time limit reached')
        if key_down(F9):
            self.abort('F9 emergency stop')
        if self.stopped.is_set():
            raise KeyboardInterrupt(self.reason)
        if user32.GetForegroundWindow() != self.hwnd or user32.IsIconic(self.hwnd):
            self.abort('WGT lost focus')
            raise RuntimeError(self.reason)
        if rect_of(self.hwnd) != self.rect:
            self.abort('WGT moved or resized')
            raise RuntimeError(self.reason)

    def _watch(self):
        while not self.closed.wait(.005):
            try:
                self.check()
            except BaseException as exc:
                self.abort(str(exc))
                return

    def start(self):
        self.check()
        self.thread = threading.Thread(target=self._watch, name='WGT-F9-stop', daemon=True)
        self.thread.start()

    def press(self):
        with self.lock:
            self.check()
            if key_down(1):
                raise RuntimeError('Left button already held before bot press; no additional mouse-down sent')
            send(DOWN)
            self.pressed = True

    def release(self):
        with self.lock:
            try:
                mouse_up()
            finally:
                self.pressed = False
        # Input accepted by Windows is not proof that its button state cleared.
        # Observe the actual state before advancing to the accuracy phase.
        end=time.perf_counter()+.030
        while time.perf_counter()<end:
            if not key_down(1):
                return
            if self.stopped.is_set():
                raise KeyboardInterrupt(self.reason)
            time.sleep(.001)
        raise RuntimeError('Left button remains held after both native release methods; downswing unverified')

    def wait(self, seconds):
        end = time.perf_counter()+seconds
        while time.perf_counter() < end:
            self.check()
            time.sleep(min(.005, max(0, end-time.perf_counter())))

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=.5)
        if self.pressed:
            self.release()


def capture(sct, rect):
    return np.asarray(sct.grab(rect))[:, :, :3].copy()


def crop(image, bounds):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bounds
    return image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]


def ocr_variants(patch, modes=(6, 7), scale=4):
    if patch is None or patch.size == 0:
        return []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    texts = []
    for picture in (big, binary):
        for psm in modes:
            texts.append(ocr_text(picture,f'--psm {psm}'))
    return texts


def read_hole_glyphs(frame,candidates=()):
    patch=crop(frame,(.055,.102,.090,.148))
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    mask=cv2.threshold(gray,130,255,cv2.THRESH_BINARY_INV)[1]
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    glyphs=sorted([s for s in stats[1:] if s[3]>=.48*patch.shape[0]
                   and 2<=s[2]<=.55*patch.shape[1] and s[4]>=20],key=lambda s:s[0])
    if not 1<=len(glyphs)<=2:
        return None
    votes=[]
    for binary in (False,True):
        digits=[]
        for x,y,w,h,_ in glyphs:
            g=(255-mask[y:y+h,x:x+w]) if binary else gray[y:y+h,x:x+w]
            g=cv2.copyMakeBorder(g,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
            g=cv2.resize(g,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
            text=ocr_text(g,'--psm 10 -c tessedit_char_whitelist=0123456789').strip()
            if not re.fullmatch('[0-9]',text):
                break
            digits.append(text)
        if len(digits)==len(glyphs) and 1<=int(''.join(digits))<=18:
            votes.append(int(''.join(digits)))
    value=consensus(votes)
    if value is not None:
        return value
    # Whole-word OCR can merge the two narrow 1s. A visible, independently
    # segmented digit count can disambiguate its existing full-number reading.
    return consensus([v for v in candidates if len(str(v))==len(glyphs)],minimum=1)


def read_hole_hud_distance(frame):
    """Independent numeric/unit rows; no club yardage or hidden-value inference."""
    votes=[]
    unit_patch=crop(frame,(.158,.066,.175,.095))
    number_patch=crop(frame,(.135,.066,.159,.095))
    for patch,whitelist in ((unit_patch,'YDSFT'),(number_patch,'0123456789.')):
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        large=cv2.resize(gray,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
        binary=cv2.threshold(large,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        values=[]
        for image in (large,binary):
            for psm in (6,7):
                text=ocr_text(image,f'--psm {psm} -c tessedit_char_whitelist={whitelist}').strip()
                if whitelist=='YDSFT':
                    if text in ('YD','YDS','FT'):
                        values.append('ft' if text=='FT' else 'yd')
                elif re.fullmatch(r'\d{1,3}(?:\.\d+)?',text) and 0<float(text)<=650:
                    values.append(float(text))
        votes.append(consensus(values))
    unit,value=votes
    return Distance(value,unit) if unit is not None and value is not None else None


def find_distance_sign(frame):
    # Reuse wgt_putt_read's white top-edge locator, extending its size range
    # to the 60px tee sign. Cyan-only components merge into the blue sky.
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # A cloud can join a dim top border and widen it beyond the actual badge.
    # Try the bright border first; retain dim passes for translucent signs.
    for level,saturation in ((210,65),(175,65),(140,100)):
        mask = cv2.inRange(hsv,(0,0,level),(179,saturation,255))
        mask[:int(.09*h)]=0;mask[int(.56*h):]=0
        mask[:,:int(.24*w)]=0;mask[:,int(.76*w):]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1,9),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,max(25,int(.025*w))),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        choices=[]
        for x,y,bw,bh,count in stats[1:]:
            if not (.035*w<bw<.21*w and bh<6 and .09*h<y<.48*h):
                continue
            width=int(bw)+2
            left=int(x)-1
            expected_bottom=int(y+width*.76)
            # Translucency varies across the sign. A dimmer bottom border is
            # accepted only with the top edge, blue interior and correct size.
            bottom_level=175 if level>=175 else 105
            white=cv2.inRange(hsv,(0,0,bottom_level),(179,65 if level>=175 else 100,255))
            white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((1,9),np.uint8))
            lo,hi=int(y+width*.65),min(h,int(y+width*.9))
            rows=(white[lo:hi,int(x):int(x+bw)]>0).sum(axis=1)
            if not len(rows) or rows.max()<.65*bw:
                continue
            bottom=lo+int(rows.argmax())
            # Text strokes also form horizontal white runs. A real badge has
            # continuous side borders; an internal digit row does not.
            side_left=white[int(y)+3:bottom-2,max(0,left-2):left+4]
            side_right=white[int(y)+3:bottom-2,left+width-4:left+width+2]
            if (not side_left.size or not side_right.size
                    or (side_left>0).any(axis=1).mean()<.55
                    or (side_right>0).any(axis=1).mean()<.55):
                continue
            inside=hsv[int(y+width*.08):bottom-2,left+4:left+width-4]
            blue_ratio=((inside[:,:,0]>80)&(inside[:,:,0]<117)&(inside[:,:,1]>25)).mean()
            if blue_ratio<.40:
                continue
            score=blue_ratio-abs(x+bw/2-w/2)/w-abs(bottom-expected_bottom)/w
            choices.append((score,left,int(y),width,bottom-int(y)+1))
        if choices:
            return max(choices)[1:]
    return None


def read_sign_distance(frame, sign):
    if sign is None:
        return None
    x, y, w, h = sign
    numeric = frame[y+int(.06*w):y+int(.39*w),x+int(.12*w):x+int(.74*w)]
    unit_patch = frame[y+int(.12*w):y+int(.40*w),x+int(.72*w):x+int(.98*w)]
    units = []
    for t in ocr_variants(unit_patch,modes=(6,7),scale=8):
        clean = re.sub('[^A-Z0-9]','',t)
        clean=re.sub(r'^[0-9O]+','',clean)
        if clean in ('YDS','YD','YOS','VDS','VOS','YD5'):
            units.append('yd')
        elif clean == 'FT':
            units.append('ft')
    unit = consensus(units)
    if unit is None:
        wider=frame[y+int(.12*w):y+int(.40*w),x+int(.66*w):x+int(.98*w)]
        for t in ocr_variants(wider,modes=(6,7),scale=8):
            clean=re.sub(r'^[0-9O]+','',re.sub('[^A-Z0-9]','',t))
            if clean in ('YDS','YD','YOS','VDS','VOS','YD5'):
                units.append('yd')
            elif clean=='FT':
                units.append('ft')
        unit=consensus(units)
    glyph_unit_used=False
    if unit is None:
        letter_votes=[]
        for left in (.60,.62):
            patch=frame[y+int(.12*w):y+int(.40*w),x+int(left*w):x+int(.98*w)]
            gray_unit=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            large=cv2.resize(gray_unit,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
            for mode in (6,7,13):
                text=ocr_text(large,f'--psm {mode} -c tessedit_char_whitelist=YDSFT').strip()
                if text in ('YDS','YD','FT'):
                    letter_votes.append('ft' if text=='FT' else 'yd')
        unit=consensus(letter_votes)
    if unit is None:
        glyph=unit_from_glyphs(frame[y+int(.14*w):y+int(.40*w),x+int(.69*w):x+int(.99*w)],w)
        if glyph in ('YDS','FT'):
            unit='yd' if glyph=='YDS' else 'ft'
            glyph_unit_used=True
    if unit is None:
        return None
    # The number crop excludes the tiny unit. Unlike V48, its digits cannot
    # absorb an S as 5/0. The independently read unit remains mandatory.
    gray = cv2.cvtColor(numeric,cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(big,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    white = cv2.inRange(numeric,(175,175,175),(255,255,255))
    white = cv2.resize(white,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
    adaptive=cv2.adaptiveThreshold(big,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,-3)
    values = []
    for picture in (big,binary,white,adaptive):
        for psm in (6,7):
            text = ocr_text(picture,f'--psm {psm} -c tessedit_char_whitelist=0123456789.').strip()
            if re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',text):
                values.append(float(text))
    # Brightness thresholding can erase faded trailing digits (200 -> 2).
    # Isolate neutral-coloured text from the teal background and include the
    # complete final large glyph, while keeping the small unit in its own crop.
    wide=frame[y+int(.06*w):y+int(.40*w),x+int(.12*w):x+int(.79*w)]
    neutral=cv2.inRange(cv2.cvtColor(wide,cv2.COLOR_BGR2HSV),(0,0,105),(179,55,255))
    neutral_votes=[]
    for text in ocr_variants(neutral,modes=(6,7,13),scale=6):
        text=text.strip('():|[] .\n\t')
        if re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',text):
            neutral_votes.append(float(text))
    neutral_value=consensus(neutral_votes,minimum=3)
    value = consensus(values)
    if neutral_value is not None:
        value=neutral_value
    elif glyph_unit_used:
        # Faded units indicate the same transparency can erase trailing
        # digits. Require the complete colour-isolated number in this case.
        return None
    return parse_distance(f'{value} {unit.upper()}') if value is not None else None


def shot_state(frame, conditions=True):
    sign = find_distance_sign(frame)
    distance = read_sign_distance(frame, sign)
    top = ' '.join(ocr_variants(crop(frame, (.055,.103,.090,.145)), modes=(7,)))
    par_patch = crop(frame, (.087,.126,.109,.145))
    par_gray = cv2.cvtColor(par_patch,cv2.COLOR_BGR2GRAY)
    par_large = cv2.resize(par_gray,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
    # PAR is low-contrast grey on white, unlike the darker hole number beside
    # it. A joint Otsu threshold erased these tiny letters in both saved HUDs.
    par_texts = [ocr_text(cv2.threshold(par_large,level,255,cv2.THRESH_BINARY)[1],
                 '--psm 7') for level in (185,210,220,230,240)]
    par_text = ' '.join(par_texts)
    hole_values = [int(m[1]) for m in re.finditer(r'\b(\d{1,2})\b', top) if 1 <= int(m[1]) <= 18]
    hole = consensus(hole_values)
    if hole is None:
        hole=read_hole_glyphs(frame,hole_values)
    par = consensus([int(m[1]) if (m:=re.search(r'PAR\s*([3-6])',re.sub(r'\bPAR\s*S\b','PAR5',t)))
                     else None for t in par_texts])
    lower = ocr_variants(crop(frame, (.012,.940,.056,.989)), modes=(7,))
    modes = []
    for t in lower:
        if re.search(r'\bPUTT(?:ER)?\b', t):
            modes.append('PUTT')
        elif re.search(r'\bFULL\b', t):
            modes.append('FULL')
        elif (match:=re.search(r'\b(PITCH|CHIP|PUNCH|FLOP)\b',t)):
            modes.append(match[1])
    mode = consensus(modes)
    # Current club's range always has its own explicit unit, including putters.
    club_texts = ocr_variants(crop(frame, (.111,.936,.172,.991)), modes=(6,7), scale=5)
    # Keep the club-name column out of the range. OCR line ordering can put
    # 'IRONS' between 187 and YDS, so read this vertical column on its own.
    range_texts = ocr_variants(crop(frame, (.140,.938,.171,.991)), modes=(6,), scale=6)
    range_texts = [re.sub(r'(?:YOS|VOS|YD5)\b','YDS',t) for t in range_texts]
    club = consensus([parse_distance(t) for t in range_texts])
    number_texts=[];unit_texts=[]
    if club is None:
        club = consensus([parse_distance(t) for t in club_texts])
    if club is None:
        # WEDGES/WOODS can be placed between the value and YDS by OCR line
        # ordering. Read the two printed rows independently, with explicit units.
        number_texts=ocr_variants(crop(frame,(.122 if mode=='PUTT' else .137,.938,.172,.966)),modes=(7,),scale=7)
        unit_patch=crop(frame,(.122 if mode=='PUTT' else .145,.966,.172,.989))
        unit_texts=ocr_variants(unit_patch,modes=(6,7,13),scale=7)
        unit_texts+=ocr_variants(cv2.inRange(unit_patch,(100,100,100),(255,255,255)),modes=(6,7),scale=7)
        numbers=[];units=[]
        for t in number_texts:
            clean=re.sub('[^0-9.]','',t)
            if re.fullmatch(r'\d{1,3}(?:\.\d+)?',clean) and 1<=float(clean)<=400:
                numbers.append(float(clean))
        for t in unit_texts:
            clean=re.sub('[^A-Z]','',t)
            if clean in ('YDS','YD','YOS'):
                units.append('yd')
            elif clean=='FT':
                units.append('ft')
        number,unit=consensus(numbers),consensus(units)
        if number is not None and unit:
            club=Distance(number,unit)
    hole_distances=ocr_variants(crop(frame,(.135,.058,.178,.095)),modes=(6,7),scale=5)
    hole_distance=consensus([parse_distance(re.sub(r'(?:YOS|VOS|YD5|Y0S)\b','YDS',t)) for t in hole_distances])
    if hole_distance is None:
        hole_distance=read_hole_hud_distance(frame)
    right_texts = ocr_variants(crop(frame, (.830,.058,.984,.104)), modes=(6,))
    shot_texts = ocr_variants(crop(frame, (.898,.082,.934,.105)), modes=(7,), scale=6)
    shots = [parse_shot(re.sub(r'\bIST\s*SHOT','1ST SHOT',t), par) for t in right_texts+shot_texts]
    right = ' '.join(right_texts)
    player = PLAYER in re.sub(r'[^a-z0-9]', '', right.lower())
    state = dict(distance=distance.value if distance else None,
                unit=distance.unit if distance else None, putt=(mode=='PUTT') if mode else None,
                distance_source='floating_target',
                hole_distance=hole_distance.value if hole_distance else None,
                hole_distance_unit=hole_distance.unit if hole_distance else None,
                mode=mode, club_range=club.value if club else None,
                club_unit=club.unit if club else None, shot=consensus(shots),
                hole=hole, par=par, complete=False, player=player, sign=sign,
                raw=dict(mode=lower, club=club_texts,club_range_number=number_texts,club_range_unit=unit_texts,
                         hole_distance=hole_distances,right=right_texts,hole=top,par=par_text,shot=shot_texts))
    if conditions:
        state.update(read_conditions(frame,sign,ocr_variants,crop))
        if mode=='PUTT':
            state['club_name']='PUTTER'
    lock=visible_hole_lock(frame,sign) if mode in ('PUTT','CHIP','PITCH') else None
    state.update(resolve_target_distance(distance,hole_distance,linked_to_hole=lock is not None))
    state['target_hole_link']=lock
    return state


def swing_button(frame, rect):
    """Use the existing template + circle/text check only as a ready indicator.

    The actual press follows the user's recorded meter anchor, never the flag.
    A plausible but unlabelled circle is no longer accepted as confidence .84.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    area = gray[int(.55*h):int(.88*h), int(.80*w):int(.99*w)]
    if SWING_TEMPLATE.is_file():
        template = cv2.imread(str(SWING_TEMPLATE), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            for scale in (.70,.85,1.,1.15,1.30):
                t = cv2.resize(template, None, fx=scale, fy=scale)
                if t.shape[0] >= area.shape[0] or t.shape[1] >= area.shape[1]:
                    continue
                _, score, _, xy = cv2.minMaxLoc(cv2.matchTemplate(area, t, cv2.TM_CCOEFF_NORMED))
                if score >= .83:
                    return (int(.80*w)+xy[0]+t.shape[1]//2,
                            int(.55*h)+xy[1]+t.shape[0]//2)
    circles = cv2.HoughCircles(cv2.GaussianBlur(area,(7,7),1.4), cv2.HOUGH_GRADIENT,
                              1.2, minDist=35, param1=115, param2=36,
                              minRadius=max(17,int(.025*w)), maxRadius=int(.058*w))
    if circles is not None:
        for cx, cy, radius in circles[0]:
            x, y, r = int(cx+.80*w), int(cy+.55*h), int(radius)
            label = frame[max(0,y-int(.3*r)):y+int(.5*r), x-int(.75*r):x+int(.75*r)]
            if any('SWING' in t for t in ocr_variants(label, modes=(7,))):
                return x, y
    return None


def saved_target(rect, meter):
    data = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    fraction = float(data['accuracy_x_fraction'])
    if not math.isfinite(fraction) or not .25 < fraction < .90:
        raise ValueError('Invalid saved accuracy calibration')
    return round(fraction*rect['width'])-(meter['left']-rect['left'])


def ready_meter(frame, rect):
    r = meter_rect(rect)
    x, y = r['left']-rect['left'], r['top']-rect['top']
    band = frame[y:y+r['height'], x:x+r['width']]
    return detect_meter(band, saved_target(rect, r))


def inspect_screen(frame,rect):
    """Read all required screen controls from exactly one captured frame."""
    state=shot_state(frame)
    button=swing_button(frame,rect)
    meter=ready_meter(frame,rect)
    region=meter_rect(rect)
    x,y=region['left']-rect['left'],region['top']-rect['top']
    details=None
    if meter:
        details=dict(**asdict(meter),capture=region,
                     full_power_x_window=x+meter.origin,neutral_x_window=x+meter.zero,
                     accuracy_x_window=x+meter.target,
                     yellow_power=yellow_power(frame[y:y+region['height'],x:x+region['width']],meter),
                     green_front=green_front(frame[y:y+region['height'],x:x+region['width']],meter))
    checks=dict(distance=state['distance'] is not None and state['unit'] in ('yd','ft'),
                club=state.get('club_name') is not None and state['club_range'] is not None,
                wind=state.get('wind') is not None,
                swing_button=button is not None,
                meter=(meter is not None and meter.origin<meter.zero<meter.blue_left<meter.target<meter.blue_right))
    return dict(state=state,checks=checks,verified=all(checks.values()),rect=rect,
                swing_button_window=list(button) if button else None,
                swing_button_desktop=[rect['left']+button[0],rect['top']+button[1]] if button else None,
                meter=details,input_sent=False)


def detection_overlay(frame,rect,reading):
    image=frame.copy()
    button=reading['swing_button_window']
    if button:
        cv2.circle(image,tuple(button),round(.036*rect['width']),(0,255,255),2)
    region=meter_rect(rect)
    x,y=region['left']-rect['left'],region['top']-rect['top']
    cv2.rectangle(image,(x,y),(x+region['width']-1,y+region['height']-1),(255,0,255),1)
    meter=reading['meter']
    if meter:
        for key,colour in [('full_power_x_window',(0,255,255)),('neutral_x_window',(255,255,0)),('accuracy_x_window',(255,255,255))]:
            cv2.line(image,(meter[key],y),(meter[key],y+region['height']-1),colour,2)
    return image


def detections_agree(a,b):
    if not a['verified'] or not b['verified'] or state_signature(a['state'])!=state_signature(b['state']):
        return False
    if a['state']['club_name']!=b['state']['club_name']:
        return False
    wa,wb=a['state']['wind'],b['state']['wind']
    if (wa['low_mph'],wa['high_mph'])!=(wb['low_mph'],wb['high_mph']):
        return False
    if math.hypot(wa['screen_dx']-wb['screen_dx'],wa['screen_dy']-wb['screen_dy'])>.20:
        return False
    if math.dist(a['swing_button_window'],b['swing_button_window'])>3:
        return False
    return all(abs(a['meter'][k]-b['meter'][k])<=2 for k in ('origin','zero','blue_left','blue_right','target'))


def stable_state(safety, sct, rect):
    last = None
    for attempt in range(4):
        safety.check()
        frame = capture(sct, rect)
        state = shot_state(frame)
        diagnostic_state(f'ready_read_{time.time_ns()}',state,frame)
        safety.check()
        if (state['player'] and state['mode'] in SUPPORTED_SHOT_MODES
                and state['distance'] is not None and state['club_range'] is not None
                and state['unit'] == state['club_unit']
                and state['putt'] == (state['unit']=='ft')
                and state.get('elevation_ft') is not None and state.get('wind') is not None
                and state.get('club_name') is not None
                and (not state['putt'] or state.get('green_speed') is not None)
                and swing_button(frame, rect) is not None and ready_meter(frame, rect) is not None):
            if last == state_signature(state):
                return state, frame
            last = state_signature(state)
        else:
            last = None
        safety.wait(.12)
    raise RuntimeError('Two stable distance/unit/mode/club/ready readings were not verified; no shot')


def select_club(safety, sct, rect, state, frame):
    """Cycle only the bottom-left club/range control, verifying after each step."""
    target = state['distance']
    def suitable(s):
        ratio = target/s['club_range']
        return (.08 <= ratio <= .92) if s['putt'] else (.35 <= ratio <= 1.10)
    if suitable(state):
        return state, frame
    direction = 1
    known_direction=None
    boundary_tried=False
    for _ in range(14):
        previous=state['club_range']
        need_larger=target/previous > (.92 if state['putt'] else 1.10)
        safety.check()
        # Wheel over the club's printed range; never wheel over the 3D course.
        move_to(rect['left']+round(.14*rect['width']), rect['top']+round(.96*rect['height']))
        safety.check()
        send(WHEEL, data=(direction*120)&0xffffffff)
        safety.wait(.20)
        new, frame = stable_state(safety, sct, rect)
        if (new['mode'] != state['mode'] or new['unit'] != state['unit']
                or abs(new['distance']-target) > .1):
            raise RuntimeError('Shot mode or target changed during club selection; stopped')
        if suitable(new):
            return new, frame
        delta=new['club_range']-previous
        if delta:
            known_direction=direction if delta>0 else -direction
            direction=known_direction if need_larger else -known_direction
            boundary_tried=False
        else:
            if need_larger and known_direction==direction:
                # A verified unchanged range while scrolling toward larger
                # clubs establishes the maximum available for a long lay-up.
                return new,frame
            if boundary_tried:
                break
            direction=-direction
            boundary_tried=True
        state = new
    raise RuntimeError(f'No usable club/range verified for {target:g} {state["unit"]}; stopped')


def drag_swing(safety, sct, rect, reference, meter, required):
    """Recorded activation + live yellow-power feedback, with unconditional up."""
    if abs(rect['width']-reference.width)>10 or abs(rect['height']-reference.height)>10:
        raise RuntimeError('Window differs from saved stroke geometry; re-record before playing')
    x = rect['left']+round(reference.anchor[0]*rect['width'])
    y = rect['top']+round(reference.anchor[1]*rect['height'])
    r = meter_rect(rect)
    if not r['left']+meter.blue_left < x < r['left']+meter.blue_right:
        raise RuntimeError('Recorded swing press does not align with the visible meter')
    if not r['top'] <= y < r['top']+r['height']:
        raise RuntimeError('Recorded swing press is outside the visible meter')
    # Keep held movement inside the same window at the recorded x. No flag or
    # camera press is possible; do not cross onto another monitor while held.
    upper = rect['top']+max(28, round(.08*rect['height']))
    if y-upper < 50:
        raise RuntimeError('Not enough upward travel')
    safety.check()
    move_to(x, y)
    safety.wait(.08)
    started = time.perf_counter()
    best = 0.0
    last_power = None
    last_seen = started
    saw_increase = False
    reference_pixels = reference.pull*rect['height']
    cadence = min(.020, max(.004, reference.drag_seconds/reference_pixels))
    try:
        safety.press()
        while time.perf_counter()-started < max(9, reference.drag_seconds*1.6):
            safety.check()
            band = capture(sct, r)
            power = yellow_power(band, meter)
            if DIAGNOSTICS:
                DIAGNOSTICS.meter('yellow',band,power)
            now = time.perf_counter()
            if power is not None:
                last_seen = now
                best = max(best, power)
                if last_power is not None and power > last_power+.002:
                    saw_increase = True
                last_power = power
                if saw_increase and power >= required-.004:
                    if power > required+max(.025, required*.10):
                        raise RuntimeError(f'Power overshot {required:.1%}: observed {power:.1%}')
                    print(f'YELLOW POWER VERIFIED {power:.1%} (requested {required:.1%})', flush=True)
                    return power
            if now-last_seen > .65:
                raise RuntimeError('Yellow power feedback missing; no blind drag')
            remaining = max(0.01, required-(power or 0))
            step = 1 if remaining < .07 or required < .15 else 4
            if y-step < upper:
                raise RuntimeError(f'Cursor travel exhausted before power target; best {best:.1%}')
            y -= step
            move_to(x, y)
            safety.wait(cadence)
        raise RuntimeError(f'Power timeout; best observed {best:.1%}')
    finally:
        try:
            safety.release()
        except BaseException:
            print('MOUSE RELEASE REQUEST SENT; cleared state not verified',flush=True)
            raise
        print('LEFT MOUSE RELEASE VERIFIED', flush=True)


def accuracy_click(safety, sct, rect, meter, reference):
    # Start tracking immediately after release. No slow OCR or repositioning
    # before the return: the saved manual accuracy click stayed at release x/y.
    tracker = AccuracyTracker(meter)
    region = meter_rect(rect)
    started = time.perf_counter()
    timeout = min(6, max(3.5, reference.accuracy_delay*2))
    records = []
    last_band = None
    clicked = False
    try:
        while time.perf_counter()-started < timeout:
            safety.check()
            last_band = capture(sct, region)
            now = time.perf_counter()
            edge = green_front(last_band, meter)
            if DIAGNOSTICS:
                DIAGNOSTICS.meter('green',last_band,edge)
            records.append(dict(t=now-started, edge=edge))
            if tracker.update(now, edge):
                # Release in this nested finally even on F9 or click-hold error.
                try:
                    safety.press()
                    safety.wait(min(.025, max(.010, reference.click_seconds)))
                finally:
                    safety.release()
                clicked = True
                print(f'ACCURACY INPUT SENT: edge={edge}, target={meter.target}', flush=True)
                return True
            safety.wait(.002)
        raise RuntimeError('Moving green return never verified a target crossing; no speculative click')
    finally:
        # Disk writes happen after the time-critical loop, never before a click.
        DEBUG.mkdir(exist_ok=True)
        (DEBUG/'accuracy_trace.json').write_text(json.dumps(dict(target=meter.target, clicked=clicked,
                                                               samples=records), indent=2), encoding='utf-8')
        if last_band is not None:
            cv2.imwrite(str(DEBUG/'last_accuracy_meter.png'), last_band)


def wait_until_next(safety, sct, rect, before, timeout=90, observer=None):
    tracker = TransitionTracker(before)
    started = time.perf_counter()
    completion_count = 0
    while time.perf_counter()-started < timeout:
        safety.wait(.35)
        frame = capture(sct, rect)
        # Completion must be explicit and repeat on the central result panel.
        panel = ' '.join(ocr_variants(crop(frame, (.18,.16,.88,.88)), modes=(11,), scale=2))
        safety.check()
        completion_count = completion_count+1 if complete_text(panel) else 0
        if completion_count >= 2:
            report=observer.report() if observer else {}
            if observer and not (report.get('scene_motion_observed') or report.get('moving_ball_candidate_tracked')):
                raise RuntimeError('Completion text appeared without observed shot motion; stopped unverified')
            diagnostic_state('completion_verified',dict(text=panel,motion=report),frame)
            return 'complete', dict(complete=True,motion=report)
        state = shot_state(frame,conditions=False)
        diagnostic_state(f'post_shot_read_{time.time_ns()}',state,frame)
        safety.check()
        ready = (state['player'] and state['mode'] is not None and state['unit']==state['club_unit']
                 and state['putt']==(state['unit']=='ft')
                 and swing_button(frame, rect) is not None and ready_meter(frame, rect) is not None)
        outcome = tracker.update(state, ready)
        if outcome:
            report=observer.report() if observer else {}
            if observer:
                if report.get('capture_error'):
                    raise RuntimeError('MSS motion observer failed: '+report['capture_error'])
                if not (report.get('scene_motion_observed') or report.get('moving_ball_candidate_tracked')):
                    raise RuntimeError('HUD changed without video motion evidence; shot movement unverified')
                same_hole=(before.get('hole') is not None and state.get('hole')==before['hole'])
                increment=(before.get('shot') is not None and state.get('shot') is not None
                           and state['shot']>before['shot'])
                if outcome=='next' and (not same_hole or not increment):
                    raise RuntimeError('Next HUD is ready but stroke increment was not verified; no repeat shot')
                state['motion']=report
                state['movement_verification']='video motion plus stable stroke/HUD advance'
                stats=shot_stats(frame)
                if stats is not None and stats!=before.get('previous_stats'):
                    state['fresh_stats']=stats
            return outcome, state
    raise RuntimeError('Could not verify the next ready shot or hole completion; no repeat shot')


def run_hole(safety, sct, rect, reference, max_shots=15, one_shot=False):
    number=0
    while max_shots==0 or number<max_shots:
        number+=1
        state, frame = stable_state(safety, sct, rect)
        DEBUG.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG/f'before_shot_{number:02d}.png'), frame)
        state, frame = select_club(safety, sct, rect, state, frame)
        if DIAGNOSTICS:
            state['previous_stats']=shot_stats(frame)
        plan=POWER_MODEL.plan(state) if POWER_MODEL else dict(power=plan_power(state),model='test baseline')
        power=plan['power']
        diagnostic_state(f'shot_{number:02d}_plan',dict(state=state,plan=plan),frame)
        if DIAGNOSTICS:
            DIAGNOSTICS.event('shot_planned',shot=number,state=state,plan=plan)
        meter = ready_meter(frame, rect)
        if meter is None:
            raise RuntimeError('Saved mark and visible meter disagree')
        print(f'SHOT {number}: hole={state["hole"]}, stroke={state["shot"]}, '
              f'{state["distance"]:g} {state["unit"]}, {state["mode"]}, '
              f'club range={state["club_range"]:g} {state["club_unit"]}, '
              f'baseline power={power:.1%}', flush=True)
        # A stroke may have begun on release even if an exception follows.
        # Never retry an interrupted stroke in the same controller run.
        observer=ShotObserver(rect,frame) if DIAGNOSTICS else None
        if observer:
            observer.start()
        try:
            actual = drag_swing(safety, sct, rect, reference, meter, power)
            # The return starts at the released yellow edge, not at the left
            # rectangle border. This is visible in green_return_review.png.
            return_meter = replace(meter, return_origin=round(meter.zero-actual*(meter.zero-meter.origin)))
            accuracy_click(safety, sct, rect, return_meter, reference)
            if observer:
                observer.arm()
            if DIAGNOSTICS:
                DIAGNOSTICS.flush_meters(number)
                DIAGNOSTICS.flush_console()
                DIAGNOSTICS.event('accuracy_input_sent',shot=number,power=actual)
            outcome, after = wait_until_next(safety, sct, rect, state,observer=observer)
        finally:
            if observer:
                observer.close()
                observer.save(number)
            if DIAGNOSTICS:
                DIAGNOSTICS.flush_meters(number)
        log = dict(attempt=number, before=state, observed_power=actual, outcome=outcome, after=after)
        (DEBUG/f'shot_{number:02d}.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
        diagnostic_state(f'shot_{number:02d}_result',log)
        if DIAGNOSTICS:
            DIAGNOSTICS.event('shot_verified',shot=number,outcome=outcome,after=after)
            if POWER_MODEL and after and after.get('fresh_stats'):
                POWER_MODEL.record(ROOT/'wgt_calibration/power_observations.json',state,after['fresh_stats'],actual,
                                   DIAGNOSTICS.path.name+f':{number}')
        if outcome == 'complete':
            print('HOLE COMPLETION OBSERVED on screen. Controller stopped.', flush=True)
            return 'complete'
        print('NEXT READY SHOT VERIFIED', flush=True)
        if one_shot:
            return 'one_shot'
    raise RuntimeError(f'Shot cap ({max_shots}) reached; hole completion not observed')


def main(argv=None):
    global DIAGNOSTICS,POWER_MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='dependencies/calibration only; no capture or input')
    group.add_argument('--observe', action='store_true', help='read one current screenshot; no input')
    group.add_argument('--one-shot', action='store_true', help='one verified stroke including accuracy')
    group.add_argument('--controlled-test', action='store_true', help='bounded one-hole live test with automatic evidence logging')
    parser.add_argument('--countdown', type=int, default=8)
    parser.add_argument('--max-shots', type=int, default=15)
    parser.add_argument('--max-seconds',type=int,default=900)
    parser.add_argument('--activate',action='store_true',help='raise the existing WGT window before input; no clicks')
    args = parser.parse_args(argv)
    if not 0 <= args.countdown <= 60 or not 0 <= args.max_shots <= 30 or not (args.max_seconds==0 or 30<=args.max_seconds<=3600):
        parser.error('countdown must be 0..60; max-shots 0..30; max-seconds 0 or 30..3600 (0 disables a cap)')
    safety = None
    hwnd=None
    rect=None
    status='stopped_unverified'
    final_data={}
    original_out,original_err=sys.stdout,sys.stderr
    try:
        setup()
        if not args.check:
            DIAGNOSTICS=Session(ROOT)
            sys.stdout=ConsoleLog(original_out,DIAGNOSTICS)
            sys.stderr=ConsoleLog(original_err,DIAGNOSTICS)
            print('Automatic diagnostics:',DIAGNOSTICS.path,flush=True)
        reference = StrokeReference.from_data(json.loads(RECORD.read_text(encoding='utf-8')))
        calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))
        fraction = float(calibration['accuracy_x_fraction'])
        if not math.isfinite(fraction) or not .25 < fraction < .90:
            raise ValueError('Invalid saved accuracy calibration')
        pytesseract.get_tesseract_version()
        POWER_MODEL=PowerCalibration.load(ROOT/'wgt_calibration/power_observations.json',ROOT/'wgt_shot_logs/shots.csv')
        print('WGT one-hole controller | F9 emergency stop | aim/camera unchanged', flush=True)
        print('Supported verified modes: '+', '.join(SUPPORTED_SHOT_MODES),flush=True)
        print(f'Saved stroke: {reference.width}x{reference.height}, meter anchor={reference.anchor}, '
              f'drag={reference.drag_seconds:.2f}s, accuracy interval={reference.accuracy_delay:.2f}s', flush=True)
        if args.check:
            print('DEPENDENCIES AND RECORDING CHECK PASSED. No live-game test performed.', flush=True)
            return 0
        if not args.observe:
            print('Activate the existing WGT Golf window. No aim or camera controls will be used.', flush=True)
            for n in range(args.countdown,0,-1):
                if key_down(F9):
                    raise KeyboardInterrupt('F9 emergency stop')
                print(f'{n}...', flush=True)
                time.sleep(.1)
                for _ in range(9):
                    if key_down(F9):
                        raise KeyboardInterrupt('F9 emergency stop')
                    time.sleep(.1)
        hwnd = find_wgt()
        if args.activate and not args.observe:
            activate_wgt(hwnd)
        rect = rect_of(hwnd)
        with mss.MSS() as sct:
            diagnostic_state('initial_geometry',dict(rect=rect,foreground=(user32.GetForegroundWindow()==hwnd),
                                                    args=vars(args)),capture(sct,rect))
            if args.observe:
                # Observation is deliberately available without activating WGT.
                readings=[]
                for i in range(2):
                    frame=capture(sct,rect)
                    reading=inspect_screen(frame,rect)
                    readings.append(reading)
                    diagnostic_state(f'detection_read_{i}',reading,frame)
                    DIAGNOSTICS.image(f'detection_read_{i}_annotated',detection_overlay(frame,rect,reading))
                state=readings[-1]['state']
                detection=dict(readings=readings,stable=detections_agree(*readings),input_sent=False)
                diagnostic_state('detection_report',detection)
                diagnostic_state('observation',state,frame)
                DIAGNOSTICS.image('hud_wind',crop(frame,(.06,.05,.13,.105)))
                DIAGNOSTICS.image('hud_club',crop(frame,(.01,.925,.18,.998)))
                if state.get('sign'):
                    x,y,w,h=state['sign']
                    DIAGNOSTICS.image('hud_distance_elevation',frame[y:y+h,x:x+w])
                print(json.dumps(dict(state=state,detection_checks=readings[-1]['checks'],
                                     stable=detection['stable'],swing_button_window=readings[-1]['swing_button_window'],
                                     meter=readings[-1]['meter']), indent=2), flush=True)
                print('OBSERVE ONLY. No input sent; verify that WGT was unobstructed.', flush=True)
                status='observed_only'
                final_data=dict(input_sent=False,state=state,detection_checks=readings[-1]['checks'],
                                detections_stable=detection['stable'])
                return 0 if detection['stable'] else 2
            safety = Safety(hwnd, rect)
            safety.deadline=time.perf_counter()+args.max_seconds if args.max_seconds else None
            safety.start()
            with clicklock_disabled():
                result=run_hole(safety, sct, rect, reference, args.max_shots, args.one_shot)
            status='hole_completion_verified' if result=='complete' else 'one_shot_verified'
            final_data=dict(outcome=result)
        return 0
    except KeyboardInterrupt as exc:
        final_data=dict(reason=str(exc))
        print(f'STOPPED: {exc}', flush=True)
        return 130
    except Exception as exc:
        final_data=dict(reason=str(exc),traceback=traceback.format_exc())
        print(f'STOPPED: {exc}. Hole completion has not been verified.', flush=True)
        return 1
    finally:
        if safety is not None:
            safety.close()
        if DIAGNOSTICS:
            try:
                try:
                    if rect and status=='stopped_unverified':
                        with mss.MSS() as sct:
                            DIAGNOSTICS.image('stopped_frame',capture(sct,rect))
                    DIAGNOSTICS.flush_meters(0)
                except Exception as exc:
                    final_data['diagnostic_capture_error']=str(exc)
                DIAGNOSTICS.close(status,**final_data)
            finally:
                sys.stdout,sys.stderr=original_out,original_err
                DIAGNOSTICS=None
        POWER_MODEL=None


if __name__ == '__main__':
    raise SystemExit(main())
