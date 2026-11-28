"""One supervised golf-automation run, with unchanged aim/camera and F9 stop.

Reuses the saved manual stroke's meter press and drag cadence, the saved
accuracy calibration, MSS capture, and the existing HUD/meter colour readers.
Distance-to-power is a baseline estimate, not an empirically fitted golf model.
--check and --observe never send mouse input. See the controlled-test documentation for limits.
"""
import argparse
from contextlib import contextmanager
from collections import deque, OrderedDict
from contextvars import ContextVar
from copy import deepcopy
import ctypes
import ctypes.wintypes as wt
from dataclasses import asdict
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

from automation_core import (AccuracyTracker, Distance, StrokeReference, TransitionTracker,
                      complete_text, consensus, detect_meter, green_front,
                      meter_rect, parse_distance, parse_shot, plan_power,
                      state_signature, yellow_power, resolve_target_distance)
from vision_engine import MotionEvidence, PowerCalibration, Session, read_conditions, elevation_arrow, read_stats_patch, unit_from_glyphs, visible_hole_lock, quick_putt_badge, read_putter_range, read_inch_elevation
from vision_engine import read_stippled_locked_putt
from context_manager import ContextObserver, GameCoordinator, ControllerLock
from game_state import GamePolicy
from terrain_handler import BUNKER_BLOCK_REASON, bunker_calibration_plan, is_bunker_state
from automation_learning import ShadowLearning
from shot_capture import build_shot_capture
from vision_engine import read_locked_short_putt_distance
from vision.regions import hud_crop, selected_region
from vision.ball_spin import read_ball_spin
from agent.shot_pipeline import ShotTiming, build_plan, execute_stroke

ROOT = Path(__file__).resolve().parent
DEBUG = ROOT/'automation_debug'
RECORD = ROOT/'wgt_calibration/last_manual_stroke.json'
CALIBRATION = ROOT/'automation_accuracy.json'
SWING_TEMPLATE = ROOT/'swing_button.png'
PLAYER = 'tigerwould52'
F9 = 0x78
DOWN, UP, WHEEL = 0x0002, 0x0004, 0x0800
MOVE, ABSOLUTE, VIRTUAL = 0x0001, 0x8000, 0x4000
user32 = ctypes.WinDLL('user32', use_last_error=True) if sys.platform == 'win32' else None
DIAGNOSTICS = None
POWER_MODEL = None
GAME = None
LEARNING = None  # Recommendations are logged only; never passed to control code.
SUPPORTED_SHOT_MODES=('FULL','CHIP','PITCH','PUTT')
OCR_CACHE=OrderedDict()
OCR_TEXT_CAPTURE=ContextVar('ocr_text_capture',default=None)


def ocr_text(image,config):
    key=(image.shape,str(image.dtype),hashlib.sha256(image.tobytes()).digest(),config)
    if key in OCR_CACHE:
        OCR_CACHE.move_to_end(key)
        text=OCR_CACHE[key]
    else:
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
    texts=OCR_TEXT_CAPTURE.get()
    if texts is not None:
        texts.append(text)
    return text


def capture_ocr_text(reader,*args):
    texts=[]
    token=OCR_TEXT_CAPTURE.set(texts)
    try:
        return reader(*args),texts
    finally:
        OCR_TEXT_CAPTURE.reset(token)


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


def learning_event(method,*args,**kwargs):
    """Shadow failures never select power, send input or retry a consumed shot."""
    if LEARNING is None:
        return None
    try:
        return getattr(LEARNING,method)(*args,**kwargs)
    except Exception as exc:
        try:
            if DIAGNOSTICS:
                DIAGNOSTICS.event('shadow_learning_error',operation=method,error=str(exc))
        except Exception:
            pass
        print(f'SHADOW LEARNING unavailable for {method}: {exc}',flush=True)
        return None


def read_shot_accuracy_trace(shot_id, number):
    """Attach shot identity after the click, never inside accuracy_click.

    before_shot_NN.png is written just before the current stroke; the saved
    trace must be newer. This only establishes trace freshness, not mouse-down
    accuracy or ball lateral motion. If anything is uncertain, abstain.
    """
    try:
        trace_file = DEBUG/'accuracy_trace.json'
        before_file = DEBUG/f'before_shot_{number:02d}.png'
        if not isinstance(shot_id, str) or not shot_id or not isinstance(number, int) or number < 1:
            return None
        if not trace_file.is_file() or not before_file.is_file():
            return None
        if trace_file.stat().st_mtime_ns <= before_file.stat().st_mtime_ns:
            return None
        trace = json.loads(trace_file.read_text(encoding='utf-8'))
        if (trace.get('clicked') is True
                and isinstance(trace.get('samples'), list) and trace['samples']
                and isinstance(trace.get('target'), (int, float))):
            return dict(trace, shot_id=shot_id)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None


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


class TITLEBARINFO(ctypes.Structure):
    _fields_=[('cbSize',wt.DWORD),('rcTitleBar',wt.RECT),('rgstate',wt.DWORD*6)]


def setup():
    if user32 is None:
        raise RuntimeError('Live WGT controller requires Windows')
    user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wt.UINT
    user32.GetForegroundWindow.restype = wt.HWND
    user32.SetForegroundWindow.argtypes=[wt.HWND]
    user32.SetForegroundWindow.restype=wt.BOOL
    user32.GetTitleBarInfo.argtypes=[wt.HWND,ctypes.POINTER(TITLEBARINFO)]
    user32.GetTitleBarInfo.restype=wt.BOOL
    user32.WindowFromPoint.argtypes=[wt.POINT]
    user32.WindowFromPoint.restype=wt.HWND
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


def display_id(hwnd):
    class MonitorInfo(ctypes.Structure):
        _fields_=[('size',wt.DWORD),('monitor',wt.RECT),('work',wt.RECT),
                  ('flags',wt.DWORD),('device',wt.WCHAR*32)]

    monitor_from_window=ctypes.WINFUNCTYPE(wt.HANDLE,wt.HWND,wt.DWORD)(
        ('MonitorFromWindow',user32))
    get_monitor_info=ctypes.WINFUNCTYPE(wt.BOOL,wt.HANDLE,ctypes.POINTER(MonitorInfo))(
        ('GetMonitorInfoW',user32))
    monitor=monitor_from_window(hwnd,2)  # MONITOR_DEFAULTTONEAREST
    info=MonitorInfo(size=ctypes.sizeof(MonitorInfo))
    if monitor and get_monitor_info(monitor,ctypes.byref(info)):
        match=re.search(r'DISPLAY(\d+)$',info.device,re.IGNORECASE)
        if match:
            return int(match[1])
    return None


def point_in_wgt(hwnd,x,y):
    """Read-only hit test for new high-level input boundaries."""
    target=user32.WindowFromPoint(wt.POINT(x,y))
    if not target:
        return False
    if target==hwnd:
        return True
    ancestor=user32.GetAncestor
    ancestor.argtypes=[wt.HWND,wt.UINT]
    ancestor.restype=wt.HWND
    return ancestor(target,2)==hwnd


def atomic_caption_click(x,y):
    """One non-interleaved native batch; no held time for a caption drag."""
    vx,vy=user32.GetSystemMetrics(76),user32.GetSystemMetrics(77)
    vw,vh=user32.GetSystemMetrics(78),user32.GetSystemMetrics(79)
    if min(vw,vh)<2 or not vx<=x<vx+vw or not vy<=y<vy+vh:
        raise RuntimeError('Caption point is outside the virtual desktop')
    ax=round((x-vx)*65535/(vw-1));ay=round((y-vy)*65535/(vh-1))
    events=(INPUT*2)()
    for i,button in enumerate((DOWN,UP)):
        events[i]=INPUT(type=0,data=INPUT_UNION(mi=MOUSEINPUT(ax,ay,0,MOVE|ABSOLUTE|VIRTUAL|button,0,0)))
    if user32.SendInput(2,events,ctypes.sizeof(INPUT))!=2:
        raise OSError(ctypes.get_last_error(),'Windows rejected atomic caption click')


def click_wgt_titlebar(hwnd):
    """Activate only a native, visible, uncovered caption; never the course."""
    if key_down(F9):
        raise KeyboardInterrupt('F9 emergency stop')
    if key_down(1):
        raise RuntimeError('Left button already held; cannot activate WGT safely')
    bounds=rect_of(hwnd)
    info=TITLEBARINFO(cbSize=ctypes.sizeof(TITLEBARINFO))
    if not user32.GetTitleBarInfo(hwnd,ctypes.byref(info)) or info.rgstate[0]&(1|0x8000|0x10000):
        raise RuntimeError('Visible WGT title bar not verified; no activation click')
    r=info.rcTitleBar
    if not (r.right-r.left>350 and 10<=r.bottom-r.top<=64
            and bounds['left']<=r.left<r.right<=bounds['left']+bounds['width']
            and bounds['top']<=r.top<r.bottom<=bounds['top']+min(80,.15*bounds['height'])):
        raise RuntimeError('WGT caption geometry not verified; no activation click')
    x,y=r.left+120,(r.top+r.bottom)//2
    if user32.WindowFromPoint(wt.POINT(x,y))!=hwnd:
        raise RuntimeError('Another window covers the WGT title bar; no activation click')
    move_to(x,y)
    if key_down(F9):
        raise KeyboardInterrupt('F9 emergency stop')
    if user32.WindowFromPoint(wt.POINT(x,y))!=hwnd or rect_of(hwnd)!=bounds:
        raise RuntimeError('WGT caption changed before activation; no click')
    try:
        atomic_caption_click(x,y)
        if key_down(F9):
            raise KeyboardInterrupt('F9 emergency stop')
    finally:
        mouse_up()
    if rect_of(hwnd)!=bounds:
        raise RuntimeError('WGT window geometry changed during activation; no swing')
    if DIAGNOSTICS:
        DIAGNOSTICS.event('wgt_titlebar_activation',x=x,y=y)


def activate_wgt(hwnd):
    """Raise the target application, falling back to one verified native caption click."""
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
    click_wgt_titlebar(hwnd)
    end=time.perf_counter()+2.
    while time.perf_counter()<end:
        if key_down(F9):
            raise KeyboardInterrupt('F9 emergency stop')
        if user32.GetForegroundWindow()==hwnd:
            return
        time.sleep(.005)
    raise RuntimeError('Windows did not give WGT foreground focus after caption activation; no swing sent')


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


def find_distance_sign(frame, region=None):
    # Reuse the putt reader's white top-edge locator, extending its size range
    # to the 60px tee sign. Cyan-only components merge into the blue sky.
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # A cloud can join a dim top border and widen it beyond the actual badge.
    # Try the bright border first; retain dim passes for translucent signs.
    # When the flag obscures the upper edge, start at the bright lower edge
    # and verify the remaining upper edge, both sides and blue fill.
    flag_gap=max(9,round(.0225*w))
    for closing,level,saturation,from_bottom in (
            (9,210,65,False),(9,175,65,False),(9,210,65,True),
            (9,140,100,False),(flag_gap,210,65,False),(5,120,100,True)):
        compact_bottom=from_bottom and level<175
        mask = cv2.inRange(hsv,(0,0,level),(179,saturation,255))
        mask[:int(.05*h)]=0;mask[int(.82*h):]=0
        # Targets can sit left/right of centre without any camera or aim
        # change. Exclude the minimap, but retain the side portions of the HUD.
        mask[:,:int(.16*w)]=0;mask[:,int(.88*w):]=0
        if region is not None:
            mask[:region.y]=0;mask[region.y+region.h:]=0
            mask[:,:region.x]=0;mask[:,region.x+region.w:]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1,closing),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((1,max(25,int(.025*w))),np.uint8))
        bottom_level=175 if level>=175 else 105
        white=cv2.inRange(hsv,(0,0,bottom_level),(179,65 if level>=175 else 100,255))
        white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((1,closing),np.uint8))
        oriented_hsv=hsv
        min_y,max_y=.05*h,.78*h
        if from_bottom:
            mask,white,oriented_hsv=mask[::-1],white[::-1],hsv[::-1]
            min_y,max_y=.22*h,.95*h
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        choices=[]
        for x,y,bw,bh,count in stats[1:]:
            if not (.035*w<bw<.21*w and bh<6 and min_y<y<max_y):
                continue
            width=int(bw)+2
            left=int(x)-1
            if from_bottom:
                # Recover a compact badge with a dim, broken top edge only
                # after the original passes, using its stronger lower edge.
                lower,upper=(.035,.07) if compact_bottom else (.07,.12)
                if not lower*w<width<upper*w:
                    continue
                # The rounded lower corner starts inside the left side rule.
                left-=1
                width+=1
            expected_bottom=int(y+width*.76)
            # Translucency varies across the sign. A dimmer bottom border is
            # accepted only with the top edge, blue interior and correct size.
            lo,hi=int(y+width*.65),min(h,int(y+width*.9))
            rows=(white[lo:hi,int(x):int(x+bw)]>0).sum(axis=1)
            edge_fraction=.60 if compact_bottom else (.44 if from_bottom else .65)
            if not len(rows) or rows.max()<edge_fraction*bw:
                continue
            bottom=lo+int(rows.argmax())
            height=bottom-int(y)+1
            if compact_bottom and not .71<=height/width<=.82:
                continue
            # Text strokes also form horizontal white runs. A real badge has
            # continuous side borders; an internal digit row does not.
            side_left=white[int(y)+3:bottom-2,max(0,left-2):left+4]
            side_right=white[int(y)+3:bottom-2,left+width-4:left+width+2]
            if (not side_left.size or not side_right.size
                    or (side_left>0).any(axis=1).mean()<.55
                    or (side_right>0).any(axis=1).mean()<.55):
                continue
            inside=oriented_hsv[int(y+width*.08):bottom-2,left+4:left+width-4]
            blue_ratio=((inside[:,:,0]>80)&(inside[:,:,0]<117)&(inside[:,:,1]>25)).mean()
            if blue_ratio<(.60 if compact_bottom else .40):
                continue
            score=blue_ratio-abs(x+bw/2-w/2)/w-abs(bottom-expected_bottom)/w
            top=h-1-bottom if from_bottom else int(y)
            choices.append((score,left,top,width,height))
        if choices:
            return max(choices)[1:]
    return None


def read_segmented_distance(patch):
    """Require matching digit OCR across separate white-text thresholds.

    Decimal punctuation is deliberately not discarded: if present between
    large glyphs this fallback abstains and leaves decimals to the main reader.
    """
    if patch is None or not patch.size:
        return None
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    values=[]
    for level in (105,150,200):
        mask=cv2.inRange(hsv,(0,0,level),(179,55,255))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        glyphs=sorted((s for s in stats[1:] if s[3]>.5*patch.shape[0] and s[4]>=8),key=lambda s:s[0])
        if not 1<=len(glyphs)<=3:
            continue
        left=glyphs[0][0];right=glyphs[-1][0]+glyphs[-1][2]
        if any(left<s[0]<right and s[3]<.25*patch.shape[0]
               and s[1]+s[3]>.65*patch.shape[0] for s in stats[1:]):
            continue
        digits=[]
        for x,y,width,height,_ in glyphs:
            image=cv2.copyMakeBorder(255-mask[y:y+height,x:x+width],4,4,4,4,cv2.BORDER_CONSTANT,value=255)
            image=cv2.resize(image,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
            digit=ocr_text(image,'--psm 13 -c tessedit_char_whitelist=0123456789').strip()
            if not re.fullmatch('[0-9]',digit):
                break
            digits.append(digit)
        if len(digits)==len(glyphs):
            values.append(float(''.join(digits)))
    return consensus(values)


def read_sign_distance(frame, sign):
    if sign is None:
        return None
    x, y, w, h = sign
    numeric = frame[y+int(.06*w):y+int(.39*w),x+int(.12*w):x+int(.74*w)]
    unit_patch = frame[y+int(.12*w):y+int(.40*w),x+int(.72*w):x+int(.98*w)]
    yard_labels = ('YDS','YD','YOS','VDS','VOS','YD5','Y0S','V0S')
    units = []
    for t in ocr_variants(unit_patch,modes=(6,7),scale=8):
        clean = re.sub('[^A-Z0-9]','',t)
        clean=re.sub(r'^[0-9O]+','',clean)
        if clean in yard_labels:
            units.append('yd')
        elif clean == 'FT':
            units.append('ft')
    unit = consensus(units)
    if unit is None:
        wider=frame[y+int(.12*w):y+int(.40*w),x+int(.66*w):x+int(.98*w)]
        for t in ocr_variants(wider,modes=(6,7),scale=8):
            clean=re.sub(r'^[0-9O]+','',re.sub('[^A-Z0-9]','',t))
            if clean in yard_labels:
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
    number_pattern=r'(?:0|[1-9]\d{0,2})(?:\.\d{1,2})?'
    values = []
    for picture in (big,binary,white,adaptive):
        for psm in (6,7):
            text = ocr_text(picture,f'--psm {psm} -c tessedit_char_whitelist=0123456789.').strip()
            if re.fullmatch(number_pattern,text):
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
            neutral_votes.append(float(text) if re.fullmatch(number_pattern,text) else text)
    neutral_value=consensus(neutral_votes,minimum=3)
    value = consensus(values)
    if isinstance(neutral_value,str):
        # A leading zero can be a misread first glyph (56 -> 06).
        # Recover the digits individually and require whole-row support.
        value=read_segmented_distance(numeric)
        if value not in values:
            return None
    elif neutral_value is not None:
        value=neutral_value
    elif glyph_unit_used:
        # Faded units indicate the same transparency can erase trailing
        # digits. Require the complete colour-isolated number in this case.
        return None
    if value is None:
        value=read_segmented_distance(numeric)
    elif value==1 and unit=='yd':
        # Whole-row OCR can merge adjacent narrow 1s into a single digit.
        # Recover repeated 1s only when each segmented digit is visible.
        segmented=read_segmented_distance(numeric)
        if segmented in (11,111):
            value=segmented
    return parse_distance(f'{value} {unit.upper()}') if value is not None else None


def reread_putt_badge(frame,sign):
    detected=find_distance_sign(frame)
    if sign is None or detected is None:
        return None
    x,y,w,h=detected
    old_x,old_y,old_w,old_h=sign
    if max(abs(x-old_x),abs(y-old_y),abs(x+w-old_x-old_w),abs(y+h-old_y-old_h))>1:
        return None
    hole_distance=read_hole_hud_distance(frame) if visible_hole_lock(frame,detected) else None
    if hole_distance and hole_distance.unit=='ft' and 1<=hole_distance.value<=5:
        distance=read_locked_short_putt_distance(frame,detected,hole_distance.value,ocr_text)
        if distance is not None:
            elevation=read_conditions(frame,detected,ocr_variants,crop,numeric_reader=read_segmented_distance,
                                      elevation_only=True)['elevation_ft']
            return dict(distance=distance,unit='ft',elevation_ft=elevation) if elevation is not None else None
        result=quick_putt_badge(frame,detected,ocr_text)
        return result if result and abs(result['distance']-hole_distance.value)<=.5 else None
    result=quick_putt_badge(frame,detected,ocr_text)
    if result is None and hole_distance and hole_distance.unit=='ft':
        result=read_stippled_locked_putt(frame,detected,hole_distance.value,ocr_text)
    return result


def reread_elevation(frame,sign):
    detected=find_distance_sign(frame)
    if detected is None or detected[:3]!=tuple(sign[:3]) or abs(detected[3]-sign[3])>1:
        return None
    inches=read_inch_elevation(frame,detected,ocr_text)
    if inches is not None:
        return inches
    x,y,w,h=detected
    unit=unit_from_glyphs(frame[y+int(.54*h):y+int(.96*h),x+int(.65*w):x+w],w,elevation=True)
    if unit not in ('FT','IN'):
        return None
    patch=frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)]
    texts=ocr_variants(patch,modes=(6,7),scale=8)
    numbers=[float(text.strip()) for text in texts
             if re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',text.strip())]
    if len(numbers)<2 or len(set(numbers))!=1:
        return None
    number=numbers[0]
    direction=elevation_arrow(frame[y+int(.46*h):y+h,x:x+int(.32*w)])
    if direction is None and number!=0:
        return None
    return number*(direction or 1)/(12 if unit=='IN' else 1)


def read_stroke_glyphs(frame,texts):
    """Use visible digit glyphs only when the ordinal suffix is recognized.

    In the live eighth-shot HUD, ordinary OCR reads STHSHOT. Never translate
    S to 8 blindly; isolate its glyph and require independent numeric votes.
    """
    counts=[]
    literal_numbers=set()
    for text in texts:
        clean=re.sub('[^A-Z0-9]','',text.upper())
        match=re.fullmatch(r'([A-Z0-9]{1,2})(ST|ND|RD|TH)SHOT',clean)
        if match:
            if match[1].isdigit():
                number=int(match[1])
                suffix='TH' if 10<=number%100<=20 else {1:'ST',2:'ND',3:'RD'}.get(number%10,'TH')
                if not 1<=number<=30 or match[2]!=suffix:
                    continue
                literal_numbers.add(number)
            counts.append(len(match[1]))
    if len(literal_numbers)>1:
        return None
    count=consensus(counts)
    if count is None:
        if len(literal_numbers)!=1:
            return None
        # One intact full label can supply the digit count only. The number
        # itself still needs the independent pixel/glyph votes below.
        count=len(str(next(iter(literal_numbers))))
    patch=crop(frame,(.895,.083,.934,.104))
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    votes=[]
    eight_votes=[]
    for threshold in (105,145,185):
        mask=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)[1]
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        glyphs=sorted((s for s in stats[1:] if s[3]>=.45*patch.shape[0] and s[4]>=8),key=lambda s:s[0])
        if len(glyphs)<count:
            continue
        selected=glyphs[:count]
        x=min(s[0] for s in selected);y=min(s[1] for s in selected)
        right=max(s[0]+s[2] for s in selected);bottom=max(s[1]+s[3] for s in selected)
        if right-x>count*frame.shape[1]*.011:
            continue
        if count==1:
            glyph=mask[y:bottom,x:right]
            contours,hierarchy=cv2.findContours(glyph,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            holes=[] if hierarchy is None else [cv2.boundingRect(c) for c,h in zip(contours,hierarchy[0]) if h[3]>=0]
            centers=[b[1]+b[3]/2 for b in holes]
            # An 8 has separate upper and lower enclosed counters. Require
            # both visible topology and numeric OCR at multiple thresholds.
            if len(centers)==2 and min(centers)<.45*glyph.shape[0] and max(centers)>.55*glyph.shape[0]:
                numeral=255-gray[y:bottom,x:right]
                numeral=cv2.copyMakeBorder(numeral,3,3,3,3,cv2.BORDER_CONSTANT,value=255)
                numeral=cv2.resize(numeral,None,fx=12,fy=12,interpolation=cv2.INTER_CUBIC)
                if ocr_text(numeral,'--psm 13 -c tessedit_char_whitelist=0123456789').strip()=='8':
                    eight_votes.append(8)
        numeric=cv2.copyMakeBorder(mask[y:bottom,x:right],4,4,4,4,cv2.BORDER_CONSTANT,value=0)
        numeric=cv2.resize(numeric,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
        for psm in (7,10,13):
            value=ocr_text(numeric,f'--psm {psm} -c tessedit_char_whitelist=0123456789').strip()
            if re.fullmatch(r'\d{1,2}',value) and len(value)==count and 1<=int(value)<=30:
                votes.append(int(value))
    number=consensus(eight_votes) or consensus(votes,minimum=3)
    return number if not literal_numbers or number in literal_numbers else None


def read_player_header(frame):
    texts=ocr_variants(crop(frame,(.830,.060,.900,.083)),modes=(6,7,13),scale=8)
    votes=[PLAYER if PLAYER in re.sub('[^a-z0-9]','',t.lower()) else None for t in texts]
    return consensus(votes)==PLAYER,texts


def shot_state(frame, conditions=True, target_source=None):
    if target_source not in (None,'floating_target','verified_hole_hud'):
        raise ValueError('Unsupported target distance source')
    report=DIAGNOSTICS if conditions else None
    timestamp=time.time() if report else None
    if report:
        report.shot_stage('observation_started',timestamp=timestamp,observation_source='live_ui')
    region = selected_region('distance')
    sign = (find_distance_sign(frame, region) if region else None) or find_distance_sign(frame)
    yards_text=[]
    elevation_text=[]
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
    putter=read_putter_range(frame,ocr_text) if mode=='PUTT' else None
    # Current club's range always has its own explicit unit, including putters.
    club_texts = [] if mode=='PUTT' else ocr_variants(hud_crop(frame,'club',(.111,.936,.172,.991),crop), modes=(6,7), scale=5)
    # Keep the club-name column out of the range. OCR line ordering can put
    # 'IRONS' between 187 and YDS, so read this vertical column on its own.
    range_texts = [] if mode=='PUTT' else ocr_variants(hud_crop(frame,'club',(.140,.938,.171,.991),crop), modes=(6,), scale=6)
    range_texts = [re.sub(r'(?:YOS|VOS|YD5)\b','YDS',t) for t in range_texts]
    club = Distance(putter['value'],'ft') if putter else consensus([parse_distance(t) for t in range_texts])
    number_texts=[];unit_texts=[]
    if club is None and mode!='PUTT':
        club = consensus([parse_distance(t) for t in club_texts])
    if club is None and mode!='PUTT':
        # WEDGES/WOODS can be placed between the value and YDS by OCR line
        # ordering. Read the two printed rows independently, with explicit units.
        number_texts=ocr_variants(crop(frame,(.122 if mode=='PUTT' else .137,.938,.172,.966)),modes=(7,),scale=7)
        # Putter arrows flank the printed FT column; excluding them prevents
        # the arrow glyphs being read as A/T and defeating unit agreement.
        unit_patch=crop(frame,(.125,.966,.149,.989) if mode=='PUTT' else (.145,.966,.172,.989))
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
        if number is None and numbers:
            # The 115-yard label can read 115/HS without padding. Preserve
            # its antialiased strokes: thresholding this tiny row drops a 1.
            # Require two padded reads to agree with the original numeral.
            patch=crop(frame,(.122 if mode=='PUTT' else .137,.938,.172,.966))
            gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            large=cv2.resize(255-gray,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
            large=cv2.copyMakeBorder(large,40,40,40,40,cv2.BORDER_CONSTANT,value=255)
            padded=[ocr_text(large,f'--psm {psm} -c tessedit_char_whitelist=0123456789.').strip()
                    for psm in (6,7)]
            number_texts.extend(padded)
            candidate=consensus([float(t) if re.fullmatch(r'\d{1,3}(?:\.\d+)?',t) else None
                                 for t in padded])
            if candidate is not None and all(n==candidate for n in numbers):
                number=candidate
        if number is not None and unit:
            club=Distance(number,unit)
    hole_distances=ocr_variants(crop(frame,(.135,.058,.178,.095)),modes=(6,7),scale=5)
    hole_distance=consensus([parse_distance(re.sub(r'(?:YOS|VOS|YD5|Y0S)\b','YDS',t)) for t in hole_distances])
    if hole_distance is None:
        if report:
            hole_distance,extra_text=capture_ocr_text(read_hole_hud_distance,frame)
            hole_distances.extend(extra_text)
        else:
            hole_distance=read_hole_hud_distance(frame)
    lock=visible_hole_lock(frame,sign) if mode in ('PUTT','CHIP','PITCH') else None
    short_locked=(mode=='PUTT' and lock and hole_distance and hole_distance.unit=='ft'
                  and 1<=hole_distance.value<=5 and target_source!='verified_hole_hud')
    short_distance=(read_locked_short_putt_distance(frame,sign,hole_distance.value,ocr_text)
                    if short_locked else None)
    badge=None
    if short_distance is not None:
        distance=Distance(short_distance,'ft')
    else:
        badge=quick_putt_badge(frame,sign,ocr_text,elevation_text=elevation_text if report else None)
        if badge is None and mode=='PUTT' and lock and hole_distance and hole_distance.unit=='ft':
            badge=read_stippled_locked_putt(frame,sign,hole_distance.value,ocr_text)
        if badge:
            distance=Distance(badge['distance'],'ft')
        elif report:
            distance,yards_text=capture_ocr_text(read_sign_distance,frame,sign)
        else:
            distance=read_sign_distance(frame,sign)
        if (short_locked and distance and (distance.unit!='ft' or abs(distance.value-hole_distance.value)>.5)):
            distance=None
    hud=GAME.observer.read_hud(frame) if GAME is not None else None
    right_texts = ocr_variants(crop(frame, (.830,.058,.984,.104)), modes=(6,)) if hud is None else []
    shot_texts = ocr_variants(crop(frame, (.898,.082,.934,.105)), modes=(7,), scale=6) if hud is None else []
    shots = [parse_shot(re.sub(r'\bIST\s*SHOT','1ST SHOT',t), par) for t in right_texts+shot_texts]
    shot=consensus(shots)
    if shot is None and hud is None:
        shot=read_stroke_glyphs(frame,shot_texts)
    right = ' '.join(right_texts)
    player = PLAYER in re.sub(r'[^a-z0-9]', '', right.lower())
    player_header=[]
    if not player and hud is None:
        player,player_header=read_player_header(frame)
    # The multiplayer highlighted row moves vertically and remains highlighted
    # during flight. Never use the fixed solo account-header crop in this HUD.
    if hud is not None:
        player=bool(hud.get('valid') and hud.get('turn_state')=='OWN_READY'
                    and hud.get('active_player')==PLAYER)
        shot=parse_shot(hud.get('shot_label') or '',par) if hud.get('valid') else None
    state = dict(distance=distance.value if distance else None,
                unit=distance.unit if distance else None, putt=(mode=='PUTT') if mode else None,
                distance_source='floating_target',
                hole_distance=hole_distance.value if hole_distance else None,
                hole_distance_unit=hole_distance.unit if hole_distance else None,
                mode=mode, club_range=club.value if club else None,
                club_unit=club.unit if club else None, shot=shot,
                hole=hole, par=par, complete=False, player=player, sign=sign,
                raw=dict(mode=lower, club=club_texts,club_range_number=number_texts,club_range_unit=unit_texts,
                         putter_range_evidence=putter,
                         hole_distance=hole_distances,right=right_texts,player_header=player_header,hole=top,par=par_text,shot=shot_texts))
    if conditions:
        state.update(read_conditions(frame,sign,ocr_variants,crop,verified_badge=badge,
                                     numeric_reader=read_segmented_distance,glyph_ocr=ocr_text))
        state['spin_indicator'] = read_ball_spin(frame)
        if mode=='PUTT':
            state['club_name']='PUTTER'
    if hud is not None:
        state['raw']['team_hud']=hud
        if (player and hud.get('shot_label')=='NOT_STARTED' and mode=='FULL'
                and GAME.observer.team_hud is not None):
            # Explicit dashes on all four counters mean no strokes this hole.
            # The independent lie reader must still verify TEE before planning.
            state['shot']=1
    state.update(resolve_target_distance(
        distance if target_source!='verified_hole_hud' else None,
        hole_distance if target_source!='floating_target' else None,
        linked_to_hole=lock is not None))
    if target_source is not None:
        state['distance_source_requested']=target_source
    state['target_hole_link']=lock
    if report:
        if state['distance_source']=='verified_hole_hud' or target_source=='verified_hole_hud':
            yards_text=hole_distances
        for key,texts in state['conditions_raw'].items():
            if key.startswith('elevation') and isinstance(texts,list):
                elevation_text.extend(texts)
        report.shot_observation(state,timestamp,yards_text,elevation_text)
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
    # The golfer can overlap the left arc and defeat full-circle matching.
    # On the recorded window geometry, independently require the exact label
    # plus substantial visible white circular border at its calibrated place.
    if abs(w-1288)<=10 and abs(h-728)<=10:
        cx,cy,r=round(.895*w),round(.727*h),round(.036*w)
        label=frame[cy-round(.28*r):cy+round(.38*r),cx-round(.75*r):cx+round(.75*r)]
        texts=ocr_variants(label,modes=(7,13),scale=6)
        if consensus(['SWING' if re.sub('[^A-Z]','',t.upper())=='SWING' else None for t in texts]):
            white=cv2.inRange(cv2.cvtColor(frame,cv2.COLOR_BGR2HSV),(0,0,180),(179,65,255))
            arcs=[]
            for angle in np.linspace(0,2*math.pi,72,endpoint=False):
                px,py=round(cx+r*math.cos(angle)),round(cy+r*math.sin(angle))
                arcs.append(bool((white[max(0,py-3):py+4,max(0,px-3):px+4]>0).any()))
            if sum(arcs)>=.60*len(arcs):
                return cx,cy
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


def readiness_problems(state, frame, rect, expected=None):
    """Report exactly which live readings are still missing or inconsistent.

    Never turn a missing measurement into a guessed value.
    """
    problems = []
    if not state.get('player'):
        problems.append('player')
    if state.get('hole') is None:
        problems.append('hole label')
    if state.get('shot') is None:
        problems.append('stroke label')
    if state.get('mode') not in SUPPORTED_SHOT_MODES:
        problems.append('shot mode')
    if state.get('distance') is None or state.get('unit') not in ('yd', 'ft'):
        problems.append('target distance/unit')
    if state.get('club_range') is None or state.get('club_unit') not in ('yd', 'ft'):
        problems.append('club range/unit')
    if (state.get('unit') is not None and state.get('club_unit') is not None
            and state['unit'] != state['club_unit']):
        problems.append('distance/club unit agreement')
    if (state.get('putt') is None or
            (state.get('unit') is not None and state['putt'] != (state['unit']=='ft'))):
        problems.append('putting-mode agreement')
    if state.get('elevation_ft') is None:
        problems.append('elevation')
    if state.get('wind') is None:
        problems.append('wind')
    if not state.get('club_name'):
        problems.append('club name')
    if state.get('putt') and state.get('green_speed') is None:
        problems.append('green speed')
    # The next-shot transition was already verified by TransitionTracker.
    # Any *visible conflicting* hole or stroke must nevertheless stop a replay.
    if expected:
        for key in ('hole', 'shot'):
            old, current = expected.get(key), state.get(key)
            if old is not None and current is not None and current != old:
                problems.append(f'{key} changed unexpectedly ({current} != {old})')
    if swing_button(frame, rect) is None:
        problems.append('Swing button')
    if ready_meter(frame, rect) is None:
        problems.append('ready power/accuracy meter')
    return problems


def stable_state(safety, sct, rect, timeout=30.0, expected=None):
    """Reacquire two agreeing *complete* readings, tolerating transient bad OCR.

    The original four attempts were too brittle immediately after a shot.
    This retry is time-bounded; F9 and desktop-focus checks remain active.
    """
    started = time.perf_counter()
    last_signature = None
    last_good_at = None
    last_wind = None
    last_report_at = started - 7
    last_problems = ['no readable frame']
    last_frame = None
    samples = 0
    while time.perf_counter()-started < timeout:
        safety.check()
        captured_at=time.perf_counter()
        frame = capture(sct, rect)
        state = shot_state(frame)
        state['captured_at']=captured_at
        solo_putt=(GAME is not None and GAME.observer.solo_confirmed is True
                   and state.get('mode')=='PUTT')
        fresh_putt=False
        if solo_putt:
            context=GAME.inspect_ready(safety,sct,rect,frame,state,captured_at)
            fresh_putt=(context.get('permission',{}).get('allowed') is True
                        and 0<=time.perf_counter()-context['observed_at']<=4)
        elif GAME is not None and getattr(GAME.observer,'practice_shot_seconds',0):
            # Include next-tee reacquisition after Continue in the practice
            # budget, rather than starting its clock after two full OCR reads.
            GAME.inspect_ready(safety,sct,rect,frame,state,captured_at)
        safety.check()
        samples += 1
        problems = readiness_problems(state, frame, rect, expected)
        if solo_putt and not fresh_putt:
            problems.append('fresh verified putt HUD')
        now = time.perf_counter()
        last_problems = problems
        last_frame = frame
        if not problems:
            wind=state['wind']
            signature = (state_signature(state),state['mode'],state['club_name'],
                         state['elevation_ft'],state.get('green_speed') if state['putt'] else None,
                         wind['low_mph'],wind['high_mph'])
            vector=(wind['screen_dx'],wind['screen_dy'])
            # Slow OCR is allowed only when each complete putt reading has
            # independently fresh ownership and measured-HUD proof.
            matching_window=timeout if fresh_putt else 8.0
            if (signature == last_signature and last_good_at is not None and now-last_good_at <= matching_window
                    and last_wind is not None and math.dist(vector,last_wind)<=.20):
                print(f'READY: two matching live readings after {now-started:.1f}s', flush=True)
                return state, frame
            last_signature, last_good_at = signature, now
            last_wind=vector
        elif expected and any('changed unexpectedly' in p for p in problems):
            diagnostic_state('transition_mismatch', dict(expected=expected, observed=state,
                                                         problems=problems), frame)
            raise RuntimeError('Next-shot hole/stroke does not match verified transition; stopped')
        elif solo_putt:
            last_signature,last_good_at,last_wind=None,None,None
        elif last_good_at is not None and now-last_good_at > 8.0:
            last_signature, last_good_at = None, None
        if now-last_report_at >= 6.0:
            print('WAITING FOR NEXT SHOT HUD: '+(', '.join(problems) if problems else
                  'one verified frame; confirming again'), flush=True)
            diagnostic_state(f'ready_wait_{samples}', dict(elapsed=round(now-started,1),
                             missing=problems, expected=expected, observed=state), frame)
            last_report_at = now
        safety.wait(.20)
    if last_frame is not None:
        DEBUG.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG/'next_shot_not_ready.png'), last_frame)
    raise RuntimeError('Next shot could not be verified within '
                       f'{timeout:g}s; missing: {", ".join(last_problems) or "two matching frames"}. '
                       'Saved automation_debug/next_shot_not_ready.png; no new swing')


def park_on_meter(safety, rect):
    """Leave hover menus without pressing, scrolling or dragging anywhere."""
    safety.check()
    if key_down(1):
        raise RuntimeError('Left button held; no pointer move to dismiss club hover')
    move_to(rect['left']+round(.72*rect['width']),
            rect['top']+round(.965*rect['height']))
    safety.check()


def select_club(safety, sct, rect, state, frame):
    """Cycle only the bottom-left club/range control, verifying after each step."""
    learning_event('note_club',state)
    if is_bunker_state(state):
        raise RuntimeError(BUNKER_BLOCK_REASON)
    target = state['distance']
    low,high=(.12,.85) if state['putt'] else (.25,.90) if state['mode'] in ('CHIP','PITCH') else (.35,.92)
    if state.get('shot')==1 and state['mode']=='FULL' and state.get('lie')=='TEE':
        low,high=(1.,1.25) if state.get('strategy',{}).get('hazards_verified') else (.90,1.10)
    def requested(s):
        if is_bunker_state(s):
            raise RuntimeError(BUNKER_BLOCK_REASON)
        if POWER_MODEL:
            return POWER_MODEL.plan(s,for_selection=True)['unclipped_power']
        return target/s['club_range']
    def suitable(s):
        if s.get('lie')=='BUNKER' and s.get('club_name') not in ('SW','LW','PW','GW'):
            return False
        power=requested(s)
        if (s['mode']=='PUTT' and s['unit']=='ft' and s.get('target_hole_link')
                and target<=2.5 and .025<=power<=high):
            return True  # Use the verified selected putter for a tap-in.
        # The preferred window is not a hard ceiling. A verified driver is
        # usable near full power, including a clipped full-power lay-up when
        # conditions demand more than its displayed range. Searching for a
        # larger club here can cycle endlessly through the bag.
        driver=(s['mode']=='FULL' and s.get('unit')=='yd' and
                re.fullmatch(r'(?:DR|D|(?:[7-9]|1[0-3])(?:[.,][05])?D)',s.get('club_name') or '') is not None)
        return low<=power<=high or (driver and power>=low)
    if suitable(state):
        if DIAGNOSTICS:
            DIAGNOSTICS.event('club_selection_selected',club=state.get('club_name'),
                              club_range=state['club_range'],requested_power=requested(state),
                              reason='usable current club; driver may use near/full power')
        return state, frame
    direction = 1
    known_direction=None
    boundary_tried=False
    visited=set()
    observed_ranges={state['club_range']}
    for step in range(1,15):
        previous=state['club_range']
        need_larger=requested(state)>high
        key=(state.get('club_name'),previous,direction)
        if key in visited:
            # Some club controls wrap instead of staying unchanged at an end.
            # A repeated selection cycle can still leave us on the shortest
            # observed range for a short shot. Use its actual verified power
            # above the execution floor, rather than requiring a nonexistent
            # smaller club to satisfy the preferred power window.
            if (len(observed_ranges)>1 and not need_larger
                    and previous==min(observed_ranges) and requested(state)>=.025):
                if DIAGNOSTICS:
                    DIAGNOSTICS.event('club_selection_selected',club=state.get('club_name'),
                                      club_range=previous,reason='shortest observed range at repeated bag boundary')
                return state,frame
            raise RuntimeError('Club selection repeated a verified club/direction; stopped before another wheel input')
        visited.add(key)
        safety.check()
        if GAME is not None:
            GAME.require_turn(safety,sct,rect,state)
        if is_bunker_state(state):
            raise RuntimeError(BUNKER_BLOCK_REASON)
        # Wheel over the club's printed range; never wheel over the 3D course.
        move_to(rect['left']+round(.14*rect['width']), rect['top']+round(.96*rect['height']))
        safety.check()
        send(WHEEL, data=(direction*120)&0xffffffff)
        # The application opens CLUB SELECTION on hover over the range. It hides Swing,
        # so leave the range immediately after the wheel event, before OCR.
        park_on_meter(safety,rect)
        safety.wait(.20)
        new, frame = stable_state(safety, sct, rect,
                                 expected={'hole':state.get('hole'),'shot':state.get('shot')})
        if GAME is not None:
            GAME.inspect(frame,new,captured_at=new.get('captured_at'))
        learning_event('note_club',new)
        diagnostic_state(f'club_h{state.get("hole")}_s{state.get("shot")}_step_{step:02d}',
                         dict(step=step,wheel_direction=direction,before=state,after=new,
                              requested_power=requested(new)),frame)
        if DIAGNOSTICS:
            DIAGNOSTICS.event('club_range_observed',step=step,wheel_direction=direction,
                              before_club=state.get('club_name'),before_range=previous,
                              after_club=new.get('club_name'),after_range=new['club_range'],
                              target=new.get('distance'))
        if (new['mode'] != state['mode'] or new['unit'] != state['unit']
                or abs(new['distance']-target) > .1):
            raise RuntimeError('Shot mode or target changed during club selection; stopped')
        if suitable(new):
            return new, frame
        observed_ranges.add(new['club_range'])
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
            if not need_larger and known_direction==-direction:
                # A tap-in may be below the preferred power window even on
                # the verified shortest range. Accept it only above the same
                # hard minimum that guards execution; never invent a range.
                if requested(new)>=.025:
                    return new,frame
                raise RuntimeError('Shortest verified club range still requires power below the controlled minimum')
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
            # Start with the normal drag step until the meter registers motion.
            # Use finer movement near the target as soon as feedback is visible.
            step = 1 if last_power is not None and (remaining < .07 or required < .15) else 4
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
    scorecard_count = 0
    while time.perf_counter()-started < timeout:
        safety.wait(.35)
        captured_at=time.perf_counter()
        frame = capture(sct, rect)
        # Completion must be explicit and repeat on the central result panel.
        panel = ' '.join(ocr_variants(crop(frame, (.18,.16,.88,.88)), modes=(11,), scale=2))
        safety.check()
        completion_count = completion_count+1 if complete_text(panel) else 0
        card=None
        if not completion_count and observer is not None:
            from round_manager import scorecard
            card=scorecard(frame,ocr_variants)
            if card is not None:
                completion_count=1
        # Count repeated result evidence independently from the generic text.
        if card is not None:
            scorecard_count+=1
        else:
            scorecard_count=0
        if scorecard_count>=2:
            completion_count=2
        if completion_count >= 2:
            report=observer.report() if observer else {}
            if observer and not (report.get('scene_motion_observed') or report.get('moving_ball_candidate_tracked')):
                raise RuntimeError('Completion text appeared without observed shot motion; stopped unverified')
            diagnostic_state('completion_verified',dict(text=panel,scorecard=card,motion=report,hole=before.get('hole')),frame)
            return 'complete', dict(complete=True,motion=report,hole=before.get('hole'),scorecard=card)
        if card is not None:
            continue  # A result modal has no live distance/club HUD to OCR.
        state = shot_state(frame,conditions=GAME is not None)
        state['captured_at']=captured_at
        if GAME is not None:
            frame,captured_at=GAME.refresh_turn_frame(safety,sct,rect,frame,state,captured_at)
        context=GAME.inspect_ready(safety,sct,rect,frame,state,captured_at) if GAME is not None else None
        if context and context['concession']['eligible']:
            GAME.concede(safety,sct,rect,state)
            if GAME.completed:
                diagnostic_state('hole_complete_after_verified_concession',context,frame)
                return 'complete',dict(complete=True,source='verified concession plus repeated completion text')
        diagnostic_state(f'post_shot_read_{time.time_ns()}',state,frame)
        safety.check()
        # Verify that the prior stroke completed independently of whether the
        # *next* target is temporarily hidden. Full measurements are required
        # again by stable_state before another mouse-down can be sent.
        ready = (state['player'] and state['mode'] in SUPPORTED_SHOT_MODES
                 and state['club_unit'] in ('yd','ft')
                 and state['putt']==(state['mode']=='PUTT')
                 and swing_button(frame, rect) is not None and ready_meter(frame, rect) is not None)
        if context is not None:
            if (ready and tracker.count and context.get('format')=='SOLO_STROKE'
                    and context.get('active_player')==PLAYER
                    and context.get('ball_owner')==PLAYER
                    and context['permission'].get('reasons')==['turn evidence stale or absent']
                    and state_signature(state)==tracker.last):
                # A matching late OCR frame neither counts nor erases an
                # earlier fresh observation. Fresh turn checks still gate input.
                continue
            ready=ready and context['permission']['allowed']
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
                if outcome=='next' and state['shot']!=before['shot']+1:
                    diagnostic_state('unexpected_stroke_advance',dict(before=before,after=state,motion=report),frame)
                    raise RuntimeError('Stroke counter advanced by more than one; possible penalty or external stroke. '
                                       'Stop for a new plan; no automatic repeat shot')
                state['motion']=report
                state['movement_verification']='video motion plus stable stroke/HUD advance'
                stats=shot_stats(frame)
                if stats is not None and stats!=before.get('previous_stats'):
                    state['fresh_stats']=stats
            return outcome, state
    raise RuntimeError('Could not verify the next ready shot or hole completion; no repeat shot')


def shot_parameters(state):
    unit=state.get('unit')
    distance=state.get('distance')
    elevation=state.get('elevation_ft')
    return dict(hole=state.get('hole'),shot=state.get('shot'),
                yards=distance if unit=='yd' else None,distance=distance,distance_unit=unit,
                elevation_feet=elevation,elevation_detected=elevation is not None,
                observation_source='live_ui',distance_source=state.get('distance_source'),
                club=state.get('club_name'))


def run_hole(safety, sct, rect, reference, max_shots=15, one_shot=False, *, bunker_trial=None,
             bunker_reference=None):
    if GAME is None:
        raise RuntimeError('Verified game/turn coordinator is required before live play')
    if bunker_trial is not None and not one_shot:
        raise ValueError('Bunker calibration is limited to a single stroke')
    number=0
    expected_next = dict(zip(('hole','shot'),bunker_trial)) if bunker_trial is not None else None
    verified_practice_transition = False
    while max_shots==0 or number<max_shots:
        number+=1
        park_on_meter(safety,rect)
        # The transition already checked ownership on this timed solo turn.
        # stable_state and the pre-input guards still check fresh frames.
        if not verified_practice_transition:
            GAME.wait_for_turn(safety,sct,rect)
        if GAME.completed:
            return 'complete'
        timing = ShotTiming()
        # The prior transition was verified, but a new HUD may need time to
        # settle. Re-read it without guessing or repeating the previous shot.
        state, frame = stable_state(safety, sct, rect,
                                    timeout=45.0 if expected_next else 30.0,
                                    expected=expected_next)
        timing.record('detection', timing.started)
        timing.require_decision_budget()
        expected_next = None
        # Stop on sand already observed by stable_state. A later OCR miss must
        # not replace that evidence with an unknown lie before the bunker guard.
        if bunker_trial is not None or not is_bunker_state(state):
            safety_started = timing.clock()
            GAME.require_turn(safety,sct,rect,state)
            timing.record('safety', safety_started)
            if DIAGNOSTICS:
                DIAGNOSTICS.shot_stage('observation_validation_passed',**shot_parameters(state))
        DEBUG.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG/f'before_shot_{number:02d}.png'), frame)
        if is_bunker_state(state) and bunker_trial is None:
            # Report the provisional model before any club-selection input.
            # Its power=None is deliberately non-executable. arm_stroke also
            # blocks sand if the lie changes during a later reread.
            try:
                estimate=build_plan(state, POWER_MODEL or PowerCalibration())
            except ValueError as exc:
                estimate=dict(power=None,automatic_swing_allowed=False,error=str(exc))
            diagnostic_state(f'shot_{number:02d}_bunker_blocked',dict(state=state,plan=estimate),frame)
            if DIAGNOSTICS:
                DIAGNOSTICS.event('bunker_swing_blocked',state=state,plan=estimate)
                shadow=learning_event('begin_shot',DIAGNOSTICS.shot_id(number),state,estimate,POWER_MODEL)
                if shadow:
                    DIAGNOSTICS.event('shadow_recommendation',shot=number,prediction=shadow)
            raise RuntimeError(BUNKER_BLOCK_REASON)
        if bunker_trial is None:
            previous_state = state
            club_started = timing.clock()
            timing.require_decision_budget()
            state, frame = select_club(safety, sct, rect, state, frame)
            timing.record('club_selection', club_started)
            if state is not previous_state or not state.get('game_context', {}).get('practice_clock'):
                safety_started = timing.clock()
                GAME.require_turn(safety,sct,rect,state)
                timing.record('safety', safety_started)
        if DIAGNOSTICS:
            DIAGNOSTICS.shot_stage('club_selected',**shot_parameters(state))
            state['previous_stats']=shot_stats(frame)
        planned_parameters=shot_parameters(state)
        planning_started = timing.clock()
        plan=(build_plan(state, POWER_MODEL) if bunker_trial is None
              else bunker_calibration_plan(state,*bunker_trial,reference=bunker_reference))
        timing.record('planning', planning_started)
        power=plan['power']
        diagnostic_state(f'shot_{number:02d}_plan',dict(state=state,plan=plan),frame)
        if DIAGNOSTICS:
            DIAGNOSTICS.event('shot_planned',shot=number,state=state,plan=plan)
            DIAGNOSTICS.shot_stage('shot_parameters_calculated',**planned_parameters,
                                   power_pct=100*power if power is not None else None,
                                   spin_indicator=state.get('spin_indicator'),spin_advice=plan.get('spin_advice'))
        if plan.get('automatic_swing_allowed') is False:
            raise RuntimeError(plan.get('block_reason',BUNKER_BLOCK_REASON))
        if DIAGNOSTICS:
            shadow=learning_event('begin_shot',DIAGNOSTICS.shot_id(number),state,plan,POWER_MODEL)
            if shadow:
                DIAGNOSTICS.event('shadow_recommendation',shot=number,prediction=shadow)
                selected=shadow['recommendation'].get('selected')
                print('SHADOW ONLY recommendation: '+json.dumps(selected or shadow['recommendation'].get('abstentions')),
                      flush=True)
        meter = ready_meter(frame, rect)
        if meter is None:
            raise RuntimeError('Saved mark and visible meter disagree')
        print(f'SHOT {number}: hole={state["hole"]}, stroke={state["shot"]}, '
              f'{state["distance"]:g} {state["unit"]}, {state["mode"]}, '
              f'club range={state["club_range"]:g} {state["club_unit"]}, '
              f'{"calibration trial" if bunker_trial is not None else "baseline"} power={power:.1%}', flush=True)
        # A stroke may have begun on release even if an exception follows.
        # Never retry an interrupted stroke in the same controller run.
        observer=ShotObserver(rect,frame) if DIAGNOSTICS else None
        if observer:
            observer.start()
        captured_before=deepcopy(state)
        accuracy_trace=None
        try:
            timing.require_decision_budget()
            safety_started = timing.clock()
            prior_deadline=(GAME.arm_stroke(safety,sct,rect,state,reference) if bunker_trial is None
                            else GAME.arm_stroke(safety,sct,rect,state,reference,bunker_trial=plan))
            timing.record('safety', safety_started)
            try:
                power=plan['power']
                captured_before=deepcopy(state)  # Freeze the final verified state before input.
                if DIAGNOSTICS:
                    DIAGNOSTICS.state(f'shot_{number:02d}_before_capture',captured_before)
                execution_parameters=shot_parameters(captured_before)
                if execution_parameters!=planned_parameters:
                    if DIAGNOSTICS:
                        DIAGNOSTICS.shot_stage('shot_parameters_changed',planned=planned_parameters,
                                               before_swing=execution_parameters)
                    raise RuntimeError('Shot parameters changed after planning; no swing')
                timing.before_swing()
                if DIAGNOSTICS:
                    DIAGNOSTICS.shot_stage('swing_initiated',label='SHOT_EXECUTION',
                                           **execution_parameters,aim_angle=0.,
                                           power_pct=100*power,retry_count=0,measurements_match_plan=True)
                actual = execute_stroke(safety, sct, rect, reference, meter, power, drag_swing, accuracy_click, timing)
            finally:
                safety.deadline=prior_deadline
            plan['performance'] = timing.summary()
            if observer:
                observer.arm()
            if DIAGNOSTICS:
                DIAGNOSTICS.event('shot_performance', shot=number, **plan['performance'])
                # Copy after release/accuracy, before waiting for the next HUD.
                # The live click routine and its timing receive no new arguments.
                accuracy_trace=read_shot_accuracy_trace(DIAGNOSTICS.shot_id(number),number)
                DIAGNOSTICS.state(f'shot_{number:02d}_swing_capture',dict(
                    shot_id=DIAGNOSTICS.shot_id(number),observed_meter_power_fraction=actual,
                    accuracy_trace=accuracy_trace))
                DIAGNOSTICS.flush_meters(number)
                DIAGNOSTICS.flush_console()
                DIAGNOSTICS.event('accuracy_input_sent',shot=number,power=actual,
                                  observed_power_pct=100*actual)
            wait_limit=300 if state.get('game_context',{}).get('format') not in (None,'SOLO_STROKE') else 90
            outcome, after = wait_until_next(safety, sct, rect, state,observer=observer,timeout=wait_limit)
            if DIAGNOSTICS:
                DIAGNOSTICS.shot_stage('post_shot_state_detected',hole=captured_before.get('hole'),
                                       shot=captured_before.get('shot'),outcome=outcome,
                                       next_hole=(after or {}).get('hole'),next_shot=(after or {}).get('shot'))
        finally:
            if observer:
                observer.close()
                observer.save(number)
            if DIAGNOSTICS:
                DIAGNOSTICS.flush_meters(number)
        shot_capture=None
        if DIAGNOSTICS:
            shot_capture=build_shot_capture(DIAGNOSTICS.shot_id(number),captured_before,after,
                                            actual,plan,accuracy_trace,outcome)
            DIAGNOSTICS.save_shot_capture(number,shot_capture)
            DIAGNOSTICS.event('new_shot_capture',shot=number,complete=shot_capture['complete'],
                              missing=shot_capture['missing'])
        log = dict(attempt=number, before=captured_before, observed_power=actual, outcome=outcome,
                   after=after, accuracy_trace=accuracy_trace, shot_capture=shot_capture)
        if bunker_trial is not None:
            next_lie=(after or {}).get('lie')
            log['bunker_trial'] = dict(plan=plan, next_lie=next_lie,
                                      escaped=outcome=='complete' or next_lie in ('GREEN','FRINGE','FAIRWAY','ROUGH','HEAVY ROUGH'))
            print(f'Bunker calibration result: next lie={next_lie}; escaped={log["bunker_trial"]["escaped"]}.',flush=True)
        (DEBUG/f'shot_{number:02d}.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
        diagnostic_state(f'shot_{number:02d}_result',log)
        if DIAGNOSTICS:
            DIAGNOSTICS.event('shot_verified',shot=number,outcome=outcome,after=after)
            if (POWER_MODEL and after and after.get('fresh_stats')
                    and (after.get('game_context') or {}).get('stats_owner') in (None,PLAYER)):
                after_context=after.get('game_context',{})
                if (after_context.get('stats_owner')==PLAYER and after_context.get('confidence',{}).get('stats_owner',0)>=.85
                        or after_context.get('format')=='SOLO_STROKE' and after_context.get('players')==[PLAYER]):
                    state.get('game_context',{})['verified_stats_owner']=PLAYER
                POWER_MODEL.record(ROOT/'wgt_calibration/power_observations.json',state,after['fresh_stats'],actual,
                                   DIAGNOSTICS.shot_id(number))
            learning_event('finish_shot',DIAGNOSTICS.shot_id(number),captured_before,after,actual,plan,
                           evidence=dict(session_path=str(DIAGNOSTICS.path),
                                         source='controller stable result and existing shot/session logs',
                                         accuracy_trace=accuracy_trace,shot_capture=shot_capture))
        if outcome == 'complete':
            GAME.completed_hole=state.get('hole')
            print('HOLE COMPLETION OBSERVED on screen.', flush=True)
            return 'complete'
        print('NEXT READY SHOT VERIFIED; reacquiring complete HUD before swing', flush=True)
        expected_next = {'hole': after.get('hole'), 'shot': after.get('shot')}
        next_context = after.get('game_context', {})
        verified_practice_transition = (next_context.get('practice_clock') is True
                                        and next_context.get('format') == 'SOLO_STROKE'
                                        and next_context.get('permission', {}).get('allowed') is True)
        if one_shot:
            return 'one_shot'
    raise RuntimeError(f'Shot cap ({max_shots}) reached; hole completion not observed')


def main(argv=None):
    global DIAGNOSTICS,POWER_MODEL,GAME,LEARNING
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='dependencies/calibration only; no capture or input')
    group.add_argument('--observe', action='store_true', help='read one current screenshot; no input')
    group.add_argument('--one-shot', action='store_true', help='one verified stroke including accuracy')
    group.add_argument('--controlled-test', action='store_true', help='bounded one-hole live test with automatic evidence logging')
    parser.add_argument('--countdown', type=int, default=8)
    parser.add_argument('--max-shots', type=int, default=15)
    parser.add_argument('--max-seconds',type=int,default=900)
    parser.add_argument('--holes',type=int,default=1,help='verified single-player holes to play, 1..9')
    parser.add_argument('--activate',action='store_true',help='raise WGT; verified title-bar click if Windows refuses focus')
    parser.add_argument('--solo-confirmed',action='store_true',
                        help='explicitly declare this run is a single-player game; still requires live player and swing verification')
    parser.add_argument('--practice-shot-clock',type=int,choices=(0,60,90),default=0,
                        help='simulate a 60- or 90-second clock in confirmed solo play, including OCR and the complete stroke')
    parser.add_argument('--resume-scorecard-hole',type=int,choices=range(1,19),default=None,
                        help='resume from the scorecard of this completed solo hole; holes counts the remaining holes')
    parser.add_argument('--resume-intro-hole',type=int,choices=range(1,19),default=None,
                        help='verify and continue past the course-info panel for this solo hole')
    parser.add_argument('--observe-seconds',type=int,default=0,help='continuously collect context evidence in --observe mode; no input')
    parser.add_argument('--concede-feet',type=float,default=None,help='override opponent-putt courtesy threshold in feet; 0 disables concessions')
    parser.add_argument('--bunker-calibration',type=int,nargs=2,metavar=('HOLE','STROKE'),
                        help='perform one explicitly scoped solo bunker calibration stroke')
    parser.add_argument('--bunker-reference',type=Path,
                        help='use a matching recorded recovery to size a single bunker layup')
    args = parser.parse_args(argv)
    if args.bunker_calibration is not None and (not args.one_shot or not args.solo_confirmed
            or not 1<=args.bunker_calibration[0]<=18 or args.bunker_calibration[1]<1):
        parser.error('bunker-calibration requires --one-shot --solo-confirmed and a valid hole and stroke')
    if args.bunker_reference is not None and args.bunker_calibration is None:
        parser.error('bunker-reference requires a scoped bunker-calibration stroke')
    if args.practice_shot_clock and not args.solo_confirmed:
        parser.error('practice-shot-clock requires --solo-confirmed')
    if args.resume_scorecard_hole is not None and (not args.solo_confirmed or args.observe or args.check or args.one_shot):
        parser.error('resume-scorecard-hole requires live confirmed solo round play')
    if args.resume_intro_hole is not None and (not args.solo_confirmed or args.observe or args.check or args.one_shot
                                             or args.resume_scorecard_hole is not None):
        parser.error('resume-intro-hole requires a confirmed solo round starting at the introduction')
    if not 1<=args.holes<=9 or args.holes>1 and (not args.solo_confirmed or args.one_shot):
        parser.error('holes must be 1..9; multiple holes require --solo-confirmed and cannot use --one-shot')
    if not 0<=args.observe_seconds<=3600 or args.observe_seconds and not args.observe:
        parser.error('observe-seconds must be 0..3600 and requires --observe')
    if args.concede_feet is not None and (not math.isfinite(args.concede_feet) or args.concede_feet<0):
        parser.error('concede-feet must be a finite nonnegative number')
    if not 0 <= args.countdown <= 60 or not 0 <= args.max_shots <= 30 or not (args.max_seconds==0 or 30<=args.max_seconds<=3600):
        parser.error('countdown must be 0..60; max-shots 0..30; max-seconds 0 or 30..3600 (0 disables a cap)')
    safety = None
    controller_lock=None
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
        policy=GamePolicy.load(ROOT/'wgt_calibration/game_policy.json',override=args.concede_feet)
        if not args.check:
            def team_ready(frame):
                h,w=frame.shape[:2]
                local=dict(left=0,top=0,width=w,height=h)
                return ready_meter(frame,local) is not None and swing_button(frame,local) is not None
            GAME=GameCoordinator(ContextObserver(ROOT,DIAGNOSTICS,policy=policy,
                              ready_reader=team_ready,solo_confirmed=args.solo_confirmed,
                              practice_shot_seconds=args.practice_shot_clock),
                                 capture,shot_state,move_to,DIAGNOSTICS,
                                 target_visible=point_in_wgt,policy=policy,
                                 attempt_store=ROOT/'wgt_calibration/concession_pending.json',
                                 putt_reader=reread_putt_badge,
                                 read_state_from_source=lambda frame,source:shot_state(frame,target_source=source),
                                 elevation_reader=reread_elevation,
                                 distance_reader=lambda frame,sign:read_sign_distance(frame,sign)
                                     if find_distance_sign(frame)==tuple(sign or ()) else None)
        reference = StrokeReference.from_data(json.loads(RECORD.read_text(encoding='utf-8')))
        calibration = json.loads(CALIBRATION.read_text(encoding='utf-8'))
        fraction = float(calibration['accuracy_x_fraction'])
        if not math.isfinite(fraction) or not .25 < fraction < .90:
            raise ValueError('Invalid saved accuracy calibration')
        pytesseract.get_tesseract_version()
        POWER_MODEL=PowerCalibration.load(ROOT/'wgt_calibration/power_observations.json',ROOT/'wgt_shot_logs/shots.csv')
        print('WGT one-hole controller | F9 emergency stop | aim/camera unchanged', flush=True)
        print('Supported verified modes: '+', '.join(SUPPORTED_SHOT_MODES),flush=True)
        print(f'Opponent-putt concession threshold: {policy.concession_max_feet:g} FT (0 disables)',flush=True)
        print(f'Saved stroke: {reference.width}x{reference.height}, meter anchor={reference.anchor}, '
              f'drag={reference.drag_seconds:.2f}s, accuracy interval={reference.accuracy_delay:.2f}s', flush=True)
        if args.check:
            print('DEPENDENCIES AND RECORDING CHECK PASSED. No live-game test performed.', flush=True)
            profile=ContextObserver(ROOT,policy=policy)
            print('FORMAT/TURN LAYOUT: '+('calibrated' if profile.layout_verified else
                  'uncalibrated; normal live input blocked until local evidence is calibrated'),flush=True)
            if args.solo_confirmed:
                print('SOLO user-attested path enabled for this run; screen checks still required.',flush=True)
            return 0
        if args.solo_confirmed:
            print('SOLO USER-CONFIRMED MODE: only an independently visible Tigerwould52 HUD and ready swing/meter can enable input; any multiplayer evidence blocks play.',flush=True)
        if not args.observe and not GAME.observer.layout_verified and not args.solo_confirmed:
            raise RuntimeError('Game format/active-turn layout is uncalibrated. Use --observe --observe-seconds 60 '
                               'to collect local evidence before live play; no game input sent')
        if not args.observe:
            controller_lock=ControllerLock(DEBUG/'live_controller.lock')
            controller_lock.acquire()
            try:
                LEARNING=ShadowLearning(ROOT,session_id=DIAGNOSTICS.path.name)
                learning_event('sync_existing')
                learning_event('write_report')
                print('Persistent learning: SHADOW ONLY; live decisions and bunker block unchanged.',flush=True)
            except Exception as exc:
                print('SHADOW LEARNING startup unavailable: '+str(exc),flush=True)
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
        DIAGNOSTICS.screen_id=display_id(hwnd)
        with mss.MSS() as sct:
            diagnostic_state('initial_geometry',dict(rect=rect,foreground=(user32.GetForegroundWindow()==hwnd),
                                                    args=vars(args)),capture(sct,rect))
            if args.observe:
                # Observation is deliberately available without activating the application.
                readings=[]
                observed_started=time.perf_counter()
                i=0
                while i<2 or time.perf_counter()-observed_started<args.observe_seconds:
                    if key_down(F9):
                        raise KeyboardInterrupt('F9 observation stop')
                    captured_at=time.perf_counter()
                    frame=capture(sct,rect)
                    reading=inspect_screen(frame,rect)
                    GAME.inspect(frame,reading['state'],captured_at=captured_at)
                    readings.append(reading)
                    diagnostic_state(f'detection_read_{i}',reading,frame)
                    DIAGNOSTICS.image(f'detection_read_{i}_annotated',detection_overlay(frame,rect,reading))
                    i+=1
                    if len(readings)>2:
                        readings.pop(0)
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
                if args.resume_scorecard_hole is not None:
                    from round_manager import continue_to_tee
                    continue_to_tee(sys.modules[__name__],safety,sct,rect,args.resume_scorecard_hole)
                if args.resume_intro_hole is not None:
                    from round_manager import await_tee
                    await_tee(sys.modules[__name__],safety,sct,rect,args.resume_intro_hole,require_intro=True)
                if args.holes>1:
                    from round_manager import run_round
                    final_data=run_round(sys.modules[__name__],safety,sct,rect,reference,args.holes,args.max_shots)
                    status='round_completion_verified'
                else:
                    options={'bunker_trial':args.bunker_calibration} if args.bunker_calibration is not None else {}
                    if args.bunker_reference is not None:
                        options['bunker_reference']=json.loads(args.bunker_reference.read_text(encoding='utf-8'))
                    result=run_hole(safety, sct, rect, reference, args.max_shots, args.one_shot,**options)
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
        try:
            if safety is not None:
                safety.close()
        finally:
            if controller_lock is not None:
                controller_lock.close()
        if LEARNING is not None:
            try:
                learning_event('write_report')
                LEARNING.close()
            except Exception as exc:
                final_data['learning_close_error']=str(exc)
            finally:
                LEARNING=None
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
        GAME=None


if __name__ == '__main__':
    raise SystemExit(main())
