"""WGT Golf single-hole controller V47 (Windows, experimental).

The controller observes the *existing* WGT window. It never logs in,
resizes the window, drags the flag, or changes the camera. After each attempted
shot it verifies a visible change before continuing. It stops on ambiguity.

Installation: put this file alongside swing_button.png and wgt_accuracy.json
in C:\\Users\\rodne\\WGT_Bot (the calibration file is optional in V47).
Needs: python -m pip install mss opencv-python numpy pytesseract
Windows Tesseract executable must be installed if OCR is used.

F9 or Ctrl+C: emergency stop (always releases left mouse button).
This is a prototype, not guaranteed to hole out; run in practice/solo mode
where automation is allowed. Never use on a paid/competitive match.
"""
import ctypes
import ctypes.wintypes as wt
import json
import math
import re
import time
from collections import deque
from pathlib import Path

import cv2
import mss
import numpy as np
import pytesseract

ROOT = Path(__file__).resolve().parent
DEBUG = ROOT / "wgt_debug"
DEBUG.mkdir(exist_ok=True)
RECORD = ROOT / "wgt_calibration" / "last_manual_stroke.json"
SWING_TEMPLATE = ROOT / "swing_button.png"
USER32 = ctypes.windll.user32
VK_F9 = 0x78
MOUSE_DOWN = 0x0002
MOUSE_UP = 0x0004
MOUSE_MOVE = 0x0001
MOUSE_ABSOLUTE = 0x8000
MOUSE_VIRTUAL = 0x4000
MAX_SHOTS = 9
SHOT_TIMEOUT = 22.0
RETURN_TIMEOUT = 3.2
CAMERA_CHANGE_FORBIDDEN = True


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


# Exact ctypes input layout is important for 64-bit Windows.
ULONG_PTR = ctypes.c_size_t
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ULONG_PTR)]
class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]
class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("data", INPUT_UNION)]

if hasattr(USER32, "SendInput"):
    USER32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    USER32.SendInput.restype = wt.UINT


def key_down(vk):
    return bool(USER32.GetAsyncKeyState(vk) & 0x8000)


def stop_if_requested(hwnd=None):
    if key_down(VK_F9):
        raise KeyboardInterrupt("F9 emergency stop")
    if hwnd is not None and USER32.GetForegroundWindow() != hwnd:
        raise RuntimeError("WGT lost focus. No more input sent.")


def send(flags, x=0, y=0):
    msg = INPUT(type=0, data=INPUT_UNION(mi=MOUSEINPUT(x, y, 0, flags, 0, 0)))
    if USER32.SendInput(1, ctypes.byref(msg), ctypes.sizeof(INPUT)) != 1:
        raise RuntimeError("SendInput failed; check permissions / game focus")


def mouse_up():
    try:
        send(MOUSE_UP)
    except Exception:
        USER32.mouse_event(MOUSE_UP, 0, 0, 0, 0)


def move_to(x, y):
    vx, vy = USER32.GetSystemMetrics(76), USER32.GetSystemMetrics(77)
    vw, vh = USER32.GetSystemMetrics(78), USER32.GetSystemMetrics(79)
    if min(vw, vh) < 2:
        raise RuntimeError("Windows virtual desktop is not available")
    ax = max(0, min(65535, round((x - vx) * 65535 / (vw - 1))))
    ay = max(0, min(65535, round((y - vy) * 65535 / (vh - 1))))
    send(MOUSE_MOVE | MOUSE_ABSOLUTE | MOUSE_VIRTUAL, ax, ay)


def sleep_safe(sec, hwnd=None):
    till = time.perf_counter() + sec
    while time.perf_counter() < till:
        stop_if_requested(hwnd)
        time.sleep(min(.01, max(0, till - time.perf_counter())))


def setup():
    try:
        USER32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        USER32.SetProcessDPIAware()
    for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            break


def find_wgt():
    found = []
    CALLBACK = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def callback(hwnd, unused):
        if not USER32.IsWindowVisible(hwnd) or USER32.IsIconic(hwnd):
            return True
        n = USER32.GetWindowTextLengthW(hwnd)
        if n:
            name = ctypes.create_unicode_buffer(n+1)
            USER32.GetWindowTextW(hwnd, name, n+1)
            if "wgt golf" in name.value.lower():
                found.append(hwnd)
        return True
    cb = CALLBACK(callback)
    USER32.EnumWindows(cb, 0)
    if not found:
        raise RuntimeError("WGT Golf window not found")
    return found[0]


def rect_of(hwnd):
    r = RECT()
    if not USER32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError("Cannot locate WGT window")
    x,y,w,h = r.left,r.top,r.right-r.left,r.bottom-r.top
    if w < 650 or h < 440:
        raise RuntimeError("WGT window too small; no automated clicks")
    return {"left":x,"top":y,"width":w,"height":h}


def capture(sct, r):
    return np.array(sct.grab(r))[:,:,:3].copy()


def clipregion(img, xyxy):
    H,W = img.shape[:2]
    x1,y1,x2,y2=xyxy
    return img[max(0,int(y1*H)):min(H,int(y2*H)),
               max(0,int(x1*W)):min(W,int(x2*W))]


def ocr(patch, psm=6):
    if patch.size == 0:
        return ""
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(gray,config=f"--psm {psm}").upper()


def shot_state(frame):
    """Use tight, tested HUD crops; never reuse the previous shot's stats."""
    # On screenshots from this WGT version, distance is on the first HUD row.
    # Putt distance is also written in large text on the floating sign.
    upper = ocr(clipregion(frame, (.115, .055, .195, .120)), 7)
    flag = ocr(clipregion(frame, (.467, .232, .555, .292)), 7)
    lower = ocr(clipregion(frame, (.005, .946, .195, .998)), 6)
    club_text = ocr(clipregion(frame, (.075, .947, .19, .997)), 6)
    putt_label = ocr(clipregion(frame, (.01, .949, .07, .998)), 7)
    right = ocr(clipregion(frame, (.82, .055, .97, .117)), 6)
    game_text = (upper+' '+flag+' '+lower+' '+right).upper()
    putt = 'PUTT' in putt_label or 'PUTT' in lower
    # The top-left HUD number is easier for OCR than its abbreviated unit.
    top_nums = re.findall(r"(?<!\d)(\d{1,3}(?:[.]\d+)?)(?!\d)", upper)
    flag = re.sub(r"(?<=\d),(?=\d)", ".", flag)
    flag_nums = re.findall(r"(?<!\d)(\d{1,3}(?:[.]\d+)?)(?!\d)", flag)
    dist = None
    unit = None
    if putt:
        nums = [float(v) for v in flag_nums if 0.8 <= float(v) <= 300]
        if nums:
            dist = nums[0]
        else:
            nums = [float(v) for v in top_nums if .8 <= float(v) <= 300]
            if nums: dist = nums[-1]
        unit = 'ft' if dist is not None else None
    else:
        nums = [float(v) for v in top_nums if 2 <= float(v) <= 600]
        if nums:dist=nums[-1];unit='yd'
    club = None
    if not putt:
        # Bottom-left selected club is shown next to its yardage, e.g. 9I 115.
        num = re.findall(r"(?<!\d)(\d{2,3})(?!\d)",lower+' '+club_text)
        nums=[int(v) for v in num if 20<=int(v)<=400]
        if nums:club=float(nums[-1])
    shot=None
    if re.search(r"[12]ST\s*SHOT",right):shot=1
    elif re.search(r"2(?:ND|4D|AD|D)\s*SHOT",right):shot=2
    elif re.search(r"3RD\s*SHOT",right):shot=3
    elif re.search(r"4TH\s*SHOT",right):shot=4
    complete=bool(re.search(r"HOLE\s*(?:COMPLETE|FINISHED)|NEXT\s*HOLE", game_text))
    return dict(distance=dist,unit=unit,putt=bool(putt),hole=None,
                shot=shot,club_yards=club,complete=complete,
                raw=game_text.replace('\n',' ')[:180])


def swing_button(frame, rect):
    """Restrict detection to the right of the WGT window, including putting."""
    H,W=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    best=(-1,None)
    if SWING_TEMPLATE.exists():
        tpl=cv2.imread(str(SWING_TEMPLATE),cv2.IMREAD_GRAYSCALE)
        if tpl is not None:
            area=gray[int(H*.35):int(H*.92),int(W*.69):int(W*.99)]
            for scale in (.7,.85,1.,1.15,1.3):
                t=cv2.resize(tpl,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                if min(t.shape)<12 or t.shape[0]>=area.shape[0] or t.shape[1]>=area.shape[1]:
                    continue
                _,v,_,pos=cv2.minMaxLoc(cv2.matchTemplate(area,t,cv2.TM_CCOEFF_NORMED))
                if v>best[0]:
                    best=(v,(rect['left']+int(W*.69)+pos[0]+t.shape[1]//2,
                             rect['top']+int(H*.35)+pos[1]+t.shape[0]//2))
    if best[0]>=.83:
        return best[1],best[0]
    # Fallback detects circle near the 'Swing' text. Must verify its text.
    crop=gray[int(H*.49):int(H*.91),int(W*.79):int(W*.99)]
    circles=cv2.HoughCircles(cv2.GaussianBlur(crop,(7,7),1.4),
                    cv2.HOUGH_GRADIENT,1.2,minDist=35,param1=115,param2=36,
                    minRadius=max(17,int(W*.024)),maxRadius=int(W*.058))
    if circles is not None:
        for x,y,r in np.asarray(circles).reshape(-1,3):
            x=int(x+W*.79);y=int(y+H*.49);r=int(r)
            zone=frame[max(0,y-r):min(H,y+r),max(0,x-r):min(W,x+r)]
            label=ocr(zone,7)
            in_right = .84*W < x < .98*W
            in_lower = .57*H < y < .88*H
            plausible_radius = .027*W < r < .060*W
            if in_right and in_lower and plausible_radius:
                return (rect['left']+x,rect['top']+y),.90 if "SWING" in label else .84
    return None,max(0,best[0])


def meter_rect(rect):
    return {'left':rect['left']+int(rect['width']*.13),
            'top':rect['top']+int(rect['height']*.932),
            'width':int(rect['width']*.77),
            'height':int(rect['height']*.063)}


def mask_columns(band, low, high, ratio=.23):
    hsv=cv2.cvtColor(band,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,low,high)
    n=max(2,int(band.shape[0]*ratio))
    return np.flatnonzero((mask>0).sum(axis=0)>=n)


def blue_target(band):
    if band.size==0:
        return None
    # Light-blue accuracy zone; ignore scattered pixels / cyan button.
    xs=mask_columns(band,(87,32,50),(130,255,255),.24)
    if len(xs)<max(35,int(band.shape[1]*.15)):
        return None
    # A few white ticks may split the blue. Use 5px adjacency gaps.
    groups=np.split(xs,np.flatnonzero(np.diff(xs)>6)+1)
    groups=[g for g in groups if len(g)>max(25,int(band.shape[1]*.11))]
    if not groups:
        return None
    first=min(int(g[0]) for g in groups)
    last=max(int(g[-1]) for g in groups)
    # Blue target must sit in the middle/right portion of the meter.
    if first<band.shape[1]*.30 or last>band.shape[1]*.995:
        return None
    return int((first+last)/2)


def green_front(band, target):
    if band.size==0 or target is None:
        return None
    xs=mask_columns(band,(35,58,50),(85,255,255),.23)
    if len(xs)<12:
        return None
    groups=np.split(xs,np.flatnonzero(np.diff(xs)>7)+1)
    groups=[g for g in groups
            if len(g)>=12 and int(g[0])<target*.66 and int(g[-1])<band.shape[1]*.97]
    if not groups:return None
    candidate=max(groups,key=len)
    return int(candidate[-1])


def annotated(img, target, edge):
    copy=img.copy()
    H=copy.shape[0]
    if target is not None: cv2.line(copy,(target,0),(target,H-1),(255,255,255),2)
    if edge is not None:cv2.line(copy,(edge,0),(edge,H-1),(0,0,255),2)
    return copy


def reference_pull():
    """Extract drag length from actual user's manual recording."""
    default=.32
    if RECORD.is_file():
        try:
            data=json.loads(RECORD.read_text(encoding='utf-8'))
            samples=data['samples']
            a=next(e for e in samples if e['down'])
            b=next(e for i,e in enumerate(samples[1:],1)
                   if not e['down'] and samples[i-1]['down'])
            return min(.65,max(.10,a['y']-b['y']))
        except Exception:
            pass
    return default


def select_power(state, rect):
    """Calibrated fallback for putts; crude for shots without known club."""
    distance=state['distance']
    if distance is None:raise RuntimeError("Can't read distance; no blind shot")
    if state['putt']:
        # Uses recorded 17.5-foot manual putt as reference, not a fixed 18%.
        ft=distance if state['unit']=='ft' else distance*3.
        ratio=math.sqrt(max(.05,ft)/17.5)
        return min(.60,max(.055,reference_pull()*ratio)), f'putt {ft:.1f}ft'
    # For the user's explicit tee policy, first shot is full power when verified.
    if state['shot']==1:
        return .83,"tee shot: maximum safe upward travel (100% requested)"
    if state['unit']!='yd' or not state['club_yards']:
        raise RuntimeError("Can't read selected club and yardage; no guessed shot")
    # WGT distance/power isn't universally linear. This is an initial cap.
    ratio=distance/state['club_yards']
    return min(.75,max(.10,.60*ratio)), f'approach estimate {distance}/{state["club_yards"]} yd'


def drag_swing(hwnd, rect, pos, frac):
    """Single held upward drag, sent via SendInput, then unconditional mouse-up."""
    x,y=pos
    screen_top=min((m['top'] for m in monitors_for_rect(rect)), default=rect['top'])
    upper=max(screen_top+20,rect['top']+12)
    available=max(0,y-upper)
    if available<35:
        raise RuntimeError("Not enough room to pull the Swing control")
    # Reference is normalized to WGT window height, bounded to desktop.
    amount=min(available,round(rect['height']*frac))
    print(f"DRAG: {amount}px upward (available {available}px)",flush=True)
    move_to(x,y);sleep_safe(.12,hwnd)
    try:
        stop_if_requested(hwnd);send(MOUSE_DOWN)
        count=max(25,int(amount/4))
        for i in range(1,count+1):
            stop_if_requested(hwnd)
            yy=max(upper,y-round(i*amount/count))
            move_to(x,yy)
            sleep_safe(.009,hwnd)
    finally:
        mouse_up()
    return amount


def monitors_for_rect(rect):
    # Use the WGT screen top; no crossing to a second, side-by-side monitor.
    x=rect['left']+rect['width']//2
    y=rect['top']+rect['height']//2
    result=[]
    if getattr(monitors_for_rect,'monitors',None):
        for m in monitors_for_rect.monitors:
            if m['left']<=x<m['left']+m['width'] and m['top']<=y<m['top']+m['height']:
                result.append(m)
    return result


def accuracy_click(hwnd, sct, rect, target):
    meter=meter_rect(rect)
    last=None
    speed=None
    history=deque(maxlen=80)
    start=time.perf_counter()
    clicked=False
    while time.perf_counter()-start<RETURN_TIMEOUT:
        stop_if_requested(hwnd)
        t=time.perf_counter()
        band=capture(sct,meter)
        now=time.perf_counter()
        edge=green_front(band,target)
        if edge is not None:
            history.append((now,edge))
            if last is not None:
                dt=now-last[0];dx=edge-last[1]
                if .004<dt<.3 and 0<dx<120:
                    v=dx/dt
                    speed=v if speed is None else .6*speed+.4*v
            last=(now,edge)
            # Predict the edge's position by the time WGT receives this click.
            lead=min(32,max(5, int((speed or 240)*.028)))
            if target-lead<=edge<=target+5:
                send(MOUSE_DOWN)
                sleep_safe(.018,hwnd)
                send(MOUSE_UP)
                clicked=True
                print(f'ACCURACY CLICK: green={edge} target={target} lead={lead}',flush=True)
                break
        if len(history)%8==1:
            # Save a small diagnostic set if no click.
            pass
        time.sleep(.002)
    # Always save an annotated end snapshot for debugging.
    band=capture(sct,meter)
    cv2.imwrite(str(DEBUG/'v47_last_meter.png'),
                annotated(band,target,history[-1][1] if history else None))
    if not clicked:
        print('RETURN METER NOT VERIFIED; no blind accuracy click',flush=True)
    return clicked


def wait_until_next(hwnd,sct,rect,before,number):
    """Require target-distance or shot-count change; never blindly repeat."""
    started=time.perf_counter()
    last=None
    equal=0
    while time.perf_counter()-started<SHOT_TIMEOUT:
        stop_if_requested(hwnd)
        sleep_safe(1.5,hwnd)
        frame=capture(sct,rect)
        state=shot_state(frame)
        if state['complete']:
            return ('complete',state)
        # Hole-end panel is often centered rather than in a HUD corner.
        if time.perf_counter()-started > 4:
            panel_text=ocr(clipregion(frame,(.22,.22,.78,.72)),11)
            if re.search(r'NEXT\s*HOLE|HOLE\s*COMPLETE|SCORECARD|HOLED\s*OUT',panel_text):
                return ('complete',state)
        if state['hole'] and before['hole'] and state['hole']!=before['hole']:
            return ('complete',state)
        new_distance=(state['distance'] is not None and before['distance'] is not None
                      and state['unit']==before['unit'] and
                      abs(state['distance']-before['distance'])>
                      (0.7 if state['unit']=='ft' else 1.5))
        new_shot=(state['shot'] is not None and before['shot'] is not None
                  and state['shot']>before['shot'])
        if new_distance or new_shot:
            signature=(state['distance'],state['unit'],state['shot'])
            if signature==last:
                equal+=1
            else:equal=0
            if equal>=1:
                return ('next',state)
        last=(state['distance'],state['unit'],state['shot'])
    cv2.imwrite(str(DEBUG/f'v47_after_shot_{number}.png'),capture(sct,rect))
    return ('unverified',None)


def run():
    setup()
    print('WGT V47 - ONE-HOLE CONTROLLER (EXPERIMENTAL)')
    print('No aim drag. No camera change. No resize. F9 = STOP.')
    print('It only continues after verifying the shot changed.')
    input('Start at a tee (or resume a current hole). Press Enter for countdown...')
    for n in range(8,0,-1):
        stop_if_requested()
        print(f'{n}...',flush=True)
        time.sleep(1)
    hwnd=find_wgt()
    stop_if_requested(hwnd)
    rect=rect_of(hwnd)
    with mss.MSS() as sct:
        monitors_for_rect.monitors=sct.monitors[1:]
        first=None
        prev=None
        for number in range(1,MAX_SHOTS+1):
            stop_if_requested(hwnd)
            current=rect_of(hwnd)
            if current!=rect:
                raise RuntimeError('WGT window moved or resized; stopped without a click')
            screenshot=capture(sct,rect)
            cv2.imwrite(str(DEBUG/f'v47_before_shot_{number}.png'),screenshot)
            state=shot_state(screenshot)
            print(f'SHOT {number}: distance={state["distance"]} {state["unit"]}; '
                  f'putt={state["putt"]}; shot={state["shot"]}; '
                  f'club={state["club_yards"]}yd',flush=True)
            if state['complete']:
                print('HOLE FINISHED (on-screen confirmation).',flush=True)
                return
            if first is None:first=state
            pos,score=swing_button(screenshot,rect)
            if pos is None:
                raise RuntimeError(f'Swing button not verified (score {score:.2f}); stopped')
            meter=meter_rect(rect)
            blue=capture(sct,meter)
            target=blue_target(blue)
            if target is None:
                cv2.imwrite(str(DEBUG/'v47_unreadable_meter.png'),blue)
                raise RuntimeError('Could not locate light-blue accuracy zone; stopped')
            pull,reason=select_power(state,rect)
            print(f'PLAN: {reason}; reference pull {pull*100:.0f}% of window height; '
                  f'live accuracy target x={target}',flush=True)
            drag_swing(hwnd,rect,pos,pull)
            print('MOUSE RELEASED. Tracking green return...',flush=True)
            hit=accuracy_click(hwnd,sct,rect,target)
            if not hit:
                print('Stopped: accuracy meter did not confirm a click. F9 always available.')
                return
            outcome,new_state=wait_until_next(hwnd,sct,rect,state,number)
            if outcome=='complete':
                print('HOLE FINISHED / new hole detected.',flush=True)
                return
            if outcome!='next':
                print('Stopped: cannot prove WGT advanced to the next shot.',flush=True)
                return
            print('Shot change verified. Preparing the next shot...',flush=True)
            sleep_safe(1.3,hwnd)
        print(f'Stopped after safety cap of {MAX_SHOTS} attempted shots.',flush=True)


if __name__=='__main__':
    try:
        run()
    except KeyboardInterrupt:
        print('Emergency stop.',flush=True)
    except Exception as exc:
        print(f'STOPPED SAFELY: {exc}',flush=True)
    finally:
        mouse_up()
        print('Mouse released. Controller closed.',flush=True)
