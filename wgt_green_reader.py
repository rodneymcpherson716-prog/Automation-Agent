"""Read-only WGT putting-grid dot motion observer (shadow/diagnostics only).

This utility does not swing, aim, click, activate the game window, or modify the
existing controller. It measures apparent motion of white green-grid dots,
which requires a steady on-screen green for multiple frames. Pixel motion is
NOT a calibrated putting break or aiming recommendation.

Usage on Windows, with WGT on the putting green (green grid visible):
  python wgt_green_reader.py --scan
  python wgt_green_reader.py --self-test

Uses only the existing WGT Python dependencies: opencv-python, numpy, mss.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SCHEMA = 1
BANDS = (('far', .46, .61), ('middle', .61, .74), ('near', .74, .88))


def detect_white_dots(bgr):
    """Candidate white-dot centers from a WGT-sized frame, not claimed lie vectors."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return np.empty((0, 2), dtype=np.float32)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 160), (179, 115, 255))
    # Avoid the player/score/Shot Stats and bottom swing meter. The main green
    # area is used; no assumptions are made about the meaning of its grid color.
    mask[:round(.46*h)] = 0
    mask[round(.88*h):] = 0
    mask[:, :round(.17*w)] = 0
    mask[:, round(.86*w):] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    result = []
    scale = (w/1288 + h/728)/2
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if (3*scale*scale <= area <= 110*scale*scale
            and 2*scale <= bw <= 16*scale and 2*scale <= bh <= 16*scale
            and .4 <= bw/bh <= 2.5):
            result.append(centers[i])
    return np.asarray(result, dtype=np.float32).reshape(-1, 2)


def _background_shift(a, b):
    """Check camera motion using an upper-scene patch, away from moving dots."""
    h, w = a.shape[:2]
    box = (round(.25*w), round(.19*h), round(.75*w), round(.38*h))
    x1, y1, x2, y2 = box
    gray_a = cv2.cvtColor(a[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if min(gray_a.shape) < 25:
        return None
    ga = cv2.resize(gray_a, (max(100, gray_a.shape[1]//2), max(50, gray_a.shape[0]//2))).astype(np.float32)
    gb = cv2.resize(gray_b, (ga.shape[1], ga.shape[0])).astype(np.float32)
    # A low-texture upper view cannot establish camera stability.
    if ga.std() < 5 or gb.std() < 5:
        return None
    try:
        (dx,dy), response = cv2.phaseCorrelate(ga, gb)
    except cv2.error:
        return None
    # Resizing may not be exactly 1/2 on tiny images.
    return dict(dx=float(dx*gray_a.shape[1]/ga.shape[1]),
                dy=float(dy*gray_a.shape[0]/ga.shape[0]), response=float(response))


def _matched_flow(a, b, dt, h):
    """Mutual nearest white-dot matches; does not interpret actual break direction."""
    if not .025 <= dt <= .3 or len(a) < 8 or len(b) < 8:
        return []
    delta = b[None,:,:]-a[:,None,:]
    squared = np.sum(delta*delta, axis=2)
    nearest_b = squared.argmin(axis=1)
    nearest_a = squared.argmin(axis=0)
    limit = (3 + dt*55)**2
    matches=[]
    for ia, ib in enumerate(nearest_b):
        if nearest_a[ib] == ia and squared[ia,ib] <= limit:
            x0,y0 = a[ia]
            dx,dy = delta[ia,ib]
            if np.isfinite(dx) and np.isfinite(dy):
                matches.append((float(x0),float(y0),float(dx/dt),float(dy/dt)))
    return matches


def analyze_frames(frames, timestamps):
    """Return conservative dot-flow metrics, or abstain from direction claims."""
    if len(frames) != len(timestamps) or len(frames) < 3:
        return dict(status='NO_READING', reason='need at least three timestamped frames')
    if any(f is None or f.shape != frames[0].shape for f in frames):
        return dict(status='NO_READING', reason='frame dimensions differ')
    if any(t2 <= t1 for t1,t2 in zip(timestamps,timestamps[1:])):
        return dict(status='NO_READING', reason='invalid capture timestamps')
    h,w = frames[0].shape[:2]
    if w < 700 or h < 450:
        return dict(status='NO_READING', reason='game window too small')
    points = [detect_white_dots(f) for f in frames]
    if float(np.median([len(p) for p in points])) < 24:
        return dict(status='NO_READING', reason='green grid dots not reliably detected',
                    median_dots=int(np.median([len(p) for p in points])))
    flows=[]; rejected_camera=0; rejected_background=0
    for i in range(len(frames)-1):
        shift = _background_shift(frames[i],frames[i+1])
        if shift is None or shift['response'] < .12:
            rejected_background += 1
            continue
        if math.hypot(shift['dx'],shift['dy']) > 1.8:
            rejected_camera += 1
            continue
        dt = timestamps[i+1]-timestamps[i]
        flows.extend(_matched_flow(points[i],points[i+1],dt,h))
    report = dict(schema=SCHEMA,status='NO_READING', median_dots=int(np.median([len(p) for p in points])),
                  captured_frames=len(frames), match_count=len(flows),
                  rejected_camera_pairs=rejected_camera, rejected_uncertain_pairs=rejected_background,
                  kind='apparent_white_dot_motion',units='screen_pixels_per_second',
                  live_aim_or_power_allowed=False, aim_recommendation=None, calibrated_break=False,
                  bands={})
    if not flows:
        report['reason']='no camera-stable matching dot motion'
        return report
    for name,lo,hi in BANDS:
        region=np.array([(dx,dy) for x,y,dx,dy in flows if lo*h <= y < hi*h],dtype=float)
        # Need both enough matches and observations across several frames;
        # no aim/slope claim is permitted on low-quality evidence.
        if len(region) < 12:
            report['bands'][name]=dict(status='INSUFFICIENT',matched_dots=len(region))
            continue
        mx,my=np.median(region,axis=0)
        spread=float(np.median(np.abs(region[:,0]-mx)))
        # Movement is relative to the screen. Sign is NOT an aiming instruction.
        label='RIGHT' if mx>1.0 else 'LEFT' if mx < -1.0 else 'NEARLY_STILL'
        report['bands'][name]=dict(status='MEASURED_DOT_FLOW',matched_dots=len(region),
                                   horizontal_flow=label,median_vx_px_s=round(float(mx),2),
                                   median_vy_px_s=round(float(my),2),
                                   horizontal_variability_px_s=round(spread,2))
    if sum(b['status']=='MEASURED_DOT_FLOW' for b in report['bands'].values()) >= 2:
        report['status']='DOT_FLOW_OBSERVED'
    else:
        report['reason']='insufficient verified coverage across the green'
    return report


def _find_wgt_window():
    if sys.platform != 'win32':
        raise RuntimeError('Live capture requires Windows')
    import ctypes
    import ctypes.wintypes as wt
    user32=ctypes.WinDLL('user32',use_last_error=True)
    # Match the existing controller's per-monitor DPI capture geometry.
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError,OSError):
        try:user32.SetProcessDPIAware()
        except (AttributeError,OSError):pass
    CALLBACK=ctypes.WINFUNCTYPE(wt.BOOL,wt.HWND,wt.LPARAM)
    user32.GetWindowTextLengthW.argtypes=[wt.HWND]
    user32.GetWindowTextLengthW.restype=ctypes.c_int
    user32.GetWindowTextW.argtypes=[wt.HWND,wt.LPWSTR,ctypes.c_int]
    user32.IsWindowVisible.argtypes=[wt.HWND]
    user32.IsIconic.argtypes=[wt.HWND]
    user32.GetWindowRect.argtypes=[wt.HWND,ctypes.POINTER(wt.RECT)]
    user32.GetForegroundWindow.restype=wt.HWND
    found=[]
    def visit(hwnd,unused):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            n=user32.GetWindowTextLengthW(hwnd)
            if n:
                buff=ctypes.create_unicode_buffer(n+1)
                user32.GetWindowTextW(hwnd,buff,n+1)
                if buff.value.strip().lower()=='wgt golf': found.append(hwnd)
        return True
    user32.EnumWindows(CALLBACK(visit),0)
    if len(found)!=1:raise RuntimeError(f'Expected one visible WGT Golf window; found {len(found)}')
    rect=wt.RECT()
    if not user32.GetWindowRect(found[0],ctypes.byref(rect)):
        raise RuntimeError('Cannot read WGT window rectangle')
    box=dict(left=rect.left,top=rect.top,width=rect.right-rect.left,height=rect.bottom-rect.top)
    if box['width']<700 or box['height']<450:
        raise RuntimeError('WGT window is too small')
    return user32,found[0],box


def scan_live(countdown=7, duration=2.2):
    """Only capture existing WGT window pixels, never activate or send input."""
    import mss
    user32, hwnd, rect = _find_wgt_window()
    print(f'Read-only scan starts in {countdown} seconds. Bring WGT to the front on its putting green.',flush=True)
    time.sleep(countdown)
    if user32.GetForegroundWindow()!=hwnd:
        raise RuntimeError('WGT is not foreground; no scan (no activation or click attempted)')
    frames=[];ts=[]
    with mss.MSS() as camera:
        started=time.perf_counter()
        while time.perf_counter()-started<duration:
            if user32.GetForegroundWindow()!=hwnd:
                raise RuntimeError('WGT lost focus during scan; result discarded')
            cur=_find_wgt_window()[2]
            if cur!=rect:
                raise RuntimeError('WGT moved/resized during scan; result discarded')
            image=np.asarray(camera.grab(rect))[:,:,:3].copy()
            frames.append(cv2.resize(image,(1288,728),interpolation=cv2.INTER_AREA))
            ts.append(time.perf_counter())
            time.sleep(.09)
    result=analyze_frames(frames,ts)
    result['at_utc']=datetime.now(timezone.utc).isoformat()
    result['scope']='read-only WGT window; no screenshot uploaded; no live input'
    folder=ROOT/'wgt_debug'/'green_scans'
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/('green_scan_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result,output


def _synthetic_frame(offset=0,window=(1288,728)):
    h,w=window[1],window[0]
    image=np.zeros((h,w,3),dtype=np.uint8)
    # An unmoving textured sky/trees reference gives camera-stability evidence.
    rng=np.random.default_rng(102)
    sky=rng.integers(20,200,(round(.19*h),round(.5*w)),dtype=np.uint8)
    y0,y1=round(.19*h),round(.38*h)
    sky=cv2.resize(sky,(round(.5*w),y1-y0))
    image[y0:y1,round(.25*w):round(.75*w)]=cv2.cvtColor(sky,cv2.COLOR_GRAY2BGR)
    image[round(.46*h):round(.88*h),:]=(50,90,40)
    for row in range(7):
        y=round((.50+row*.05)*h)
        for col in range(12):
            x=round((.22+col*.05)*w)+offset
            cv2.circle(image,(x,y),3,(235,235,235),-1)
    return image


def self_test():
    assert len(detect_white_dots(_synthetic_frame()))>=60
    frames=[_synthetic_frame(i) for i in range(6)]
    r=analyze_frames(frames,[i*.1 for i in range(6)])
    assert r['status']=='DOT_FLOW_OBSERVED',r
    assert all(b['horizontal_flow']=='RIGHT' for b in r['bands'].values() if b['status']=='MEASURED_DOT_FLOW'),r
    left=analyze_frames([_synthetic_frame(-i) for i in range(6)],[i*.1 for i in range(6)])
    assert left['status']=='DOT_FLOW_OBSERVED',left
    assert all(b['horizontal_flow']=='LEFT' for b in left['bands'].values() if b['status']=='MEASURED_DOT_FLOW'),left
    still=analyze_frames([_synthetic_frame(0) for i in range(6)],[i*.1 for i in range(6)])
    assert still['status']=='DOT_FLOW_OBSERVED',still
    assert all(b['horizontal_flow']=='NEARLY_STILL' for b in still['bands'].values() if b['status']=='MEASURED_DOT_FLOW'),still
    assert analyze_frames(frames[:2],[0,.1])['status']=='NO_READING'
    assert analyze_frames([np.zeros((728,1288,3),dtype=np.uint8)]*5,[i*.1 for i in range(5)])['status']=='NO_READING'
    assert analyze_frames(frames,[0,1,1,2,3,4])['status']=='NO_READING'
    return '6 synthetic offline checks passed; no WGT input'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',action='store_true',help='read WGT putting grid without clicks')
    parser.add_argument('--self-test',action='store_true',help='offline synthetic checks only')
    parser.add_argument('--image',type=Path,help='read dot count in a saved image (not slope)')
    parser.add_argument('--countdown',type=int,default=7)
    args=parser.parse_args(argv)
    if args.self_test or not (args.scan or args.image):
        print(self_test());return 0
    if args.image:
        frame=cv2.imread(str(args.image))
        if frame is None: raise FileNotFoundError(args.image)
        print('White dot candidates:',len(detect_white_dots(cv2.resize(frame,(1288,728)))))
        print('Single image cannot establish dot movement or putting break.')
        return 0
    try:
        result,path=scan_live(args.countdown)
    except Exception as exc:
        print('NO READING:',exc)
        print('No mouse/keyboard input was sent. Your existing controller is unchanged.')
        return 2
    print('GREEN SCAN:',result['status'])
    for name,values in result.get('bands',{}).items():
        if values['status']=='MEASURED_DOT_FLOW':
            print(f'  {name}: {values["horizontal_flow"]} dot motion, '
                  f'{values["median_vx_px_s"]} px/s horizontal, '
                  f'{values["matched_dots"]} matches')
        else: print(f'  {name}: not enough reliable dot motion')
    if result.get('reason'):print('WHY:',result['reason'])
    print('REPORT:',path)
    print('READ-ONLY. No aim, power, DING, swing, or F9 control changed.')
    return 0 if result['status']=='DOT_FLOW_OBSERVED' else 1


if __name__=='__main__':
    raise SystemExit(main())
