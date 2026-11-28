"""WGT single-stroke calibration and replay for Windows.

Records your own successful manual swing/putt as a sequence of mouse input
positions, press, release and accuracy click; replays once on demand.
No auto aiming, no window resize, no guessed percentages.

Run with your existing WGT_Bot virtual environment:
    python automation_calibrate_replay_v45.py

Use only in a game mode where automation is permitted.
"""
import ctypes
import ctypes.wintypes as wt
import json
import time
from pathlib import Path

user32 = ctypes.windll.user32
VK_LEFT = 0x01
VK_F9 = 0x78
MOUSE_LEFT_DOWN = 0x0002
MOUSE_LEFT_UP = 0x0004
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'wgt_calibration'
OUT.mkdir(exist_ok=True)
RECORD = OUT / 'last_manual_stroke.json'

class POINT(ctypes.Structure):
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

class RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

CALLBACK = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

def dpi_setup():
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try: user32.SetProcessDPIAware()
        except Exception: pass


def held(key):
    return bool(user32.GetAsyncKeyState(key) & 0x8000)


def stop_check():
    if held(VK_F9):
        raise KeyboardInterrupt('F9 emergency stop')


def pointer():
    p = POINT()
    if not user32.GetCursorPos(ctypes.byref(p)):
        raise RuntimeError('Cannot read cursor')
    return p.x, p.y


def window_rect(hwnd):
    r = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError('WGT rectangle unavailable')
    return r.left, r.top, r.right-r.left, r.bottom-r.top


def find_wgt():
    found = []
    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length+1)
                user32.GetWindowTextW(hwnd, buf, length+1)
                if 'wgt golf' in buf.value.lower():
                    found.append(hwnd)
        return True
    cb = CALLBACK(visit)
    user32.EnumWindows(cb, 0)
    if not found:
        raise RuntimeError('WGT Golf window not found. Open the game first.')
    return found[0]


def wait_for_wgt(seconds=8):
    print(f'Activate WGT on Monitor 1. Starting in {seconds} seconds...', flush=True)
    for n in range(seconds, 0, -1):
        stop_check()
        print(f'{n}...',flush=True)
        time.sleep(1)
    hwnd = find_wgt()
    if user32.GetForegroundWindow() != hwnd:
        raise RuntimeError('WGT is not the active window. No mouse actions.')
    return hwnd


def mouse_event(flag):
    user32.mouse_event(flag, 0, 0, 0, 0)


def record_once():
    hwnd = wait_for_wgt()
    x0,y0,w,h = window_rect(hwnd)
    if min(w,h) < 350:
        raise RuntimeError('WGT game window too small')
    print('RECORDING: make ONE complete manual putt/swing, including accuracy click.', flush=True)
    print('Recorder stops about 2 seconds after your final mouse click.',flush=True)
    trace=[]
    t0=time.perf_counter()
    last_down=held(VK_LEFT)
    had_down=False
    rises=0
    end_pending=None
    prev_pos=None
    max_length=18.0
    while time.perf_counter()-t0 < max_length:
        stop_check()
        if user32.GetForegroundWindow() != hwnd:
            raise RuntimeError('WGT lost focus while recording; cancelled')
        now=time.perf_counter()-t0
        down=held(VK_LEFT)
        xx,yy=pointer()
        # Store window-relative coordinates for a stable repeat even if window moved.
        pos=(round((xx-x0)/w,6),round((yy-y0)/h,6))
        if pos!=prev_pos or down!=last_down:
            trace.append({'t':round(now,4),'x':pos[0],'y':pos[1],'down':down})
            prev_pos=pos
        if down and not last_down:
            had_down=True
            end_pending=None
        if last_down and not down:
            rises+=1
            end_pending=now+2.0
        last_down=down
        if had_down and rises>=2 and end_pending is not None and now>=end_pending:
            break
        time.sleep(.008)
    if rises<2 or not had_down:
        raise RuntimeError('No full manual stroke + accuracy click was detected. Nothing saved.')
    # Trim startup pause and normalize time; preserve first movement before mouse down.
    first_down=next(i for i,e in enumerate(trace) if e['down'])
    start=max(0,first_down-15)
    trace=trace[start:]
    base=trace[0]['t']
    for e in trace: e['t']=round(e['t']-base,4)
    data={'width':w,'height':h,'samples':trace,'duration':trace[-1]['t'],
          'description':'one manually completed WGT swing and accuracy click',
          'mouse_releases':rises}
    RECORD.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(f'RECORDED: {len(trace)} events, {data["duration"]:.2f}s; saved {RECORD}',flush=True)
    return data


def replay(data):
    print('Place WGT on the SAME shot screen (or a practice shot).',flush=True)
    hwnd=wait_for_wgt()
    left,top,w,h=window_rect(hwnd)
    # No resizing or camera change; avoid replay on altered window geometry.
    if abs(w-data['width'])>10 or abs(h-data['height'])>10:
        raise RuntimeError('WGT window size changed; no replay. Use the recorded size.')
    seq=data['samples']
    if not seq:
        raise RuntimeError('Recording is empty')
    if any(not (-.02<=e['x']<=1.02 and -.02<=e['y']<=1.02) for e in seq):
        raise RuntimeError('Recorded mouse left game window; refusing replay')
    print('REPLAYING ONE stroke. F9 stops immediately.',flush=True)
    is_down=False
    started=time.perf_counter()
    try:
        for e in seq:
            stop_check()
            if user32.GetForegroundWindow()!=hwnd:
                raise RuntimeError('WGT lost focus; stopping')
            deadline=started+float(e['t'])
            while time.perf_counter()<deadline:
                stop_check()
                time.sleep(.001)
            px=left+round(e['x']*w)
            py=top+round(e['y']*h)
            user32.SetCursorPos(px,py)
            if e['down'] and not is_down:
                mouse_event(MOUSE_LEFT_DOWN);is_down=True
            elif not e['down'] and is_down:
                mouse_event(MOUSE_LEFT_UP);is_down=False
        print('Replay finished. Check whether the ball moved.',flush=True)
    finally:
        mouse_event(MOUSE_LEFT_UP)
        print('Mouse button released.',flush=True)


def main():
    dpi_setup()
    print('WGT CALIBRATE + REPLAY V45 | no aim/camera/window changes')
    print('F9 = emergency stop | Ctrl+C = stop from terminal')
    print('1 = Record your normal manual putt, 2 = Replay the recorded stroke')
    print('Recording includes your accuracy click; neither power nor aim is guessed.')
    choice=input('Choose 1 or 2, then Enter: ').strip()
    if choice=='1': record_once()
    elif choice=='2':
        if not RECORD.exists():
            print('No recording yet. Run option 1 first.');return
        replay(json.loads(RECORD.read_text(encoding='utf-8')))
    else:
        print('Choose 1 or 2.')

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('Stopped by user.')
    except Exception as ex: print('Stopped:',ex)
    finally: mouse_event(MOUSE_LEFT_UP)
