"""Separate, read-only BEFORE/AFTER recorder for one manual SOLO bunker shot.

Never launches the target application, moves the pointer, clicks, swings or changes aim.
The player opens Shot Stats, chooses the setup and takes the shot manually.
"""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import time
import uuid

import cv2

from terrain_recording import ManualBunkerStore, atomic_json, read_manual_stats_patch
from context_manager import ContextObserver, ControllerLock

ROOT = Path(__file__).resolve().parent


class ReadOnlyNative:
    """Expose only the Windows queries used by existing screen readers."""
    ALLOWED = frozenset(('GetForegroundWindow', 'IsIconic', 'IsWindowVisible',
                         'GetWindowRect', 'GetWindowTextLengthW', 'GetWindowTextW',
                         'EnumWindows', 'WindowFromPoint', 'GetAncestor', 'GetAsyncKeyState'))

    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        if name not in self.ALLOWED:
            raise RuntimeError('Manual recorder forbids native action: ' + name)
        return getattr(self.native, name)


@contextmanager
def read_only_native(live):
    native = live.user32
    live.user32 = ReadOnlyNative(native)
    try:
        yield
    finally:
        live.user32 = native


def check_stop(live):
    if live.key_down(live.F9):
        raise KeyboardInterrupt('F9: recording stopped; no input was sent')


def capture_pair(root, phase, countdown=8):
    """The only live acquisition path. Called only by the opt-in recorder UI.

    Acquire both images before slow OCR, so each phase measures one short
    interval. Keep every raw reading with the PNG's timestamp and SHA-256.
    """
    import execution_engine as live
    if phase not in ('before', 'after'):
        raise ValueError('Unknown recording phase')
    live.setup()  # Existing DPI/signature/Tesseract setup; does not send input.
    folder = Path(root)/'automation_debug/manual_bunker'/uuid.uuid4().hex
    folder.mkdir(parents=True)
    with read_only_native(live):
        print(f'{phase.upper()}: switch to your already-open WGT window; keep the HUD and Shot Stats visible.', flush=True)
        for seconds in range(countdown, 0, -1):
            print(f'Capture in {seconds}...', flush=True)
            for _ in range(10):
                check_stop(live)
                time.sleep(.1)
        hwnd = live.find_wgt()  # Finds an existing window; never starts a game.
        rect = live.rect_of(hwnd)
        if abs(rect['width']-1288) > 10 or abs(rect['height']-728) > 10:
            raise ValueError('Use the established 1288 x 728 WGT layout; recorder will not resize it')
        frames = []
        with live.mss.MSS() as sct:
            for index in (1, 2):
                check_stop(live)
                if (live.user32.GetForegroundWindow() != hwnd or live.user32.IsIconic(hwnd)
                        or live.rect_of(hwnd) != rect):
                    raise ValueError('WGT must remain foreground, visible and at unchanged geometry')
                for fx, fy in ((.86, .08), (.14, .96), (.88, .925), (.85, .23), (.5, .5)):
                    if not live.point_in_wgt(hwnd, rect['left']+round(fx*rect['width']),
                                             rect['top']+round(fy*rect['height'])):
                        raise ValueError('Another window covers a measurement region; no capture accepted')
                stamp = time.time()
                captured = time.perf_counter()
                frame = live.capture(sct, rect)
                path = folder/f'{phase}_{index}.png'
                if not cv2.imwrite(str(path), frame):
                    raise OSError('Could not save manual-shot capture evidence')
                frames.append((frame, stamp, captured, path))
                if index == 1:
                    time.sleep(.5)
        print('Two local captures saved. Reading them; do not take another shot yet.', flush=True)
        def ready(frame):
            local = dict(left=0, top=0, width=frame.shape[1], height=frame.shape[0])
            return live.ready_meter(frame, local) is not None and live.swing_button(frame, local) is not None
        observer = ContextObserver(root, solo_confirmed=True, ready_reader=ready)
        pair = []
        for frame, stamp, captured, path in frames:
            check_stop(live)
            state = live.shot_state(frame)
            check_stop(live)
            context = observer.observe(frame, state, captured_at=captured)
            check_stop(live)
            stats = read_manual_stats_patch(live.crop(frame, (955/1257, 115/718, 1160/1257, 236/718)),
                                           live.ocr_variants)
            pair.append(dict(capture_id=uuid.uuid4().hex, captured_at=stamp,
                             evidence=dict(path=str(path.relative_to(Path(root))),
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                             window=dict(rect=rect, foreground_verified=True, regions_uncovered=True),
                             state=state, context=context, stats=stats, ready=ready(frame)))
        atomic_json(folder/'capture.json', dict(schema=1, phase=phase, captures=pair,
                                               use_for_automatic_power=False))
        return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check imports only; no window queries, capture or input')
    args = parser.parse_args(argv)
    if args.check:
        import mss
        import pytesseract
        print('Manual bunker recorder imports OK. No capture or input. Automatic bunker swings remain blocked.')
        return 0
    store = ManualBunkerStore(ROOT)
    lock = ControllerLock(ROOT/'automation_debug/live_controller.lock')
    try:
        lock.acquire()  # Also prevents a nine-hole controller starting alongside this recorder.
        print('READ-ONLY MANUAL BUNKER RECORDER. Use SOLO play only. No automation will run.')
        print('Open Shot Stats yourself and keep it visible BEFORE and AFTER. Do not use this while another bot runs.')
        while True:
            print('\n1 = capture BEFORE   2 = capture AFTER   3 = discard pending BEFORE   Q = quit')
            choice = input('Choose: ').strip().upper()
            if choice == 'Q':
                return 0
            try:
                if choice == '1':
                    if store.pending():
                        raise ValueError('Finish or explicitly discard the pending BEFORE first')
                    if input('Type SOLO to confirm single-player play and that YOU will take the manual shot: ').strip().upper() != 'SOLO':
                        continue
                    pending = store.stage(capture_pair(ROOT, 'before'), solo_confirmed=True)
                    missing = [k for k, v in pending['before_fields'].items() if v is None]
                    print('BEFORE SAVED. Return to this console to read this message before swinging.')
                    print('Missing conditions:', ', '.join(missing) or 'none')
                    print('Baseline Shot Stats:', pending['baseline_stats'])
                    if any(v is None for v in pending['baseline_stats'].values()):
                        print('BASELINE INCOMPLETE: fresh result attribution cannot be verified. Open Shot Stats, discard and recapture BEFORE.')
                    print('If anything is missing, discard and recapture for a fully verified measurement.')
                    print('Now take exactly ONE manual shot. Do not change aim, club or mode. Then choose 2.')
                elif choice == '2':
                    if not store.pending():
                        raise ValueError('Capture BEFORE first; a missing setup will not be reconstructed')
                    answer = input('Type ONE to confirm exactly one manual shot, unchanged aim/club/mode, and no noticed condition changes since BEFORE: ')
                    if answer.strip().upper() != 'ONE':
                        continue
                    record = store.finish(capture_pair(ROOT, 'after'), one_manual_shot_confirmed=True,
                                          setup_unchanged_confirmed=True)
                    print('RESULT:', record['status'].upper())
                    print('Fields:', record['fields'])
                    print('Missing:', ', '.join(record['missing_fields']) or 'none')
                    print('Attribution issues:', '; '.join(record['rejection_reasons']) or 'none')
                    print('Saved:', store.folder/'records'/(record['id'] + '.json'))
                    print('Excluded from automatic power. Automatic bunker swings remain blocked.')
                    # Research data can update shadow learning immediately;
                    # it still cannot enable or control a bunker swing.
                    from automation_learning import ShadowLearning
                    learner = None
                    try:
                        learner = ShadowLearning(ROOT)
                        learner.sync_existing()
                        report = learner.write_report()
                        print('Shadow memory updated. Eligible research measurements:', report['eligible'])
                    except Exception as exc:
                        print('Measurement saved; shadow update unavailable:', exc)
                    finally:
                        if learner is not None:
                            learner.close()
                elif choice == '3':
                    if input('Type DISCARD to retire the pending setup (local evidence will be kept): ').strip().upper() == 'DISCARD':
                        store.discard()
                        print('Pending setup discarded; evidence retained.')
            except (ValueError, OSError, RuntimeError) as exc:
                print('NOT ACCEPTED:', exc)
                print('No input was sent. Do not repeat a shot just to repair its recording.')
    except (KeyboardInterrupt, EOFError):
        print('\nRecorder stopped. Pending data retained; no input was sent.')
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print('Recorder could not start:', exc)
        return 1
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
