"""Run regressions with native input and real MSS capture blocked at the boundary.

Usage: python -B run_offline_tests.py [test_bunker_power ...]
Without names, discover test_*.py in this workspace. Never launch the game.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import execution_engine as live


def main(names=()):
    attempts = []

    def forbidden(action):
        def deny(*args, **kwargs):
            attempts.append(action)
            raise AssertionError('Offline tests forbid native input/capture: ' + action)
        return deny

    class OfflineCapture:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        grab = staticmethod(forbidden('MSS.grab'))

    with ExitStack() as guards:
        guards.enter_context(patch.object(live.mss, 'mss', return_value=OfflineCapture()))
        guards.enter_context(patch.object(live.mss, 'MSS', return_value=OfflineCapture()))
        if live.user32 is not None:
            for name in ('SendInput', 'mouse_event', 'keybd_event', 'SetCursorPos', 'SetForegroundWindow'):
                guards.enter_context(patch.object(live.user32, name, side_effect=forbidden(name)))
        loader = unittest.TestLoader()
        suite = (loader.loadTestsFromNames(names) if names else
                 loader.discover(str(Path(__file__).resolve().parent), pattern='test_*.py'))
        result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    print('Native input / real capture attempts:', len(attempts), flush=True)
    if attempts:
        print('Blocked attempts:', ', '.join(attempts), flush=True)
    return 0 if result.wasSuccessful() and not attempts else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
