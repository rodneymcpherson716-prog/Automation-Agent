"""Exercise the calibration workflow without opening a live WGT window."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vision.calibration.calibration_tool import calibrate
from vision.calibration.profile_manager import ProfileManager, Region


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory())
        self.manager = ProfileManager(Path(folder) / 'profiles.json')
        self.frame = SimpleNamespace(shape=(720, 1280, 3))

    def test_skip_retry_undo_preview_and_save_use_only_accepted_regions(self):
        first, final = Region(100, 100, 80, 40), Region(120, 130, 100, 80)
        wind = Region(78, 40, 64, 24)
        choices = ['r', 'a', 's', 'u', 'a', 's', 's', 'p', 's', 'y']
        selections = [first, final, None, None, wind, None, None]
        with redirect_stdout(StringIO()), patch('builtins.input', side_effect=choices), \
                patch('vision.calibration.calibration_tool.select_region', side_effect=selections), \
                patch('vision.calibration.calibration_tool.show_preview') as preview:
            self.assertTrue(calibrate(self.frame, self.manager))
        accepted = {'distance': final, 'wind': wind}
        self.assertEqual(self.manager.load()[(1280, 720)], accepted)
        self.assertEqual(preview.call_count, 2)
        preview.assert_called_with(self.frame, accepted)

    def test_backup_restore_and_scaling_preserve_layout(self):
        original, replacement = Region(100, 100, 80, 40), Region(120, 100, 80, 40)
        self.manager.save(1280, 720, {'distance': original})
        self.manager.save(1280, 720, {'distance': replacement})
        self.manager.restore()
        profiles = self.manager.load()
        self.assertEqual(profiles[(1280, 720)]['distance'], original)
        self.assertEqual(self.manager.resolve(profiles, 2560, 1440), {'distance': Region(200, 200, 160, 80)})
        self.assertEqual(self.manager.resolve(profiles, 1024, 768), {})

    def test_restart_then_cancel_leaves_saved_profile_in_place(self):
        original = Region(100, 100, 80, 40)
        self.manager.save(1280, 720, {'distance': original})
        with redirect_stdout(StringIO()), patch('builtins.input', side_effect=['a', 'x', 'q']), \
                patch('vision.calibration.calibration_tool.select_region', return_value=Region(140, 120, 80, 40)):
            self.assertFalse(calibrate(self.frame, self.manager))
        self.assertEqual(self.manager.load()[(1280, 720)], {'distance': original})
