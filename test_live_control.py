import json
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import execution_engine as live
from automation_core import Distance,resolve_target_distance
from vision_engine import visible_hole_lock

PROJECT=Path(r'Automation-Agent project root')
DRIVER=Path(__file__).with_name('preflight.png')
if not DRIVER.is_file():
    DRIVER=PROJECT/'automation_debug/driver_preflight_fixture_20261110.png'


class LiveReadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()

    def test_driver_tee_target_is_separate_from_range_and_hole_distance(self):
        frame=cv2.imread(str(DRIVER))
        self.assertIsNotNone(frame)
        h,w=frame.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            reading=live.inspect_screen(frame,dict(left=37,top=46,width=w,height=h))
        s=reading['state']
        self.assertEqual(s['distance'],282,s)
        self.assertEqual(s['club_range'],282,s)
        self.assertEqual(s['club_name'],'9.0D',s)
        self.assertEqual(s['hole'],11,s)
        self.assertEqual(s['shot'],1,s)
        self.assertEqual(s['hole_distance'],425,s)
        self.assertTrue(reading['verified'],reading)
        send.assert_not_called()

    def test_club_range_and_hole_hud_do_not_replace_hidden_aim_on_tee(self):
        frame=cv2.imread(str(DRIVER))
        with patch.object(live,'read_sign_distance',return_value=None):
            state=live.shot_state(frame)
        self.assertEqual(state['club_range'],282)
        self.assertEqual(state['hole_distance'],425)
        self.assertIsNone(state['distance'])

    def test_live_second_shot_sign_border_does_not_merge_into_cloud(self):
        path=Path(__file__).with_name('approach_live.png')
        if not path.is_file():
            path=PROJECT/'automation_debug/approach_live_fixture_20261110.png'
        frame=cv2.imread(str(path))
        self.assertIsNotNone(frame)
        h,w=frame.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            reading=live.inspect_screen(frame,dict(left=37,top=46,width=w,height=h))
        state=reading['state']
        self.assertEqual(state['distance'],174,reading)
        self.assertEqual(state['club_range'],187,reading)
        self.assertEqual(state['shot'],2,reading)
        self.assertEqual(state['hole'],11,reading)
        self.assertEqual(state['distance_source'],'floating_target')
        self.assertTrue(reading['verified'],reading)
        send.assert_not_called()

    def test_verified_locked_hole_hud_is_fallback_only_with_adequate_precision(self):
        value=resolve_target_distance(None,Distance(12,'ft'),True)
        self.assertEqual(value['distance'],12)
        self.assertEqual(value['distance_source'],'verified_hole_hud')
        self.assertIsNone(resolve_target_distance(None,Distance(12,'ft'),False)['distance'])
        self.assertIsNone(resolve_target_distance(None,Distance(2,'ft'),True)['distance'])
        exact=resolve_target_distance(Distance(2.4,'ft'),Distance(2,'ft'),True)
        self.assertEqual(exact['distance'],2.4)

    def test_actual_putting_lock_is_recognized_but_tee_has_no_lock(self):
        frame=cv2.imread(str(PROJECT/'automation_debug/short_game_fixture_20261110.png'))
        self.assertIsNotNone(visible_hole_lock(frame,live.find_distance_sign(frame)))
        tee=cv2.imread(str(DRIVER))
        self.assertIsNone(visible_hole_lock(tee,live.find_distance_sign(tee)))

    def test_activation_raises_only_existing_window_without_game_input(self):
        with patch.object(live,'user32') as api,patch.object(live,'key_down',return_value=False), \
                patch.object(live,'send') as send:
            api.IsIconic.return_value=False
            api.GetForegroundWindow.side_effect=[20,10]
            live.activate_wgt(10)
            api.SetForegroundWindow.assert_called_once_with(10)
            send.assert_not_called()

    def test_f9_prevents_activation(self):
        with patch.object(live,'key_down',return_value=True),patch.object(live,'user32') as api:
            with self.assertRaises(KeyboardInterrupt):
                live.activate_wgt(10)
            api.SetForegroundWindow.assert_not_called()

    def test_release_uses_both_native_paths_even_when_sendinput_reports_success(self):
        with patch.object(live,'send') as send,patch.object(live,'user32') as api:
            live.mouse_up()
            send.assert_called_once_with(live.UP)
            api.mouse_event.assert_called_once_with(live.UP,0,0,0,0)

    def test_release_requires_observed_button_up(self):
        safety=live.Safety(10,{})
        safety.pressed=True
        with patch.object(live,'mouse_up') as up,patch.object(live,'key_down',return_value=False):
            safety.release()
            up.assert_called_once()
            self.assertFalse(safety.pressed)

    def test_stuck_button_stops_before_accuracy_phase(self):
        safety=live.Safety(10,{})
        safety.pressed=True
        with patch.object(live,'mouse_up') as up,patch.object(live,'key_down',return_value=True), \
                patch.object(live.time,'perf_counter',side_effect=[0,1]):
            with self.assertRaisesRegex(RuntimeError,'Left button remains held'):
                safety.release()
            up.assert_called_once()
            self.assertFalse(safety.pressed)

    def test_existing_held_button_blocks_another_mouse_down(self):
        safety=live.Safety(10,{})
        with patch.object(safety,'check'),patch.object(live,'key_down',return_value=True), \
                patch.object(live,'send') as send:
            with self.assertRaisesRegex(RuntimeError,'already held'):
                safety.press()
            send.assert_not_called()

    def test_clicklock_disabled_only_during_input_and_restored(self):
        with patch.object(live,'clicklock_state',side_effect=[True,False,True]), \
                patch.object(live,'set_clicklock') as setting:
            with live.clicklock_disabled():
                setting.assert_called_once_with(False)
            self.assertEqual([x.args[0] for x in setting.call_args_list],[False,True])

    def test_clicklock_restored_on_emergency_stop(self):
        with patch.object(live,'clicklock_state',side_effect=[True,False,True]), \
                patch.object(live,'set_clicklock') as setting:
            with self.assertRaises(KeyboardInterrupt):
                with live.clicklock_disabled():
                    raise KeyboardInterrupt('F9')
            self.assertEqual([x.args[0] for x in setting.call_args_list],[False,True])

    def test_clicklock_already_off_is_unchanged(self):
        with patch.object(live,'clicklock_state',return_value=False), \
                patch.object(live,'set_clicklock') as setting:
            with live.clicklock_disabled():
                pass
            setting.assert_not_called()

    def test_failed_clicklock_disable_blocks_input_and_restores_setting(self):
        with patch.object(live,'clicklock_state',side_effect=[True,True,True]), \
                patch.object(live,'set_clicklock') as setting:
            with self.assertRaisesRegex(RuntimeError,'remained enabled'):
                with live.clicklock_disabled():
                    self.fail('Must not enter live input with ClickLock still on')
            self.assertEqual([x.args[0] for x in setting.call_args_list],[False,True])


if __name__=='__main__':
    unittest.main(verbosity=2)
