"""Offline regressions. Does not capture the desktop or send mouse input."""
import ctypes
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import automation_core as core
import execution_engine as live

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT if (ROOT/'automation_debug/v48_before_shot_1.png').is_file() else Path(r'Automation-Agent project root')
live.SWING_TEMPLATE = FIXTURES/'swing_button.png'
live.CALIBRATION = FIXTURES/'automation_accuracy.json'


def band(power=None, green=None):
    image = np.zeros((40, 1000, 3), dtype=np.uint8)
    cv2.rectangle(image, (70, 5), (940, 32), (190, 190, 190), 1)
    cv2.rectangle(image, (550, 7), (900, 30), (190, 150, 100), -1)
    cv2.line(image,(500,2),(500,36),(125,125,125),1)
    if power is not None:
        cv2.rectangle(image, (round(500-430*power), 7), (499, 30), (20, 165, 210), -1)
    if green is not None:
        cv2.rectangle(image, (502, 7), (green, 30), (30, 180, 65), -1)
    return image


class CoreTests(unittest.TestCase):
    def test_units_and_decimals(self):
        self.assertEqual(core.parse_distance('282 YDS'), core.Distance(282, 'yd'))
        self.assertEqual(core.parse_distance('5.9 FT'), core.Distance(5.9, 'ft'))
        self.assertEqual(core.parse_distance('17.5FT'), core.Distance(17.5, 'ft'))
        self.assertIsNone(core.parse_distance('2820'))
        self.assertIsNone(core.parse_distance('NaN FT'))

    def test_conflicting_ocr_abstains(self):
        self.assertIsNone(core.consensus([12, 12, 120, 120]))
        self.assertIsNone(core.consensus([12, None]))
        self.assertEqual(core.consensus([12, 12, 120]), 12)

    def test_completion_not_scorecard_or_scoring_label(self):
        for s in ('FOR BOGEY', 'SCORECARD', 'PAR 4'):
            self.assertFalse(core.complete_text(s))
        self.assertTrue(core.complete_text('NEXT HOLE'))
        self.assertEqual(core.parse_shot('FOR BOGEY', 4), 5)
        self.assertEqual(core.parse_shot('1ST SHOT'), 1)

    def test_saved_recording_is_two_cycles_not_distance_calibration(self):
        data = json.loads((FIXTURES/'wgt_calibration/last_manual_stroke.json').read_text())
        ref = core.StrokeReference.from_data(data)
        self.assertAlmostEqual(ref.anchor[0], .708851)
        self.assertAlmostEqual(ref.pull, .34478)
        self.assertAlmostEqual(ref.accuracy_delay, 1.8984)
        self.assertAlmostEqual(ref.drag_seconds, 5.6668)
        bad = json.loads(json.dumps(data))
        bad['samples'][-1]['down'] = True
        with self.assertRaises(ValueError):
            core.StrokeReference.from_data(bad)
        bad = json.loads(json.dumps(data))
        bad['samples'][0]['x'] = float('nan')
        with self.assertRaises(ValueError):
            core.StrokeReference.from_data(bad)

    def test_meter_origin_and_power(self):
        meter = core.detect_meter(band())
        self.assertIsNotNone(meter)
        self.assertAlmostEqual(meter.origin, 70, delta=1)
        for power in (.04, .18, .50, .80, .99):
            self.assertAlmostEqual(core.yellow_power(band(power=power), meter), power, delta=.015)
        self.assertIsNone(core.yellow_power(band(), meter))
        self.assertIsNone(core.green_front(band(), meter))
        self.assertAlmostEqual(core.green_front(band(green=650), meter), 650, delta=1)

    def test_saved_mark_requires_visible_blue_zone(self):
        self.assertIsNotNone(core.detect_meter(band(), 725))
        self.assertIsNone(core.detect_meter(band(), 300))
        self.assertIsNone(core.detect_meter(np.zeros((40, 1000, 3), dtype=np.uint8)))

    def test_real_pitch_meter_uses_visible_corners_when_one_border_extends(self):
        run=FIXTURES/'automation_debug/runs/20261122_214328_9c21de5b'
        frame=cv2.imread(str(run/'hole_01_post_shot_read_1795405653711724900.png'))
        self.assertIsNotNone(frame)
        height,width=frame.shape[:2]
        rect=core.meter_rect(dict(left=0,top=0,width=width,height=height))
        x,y=rect['left'],rect['top']
        image=frame[y:y+rect['height'],x:x+rect['width']].copy()
        self.assertEqual(core.meter_outline(image),(73,5,26))
        meter=core.detect_meter(image,767)
        self.assertIsNotNone(meter)
        self.assertEqual((meter.origin,meter.zero,meter.target),(73,535,767))
        image[6:26,995:998]=0
        self.assertIsNone(core.detect_meter(image,767))

    def test_static_green_never_clicks(self):
        m = core.detect_meter(band())
        tracker = core.AccuracyTracker(m)
        self.assertFalse(any(tracker.update(i*.01, m.target-5) for i in range(400)))
        tracker = core.AccuracyTracker(m)
        self.assertFalse(any(tracker.update(i*.01, m.green_origin+30) for i in range(400)))

    def test_accuracy_motion_clicks_once_before_crossing(self):
        m = core.detect_meter(band())
        tracker = core.AccuracyTracker(m)
        clicks = []
        for i, edge in enumerate(range(m.green_origin+5, m.target+100, 9)):
            if tracker.update(i*.012, edge):
                clicks.append(edge)
        self.assertEqual(len(clicks), 1)
        self.assertLessEqual(clicks[0], m.target+5)
        self.assertGreater(clicks[0], m.target-50)

    def test_noise_backward_and_late_returns_abstain(self):
        m = core.detect_meter(band())
        for observations in ([m.target+20]*10, [100, 110, 95, m.target], [100, 105, m.target]):
            tracker = core.AccuracyTracker(m)
            self.assertFalse(any(tracker.update(i*.01, x) for i, x in enumerate(observations)))

    def test_power_uses_selected_range_including_putting(self):
        state = dict(distance=50, unit='yd', club_range=100, club_unit='yd', putt=False, shot=1)
        self.assertEqual(core.plan_power(state), .5)
        self.assertEqual(core.plan_power(dict(state, distance=282, club_range=187)), .99)
        putt = dict(distance=5.9, unit='ft', club_range=30, club_unit='ft', putt=True)
        self.assertAlmostEqual(core.plan_power(putt), 5.9/30)
        with self.assertRaises(ValueError):
            core.plan_power(dict(putt, club_unit='yd'))
        with self.assertRaises(ValueError):
            core.plan_power(dict(putt, club_range=300))

    def test_transition_requires_stable_ready_state_and_handles_putting(self):
        before = dict(distance=70, unit='yd', club_range=100, club_unit='yd', putt=False, shot=4, hole=1)
        after = dict(distance=5.9, unit='ft', club_range=30, club_unit='ft', putt=True, shot=5, hole=1)
        t = core.TransitionTracker(before)
        self.assertIsNone(t.update(before, True))
        self.assertIsNone(t.update(after, False))
        self.assertIsNone(t.update(after, True))
        self.assertEqual(t.update(after, True), 'next')
        self.assertEqual(t.update(dict(after, complete=True), False), 'complete')

    def test_new_hole_requires_two_stable_ready_observations(self):
        before=dict(hole=1,shot=5,distance=5,unit='ft',putt=True,club_range=30,club_unit='ft')
        after=dict(hole=2,shot=1,distance=250,unit='yd',putt=False,club_range=270,club_unit='yd')
        t=core.TransitionTracker(before)
        self.assertIsNone(t.update(after,False))
        self.assertIsNone(t.update(after,True))
        self.assertEqual(t.update(after,True),'complete')

    def test_saved_screenshots_have_real_meter(self):
        for name in ('v48_before_shot_1.png', 'v47_before_shot_1.png'):
            with self.subTest(name=name):
                frame = cv2.imread(str(FIXTURES/'automation_debug'/name))
                h, w = frame.shape[:2]
                r = core.meter_rect(dict(left=0, top=0, width=w, height=h))
                image = frame[r['top']:r['top']+r['height'], r['left']:r['left']+r['width']]
                target = round(.7150621118012422*w)-r['left']
                meter = core.detect_meter(image, target)
                self.assertIsNotNone(meter)
                self.assertIsNone(core.yellow_power(image, meter))
                self.assertIsNone(core.green_front(image, meter))

    def test_real_saved_yellow_fill_expands_left_of_neutral(self):
        readings = []
        for name in ('power_check_5.png','power_check_15.png','power_check_30.png'):
            frame = cv2.imread(str(FIXTURES/'automation_debug'/name))
            meter = core.detect_meter(frame)
            self.assertIsNotNone(meter)
            self.assertAlmostEqual(meter.zero,535,delta=3)
            value = core.yellow_power(frame,meter)
            self.assertIsNotNone(value)
            readings.append(value)
        self.assertTrue(.10 < readings[0] < .17, readings)
        self.assertTrue(.16 < readings[1] < .23, readings)
        self.assertTrue(.23 < readings[2] < .31, readings)
        self.assertEqual(readings, sorted(readings))

    def test_real_saved_green_return_starts_at_released_yellow_edge(self):
        image = cv2.imread(str(FIXTURES/'automation_debug/green_return_review.png'))
        # Existing producer writes 25 tiles of 850x60 (wgt_before_green_fix_v2).
        self.assertEqual(image.shape[:2],(1500,850))
        first = image[:60]
        meter = core.detect_meter(first)
        self.assertIsNotNone(meter)
        # Yellow annotation at x=359 is the released front, before neutral 430.
        meter = replace(meter, return_origin=359)
        edges = [core.green_front(image[y:y+60],meter) for y in range(0,1500,60)]
        seen = [e for e in edges if e is not None]
        self.assertGreater(len(seen),10)
        self.assertLess(seen[0],380)
        self.assertGreater(seen[-1],460)
        self.assertEqual(seen,sorted(seen))
        tracker = core.AccuracyTracker(meter)
        self.assertFalse(any(tracker.update(i*.03,edge) for i,edge in enumerate(edges)))


class ScreenshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()

    def test_tee_sign_is_282_not_hole_length_527(self):
        frame = cv2.imread(str(FIXTURES/'automation_debug/v48_before_shot_1.png'))
        state = live.shot_state(frame)
        self.assertEqual(state['distance'], 282)
        self.assertEqual(state['unit'], 'yd')
        self.assertEqual(state['mode'], 'FULL')
        self.assertEqual(state['club_range'], 187)
        self.assertEqual(state['club_unit'], 'yd')
        self.assertEqual(state['shot'], 1)
        self.assertEqual(state['par'],5,state['raw'])
        self.assertTrue(state['player'])
        r = dict(left=0,top=0,width=1288,height=728)
        self.assertIsNotNone(live.swing_button(frame, r))

    def test_putt_sign_59_decimal_and_range_300_feet(self):
        frame = cv2.imread(str(FIXTURES/'automation_debug/v47_before_shot_1.png'))
        state = live.shot_state(frame)
        self.assertEqual(state['distance'], 5.9)
        self.assertEqual(state['unit'], 'ft')
        self.assertEqual(state['mode'], 'PUTT')
        self.assertEqual(state['club_range'], 300)
        self.assertEqual(state['club_unit'], 'ft')
        self.assertTrue(state['player'])
        self.assertFalse(state['complete'])
        self.assertEqual(state['par'],4,state['raw'])
        self.assertEqual(state['shot'],5,state['raw'])

class RuntimeTests(unittest.TestCase):
    def test_windows_input_layout(self):
        if sys.platform == 'win32':
            self.assertEqual(ctypes.sizeof(live.INPUT), 40 if ctypes.sizeof(ctypes.c_void_p)==8 else 28)

    def test_mouse_up_fallback_on_sendinput_error(self):
        with patch.object(live, 'send', side_effect=OSError('rejected')), patch.object(live, 'user32') as u:
            live.mouse_up()
            u.mouse_event.assert_called_once_with(live.UP,0,0,0,0)

    def test_f9_is_latched_and_releases_pressed_button(self):
        s = live.Safety(10, {})
        s.pressed = True
        with patch.object(live, 'key_down', return_value=True), patch.object(live, 'mouse_up') as up:
            with self.assertRaises(KeyboardInterrupt):
                s.check()
            self.assertTrue(s.stopped.is_set())
            self.assertFalse(s.pressed)
            up.assert_called_once()
        with patch.object(live, 'key_down', return_value=False):
            with self.assertRaises(KeyboardInterrupt):
                s.check()

    def test_focus_loss_and_resize_release(self):
        for moved in (False,True):
            s = live.Safety(10, {'left':0})
            s.pressed = True
            with patch.object(live,'key_down',return_value=False), patch.object(live,'user32') as u, \
                    patch.object(live,'rect_of',return_value={'left':1}), patch.object(live,'mouse_up') as up:
                u.GetForegroundWindow.return_value = 10 if moved else 20
                u.IsIconic.return_value = False
                with self.assertRaises(RuntimeError):
                    s.check()
                up.assert_called_once()

    def test_emergency_watcher_releases_while_main_thread_is_busy(self):
        s=live.Safety(10,{'left':0})
        with patch.object(live,'key_down',return_value=False) as key, \
                patch.object(live,'user32') as u, patch.object(live,'rect_of',return_value={'left':0}), \
                patch.object(live,'mouse_up') as up:
            u.GetForegroundWindow.return_value=10
            u.IsIconic.return_value=False
            s.start()
            s.pressed=True
            key.return_value=True
            self.assertTrue(s.stopped.wait(.5))
            s.close()
            up.assert_called_once()

    def test_virtual_desktop_negative_coordinates(self):
        with patch.object(live,'user32') as u,patch.object(live,'send') as send, \
                patch.object(live,'cursor',return_value=(-1200,-400)):
            u.GetSystemMetrics.side_effect=lambda n:{76:-1920,77:-1080,78:3840,79:2160}[n]
            live.move_to(-1200,-400)
            flags,x,y=send.call_args.args
            self.assertEqual(flags,live.MOVE|live.ABSOLUTE|live.VIRTUAL)
            self.assertAlmostEqual(x,round(720*65535/3839))
            self.assertAlmostEqual(y,round(680*65535/2159))

    def test_check_mode_sends_no_input(self):
        output=io.StringIO()
        with patch.object(live,'setup'),patch.object(live,'RECORD',FIXTURES/'wgt_calibration/last_manual_stroke.json'), \
                patch.object(live.pytesseract,'get_tesseract_version',return_value='5'), \
                patch.object(live,'send') as send,patch.object(live,'capture') as capture, \
                patch.object(live,'find_wgt') as find,redirect_stdout(output):
            result=live.main(['--check'])
            self.assertEqual(result,0,output.getvalue())
            send.assert_not_called()
            capture.assert_not_called()
            find.assert_not_called()

    def test_capture_error_during_drag_always_releases(self):
        from unittest.mock import Mock
        ref = core.StrokeReference.from_data(json.loads((FIXTURES/'wgt_calibration/last_manual_stroke.json').read_text()))
        safety = Mock()
        meter = core.Meter(73, 580, 950, 767,1056,535)
        rect = dict(left=0,top=0,width=1288,height=728)
        with patch.object(live,'move_to'), patch.object(live,'capture',side_effect=OSError('capture failed')):
            with self.assertRaises(OSError):
                live.drag_swing(safety,None,rect,ref,meter,.2)
        safety.press.assert_called_once()
        safety.release.assert_called_once()

    def test_short_putt_uses_normal_start_then_fine_meter_steps(self):
        from unittest.mock import Mock
        ref = core.StrokeReference.from_data(json.loads((FIXTURES/'wgt_calibration/last_manual_stroke.json').read_text()))
        safety = Mock()
        meter = core.Meter(73, 580, 950, 767,1056,535)
        rect = dict(left=0,top=0,width=1288,height=728)
        with patch.object(live,'move_to') as move, patch.object(live,'capture',return_value=band()), \
                patch.object(live,'yellow_power',side_effect=[None,None,.08,.126]):
            self.assertEqual(live.drag_swing(safety,None,rect,ref,meter,.127),.126)
        positions=[call.args[1] for call in move.call_args_list]
        self.assertEqual([first-second for first,second in zip(positions,positions[1:])],[4,4,1])
        safety.press.assert_called_once()
        safety.release.assert_called_once()

    def test_accuracy_interrupt_during_click_always_releases(self):
        from unittest.mock import Mock
        safety = Mock()
        safety.wait.side_effect = KeyboardInterrupt('F9')
        ref = core.StrokeReference.from_data(json.loads((FIXTURES/'wgt_calibration/last_manual_stroke.json').read_text()))
        with patch.object(live,'capture',return_value=band()), patch.object(live,'green_front',return_value=700), \
                patch.object(core.AccuracyTracker,'update',return_value=True), patch.object(live,'DEBUG',ROOT), \
                patch.object(Path,'write_text'), patch.object(cv2,'imwrite'):
            with self.assertRaises(KeyboardInterrupt):
                live.accuracy_click(safety,None,dict(left=0,top=0,width=1288,height=728),
                                    core.detect_meter(band()),ref)
        safety.press.assert_called_once()
        safety.release.assert_called_once()

    def test_simulated_hole_reaches_putting_then_completion(self):
        self.check_simulated_hole(practice_clock=False)

    def test_timed_solo_hole_keeps_preinput_checks_without_duplicate_reads(self):
        self.check_simulated_hole(practice_clock=True)

    def check_simulated_hole(self, practice_clock):
        from unittest.mock import Mock
        tee = dict(hole=1,shot=1,distance=210,unit='yd',mode='FULL',putt=False,club_range=240,club_unit='yd')
        if practice_clock:
            tee['game_context'] = dict(practice_clock=True,format='SOLO_STROKE',permission={'allowed': True})
        approach = dict(tee,shot=2,distance=60,club_range=80)
        putt = dict(tee,shot=3,distance=5.9,unit='ft',mode='PUTT',putt=True,club_range=30,club_unit='ft')
        frame = band()
        meter = core.detect_meter(frame)
        ref = Mock()
        with patch.object(live,'park_on_meter'),patch.object(live,'GAME') as game, \
                patch.object(live,'stable_state',side_effect=[(tee,frame),(approach,frame),(putt,frame)]), \
                patch.object(live,'select_club',side_effect=lambda a,b,c,state,frame:(state,frame)), \
                patch.object(live,'ready_meter',return_value=meter), \
                patch.object(live,'drag_swing',return_value=.5) as drag, \
                patch.object(live,'accuracy_click',return_value=True) as click, \
                patch.object(live,'wait_until_next',side_effect=[('next',approach),('next',putt),('complete',None)]), \
                patch.object(live,'DEBUG',ROOT),patch.object(Path,'write_text'),patch.object(cv2,'imwrite'), \
                redirect_stdout(io.StringIO()):
            game.completed=False
            self.assertEqual(live.run_hole(Mock(),None,{},ref), 'complete')
            self.assertEqual(drag.call_count,3)
            self.assertEqual(click.call_count,3)
            self.assertEqual(game.arm_stroke.call_count,3)
            self.assertEqual(game.wait_for_turn.call_count,1 if practice_clock else 3)
            self.assertEqual(game.require_turn.call_count,3 if practice_clock else 6)
            self.assertAlmostEqual(drag.call_args.args[-1],5.9/30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
