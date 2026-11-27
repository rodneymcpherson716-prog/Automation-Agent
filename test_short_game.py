"""Offline short-game regressions. Every mouse/input path is mocked."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import execution_engine as live
from automation_core import Meter
from vision_engine import PowerCalibration

PROJECT=Path(r'Automation-Agent project root')
FIXTURE=Path(__file__).with_name('short_game_live.png')
if not FIXTURE.is_file():
    FIXTURE=PROJECT/'wgt_debug/short_game_fixture_20261110.png'


def state(mode='CHIP',**changes):
    value=dict(player=True,hole=10,par=5,shot=5,distance=10.,unit='yd',putt=False,mode=mode,
               club_name='PW',club_range=17.,club_unit='yd',elevation_ft=4/12,
               wind=dict(low_mph=11,high_mph=13,mean_mph=12.,screen_dx=0.,screen_dy=-1.),green_speed=10.)
    return dict(value,**changes)


class ShortGameTests(unittest.TestCase):
    def test_latest_mss_putting_frame_preserves_decimal_distance_and_units(self):
        live.setup()
        frame=cv2.imread(str(FIXTURE))
        self.assertIsNotNone(frame)
        h,w=frame.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            reading=live.inspect_screen(frame,dict(left=37,top=46,width=w,height=h))
        self.assertEqual(reading['state']['mode'],'PUTT',reading)
        self.assertEqual(reading['state']['distance'],2.4,reading)
        self.assertEqual(reading['state']['unit'],'ft',reading)
        self.assertEqual(reading['state']['club_range'],30,reading)
        self.assertEqual(reading['state']['club_unit'],'ft',reading)
        self.assertAlmostEqual(reading['state']['elevation_ft'],-.7/12,msg=str(reading))
        self.assertTrue(reading['verified'],reading)
        send.assert_not_called()

    def test_chip_and_pitch_can_pass_ready_guard_without_sending_input(self):
        for mode in ('CHIP','PITCH'):
            s=state(mode)
            with patch.object(live,'capture',return_value=np.zeros((728,1288,3),np.uint8)), \
                    patch.object(live,'shot_state',return_value=s),patch.object(live,'swing_button',return_value=(1153,529)), \
                    patch.object(live,'ready_meter',return_value=Meter(73,598,932,767,1056,535)), \
                    patch.object(live,'send') as send:
                observed,_=live.stable_state(Mock(),None,{})
                self.assertEqual(observed['mode'],mode)
                send.assert_not_called()

    def test_hidden_target_still_blocks_short_game(self):
        with patch.object(live,'capture',return_value=np.zeros((728,1288,3),np.uint8)), \
                patch.object(live,'shot_state',return_value=state('PITCH',distance=None)), \
                patch.object(cv2,'imwrite'),patch.object(live,'send') as send:
            with self.assertRaises(RuntimeError):
                live.stable_state(Mock(),None,dict(left=0,top=0,width=1288,height=728),timeout=.01)
            send.assert_not_called()

    def test_mixed_units_and_unsupported_modes_still_block(self):
        for s in (state('PUNCH'),state('FLOP'),state(unit='ft')):
            with patch.object(live,'capture',return_value=np.zeros((728,1288,3),np.uint8)), \
                    patch.object(live,'shot_state',return_value=s),patch.object(cv2,'imwrite'),patch.object(live,'send') as send:
                with self.assertRaises(RuntimeError):
                    live.stable_state(Mock(),None,dict(left=0,top=0,width=1288,height=728),timeout=.01)
                send.assert_not_called()

    def test_short_game_power_uses_selected_mode_range_and_conditions(self):
        model=PowerCalibration()
        flat=model.plan(state(elevation_ft=0,wind=dict(mean_mph=0,screen_dx=0,screen_dy=0)))
        self.assertAlmostEqual(flat['power'],10/17)
        chip=model.plan(state())
        self.assertAlmostEqual(chip['power'],(10+(4/12)/3)*(1-.003*12)/17)
        self.assertIn('chip baseline',chip['correction'])
        pitch=model.plan(state('PITCH',distance=25,club_range=32))
        self.assertLess(pitch['power'],1)
        self.assertIn('pitch baseline',pitch['correction'])

    def test_short_game_calibration_records_carry_plus_roll(self):
        stats=dict(carry_yd=19.56,roll_yd=16.05,total_yd=35.61,power_pct=99,spin_pct=0,apex_ft=11.84)
        with tempfile.TemporaryDirectory() as folder:
            model=PowerCalibration()
            for mode in ('CHIP','PITCH','FULL'):
                model.record(Path(folder)/(mode+'.json'),state(mode),stats,.99,mode)
            self.assertEqual(model.records[0]['travel'],35.61)
            self.assertEqual(model.records[1]['travel'],35.61)
            self.assertEqual(model.records[2]['travel'],19.56)

    def test_short_game_rejects_old_carry_only_calibration(self):
        points=[dict(shot_id=str(i),verified=True,mode='CHIP',unit='yd',club_name='PW',club_range=17,
                     wind_mph=0,elevation_ft=0,power=p,travel=d) for i,(p,d) in enumerate(((.3,4),(.8,14)))]
        s=state(elevation_ft=0,wind=dict(mean_mph=0,screen_dx=0,screen_dy=0))
        self.assertIsNone(PowerCalibration(points).curve(s))
        corrected=[dict(p,travel_basis='total') for p in points]
        self.assertIsNotNone(PowerCalibration(corrected).curve(s))

    def test_simulated_pitch_chip_putt_hole_uses_same_guarded_swing_path(self):
        shots=[state('PITCH',distance=25,club_range=32,shot=4),state('CHIP'),
               state('PUTT',distance=6,unit='ft',club_range=30,club_unit='ft',putt=True,shot=6,club_name='PUTTER')]
        frame=np.zeros((728,1288,3),np.uint8)
        meter=Meter(73,598,932,767,1056,535)
        with patch.object(live,'GAME') as game, \
                patch.object(live,'park_on_meter'), \
                patch.object(live,'stable_state',side_effect=[(s,frame) for s in shots]), \
                patch.object(live,'select_club',side_effect=lambda a,b,c,s,f:(s,f)), \
                patch.object(live,'ready_meter',return_value=meter), \
                patch.object(live,'drag_swing',return_value=.5) as drag, \
                patch.object(live,'accuracy_click',return_value=True) as accuracy, \
                patch.object(live,'wait_until_next',side_effect=[('next',shots[1]),('next',shots[2]),('complete',None)]), \
                patch.object(live,'DEBUG',Path(__file__).parent),patch.object(Path,'write_text'), \
                patch.object(cv2,'imwrite'),redirect_stdout(io.StringIO()):
            game.completed=False
            result=live.run_hole(Mock(),None,{},Mock(),max_shots=0)
        self.assertEqual(result,'complete')
        self.assertEqual(drag.call_count,3)
        self.assertEqual(accuracy.call_count,3)
        self.assertEqual(game.arm_stroke.call_count,3)


if __name__=='__main__':
    unittest.main(verbosity=2)
