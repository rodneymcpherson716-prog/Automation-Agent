"""V48 transition integration and actual diagnostic-frame regressions; no input."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
import execution_engine as live
from automation_core import Meter, TransitionTracker
from vision_engine import PowerCalibration

PROJECT=Path(r'Automation-Agent project root')

def state(**changes):
    return dict(dict(player=True,hole=11,shot=7,par=4,distance=38.,unit='yd',
        club_range=190.,club_unit='yd',club_name='3HY',mode='FULL',putt=False,
        elevation_ft=2.,green_speed=10.,wind=dict(low_mph=13,high_mph=15,
            mean_mph=14.,screen_dx=0.,screen_dy=1.)),**changes)

class RepairTests(unittest.TestCase):
    def test_penalty_stroke_advance_stops_before_automatic_repeat(self):
        before=state(shot=3,distance=131.)
        after=state(shot=5,distance=131.)
        observer=Mock()
        observer.report.return_value=dict(scene_motion_observed=True,moving_ball_candidate_tracked=True,capture_error=None)
        with patch.object(live,'GAME',None),patch.object(live,'capture',return_value=np.zeros((728,1288,3),np.uint8)), \
                patch.object(live,'ocr_variants',return_value=['']),patch.object(live,'shot_state',return_value=after), \
                patch.object(live,'swing_button',return_value=(1000,500)),patch.object(live,'ready_meter',return_value=Mock()), \
                patch.object(live,'diagnostic_state'),patch.object(live,'send') as send:
            with self.assertRaisesRegex(RuntimeError,'possible penalty or external stroke'):
                live.wait_until_next(Mock(),None,dict(width=1288,height=728),before,observer=observer)
        send.assert_not_called()

    def test_real_left_badge_is_38_not_hole_146_or_hybrid_range_190(self):
        live.setup()
        fixture=Path(__file__).with_name('left_target_hybrid.png')
        if not fixture.is_file():
            fixture=PROJECT/'automation_debug/left_target_hybrid_fixture_20261110.png'
        f=cv2.imread(str(fixture))
        self.assertIsNotNone(f)
        h,w=f.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            r=live.inspect_screen(f,dict(left=37,top=46,width=w,height=h))
        s=r['state']
        self.assertEqual(s['distance'],38,s)
        self.assertEqual(s['hole_distance'],146,s)
        self.assertEqual(s['club_range'],190,s)
        self.assertEqual(s['club_name'],'3HY',s)
        self.assertEqual(s['shot'],8,s)
        self.assertEqual(s['elevation_ft'],2,s)
        self.assertTrue(r['verified'],r)
        send.assert_not_called()

    def test_transition_can_verify_stroke_advance_with_hidden_next_target(self):
        tracker=TransitionTracker(state())
        after=state(shot=8,distance=None,unit=None)
        self.assertIsNone(tracker.update(after,True))
        self.assertEqual(tracker.update(after,True),'next')

    def test_unseen_stroke_glyph_is_not_guessed_from_sthshot(self):
        frame=np.zeros((728,1288,3),np.uint8)
        self.assertIsNone(live.read_stroke_glyphs(frame,['STHSHOT','STHSHOT']))
        self.assertIsNone(live.read_stroke_glyphs(frame,['FOR PAR','FOR PAR']))

    def test_target_change_without_stroke_increment_cannot_complete_shot(self):
        tracker=TransitionTracker(state())
        for _ in range(3):
            self.assertIsNone(tracker.update(state(distance=45),True))

    def test_live_wait_verifies_hidden_next_target_only_with_motion_and_counter(self):
        before=state()
        after=state(shot=8,distance=None,unit=None)
        observer=Mock()
        observer.report.return_value=dict(scene_motion_observed=True,moving_ball_candidate_tracked=True,capture_error=None)
        with patch.object(live,'capture',return_value=np.zeros((728,1288,3),np.uint8)), \
                patch.object(live,'ocr_variants',return_value=['']),patch.object(live,'shot_state',return_value=after), \
                patch.object(live,'swing_button',return_value=(1000,500)),patch.object(live,'ready_meter',return_value=Mock()), \
                patch.object(live,'shot_stats',return_value=None),patch.object(live,'diagnostic_state'), \
                patch.object(live,'send') as send:
            result,observed=live.wait_until_next(Mock(),None,dict(width=1288,height=728),before,observer=observer)
        self.assertEqual(result,'next')
        self.assertIsNone(observed['distance'])
        self.assertEqual(observed['shot'],8)
        self.assertIn('movement_verification',observed)
        send.assert_not_called()

    def test_missing_transition_identity_and_hidden_target_block_new_shot(self):
        with patch.object(live,'swing_button',return_value=(1000,500)), \
                patch.object(live,'ready_meter',return_value=Mock()):
            problems=live.readiness_problems(state(hole=None,shot=None,distance=None,unit=None),None,{},
                                             expected=dict(hole=11,shot=8))
        self.assertIn('hole label',problems)
        self.assertIn('stroke label',problems)
        self.assertIn('target distance/unit',problems)

    def stable_sequence(self,values,expected=None):
        clock=[0.]
        index=[0]
        def capture(*args):
            clock[0]+=.2
            return np.zeros((2,2,3),np.uint8)
        def read(*args):
            i=min(index[0],len(values)-1)
            index[0]+=1
            return values[i]
        with patch.object(live.time,'perf_counter',side_effect=lambda:clock[0]), \
                patch.object(live,'capture',side_effect=capture),patch.object(live,'shot_state',side_effect=read), \
                patch.object(live,'swing_button',return_value=(1000,500)), \
                patch.object(live,'ready_meter',return_value=Mock()),patch.object(live,'send') as send, \
                redirect_stdout(io.StringIO()):
            result=live.stable_state(Mock(),None,{},timeout=2,expected=expected)
            send.assert_not_called()
        return result,index[0]

    def test_transient_missing_hud_then_two_complete_readings(self):
        s=state(shot=8)
        (result,_),reads=self.stable_sequence([state(shot=8,distance=None),s,s],dict(hole=11,shot=8))
        self.assertEqual(result['shot'],8)
        self.assertEqual(reads,3)

    def test_two_matching_ranges_with_different_clubs_do_not_agree(self):
        a=state(club_name='3I')
        b=state(club_name='4I')
        (_, _),reads=self.stable_sequence([a,b,b])
        self.assertEqual(reads,3)

    def test_visible_wrong_stroke_stops_reacquisition(self):
        with self.assertRaisesRegex(RuntimeError,'does not match'):
            self.stable_sequence([state(shot=7)],dict(hole=11,shot=8))

    def test_hidden_next_target_never_repeats_previous_swing(self):
        before=state()
        after=state(shot=8,distance=None,unit=None)
        with patch.object(live,'park_on_meter'),patch.object(live,'GAME') as game, \
                patch.object(live,'stable_state',side_effect=[(before,np.zeros((2,2,3),np.uint8)),RuntimeError('target hidden')]) as ready, \
                patch.object(live,'select_club',side_effect=lambda a,b,c,s,f:(s,f)), \
                patch.object(live,'ready_meter',return_value=Meter(73, 200, 540, 767, 900, zero=535)), \
                patch.object(live,'drag_swing',return_value=.8) as drag, \
                patch.object(live,'accuracy_click') as accuracy, \
                patch.object(live,'wait_until_next',return_value=('next',after)), \
                patch.object(live,'DEBUG',Path(__file__).parent),patch.object(Path,'write_text'), \
                patch.object(cv2,'imwrite'), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'target hidden'):
                game.completed=False
                live.run_hole(Mock(),None,{},Mock(),max_shots=0)
        self.assertEqual(drag.call_count,1)
        self.assertEqual(accuracy.call_count,1)
        self.assertEqual(game.arm_stroke.call_count,1)
        self.assertEqual(ready.call_args.kwargs['expected'],dict(hole=11,shot=8))

    def test_club_selection_uses_headwind_adjusted_power_not_distance_ratio(self):
        model=PowerCalibration()
        s=state(distance=90,club_range=100,elevation_ft=0,wind=dict(low_mph=20,high_mph=20,mean_mph=20,screen_dx=0,screen_dy=1))
        bigger=dict(s,club_range=120,club_name='9I')
        with patch.object(live,'key_down',return_value=False),patch.object(live,'POWER_MODEL',model),patch.object(live,'move_to'), \
                patch.object(live,'send') as send,patch.object(live,'stable_state',return_value=(bigger,None)):
            chosen,_=live.select_club(Mock(),None,dict(left=0,top=0,width=1288,height=728),s,None)
        self.assertEqual(chosen['club_range'],120)
        self.assertEqual(send.call_count,1)

    def test_oversized_putter_can_be_reselected_but_cannot_be_swung(self):
        s=state(distance=.6,club_range=300,mode='PUTT',putt=True,unit='ft',club_unit='ft',elevation_ft=0,club_name='PUTTER')
        model=PowerCalibration()
        self.assertLess(model.plan(s,for_selection=True)['unclipped_power'],.025)
        with self.assertRaises(ValueError):
            model.plan(s)

    def test_tap_in_uses_verified_shortest_range_above_hard_minimum(self):
        s=state(distance=.6,club_range=30,mode='PUTT',putt=True,unit='ft',club_unit='ft',elevation_ft=0,club_name='PUTTER')
        choices=[dict(s,club_range=r) for r in (40,30,10,10)]
        model=PowerCalibration()
        with patch.object(live,'key_down',return_value=False),patch.object(live,'POWER_MODEL',model),patch.object(live,'move_to'), \
                patch.object(live,'send') as send,patch.object(live,'stable_state',side_effect=[(v,None) for v in choices]):
            chosen,_=live.select_club(Mock(),None,dict(left=0,top=0,width=1288,height=728),s,None)
        self.assertEqual(chosen['club_range'],10)
        self.assertAlmostEqual(model.plan(chosen)['power'],.06)
        self.assertEqual(send.call_count,4)

    def test_putting_curve_does_not_discard_verified_samples_for_air_wind(self):
        s=state(distance=10,club_range=30,mode='PUTT',putt=True,unit='ft',club_unit='ft',elevation_ft=0,club_name='PUTTER')
        records=[dict(shot_id=str(i),verified=True,mode='PUTT',unit='ft',club_name='PUTTER',club_range=30,
                      elevation_ft=0,wind_mph=15,green_speed=10,travel_basis='total',power=p,travel=d)
                 for i,(p,d) in enumerate(((.2,5),(.6,15)))]
        self.assertAlmostEqual(PowerCalibration(records).plan(s)['power'],.4)

if __name__=='__main__':
    unittest.main()
