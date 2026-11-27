from pathlib import Path
import json
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
import execution_engine as live
from vision_engine import PowerCalibration

PROJECT=Path(r'Automation-Agent project root')
FIXTURE=Path(__file__).with_name('bunker.png')
if not FIXTURE.is_file():
    FIXTURE=PROJECT/'wgt_debug/bunker_transition_fixture_20261110.png'

class BunkerTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()
        cls.frame=cv2.imread(str(FIXTURE))

    def test_actual_player_name_and_birdie_label_verify_stroke_four(self):
        h,w=self.frame.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'),patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            r=live.inspect_screen(self.frame,dict(left=29,top=48,width=w,height=h))
        s=r['state']
        self.assertTrue(s['player'],s)
        self.assertEqual(s['shot'],4,s)
        self.assertEqual(s['distance'],12,s)
        self.assertEqual(s['club_name'],'LW',s)
        self.assertEqual(s['club_range'],52,s)
        self.assertTrue(r['verified'],r)
        send.assert_not_called()

    def test_replay_logged_third_shot_timeout_recovers_ready_fourth_shot(self):
        run=PROJECT/'wgt_debug/runs/20261110_142623_d97cc5cd'
        paths=sorted(run.glob('post_shot_read_*.png'))[-2:]
        self.assertEqual(len(paths),2)
        frames=[cv2.imread(str(path)) for path in paths]
        self.assertTrue(all(frame is not None for frame in frames))
        before=json.loads((run/'shot_03_plan.json').read_text())['state']
        rect=json.loads((run/'initial_geometry.json').read_text())['rect']
        motion=json.loads((run/'shot_03_motion.json').read_text())
        observer=Mock()
        observer.report.return_value=motion
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'), \
                patch.object(live,'capture',side_effect=frames),patch.object(live,'DIAGNOSTICS',None), \
                patch.object(live,'send') as send:
            outcome,after=live.wait_until_next(Mock(),None,rect,before,observer=observer)
        self.assertEqual(outcome,'next')
        self.assertEqual((after['hole'],after['shot']),(10,4))
        self.assertTrue(after['player'])
        self.assertEqual(after['distance'],12)
        self.assertTrue(after['motion']['scene_motion_observed'])
        send.assert_not_called()

    def test_unknown_s2_name_is_not_blindly_changed_to_52(self):
        with patch.object(live,'ocr_variants',return_value=['TIGERWOULDS2']*6):
            known,_=live.read_player_header(np.zeros((728,1288,3),np.uint8))
        self.assertFalse(known)

    def test_actual_fresh_shot_statistics_preserve_decimals_and_zero_spin(self):
        stats=live.shot_stats(self.frame)
        self.assertIsNotNone(stats)
        self.assertEqual(stats['carry_yd'],126.22)
        self.assertEqual(stats['roll_yd'],7.87)
        self.assertEqual(stats['power_pct'],87)
        self.assertEqual(stats['spin_pct'],0)
        self.assertEqual(stats['apex_ft'],51.35)

    def test_wrapping_bag_accepts_verified_shortest_wedge_for_short_shot(self):
        s=dict(distance=12,unit='yd',club_range=52,club_unit='yd',club_name='LW',putt=False,mode='FULL',
               elevation_ft=3,green_speed=10,hole=10,shot=4,wind=dict(mean_mph=11,screen_dx=0,screen_dy=1))
        states=[dict(s,club_name=c,club_range=r) for c,r in [('PW',64),('LW',52),('9.0D',282),('LW',52)]]
        with patch.object(live,'POWER_MODEL',PowerCalibration()),patch.object(live,'move_to'),patch.object(live,'send') as send, \
                patch.object(live,'stable_state',side_effect=[(v,None) for v in states]),patch.object(live,'diagnostic_state'):
            selected,_=live.select_club(Mock(),None,dict(left=29,top=48,width=1288,height=728),s,None)
        self.assertEqual(selected['club_name'],'LW')
        self.assertEqual(send.call_count,4)

if __name__=='__main__':
    unittest.main()
