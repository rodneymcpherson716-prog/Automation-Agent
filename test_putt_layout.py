"""The actual first green reached by the automatic controller; no live input."""
from pathlib import Path
from unittest.mock import Mock,patch
import json,unittest
import cv2
import execution_engine as live

ROOT=Path(r'Automation-Agent project root')
RUN=ROOT/'wgt_debug/runs/20261112_020415_9840572f'
RECT=dict(left=37,top=46,width=1288,height=728)

class PuttLayoutTests(unittest.TestCase):
    def test_actual_high_approach_badge_reads_without_camera_movement(self):
        frame=cv2.imread(str(ROOT/'wgt_debug/runs/20261114_120746_eaa7ed9f/hole_01_ready_wait_5.png'))
        sign=live.find_distance_sign(frame)
        self.assertIsNotNone(sign)
        self.assertLess(sign[1],.09*frame.shape[0])
        with patch.object(live,'GAME',None),patch.object(live,'send') as send:
            state=live.shot_state(frame)
        self.assertEqual((state['distance'],state['unit']),(131.,'yd'))
        self.assertEqual(state['elevation_ft'],1.)
        self.assertEqual(live.readiness_problems(state,frame,RECT),[])
        send.assert_not_called()

    def test_actual_flag_obscured_badge_with_sixty_foot_putter(self):
        run=ROOT/'wgt_debug/runs/20261117_220207_51c60201'
        for name in ('ready_wait_1.png','ready_wait_19.png'):
            with self.subTest(frame=name):
                frame=cv2.imread(str(run/name))
                with patch.object(live,'GAME',None):
                    state=live.shot_state(frame)
                self.assertEqual((state['distance'],state['unit']),(14.6,'ft'))
                self.assertEqual((state['club_range'],state['club_unit']),(60.,'ft'))
                self.assertAlmostEqual(state['elevation_ft'],.6/12)
                self.assertEqual(live.readiness_problems(state,frame,RECT),[])

    def test_actual_stippled_golfer_badge_reads_and_missing_fields_still_refuse(self):
        from vision_engine import quick_putt_badge
        frame=cv2.imread(str(ROOT/'wgt_debug/runs/20261114_115254_1e8a1fc2/initial_geometry.png'))
        sign=live.find_distance_sign(frame)
        badge=quick_putt_badge(frame,sign,live.ocr_text)
        self.assertIsNotNone(badge)
        self.assertEqual((badge['distance'],badge['unit']),(7.6,'ft'))
        self.assertAlmostEqual(badge['elevation_ft'],.1/12)
        self.assertEqual(badge['recovery'],'verified_periodic_golfer_stipple')
        other=cv2.imread(str(ROOT/'wgt_debug/runs/20261114_120134_67092f58/initial_geometry.png'))
        other_badge=quick_putt_badge(other,live.find_distance_sign(other),live.ocr_text)
        self.assertEqual(other_badge,badge)
        x,y,w,h=sign
        for a,b,c,d in ((.12*w,.06*w,.74*w,.40*w),(.69*w,.14*w,.99*w,.40*w),
                        (.33*w,.49*h,.73*w,.95*h),(.65*w,.54*h,w,.96*h)):
            altered=frame.copy()
            altered[y+int(b):y+int(d),x+int(a):x+int(c)]=0
            self.assertIsNone(quick_putt_badge(altered,sign,live.ocr_text))

    @classmethod
    def setUpClass(cls):
        live.setup()
        cls.frames=[cv2.imread(str(RUN/f'post_shot_read_{stamp}.png')) for stamp in
            ('1794470885672166200','1794470933101090500')]

    def test_two_real_green_frames_read_decimal_distance_range_and_elevation(self):
        for frame in self.frames:
            with self.subTest(frame=id(frame)),patch.object(live,'CALIBRATION',ROOT/'automation_accuracy.json'),patch.object(live,'SWING_TEMPLATE',ROOT/'swing_button.png'),patch.object(live,'GAME',None),patch.object(live,'send') as send:
                state=live.shot_state(frame)
                self.assertEqual((state['distance'],state['unit']),(17.5,'ft'))
                self.assertEqual(state['distance_source'],'floating_target')
                self.assertEqual((state['club_range'],state['club_unit']),(30,'ft'))
                self.assertEqual((state['mode'],state['club_name'],state['shot']),('PUTT','PUTTER',4))
                self.assertAlmostEqual(state['elevation_ft'],.6/12)
                self.assertEqual(live.readiness_problems(state,frame,RECT),[])
                send.assert_not_called()

    def test_hidden_badge_digits_are_not_reconstructed(self):
        frame=self.frames[0].copy();sign=live.find_distance_sign(frame)
        x,y,w,h=sign
        frame[y+int(.04*w):y+int(.42*w),x+int(.10*w):x+int(.80*w)]=0
        self.assertIsNone(live.read_sign_distance(frame,sign))

    def test_actual_lower_badge_position_reads_without_camera_input(self):
        path=Path(__file__).with_name('low_badge.png')
        if not path.is_file():path=ROOT/'wgt_debug/putt_low_badge_20261112.png'
        frame=cv2.imread(str(path))
        with patch.object(live,'GAME',None),patch.object(live,'send') as send:
            state=live.shot_state(frame)
        self.assertEqual((state['distance'],state['unit']),(17.5,'ft'))
        self.assertEqual((state['club_range'],state['club_unit']),(30,'ft'))
        self.assertAlmostEqual(state['elevation_ft'],.6/12)
        self.assertGreater(state['sign'][1],frame.shape[0]*.5)
        self.assertEqual(live.readiness_problems(state,frame,RECT),[])
        send.assert_not_called()

    def test_hidden_elevation_unit_stays_unknown(self):
        frame=self.frames[0].copy();x,y,w,h=live.find_distance_sign(frame)
        frame[y+int(.50*h):y+int(.96*h),x+int(.73*w):x+int(.98*w)]=0
        with patch.object(live,'GAME',None):state=live.shot_state(frame)
        self.assertIsNone(state['elevation_ft'])

    def test_large_inch_variant_is_restricted_to_elevation_labels(self):
        import vision_engine as obs
        path=Path(__file__).with_name('low_badge.png')
        if not path.is_file():path=ROOT/'wgt_debug/putt_low_badge_20261112.png'
        frame=cv2.imread(str(path));x,y,w,h=live.find_distance_sign(frame)
        unit=frame[y+int(.54*h):y+int(.96*h),x+int(.65*w):x+w]
        self.assertIsNone(obs.unit_from_glyphs(unit,w))
        self.assertEqual(obs.unit_from_glyphs(unit,w,elevation=True),'IN')

    def test_missing_printed_putter_unit_cannot_be_assumed_from_putt_mode(self):
        frame=self.frames[0].copy();h,w=frame.shape[:2]
        frame[int(.966*h):int(.989*h),int(.122*w):int(.172*w)]=0
        with patch.object(live,'GAME',None):state=live.shot_state(frame)
        self.assertIsNone(state['club_unit'])
        self.assertIsNone(state['club_range'])

    def test_missing_vertical_badge_edge_is_rejected(self):
        frame=self.frames[0].copy()
        frame[138:234,698:708]=frame[138:234,690:700]
        self.assertIsNone(live.find_distance_sign(frame))

    def test_actual_third_shot_motion_and_new_putter_hud_verify_transition(self):
        before=json.loads((RUN/'shot_03_plan.json').read_text())['state']
        observer=Mock();observer.report.return_value=json.loads((RUN/'shot_03_motion.json').read_text())
        with patch.object(live,'GAME',None),patch.object(live,'CALIBRATION',ROOT/'automation_accuracy.json'),patch.object(live,'SWING_TEMPLATE',ROOT/'swing_button.png'),patch.object(live,'DIAGNOSTICS',None),patch.object(live,'capture',side_effect=self.frames),patch.object(live,'send') as send:
            outcome,after=live.wait_until_next(Mock(),None,RECT,before,observer=observer)
        self.assertEqual(outcome,'next')
        self.assertEqual((after['shot'],after['distance'],after['unit']),(4,17.5,'ft'))
        self.assertTrue(after['motion']['scene_motion_observed'])
        send.assert_not_called()

if __name__=='__main__':unittest.main()
