import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import tempfile
import unittest
import cv2
import numpy as np
import execution_engine as live
from vision_engine import Session,quick_putt_badge,read_putter_range
from round_manager import scorecard,run_round,continue_to_tee,next_hole
import test_putt_freshness as freshness_tests

ROOT=Path(__file__).parent

class NineHoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()

    def test_actual_live_putt_decimal_and_units(self):
        frame=cv2.imread(str(ROOT/'automation_debug/nine_hole_putt_reference_20261112.png'))
        badge=quick_putt_badge(frame,live.find_distance_sign(frame),live.ocr_text)
        self.assertIsNotNone(badge)
        self.assertEqual((badge['distance'],badge['unit']),(17.5,'ft'))
        self.assertAlmostEqual(badge['elevation_ft'],.05)

    def test_hidden_putt_unit_and_number_refuse(self):
        frame=cv2.imread(str(ROOT/'automation_debug/nine_hole_putt_reference_20261112.png'))
        sign=live.find_distance_sign(frame);x,y,w,h=sign
        for region in ((x+int(.69*w),y+int(.12*w),x+w,y+int(.40*w)),
                       (x+int(.12*w),y+int(.06*w),x+int(.74*w),y+int(.40*w))):
            altered=frame.copy();a,b,c,d=region;altered[b:d,a:c]=0
            self.assertIsNone(quick_putt_badge(altered,sign,live.ocr_text))

    def test_faded_target_cannot_turn_five_point_nine_into_nine_point_nine(self):
        frame=cv2.imread(str(ROOT/'automation_debug/v47_before_shot_1.png'))
        sign=live.find_distance_sign(frame)
        self.assertIsNone(quick_putt_badge(frame,sign,live.ocr_text))
        state=live.shot_state(frame)
        self.assertEqual((state['distance'],state['unit']),(5.9,'ft'))

    def test_actual_putter_range_excludes_arrow_artifact(self):
        frame=cv2.imread(str(ROOT/'automation_debug/nine_hole_putt_reference_20261112.png'))
        state=live.shot_state(frame)
        self.assertEqual((state['club_range'],state['club_unit']),(30.,'ft'))
        self.assertEqual(json.loads(json.dumps(state))['raw']['putter_range_evidence']['glyph_bounds'],
                         state['raw']['putter_range_evidence']['glyph_bounds'])

    def test_real_30_150_300_ranges_keep_all_digits(self):
        for path,value in (('nine_hole_putt_reference_20261112.png',30),
            ('runs/20261112_034917_3d0c2dc8/ready_wait_4.png',150),
            ('runs/20261112_040025_fd6264f3/ready_wait_4.png',300)):
            frame=cv2.imread(str(ROOT/'automation_debug'/path))
            with self.subTest(value=value):
                self.assertEqual(read_putter_range(frame,live.ocr_text)['value'],value)

    def fifty_frame(self):
        frame=cv2.imread(str(ROOT/'automation_debug/nine_hole_putt_reference_20261112.png'))
        source=cv2.imread(str(ROOT/'automation_debug/runs/20261112_034917_3d0c2dc8/ready_wait_4.png'))
        h,w=frame.shape[:2];left,top=int(.122*w),int(.938*h)
        digits=source[top:int(.966*h),left:int(.172*w)]
        frame[top:int(.966*h),left:int(.172*w)]=20
        frame[top+6:top+17,left+13:left+20]=digits[6:17,15:22]
        frame[top+6:top+17,left+21:left+28]=digits[6:17,24:31]
        return frame

    def test_fifty_cannot_become_five_when_zero_is_hidden(self):
        frame=self.fifty_frame()
        self.assertEqual(read_putter_range(frame,live.ocr_text)['value'],50)
        h,w=frame.shape[:2]
        frame[int(.938*h)+6:int(.938*h)+17,int(.122*w)+21:int(.122*w)+28]=20
        self.assertIsNone(read_putter_range(frame,live.ocr_text))

    def test_visible_five_foot_range_is_accepted_when_centred(self):
        frame=self.fifty_frame();h,w=frame.shape[:2]
        left,top=int(.122*w),int(.938*h)
        five=frame[top+6:top+17,left+13:left+20].copy()
        frame[top:int(.966*h),left:int(.172*w)]=20
        frame[top+6:top+17,left+17:left+24]=five
        found=read_putter_range(frame,live.ocr_text)
        self.assertIsNotNone(found)
        self.assertEqual((found['value'],found['unit']),(5.,'ft'))

    def test_partially_cropped_fifty_is_never_accepted_as_five(self):
        for removed_columns in (1,2,4,7):
            frame=self.fifty_frame();h,w=frame.shape[:2]
            left,top=int(.122*w),int(.938*h)
            frame[top+6:top+17,left+28-removed_columns:left+28]=20
            found=read_putter_range(frame,live.ocr_text)
            self.assertTrue(found is None or found['value']!=5.)

    def test_range_conflict_digit_count_and_unit_conflict_refuse(self):
        frame=self.fifty_frame()
        for votes in (['50','60'],['5','5'],['50','50','FT','F'],['50','50','FT','TF']):
            with self.subTest(votes=votes):
                self.assertIsNone(read_putter_range(frame,Mock(side_effect=votes)))

    def test_erased_unit_cannot_be_inferred_from_putt_mode(self):
        frame=self.fifty_frame();h,w=frame.shape[:2]
        frame[int(.966*h):int(.989*h),int(.125*w):int(.149*w)]=0
        self.assertIsNone(read_putter_range(frame,live.ocr_text))

    def test_golf_hud_and_permanent_scorecard_label_do_not_complete(self):
        frame=cv2.imread(str(ROOT/'automation_debug/nine_hole_putt_reference_20261112.png'))
        self.assertIsNone(scorecard(frame,Mock(return_value=['SCORECARD'])))

    def test_result_modal_requires_all_labels_and_two_continue_reads(self):
        frame=np.zeros((728,1288,3),np.uint8)
        cv2.rectangle(frame,(355,223),(930,550),(250,250,250),-1)
        cv2.rectangle(frame,(820,510),(914,537),(105,105,105),-1)
        ocr=Mock(side_effect=[['SCORECARD STATS METER STATS']*2,['PUTTS PAR'],['CONTINUE']*2])
        self.assertIsNotNone(scorecard(frame,ocr))
        ocr=Mock(side_effect=[['SCORECARD STATS METER STATS']*2,['PUTTS PAR'],['CONTINUE','EXIT']])
        self.assertIsNone(scorecard(frame,ocr))

    def test_nine_completions_and_only_eight_advances(self):
        game=SimpleNamespace(completed=False,completed_hole=None)
        controller=SimpleNamespace(GAME=game,DIAGNOSTICS=Mock())
        def hole(*args):
            game.completed_hole=9+controller.run_hole.call_count
            return 'complete'
        controller.run_hole=Mock(side_effect=hole)
        with patch('round_manager.continue_to_tee') as advance:
            result=run_round(controller,Mock(),None,{},None,9,15)
        self.assertEqual(result['completed_holes'],list(range(10,19)))
        self.assertEqual(advance.call_count,8)

    def test_no_advance_after_unverified_hole(self):
        controller=SimpleNamespace(GAME=SimpleNamespace(completed=False),DIAGNOSTICS=Mock(),run_hole=Mock(return_value='one_shot'))
        with patch('round_manager.continue_to_tee') as advance:
            with self.assertRaisesRegex(RuntimeError,'unverified'):run_round(controller,Mock(),None,{},None,9,15)
            advance.assert_not_called()

    def test_f9_prevents_continue_capture_and_click(self):
        safety=Mock();safety.check.side_effect=KeyboardInterrupt('F9')
        controller=Mock()
        with self.assertRaises(KeyboardInterrupt):continue_to_tee(controller,safety,None,{},10)
        controller.capture.assert_not_called();safety.press.assert_not_called()

    def test_continue_is_not_reclicked_when_next_tee_is_wrong(self):
        controller=Mock();safety=Mock();safety.hwnd=1
        controller.capture.return_value=np.zeros((200,300,3),np.uint8)
        controller.stable_state.return_value=({'hole':10,'shot':4},None)
        found=dict(button=[100,100,50,20])
        with patch('round_manager.scorecard',return_value=found):
            with self.assertRaisesRegex(RuntimeError,'Next tee identity'):
                continue_to_tee(controller,safety,None,dict(left=0,top=0),10)
        self.assertEqual(safety.press.call_count,1)
        self.assertEqual(safety.release.call_count,1)
        controller.run_hole.assert_not_called()

    def test_continue_clicks_scorecard_and_intro_once_each(self):
        controller, safety = Mock(), Mock()
        safety.hwnd = 1
        controller.capture.return_value = np.zeros((728,1288,3),np.uint8)
        controller.swing_button.side_effect = [None, None, True]
        controller.stable_state.return_value = ({'hole':12,'shot':1}, None)
        score = dict(button=[820,510,94,27])
        intro = dict(kind='hole_intro',hole=12,button=[817,505,102,37])
        with patch('round_manager.scorecard',return_value=score), patch('round_manager.hole_intro',return_value=intro) as find_intro:
            result = continue_to_tee(controller,safety,None,dict(left=0,top=0),11)
        self.assertEqual(result, {'hole':12,'shot':1})
        self.assertEqual(find_intro.call_count, 2)
        self.assertEqual(safety.press.call_count, 2)
        self.assertEqual(safety.release.call_count, 2)

    def test_session_scopes_preserve_each_hole_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            s=Session(root)
            for number in (1,2):
                s.scope=f'hole_{number:02d}_';s.state('shot_01_result',{'hole':number},np.zeros((10,10,3),np.uint8))
            s.close('offline_test_only')
            self.assertTrue((s.path/'hole_01_shot_01_result.json').is_file())
            self.assertTrue((s.path/'hole_02_shot_01_result.png').is_file())

    def test_calibration_shot_ids_do_not_collide_between_holes(self):
        with tempfile.TemporaryDirectory() as root:
            s=Session(root)
            s.scope='hole_01_';first=s.shot_id(1)
            s.scope='hole_02_';second=s.shot_id(1)
            s.close('offline_test_only')
            self.assertNotEqual(first,second)

    def test_hole_wrap_and_unknown_hole(self):
        self.assertEqual(next_hole(18),1)
        with self.assertRaises(RuntimeError):next_hole(None)

    def test_changed_badge_background_requires_fresh_matching_semantic_read(self):
        helper=freshness_tests.PuttFreshnessTests();helper.setUpClass()
        fresh=helper.frames[0].copy();x,y,w,h=helper.sign
        fresh[y+int(.12*h),x+int(.73*w)]=(150,150,150)
        game=helper.game(fresh)
        game.putt_reader=Mock(return_value=dict(distance=17.5,unit='ft',elevation_ft=.05))
        context=helper.renew(game)
        self.assertTrue(context['putt_revalidation']['badge_semantically_reread'])
        self.assertEqual(context['observed_at'],110.)
        self.assertEqual(context['stable_frames'],2)

    def test_changed_badge_measurement_cannot_renew_old_distance(self):
        helper=freshness_tests.PuttFreshnessTests();helper.setUpClass()
        fresh=helper.frames[0].copy();x,y,w,h=helper.sign
        fresh[y+int(.12*h),x+int(.73*w)]=(150,150,150)
        for decoded in (None,dict(distance=7.3,unit='ft',elevation_ft=.05)):
            game=helper.game(fresh);game.putt_reader=Mock(return_value=decoded)
            self.assertEqual(helper.renew(game)['observed_at'],100.)

if __name__=='__main__':unittest.main()
