"""Replay the actual Nov 18 tee-to-second-shot failure; no live input."""
import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
import execution_engine as live
from context_manager import ContextObserver,GameCoordinator
from round_manager import scorecard

ROOT=Path(__file__).parent
RUN=ROOT/'wgt_debug/runs/20261118_034307_26a058c3'
RECT=dict(left=37,top=46,width=1288,height=728)

class LiveSecondShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()
        paths=sorted(RUN.glob('post_shot_read_*.png'))
        cls.frames=[cv2.imread(str(path)) for path in paths[-2:]]
        cls.frame=cv2.imread(str(RUN/'stopped_frame.png'))

    def test_actual_single_intact_label_requires_matching_independent_digit(self):
        self.assertEqual(live.read_stroke_glyphs(self.frame,['24D SHOT','2ND SHOT']),2)

    def test_conflicting_full_labels_stay_unverified(self):
        self.assertIsNone(live.read_stroke_glyphs(self.frame,['2ND SHOT','3RD SHOT']))

    def test_one_label_cannot_authorize_a_blank_digit(self):
        self.assertIsNone(live.read_stroke_glyphs(np.zeros_like(self.frame),['24D SHOT','2ND SHOT']))

    def test_numeric_evidence_must_match_the_intact_label(self):
        self.assertIsNone(live.read_stroke_glyphs(self.frame,['3RD SHOT']))

    def test_invalid_ordinal_is_not_an_intact_label(self):
        self.assertIsNone(live.read_stroke_glyphs(self.frame,['2RD SHOT','2RD SHOT']))

    def test_actual_complete_reader_verifies_both_saved_second_shot_frames(self):
        for frame in self.frames:
            with patch.object(live,'GAME',None),patch.object(live,'send') as send:
                state=live.shot_state(frame)
            self.assertEqual((state['hole'],state['shot'],state['player']),(10,2,True))
            self.assertEqual((state['distance'],state['unit']),(200.,'yd'))
            send.assert_not_called()

    def test_real_transition_with_real_context_observer_reaches_second_shot(self):
        before=json.loads((RUN/'shot_01_plan.json').read_text())['state']
        motion=Mock()
        motion.report.return_value=json.loads((RUN/'shot_01_motion.json').read_text())
        frames=itertools.cycle(self.frames)
        def ready(frame):
            return live.swing_button(frame,RECT) is not None and live.ready_meter(frame,RECT) is not None
        observer=ContextObserver(ROOT,solo_confirmed=True,ready_reader=ready,practice_shot_seconds=90)
        game=GameCoordinator(observer,Mock(return_value=self.frames[-1]),live.shot_state,Mock())
        actual_ocr=live.ocr_variants
        def ocr(image,modes=(6,7),scale=4):
            return [''] if modes==(11,) and scale==2 else actual_ocr(image,modes,scale)
        with patch.object(live,'GAME',game),patch.object(live,'DIAGNOSTICS',None), \
                patch.object(live,'capture',side_effect=lambda *args:next(frames)), \
                patch.object(live,'ocr_variants',side_effect=ocr), \
                patch('round_manager.scorecard',return_value=None),patch.object(live,'shot_stats',return_value=None), \
                patch.object(live,'send') as send,patch.object(live,'drag_swing') as swing:
            outcome,after=live.wait_until_next(Mock(),None,RECT,before,timeout=30,observer=motion)
        self.assertEqual(outcome,'next')
        self.assertEqual((after['hole'],after['shot']),(10,2))
        self.assertTrue(after['game_context']['permission']['allowed'])
        send.assert_not_called()
        swing.assert_not_called()
        game.move_to.assert_not_called()

    def test_swing_handoff_f9_and_previous_putt_fix_are_unchanged(self):
        def definitions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old=definitions(ROOT/'repair_backups/stroke_consensus_20261118/wgt_one_hole_v48.py')
        current=definitions(ROOT/'execution_engine.py')
        for name in old:
            if name not in ('read_stroke_glyphs','ocr_text','find_distance_sign','shot_state','run_hole','main'):
                self.assertEqual(current[name],old[name],name)
        old_round=definitions(ROOT/'repair_backups/stroke_consensus_20261118/round_manager.py')
        current_round=definitions(ROOT/'round_manager.py')
        for name in old_round:
            if name!='scorecard':
                self.assertEqual(current_round[name],old_round[name],name)

class RealCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()
        cls.frame=cv2.imread(str(ROOT/'wgt_debug/runs/20261114_120517_84975c28/hole_01_stopped_frame.png'))

    def test_real_completed_hole_scorecard_and_continue_are_verified(self):
        found=scorecard(self.frame,live.ocr_variants)
        self.assertIsNotNone(found)
        self.assertEqual(found['kind'],'completed_scorecard')
        self.assertEqual(found['button'],[820,509,95,27])

    def test_erased_continue_text_cannot_complete_a_hole(self):
        frame=self.frame.copy()
        frame[512:533,830:905]=np.median(frame[512:533,830:905].reshape(-1,3),axis=0).astype(np.uint8)
        self.assertIsNone(scorecard(frame,live.ocr_variants))

    def test_actual_completion_still_requires_two_frames_and_shot_motion(self):
        actual_ocr=live.ocr_variants
        def ocr(image,modes=(6,7),scale=4):
            return [''] if modes==(11,) else actual_ocr(image,modes,scale)
        for moved in (False,True):
            with self.subTest(motion=moved):
                observer=Mock()
                observer.report.return_value=dict(scene_motion_observed=moved,moving_ball_candidate_tracked=False)
                with patch.object(live,'GAME',None),patch.object(live,'DIAGNOSTICS',None), \
                        patch.object(live,'capture',return_value=self.frame) as capture, \
                        patch.object(live,'ocr_variants',side_effect=ocr),patch.object(live,'send') as send:
                    if moved:
                        outcome,after=live.wait_until_next(Mock(),None,RECT,dict(hole=10,shot=4),timeout=10,observer=observer)
                        self.assertEqual(outcome,'complete')
                        self.assertTrue(after['complete'])
                    else:
                        with self.assertRaisesRegex(RuntimeError,'without observed shot motion'):
                            live.wait_until_next(Mock(),None,RECT,dict(hole=10,shot=4),timeout=10,observer=observer)
                self.assertEqual(capture.call_count,2)
                send.assert_not_called()

if __name__=='__main__':
    unittest.main()
