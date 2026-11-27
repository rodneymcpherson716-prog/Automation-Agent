"""Final putt distance-source consistency; all capture/input is mocked."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np
import execution_engine as live
from context_manager import GameCoordinator
from game_state import PLAYER

ROOT=Path(__file__).parent

def planned():
    return dict(player=True,hole=10,shot=6,mode='PUTT',putt=True,
                distance=14.6,unit='ft',distance_source='floating_target',
                club_name='PUTTER',club_range=60.,club_unit='ft',green_speed=10.,
                elevation_ft=.05,lie='GREEN',wind=dict(mean_mph=0,screen_dx=0,screen_dy=0))

def authorized():
    return dict(observed_at=100.,layout_verified=True,stable_frames=2,
        source='solo_user_confirmed_live_hud',format='SOLO_STROKE',players=[PLAYER],
        active_player=PLAYER,ball_owner=PLAYER,turn_state='OWN_READY',ready_controls=True,
        solo_session_confirmed=True,solo_evidence=dict(player_header_visible=True,
            swing_and_meter_visible=True,multiplayer_conflicts_absent=True))

class SourceBoundaryTests(unittest.TestCase):
    def game(self,readings,context=None):
        observer=SimpleNamespace(solo_confirmed=True,policy=None,turn_deadline=190.)
        frame=np.zeros((2,2,3),np.uint8)
        default=Mock(return_value=dict(planned(),distance=15.,distance_source='verified_hole_hud'))
        source_reader=Mock(side_effect=deepcopy(readings))
        game=GameCoordinator(observer,Mock(return_value=frame),default,Mock(),session=Mock(),
                             read_state_from_source=source_reader)
        game.refresh_turn_frame=Mock(side_effect=lambda safety,sct,rect,f,s,t:(f,t))
        def inspect(safety,sct,rect,f,s,t):
            c=deepcopy(context or authorized())
            s['game_context']=c
            return c
        game.inspect_ready=Mock(side_effect=inspect)
        return game,source_reader,default

    def test_missing_precise_read_is_reacquired_without_using_rounded_hud(self):
        original=planned()
        missing=dict(original,distance=None,unit=None)
        game,reader,default=self.game([missing,original])
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101.):
            result=game.require_turn(safety,None,{},original)
        self.assertEqual(result['active_player'],PLAYER)
        self.assertEqual(original['distance'],14.6)
        self.assertEqual([call.args[1] for call in reader.call_args_list],['floating_target']*2)
        default.assert_not_called()
        self.assertEqual(game.capture.call_count,2)
        self.assertEqual(game.observer.turn_deadline,190.)
        game.inspect_ready.assert_called_once()
        safety.press.assert_not_called()
        game.move_to.assert_not_called()

    def test_changed_precise_value_stops_without_retry(self):
        game,reader,_=self.game([dict(planned(),distance=14.7)])
        with self.assertRaisesRegex(RuntimeError,'before input: distance'):
            game.require_turn(Mock(),None,{},planned())
        self.assertEqual(reader.call_count,1)
        game.inspect_ready.assert_not_called()
        game.move_to.assert_not_called()

    def test_different_source_is_rejected_even_with_the_same_number(self):
        game,reader,_=self.game([dict(planned(),distance_source='verified_hole_hud')])
        with self.assertRaisesRegex(RuntimeError,'distance_source'):
            game.require_turn(Mock(),None,{},planned())
        self.assertEqual(reader.call_count,1)
        game.inspect_ready.assert_not_called()

    def test_unreadable_distance_remains_blocked_after_three_read_only_attempts(self):
        game,reader,_=self.game([dict(planned(),distance=None,unit=None)]*3)
        with self.assertRaisesRegex(RuntimeError,'before input: distance'):
            game.require_turn(Mock(),None,{},planned())
        self.assertEqual(reader.call_count,3)
        game.inspect_ready.assert_not_called()
        game.move_to.assert_not_called()
        self.assertEqual(game.observer.turn_deadline,190.)
        self.assertEqual(game.session.state.call_count,3)

    def test_identity_change_is_never_retried(self):
        game,reader,_=self.game([dict(planned(),shot=7,distance=None,unit=None)])
        with self.assertRaisesRegex(RuntimeError,'before input: shot'):
            game.require_turn(Mock(),None,{},planned())
        self.assertEqual(reader.call_count,1)
        game.inspect_ready.assert_not_called()

    def test_unknown_player_is_not_retried(self):
        game,reader,_=self.game([dict(planned(),player=False,distance=None,unit=None)])
        with self.assertRaises(RuntimeError):
            game.require_turn(Mock(),None,{},planned())
        self.assertEqual(reader.call_count,1)
        game.inspect_ready.assert_not_called()

    def test_f9_during_reacquisition_stops_before_another_capture(self):
        game,reader,_=self.game([dict(planned(),distance=None,unit=None),planned()])
        safety=Mock()
        safety.wait.side_effect=KeyboardInterrupt('F9 emergency stop')
        with self.assertRaises(KeyboardInterrupt):
            game.require_turn(safety,None,{},planned())
        self.assertEqual(reader.call_count,1)
        safety.press.assert_not_called()

    def test_fresh_authorization_is_still_required_after_the_exact_distance_matches(self):
        stale=authorized();stale['observed_at']=90.
        game,_,_=self.game([planned()],context=stale)
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101.):
            with self.assertRaisesRegex(RuntimeError,'No game input'):
                game.require_turn(safety,None,{},planned())
        safety.press.assert_not_called()
        game.move_to.assert_not_called()

    def test_handoff_and_existing_swing_code_are_unchanged_by_this_fix(self):
        def definitions(path):
            return {n.name:ast.dump(n,include_attributes=False)
                    for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old=definitions(ROOT/'repair_backups/putt_distance_source_20261118/wgt_one_hole_v48.py')
        now=definitions(ROOT/'execution_engine.py')
        for name in old:
            if name not in ('shot_state','main','read_stroke_glyphs','ocr_text','run_hole','find_distance_sign'):
                self.assertEqual(now[name],old[name],name)

class RealDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()
        cls.frame=cv2.imread(str(ROOT/'wgt_debug/runs/20261117_220207_51c60201/ready_wait_10.png'))

    def test_real_14_point_6_marker_and_15_hud_are_distinct_sources(self):
        with patch.object(live,'GAME',None),patch.object(live,'send') as send:
            precise=live.shot_state(self.frame,target_source='floating_target')
            rounded=live.shot_state(self.frame,target_source='verified_hole_hud')
        self.assertEqual((precise['distance'],precise['unit'],precise['distance_source']),(14.6,'ft','floating_target'))
        self.assertEqual((rounded['distance'],rounded['unit'],rounded['distance_source']),(15.,'ft','verified_hole_hud'))
        send.assert_not_called()

    def test_hidden_marker_cannot_substitute_15_for_the_planned_14_point_6(self):
        frame=self.frame.copy()
        x,y,w,h=live.find_distance_sign(frame)
        hsv=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
        blue=(hsv[:,:,0]>80)&(hsv[:,:,0]<117)&(hsv[:,:,1]>25)
        fill=np.median(frame[y:y+h,x:x+w][blue],axis=0).astype(np.uint8)
        frame[y+int(.04*w):y+int(.44*w),x+int(.10*w):x+int(.74*w)]=fill
        with patch.object(live,'GAME',None),patch.object(live,'send') as send:
            fallback=live.shot_state(frame)
            pinned=live.shot_state(frame,target_source='floating_target')
        self.assertEqual((fallback['distance'],fallback['distance_source']),(15.,'verified_hole_hud'))
        self.assertIsNone(pinned['distance'])
        self.assertIsNone(pinned['unit'])
        send.assert_not_called()

if __name__=='__main__':
    unittest.main()
