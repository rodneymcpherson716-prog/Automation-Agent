"""Context/turn input boundaries and learning regressions, all input mocked."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import ast
import json
import unittest
from unittest.mock import Mock,patch
from contextlib import redirect_stdout
import io

import cv2
import numpy as np

import execution_engine as live
from context_manager import ContextObserver,GameCoordinator,ControllerLock
from game_state import GamePolicy
from game_state import turn_permission
from vision_engine import PowerCalibration
from test_game_intelligence import SOLO,MATCH,ALT,PUTT,CONTROL,context


def state(**changes):
    return dict(dict(hole=1,shot=1,mode='FULL',putt=False,club_range=200.,club_unit='yd',
                     club_name='9.0D',distance=100.,unit='yd',elevation_ft=0.,green_speed=10.,
                     wind=dict(mean_mph=0,screen_dx=0,screen_dy=0)),**changes)


class ContextRuntimeTests(unittest.TestCase):
    def test_actual_alternate_shot_watching_opponent_banner_blocks_swings(self):
        root=Path(live.__file__).parent
        frame=cv2.imread(str(root/'wgt_debug/runs/20261111_065917_de00af09/initial_geometry.png'))
        self.assertIsNotNone(frame)
        live.setup()
        with TemporaryDirectory() as folder:
            observer=ContextObserver(folder)
            observer.regions={'watch_banner':observer.regions['watch_banner']}
            c=observer.observe(frame,state(),now=100)
        self.assertEqual(c['turn_state'],'WATCHING_OPPONENT',c)
        self.assertFalse(c['permission']['allowed'])
    def test_actual_mss_tee_lie_is_read_without_granting_our_turn(self):
        root=Path(live.__file__).parent
        frame=cv2.imread(str(root/'wgt_debug/runs/20261111_045104_9c05321e/observation.png'))
        self.assertIsNotNone(frame)
        live.setup()
        with TemporaryDirectory() as folder:
            observer=ContextObserver(folder)
            observer.regions={'lie':observer.regions['lie']}
            c=observer.observe(frame,state(),now=100)
        self.assertEqual(c['lie'],'TEE',c)
        self.assertGreaterEqual(c['confidence']['lie'],.90)
        self.assertFalse(c['permission']['allowed'])

    def test_preserved_swing_release_f9_and_motion_code_matches_nov10_backup(self):
        root=Path(live.__file__).parent
        def definitions(path):
            return {n.name:ast.dump(n,include_attributes=False)
                    for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        current=definitions(root/'execution_engine.py')
        previous=definitions(root/'wgt_one_hole_v48_backup_nov10.py')
        for symbol in ('setup','send','mouse_up','move_to','clicklock_state','set_clicklock',
                       'clicklock_disabled','activate_wgt','Safety','drag_swing','accuracy_click','ShotObserver'):
            self.assertEqual(current[symbol],previous[symbol],symbol)

    def observer(self,folder,text=SOLO):
        observer=ContextObserver(folder,reader=lambda image,psm:dict(text=text,confidence=96,words=[]))
        observer.layout_verified=True  # Synthetic test layout only.
        return observer

    def coordinator(self,contexts,s=None,**kwargs):
        observer=Mock()
        observer.observe.side_effect=[deepcopy(c) for c in contexts]
        observed=state() if s is None else s
        capture=Mock(return_value=np.zeros((200,300,3),np.uint8))
        game=GameCoordinator(observer,capture,Mock(return_value=deepcopy(observed)),Mock(),target_visible=Mock(return_value=True),**kwargs)
        return game,capture

    def test_only_one_live_controller_can_hold_the_os_lock(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'controller.lock'
            first,second=ControllerLock(path),ControllerLock(path)
            try:
                first.acquire()
                with self.assertRaisesRegex(RuntimeError,'already running'):
                    second.acquire()
                first.close()
                second.acquire()
            finally:
                first.close();second.close()

    def test_cli_concession_override_does_not_rewrite_saved_policy(self):
        root=Path(live.__file__).parent
        policy=root/'wgt_calibration/game_policy.json'
        original=policy.read_bytes()
        output=io.StringIO()
        with patch.object(live,'setup'),patch.object(live.pytesseract,'get_tesseract_version',return_value='5'), \
                patch.object(live,'send') as send,patch.object(live,'capture') as capture,redirect_stdout(output):
            self.assertEqual(live.main(['--check','--concede-feet','1.5']),0)
        self.assertIn('threshold: 1.5 FT',output.getvalue())
        self.assertEqual(policy.read_bytes(),original)
        send.assert_not_called();capture.assert_not_called()

    def test_uncalibrated_live_start_stops_before_activation_or_game_input(self):
        root=Path(live.__file__).parent
        with TemporaryDirectory() as folder,patch.object(live,'ROOT',Path(folder)),patch.object(live,'setup'), \
                patch.object(live,'RECORD',root/'wgt_calibration/last_manual_stroke.json'), \
                patch.object(live,'CALIBRATION',root/'automation_accuracy.json'), \
                patch.object(live.pytesseract,'get_tesseract_version',return_value='5'), \
                patch.object(live,'find_wgt') as find,patch.object(live,'activate_wgt') as activate, \
                patch.object(live,'send') as send,patch.object(live,'capture') as capture,redirect_stdout(io.StringIO()):
            self.assertEqual(live.main(['--controlled-test','--activate','--countdown','0','--concede-feet','2']),1)
            report=json.loads((Path(folder)/'wgt_debug/latest_run.json').read_text())
            self.assertIn('uncalibrated',report['reason'])
        find.assert_not_called();activate.assert_not_called();send.assert_not_called();capture.assert_not_called()

    def test_observer_requires_two_captures_and_resets_on_shot_change(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder)
            frame=np.zeros((72,128,3),np.uint8)
            a=observer.observe(frame,state(),now=100)
            b=observer.observe(frame,state(),now=101)
            c=observer.observe(frame,state(shot=2),now=102)
        self.assertFalse(a['permission']['allowed'])
        self.assertTrue(b['permission']['allowed'])
        self.assertFalse(c['permission']['allowed'])

    def test_old_capture_is_not_restamped_after_slow_ocr(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder)
            frame=np.zeros((72,128,3),np.uint8)
            observer.observe(frame,state(),now=100,captured_at=99)
            c=observer.observe(frame,state(),now=110,captured_at=100)
        self.assertEqual(c['observed_at'],100)
        self.assertFalse(c['permission']['allowed'])

    def test_live_profile_name_frame_has_no_turn_permission(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder,'TIGERWOULD52\n1ST SHOT\nSWING')
            frame=np.zeros((72,128,3),np.uint8)
            observer.observe(frame,state(),now=100)
            c=observer.observe(frame,state(),now=101)
        self.assertFalse(c['permission']['allowed'])

    def test_uncalibrated_layout_never_grants_a_turn_despite_labelled_text(self):
        with TemporaryDirectory() as folder:
            observer=ContextObserver(folder,reader=lambda image,psm:dict(text=SOLO,confidence=99,words=[]))
            frame=np.zeros((72,128,3),np.uint8)
            observer.observe(frame,state(),now=100)
            c=observer.observe(frame,state(),now=101)
        self.assertFalse(c['permission']['allowed'])
        self.assertIn('WGT format/turn layout not calibrated',c['permission']['reasons'])

    def test_chat_or_wrong_region_cannot_supply_active_turn_or_opponent_distance(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder)
            # Even perfectly legible text in the roster area is not an active-
            # player indicator or an opponent-putt control/dialog.
            observer.regions={'players_turn':(0,0,1,1)}
            frame=np.zeros((72,128,3),np.uint8)
            observer.observe(frame,state(),now=100)
            c=observer.observe(frame,state(),now=101)
        self.assertNotIn('active_player',c)
        self.assertNotIn('ball_owner',c)
        self.assertFalse(c['permission']['allowed'])

    def test_timer_ocr_cannot_restart_an_existing_turn_countdown(self):
        text=[MATCH.replace('60','30')]
        with TemporaryDirectory() as folder:
            observer=ContextObserver(folder,reader=lambda image,psm:dict(text=text[0],confidence=96,words=[]))
            observer.layout_verified=True
            a=np.zeros((72,128,3),np.uint8)
            observer.observe(a,state(),now=100)
            text[0]=MATCH
            c=observer.observe(a+1,state(),now=101)
        self.assertNotIn('seconds_left',c)
        self.assertFalse(c['permission']['allowed'])

    def test_wrong_active_player_blocks_input_boundary(self):
        c=context(MATCH.replace('ACTIVE PLAYER: Tigerwould52','ACTIVE PLAYER: Opponent1'))
        game,_=self.coordinator([c])
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101):
            with self.assertRaisesRegex(RuntimeError,'No game input'):
                game.require_turn(safety,None,{},state())
        safety.press.assert_not_called()
        game.move_to.assert_not_called()

    def test_changed_shot_identity_blocks_even_with_own_turn_text(self):
        game,_=self.coordinator([context(SOLO)],s=state(shot=2))
        with self.assertRaisesRegex(RuntimeError,'Shot state changed'):
            game.require_turn(Mock(),None,{},state())
        game.move_to.assert_not_called()

    def test_timer_deadline_is_armed_before_stroke_and_never_extended(self):
        game,_=self.coordinator([context(MATCH.replace('60','30'))])
        safety=Mock(deadline=120.)
        reference=Mock(drag_seconds=5.67,accuracy_delay=1.9,anchor=(.708851,.956044))
        s=state()
        from game_state import strategy
        s['strategy']=strategy(s)
        with patch('context_manager.time.perf_counter',return_value=101):
            previous=game.arm_stroke(safety,None,dict(left=0,top=0,width=1288,height=728),s,reference)
        self.assertEqual(previous,120.)
        self.assertEqual(safety.deadline,120.)
        safety.press.assert_not_called()

    def test_insufficient_clock_never_starts_a_stroke(self):
        game,_=self.coordinator([context(MATCH.replace('60','10'))])
        safety=Mock(deadline=900.)
        s=state()
        from game_state import strategy
        s['strategy']=strategy(s)
        with patch('context_manager.time.perf_counter',return_value=101):
            with self.assertRaisesRegex(RuntimeError,'insufficient verified turn time'):
                game.arm_stroke(safety,None,{},s,Mock(drag_seconds=5.67,accuracy_delay=1.9))
        self.assertEqual(safety.deadline,900.)
        safety.press.assert_not_called()

    def test_concession_click_is_once_and_requires_explicit_owner_result(self):
        before=context(PUTT);before['concede_control']=deepcopy(CONTROL)
        after=context(PUTT+'\nPUTT CONCEDED: Opponent1')
        after['concede_control']=None
        game,_=self.coordinator([before,after,after,before])
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101):
            self.assertTrue(game.concede(safety,None,dict(left=20,top=30),state()))
            self.assertFalse(game.concede(safety,None,dict(left=20,top=30),state()))
        safety.press.assert_called_once()
        safety.release.assert_called_once()

    def test_same_putt_distance_change_does_not_unlock_a_second_click(self):
        before=context(PUTT);before['concede_control']=deepcopy(CONTROL)
        after=context(PUTT+'\nPUTT CONCEDED: Opponent1');after['concede_control']=None
        changed=context(PUTT.replace('2.0 FT','1.9 FT'));changed['concede_control']=deepcopy(CONTROL)
        game,_=self.coordinator([before,after,after,changed])
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101):
            self.assertTrue(game.concede(safety,None,dict(left=0,top=0),state()))
            self.assertFalse(game.concede(safety,None,dict(left=0,top=0),state()))
        safety.press.assert_called_once()

    def test_unresolved_concession_remains_locked_across_restart_and_policy_change(self):
        before=context(PUTT);before['concede_control']=deepcopy(CONTROL)
        changed=context(PUTT.replace('2.0 FT','1.9 FT'));changed['concede_control']=deepcopy(CONTROL)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'pending.json'
            game,_=self.coordinator([before],attempt_store=path)
            first=Mock();first.wait.side_effect=KeyboardInterrupt('F9')
            with patch('context_manager.time.perf_counter',return_value=101):
                with self.assertRaises(KeyboardInterrupt):
                    game.concede(first,None,dict(left=0,top=0),state())
                restarted,_=self.coordinator([changed],attempt_store=path,policy=GamePolicy(3.))
                second=Mock()
                self.assertFalse(restarted.concede(second,None,dict(left=0,top=0),state()))
            self.assertEqual(len(json.loads(path.read_text())['pending']),1)
        first.press.assert_called_once();first.release.assert_called_once();second.press.assert_not_called()

    def test_shared_ball_courtesy_is_once_when_opposing_player_or_team_label_changes(self):
        text=ALT.replace('ACTIVE PLAYER: Tigerwould52','ACTIVE PLAYER: Opponent1')
        text=text.replace('SHARED BALL: TEAM A','SHARED BALL: TEAM B')
        text+='\nBALL OWNER: Opponent1\nLIE: GREEN\nOPPONENT PUTT: Opponent1 2.0 FT'
        before=context(text);before['concede_control']=deepcopy(CONTROL)
        after=context(text+'\nPUTT CONCEDED: Opponent1');after['concede_control']=None
        changed=context(text.replace('ACTIVE PLAYER: Opponent1','ACTIVE PLAYER: Opponent2')
                        .replace('BALL OWNER: Opponent1','BALL OWNER: Opponent2')
                        .replace('OPPONENT PUTT: Opponent1 2.0 FT','OPPONENT PUTT: Opponent2 1.9 FT')
                        .replace('TEAM A','TEAM X').replace('TEAM B','TEAM Y'))
        changed['concede_control']=deepcopy(CONTROL)
        game,_=self.coordinator([before,after,after,changed]);safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101):
            self.assertTrue(game.concede(safety,None,dict(left=0,top=0),state()))
            self.assertFalse(game.concede(safety,None,dict(left=0,top=0),state()))
        safety.press.assert_called_once()

    def test_verified_concession_clears_durable_pending_but_keeps_session_lock(self):
        before=context(PUTT);before['concede_control']=deepcopy(CONTROL)
        after=context(PUTT+'\nPUTT CONCEDED: Opponent1');after['concede_control']=None
        with TemporaryDirectory() as folder:
            path=Path(folder)/'pending.json'
            game,_=self.coordinator([before,after,after],attempt_store=path)
            with patch('context_manager.time.perf_counter',return_value=101):
                self.assertTrue(game.concede(Mock(),None,dict(left=0,top=0),state()))
            self.assertEqual(json.loads(path.read_text())['pending'],{})
            self.assertEqual(len(game.pending_concessions),1)

    def test_pending_journal_failure_stops_before_mouse_down(self):
        c=context(PUTT);c['concede_control']=deepcopy(CONTROL)
        game,_=self.coordinator([c]);safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101), \
                patch.object(game,'_save_pending',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                game.concede(safety,None,dict(left=0,top=0),state())
        safety.press.assert_not_called()

    def test_wrong_game_result_never_confirms_a_concession(self):
        before=context(PUTT);before['concede_control']=deepcopy(CONTROL)
        wrong=context(PUTT.replace('MATCH PLAY','STROKE PLAY')+'\nPUTT CONCEDED: Opponent1')
        clock=[101.]
        calls=[0]
        game,_=self.coordinator([])
        def read(*args,**kwargs):
            calls[0]+=1
            c=deepcopy(before if calls[0]==1 else wrong)
            c['observed_at']=clock[0]
            return c
        game.observer.observe.side_effect=read
        safety=Mock()
        def advance(seconds):
            clock[0]+=1
        safety.wait.side_effect=advance
        with patch('context_manager.time.perf_counter',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(RuntimeError,'explicit result unverified'):
                game.concede(safety,None,dict(left=0,top=0),state())
        safety.press.assert_called_once();safety.release.assert_called_once()
        self.assertFalse(game.completed)

    def test_concession_f9_still_releases_and_latches_the_attempt(self):
        c=context(PUTT);c['concede_control']=deepcopy(CONTROL)
        game,_=self.coordinator([c])
        safety=Mock()
        safety.wait.side_effect=KeyboardInterrupt('F9')
        with patch('context_manager.time.perf_counter',return_value=101):
            with self.assertRaises(KeyboardInterrupt):
                game.concede(safety,None,dict(left=20,top=30),state())
        safety.release.assert_called_once()
        self.assertEqual(len(game.pending_concessions),1)

    def test_concession_can_finish_hole_only_after_repeated_explicit_result(self):
        before=context(PUTT);before['concede_control']=deepcopy(CONTROL)
        after=context(PUTT+'\nPUTT CONCEDED: Opponent1\nHOLE COMPLETE')
        after['concede_control']=None
        game,_=self.coordinator([before,after,after])
        with patch('context_manager.time.perf_counter',return_value=101):
            self.assertTrue(game.concede(Mock(),None,dict(left=0,top=0),state()))
        self.assertTrue(game.completed)

    def test_occluded_concession_control_sends_no_mouse_input(self):
        c=context(PUTT);c['concede_control']=deepcopy(CONTROL)
        game,_=self.coordinator([c]);game.target_visible.return_value=False
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=101):
            with self.assertRaisesRegex(RuntimeError,'covered by another window'):
                game.concede(safety,None,dict(left=0,top=0),state())
        safety.press.assert_not_called();game.move_to.assert_not_called()

    def test_control_template_needs_matching_hash_and_exact_label(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            template=np.random.default_rng(10).integers(0,255,(30,140,3),dtype=np.uint8)
            cv2.imwrite(str(root/'control.png'),template)
            spec=dict(template='control.png',sha256=hashlib.sha256((root/'control.png').read_bytes()).hexdigest(),
                      verified_control=True,label='CONCEDE PUTT')
            observer=self.observer(folder)
            observer.control_spec=spec
            observer.reader=lambda image,psm:dict(text='CONCEDE PUTT',confidence=97,words=[])
            frame=np.zeros((200,300,3),np.uint8);frame[50:80,40:180]=template
            self.assertEqual(observer.control(frame)['rect'],[40,50,140,30])
            self.assertIsNone(observer.control((frame*.6).astype(np.uint8)))
            observer.reader=lambda image,psm:dict(text='CONCEDE HOLE',confidence=97,words=[])
            self.assertIsNone(observer.control(frame))
            observer.control_spec=dict(spec,sha256='changed')
            self.assertIsNone(observer.control(frame))

    def test_run_hole_without_turn_coordinator_never_swings(self):
        with patch.object(live,'GAME',None),patch.object(live,'drag_swing') as drag:
            with self.assertRaisesRegex(RuntimeError,'coordinator is required'):
                live.run_hole(Mock(),None,{},Mock())
        drag.assert_not_called()

    def test_turn_loss_before_club_wheel_sends_no_input(self):
        game=Mock()
        game.require_turn.side_effect=RuntimeError('other player turn')
        with patch.object(live,'GAME',game),patch.object(live,'POWER_MODEL',None), \
                patch.object(live,'send') as send,patch.object(live,'move_to') as move:
            with self.assertRaisesRegex(RuntimeError,'other player turn'):
                live.select_club(Mock(),None,{},state(distance=190,club_range=100,club_name='3I'),None)
        send.assert_not_called();move.assert_not_called()


class StrategyLearningTests(unittest.TestCase):
    def test_full_power_still_requires_visible_feedback_and_release(self):
        from automation_core import Meter,StrokeReference
        root=Path(live.__file__).parent
        reference=StrokeReference.from_data(json.loads((root/'wgt_calibration/last_manual_stroke.json').read_text()))
        safety=Mock()
        with patch.object(live,'move_to'),patch.object(live,'capture',return_value=np.zeros((40,1056,3),np.uint8)), \
                patch.object(live,'yellow_power',side_effect=[.2,.7,.998]),patch.object(live,'DIAGNOSTICS',None):
            actual=live.drag_swing(safety,None,dict(left=0,top=0,width=1288,height=728),
                                   reference,Meter(73,580,950,767,1056,535),1.)
        self.assertEqual(actual,.998)
        safety.press.assert_called_once()
        safety.release.assert_called_once()

    def test_first_tee_shot_requests_full_power_on_every_hole(self):
        for hole in (1,10,18):
            s=state(hole=hole,lie='TEE')
            self.assertEqual(PowerCalibration().plan(s)['power'],1.)
            self.assertEqual(PowerCalibration().plan(s,for_selection=True)['power'],.5)
        self.assertEqual(PowerCalibration().plan(state(lie='FAIRWAY'))['power'],.5)
        self.assertEqual(PowerCalibration().plan(state(lie='TEE',shot=2))['power'],.5)

    def test_strategy_layup_preserves_original_target_distance(self):
        s=state(strategy=dict(target=75))
        plan=PowerCalibration().plan(s)
        self.assertEqual(plan['power'],75/200)
        self.assertEqual(s['distance'],100)

    def test_foreign_or_unattributed_stats_never_enter_modern_learning(self):
        s=state(lie='FAIRWAY',game_context=context(SOLO))
        s['game_context']['permission']=dict(allowed=True)
        stats=dict(carry_yd=90,roll_yd=10,total_yd=100,power_pct=50,spin_pct=0,apex_ft=20)
        model=PowerCalibration()
        with TemporaryDirectory() as folder:
            path=Path(folder)/'power.json'
            self.assertFalse(model.record(path,s,stats,.5,'session:1'))
            self.assertFalse(path.exists())
            s['game_context']['verified_stats_owner']='opponent1'
            self.assertFalse(model.record(path,s,stats,.5,'session:1'))
            s['game_context']['verified_stats_owner']='tigerwould52'
            self.assertTrue(model.record(path,s,stats,.5,'session:1'))
            self.assertFalse(model.record(path,s,stats,.5,'session:1'))
        self.assertEqual(len(model.records),1)
        self.assertTrue(model.records[0]['owner_verified'])


if __name__=='__main__':
    unittest.main()
