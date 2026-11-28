"""Actual MSS solo/team regressions; every native input is mocked."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import cv2
import numpy as np
import execution_engine as live
from context_manager import ContextObserver,GameCoordinator,solo_header_layout

ROOT=Path(r'Automation-Agent project root')
FIXTURE=Path(__file__).with_name('solo_tee.png')
if not FIXTURE.is_file():
    FIXTURE=ROOT/'automation_debug/solo_profile_tee_20261111.png'

class SoloProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()
        cls.frame=cv2.imread(str(FIXTURE))
        def ready(frame):
            h,w=frame.shape[:2]
            rect=dict(left=0,top=0,width=w,height=h)
            return live.ready_meter(frame,rect) is not None and live.swing_button(frame,rect) is not None
        cls.ready=staticmethod(ready)

    def observer(self,solo=True):
        observer=ContextObserver(ROOT,ready_reader=self.ready,solo_confirmed=solo)
        observer.regions={}
        return observer

    def test_actual_solo_tee_with_old_team_profile_reads_complete_hud(self):
        observer=self.observer()
        with patch.object(live,'GAME',SimpleNamespace(observer=observer)):
            state=live.shot_state(self.frame)
        self.assertTrue(state['player'])
        self.assertEqual((state['shot'],state['hole'],state['par']),(1,10,4))
        self.assertEqual((state['distance'],state['hole_distance'],state['club_range']),(282,378,205))
        self.assertEqual(state['club_name'],'3W')
        self.assertEqual(state['elevation_ft'],-3)
        self.assertEqual(live.readiness_problems(state,self.frame,dict(left=0,top=0,width=1288,height=728)),[])
        observer.observe(self.frame,state,now=100)
        context=observer.observe(self.frame,state,now=101)
        self.assertTrue(context['permission']['allowed'],context['permission'])
        self.assertFalse(context['layout_verified'])

    def test_without_explicit_solo_selection_stays_blocked(self):
        observer=self.observer(False)
        observer.regions={}
        c=observer.observe(self.frame,dict(player=True,hole=10,shot=1),now=100)
        self.assertFalse(c['permission']['allowed'])
        self.assertIsNotNone(observer.read_hud(self.frame))

    def test_single_frame_and_missing_controls_never_permit_input(self):
        observer=self.observer();observer.regions={}
        s=dict(player=True,hole=10,shot=1)
        self.assertFalse(observer.observe(self.frame,s,now=100)['permission']['allowed'])
        observer.solo_ready_reader=lambda frame:False
        self.assertFalse(observer.observe(self.frame,s,now=101)['permission']['allowed'])

    def test_actual_team_frames_remain_team_even_with_solo_flag(self):
        observer=self.observer();observer.regions={}
        for n in (0,14,28,57,71,79):
            with self.subTest(frame=n):
                frame=cv2.imread(str(ROOT/f'automation_debug/multiplayer_20261111_070215/frame_{n:03d}.png'))
                self.assertFalse(solo_header_layout(frame))
                self.assertIsNotNone(observer.read_hud(frame))
                c=observer.observe(frame,dict(player=True,hole=2,shot=1),now=100+n)
                self.assertFalse(c['solo_evidence']['multiplayer_conflicts_absent'])

    def test_refresh_solo_pixels_without_reviving_team_profile(self):
        observer=self.observer()
        frame=self.frame
        game=GameCoordinator(observer,Mock(return_value=frame.copy()),Mock(),Mock())
        state=dict(sign=[720,82,61,46])
        with patch('context_manager.time.perf_counter',return_value=110):
            _,stamp=game.refresh_turn_frame(Mock(),None,{},frame,state,100)
        self.assertEqual(stamp,110)
        fresh=frame.copy();fresh[60,1080]=0
        game.capture.return_value=fresh
        with patch('context_manager.time.perf_counter',return_value=110):
            _,stamp=game.refresh_turn_frame(Mock(),None,{},frame,state,100)
        self.assertEqual(stamp,100)

    def test_f9_or_focus_guard_prevents_any_revalidation(self):
        observer=self.observer()
        capture=Mock()
        game=GameCoordinator(observer,capture,Mock(),Mock())
        safety=Mock();safety.check.side_effect=KeyboardInterrupt('F9 emergency stop')
        with self.assertRaises(KeyboardInterrupt):
            game.refresh_turn_frame(safety,None,{},self.frame,dict(sign=[720,82,61,46]),0)
        capture.assert_not_called()

if __name__=='__main__':unittest.main()
