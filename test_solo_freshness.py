"""Solo turn OCR budget and guard regressions; native input is never sent."""
from pathlib import Path
from unittest.mock import Mock,patch
import unittest
import cv2
import execution_engine as live
from context_manager import ContextObserver,GameCoordinator,_solo_static_hud_unchanged
from game_state import turn_permission

ROOT=Path(r'Automation-Agent project root')

class SoloFreshnessTests(unittest.TestCase):
    def test_chip_flag_pixels_require_matching_fresh_yardage_and_unit(self):
        live.setup()
        run=ROOT/'automation_debug/runs/20261122_130610_a294c8f6'
        previous,fresh=[cv2.imread(str(run/f'hole_01_context_{i:05d}.png')) for i in (37,38)]
        self.assertTrue(all(frame is not None for frame in (previous,fresh)))
        state=live.shot_state(previous)
        self.assertEqual((state['mode'],state['distance'],state['unit'],state['elevation_ft']),
                         ('CHIP',10,'yd',15/12))
        self.assertFalse(_solo_static_hud_unchanged(previous,fresh,state,live.reread_elevation))
        self.assertTrue(_solo_static_hud_unchanged(previous,fresh,state,live.reread_elevation,
                                                  live.read_sign_distance))
        for distance in (None,live.Distance(11,'yd'),live.Distance(10,'ft')):
            with self.subTest(distance=distance):
                self.assertFalse(_solo_static_hud_unchanged(previous,fresh,state,live.reread_elevation,
                                                           Mock(return_value=distance)))

    def test_animated_pitch_badge_requires_matching_fresh_elevation(self):
        live.setup()
        run=ROOT/'automation_debug/runs/20261120_061823_6fe89608'
        previous=cv2.imread(str(run/'context_00009.png'))
        fresh=cv2.imread(str(run/'context_00010.png'))
        self.assertIsNotNone(previous)
        self.assertIsNotNone(fresh)
        state=live.shot_state(previous)
        self.assertEqual((state['hole'],state['shot'],state['distance'],state['elevation_ft']),
                         (1,3,14,23/12))
        self.assertFalse(_solo_static_hud_unchanged(previous,fresh,state))
        self.assertTrue(_solo_static_hud_unchanged(previous,fresh,state,live.reread_elevation))

        observer=Mock(solo_confirmed=True,team_hud=None)
        game=GameCoordinator(observer,Mock(return_value=fresh),Mock(),Mock(),
                             elevation_reader=live.reread_elevation)
        with patch('context_manager.time.perf_counter',return_value=105.):
            selected,captured=game.refresh_turn_frame(Mock(),None,{},previous,state,100.)
        self.assertIs(selected,fresh)
        self.assertEqual(captured,105.)

        wrong=dict(state,elevation_ft=24/12)
        self.assertFalse(_solo_static_hud_unchanged(previous,fresh,wrong,live.reread_elevation))
        x,y,w,h=state['sign']
        erased=fresh.copy()
        erased[y+int(.54*h):y+int(.96*h),x+int(.74*w):x+int(.97*w)]=0
        self.assertFalse(_solo_static_hud_unchanged(previous,erased,state,live.reread_elevation))

    def test_static_refresh_rejects_changed_distance_elevation_unit_and_arrow(self):
        frame=cv2.imread(str(ROOT/'automation_debug/runs/20261114_120746_eaa7ed9f/hole_01_ready_wait_5.png'))
        x,y,w,h=548,65,81,62
        state=dict(mode='FULL',sign=[x,y,w,h],wind=dict(high_mph=0))
        self.assertTrue(_solo_static_hud_unchanged(frame,frame.copy(),state))
        regions={
            'distance':(round(.12*w),round(.06*w),round(.74*w),round(.40*w)),
            'elevation':(round(.33*w),round(.49*h),round(.73*w),round(.95*h)),
            'elevation_unit':(round(.74*w),round(.54*h),round(.97*w),round(.96*h)),
            'elevation_arrow':(0,round(.46*h),round(.32*w),h),
        }
        for name,(a,b,c,d) in regions.items():
            with self.subTest(field=name):
                changed=frame.copy()
                changed[y+b:y+d,x+a:x+c]=0
                self.assertFalse(_solo_static_hud_unchanged(frame,changed,state))

    @classmethod
    def setUpClass(cls):
        cls.frame=cv2.imread(str(ROOT/'automation_debug/solo_profile_tee_20261111.png'))

    def observer(self,solo=True,player=True,text=''):
        self.clock=[100.]
        def reader(image,psm):
            self.clock[0]+=.3
            return dict(text=text,confidence=99,words=[])
        observer=ContextObserver(ROOT,reader=reader,solo_confirmed=solo,ready_reader=lambda frame:True)
        return observer,dict(player=player,hole=10,shot=1)

    def test_solo_reads_only_lie_and_watching_banner(self):
        observer,state=self.observer()
        with patch('context_manager.time.perf_counter',side_effect=lambda:self.clock[0]):
            c=observer.observe(self.frame,state,captured_at=100.)
        self.assertEqual(set(c['raw']),{'lie','watch_banner'})
        self.assertAlmostEqual(self.clock[0]-100,1.2)
        self.assertFalse(c['permission']['allowed'])  # Two captured frames required.
        self.clock[0]+=1
        with patch('context_manager.time.perf_counter',side_effect=lambda:self.clock[0]):
            c=observer.observe(self.frame,state,captured_at=self.clock[0])
        self.assertTrue(c['permission']['allowed'],c['permission'])
        self.assertNotIn('hazards',c)
        self.assertNotIn('stats_owner',c)

    def test_stale_solo_capture_still_blocked(self):
        observer,state=self.observer()
        observer.regions={}
        observer.observe(self.frame,state,now=100)
        c=observer.observe(self.frame,state,now=101)
        self.assertFalse(turn_permission(c,106)['allowed'])
        self.assertIn('turn evidence stale or absent',turn_permission(c,106)['reasons'])

    def test_watch_banner_vetoes_solo_even_with_ready_controls(self):
        observer,state=self.observer(text='WATCHING OPPONENT')
        observer.observe(self.frame,state,now=100)
        c=observer.observe(self.frame,state,now=101)
        self.assertFalse(c['permission']['allowed'])
        self.assertFalse(c['solo_evidence']['multiplayer_conflicts_absent'])

    def test_missing_solo_player_cannot_take_fast_path(self):
        observer,state=self.observer(player=False)
        c=observer.observe(self.frame,state,now=100)
        self.assertIn('players_turn',c['raw'])
        self.assertFalse(c['permission']['allowed'])

    def test_no_solo_flag_keeps_complete_context_scan(self):
        observer,state=self.observer(solo=False)
        c=observer.observe(self.frame,state,now=100)
        self.assertIn('current_aim_hazards',c['raw'])
        self.assertFalse(c['permission']['allowed'])

    def test_actual_team_panel_does_not_use_solo_fast_path(self):
        observer,state=self.observer()
        frame=cv2.imread(str(ROOT/'automation_debug/multiplayer_20261111_070215/frame_028.png'))
        c=observer.observe(frame,state,now=100)
        self.assertIn('current_aim_hazards',c['raw'])
        self.assertFalse(c['permission']['allowed'])

    def test_focus_loss_before_turn_check_sends_no_input(self):
        observer,state=self.observer()
        game=GameCoordinator(observer,Mock(),Mock(),Mock())
        safety=Mock();safety.check.side_effect=RuntimeError('WGT lost focus')
        with self.assertRaisesRegex(RuntimeError,'WGT lost focus'):
            game.require_turn(safety,None,{},state)
        game.capture.assert_not_called();game.move_to.assert_not_called()

if __name__=='__main__':unittest.main()
