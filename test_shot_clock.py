"""Shot-clock invariants; no desktop or game input."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from context_manager import ContextObserver,GameCoordinator
from game_state import PLAYER,combine_context,swing_budget
from test_game_intelligence import ALT

SOLO='SOLO STROKE PLAY\nPLAYER: Tigerwould52\nACTIVE PLAYER: Tigerwould52\nBALL OWNER: Tigerwould52'
MATCH='MATCH PLAY\nPLAYERS: Tigerwould52, Opponent1\nACTIVE PLAYER: Tigerwould52\nBALL OWNER: Tigerwould52\nTIME LEFT: 60'

def context(text=SOLO,now=100.):
    c=combine_context([dict(text=text,confidence=99)]*2,now)
    return dict(c,stable_frames=2,layout_verified=True)

class ShotClockTests(unittest.TestCase):
    def test_ninety_second_practice_counts_ocr_and_stops_when_expired(self):
        with TemporaryDirectory() as folder:
            observer=ContextObserver(folder,solo_confirmed=True,ready_reader=lambda frame:True,
                                     practice_shot_seconds=90)
            observer.regions={}
            frame=np.zeros((72,128,3),np.uint8)
            state=dict(player=True,hole=10,shot=4,captured_at=100.)
            observer.observe(frame,state,now=105.)
            c=observer.observe(frame,state,now=106.)
            self.assertEqual(c['timer_deadline'],190.)
            self.assertEqual(c['turn_limit'],90)
            game=GameCoordinator(observer,Mock(),Mock(),Mock())
            with patch.object(game,'inspect',return_value=c),patch.object(game,'revalidate_putt',return_value=c),patch('context_manager.time.perf_counter',return_value=191.):
                with self.assertRaisesRegex(RuntimeError,'90-second practice shot clock expired'):
                    game.inspect_ready(Mock(),None,{},frame,state,100.)
            game.move_to.assert_not_called()

    def practice_observer(self,folder):
        observer=ContextObserver(folder,solo_confirmed=True,ready_reader=lambda frame:True,
                                 practice_shot_seconds=60)
        observer.regions={}
        return observer

    def test_practice_clock_counts_original_capture_before_ocr(self):
        with TemporaryDirectory() as folder:
            observer=self.practice_observer(folder)
            frame=np.zeros((72,128,3),np.uint8)
            state=dict(player=True,hole=10,shot=4,captured_at=90.)
            first=observer.observe(frame,state,now=100.)
            second=observer.observe(frame,state,now=101.)
        self.assertEqual(first['timer_deadline'],150.)
        self.assertTrue(second['practice_clock'])
        self.assertNotIn('seconds_left',second)
        self.assertNotIn('turn_limit',second['confidence'])
        self.assertTrue(swing_budget(second,102.,stroke_seconds=15)['allowed'])
        late=swing_budget(dict(second,observed_at=140.),140.,stroke_seconds=15)
        self.assertFalse(late['allowed'])
        self.assertIn('insufficient verified turn time',late['reasons'][0])

    def test_practice_clock_survives_missing_hud_and_resets_only_on_new_shot(self):
        with TemporaryDirectory() as folder:
            observer=self.practice_observer(folder)
            frame=np.zeros((72,128,3),np.uint8)
            first=observer.observe(frame,dict(player=True,hole=10,shot=4),now=100.)
            observer.observe(frame,dict(player=False,hole=None,shot=None),now=105.)
            recovered=observer.observe(frame,dict(player=True,hole=10,shot=4),now=110.)
            new=observer.observe(frame,dict(player=True,hole=10,shot=5),now=120.)
        self.assertEqual(first['timer_deadline'],160.)
        self.assertEqual(recovered['timer_deadline'],160.)
        self.assertEqual(new['timer_deadline'],180.)

    def test_practice_clock_requires_solo_and_cannot_supply_a_multiplayer_timer(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                ContextObserver(folder,practice_shot_seconds=60)
            observer=self.practice_observer(folder)
            observer.regions={'match_header':(.1,.1,.9,.9),'players_turn':(.1,.1,.9,.9),
                              'turn_header':(.1,.1,.9,.9),'watch_banner':(.1,.1,.9,.9)}
            observer.reader=lambda image,psm:dict(text=MATCH.replace('\nTIME LEFT: 60',''),confidence=99,words=[])
            c=observer.observe(np.zeros((72,128,3),np.uint8),dict(player=True,hole=10,shot=4),now=100.)
        self.assertNotIn('practice_clock',c)
        self.assertNotIn('timer_deadline',c)
        self.assertFalse(c['permission']['allowed'])

    def test_alternate_shot_uses_remaining_time_not_a_fresh_sixty_seconds(self):
        for seconds in (60,45,20):
            c=context(ALT.replace('TIME LEFT: 60','TIME LEFT: '+str(seconds)),100)
            result=swing_budget(c,103,stroke_seconds=15,margin=3)
            self.assertEqual(result['deadline'],100+seconds-3)
            self.assertEqual(result['remaining'],seconds-6)
            self.assertEqual(result['allowed'],seconds>=21)

    def test_alternate_shot_rejects_old_ninety_second_clock_and_missing_timer(self):
        for seconds in (61,83,90,True,float('nan')):
            c=context(ALT);c['seconds_left']=seconds
            self.assertFalse(swing_budget(c,101)['allowed'])
        self.assertFalse(swing_budget(context(ALT.replace('\nTIME LEFT: 60','')),101)['allowed'])

    def test_alternate_shot_config_and_deadline_survive_rereads(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder)
            self.text=ALT
            first=self.read(observer,100)
            self.assertEqual(first['turn_limit'],60)
            self.assertIn('user-confirmed',first['turn_limit_source'])
            self.assertNotIn('turn_limit',first['confidence'])
            self.assertEqual(first['timer_deadline'],160)
            self.text=ALT.replace('TIME LEFT: 60','TIME LEFT: 90')
            conflict=self.read(observer,105)
            self.assertEqual(conflict['timer_deadline'],160)
            self.assertFalse(conflict['permission']['allowed'])
            self.text=ALT.replace('TIME LEFT: 60','TIME LEFT: 50')
            self.assertEqual(self.read(observer,110)['timer_deadline'],160)
            self.text=ALT
            new=self.read(observer,120,shot=2)
            self.assertEqual(new['timer_deadline'],180)

    def test_known_sixty_second_clock_without_remaining_time_is_blocked(self):
        c=context();c['turn_limit']=60
        result=swing_budget(c,101,stroke_seconds=15)
        self.assertFalse(result['allowed'])
        self.assertIsNone(result['deadline'])

    def test_explicitly_untimed_existing_path_is_preserved(self):
        self.assertTrue(swing_budget(context(),101)['allowed'])

    def test_tracked_deadline_survives_temporary_missing_timer_read(self):
        c=context(now=120);c.update(turn_limit=60,timer_deadline=160)
        result=swing_budget(c,121,stroke_seconds=15)
        self.assertTrue(result['allowed'])
        self.assertEqual(result['deadline'],157)
        self.assertEqual(result['remaining'],36)

    def test_clock_budget_includes_ocr_time_and_full_stroke_reserve(self):
        c=context(MATCH,100)
        result=swing_budget(c,103,stroke_seconds=15,margin=3)
        self.assertEqual(result['remaining'],54)
        self.assertTrue(result['allowed'])
        c=context(MATCH.replace('60','16'),144);c['timer_deadline']=160
        self.assertFalse(swing_budget(c,146,stroke_seconds=15,margin=3)['allowed'])

    def observer(self,folder):
        self.text=MATCH+'\nLIE: FAIRWAY'
        observer=ContextObserver(folder,reader=lambda image,psm:dict(text=self.text,confidence=99,words=[]))
        observer.layout_verified=True
        observer.regions={key:(.1,.1,.9,.9) for key in ('match_header','players_turn','turn_header','lie','watch_banner')}
        return observer

    def read(self,observer,now,shot=1):
        observer.ocr_cache.clear()
        return observer.observe(np.zeros((72,128,3),np.uint8),dict(hole=10,shot=shot),now=now)

    def test_lie_or_readiness_change_cannot_restart_same_turn_clock(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder)
            self.assertEqual(self.read(observer,100)['timer_deadline'],160)
            self.text=MATCH+'\nLIE: GREEN\nWATCHING OPPONENT'
            c=self.read(observer,110)
        self.assertEqual(c['timer_deadline'],160)
        self.assertNotIn('seconds_left',c)  # Inflated repeat of 60 is rejected.

    def test_missing_identity_then_recovery_keeps_original_deadline(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder);self.read(observer,100)
            self.text='';self.read(observer,105,shot=None)
            self.text=MATCH+'\nLIE: FAIRWAY'
            c=self.read(observer,110)
        self.assertEqual(c['timer_deadline'],160)
        self.assertNotIn('seconds_left',c)

    def test_verified_new_stroke_can_start_its_own_clock(self):
        with TemporaryDirectory() as folder:
            observer=self.observer(folder);self.read(observer,100)
            c=self.read(observer,110,shot=2)
        self.assertEqual(c['timer_deadline'],170)
        self.assertEqual(c['seconds_left'],60)

    def test_fresh_hud_updates_do_not_refresh_remaining_sixty_seconds(self):
        observer=SimpleNamespace(solo_confirmed=True,last_time=100.,turn_deadline=None,
            read_hud=lambda frame:None,solo_ready_reader=lambda frame:True,sequence=1)
        image=np.zeros((16,16,3),np.uint8)
        game=GameCoordinator(observer,Mock(return_value=image),Mock(),Mock())
        state=dict(mode='PUTT',player=True,sign=[0,0,10,10],distance=1.,unit='ft',club_range=30.,
            club_unit='ft',elevation_ft=0.,green_speed=10.,wind=dict(high_mph=0,screen_dx=0,screen_dy=0))
        c=dict(observed_at=100.,seconds_left=60,turn_limit=60,source='solo_user_confirmed_live_hud',
            format='SOLO_STROKE',players=[PLAYER],active_player=PLAYER,ball_owner=PLAYER,
            ready_controls=True,turn_state='OWN_READY',solo_session_confirmed=True,stable_frames=2,
            solo_evidence=dict(player_header_visible=True,swing_and_meter_visible=True,multiplayer_conflicts_absent=True))
        with patch('context_manager.solo_header_layout',return_value=True),patch('context_manager.putt_measurement_masks',return_value={'measured_text':np.ones(2)}),patch('context_manager.triangle_vector',return_value=(0.,0.)):
            for now in (110.,120.):
                with patch('context_manager.time.perf_counter',return_value=now):
                    c=game.revalidate_putt(Mock(),None,{},image,state,c)
                self.assertEqual(c['observed_at'],now)
                self.assertEqual(c['timer_deadline'],160)
                self.assertEqual(observer.turn_deadline,160)
        self.assertEqual(swing_budget(c,121,stroke_seconds=15)['deadline'],157)
        game.move_to.assert_not_called()

if __name__=='__main__':unittest.main()
