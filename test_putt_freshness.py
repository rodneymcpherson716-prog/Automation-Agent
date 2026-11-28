"""Fresh MSS putt glyph proof; all input is mocked and never sent."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from copy import deepcopy
import unittest
import json
import time
import cv2,numpy as np
import execution_engine as live
from context_manager import GameCoordinator,putt_measurement_masks
from vision_engine import triangle_vector,quick_putt_badge,_visible_decimal_text
from game_state import turn_permission,PLAYER

ROOT=Path(r'Automation-Agent project root')
LOCAL=Path(__file__).parent
RECT=dict(left=37,top=46,width=1288,height=728)

class PuttFreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()
        paths=[LOCAL/f'probe_{i}.png' for i in (0,1)]
        if not all(p.is_file() for p in paths):
            paths=[ROOT/f'automation_debug/putt_freshness_{i}_20261112.png' for i in (0,1)]
        cls.frames=[cv2.imread(str(p)) for p in paths]
        cls.sign=live.find_distance_sign(cls.frames[0])
        f=cls.frames[0];h,w=f.shape[:2]
        vector=triangle_vector(f[int(.064*h):int(.097*h),int(.061*w):int(.078*w)])
        cls.state=dict(mode='PUTT',putt=True,player=True,hole=10,shot=4,distance=17.5,unit='ft',
            club_range=60.,club_unit='ft',club_name='PUTTER',elevation_ft=.05,green_speed=10.,sign=cls.sign,
            wind=dict(low_mph=11,high_mph=13,mean_mph=12.,screen_dx=vector[0],screen_dy=vector[1]))

    def context(self,stable=2):
        return dict(observed_at=100.,source='solo_user_confirmed_live_hud',format='SOLO_STROKE',
            players=[PLAYER],active_player=PLAYER,ball_owner=PLAYER,turn_state='OWN_READY',
            ready_controls=True,solo_session_confirmed=True,stable_frames=stable,
            solo_evidence=dict(player_header_visible=True,swing_and_meter_visible=True,
                               multiplayer_conflicts_absent=True))

    def game(self,fresh=None,ready=True):
        self.observer=SimpleNamespace(solo_confirmed=True,read_hud=lambda frame:None,
            solo_ready_reader=Mock(return_value=ready),last_time=100.,sequence=1)
        self.capture=Mock(return_value=self.frames[1] if fresh is None else fresh)
        return GameCoordinator(self.observer,self.capture,Mock(),Mock())

    def renew(self,game,state=None,context=None,safety=None,attempts=1):
        with patch('context_manager.time.perf_counter',return_value=110.):
            return game.revalidate_putt(safety or Mock(),None,RECT,self.frames[0],
                                       deepcopy(self.state) if state is None else state,
                                       self.context() if context is None else context,attempts=attempts)

    def test_actual_animated_frames_match_each_measured_label(self):
        self.assertFalse(np.array_equal(*self.frames))
        a,b=[putt_measurement_masks(f,self.sign) for f in self.frames]
        self.assertIsNotNone(a);self.assertIsNotNone(b)
        self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a))

    def test_matching_live_hud_renews_age_without_forging_stability(self):
        game=self.game();c=self.renew(game)
        self.assertEqual(c['observed_at'],110)
        self.assertEqual(c['ocr_frame_at'],100)
        self.assertEqual(c['stable_frames'],2)
        self.assertTrue(c['permission']['allowed'],c['permission'])
        self.assertFalse(turn_permission(c,115)['allowed'])
        game.move_to.assert_not_called()

    def test_one_reading_stays_blocked_after_revalidation(self):
        game=self.game();c=self.renew(game,context=self.context(stable=1))
        self.assertEqual(c['stable_frames'],1)
        self.assertFalse(c['permission']['allowed'])

    def test_final_turn_check_recaptures_the_actual_animated_flag(self):
        run=LOCAL/'automation_debug/runs/20261121_125508_3c47a9e1'
        old,covered,fresh=[cv2.imread(str(run/f'hole_01_context_{i:05d}.png')) for i in (1,2,3)]
        self.assertTrue(all(frame is not None for frame in (old,covered,fresh)))
        sign=live.find_distance_sign(old)
        # The real flag frame is now readable with the alternate grayscale
        # layouts. Mask the numeral row to exercise a genuinely unreadable frame.
        x,y,width,height=sign
        covered=covered.copy()
        covered[y+int(.08*width):y+int(.39*width),
                x+int(.22*width):x+int(.73*width)]=0
        h,w=old.shape[:2]
        dx,dy=triangle_vector(old[int(.064*h):int(.097*h),int(.061*w):int(.078*w)])
        state=dict(self.state,sign=sign,distance=31.5,elevation_ft=4.9/12,
                   wind=dict(low_mph=14,high_mph=16,mean_mph=15,screen_dx=dx,screen_dy=dy))
        game=self.game()
        self.capture.side_effect=[old,covered,fresh]
        game.read_state.return_value=deepcopy(state)
        game.refresh_turn_frame=Mock(return_value=(old,100.))
        game.inspect_ready=Mock(return_value=self.context())
        game.putt_reader=Mock(side_effect=lambda frame,rect:
                             quick_putt_badge(frame,rect,live.ocr_text))
        safety=Mock()
        with patch('context_manager.time.perf_counter',return_value=110.):
            context=game.require_turn(safety,None,RECT,state)
        self.assertTrue(context['permission']['allowed'])
        self.assertEqual(context['putt_revalidation']['unreadable_badge_retries'],1)
        self.assertEqual(self.capture.call_count,3)
        self.assertEqual(game.putt_reader.call_count,2)
        safety.wait.assert_called_once_with(.25)
        game.move_to.assert_not_called()

    def test_recapture_stops_on_a_readable_disagreement_or_after_three_unreadable_frames(self):
        x,y,w,h=self.sign
        changed=self.frames[0].copy()
        patch=changed[y+int(.50*h):y+int(.94*h),x+int(.33*w):x+int(.73*w)]
        core=cv2.inRange(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV),(0,0,175),(179,35,255))
        row,col=np.argwhere(core!=0)[0]
        patch[row,col]=0
        for decoded,expected_captures in ((dict(distance=17.5,unit='ft',elevation_ft=.06),1),(None,3)):
            with self.subTest(decoded=decoded):
                game=self.game(changed)
                game.putt_reader=Mock(return_value=decoded)
                context=self.renew(game,attempts=3)
                self.assertEqual(context['observed_at'],100.)
                self.assertEqual(self.capture.call_count,expected_captures)
                self.assertEqual(bool(context.get('putt_measurements_changed')),decoded is not None)
                game.move_to.assert_not_called()

    def test_changed_or_hidden_measured_fields_cannot_be_renewed(self):
        a=self.frames[0];h,w=a.shape[:2];x,y,sw,sh=self.sign
        areas={
            'target':(x+int(.12*sw),y+int(.06*sw),x+int(.74*sw),y+int(.40*sw)),
            'target unit':(x+int(.72*sw),y+int(.12*sw),x+int(.98*sw),y+int(.40*sw)),
            'elevation unit':(x+int(.73*sw),y+int(.54*sh),x+int(.94*sw),y+int(.96*sh)),
            'range':(int(.122*w),int(.938*h),int(.172*w),int(.966*h)),
            'player':(int(.830*w),int(.060*h),int(.900*w),int(.083*h)),
            'stroke':(int(.898*w),int(.082*h),int(.934*w),int(.105*h)),
            'green speed':(int(.094*w),int(.145*h),int(.119*w),int(.166*h)),
        }
        for field,(x1,y1,x2,y2) in areas.items():
            with self.subTest(field=field):
                fresh=a.copy();fresh[y1:y2,x1:x2]=0
                c=self.renew(self.game(fresh))
                self.assertEqual(c['observed_at'],100)
                self.assertNotIn('putt_revalidation',c)

    def test_decimal_point_removal_is_detected(self):
        fresh=self.frames[0].copy();x,y,sw,sh=self.sign
        x1,y1=x+int(.12*sw),y+int(.06*sw)
        patch=fresh[y1:y+int(.40*sw),x1:x+int(.74*sw)]
        mask=cv2.inRange(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV),(0,0,175),(179,35,255))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        small=[s for s in stats[1:] if s[3]<=6 and s[4]>=2 and s[1]>patch.shape[0]*.45]
        self.assertTrue(small,'Visible decimal point must be represented by the mask')
        for sx,sy,bw,bh,count in small:fresh[y1+sy:y1+sy+bh,x1+sx:x1+sx+bw]=0
        self.assertEqual(self.renew(self.game(fresh))['observed_at'],100)

    def test_elevation_arrow_change_is_detected(self):
        fresh=self.frames[0].copy();x,y,sw,sh=self.sign
        x1,x2=x+int(.06*sw),x+int(.30*sw);y1,y2=y+int(.50*sh),y+int(.92*sh)
        fresh[y1:y2,x1:x2]=fresh[y1:y2,x1:x2][::-1]
        self.assertEqual(self.renew(self.game(fresh))['observed_at'],100)

    def test_missing_controls_or_foreign_team_hud_prevents_renewal(self):
        self.assertEqual(self.renew(self.game(ready=False))['observed_at'],100)
        game=self.game();self.observer.read_hud=lambda frame:dict(valid=True)
        self.assertEqual(self.renew(game)['observed_at'],100)

    def test_new_wind_direction_or_expired_fresh_capture_stays_blocked(self):
        game=self.game()
        with patch('context_manager.triangle_vector',return_value=(-1.,0.)):
            self.assertEqual(self.renew(game)['observed_at'],100)
        clock=[110.];game=self.game()
        self.observer.solo_ready_reader.side_effect=lambda frame:clock.__setitem__(0,115.) or True
        with patch('context_manager.time.perf_counter',side_effect=lambda:clock[0]):
            c=game.revalidate_putt(Mock(),None,RECT,self.frames[0],deepcopy(self.state),self.context())
        self.assertEqual(c['observed_at'],100)

    def test_no_solo_or_non_putt_never_uses_this_path(self):
        game=self.game();self.observer.solo_confirmed=False
        self.renew(game);self.capture.assert_not_called()
        game=self.game();s=dict(self.state,mode='FULL')
        self.renew(game,state=s);self.capture.assert_not_called()

    def test_f9_and_focus_abort_before_capture_or_input(self):
        for error in (KeyboardInterrupt('F9'),RuntimeError('WGT lost focus')):
            with self.subTest(error=str(error)):
                game=self.game();safety=Mock();safety.check.side_effect=error
                with self.assertRaises(type(error)):self.renew(game,safety=safety)
                self.capture.assert_not_called();game.move_to.assert_not_called()

    def test_slow_ocr_still_requires_two_consecutive_fresh_complete_putt_reads(self):
        clock=[0.];calls=[];game=Mock();game.observer.solo_confirmed=True
        state=deepcopy(self.state)
        def capture(*args):clock[0]+=12;calls.append(clock[0]);return self.frames[0]
        def inspect(*args):return dict(observed_at=clock[0],permission=dict(allowed=True))
        game.inspect_ready.side_effect=inspect
        safety=Mock();safety.wait.side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
        with patch.object(live,'GAME',game),patch.object(live,'capture',side_effect=capture),patch.object(live,'shot_state',side_effect=lambda frame:deepcopy(state)),patch.object(live,'readiness_problems',return_value=[]),patch.object(live,'diagnostic_state'),patch.object(live.time,'perf_counter',side_effect=lambda:clock[0]):
            result,_=live.stable_state(safety,None,RECT,timeout=30)
        self.assertEqual(len(calls),2)
        self.assertEqual(result['distance'],17.5)

    def test_stable_putt_reacquisition_rejects_expired_proof(self):
        clock=[0.];game=Mock();game.observer.solo_confirmed=True
        def capture(*args):clock[0]+=12;return self.frames[0]
        game.inspect_ready.side_effect=lambda *args:dict(observed_at=clock[0]-5,permission=dict(allowed=True))
        safety=Mock();safety.wait.side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
        with patch.object(live,'GAME',game),patch.object(live,'capture',side_effect=capture),patch.object(live,'shot_state',side_effect=lambda frame:deepcopy(self.state)),patch.object(live,'readiness_problems',side_effect=lambda *args:[]),patch.object(live,'diagnostic_state'),patch.object(cv2,'imwrite'),patch.object(live.time,'perf_counter',side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(RuntimeError,'fresh verified putt HUD'):
                live.stable_state(safety,None,RECT,timeout=30)

    def test_current_missing_player_is_not_replaced_with_prior_player_before_input(self):
        observer=SimpleNamespace(solo_confirmed=False,observe=Mock())
        observer.observe.side_effect=lambda frame,state,**kwargs:dict(
            observed_at=110,confidence={},stable_frames=2,format=None,permission=dict(allowed=False))
        game=GameCoordinator(observer,Mock(return_value=self.frames[0]),
            Mock(return_value=dict(self.state,player=False)),Mock())
        game.refresh_turn_frame=Mock(return_value=(self.frames[0],110))
        with patch('context_manager.time.perf_counter',return_value=110):
            with self.assertRaisesRegex(RuntimeError,'No game input'):
                game.require_turn(Mock(),None,RECT,deepcopy(self.state))
        self.assertFalse(observer.observe.call_args.args[1]['player'])
        game.move_to.assert_not_called()

class LiveGreenDecimalTests(unittest.TestCase):
    def test_short_putt_rechecks_decimal_after_one_pixel_badge_border_change(self):
        live.setup()
        run=LOCAL/'automation_debug/runs/20261126_004514_e82c52a1'
        old,fresh=[cv2.imread(str(run/f'hole_01_context_{index:05d}.png')) for index in (5,6)]
        self.assertTrue(all(frame is not None for frame in (old,fresh)))
        sign=live.find_distance_sign(old)
        self.assertNotEqual(sign,live.find_distance_sign(fresh))
        expected=dict(distance=4.3,unit='ft',elevation_ft=-2/12)
        self.assertEqual(live.reread_putt_badge(old,sign),expected)
        self.assertEqual(live.reread_putt_badge(fresh,sign),expected)
        x,y,w,h=sign
        self.assertIsNone(live.reread_putt_badge(fresh,(x+4,y,w,h)))

    def test_birdie_putt_rechecks_both_digits_of_eleven_inches(self):
        live.setup()
        run=LOCAL/'automation_debug/runs/20261126_002605_7d3da041'
        for name in ('hole_01_stopped_frame.png','hole_01_post_shot_read_1795674604319273200.png'):
            with self.subTest(name=name):
                frame=cv2.imread(str(run/name))
                self.assertIsNotNone(frame)
                sign=live.find_distance_sign(frame)
                badge=quick_putt_badge(frame,sign,live.ocr_text)
                self.assertEqual(badge,dict(distance=39.5,unit='ft',elevation_ft=-11/12))
                x,y,w,h=sign
                frame[y+int(.49*h)+31:y+int(.49*h)+33,
                      x+int(.33*w)+32:x+int(.33*w)+34]=255
                changed=quick_putt_badge(frame,sign,live.ocr_text)
                self.assertNotEqual(changed,badge)

    def test_birdie_putt_confirms_fractional_elevation_after_threshold_disagreement(self):
        run=LOCAL/'automation_debug/runs/20261122_124116_eb483400'
        live.setup()
        for name in ('initial_geometry.png','hole_01_context_00005.png'):
            with self.subTest(name=name):
                frame=cv2.imread(str(run/name))
                self.assertIsNotNone(frame)
                reads=[]
                badge=quick_putt_badge(frame,live.find_distance_sign(frame),live.ocr_text,
                                       elevation_text=reads)
                self.assertEqual(badge,dict(distance=16.3,unit='ft',elevation_ft=3.5/12))
                self.assertEqual([read.strip() for read in reads],['3.9','3.5','3.5','3.5','3.5'])

    def test_fringe_putt_confirms_distance_after_threshold_disagreement(self):
        live.setup()
        path=LOCAL/'automation_debug/runs/20261122_130125_497be5c6/initial_geometry.png'
        frame=cv2.imread(str(path))
        self.assertIsNotNone(frame)
        sign=live.find_distance_sign(frame)
        badge=quick_putt_badge(frame,sign,live.ocr_text)
        self.assertEqual(badge,dict(distance=25.5,unit='ft',elevation_ft=-.2/12))

    def test_current_par_putt_keeps_decimal_distance_and_two_digit_elevation(self):
        live.setup()
        paths = ('20261122_051210_d12149fd/stopped_frame.png',
                 '20261122_051851_85c8a17e/initial_geometry.png')
        for path in paths:
            with self.subTest(path=path):
                frame=cv2.imread(str(LOCAL/'automation_debug/runs'/path))
                self.assertIsNotNone(frame)
                badge=quick_putt_badge(frame,live.find_distance_sign(frame),live.ocr_text)
                self.assertEqual(badge,dict(distance=32.8,unit='ft',elevation_ft=1.0))

    @classmethod
    def setUpClass(cls):
        live.setup()
        run=ROOT/'automation_debug/runs/20261118_025714_0df1a399'
        cls.cases=[]
        for path in sorted(run.glob('*post_shot_read*.json')):
            state=json.loads(path.read_text(encoding='utf-8-sig'))
            if state.get('mode')=='PUTT' and state.get('distance')==21.9:
                cls.cases.append((cv2.imread(str(path.with_suffix('.png'))),state))
        assert len(cls.cases)>=2

    def test_actual_flag_frames_keep_visible_elevation_decimal(self):
        for frame,state in self.cases:
            with self.subTest(capture=state['game_context']['observed_at']):
                badge=quick_putt_badge(frame,live.find_distance_sign(frame),live.ocr_text)
                self.assertIsNotNone(badge)
                self.assertEqual((badge['distance'],badge['unit']),(21.9,'ft'))
                self.assertAlmostEqual(badge['elevation_ft'],-2.9/12)

    def test_missing_or_displaced_decimal_is_never_reconstructed(self):
        frame,state=self.cases[-1]
        x,y,w,h=state['sign']
        p=frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)].copy()
        self.assertEqual(_visible_decimal_text(p,'29'),'2.9')
        _,_,stats,_=cv2.connectedComponentsWithStats(cv2.inRange(p,(200,)*3,(255,)*3))
        dots=[s for s in stats[1:] if s[3]<=6 and s[4]>=2]
        self.assertEqual(len(dots),1)
        a,b,c,d,_=map(int,dots[0])
        absent=p.copy();absent[b:b+d,a:a+c]=0
        self.assertIsNone(_visible_decimal_text(absent,'29'))
        displaced=absent.copy();displaced[1:1+d,a:a+c]=p[b:b+d,a:a+c]
        self.assertIsNone(_visible_decimal_text(displaced,'29'))
        self.assertIsNone(_visible_decimal_text(p,'299'))  # Digit count must also match.

    def test_actual_green_transition_reaches_the_putting_handoff(self):
        old,state=self.cases[0]
        fresh,_=self.cases[-1]
        run=ROOT/'automation_debug/runs/20261118_025714_0df1a399'
        before=json.loads((run/'hole_01_shot_03_plan.json').read_text())['state']
        motion=Mock()
        motion.report.return_value=json.loads((run/'hole_01_shot_03_motion.json').read_text())
        deadline=time.perf_counter()+90
        observer=SimpleNamespace(solo_confirmed=True,team_hud=None,read_hud=lambda frame:None,
            solo_ready_reader=lambda frame:live.swing_button(frame,RECT) is not None and live.ready_meter(frame,RECT) is not None,
            last_time=time.perf_counter()-10,turn_deadline=deadline,sequence=1)
        reader=lambda frame,sign:quick_putt_badge(frame,sign,live.ocr_text) if live.find_distance_sign(frame)==tuple(sign) else None
        game=GameCoordinator(observer,Mock(return_value=fresh),Mock(),Mock(),putt_reader=reader)
        def inspect(safety,sct,rect,frame,current,captured_at):
            context=deepcopy(state['game_context'])
            context.update(observed_at=time.perf_counter()-10,stable_frames=2,timer_deadline=deadline)
            current['game_context']=game.revalidate_putt(safety,sct,rect,frame,current,context)
            return current['game_context']
        game.inspect_ready=inspect
        with patch.object(live,'GAME',game),patch.object(live,'capture',return_value=old), \
                patch.object(live,'shot_state',side_effect=lambda *args,**kwargs:deepcopy(state)), \
                patch.object(live,'ocr_variants',return_value=['']),patch('round_manager.scorecard',return_value=None), \
                patch.object(live,'shot_stats',return_value=None),patch.object(live,'diagnostic_state'), \
                patch.object(live,'send') as send:
            outcome,after=live.wait_until_next(Mock(),None,RECT,before,timeout=8,observer=motion)
        self.assertEqual(outcome,'next')
        self.assertEqual((after['mode'],after['shot'],after['distance']),('PUTT',4,21.9))
        game.move_to.assert_not_called()
        send.assert_not_called()

    def test_real_fresh_frame_renews_proof_without_extending_practice_clock(self):
        old,state=self.cases[0]
        fresh,_=self.cases[-1]
        original=time.perf_counter()-10
        deadline=original+90
        context=deepcopy(state['game_context'])
        context.update(observed_at=original,stable_frames=2,timer_deadline=deadline)
        observer=SimpleNamespace(solo_confirmed=True,read_hud=lambda frame:None,
            solo_ready_reader=lambda frame:live.swing_button(frame,RECT) is not None and live.ready_meter(frame,RECT) is not None,
            last_time=original,turn_deadline=deadline,sequence=1)
        reader=lambda frame,sign:quick_putt_badge(frame,sign,live.ocr_text) if live.find_distance_sign(frame)==tuple(sign) else None
        game=GameCoordinator(observer,Mock(return_value=fresh),Mock(),Mock(),putt_reader=reader)
        renewed=game.revalidate_putt(Mock(),None,RECT,old,deepcopy(state),context)
        self.assertGreater(renewed['observed_at'],original)
        self.assertTrue(renewed['permission']['allowed'],renewed['permission'])
        self.assertEqual(renewed['stable_frames'],2)
        self.assertEqual(renewed['timer_deadline'],deadline)
        self.assertEqual(observer.turn_deadline,deadline)
        game.move_to.assert_not_called()


if __name__=='__main__':unittest.main()
