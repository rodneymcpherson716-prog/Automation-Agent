"""Replay real MSS multiplayer frames; no native input or live gameplay."""
import json,unittest
from unittest.mock import Mock,patch
from pathlib import Path
import cv2
import execution_engine as live
from context_manager import ContextObserver,GameCoordinator,text_data
from multiplayer_hud import TeamHudReader
from game_state import turn_permission

ROOT=Path(__file__).parent
class MultiplayerHudTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  live.setup()
  cls.spec=json.loads((ROOT/'wgt_calibration/alternate_shot_v1/profile.candidate.json').read_text())
  def ready(im):
   h,w=im.shape[:2];r=dict(left=0,top=0,width=w,height=h)
   return live.ready_meter(im,r) is not None and live.swing_button(im,r) is not None
  cls.reader=TeamHudReader(ROOT,cls.spec,text_data,ready)
 def frame(self,n,second=False):
  return cv2.imread(str(ROOT/f'wgt_debug/multiplayer_20261111_{"071035" if second else "070215"}/frame_{n:03d}.png'))
 def context(self,frame):
  observer=ContextObserver(ROOT)
  observer.team_hud=self.reader;observer.layout_verified=True
  observer.regions={}
  observer.observe(frame,dict(hole=2,shot=1),now=100)
  return observer.observe(frame,dict(hole=2,shot=1),now=101)
 def test_own_tee_requires_ready_controls_timer_and_four_reviewed_names(self):
  c=self.context(self.frame(57))
  self.assertFalse(c['permission']['allowed'])  # Historical 90-second session.
  self.assertIn('60-second', ' '.join(c['permission']['reasons']))
  self.assertEqual((c['seconds_left'],c['shot_label']),(90,'NOT_STARTED'))
  self.assertEqual(c['turn_limit'],60)
  self.assertNotIn('timer_deadline',c)
  self.assertEqual(c['permission']['teammate'],'clarencelogan')
  self.assertEqual(set(c['permission']['opponents']),{'briancolley02','ptsun'})
 def test_second_shot_survives_score_change_from_as_to_one_down(self):
  c=self.context(self.frame(20,True))
  self.assertFalse(c['permission']['allowed'])  # 83 cannot be this session's remaining time.
  self.assertIn('60-second', ' '.join(c['permission']['reasons']))
  self.assertEqual((c['seconds_left'],c['shot_label']),(83,'2ND SHOT'))
 def test_verified_own_team_hud_accepts_sixty_second_session(self):
  actual=self.reader._clock_and_label
  def current_session(*args):
   seconds,label,raw=actual(*args)
   return 60,label,raw  # Synthetic timer; saved image is left unchanged.
  with patch.object(self.reader,'_clock_and_label',side_effect=current_session):
   c=self.context(self.frame(57))
  self.assertTrue(c['permission']['allowed'],c['permission'])
  self.assertEqual(c['timer_deadline'],160)
  self.assertEqual(c['turn_limit'],60)
 def test_reviewed_scoring_putt_labels_remain_bound_to_the_opponent(self):
  for n,label in ((0,'FOR BIRDIE'),(28,'FOR PAR'),(43,'FOR BOGEY')):
   with self.subTest(frame=n):
    c=self.context(self.frame(n));self.assertEqual(c['shot_label'],label)
    self.assertFalse(c['permission']['allowed'])
 def test_own_highlight_during_flight_never_permits_another_swing(self):
  for n in (71,79):
   with self.subTest(frame=n):
    c=self.context(self.frame(n));self.assertEqual(c['active_player'],'tigerwould52')
    self.assertFalse(c['permission']['allowed']);self.assertEqual(c['turn_state'],'OWN_SHOT_IN_PROGRESS')
 def test_opponents_and_teammate_loading_are_blocked(self):
  for n in (0,14,28):
   with self.subTest(frame=n):self.assertFalse(self.context(self.frame(n))['permission']['allowed'])
 def test_occluded_roster_is_blocked(self):
  im=self.frame(57);im[43:65,1155:1228]=0
  self.assertFalse(self.context(im)['permission']['allowed'])
 def test_missing_ready_detector_cannot_grant_our_turn(self):
  reader=TeamHudReader(ROOT,self.spec,text_data,None)
  c=self.context(self.frame(57));r=reader.read(self.frame(57))
  c.update(turn_state=r['turn_state'],ready_controls=r['ready_controls'])
  self.assertFalse(turn_permission(c,101)['allowed'])
 def test_unknown_format_cannot_reuse_team_profile(self):
  with self.assertRaises(ValueError):TeamHudReader(ROOT,dict(self.spec,format='STROKE_PLAY'),text_data,None)
 def test_changed_roster_template_is_rejected(self):
  import copy
  s=copy.deepcopy(self.spec);s['roster_templates'][0]['sha256']='0'*64
  with self.assertRaises(ValueError):TeamHudReader(ROOT,s,text_data,None)
 def refresh(self,frame,fresh):
  observer=ContextObserver(ROOT);observer.team_hud=self.reader
  game=GameCoordinator(observer,Mock(return_value=fresh),Mock(),Mock())
  with patch('context_manager.time.perf_counter',return_value=110):
   return game.refresh_turn_frame(Mock(),None,{},frame,dict(sign=[616,89,61,46]),100)
 def test_slow_ocr_refresh_requires_identical_measurement_pixels(self):
  frame=self.frame(57)
  fresh,stamp=self.refresh(frame,frame.copy())
  self.assertEqual(stamp,110)
  for y,x in ((60,130),(695,170),(95,630),(674,1100)):
   with self.subTest(pixel=(y,x)):
    changed=frame.copy();changed[y,x]^=255
    result,stamp=self.refresh(frame,changed)
    self.assertIs(result,frame);self.assertEqual(stamp,100)
 def test_slow_ocr_refresh_cannot_renew_flight_or_another_players_turn(self):
  frame=self.frame(57)
  for fresh in (self.frame(71),self.frame(28)):
   result,stamp=self.refresh(frame,fresh)
   self.assertIs(result,frame);self.assertEqual(stamp,100)
if __name__=='__main__':unittest.main()
