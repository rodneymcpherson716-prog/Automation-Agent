"""Offline role/turn/clock/etiquette policy tests; no desktop input."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
import game_state as game


def context(text,now=100.):
    c=game.combine_context([dict(text=text,confidence=96),dict(text=text,confidence=94)],now)
    c['stable_frames']=2  # Synthetic two-frame evidence; production counts MSS captures.
    c['layout_verified']=True  # Policy fixture, not a real WGT layout calibration.
    return c


SOLO='SOLO STROKE PLAY\nPLAYER: Tigerwould52\nACTIVE PLAYER: Tigerwould52\nBALL OWNER: Tigerwould52\nLIE: TEE'
MATCH='MATCH PLAY\nPLAYERS: Tigerwould52, Opponent1\nACTIVE PLAYER: Tigerwould52\nBALL OWNER: Tigerwould52\nTIME LEFT: 60'
ALT='ALTERNATE SHOT\nTEAM A: Tigerwould52, Mate1\nTEAM B: Opponent1, Opponent2\nACTIVE PLAYER: Tigerwould52\nSHARED BALL: TEAM A\nTIME LEFT: 60'
PUTT='MATCH PLAY\nPLAYERS: Tigerwould52, Opponent1\nACTIVE PLAYER: Opponent1\nBALL OWNER: Opponent1\nOPPONENT PUTT: Opponent1 2.0 FT\nLIE: GREEN'
CONTROL=dict(label='CONCEDE PUTT',confidence=.96,calibrated=True,enabled=True,rect=[10,10,140,40])


class GamePolicyTests(unittest.TestCase):
    def test_real_wgt_watching_banners_veto_a_contradictory_own_turn(self):
        for role in ('TEAMMATE','OPPONENT'):
            c=context(SOLO+'\nWATCHING '+role)
            self.assertEqual(c['turn_state'],'WATCHING_'+role)
            self.assertFalse(game.turn_permission(c,101)['allowed'])
        c=context(PUTT+'\nWATCHING TEAMMATE')
        self.assertFalse(game.concession_decision(c,CONTROL,101)['eligible'])
    def test_saved_policy_override_and_bom_preserve_the_settings_file(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'game_policy.json'
            self.assertEqual(game.GamePolicy.load(path).concession_max_feet,2.)
            path.write_text(json.dumps(dict(schema=1,concession_max_feet=1.5)),encoding='utf-8-sig')
            original=path.read_bytes()
            self.assertEqual(game.GamePolicy.load(path).concession_max_feet,1.5)
            self.assertEqual(game.GamePolicy.load(path,override=2.1).concession_max_feet,2.1)
            self.assertEqual(path.read_bytes(),original)

    def test_invalid_policy_values_fail_closed(self):
        for value in (-1,float('nan'),float('inf'),True,'2',None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    game.GamePolicy(value)
                self.assertFalse(game.concession_decision(context(PUTT),CONTROL,101,max_feet=value)['eligible'])
        with TemporaryDirectory() as folder:
            path=Path(folder)/'game_policy.json'
            for document in (dict(schema=2,concession_max_feet=2),dict(schema=1),
                             dict(schema=1,concession_max_feet=2,conceed_feet=3)):
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    game.GamePolicy.load(path)
    def test_profile_name_and_swing_are_not_a_turn(self):
        c=context('TIGERWOULD52\nSWING\n1ST SHOT')
        self.assertFalse(game.turn_permission(c,101)['allowed'])

    def test_explicit_solo_owner_and_turn_are_required(self):
        self.assertTrue(game.turn_permission(context(SOLO),101)['allowed'])
        for label in ('ACTIVE PLAYER: Tigerwould52','BALL OWNER: Tigerwould52','SOLO STROKE PLAY'):
            self.assertFalse(game.turn_permission(context(SOLO.replace(label,'')),101)['allowed'])

    def test_solo_cannot_hide_a_second_player(self):
        self.assertFalse(game.turn_permission(context(SOLO+'\nPLAYER: Opponent1'),101)['allowed'])

    def test_opponents_and_teammates_never_receive_our_swing_permission(self):
        for who in ('Mate1','Opponent1','Opponent2'):
            c=context(ALT.replace('ACTIVE PLAYER: Tigerwould52','ACTIVE PLAYER: '+who))
            self.assertFalse(game.turn_permission(c,101)['allowed'])

    def test_alternate_shot_requires_complete_teams_and_our_shared_ball(self):
        allowed=game.turn_permission(context(ALT),101)
        self.assertTrue(allowed['allowed'],allowed)
        self.assertEqual(allowed['teammate'],'mate1')
        self.assertEqual(allowed['opponents'],['opponent1','opponent2'])
        for text in (ALT.replace('TEAM B: Opponent1, Opponent2',''),ALT.replace('SHARED BALL: TEAM A','SHARED BALL: TEAM B')):
            self.assertFalse(game.turn_permission(context(text),101)['allowed'])

    def test_stroke_play_does_not_mean_solo(self):
        self.assertFalse(game.turn_permission(context(SOLO.replace('SOLO STROKE PLAY','STROKE PLAY')),101)['allowed'])

    def test_conflicting_ocr_and_low_confidence_abstain(self):
        for second,confidence in ((MATCH.replace('Tigerwould52','Opponent1'),96),(MATCH,40)):
            c=game.combine_context([dict(text=MATCH,confidence=96),dict(text=second,confidence=confidence)],100)
            self.assertFalse(game.turn_permission(c,101)['allowed'])

    def test_stale_and_future_evidence_never_grants_a_turn(self):
        for now in (99,105):
            self.assertFalse(game.turn_permission(context(SOLO),now)['allowed'])

    def test_one_frame_and_conflicting_ball_owner_do_not_authorize(self):
        c=context(SOLO)
        c['stable_frames']=1
        self.assertFalse(game.turn_permission(c,101)['allowed'])
        self.assertFalse(game.turn_permission(context(SOLO+'\nBALL OWNER: Opponent1'),101)['allowed'])

    def test_30_60_90_second_timers_keep_the_original_capture_time(self):
        for seconds in (30,60,90):
            c=context(MATCH.replace('TIME LEFT: 60','TIME LEFT: '+str(seconds)))
            b=game.swing_budget(c,103,stroke_seconds=12,margin=3)
            self.assertTrue(b['allowed'],b)
            self.assertEqual(b['deadline'],100+seconds-3)
            self.assertEqual(b['remaining'],seconds-6)

    def test_low_or_missing_time_blocks_multiplayer(self):
        for text in (MATCH.replace('TIME LEFT: 60','TIME LEFT: 10'),MATCH.replace('TIME LEFT: 60','')):
            self.assertFalse(game.swing_budget(context(text),101)['allowed'])

    def test_concede_at_2_feet_and_below_with_opponent_owned_green_ball(self):
        for feet in ('0.5','1.9','2.0'):
            c=context(PUTT.replace('2.0 FT',feet+' FT'))
            self.assertTrue(game.concession_decision(c,CONTROL,101)['eligible'])

    def test_concession_limit_is_configurable_without_changing_ownership_checks(self):
        c=context(PUTT.replace('2.0 FT','2.01 FT'))
        self.assertFalse(game.concession_decision(c,CONTROL,101)['eligible'])
        self.assertTrue(game.concession_decision(c,CONTROL,101,max_feet=2.1)['eligible'])
        self.assertFalse(game.concession_decision(c,CONTROL,101,max_feet=0)['eligible'])
        own=context(PUTT.replace('BALL OWNER: Opponent1','BALL OWNER: Tigerwould52'))
        self.assertFalse(game.concession_decision(own,CONTROL,101,max_feet=3)['eligible'])

    def test_concession_rejects_too_far_missing_units_own_ball_and_wrong_mode(self):
        for old,new in (('2.0 FT','2.01 FT'),('2.0 FT','2.0'),('LIE: GREEN','LIE: FRINGE'),
                        ('MATCH PLAY','STROKE PLAY'),('OPPONENT PUTT: Opponent1','OPPONENT PUTT: Tigerwould52'),
                        ('BALL OWNER: Opponent1','BALL OWNER: Tigerwould52')):
            self.assertFalse(game.concession_decision(context(PUTT.replace(old,new)),CONTROL,101)['eligible'])

    def test_concession_never_uses_concede_hole_or_an_uncalibrated_control(self):
        for control in (None,dict(CONTROL,label='CONCEDE HOLE'),dict(CONTROL,calibrated=False),dict(CONTROL,confidence=.5),dict(CONTROL,enabled=False)):
            self.assertFalse(game.concession_decision(context(PUTT),control,101)['eligible'])

    def test_concession_never_grants_teammates_shared_ball_putts(self):
        c=context(ALT.replace('ACTIVE PLAYER: Tigerwould52','ACTIVE PLAYER: Mate1')+
                  '\nBALL OWNER: Mate1\nLIE: GREEN\nOPPONENT PUTT: Mate1 1.0 FT')
        self.assertFalse(game.concession_decision(c,CONTROL,101)['eligible'])

    def test_alternate_shot_opposing_shared_ball_is_eligible(self):
        c=context(ALT.replace('ACTIVE PLAYER: Tigerwould52','ACTIVE PLAYER: Opponent1')
                  .replace('SHARED BALL: TEAM A','SHARED BALL: TEAM B')+
                  '\nBALL OWNER: Opponent1\nLIE: GREEN\nOPPONENT PUTT: Opponent1 2.0 FT')
        self.assertTrue(game.concession_decision(c,CONTROL,101)['eligible'])
        c['ball_team']='A'
        self.assertFalse(game.concession_decision(c,CONTROL,101)['eligible'])

    def test_explicit_tee_policy_does_not_apply_to_first_putt_or_recovery(self):
        state=dict(distance=200,unit='yd',shot=1,mode='FULL',lie='TEE')
        self.assertTrue(game.strategy(state)['tee_full_power'])
        for changes in (dict(shot=2),dict(lie='BUNKER'),dict(mode='PUTT'),dict(lie=None)):
            self.assertFalse(game.strategy(dict(state,**changes))['tee_full_power'])

    def test_fixed_aim_hazard_layup_never_moves_flag_or_camera(self):
        s=dict(distance=180,unit='yd',shot=2,mode='FULL',lie='FAIRWAY',
               verified_hazards=[dict(kind='WATER',start=165,end=200,unit='yd',current_aim=True)])
        plan=game.strategy(s)
        self.assertEqual(plan['target'],155)
        self.assertFalse(plan['aim_adjustment'])
        self.assertEqual(s['distance'],180)
        self.assertFalse(game.strategy(dict(s,verified_hazards=[]))['hazards_verified'])


if __name__=='__main__':
    unittest.main()
