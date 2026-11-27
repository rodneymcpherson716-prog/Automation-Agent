"""Evidence adapters for shadow learning. No capture, input, fitting or guesses."""
from copy import deepcopy
import hashlib
import json
import math

from environmental_evidence import observe_wind_trigger

PLAYER = 'tigerwould52'
LIES = {'TEE', 'FAIRWAY', 'ROUGH', 'HEAVY ROUGH', 'BUNKER', 'GREEN', 'FRINGE'}
MODES = {'FULL', 'CHIP', 'PITCH', 'PUTT', 'FLOP', 'PUNCH'}
BASELINE_VERSION = 'wgt-range-conditions-forward-v1'


def number(x, low=-math.inf, high=math.inf):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and low <= x <= high


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def lie_range(value):
    return (list(value) if isinstance(value, (list, tuple)) and len(value) == 2
            and all(number(x, 0, 100) for x in value) and value[0] <= value[1] else None)


def compact_state(state):
    keys = ('club_name', 'club_range', 'club_unit', 'mode', 'putt', 'distance', 'unit',
            'distance_source', 'lie', 'lie_percent', 'lie_loss_percent', 'wind',
            'elevation_ft', 'green_speed', 'hole', 'shot', 'target_hole_link')
    result = {key: deepcopy(state.get(key)) for key in keys}
    result['lie_percent'] = lie_range(result['lie_percent'])
    result['lie_loss_percent'] = lie_range(result['lie_loss_percent'])
    return result


def group_key(state):
    if (state.get('lie') not in LIES or state.get('mode') not in MODES
            or not isinstance(state.get('club_name'), str) or not state['club_name']
            or not number(state.get('club_range'), .01, 1000)
            or state.get('unit') != state.get('club_unit')
            or state.get('unit') not in ('yd', 'ft')
            or state.get('putt') is not (state['mode'] == 'PUTT')
            or (state['mode'] == 'PUTT') != (state['unit'] == 'ft')):
        return None
    return json.dumps([state['mode'], state['lie'], state['club_name'], float(state['club_range']),
                       state['unit'], 'displayed_lie' if lie_range(state.get('lie_percent')) else 'no_lie_display',
                       'explicit_loss' if lie_range(state.get('lie_loss_percent')) else 'no_explicit_loss'],
                      separators=(',', ':'))


def condition_features(state):
    """Raw lie display is a covariate, never an assumed retention/loss factor."""
    if group_key(state) is None or not number(state.get('distance'), .01, 1000):
        raise ValueError('club/mode/lie/target units missing or inconsistent')
    if not number(state.get('elevation_ft'), -300, 300):
        raise ValueError('elevation unverified')
    features = dict(elevation=state['elevation_ft'] / (state['club_range'] * (1 if state['putt'] else 3)))
    if state['putt']:
        if not number(state.get('green_speed'), 4, 20):
            raise ValueError('putting green speed unverified')
        features['green_speed'] = state['green_speed']/10
    else:
        wind = state.get('wind') or {}
        lo, hi, dx, dy = [wind.get(k) for k in ('low_mph', 'high_mph', 'screen_dx', 'screen_dy')]
        if (not all(number(x, 0, 100) for x in (lo, hi)) or lo > hi
                or not all(number(x, -1, 1) for x in (dx, dy))
                or (hi > 0 and not .9 <= math.hypot(dx, dy) <= 1.05)):
            raise ValueError('wind range/direction unverified')
        features.update(wind_screen_y=(lo+hi)/2*dy/20,
                        wind_screen_x=(lo+hi)/2*dx/20, wind_spread=(hi-lo)/20)
    percent = lie_range(state.get('lie_percent'))
    if state['lie'] in ('BUNKER', 'ROUGH', 'HEAVY ROUGH') and percent is None:
        raise ValueError('recovery lie percentage unverified')
    if percent is not None:
        features.update(lie_midpoint=sum(percent)/200, lie_spread=(percent[1]-percent[0])/100)
    loss = lie_range(state.get('lie_loss_percent'))
    if loss is not None:
        if loss[1] >= 100:
            raise ValueError('invalid explicitly labelled lie loss')
        features['explicit_loss_midpoint'] = sum(loss)/200
    return features


def baseline_descriptor(plan=None, calibration=None, state=None):
    label = (plan or {}).get('model', 'selected-range baseline')
    curve = calibration.curve(state) if calibration is not None and state is not None else None
    return dict(algorithm=BASELINE_VERSION, label=label, curve=curve,
                empirical_curve_unavailable=('empirical' in label and curve is None))


def baseline_distance(state, power, descriptor=None):
    """Inverse of existing planner's distance corrections, explicitly predicted.

    The first-tee 100% policy remains an action policy, not a carry measurement.
    Historical empirical curves cannot be reconstructed if their knots weren't saved.
    """
    if not number(power, 0, 1):
        return None
    try:
        condition_features(state)
        desc = descriptor or baseline_descriptor()
        if desc.get('empirical_curve_unavailable'):
            return None
        effective = power * state['club_range']
        curve = desc.get('curve')
        if curve and curve[0][0] <= power <= curve[-1][0]:
            import numpy as np
            effective = float(np.interp(power, [p for p, d in curve], [d for p, d in curve]))
        if state['putt']:
            return max(0., effective-state['elevation_ft']*12)
        loss = lie_range(state.get('lie_loss_percent'))
        if loss:
            effective *= 1-sum(loss)/200
        wind = state['wind']
        rate = .003 if state['mode'] in ('CHIP', 'PITCH') else .005
        factor = 1+rate*wind['screen_dy']*(wind['low_mph']+wind['high_mph'])/2
        return max(0., effective/factor-state['elevation_ft']/3)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return None


def verified_lie(state):
    context = state.get('game_context') or {}
    return (state.get('lie') in LIES and context.get('lie') == state['lie']
            and number(context.get('confidence', {}).get('lie'), .85, 1))


def solo_owner(state):
    c = state.get('game_context') or {}
    e = c.get('solo_evidence') or {}
    return (state.get('player') is True and c.get('source') == 'solo_user_confirmed_live_hud'
            and c.get('format') == 'SOLO_STROKE' and c.get('players') == [PLAYER]
            and c.get('active_player') == PLAYER and c.get('ball_owner') == PLAYER
            and all(e.get(k) is True for k in ('player_header_visible', 'swing_and_meter_visible',
                                               'multiplayer_conflicts_absent', 'user_confirmed')))


def stats_values(stats):
    if not isinstance(stats, dict):
        return dict(carry_yd=None, roll_yd=None, power_pct=None)
    return {key: stats.get(key) if number(stats.get(key), 0, high) else None
            for key, high in (('carry_yd', 600), ('roll_yd', 200), ('power_pct', 100))}


def hole_distance(after):
    link = after.get('target_hole_link') or {}
    if (link.get('source') == 'visible_lock_beside_target'
            and after.get('distance_source') in ('floating_target', 'verified_hole_hud')
            and after.get('unit') in ('yd', 'ft') and number(after.get('distance'), 0, 1000)):
        return dict(value=after['distance'], unit=after['unit'], source='stable ready HUD locked to hole')
    # Completing a hole/scorecard/concession does NOT prove a measured zero.
    return None


def verified_aim(state):
    value = state.get('aim_measurement') or {}
    if (value.get('source') == 'current_aim_marker_ocr' and value.get('verified') is True
            and number(value.get('confidence'), .85, 1)
            and all(number(value.get(k), 0, 1) for k in ('x', 'y'))):
        return deepcopy(value)
    return None


def area_result(intent, after):
    if not intent or not verified_lie(after):
        return None
    if intent == 'OUTSIDE_BUNKER':
        return after['lie'] != 'BUNKER'
    return after['lie'] == intent if intent in LIES else None


def automatic_record(shot_id, session_id, before, after, observed_power, plan=None, prediction=None,
                     evidence=None):
    """Consume the controller's stable-result contract; preserve excluded rows too."""
    after = after or {}
    plan = plan or {}
    state = compact_state(before)
    c = before.get('game_context') or {}
    ac = after.get('game_context') or {}
    reasons = []
    if before.get('player') is not True or c.get('permission', {}).get('allowed') is not True:
        reasons.append('pre-shot owner/turn not verified')
    if not verified_lie(before):
        reasons.append('lie label unverified')
        state['lie'] = None
    if state.get('lie_percent') is not None and (c.get('lie_percent') != state['lie_percent']
            or not number(c.get('confidence', {}).get('lie_percent'), .85, 1)):
        state['lie_percent'] = None
        reasons.append('lie percentage unverified')
    if state.get('lie_loss_percent') is not None and (c.get('lie_loss_percent') != state['lie_loss_percent']
            or not number(c.get('confidence', {}).get('lie_loss_percent'), .85, 1)):
        state['lie_loss_percent'] = None
        reasons.append('explicit lie-loss label unverified')
    motion = after.get('motion') or {}
    transition = (after.get('movement_verification') == 'video motion plus stable stroke/HUD advance'
                  and before.get('hole') == after.get('hole')
                  and number(before.get('shot'), 1, 30) and after.get('shot') == before['shot']+1
                  and not motion.get('capture_error')
                  and (motion.get('scene_motion_observed') is True or motion.get('moving_ball_candidate_tracked') is True))
    # An explicit foreign Shot Stats owner vetoes a solo/profile header.
    owner = (ac.get('stats_owner') in (None, PLAYER)
             and (solo_owner(after) or ac.get('stats_owner') == PLAYER
                  and number(ac.get('confidence', {}).get('stats_owner'), .85, 1)))
    stats = stats_values(after.get('fresh_stats'))
    old = stats_values(before.get('previous_stats'))
    fresh = (all(v is not None for v in stats.values()) and all(v is not None for v in old.values())
             and stats != old)
    if not transition:
        reasons.append('exactly one stable stroke advance with motion unverified')
    if not owner:
        reasons.append('fresh Shot Stats owner unverified')
    if not fresh:
        reasons.append('complete changed Shot Stats/baseline comparison unverified')
    attributed = bool(transition and owner and fresh)
    actual_power = stats['power_pct']/100 if attributed else None
    if not number(observed_power, .025, 1):
        reasons.append('observed meter power unverified')
    if actual_power is not None and number(observed_power, .025, 1) and abs(actual_power-observed_power) > .035:
        reasons.append('game POWER disagrees with observed meter; result attribution uncertain')
        attributed = False
        actual_power = None
    if group_key(state) is None:
        reasons.append('context group incomplete')
    try:
        condition_features(state)
    except ValueError as exc:
        reasons.append(str(exc))
    if not number(actual_power, .025, 1):
        reasons.append('actual game power unavailable/outside training domain')
    carry = stats['carry_yd'] if attributed else None
    roll = stats['roll_yd'] if attributed else None
    total = carry+roll if carry is not None and roll is not None else None
    travel = (None if total is None else total*3) if state.get('putt') else (
        total if state.get('mode') in ('CHIP', 'PITCH') else carry)
    basis = 'total' if state.get('mode') in ('CHIP', 'PITCH', 'PUTT') else 'carry' if state.get('mode') in MODES else None
    if basis is None:
        travel = None
    desc = (prediction or {}).get('baseline_descriptor') or baseline_descriptor(plan)
    intended = plan.get('intended_area') if plan.get('intent_source') == 'planner' else None
    reached = area_result(intended, after) if transition and owner else None
    target = (before.get('strategy') or {}).get('target', state.get('distance'))
    delta = travel-target if travel is not None and number(target, .01, 1000) else None
    return dict(schema=1, id=shot_id, session_id=session_id, source='automatic verified session',
                state=state, group=group_key(state), eligible=not reasons, exclusions=sorted(set(reasons)),
                selected_power=plan.get('power') if number(plan.get('power'), 0, 1) else None,
                observed_meter_power=observed_power if number(observed_power, 0, 1) else None,
                actual_game_power=actual_power, aim=verified_aim(before), aim_policy='unchanged current game aim',
                actual=dict(carry_yd=carry, roll_yd=roll, travel=travel,
                            travel_basis=basis,
                            unit=state.get('unit'), final_distance_to_hole=hole_distance(after) if transition else None,
                            reached_intended_area=reached, intended_area=intended,
                            recovery_failed=(not reached) if intended == 'OUTSIDE_BUNKER' and reached is not None else None,
                            numeric_travel_minus_target=delta, comparison_basis='travel versus planned distance; not measured ball direction'),
                verified=dict(outcome_attributed=attributed, final_distance=transition and hole_distance(after) is not None,
                              aim=verified_aim(before) is not None, intended_area=reached is not None),
                prediction=prediction, baseline_descriptor=desc,
                baseline_at_actual_power=baseline_distance(state, actual_power, desc),
                evidence=evidence or {},
                wind_accuracy=observe_wind_trigger(state, evidence, shot_id),
                use_for_live_control=False)


def manual_record(root, record):
    """Recheck manual-record evidence, not just the stored 'verified' boolean."""
    from datetime import datetime
    from terrain_recording import (ManualBunkerStore, paired_values, before_values,
                                      stats_values as manual_stats, local_solo, pair_identity)
    if (record.get('schema') != 1 or record.get('kind') != 'manual_bunker_shot'
            or record.get('measurement_verified') is not True):
        raise ValueError('manual capture is partial/unverified')
    before, after = record['before'], record['after']
    stamp = datetime.fromisoformat(record['recorded_at_utc']).timestamp()
    store = ManualBunkerStore(root)
    store.validate_pair(before, stamp)
    store.validate_pair(after, stamp)
    if len({s['capture_id'] for s in before+after}) != 4 or after[0]['captured_at'] <= before[-1]['captured_at']:
        raise ValueError('manual before/after captures reused or out of order')
    facts = paired_values(before, before_values)
    stats = paired_values(after, manual_stats)
    previous = paired_values(before, manual_stats)
    b, a = pair_identity(before), pair_identity(after)
    attest = record.get('manual_attestation') or {}
    if (not all(local_solo(s) for s in before+after) or None in b.values()
            or a != dict(hole=b['hole'], shot=b['shot']+1)
            or any(x is None for x in list(facts.values())+list(stats.values())+list(previous.values()))
            or stats == previous or attest.get('one_shot') is not True or attest.get('setup_unchanged') is not True
            or facts['club'] != 'LW' or facts['lie'] != 'BUNKER'):
        raise ValueError('manual attribution/conditions cannot be independently rechecked')
    s = dict(club_name=facts['club'], club_range=facts['club_range_yd'], club_unit='yd', unit='yd',
             mode=facts['mode'], putt=False, distance=facts['target_distance_yd'], distance_source='floating_target',
             lie='BUNKER', lie_percent=facts['lie_percent'], lie_loss_percent=None,
             elevation_ft=facts['elevation_ft'], green_speed=None, hole=b['hole'], shot=b['shot'],
             wind=dict(low_mph=facts['wind_mph'][0], high_mph=facts['wind_mph'][1],
                       screen_dx=facts['wind_screen_vector'][0], screen_dy=facts['wind_screen_vector'][1]))
    condition_features(s)
    total = stats['carry_yd']+stats['roll_yd']
    travel = total if s['mode'] in ('CHIP', 'PITCH') else stats['carry_yd']
    return dict(schema=1, id='manual:'+record['id'], session_id='manual-day:'+record['recorded_at_utc'][:10],
                source='manual HUD measurement with player attestation', state=s, group=group_key(s), eligible=True,
                exclusions=[], selected_power=None, observed_meter_power=None, actual_game_power=stats['power_percent']/100,
                aim=None, aim_policy='manual setup; exact aim unreadable',
                actual=dict(carry_yd=stats['carry_yd'], roll_yd=stats['roll_yd'], travel=travel,
                            travel_basis='total' if s['mode'] in ('CHIP', 'PITCH') else 'carry', unit='yd',
                            final_distance_to_hole=None, reached_intended_area=None, intended_area=None,
                            recovery_failed=None, numeric_travel_minus_target=travel-s['distance'],
                            comparison_basis='travel versus target only; not measured ball direction'),
                verified=dict(outcome_attributed=True, final_distance=False, aim=False, intended_area=False),
                prediction=None, baseline_descriptor=baseline_descriptor(),
                baseline_at_actual_power=baseline_distance(s, stats['power_percent']/100),
                evidence=dict(manual_id=record['id'], capture_hashes=[x['evidence'] for x in before+after],
                              attribution_method=record['attribution_method']), use_for_live_control=False)
