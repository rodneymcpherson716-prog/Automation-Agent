"""New-shot evidence bundles. No input, capture, OCR, fitting or old-record repair.

Readers supply observations; missing facts stay unknown. Completeness is an
audit of the requested recording fields, separate from existing training rules.
"""
from copy import deepcopy
import math

from learning_data import (PLAYER, automatic_record, number, solo_owner,
                               stats_values, verified_lie)


def ding_reading(trace, shot_id):
    if (not isinstance(trace, dict) or not isinstance(shot_id, str) or not shot_id
            or trace.get('shot_id') != shot_id or trace.get('clicked') is not True
            or not number(trace.get('target'), 0, 2000)):
        return None
    samples = trace.get('samples')
    if not isinstance(samples, list) or len(samples) < 3:
        return None
    if any(not isinstance(s, dict) or not number(s.get('t'), 0, 60) for s in samples):
        return None
    if any(b['t'] <= a['t'] for a, b in zip(samples, samples[1:])):
        return None
    valid = [s for s in samples if number(s.get('edge'), 0, 2000)]
    if len(valid) < 3 or valid[-1] is not samples[-1]:
        return None
    target, edge = trace['target'], valid[-1]['edge']
    return dict(shot_id=shot_id, target_x_px=target,
                observed_edge_x_at_trigger_px=edge, edge_minus_target_px=edge-target,
                samples=deepcopy(samples), current_shot_trace=True,
                exact_mouse_down_verified=False, lateral_ball_miss_verified=False,
                can_change_live_accuracy=False)


def build_shot_capture(shot_id, before, after, actual_power, plan, trace, outcome):
    """Freeze one new shot's full observations and disclose every missing field.

Uses the unchanged automatic_record eligibility rules for the distance-learning
status. Neither a complete packet nor a DING trigger proves impact accuracy.
Final position means a verified remaining hole distance and observed lie;
absolute course coordinates are not supplied by the current readers.
"""
    before, after, plan = deepcopy(before), deepcopy(after or {}), deepcopy(plan or {})
    c, ac = before.get('game_context') or {}, after.get('game_context') or {}
    derived = automatic_record(shot_id, 'capture-audit-only', before, after, actual_power, plan)
    pre_owner = (before.get('player') is True and c.get('permission', {}).get('allowed') is True
                 and c.get('active_player') == PLAYER and PLAYER in c.get('players', [])
                 and (c.get('ball_owner') == PLAYER or c.get('format') == 'ALTERNATE_SHOT'
                      and c.get('teams', {}).get(PLAYER) is not None
                      and c.get('ball_team') == c['teams'][PLAYER]))
    stats_owner = (ac.get('stats_owner') in (None, PLAYER)
                   and (solo_owner(after) or ac.get('stats_owner') == PLAYER
                        and number(ac.get('confidence', {}).get('stats_owner'), .85, 1)))
    attributed = derived['verified']['outcome_attributed']
    result_owner = bool(pre_owner and stats_owner and attributed)
    wind = before.get('wind') or {}
    lo, hi, dx, dy = (wind.get(k) for k in ('low_mph', 'high_mph', 'screen_dx', 'screen_dy'))
    wind_ok = (all(number(x, 0, 100) for x in (lo, hi)) and lo <= hi
               and all(number(x, -1, 1) for x in (dx, dy))
               and (hi == 0 or .9 <= math.hypot(dx, dy) <= 1.05))
    old_stats = stats_values(before.get('previous_stats'))
    ding = ding_reading(trace, shot_id)
    final_distance = derived['actual']['final_distance_to_hole'] if result_owner else None
    final_lie = after.get('lie') if result_owner and verified_lie(after) else None
    checks = dict(
        shot_identity=isinstance(shot_id, str) and bool(shot_id),
        before_tigerwould52_turn=bool(pre_owner),
        before_club_and_units=(isinstance(before.get('club_name'), str) and bool(before['club_name'])
                               and number(before.get('club_range'), .01, 1000)
                               and before.get('unit') == before.get('club_unit')
                               and before.get('unit') in ('yd', 'ft')
                               and before.get('putt') is (before.get('mode') == 'PUTT')
                               and (before.get('mode') == 'PUTT') == (before.get('unit') == 'ft')),
        before_distance=number(before.get('distance'), .01, 1000),
        before_lie=bool(verified_lie(before)),
        before_wind=bool(wind_ok),
        before_elevation=number(before.get('elevation_ft'), -300, 300),
        before_stats_baseline=all(v is not None for v in old_stats.values()),
        actual_meter_power=number(actual_power, .025, 1),
        ding_trace=ding is not None,
        attributed_changed_stats=bool(attributed),
        result_belongs_to_tigerwould52=result_owner,
        distance_learning_evidence=derived['eligible'],
        final_position=final_distance is not None and final_lie is not None)
    missing = [key for key, verified in checks.items() if not verified]
    return dict(schema=1, kind='new_shot_capture', shot_id=shot_id if checks['shot_identity'] else None,
                outcome=outcome, before=before, after=after,
                performance=plan.get('performance'), risk_level=plan.get('risk_level'),
                expected_score=plan.get('expected_score'), spin_advice=plan.get('spin_advice'),
                swing=dict(requested_power_fraction=plan.get('power'),
                           observed_meter_power_fraction=actual_power,
                           attributed_game_power_fraction=derived['actual_game_power'] if result_owner else None,
                           ding=ding, raw_accuracy_trace=deepcopy(trace) if isinstance(trace, dict) else None),
                result=dict(owner=PLAYER if result_owner else None,
                            carry_yd=derived['actual']['carry_yd'] if result_owner else None,
                            roll_yd=derived['actual']['roll_yd'] if result_owner else None,
                            final_position=dict(distance_to_hole=final_distance, lie=final_lie,
                                                absolute_coordinates=None),
                            actual_score=None,
                            hole_completion_observed=after.get('complete') is True),
                verification=checks, complete=not missing, missing=missing,
                distance_training_eligible=derived['eligible'],
                distance_training_exclusions=derived['exclusions'],
                completeness_changes_training_rules=False, use_for_live_control=False,
                limitations=['DING edge is the click trigger, not verified mouse-down or ball impact.',
                             'Hole completion does not establish measured zero distance or carry.',
                             'Missing final position is never reconstructed from carry and roll.'])
