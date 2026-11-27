"""Bunker sensitivity model and an explicitly requested calibration trial.

The HUD's LIE percentage has no established physical meaning in this project.
Keep it distinct from an explicitly labelled LIE LOSS. Neither the selected
club range nor the manual hole-out establishes a sand carry/power curve.
"""
from copy import deepcopy
import math
import re


BUNKER_BLOCK_REASON = ('Automatic bunker swings blocked: bunker power, lie semantics, '
                       'wind and elevation response have not been validated; no swing')


def is_bunker_state(state):
    if str(state.get('lie', '')).upper() in ('BUNKER', 'SAND'):
        return True
    context = state.get('game_context') or {}
    if str(context.get('lie', '')).upper() in ('BUNKER', 'SAND'):
        return True
    raw = context.get('raw', {})
    if isinstance(raw, dict):
        readings = raw.get('lie', [])
        pattern = r'\b(BUNKER|SAND)\b'
    else:
        # combine_context stores a flat list; ContextObserver stores regions.
        # In flat full-context text, distinguish the lie from an aim hazard.
        readings = raw if isinstance(raw, (list, tuple)) else []
        pattern = r'^\s*(?:LIE\s*:\s*)?(BUNKER|SAND)\b'
    return any(isinstance(r, dict) and
               re.search(pattern, str(r.get('text', '')), re.I | re.M)
               for r in readings)


def number(value, label, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError('Invalid or missing bunker ' + label)
    return float(value)


def percent_range(value, label, loss=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Invalid or missing bunker ' + label)
    lo, hi = [number(v, label, 0 if loss else 1, 100) for v in value]
    if lo > hi or (loss and hi >= 100):
        raise ValueError('Invalid bunker ' + label)
    return [lo, hi]


def bunker_power_estimate(state, reference=None):
    """Return a diagnostic estimate and alternative lie hypotheses, never power.

    Nominal effective yards = (target yards + elevation feet/3) *
    (1 - .003 * screen-top tailwind mph). Both coefficients and the linear
    range ratio are unverified assumptions for sand. No lie interpretation
    changes the nominal estimate, including an explicit loss label: its
    relationship to recovery distance is still unmeasured. Crosswind is reported;
    its effect on carry/roll and aim is unresolved. Scenario ranges below
    are sensitivity calculations, NOT confidence bounds or recommendations.
    """
    if not is_bunker_state(state):
        raise ValueError('Bunker estimate requires bunker evidence')
    if (state.get('unit') != 'yd' or state.get('club_unit') != 'yd'
            or state.get('putt') is not False
            or state.get('mode') not in ('FULL', 'CHIP', 'PITCH')):
        raise ValueError('Bunker recovery requires verified yard units and airborne mode')
    if state.get('club_name') not in ('LW', 'SW', 'PW', 'GW'):
        raise ValueError('Bunker recovery requires a verified wedge')
    distance = number(state.get('distance'), 'target distance', .01, 650)
    target = number((state.get('strategy') or {}).get('target', distance),
                    'strategy target distance', .01, 650)
    club = number(state.get('club_range'), 'club range', .01, 650)
    elevation = number(state.get('elevation_ft'), 'elevation', -300, 300)
    lie = percent_range(state.get('lie_percent'), 'displayed lie range')
    wind = state.get('wind')
    if not isinstance(wind, dict):
        raise ValueError('Live bunker wind must be verified')
    mean = number(wind.get('mean_mph'), 'mean wind', 0, 100)
    low = number(wind.get('low_mph', mean), 'low wind', 0, 100)
    high = number(wind.get('high_mph', mean), 'high wind', 0, 100)
    dx = number(wind.get('screen_dx'), 'wind direction x', -1, 1)
    dy = number(wind.get('screen_dy'), 'wind direction y', -1, 1)
    norm = math.hypot(dx, dy)
    if not low <= mean <= high or norm > 1.05 or (mean > 0 and norm < .9):
        raise ValueError('Invalid bunker wind range or direction')
    elevated = target + elevation / 3.
    if elevated <= 0:
        raise ValueError('Bunker elevation estimate is outside its usable domain')
    effective = elevated * (1 + .003 * dy * mean)
    raw = effective / club
    wind_range = sorted(elevated * (1 + .003 * dy * mph) / club for mph in (low, high))
    scenarios = {
        'no_lie_conversion': wind_range,
        'if_display_means_retained_effectiveness':
            [wind_range[0] / (lie[1]/100), wind_range[1] / (lie[0]/100)],
        'if_display_means_loss':
            [wind_range[0] / (1-lie[0]/100) if lie[0] < 100 else None,
             wind_range[1] / (1-lie[1]/100) if lie[1] < 100 else None],
    }
    explicit_loss = state.get('lie_loss_percent')
    if explicit_loss is not None:
        explicit_loss = percent_range(explicit_loss, 'explicit lie-loss range', loss=True)
        scenarios['explicit_loss_label_linear_hypothesis'] = [
            wind_range[0] / (1-explicit_loss[0]/100),
            wind_range[1] / (1-explicit_loss[1]/100)]
    contextual_reference = None
    if reference is not None:
        contextual_reference = dict(
            reported_facts=deepcopy(reference), evidence='player-reported contextual hole-out',
            used_for_fit=False, inferred_carry_yd=None,
            limitation='Wind direction, carry, roll and impact accuracy are unknown; '
                       'a hole-out does not establish carry distance or a reusable power curve')
    return dict(
        power=None, estimated_power=raw, unclipped_power=raw,
        effective_distance=effective, strategy_target=target,
        model='unvalidated bunker sensitivity estimate',
        automatic_swing_allowed=False, block_reason=BUNKER_BLOCK_REASON,
        lie='BUNKER', lie_percent=lie, lie_loss_percent=explicit_loss,
        lie_percent_semantics='uninterpreted HUD percentage; no loss inferred',
        nominal_lie_conversion='none',
        lie_scenarios=scenarios, scenario_ranges_are_confidence_intervals=False,
        observed_inputs=dict(target_distance_yd=distance, strategy_target_yd=target,
                             club_name=state['club_name'], club_range_yd=club,
                             shot_mode=state['mode'], elevation_ft=elevation,
                             lie_percent=deepcopy(lie), lie_loss_percent=deepcopy(explicit_loss),
                             wind=dict(mean_mph=mean, low_mph=low, high_mph=high,
                                       screen_dx=dx, screen_dy=dy)),
        input_evidence='caller-supplied readings; this model does not independently verify OCR',
        assumptions=dict(validation='unverified for bunker recovery',
                         distance_response='linear target distance / selected club range',
                         elevation_yards_per_foot=1/3, wind_fraction_per_mph=.003,
                         wind_axis='screen top treated as tailwind; aim alignment unverified',
                         carry_roll_split='unknown'),
        correction='estimated elevation ft/3 and 0.3% per mph screen-top tailwind; '
                   'linear distance/selected range; lie scenarios are unverified hypotheses',
        crosswind_mph=dx*mean, aim_adjustment=False, calibration_points=0,
        outside_controlled_power_range=not .025 <= raw <= .99,
        contextual_reference=contextual_reference,
        uncertainties=['sand carry/roll response', 'displayed lie semantics',
                       'wind projection relative to current aim', 'crosswind carry effect',
                       'elevation response', 'partial-power range response',
                       'bunker lip clearance and landing surface'])


def bunker_calibration_plan(state, hole, stroke, *, reference=None):
    """Choose a trial power that balances the competing lie hypotheses."""
    if ((state.get('hole'),state.get('shot')) != (hole,stroke) or state.get('lie') != 'BUNKER'
            or state.get('player') is not True or state.get('club_name') != 'LW'
            or state.get('mode') != 'FULL'):
        raise ValueError('Bunker calibration requires the specified stroke, a verified sand lie and full lob wedge')
    model='single bunker calibration trial'
    correction='wind/elevation adjusted; geometric midpoint of the lie-hypothesis power extremes'
    if reference is not None:
        if (reference.get('kind')!='single_bunker_calibration_measurement'
                or reference.get('club')!=state.get('club_name') or reference.get('mode')!=state.get('mode')
                or reference.get('club_range_yd')!=state.get('club_range')
                or reference.get('displayed_lie_percent')!=state.get('lie_percent')
                or reference.get('after',{}).get('lie') not in ('GREEN','FRINGE','FAIRWAY','ROUGH','HEAVY ROUGH')):
            raise ValueError('Measured bunker reference does not match the wedge, mode, range and sand lie')
        travel=number(reference['shot_stats'].get('total_yd'),'reference travel',.01,100)
        aim=number(state.get('distance'),'target distance',.01,650)
        state=dict(state,strategy=dict(state.get('strategy') or {},target=min(aim,travel)))
    result=bunker_power_estimate(state)
    if result['lie_percent'] != [40,50] or result['lie_loss_percent'] is not None:
        raise ValueError('Bunker calibration requires the displayed 40–50% lie band')
    # Treat the display as an unresolved hypothesis. The geometric midpoint
    # minimizes the largest multiplicative miss between the extreme scenarios.
    if reference is None:
        candidates=[sum(bounds)/2 for bounds in result['lie_scenarios'].values()]
        power=math.sqrt(min(candidates)*max(candidates))
    else:
        wind=reference.get('wind') or {}
        dy=number(wind.get('screen_dy'),'reference wind direction',-1,1)
        speed=number(wind.get('mean_mph'),'reference wind speed',0,100)
        slope=number(reference.get('elevation_ft'),'reference elevation',-300,300)
        adjusted_travel=(travel+slope/3)*(1+.003*dy*speed)
        if adjusted_travel<=0:
            raise ValueError('Invalid adjusted travel in bunker calibration reference')
        power=(number(reference.get('observed_meter_power'),'reference power',.025,.99)
               *result['effective_distance']/adjusted_travel)
        model='matched measured bunker layup'
        correction='scale the observed recovery power locally; adjust for measured wind and elevation'
    if not .025<=power<=.99:
        raise ValueError('Calculated bunker trial is outside the verified meter range')
    result.update(power=power,unclipped_power=power,automatic_swing_allowed=True,block_reason=None,
                  bunker_trial=[hole,stroke],model=model,correction=correction)
    if reference is not None:
        result['calibration_reference']=reference
    return result
