"""Read-only wind / accuracy-meter trigger evidence.

This DOES NOT learn an effective offset or move the DING click. The controller's
existing meter trace records the green edge at the moment the click is TRIGGERED;
that is not proof of the exact mouse-down position or ball trajectory.
"""
import math


def _number(value, low, high):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high)


def observe_wind_trigger(state, evidence, shot_id):
    """Return a labeled, untrained telemetry record or None if unverified."""
    if not isinstance(evidence, dict):
        return None
    trace = evidence.get('accuracy_trace')
    wind = (state or {}).get('wind')
    if not isinstance(trace, dict) or not isinstance(wind, dict):
        return None
    if trace.get('shot_id') != shot_id or trace.get('clicked') is not True:
        return None
    lo, hi = wind.get('low_mph'), wind.get('high_mph')
    dx, dy = wind.get('screen_dx'), wind.get('screen_dy')
    if (not _number(lo, 0, 50) or not _number(hi, 0, 50) or lo > hi
            or not _number(dx, -1, 1) or not _number(dy, -1, 1)
            or (hi > 0 and not .9 <= math.hypot(dx, dy) <= 1.05)):
        return None
    target = trace.get('target')
    samples = trace.get('samples')
    if not _number(target, 0, 2000) or not isinstance(samples, list) or not samples:
        return None
    last = samples[-1]
    if not isinstance(last, dict) or not _number(last.get('edge'), 0, 2000):
        return None
    edge = last['edge']
    offset = round(edge - target, 3)
    side = 'left' if offset < 0 else 'right' if offset > 0 else 'center'
    # More than two valid, genuinely changing edges are required for direction.
    edges = [sample.get('edge') for sample in samples[-16:]
             if isinstance(sample, dict) and _number(sample.get('edge'), 0, 2000)]
    movement = 'unknown'
    changes = [b-a for a, b in zip(edges, edges[1:]) if abs(b-a) > 0.5]
    if len(changes) >= 2 and all(v > 0 for v in changes):
        movement = 'right'
    elif len(changes) >= 2 and all(v < 0 for v in changes):
        movement = 'left'
    return {
        'source': 'verified_current_shot_accuracy_trace',
        'wind_mph': [lo, hi], 'wind_screen_dx': dx, 'wind_screen_dy': dy,
        'ding_target_x_px': target, 'observed_green_edge_x_at_trigger_px': edge,
        'edge_minus_ding_px': offset, 'edge_side_of_ding': side,
        'recent_meter_movement': movement,
        'intentional_wind_offset': False,
        'exact_mouse_down_location_verified': False,
        'lateral_ball_miss_verified': False,
        'eligible_to_train_wind_offset': False,
        'learning_status': 'telemetry_only; effectiveness unmeasured',
    }
