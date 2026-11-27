"""Install read-only wind + DING trigger telemetry, in shadow mode only.

Run from WGT_Bot: python wgt_install_wind_evidence.py
Requires existing controller, wgt_learning.py, wgt_learning_data.py.
Does not start WGT, open windows, or send input. Backups made before edits.
"""
from pathlib import Path
from datetime import datetime
import ast
import shutil
import sys

ROOT = Path(__file__).resolve().parent

WIND_MODULE = '''"""Read-only WGT wind / accuracy-meter trigger evidence.

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
'''

# Require exact anchors, fail rather than make uncertain edits.
def once(text, old, new, name):
    n = text.count(old)
    if n != 1:
        raise ValueError(f'{name}: expected exactly one patch location, found {n}. No files changed.')
    return text.replace(old, new, 1)


def prepare(root):
    controller = root / 'execution_engine.py'
    data = root / 'wgt_learning_data.py'
    learning = root / 'wgt_learning.py'
    module = root / 'wgt_wind_evidence.py'
    required = [controller, data, learning]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')
    if module.exists():
        raise RuntimeError('wgt_wind_evidence.py already exists. Not overwriting; inspect manually.')
    original = {p: p.read_text(encoding='utf-8-sig') for p in required}

    c = original[controller]
    c = once(c, 'def accuracy_click(safety, sct, rect, meter, reference):',
             'def accuracy_click(safety, sct, rect, meter, reference, shot_id=None):', 'controller signature')
    c = once(c, "json.dumps(dict(target=meter.target, clicked=clicked,",
             "json.dumps(dict(shot_id=shot_id, target=meter.target, clicked=clicked,", 'trace shot ID')
    c = once(c, 'accuracy_click(safety, sct, rect, return_meter, reference)',
             "accuracy_click(safety, sct, rect, return_meter, reference,\n                               shot_id=DIAGNOSTICS.shot_id(number) if DIAGNOSTICS else None)",
             'accuracy call')
    c = once(c, "def shot_stats(frame):\n", '''def read_shot_accuracy_trace(shot_id):
    """Read only the trace tagged by the just-finished controller shot."""
    try:
        trace = json.loads((DEBUG/'accuracy_trace.json').read_text(encoding='utf-8'))
        if (trace.get('shot_id') == shot_id and trace.get('clicked') is True
                and isinstance(trace.get('samples'), list) and trace['samples']):
            return trace
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None


def shot_stats(frame):
''', 'controller trace reader')
    c = once(c, "source='controller stable result and existing shot/session logs'))",
             "source='controller stable result and existing shot/session logs',\n                                         accuracy_trace=read_shot_accuracy_trace(DIAGNOSTICS.shot_id(number))))",
             'pass current shot trace')

    d = original[data]
    d = once(d, "import math\n", "import math\n\nfrom wgt_wind_evidence import observe_wind_trigger\n", 'data import')
    d = once(d, "                evidence=evidence or {}, use_for_live_control=False)",
             "                evidence=evidence or {},\n                wind_accuracy=observe_wind_trigger(state, evidence, shot_id),\n                use_for_live_control=False)", 'data wind telemetry')

    l = original[learning]
    l = once(l, "        sync = self.db.execute(\"SELECT value FROM meta WHERE key='last_sync'\").fetchone()",
             """        wind_telemetry = [r['wind_accuracy'] for r in rows if isinstance(r.get('wind_accuracy'), dict)]
        wind_summary = dict(observed_trigger_offsets=len(wind_telemetry),
                            validated_lateral_outcomes=0,
                            learned_wind_click_policy=False,
                            live_accuracy_offset_allowed=False,
                            status='telemetry only; meter offset has no measured lateral efficacy')
        sync = self.db.execute("SELECT value FROM meta WHERE key='last_sync'").fetchone()""",
             'learning wind summary')
    l = once(l, "                    performance=trends, protocol=PROTOCOL, latest_shadow_predictions=latest,",
             "                    performance=trends, wind_accuracy=wind_summary,\n                    protocol=PROTOCOL, latest_shadow_predictions=latest,",
             'learning report inclusion')
    pending = {controller: c, data: d, learning: l, module: WIND_MODULE}
    for path, content in pending.items():
        try:
            ast.parse(content, filename=str(path))
        except SyntaxError as exc:
            raise RuntimeError(f'Patch syntax check failed for {path.name}: {exc}') from exc
    return pending, original


def main():
    try:
        pending, original = prepare(ROOT)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Back up originals before replacing any of the working project.
        backup_dir = ROOT / 'wgt_debug' / ('backup_wind_evidence_' + stamp)
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path, content in original.items():
            (backup_dir / path.name).write_text(content, encoding='utf-8')
        staged = []
        for path, content in pending.items():
            staged_path = path.with_name(path.name + '.wind_new')
            staged_path.write_text(content, encoding='utf-8')
            staged.append((staged_path, path))
        for staged_path, path in staged:
            staged_path.replace(path)
        # Offline test on installed wind observation; no imports of game input.
        from wgt_wind_evidence import observe_wind_trigger
        s = {'wind': {'low_mph': 11, 'high_mph': 13, 'screen_dx': .1, 'screen_dy': -.995}}
        trace = {'shot_id':'session:1', 'clicked':True, 'target':100,
                 'samples':[{'edge':104},{'edge':102},{'edge':99}]}
        assert observe_wind_trigger(s, {'accuracy_trace':trace}, 'session:1')['edge_side_of_ding']=='left'
        assert observe_wind_trigger(s, {'accuracy_trace':trace}, 'different:1') is None
        assert observe_wind_trigger(s, {'accuracy_trace':{**trace,'clicked':False}}, 'session:1') is None
        assert observe_wind_trigger({'wind':None}, {'accuracy_trace':trace}, 'session:1') is None
        assert observe_wind_trigger(s, None, 'session:1') is None
        assert observe_wind_trigger(s, {'accuracy_trace':{**trace,'samples':[{'edge':None}]}}, 'session:1') is None
        print('PATCH APPLIED: wind/DING trigger evidence now stored with verified shots.')
        print('OFFLINE PASS: six telemetry checks; all changed Python files compile.')
        print('SHADOW ONLY: wind offsets do not affect live accuracy clicks.')
        print('BACKUP:', backup_dir)
        print('NO GAME INPUT SENT.')
    except Exception as exc:
        print('NOT INSTALLED:', exc)
        sys.exit(1)


if __name__ == '__main__':
    main()
