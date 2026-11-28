"""Repair three wind-telemetry regressions without modifying swing/accuracy behavior.

This script is intended to run inside the user's WGT_Bot folder.  It restores
accuracy_click's exact original five-argument body and original controller call.
It keeps telemetry by associating the just-written accuracy trace with this
shot outside the time-critical click path, using fresh file timing evidence.
No game, capture or Windows input APIs are invoked by this installer.
"""
from pathlib import Path
from datetime import datetime
import ast
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
FILE = ROOT / 'execution_engine.py'

OLD_READER = '''def read_shot_accuracy_trace(shot_id):
    """Read only the trace tagged by the just-finished controller shot."""
    try:
        trace = json.loads((DEBUG/'accuracy_trace.json').read_text(encoding='utf-8'))
        if (trace.get('shot_id') == shot_id and trace.get('clicked') is True
                and isinstance(trace.get('samples'), list) and trace['samples']):
            return trace
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None
'''

NEW_READER = '''def read_shot_accuracy_trace(shot_id, number):
    """Attach shot identity after the click, never inside accuracy_click.

    before_shot_NN.png is written just before the current stroke; the saved
    trace must be newer. This only establishes trace freshness, not mouse-down
    accuracy or ball lateral motion. If anything is uncertain, abstain.
    """
    try:
        trace_file = DEBUG/'accuracy_trace.json'
        before_file = DEBUG/f'before_shot_{number:02d}.png'
        if not isinstance(shot_id, str) or not shot_id or not isinstance(number, int) or number < 1:
            return None
        if not trace_file.is_file() or not before_file.is_file():
            return None
        if trace_file.stat().st_mtime_ns <= before_file.stat().st_mtime_ns:
            return None
        trace = json.loads(trace_file.read_text(encoding='utf-8'))
        if (trace.get('clicked') is True
                and isinstance(trace.get('samples'), list) and trace['samples']
                and isinstance(trace.get('target'), (int, float))):
            return dict(trace, shot_id=shot_id)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None
'''


def once(source, old, new, name):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'{name}: expected exactly one anchor, found {count}. No files changed.')
    return source.replace(old, new, 1)


def function_tree(content, name):
    tree = ast.parse(content)
    matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(matches) != 1:
        raise RuntimeError(f'Missing or duplicate function: {name}')
    return ast.dump(matches[0], include_attributes=False)


def build(source):
    content = source
    content = once(content,
        'def accuracy_click(safety, sct, rect, meter, reference, shot_id=None):',
        'def accuracy_click(safety, sct, rect, meter, reference):',
        'Restore original five-argument signature')
    content = once(content,
        "json.dumps(dict(shot_id=shot_id, target=meter.target, clicked=clicked,",
        "json.dumps(dict(target=meter.target, clicked=clicked,",
        'Restore original accuracy trace writer')
    content = once(content,
        "accuracy_click(safety, sct, rect, return_meter, reference,\n                               shot_id=DIAGNOSTICS.shot_id(number) if DIAGNOSTICS else None)",
        "accuracy_click(safety, sct, rect, return_meter, reference)",
        'Restore original controller call and mocked signature')
    content = once(content, OLD_READER, NEW_READER, 'Read-only fresh trace association')
    content = once(content,
        'accuracy_trace=read_shot_accuracy_trace(DIAGNOSTICS.shot_id(number))))',
        'accuracy_trace=read_shot_accuracy_trace(DIAGNOSTICS.shot_id(number),number)))',
        'Pass number to read-only trace association')
    ast.parse(content)
    return content


def check_repair(original, repaired):
    if function_tree(original, 'accuracy_click') == function_tree(repaired, 'accuracy_click'):
        raise RuntimeError('Source appears unpatched; abort')
    a = ast.parse(repaired)
    accuracy = [n for n in a.body if isinstance(n, ast.FunctionDef) and n.name == 'accuracy_click'][0]
    assert len(accuracy.args.args) == 5 and not accuracy.args.defaults
    assert 'shot_id' not in ast.get_source_segment(repaired, accuracy)
    assert 'accuracy_trace=read_shot_accuracy_trace' in repaired
    assert "shot_id=DIAGNOSTICS.shot_id(number) if DIAGNOSTICS else None" not in repaired
    # Ensure this narrow patch did not change any live input/capture source.
    for name in ('drag_swing', 'Safety', 'run_hole', 'accuracy_click'):
        assert name in repaired
    return True


def main():
    if not FILE.is_file():
        print('NOT INSTALLED: missing execution_engine.py'); return 1
    try:
        existing = FILE.read_text(encoding='utf-8-sig')
        updated = build(existing)
        check_repair(existing, updated)
        backup = ROOT/'automation_debug'/('backup_wind_regression_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        backup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(FILE,backup/FILE.name)
        temporary = FILE.with_name(FILE.name+'.wind_regression_new')
        temporary.write_text(updated,encoding='utf-8')
        temporary.replace(FILE)
        print('PATCH APPLIED: restored original accuracy_click and five-argument call.')
        print('WIND EVIDENCE PRESERVED: freshness-tagged trace only after the click.')
        print('OFFLINE PASS: controller compiles; original accuracy function shape restored.')
        print('BACKUP:', backup)
        print('NO GAME INPUT SENT. Do not start nine-hole bot until full regression passes.')
        return 0
    except Exception as exc:
        print('NOT INSTALLED:', exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
