"""Fix redundant putting-readiness gating in WGT v48, without granting input.

Usage (from C:\\Users\\rodne\\WGT_Bot):
    & '.\\.venv\\Scripts\\python.exe' '.\\wgt_fix_putt_ready.py'

Changes ONLY stable_state() in execution_engine.py. Fails closed if the
reviewed source pattern is absent. Backups are retained in wgt_debug.
"""

import ast
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'execution_engine.py'

REPLACEMENTS = [
    (
        """        fresh_putt=False
        if solo_putt:
            context=GAME.inspect_ready(safety,sct,rect,frame,state,captured_at)
            fresh_putt=(context.get('permission',{}).get('allowed') is True
                        and 0<=time.perf_counter()-context['observed_at']<=4)
""",
        """        if solo_putt:
            # Preserve full live context observation and the shot-clock guard.
            # This phase verifies the shot measurements; it must not depend on
            # permission becoming fresh during lengthy OCR. require_turn()
            # independently verifies ownership immediately before input.
            GAME.inspect_ready(safety,sct,rect,frame,state,captured_at)
""",
    ),
    (
        """        if solo_putt and not fresh_putt:
            problems.append('fresh verified putt HUD')
""",
        """""",
    ),
    (
        """            # Two consecutive complete putt readings each have independent
            # fresh glyph/control proof, even when OCR itself takes >8s.
            matching_window=timeout if fresh_putt else 8.0
""",
        """            # Match TWO independently captured, fully decoded putting
            # states even if OCR is slow. This is HUD agreement, not input
            # authorization. require_turn() still supplies the fresh owner gate.
            matching_window=timeout if solo_putt else 8.0
""",
    ),
]


def node_invariants(tree):
    candidates=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='stable_state']
    if len(candidates)!=1:
        raise RuntimeError('Expected exactly one stable_state')
    code=ast.unparse(candidates[0])
    for marker in ('safety.check()', 'readiness_problems(', 'state_signature(',
                   'safety.wait(', 'GAME.inspect_ready(', 'GAME.observer.solo_confirmed'):
        if marker not in code:
            raise RuntimeError('Required HUD safety condition missing: '+marker)


def only_stable_state_changed(before,after):
    def scrub(tree):
        for item in ast.walk(tree):
            if isinstance(item,ast.FunctionDef) and item.name=='stable_state':
                item.body=[]
        return ast.dump(tree,include_attributes=False)
    return scrub(before)==scrub(after)


def main():
    if not TARGET.exists():
        raise RuntimeError('NOT INSTALLED: execution_engine.py is missing')
    old=TARGET.read_text(encoding='utf-8-sig')
    updated=old
    for before,after in REPLACEMENTS:
        if updated.count(before)!=1:
            if 'matching_window=timeout if solo_putt else 8.0' in old:
                raise RuntimeError('NOT INSTALLED: this fix appears already installed')
            raise RuntimeError('NOT INSTALLED: controller differs from reviewed version; no change made')
        updated=updated.replace(before,after,1)
    old_tree=ast.parse(old)
    new_tree=ast.parse(updated)
    node_invariants(new_tree)
    if not only_stable_state_changed(old_tree,new_tree):
        raise RuntimeError('NOT INSTALLED: unexpected program change')
    compile(updated,str(TARGET),'exec')
    # Independently assert the final two input boundaries remain present.
    for marker in ('GAME.require_turn(safety,sct,rect,state)',
                   'context=self.require_turn(safety,sct,rect,state)',
                   'if is_bunker_state(state):', 'F9 = 0x78'):
        if marker not in updated and marker not in (ROOT/'context_manager.py').read_text(encoding='utf-8-sig'):
            raise RuntimeError('NOT INSTALLED: safety boundary is missing: '+marker)
    backup=ROOT/'wgt_debug'/('backup_putt_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(TARGET,backup/TARGET.name)
    temporary=TARGET.with_name(TARGET.name+'.putt_tmp')
    try:
        temporary.write_text(updated,encoding='utf-8')
        compile(temporary.read_text(encoding='utf-8'),str(temporary),'exec')
        temporary.replace(TARGET)
    finally:
        if temporary.exists():
            temporary.unlink()
    print('PUTTING READINESS FIX APPLIED')
    print('Now requires two matching complete putt HUD readings;')
    print('fresh ownership and turn are still mandatory before any swing.')
    print('PRESERVED: F9, meter control, DING, bunker guard, shot clock.')
    print('BACKUP:',backup)
    print('NO GAME INPUT SENT')


if __name__=='__main__':
    try:
        if sys.argv[1:] not in ([], ['--putt-now']):
            raise RuntimeError('Use without arguments or with --putt-now')
        main()
        if sys.argv[1:] == ['--putt-now']:
            print('STARTING SUPERVISED SINGLE-SHOT WGT ATTEMPT',flush=True)
            print('Must already be in a solo game on the putting green; F9 stops input.',flush=True)
            command = [sys.executable, '-B', '-u', str(TARGET),
                       '--one-shot', '--solo-confirmed', '--holes', '1',
                       '--max-shots', '1', '--max-seconds', '150',
                       '--activate', '--countdown', '5', '--concede-feet', '0']
            sys.exit(subprocess.call(command,cwd=ROOT))
    except Exception as exc:
        print('ERROR:',exc)
        sys.exit(1)
