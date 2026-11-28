"""Install a bounded, fresh-frame solo turn recheck for WGT.

Run in C:\\Users\\rodne\\WGT_Bot, from the project's Python venv. Installs only
into context_manager.py; backs up the entire original. Does not launch WGT or send
any input. Fails closed if expected source anchors do not match.
"""
from pathlib import Path
from datetime import datetime
import ast
import shutil
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'context_manager.py'

HELPER = '''\ndef _solo_static_hud_unchanged(old, fresh, state):
    """Verify displayed solo shot identity/measurements still match new pixels.

    Dynamic course and animated meter pixels are deliberately excluded, but
    high-contrast text cores must be exactly equal. Uncertainty stops input.
    This does not confer turn permission: a full fresh ContextObserver reading
    must still pass the unchanged turn_permission policy.
    """
    if (old is None or fresh is None or old.shape != fresh.shape
            or not solo_header_layout(old) or not solo_header_layout(fresh)):
        return False
    h, w = old.shape[:2]
    crops = (
        (.830, .058, .984, .105),  # Actual player and stroke text
        (.055, .102, .109, .145),  # Hole and par
        (.012, .940, .056, .989),  # Shot mode
        (.111, .936, .172, .991),  # Club/range text
        (.823, .905, .940, .940),  # Lie and displayed lie range
        (.078, .063, .125, .095),  # Wind speed digits
        (.135, .058, .178, .095),  # Hole distance and unit
    )
    if state.get('mode') == 'PUTT':
        crops += ((.094, .145, .119, .166),)  # Green speed text
    for x1, y1, x2, y2 in crops:
        a = old[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        b = fresh[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        if not a.size or a.shape != b.shape:
            return False
        ah = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        bh = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # Compare only confidently bright, near-neutral glyph pixels; this
        # avoids scenery movement seen through translucent game backgrounds.
        am = cv2.inRange(ah, (0, 0, 160), (179, 65, 255))
        bm = cv2.inRange(bh, (0, 0, 160), (179, 65, 255))
        if not np.array_equal(am, bm):
            return False
    sign = state.get('sign')
    if not (isinstance(sign, (tuple, list)) and len(sign) == 4):
        return False
    x, y, sw, sh = (int(z) for z in sign)
    if not (0 <= x < x+sw <= w and 0 <= y < y+sh <= h):
        return False
    # A separate fresh wind-arrow reading must agree with the original
    # conditions; speed digits alone cannot establish unchanged wind.
    wind = state.get('wind') or {}
    if wind.get('high_mph', 0) > 0:
        current_vector = triangle_vector(fresh[int(.064*h):int(.097*h),
                                               int(.061*w):int(.078*w)])
        expected_vector = (wind.get('screen_dx'), wind.get('screen_dy'))
        if (current_vector is None or any(v is None for v in expected_vector)
                or np.linalg.norm(np.array(current_vector)-np.array(expected_vector)) > .20):
            return False
    # Floating target's numeric text, excluding its animated flag/arrow.
    a = old[y+round(.06*sw):y+round(.40*sw), x+round(.12*sw):x+round(.98*sw)]
    b = fresh[y+round(.06*sw):y+round(.40*sw), x+round(.12*sw):x+round(.98*sw)]
    if not a.size or a.shape != b.shape:
        return False
    ah = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    bh = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    return np.array_equal(cv2.inRange(ah, (0, 0, 160), (179, 65, 255)),
                          cv2.inRange(bh, (0, 0, 160), (179, 65, 255)))
\n'''

OLD = '''        frame,captured=self.refresh_turn_frame(safety,sct,rect,frame,current,captured)
        context=self.inspect_ready(safety,sct,rect,frame,current,captured)
        for key in ('game_context','lie','lie_percent','lie_loss_percent','verified_hazards','strategy'):
            state[key]=current.get(key)
        result=turn_permission(context,time.perf_counter())
        safety.check()
        if not result['allowed']:
            raise RuntimeError('No game input: '+', '.join(result['reasons']))
        return context'''

NEW = '''        frame,captured=self.refresh_turn_frame(safety,sct,rect,frame,current,captured)
        context=self.inspect_ready(safety,sct,rect,frame,current,captured)
        original_lie=(current.get('lie'),deepcopy(current.get('lie_percent')),
                      deepcopy(current.get('lie_loss_percent')))
        original_owner=(context.get('format'),context.get('players'),
                        context.get('active_player'),context.get('ball_owner'))
        result=turn_permission(context,time.perf_counter())
        # Full shot OCR can consume the four-second turn freshness budget.
        # Only a user-confirmed, already recognized solo owner may get a few
        # extra *fresh, independently captured and fully OCR-checked* attempts.
        # Never refresh timestamps, synthesize stable frames, or waive a check.
        transient={'turn evidence stale or absent',
                   'turn not stable across two captured frames'}
        if (not result['allowed'] and set(result['reasons']) <= transient
                and self.observer.solo_confirmed is True
                and context.get('source') == 'solo_user_confirmed_live_hud'):
            source_frame=frame
            for _ in range(3):
                safety.check()
                fresh_at=time.perf_counter()
                fresh=self.capture(sct,rect)
                if not _solo_static_hud_unchanged(source_frame,fresh,current):
                    raise RuntimeError('No game input: fresh solo HUD measurements changed or unverified')
                # Fresh observation must recheck the actual account/ready
                # controls, all conflict indicators and the independent lie.
                context=self.inspect_ready(safety,sct,rect,fresh,current,fresh_at)
                if (original_lie!=(current.get('lie'),current.get('lie_percent'),
                                  current.get('lie_loss_percent')) or
                    original_owner!=(context.get('format'),context.get('players'),
                                     context.get('active_player'),context.get('ball_owner'))):
                    raise RuntimeError('No game input: live lie or owner changed during turn refresh')
                result=turn_permission(context,time.perf_counter())
                safety.check()
                if result['allowed']:
                    break
                if not set(result['reasons']) <= transient:
                    break
        for key in ('game_context','lie','lie_percent','lie_loss_percent','verified_hazards','strategy'):
            state[key]=current.get(key)
        safety.check()
        if not result['allowed']:
            raise RuntimeError('No game input: '+', '.join(result['reasons']))
        return context'''

def check_compiled(data):
    tree = ast.parse(data)
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'require_turn']
    assert len(methods) == 1, 'Expected exactly one require_turn method'
    assert any(isinstance(n, ast.FunctionDef) and n.name == '_solo_static_hud_unchanged' for n in tree.body)
    # Assert that the important guards remain present rather than touching them.
    assert 'def arm_stroke(' in data and 'is_bunker_state(state)' in data
    assert 'turn_permission(context,time.perf_counter())' in data
    assert 'safety.check()' in data
    return tree

def main():
    if not TARGET.is_file():
        raise RuntimeError('NOT INSTALLED: context_manager.py is missing from this directory')
    source = TARGET.read_text(encoding='utf-8-sig')
    if source.count(OLD) != 1 or source.count('\nclass GameCoordinator:') != 1:
        raise RuntimeError('NOT INSTALLED: current controller differs from reviewed file; no changes made')
    updated = source.replace('\nclass GameCoordinator:', '\n' + HELPER + 'class GameCoordinator:', 1).replace(OLD,NEW,1)
    before = ast.parse(source)
    after = check_compiled(updated)
    # Verify changes are confined to the intentionally added helper and the
    # require_turn body, leaving every other definition/source statement equal.
    def stripped(tree):
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name == '_solo_static_hud_unchanged':
                tree.body.remove(n)
                break
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == 'require_turn':
                n.body = []
        return ast.dump(tree, include_attributes=False)
    assert stripped(before) == stripped(after), 'NOT INSTALLED: unexpected AST changes'
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tempfile_path=Path(td)/'context.py'
        tempfile_path.write_text(updated,encoding='utf-8')
        compile(updated,str(tempfile_path),'exec')
    backup_dir=ROOT/'automation_debug'/('backup_fresh_turn_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup_dir.mkdir(parents=True,exist_ok=False)
    shutil.copy2(TARGET,backup_dir/TARGET.name)
    TARGET.write_text(updated,encoding='utf-8')
    print('PATCH APPLIED: bounded fresh solo turn OCR retry')
    print('PRESERVED: F9, swing, accuracy, bunker guards, multiplayer checks')
    print('SAFETY: never renews old timestamps or invents stable-frame counts')
    print('BACKUP:',backup_dir)
    print('NO GAME INPUT SENT')

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('ERROR:',exc)
        sys.exit(1)
