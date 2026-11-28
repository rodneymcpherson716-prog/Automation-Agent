"""One supervised recovery attempt for the observed hole-11 rough position.

The live trial failed: the longer iron carried 121.09 yards, short of the
130-yard target. Live execution is disabled. The estimate remains here only
as a reproducible diagnostic; it is not a fitted distance model.
"""
import math
import sys

import execution_engine as live
from vision_engine import PowerCalibration


def recovery_plan(original,state,for_selection=False):
    if (state.get('hole')!=11 or state.get('shot')!=7 or state.get('mode')!='FULL'
            or state.get('unit')!='yd' or not 129.5<=state.get('distance',0)<=130.5):
        raise RuntimeError('Recovery is restricted to the verified hole-11 seventh-shot position')
    context=state.get('game_context',{})
    readings=context.get('raw',{}).get('lie',[])
    if context.get('format')!='SOLO_STROKE' or len(readings)!=2:
        raise RuntimeError('Recovery requires the independently verified solo rough HUD')
    for reading in readings:
        words=reading.get('words',[])
        if not all(any(str(word.get('text','')).upper()==label and word.get('confidence',0)>=90
                       for word in words) for label in ('CUT','ROUGH')):
            raise RuntimeError('Recovery rough label changed or became unreadable')
    wind=state.get('wind') or {}
    if not (14<=wind.get('mean_mph',0)<=16 and
            math.hypot(wind.get('screen_dx',0)-.9029,wind.get('screen_dy',0)+.4299)<=.20):
        raise RuntimeError('Wind no longer matches the two observed short carries')
    plan=dict(original)
    # Two actual 7I/143-yard shots at about .885 power carried 107.88 and
    # 111.57 yards. Their mean carry / nominal response is approximately .867.
    # Use .86 as a local estimate, with four yards of carry beyond the target.
    # This does not interpret the displayed lie percentage as a loss percentage.
    effective=(state['distance']+4.)/.86
    requested=effective/state['club_range']
    plan.update(power=min(.99,requested),unclipped_power=requested,
                effective_distance=effective,model='single-position supervised rough recovery estimate',
                correction='local observed carry response estimate .86; four-yard carry margin',
                practice_only=True,clipped_to_full=requested>.99)
    return plan


def check():
    sample=dict(hole=11,shot=7,mode='FULL',unit='yd',distance=130.,club_range=143.,
                game_context=dict(format='SOLO_STROKE',raw=dict(lie=[dict(words=[
                    dict(text='CUT',confidence=94),dict(text='ROUGH',confidence=94)])]*2)),
                wind=dict(mean_mph=15.,screen_dx=.9029,screen_dy=-.4299))
    assert recovery_plan({},sample)['unclipped_power']>1.
    assert .80<recovery_plan({},dict(sample,club_range=187.))['power']<.85
    for changed in (dict(sample,shot=8),dict(sample,hole=12),dict(sample,distance=131.),
                    dict(sample,game_context={}),dict(sample,wind={})):
        try:
            recovery_plan({},changed)
        except RuntimeError:
            continue
        raise AssertionError('Mismatched recovery position was accepted')
    print('Recovery scope checks passed; no game input.')


if __name__=='__main__':
    if sys.argv[1:]==['--check']:
        check()
    else:
        raise SystemExit('Live recovery estimate disabled after the short-carry trial; no game input.')
