"""MSS-frame context OCR, local evidence, and guarded high-level decisions.

No native input is implemented here. V48 supplies its existing input/release
callbacks. Calibration refers to a locally captured real control, not a guess.
"""
from collections import defaultdict,OrderedDict,deque
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
import re
from pathlib import Path
import time

import cv2
import numpy as np
import pytesseract

from game_state import (PLAYER, combine_context, turn_permission, swing_budget,
                      concession_decision, strategy,parse_context,GamePolicy,
                      ALTERNATE_SHOT_TURN_SECONDS)
from vision_engine import triangle_vector
from terrain_handler import BUNKER_BLOCK_REASON, bunker_calibration_plan, is_bunker_state

REGIONS={
    'match_header':(.005,.035,.995,.215),
    'players_turn':(.005,.20,.30,.88),
    'turn_header':(.67,.035,.995,.23),
    'concession_dialog':(.22,.22,.88,.83),
    'lie':(.83,.916,.92,.934),
    'current_aim_hazards':(.31,.20,.66,.45),
    'shot_stats':(.74,.16,.90,.34),
    'watch_banner':(.36,.098,.65,.145),
}
FIELD_SOURCES={
    'match_header':{'format'},
    'players_turn':{'players','teams','scores'},
    'turn_header':{'active_player','ball_owner','ball_team','seconds_left','turn_limit'},
    'concession_dialog':{'opponent_putt','conceded_putt_owner','hole_complete'},
    'lie':{'lie','lie_loss_percent','lie_percent'},
    'current_aim_hazards':{'hazards'},
    'shot_stats':{'stats_owner'},
    'watch_banner':{'turn_state'},
}


def putt_measurement_masks(frame,sign):
    """Measured HUD glyphs, excluding the animated course and green grid."""
    if sign is None or len(sign)!=4:
        return None
    h,w=frame.shape[:2];x,y,sw,sh=map(int,sign)
    if not (0<=x<x+sw<=w and 0<=y<y+sh<=h):
        return None
    regions={
        'player':(.830,.060,.900,.083),'stroke':(.898,.082,.934,.105),
        'hole_par':(.055,.103,.109,.145),'mode':(.012,.940,.056,.989),
        'putter_range':(.122,.938,.172,.966),'putter_unit':(.125,.966,.149,.989),
        'wind_speed':(.078,.063,.125,.095),'green_speed':(.094,.145,.119,.166),
        'lie':(.887,.916,.920,.934),
    }
    patches={key:frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)]
             for key,(x1,y1,x2,y2) in regions.items()}
    patches.update({
        'target_value':frame[y+int(.06*sw):y+int(.40*sw),x+int(.12*sw):x+int(.74*sw)],
        'target_unit':frame[y+int(.12*sw):y+int(.40*sw),x+int(.72*sw):x+int(.98*sw)],
        'elevation_value':frame[y+int(.50*sh):y+int(.94*sh),x+int(.33*sw):x+int(.73*sw)],
        'elevation_unit':frame[y+int(.54*sh):y+int(.96*sh),x+int(.73*sw):x+int(.94*sw)],
        'badge_top':frame[y:y+4,x+12:x+sw-12],
        'badge_bottom':frame[y+sh-4:y+sh,x+12:x+sw-12],
        'badge_left':frame[y+12:y+sh-12,x:x+4],
        'badge_right':frame[y+12:y+sh-12,x+sw-4:x+sw],
    })
    masks={}
    for key,patch in patches.items():
        if not patch.size:
            return None
        if key=='hole_par':
            gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            masks[key]=np.stack([gray<level for level in (130,175,210)])
        else:
            hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
            masks[key]=np.stack([cv2.inRange(hsv,(0,0,level),(179,35,255))
                                 for level in (130,175,210)])
    arrow=frame[y+int(.50*sh):y+int(.92*sh),x+int(.06*sw):x+int(.30*sw)]
    masks['elevation_arrow']=cv2.inRange(cv2.cvtColor(arrow,cv2.COLOR_BGR2HSV),(35,70,70),(90,255,255))
    required=('player','stroke','hole_par','mode','putter_range','putter_unit',
              'wind_speed','green_speed','target_value','target_unit','elevation_value','elevation_unit')
    if any(np.count_nonzero(masks[key])<4 for key in required):
        return None
    return masks


class ControllerLock:
    """OS-owned lock: only one live V48 controller, released even on exit."""
    def __init__(self,path):
        self.path=Path(path)
        self.stream=None

    def acquire(self):
        if self.stream is not None:
            raise RuntimeError('Controller lock is already held')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        stream=self.path.open('a+b')
        if stream.seek(0,2)==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError('Another live V48 controller is already running; no second controller started') from exc
        self.stream=stream

    def close(self):
        if self.stream is None:
            return
        stream,self.stream=self.stream,None
        try:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        finally:
            stream.close()


def solo_header_layout(frame):
    """Recognize the fixed navy solo panel, independently of player OCR.

    Calibrated from the local 1288x728 MSS solo capture. The left edge and
    bottom rule are outside the name/shot glyphs; a team roster's highlighted
    row or scenery cannot supply this full panel geometry.
    """
    image=cv2.resize(frame,(1288,728),interpolation=cv2.INTER_AREA)
    navy=np.max(np.abs(image.astype(np.int16)-np.array([71,50,23])),axis=2)<=12
    return all(navy[y1:y2,x1:x2].mean()>=.90 for x1,y1,x2,y2 in
               ((1072,44,1075,74),(1072,72,1210,74),(1190,44,1220,47)))


def text_data(patch,psm):
    if patch.size==0:
        return dict(text='',confidence=0,words=[])
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    large=cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    try:
        data=pytesseract.image_to_data(large,config=f'--psm {psm}',timeout=2,
                                      output_type=pytesseract.Output.DICT)
    except RuntimeError as exc:
        if 'timeout' not in str(exc).lower():
            raise
        return dict(text='',confidence=0,words=[],error='OCR timeout')
    lines=defaultdict(list)
    words=[]
    for i,text in enumerate(data['text']):
        if not text.strip():
            continue
        confidence=float(data['conf'][i])
        key=(data['block_num'][i],data['par_num'][i],data['line_num'][i])
        # Keep every word: dropping an uncertain digit could silently shorten
        # an opponent's putt. Low confidence invalidates the containing block.
        lines[key].append(text)
        words.append(dict(text=text,confidence=confidence,
                          rect=[int(data[k][i]/3) for k in ('left','top','width','height')]))
    # Confidence belongs to the labelled evidence, not unrelated scenery OCR.
    # Every word on a relevant line remains mandatory, including all digits.
    text='\n'.join(' '.join(v) for v in lines.values())
    relevant=[]
    for key,parts in lines.items():
        line=' '.join(parts)
        if parse_context(line):
            indices=[i for i,t in enumerate(data['text']) if t.strip()
                     and (data['block_num'][i],data['par_num'][i],data['line_num'][i])==key]
            relevant.extend(float(data['conf'][i]) for i in indices)
    return dict(text=text,confidence=min(relevant) if relevant else min((w['confidence'] for w in words),default=0),words=words)




def prepare_lie_patch(patch):
    mask=cv2.inRange(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV),(0,0,120),(179,70,255))
    large=cv2.resize(cv2.cvtColor(255-mask,cv2.COLOR_GRAY2BGR),None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(large,12,12,12,12,cv2.BORDER_CONSTANT,value=(255,255,255))


def verified_bunker_readings(readings, frame=None):
    """Verify bunker lie from the full HUD label."""

    if frame is None or not any(
        re.search(r"\bBUNKER\b",
                  str(r.get("text", "")), re.I)
        for r in readings
    ):
        return None

    h, w = frame.shape[:2]

    if abs(w - 1288) > 10 or abs(h - 728) > 10:
        return None

    patch = frame[
        round(h * .905):round(h * .940),
        round(w * .823):round(w * .940)
    ]

    if patch.size == 0:
        return None

    variants = (
        patch,
        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    )

    verified = []

    for variant in variants:
        big = cv2.resize(
            variant, None,
            fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        try:
            data = pytesseract.image_to_data(
                big,
                config="--psm 6",
                timeout=2,
                output_type=pytesseract.Output.DICT
            )
        except RuntimeError:
            return None

        words = [
            (str(t).upper().strip(), float(c))
            for t, c in zip(data["text"], data["conf"])
            if str(t).strip()
        ]

        labels = [
            confidence for text, confidence in words
            if text == "BUNKER"
        ]

        ranges = [
            (match, confidence)
            for text, confidence in words
            if (match := re.fullmatch(
                r"(\d{1,3})[-\u2013\u2014](\d{1,3})%",
                text
            ))
        ]

        if (len(labels) != 1 or
            len(ranges) != 1 or
            min(labels[0], ranges[0][1]) < 85):
            return None

        low, high = map(
            int, ranges[0][0].groups()
        )

        if not 0 < low <= high <= 100:
            return None

        verified.append((
            (low, high),
            min(labels[0], ranges[0][1]) / 100
        ))

    if verified[0][0] != verified[1][0]:
        return None

    return {
        "lie": "BUNKER",
        "lie_percent": list(verified[0][0]),
        "confidence": min(v[1] for v in verified)
    }


class ContextObserver:
    def __init__(self,root,session=None,reader=text_data,policy=None,ready_reader=None,
                 solo_confirmed=False,practice_shot_seconds=0):
        self.root=Path(root)
        self.session=session
        self.reader=reader
        # Explicit per-run selection by the user; not inferred from a lone
        # username or from the presence of the Swing button.
        self.solo_confirmed=bool(solo_confirmed)
        if practice_shot_seconds not in (0,60,90) or practice_shot_seconds and not self.solo_confirmed:
            raise ValueError('Practice shot clock requires confirmed solo play and 60 or 90 seconds')
        self.practice_shot_seconds=practice_shot_seconds
        self.solo_ready_reader=ready_reader
        self.policy=policy if policy is not None else GamePolicy.load(self.root/'wgt_calibration/game_policy.json')
        if not isinstance(self.policy,GamePolicy):
            raise ValueError('Invalid game policy')
        self.last=None
        self.last_key=None
        self.last_time=None
        self.count=0
        self.turn_deadline=None
        self.clock_turn_key=None
        self.last_image_at=float('-inf')
        self.ocr_cache=OrderedDict()
        self.turn_history=deque(maxlen=200)
        self.sequence=0
        self.regions=dict(REGIONS)
        self.control_spec=None
        self.layout_verified=False
        self.team_hud=None
        self.hud_cache=None
        profile=self.root/'wgt_calibration/game_context_profile.json'
        if profile.is_file():
            data=json.loads(profile.read_text(encoding='utf-8-sig'))
            if data.get('schema')!=1:
                raise ValueError('Invalid game context profile schema')
            for key,bounds in data.get('regions',{}).items():
                if (key not in REGIONS or len(bounds)!=4 or
                        not 0<=bounds[0]<bounds[2]<=1 or not 0<=bounds[1]<bounds[3]<=1):
                    raise ValueError('Invalid game context OCR region: '+key)
                self.regions[key]=bounds
            self.control_spec=data.get('concede_putt_control')
            reference=(self.root/data.get('reference_capture','')).resolve()
            self.layout_verified=(data.get('verified_layout') is True
                                  and reference.is_relative_to(self.root.resolve()) and reference.is_file()
                                  and hashlib.sha256(reference.read_bytes()).hexdigest()==data.get('reference_sha256'))
            if data.get('reader')=='wgt_team_hud_v1' and self.layout_verified:
                from multiplayer_hud import TeamHudReader
                self.team_hud=TeamHudReader(self.root,data,reader,ready_reader)

    def read_hud(self,frame):
        if self.team_hud is None:
            return None
        key=hashlib.sha256(frame.tobytes()).digest()
        if self.hud_cache is None or self.hud_cache[0]!=key:
            self.hud_cache=(key,self.team_hud.read(frame))
        hud=self.hud_cache[1]
        # A stored team profile describes a past game, not every future frame.
        # Only an explicit solo run plus the actual solo panel may fall back to
        # the fixed player/shot OCR. A valid or partial team HUD remains blocking.
        if self.solo_confirmed and not hud.get('valid') and solo_header_layout(frame):
            return None
        return hud

    def control(self,frame):
        spec=self.control_spec
        if not spec or spec.get('verified_control') is not True or spec.get('label')!='CONCEDE PUTT':
            return None
        template=(self.root/spec['template']).resolve()
        if not template.is_relative_to(self.root.resolve()) or not template.is_file():
            return None
        if hashlib.sha256(template.read_bytes()).hexdigest()!=spec.get('sha256'):
            return None
        sample=cv2.imread(str(template))
        if sample is None or sample.std()<8 or any(a>b for a,b in zip(sample.shape[:2],frame.shape[:2])):
            return None
        _,score,_,xy=cv2.minMaxLoc(cv2.matchTemplate(frame,sample,cv2.TM_CCOEFF_NORMED))
        if score<.95:
            return None
        x,y=xy;h,w=sample.shape[:2]
        actual=frame[y:y+h,x:x+w]
        # Normalized correlation alone can match a dim disabled button.
        # Preserve the calibrated enabled appearance as well as its label.
        if cv2.absdiff(actual,sample).mean()>8:
            return None
        texts=[self.reader(frame[y:y+h,x:x+w],psm) for psm in (6,7)]
        if any(' '.join(t['text'].upper().split())!='CONCEDE PUTT' or t['confidence']<90 for t in texts):
            return None
        return dict(label='CONCEDE PUTT',confidence=min(score,min(t['confidence'] for t in texts)/100),
                    calibrated=True,enabled=True,rect=[x,y,w,h],template_sha256=spec['sha256'])

    def observe(self,frame,state,now=None,captured_at=None):
        captured=captured_at if captured_at is not None else time.perf_counter() if now is None else now
        height,width=frame.shape[:2]
        context=dict(observed_at=captured,observed_at_utc=datetime.now(timezone.utc).isoformat(),layout_verified=self.layout_verified,
                     confidence={},raw={},stable_frames=0)
        conflicts=set()
        hud=self.read_hud(frame)
        if hud is None and self.team_hud is not None:
            context['layout_verified']=False  # The saved team layout is absent.
        if hud is not None:
            context['raw']['team_hud']=hud
            context['source']='wgt_team_hud_v1'
            context['layout_verified']=self.layout_verified and hud['valid']
            if hud['valid']:
                for key in ('format','players','teams','active_player','ball_owner','ball_team',
                            'turn_state','seconds_left','shot_label','ready_controls'):
                    if hud.get(key) is not None:
                        context[key]=hud[key]
                        context['confidence'][key]=hud['confidence']
        for region,bounds in self.regions.items():
            # The explicitly selected solo HUD has a fixed account/stroke
            # panel. Scanning scenery for labelled multiplayer facts consumes
            # the entire freshness budget and cannot establish a solo roster.
            # Current team geometry and the watching banner still veto input.
            # Optional hazard/concession/stat attribution stays unverified.
            if (self.solo_confirmed and hud is None and state.get('player') is True
                    and solo_header_layout(frame) and region not in ('lie','watch_banner')):
                continue
            if hud is not None and region in ('match_header','players_turn','turn_header','watch_banner'):
                continue
            x1,y1,x2,y2=bounds
            patch=frame[int(y1*height):int(y2*height),int(x1*width):int(x2*width)]
            modes=(6,11)
            variants=None
            if region=='lie':
                patch=prepare_lie_patch(patch)
                modes=(6,7)
            elif region=='watch_banner':
                mask=cv2.inRange(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV),(0,0,195),(179,55,255))
                contrast=cv2.cvtColor(255-mask,cv2.COLOR_GRAY2BGR)
                variants=[cv2.copyMakeBorder(image,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
                          for image in (patch,contrast)]
                modes=(7,7)
            readings=[]
            for index,psm in enumerate(modes):
                picture=variants[index] if variants is not None else patch
                cache_key=(region,picture.shape,hashlib.sha256(picture.tobytes()).digest(),psm)
                if cache_key not in self.ocr_cache:
                    observed=self.reader(picture,psm)
                    if not observed.get('error'):
                        self.ocr_cache[cache_key]=observed
                else:
                    observed=self.ocr_cache[cache_key]
                    self.ocr_cache.move_to_end(cache_key)
                readings.append(dict(observed,region=region))
                while len(self.ocr_cache)>256:
                    self.ocr_cache.popitem(last=False)
            facts=combine_context(readings,captured)
            if region == 'lie':
                # Verify essential words, not the noisy separator or suffix.
                trusted=verified_bunker_readings(readings,frame)
                if trusted is not None:
                    for field in ('lie','lie_percent'):
                        facts[field]=trusted[field]
                        facts['confidence'][field]=trusted['confidence']
            context['raw'][region]=readings
            for key,confidence in facts['confidence'].items():
                if key not in FIELD_SOURCES[region]:
                    continue
                value=facts[key]
                if key in context and context[key]!=value:
                    conflicts.add(key)
                context[key]=value
                context['confidence'][key]=confidence
        for key in conflicts:
            context.pop(key,None)
            context['confidence'].pop(key,None)
        if self.solo_confirmed:
            # User has explicitly selected SOLO for *this run*. The image must
            # independently contain the recognized local player and calibrated
            # swing controls. Any roster, team, watching banner or named
            # multiplayer format defeats that declaration. Missing evidence
            # never gives permission to send a stroke.
            raw_text='\n'.join(str(item.get('text','')) for region in
                ('match_header','players_turn','turn_header','watch_banner')
                for item in context['raw'].get(region,[]) if isinstance(item,dict)).upper()
            multi_words=bool(re.search(
                r'\b(ALTERNATE SHOT|ALT SHOT|MATCH PLAY|BEST BALL|SKINS|'
                r'WATCHING (?:TEAMMATE|OPPONENT)|PLAYERS?\s*:|TEAM\s+[A-Z0-9]+\s*:)',
                raw_text))
            conflicts_present=(
                hud is not None or multi_words
                or context.get('format') not in (None,'SOLO_STROKE')
                or bool(context.get('teams'))
                or len(context.get('players',[]))>1
                or context.get('turn_state') in ('WATCHING_TEAMMATE','WATCHING_OPPONENT')
                or context.get('active_player') not in (None,PLAYER)
            )
            header_visible=state.get('player') is True
            controls_visible=bool(self.solo_ready_reader(frame)) if (
                header_visible and not conflicts_present and
                self.solo_ready_reader is not None) else False
            evidence=dict(user_confirmed=True,player_header_visible=header_visible,
                          swing_and_meter_visible=controls_visible,
                          multiplayer_conflicts_absent=not conflicts_present)
            context['solo_evidence']=evidence
            if header_visible and controls_visible and not conflicts_present:
                context.update(format='SOLO_STROKE',players=[PLAYER],active_player=PLAYER,
                               ball_owner=PLAYER,turn_state='OWN_READY',ready_controls=True,
                               solo_session_confirmed=True,
                               source='solo_user_confirmed_live_hud')
            # Do not mark an unknown visual layout as calibrated. The verified
            # solo alternative is checked separately in turn_permission().
        key=json.dumps({k:context.get(k) for k in ('format','players','teams','active_player','ball_owner',
                        'ball_team','lie','lie_percent','lie_loss_percent',
                        'opponent_putt','turn_state','ready_controls')},sort_keys=True)
        # Include hole/stroke to prevent carrying evidence into another turn.
        key+=(str(state.get('hole')),str(state.get('shot'))).__repr__()
        previous_key=self.last_key
        self.count=self.count+1 if (key==previous_key and self.last_time is not None
                                   and 0<captured-self.last_time<=15) else 1
        # A menu/lie/readiness change or missing OCR is not a new timed turn.
        # Reset only after a complete, independently recognized turn identity.
        clock_key=(state.get('hole'),state.get('shot'),context.get('format'),
                   context.get('active_player'),context.get('ball_team') if
                   context.get('format')=='ALTERNATE_SHOT' else context.get('ball_owner'))
        if (all(value is not None for value in clock_key)
                and context.get('active_player') in context.get('players',[])
                and clock_key!=self.clock_turn_key):
            self.turn_deadline=None
            self.clock_turn_key=clock_key
        remaining=context.get('seconds_left')
        alternate=context.get('format')=='ALTERNATE_SHOT'
        if alternate:
            # Configuration supplied by the player, not inferred OCR confidence.
            context['turn_limit']=ALTERNATE_SHOT_TURN_SECONDS
            context['turn_limit_source']='user-confirmed alternate-shot team session'
        if remaining is not None and (not alternate or 0<=remaining<=ALTERNATE_SHOT_TURN_SECONDS):
            candidate=captured+remaining
            if self.turn_deadline is not None and candidate>self.turn_deadline+2:
                context.pop('seconds_left',None)
                context['confidence'].pop('seconds_left',None)
            else:
                self.turn_deadline=min(self.turn_deadline,candidate) if self.turn_deadline is not None else candidate
        if (self.practice_shot_seconds and context.get('format')=='SOLO_STROKE'
                and context.get('source')=='solo_user_confirmed_live_hud'
                and context.get('turn_state')=='OWN_READY' and all(v is not None for v in clock_key)):
            context['turn_limit']=self.practice_shot_seconds
            context['turn_limit_source']='simulated solo practice clock; not a visible multiplayer timer'
            context['practice_clock']=True
            # Preserve the capture before OCR or measurement revalidation. A
            # reread, club change or temporarily missing identity never renews it.
            candidate=min(captured,state.get('captured_at',captured))+self.practice_shot_seconds
            if self.turn_deadline is None:
                self.turn_deadline=candidate
                print(f'PRACTICE CLOCK: {self.practice_shot_seconds}s from first captured ready HUD '
                      f'(hole {state.get("hole")}, shot {state.get("shot")}); OCR time counts',flush=True)
                if self.session:
                    self.session.event('practice_clock_started',hole=state.get('hole'),shot=state.get('shot'),
                                       deadline=candidate,seconds=self.practice_shot_seconds)
        if self.turn_deadline is not None:
            context['timer_deadline']=self.turn_deadline
        context['stable_frames']=self.count
        self.last_key,self.last_time=key,captured
        context['concede_control']=self.control(frame)
        context['permission']=turn_permission(context,time.perf_counter() if now is None else now)
        context['concession']=concession_decision(context,context['concede_control'],
                                                 time.perf_counter() if now is None else now,
                                                 max_feet=self.policy.concession_max_feet)
        context['policy']=self.policy.as_dict()
        turn=dict(hole=state.get('hole'),shot=state.get('shot'),player=context.get('active_player'),
                  team=context.get('teams',{}).get(context.get('active_player')))
        if (self.count>=2 and context.get('active_player') in context.get('players',[])
                and context.get('format') and (not self.turn_history or self.turn_history[-1]!=turn)):
            self.turn_history.append(turn)
        context['observed_turn_order']=list(self.turn_history)
        self.last=context
        self.sequence+=1
        if self.session:
            stamp=f'context_{self.sequence:05d}'
            self.session.state(stamp,context)
            if key!=previous_key or captured-self.last_image_at>=5:
                self.session.image(stamp,frame)
                self.session.image(stamp+'_annotated',self.annotate(frame,context))
                self.last_image_at=captured
        return context

    def annotate(self,frame,context):
        image=frame.copy()
        h,w=image.shape[:2]
        for region,bounds in self.regions.items():
            x1,y1,x2,y2=bounds
            a,b=(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h))
            cv2.rectangle(image,a,b,(0,180,255),1)
            cv2.putText(image,region,(a[0],max(12,a[1])),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,180,255),1)
        label='OUR TURN VERIFIED' if context['permission']['allowed'] else 'INPUT BLOCKED: '+', '.join(context['permission']['reasons'])
        cv2.putText(image,label[:155],(12,h-55),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
        control=context.get('concede_control')
        if control:
            x,y,cw,ch=control['rect']
            cv2.rectangle(image,(x,y),(x+cw,y+ch),(0,255,0),2)
        return image



def _solo_static_hud_unchanged(old, fresh, state, elevation_reader=None, distance_reader=None):
    """Verify displayed solo shot identity/measurements still match new pixels.

    Dynamic course and animated meter pixels are deliberately excluded, but
    High-contrast text cores must agree. Reread the value and unit of a
    changed target row, and the value, unit and direction of elevation.
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
    if not np.array_equal(cv2.inRange(ah, (0, 0, 160), (179, 65, 255)),
                          cv2.inRange(bh, (0, 0, 160), (179, 65, 255))):
        if distance_reader is None or state.get('distance_source') != 'floating_target':
            return False
        distance = distance_reader(fresh, sign)
        if distance is None or (distance.value, distance.unit) != (state.get('distance'), state.get('unit')):
            return False
    # Elevation value/unit and its direction are measurements too. A matching
    # distance row alone cannot renew a stale shot with changed elevation.
    a = old[y+round(.46*sh):y+sh, x:x+sw]
    b = fresh[y+round(.46*sh):y+sh, x:x+sw]
    ah = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    bh = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    if (np.array_equal(cv2.inRange(ah, (0, 0, 160), (179, 65, 255)),
                       cv2.inRange(bh, (0, 0, 160), (179, 65, 255)))
            and np.array_equal(cv2.inRange(ah, (35, 70, 70), (90, 255, 255)),
                               cv2.inRange(bh, (35, 70, 70), (90, 255, 255)))):
        return True
    return (elevation_reader is not None and state.get('mode') in ('FULL','PITCH','CHIP')
            and state.get('elevation_ft') is not None
            and elevation_reader(fresh,sign)==state['elevation_ft'])

class GameCoordinator:
    """Input boundaries around existing V48 controls; unknown turns wait."""
    def __init__(self,observer,capture,read_state,move_to,session=None,target_visible=None,policy=None,attempt_store=None,putt_reader=None,read_state_from_source=None,elevation_reader=None,distance_reader=None):
        self.observer=observer
        self.capture=capture
        self.read_state=read_state
        self.putt_reader=putt_reader
        self.read_state_from_source=read_state_from_source
        self.elevation_reader=elevation_reader
        self.distance_reader=distance_reader
        self.move_to=move_to
        self.session=session
        self.pending_concessions=set()
        self.last_state=None
        self.target_visible=target_visible
        self.completed=False
        candidate=getattr(observer,'policy',None)
        self.policy=policy if policy is not None else candidate if isinstance(candidate,GamePolicy) else GamePolicy()
        if not isinstance(self.policy,GamePolicy):
            raise ValueError('Invalid game policy')
        self.attempt_store=Path(attempt_store) if attempt_store is not None else None
        self.durable_pending={}
        if self.attempt_store is not None and self.attempt_store.is_file():
            data=json.loads(self.attempt_store.read_text(encoding='utf-8-sig'))
            if not isinstance(data,dict) or data.get('schema')!=1 or not isinstance(data.get('pending'),dict):
                raise ValueError('Invalid unresolved-concession journal; no game input')
            self.durable_pending=data['pending']
            self.pending_concessions.update(self.durable_pending)

    def _save_pending(self):
        if self.attempt_store is None:
            return
        self.attempt_store.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.attempt_store.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(schema=1,pending=self.durable_pending),indent=2),encoding='utf-8')
        temporary.replace(self.attempt_store)

    def _putt_key(self,context,state,owner):
        teams=context.get('teams',{})
        groups={team:sorted(player for player in teams if teams[player]==team) for team in set(teams.values())}
        ball=('team',groups.get(context.get('ball_team'),[])) if context.get('format')=='ALTERNATE_SHOT' else ('player',owner)
        return json.dumps(dict(format=context.get('format'),match=context.get('match_id'),
                               players=sorted(context.get('players',[])),teams=sorted(groups.values()),
                               hole=state.get('hole'),shot=state.get('shot'),ball=ball),sort_keys=True)

    def inspect(self,frame,state,captured_at=None):
        started = time.perf_counter()
        state['game_context'] = self.observer.observe(frame, state, captured_at=captured_at)
        print(f"OBSERVE TIME: {time.perf_counter() - started:.3f}s", flush=True)
        context=state['game_context']
        state['lie']=context.get('lie')
        state['lie_percent']=context.get('lie_percent')
        state['lie_loss_percent']=context.get('lie_loss_percent')
        state['verified_hazards']=context.get('hazards',[])
        state['strategy']=strategy(state)
        self.last_state=state
        return context

    def inspect_ready(self,safety,sct,rect,frame,state,captured_at):
        context=self.inspect(frame,state,captured_at=captured_at)
        context=self.revalidate_putt(safety,sct,rect,frame,state,context)
        if context.get('practice_clock') and time.perf_counter()>=context['timer_deadline']:
            raise RuntimeError(f'{context["turn_limit"]}-second practice shot clock expired; no swing')
        return context

    def revalidate_putt(self,safety,sct,rect,frame,state,context,attempts=1):
        """Bind slow putt OCR to a fresh, independently ready matching HUD.

        The four-second limit is unchanged. Every measured glyph mask, the
        account/stroke panel, fresh wind vector and live controls must agree.
        No prior value is renewed when its visible label changes or vanishes.
        """
        evidence=context.get('solo_evidence') or {}
        if (self.observer.solo_confirmed is not True or state.get('mode')!='PUTT'
                or context.get('source')!='solo_user_confirmed_live_hud'
                or context.get('format')!='SOLO_STROKE' or state.get('player') is not True
                or not all(evidence.get(k) is True for k in ('player_header_visible',
                    'swing_and_meter_visible','multiplayer_conflicts_absent'))
                or not state.get('wind') or not solo_header_layout(frame)
                or any(state.get(k) is None for k in ('distance','unit','club_range','club_unit',
                                                      'elevation_ft','green_speed'))):
            return context
        if time.perf_counter()-context['observed_at']<=1:
            return context
        masks=putt_measurement_masks(frame,state.get('sign'))
        if masks is None:
            return context
        badge_fields={'target_value','target_unit','elevation_value','elevation_unit','elevation_arrow',
                      'badge_top','badge_bottom','badge_left','badge_right'}
        semantic_badge=False
        for attempt in range(attempts):
            safety.check()
            captured=time.perf_counter()
            fresh=self.capture(sct,rect)
            if fresh.shape!=frame.shape or not solo_header_layout(fresh):
                return context
            current=putt_measurement_masks(fresh,state.get('sign'))
            if current is None:
                return context
            changed=[key for key in masks if not np.array_equal(masks[key],current[key])]
            if not changed:
                break
            if self.putt_reader is None or not set(changed)<=badge_fields:
                if self.session:
                    self.session.event('putt_revalidation_refused',changed_fields=changed)
                return context
            decoded=self.putt_reader(fresh,state.get('sign'))
            if decoded is not None and all(decoded.get(k)==state.get(k)
                                           for k in ('distance','unit','elevation_ft')):
                semantic_badge=True
                break
            if self.session:
                self.session.event('putt_revalidation_refused',changed_fields=changed)
            # Camera animation can hide a glyph for a frame. A readable
            # disagreement still ends this proof and is never retried here.
            if decoded is not None:
                context['putt_measurements_changed']=True
                return context
            if attempt+1<attempts:
                safety.wait(.25)
        else:
            return context
        h,w=fresh.shape[:2]
        vector=triangle_vector(fresh[int(.064*h):int(.097*h),int(.061*w):int(.078*w)])
        wind=state['wind']
        if vector is None:
            if wind['high_mph']!=0:
                return context
            vector=(0.,0.)
        if np.linalg.norm(np.array(vector)-np.array([wind['screen_dx'],wind['screen_dy']]))>.20:
            return context
        if self.observer.read_hud(fresh) is not None:
            return context
        if self.observer.solo_ready_reader is None or not self.observer.solo_ready_reader(fresh):
            return context
        safety.check()
        if time.perf_counter()-captured>4:
            return context
        original=context['observed_at']
        if context.get('seconds_left') is not None:
            # Renewing measurement evidence never renews shot-clock time.
            deadline=min(context.get('timer_deadline',float('inf')),original+context['seconds_left'])
            context['timer_deadline']=deadline
            tracked=getattr(self.observer,'turn_deadline',None)
            self.observer.turn_deadline=min(tracked,deadline) if tracked is not None else deadline
        context['ocr_frame_at']=original
        context['observed_at']=captured
        self.observer.last_time=captured  # Renew time, never the stable-frame count.
        context['observed_at_utc']=datetime.now(timezone.utc).isoformat()
        context['putt_revalidation']=dict(original_captured_at=original,captured_at=captured,
            matching_fields=list(masks),fresh_controls=True,fresh_wind_vector=list(vector),
            badge_semantically_reread=semantic_badge,unreadable_badge_retries=attempt)
        context['permission']=turn_permission(context,time.perf_counter())
        if self.session:
            self.session.event('putt_hud_revalidated',**context['putt_revalidation'])
            self.session.state(f'putt_live_{self.observer.sequence:05d}',context,fresh)
        return context

    def refresh_turn_frame(self,safety,sct,rect,frame,state,captured):
        """Re-capture after slow shot OCR without renewing old pixel evidence.

        Compare identical measurement crops or, for the calibrated solo HUD,
        verify its visible glyphs and reread a changed elevation row. The
        moving timer and turn permission use the actual new captured frame.
        """
        if (self.observer.team_hud is None and not self.observer.solo_confirmed) or time.perf_counter()-captured<=1:
            return frame,captured
        safety.check()
        stamp=time.perf_counter()
        fresh=self.capture(sct,rect)
        if fresh.shape!=frame.shape:
            return frame,captured
        h,w=frame.shape[:2]
        bounds=[(.005,.05,.18,.18),(.008,.93,.175,.989),(.83,.916,.92,.934)]
        if self.observer.solo_confirmed:
            bounds += [(.83,.058,.984,.104),(.80,.55,.99,.88),(.055,.10,.12,.15),
                       (.01,.94,.06,.99),(.05,.02,.15,.12)]
        sign=state.get('sign')
        if sign is None:
            return frame,captured
        x,y,sw,sh=sign
        identical=(np.array_equal(frame[y:y+sh,x:x+sw],fresh[y:y+sh,x:x+sw])
                   and all(np.array_equal(frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)],
                                          fresh[int(y1*h):int(y2*h),int(x1*w):int(x2*w)])
                           for x1,y1,x2,y2 in bounds))
        if not identical:
            if not (self.observer.solo_confirmed is True and self.elevation_reader is not None
                    and state.get('player') is True and state.get('mode') in ('FULL','PITCH','CHIP')
                    and _solo_static_hud_unchanged(frame,fresh,state,self.elevation_reader,self.distance_reader)):
                return frame,captured
        if self.observer.team_hud is not None:
            previous=self.observer.read_hud(frame)
            current=self.observer.read_hud(fresh)
            if previous is None and current is None and self.observer.solo_confirmed:
                pass  # Both independently match the calibrated solo panel.
            elif previous is None or current is None or not previous.get('valid') or not current.get('valid') or any(
                    previous.get(k)!=current.get(k) for k in
                    ('active_player','players','teams','turn_state','shot_label','ready_controls')):
                return frame,captured
        safety.check()
        if self.session:
            self.session.event('measurement_pixels_revalidated',captured_at=stamp,
                               original_captured_at=captured,input_sent=False,
                               proof='identical_crops' if identical else 'verified_solo_hud')
        return fresh,stamp

    def require_turn(self,safety,sct,rect,state,*,strict_conditions=False):
        safety.check()
        required=('hole','shot','distance','unit','mode','club_range','club_unit','club_name','elevation_ft')
        source=state.get('distance_source')
        pinned=(state.get('mode')=='PUTT' and self.read_state_from_source is not None
                and source in ('floating_target','verified_hole_hud'))
        if state.get('mode')=='PUTT':
            required+=('green_speed',)
        if pinned:
            required+=('distance_source',)
        measurement_keys=('distance','unit','elevation_ft','distance_source')
        for attempt in range(3):
            safety.check()
            captured=time.perf_counter()
            frame=self.capture(sct,rect)
            current=(self.read_state_from_source(frame,source) if pinned else self.read_state(frame))
            current['captured_at']=captured
            changed=[key for key in required if current.get(key) is None or current[key]!=state.get(key)]
            if not changed:
                break
            # This is a read-only reacquisition before any stroke input, never
            # a retry of a consumed/uncertain shot. Exact equality is retained.
            reread=(pinned and self.observer.solo_confirmed is True
                    and current.get('player') is True and source=='floating_target'
                    and set(changed)<= {'distance','unit'}
                    and all(current.get(key) is None for key in changed) and attempt<2)
            if self.session:
                self.session.state(f'preinput_distance_{time.time_ns()}',dict(
                    attempt=attempt+1,changed=changed,requested_source=source,
                    planned=state,reread=current,read_again=reread),frame)
                self.session.shot_stage('shot_measurements_rejected',hole=state.get('hole'),
                                        shot=state.get('shot'),changed=changed,read_again=reread,
                                        expected={key:state.get(key) for key in measurement_keys},
                                        observed={key:current.get(key) for key in measurement_keys})
            if not reread:
                raise RuntimeError('Shot state changed or became unreadable before input: '+changed[0])
            safety.wait(.15)
        if state.get('wind') and not current.get('wind'):
            raise RuntimeError('Wind became unreadable before input; re-observe before swinging')
        if strict_conditions and current.get('spin_indicator')!=state.get('spin_indicator'):
            raise RuntimeError('Spin changed before the bunker calibration stroke; replan before swinging')
        if state.get('wind') and current.get('wind'):
            if (abs(state['wind']['mean_mph']-current['wind']['mean_mph'])>2
                    or np.linalg.norm(np.array([state['wind']['screen_dx'],state['wind']['screen_dy']])-
                                      np.array([current['wind']['screen_dx'],current['wind']['screen_dy']]))>.20):
                raise RuntimeError('Wind changed before input; re-observe before swinging')
        frame,captured=self.refresh_turn_frame(safety,sct,rect,frame,current,captured)
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
                and context.get('source') == 'solo_user_confirmed_live_hud'
                and not context.get('putt_measurements_changed')):
            if current.get('mode')=='PUTT':
                context=self.revalidate_putt(safety,sct,rect,frame,current,context,attempts=3)
                result=turn_permission(context,time.perf_counter())
            else:
                source_frame=frame
                for _ in range(3):
                    safety.check()
                    fresh_at=time.perf_counter()
                    fresh=self.capture(sct,rect)
                    if not _solo_static_hud_unchanged(source_frame,fresh,current,self.elevation_reader,self.distance_reader):
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
        if context.get('putt_measurements_changed'):
            raise RuntimeError('No game input: the fresh putt badge disagrees with the planned measurements')
        for key in ('game_context','lie','lie_percent','lie_loss_percent','verified_hazards','strategy'):
            state[key]=current.get(key)
        safety.check()
        if not result['allowed']:
            raise RuntimeError('No game input: '+', '.join(result['reasons']))
        if strict_conditions:
            state['wind']=current['wind']
        if self.session:
            self.session.shot_stage('shot_measurements_revalidated',hole=state.get('hole'),
                                    shot=state.get('shot'),observation_retries=attempt,
                                    expected={key:state.get(key) for key in measurement_keys},
                                    observed={key:current.get(key) for key in measurement_keys})
        return context

    def wait_for_turn(self,safety,sct,rect,timeout=300):
        started=time.perf_counter()
        report_at=started-6
        solo_hint_shown=False
        while time.perf_counter()-started<timeout:
            safety.check()
            captured=time.perf_counter()
            frame=self.capture(sct,rect)
            state=self.read_state(frame)
            state['captured_at']=captured
            frame,captured=self.refresh_turn_frame(safety,sct,rect,frame,state,captured)
            context=self.inspect_ready(safety,sct,rect,frame,state,captured)
            if turn_permission(context,time.perf_counter())['allowed']:
                return state,frame
            if context['concession']['eligible']:
                self.concede(safety,sct,rect,state)
                if self.completed:
                    return dict(complete=True),frame
            if self.session and time.perf_counter()-report_at>=6:
                self.session.event('waiting_for_verified_turn',context=context)
                report_at=time.perf_counter()
            print('WAITING FOR VERIFIED TURN: '+', '.join(context['permission']['reasons']),flush=True)
            if (not solo_hint_shown and self.observer.solo_confirmed is False
                    and not context.get('format') and solo_header_layout(frame)):
                print('Solo HUD detected. If this is a solo round, restart with --solo-confirmed.',
                      flush=True)
                solo_hint_shown=True
            safety.wait(.25)
        raise RuntimeError('Game format/roster/our active turn not verified before turn-wait limit; no swing')

    def arm_stroke(self,safety,sct,rect,state,reference,*,bunker_trial=None):
        # A failed later OCR read must not erase already observed sand evidence.
        if is_bunker_state(state) and bunker_trial is None:
            raise RuntimeError(BUNKER_BLOCK_REASON)
        planned_lie=state.get('lie')
        planned_percentages=deepcopy((state.get('lie_percent'),state.get('lie_loss_percent')))
        planned_strategy=deepcopy(state.get('strategy'))
        context=(self.require_turn(safety,sct,rect,state,strict_conditions=True) if bunker_trial is not None
                 else self.require_turn(safety,sct,rect,state))
        if (planned_lie!=state.get('lie') or planned_strategy!=state.get('strategy')
                or planned_percentages!=(state.get('lie_percent'),state.get('lie_loss_percent'))):
            raise RuntimeError('Lie or strategy changed before stroke; re-observe before swinging')
        if bunker_trial is not None:
            if (context.get('format')!='SOLO_STROKE' or context.get('players')!=[PLAYER]
                    or context.get('active_player')!=PLAYER or context.get('ball_owner')!=PLAYER):
                raise RuntimeError('Bunker calibration requires the verified solo player and ball owner')
            fresh=bunker_calibration_plan(state,*bunker_trial['bunker_trial'],
                                          reference=bunker_trial.get('calibration_reference'))
            previous_inputs=bunker_trial['observed_inputs']
            if any(value!=previous_inputs[key] for key,value in fresh['observed_inputs'].items() if key!='wind'):
                raise RuntimeError('Bunker calibration inputs changed; replan before swinging')
            if self.session:
                self.session.event('bunker_trial_final_power',planned=bunker_trial['power'],
                                   calculated=fresh['power'],wind=state['wind'])
            bunker_trial.update(fresh)
        elif is_bunker_state(state):
            if state.get('lie')!='BUNKER' or not state.get('lie_percent'):
                raise RuntimeError('Bunker visible but lie or percentage unverified; no swing')
            raise RuntimeError(BUNKER_BLOCK_REASON)
        worst=max(9,reference.drag_seconds*1.6)+min(6,max(3.5,reference.accuracy_delay*2))+1
        budget=swing_budget(context,time.perf_counter(),stroke_seconds=worst)
        if not budget['allowed']:
            raise RuntimeError('No swing: '+', '.join(budget['reasons']))
        x=rect['left']+round(reference.anchor[0]*rect['width'])
        y=rect['top']+round(reference.anchor[1]*rect['height'])
        if self.target_visible is None or not self.target_visible(safety.hwnd,x,y):
            raise RuntimeError('Another window may cover the swing-meter anchor; no swing')
        previous=safety.deadline
        if budget['deadline'] is not None:
            safety.deadline=min(previous,budget['deadline']) if previous is not None else budget['deadline']
        if self.session:
            self.session.event('verified_turn_stroke_budget',budget=budget)
        return previous

    def concede(self,safety,sct,rect,state):
        captured=time.perf_counter()
        frame=self.capture(sct,rect)
        current=self.read_state(frame)
        if any(current.get(k) is None or current.get(k)!=state.get(k) for k in ('hole','shot')):
            return False
        context=self.inspect(frame,state,captured_at=captured)
        control=context.get('concede_control')
        decision=concession_decision(context,control,time.perf_counter(),max_feet=self.policy.concession_max_feet)
        if not decision['eligible']:
            return False
        key=self._putt_key(context,state,decision['owner'])
        if state.get('hole') is None or state.get('shot') is None or key in self.pending_concessions:
            return False
        x,y,w,h=control['rect']
        if not 0<=x<x+w<=frame.shape[1] or not 0<=y<y+h<=frame.shape[0]:
            raise RuntimeError('Concession control is outside the captured WGT window')
        tx,ty=rect['left']+x+w//2,rect['top']+y+h//2
        if self.target_visible is None or not self.target_visible(safety.hwnd,tx,ty):
            raise RuntimeError('Concession control is covered by another window; no click')
        self.move_to(tx,ty)
        safety.check()
        if not self.target_visible(safety.hwnd,tx,ty):
            raise RuntimeError('Concession control became covered; no click')
        # Mark before sending input: an exception or an unknown UI outcome must
        # never retry a possibly consumed concession.
        self.pending_concessions.add(key)
        self.durable_pending[key]=dict(owner=decision['owner'],feet=decision['feet'],
                                       threshold_feet=decision['threshold_feet'],
                                       attempted_at_utc=datetime.now(timezone.utc).isoformat())
        self._save_pending()  # A persistence failure stops before mouse-down.
        try:
            safety.press()
            safety.wait(.015)
        finally:
            safety.release()
        if self.session:
            self.session.event('concede_putt_input_sent',decision=decision,control=control,
                               outcome_verified=False)
        # Verify an explicit owner-bound result twice; a sent click or a
        # disappearing button is not a successful concession.
        successes=0
        completion_count=0
        end=time.perf_counter()+12
        while time.perf_counter()<end:
            safety.wait(.25)
            captured=time.perf_counter()
            frame=self.capture(sct,rect)
            after=self.read_state(frame)
            observed=self.inspect(frame,after,captured_at=captured)
            fresh=0<=time.perf_counter()-observed.get('observed_at',float('-inf'))<=4
            same_game=(observed.get('layout_verified') is True
                       and all(observed.get(k)==context.get(k) for k in ('format','players','teams')))
            recorded=(fresh and same_game and observed.get('conceded_putt_owner')==decision['owner']
                      and observed.get('confidence',{}).get('conceded_putt_owner',0)>=.85)
            successes=successes+1 if recorded else 0
            completion_count=completion_count+1 if recorded and observed.get('hole_complete') else 0
            if successes>=2:
                self.completed=completion_count>=2
                self.durable_pending.pop(key,None)
                self._save_pending()
                if self.session:
                    self.session.event('concession_verified',decision=decision)
                return True
        raise RuntimeError('Concede Putt input sent once; explicit result unverified; no automatic retry')
