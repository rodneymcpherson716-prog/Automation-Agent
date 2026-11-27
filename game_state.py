"""Verified game/turn and sportsmanship policies. No screen or input APIs.

UI labels below are evidence contracts, not claims about uncalibrated application
layouts. Unknown, conflicting or stale evidence never grants input permission.
"""
from collections import Counter
from dataclasses import dataclass,asdict
import json
import math
from pathlib import Path
import re

PLAYER='tigerwould52'
FORMATS={'SOLO_STROKE','MATCH_PLAY','ALTERNATE_SHOT','STROKE_PLAY','BEST_BALL','SKINS'}
MULTIPLAYER=FORMATS-{'SOLO_STROKE'}
LIVE_FORMATS=FORMATS-{'BEST_BALL'}  # Label recognized; application-specific rules not verified yet.
CONCESSIONS={'MATCH_PLAY','ALTERNATE_SHOT'}
ALTERNATE_SHOT_TURN_SECONDS=60  # User-confirmed team-session limit.


def concession_limit(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Concession threshold must be a finite, nonnegative number of feet')
    return float(value)


@dataclass(frozen=True)
class GamePolicy:
    concession_max_feet:float=2.0

    def __post_init__(self):
        object.__setattr__(self,'concession_max_feet',concession_limit(self.concession_max_feet))

    @classmethod
    def load(cls,path,override=None):
        values={}
        path=Path(path)
        if path.is_file():
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            if (not isinstance(data,dict) or data.get('schema')!=1
                    or set(data)-{'schema','concession_max_feet'}):
                raise ValueError('Invalid game policy schema or unrecognized setting')
            if 'concession_max_feet' not in data:
                raise ValueError('Game policy is missing concession_max_feet')
            values['concession_max_feet']=data['concession_max_feet']
        if override is not None:
            values['concession_max_feet']=override
        return cls(**values)

    def as_dict(self):
        return asdict(self)


def name(value):
    value=str(value or '').strip().lower()
    return value if re.fullmatch(r'[a-z0-9_]{2,32}',value) else None


def agreed(values):
    """Both OCR variants must agree; one plausible reading is insufficient."""
    if len(values)<2 or any(value is None for value in values):
        return None
    return values[0] if all(value==values[0] for value in values) else None


def parse_context(text,player=PLAYER):
    """Parse labelled facts; never turn a profile name or Swing into a turn."""
    result={}
    conflicts=set()
    def assign(key,value):
        if key in result and result[key]!=value:
            conflicts.add(key)
        result[key]=value
    formats=[]
    players=[]
    teams={}
    active=[]
    for line in text.upper().splitlines():
        line=line.strip()
        for label,value in (('SOLO STROKE PLAY','SOLO_STROKE'),('SINGLE PLAYER','SOLO_STROKE'),
                            ('ALTERNATE SHOT','ALTERNATE_SHOT'),('ALT SHOT','ALTERNATE_SHOT'),
                            ('MATCH PLAY','MATCH_PLAY'),('BEST BALL','BEST_BALL'),('SKINS','SKINS')):
            if re.fullmatch(r'(?:FORMAT\s*:\s*)?'+label,line):
                formats.append(value)
        if re.fullmatch(r'(?:FORMAT\s*:\s*)?STROKE PLAY',line):
            formats.append('STROKE_PLAY')
        m=re.fullmatch(r'(?:ACTIVE PLAYER|TURN|CURRENT PLAYER)\s*:\s*([A-Z0-9_]+)',line)
        if m and name(m[1]):
            active.append(name(m[1]))
        m=re.fullmatch(r"([A-Z0-9_]+)'S TURN",line)
        if m and name(m[1]):
            active.append(name(m[1]))
        if line=='YOUR TURN':
            active.append(name(player))
        m=re.fullmatch(r'WATCHING\s+(TEAMMATE|OPPONENT)',line)
        if m:
            assign('turn_state','WATCHING_'+m[1])
        if line in ('HOLE COMPLETE','NEXT HOLE','HOLED OUT'):
            assign('hole_complete',True)
        m=re.fullmatch(r'PLAYERS?\s*:\s*([A-Z0-9_, ]+)',line)
        if m:
            entries=[name(x) for x in re.split(r'[, ]+',m[1]) if x]
            if all(entries):
                players.extend(entries)
        m=re.fullmatch(r'TEAM ([A-Z0-9]+)\s*:\s*([A-Z0-9_, ]+)',line)
        if m:
            entries=[name(x) for x in re.split(r'[, ]+',m[2]) if x]
            if all(entries):
                players.extend(entries)
                for who in entries:
                    teams[who]=m[1]
        m=re.fullmatch(r'(?:BALL OWNER|BALL)\s*:\s*([A-Z0-9_]+)',line)
        if m:
            assign('ball_owner',name(m[1]))
        m=re.fullmatch(r'SHARED BALL\s*:\s*TEAM ([A-Z0-9]+)',line)
        if m:
            assign('ball_team',m[1])
        m=re.fullmatch(r'(?:TIME LEFT|TIME REMAINING|SHOT TIME LEFT)\s*:\s*(?:(\d):)?(\d{1,2})(?:\s*(?:S|SEC|SECONDS))?',line)
        if m:
            seconds=int(m[2])+60*int(m[1] or 0)
            if 0<=seconds<=90:
                assign('seconds_left',seconds)
        m=re.fullmatch(r'TURN LIMIT\s*:\s*(30|60|90)(?:\s*(?:S|SEC|SECONDS))?',line)
        if m:
            assign('turn_limit',int(m[1]))
        m=re.fullmatch(r'(?:LIE\s*:\s*)?(TEE(?: BOX)?|FAIRWAY|ROUGH|HEAVY ROUGH|BUNKER|SAND|GREEN|FRINGE)(?:\s+\((\d{1,3})\s*[-\u2013\u2014]\s*(\d{1,3})%\)|\s*\|\s*(\d{1,3})\s*[-\u2013\u2014]\s*(\d{1,3})%\s*(?:LIE|UE)?)?',line)
        if m:
            assign('lie',{'SAND':'BUNKER','TEE BOX':'TEE'}.get(m[1],m[1]))
            lo,hi=(m[2],m[3]) if m[2] else (m[4],m[5])
            if lo and 0<int(lo)<=int(hi)<=100:
                # A displayed LIE is a label, not evidence of power loss.
                assign('lie_percent',[int(lo),int(hi)])
        m=re.fullmatch(r'(?:SCORE|TEAM ([A-Z0-9]+) SCORE)\s*:\s*(E|[+-]\d{1,2})',line)
        if m:
            result.setdefault('scores',{})[m[1] or 'player']=m[2]
        m=re.fullmatch(r'OPPONENT PUTT\s*:\s*([A-Z0-9_]+)\s+(\d{1,3}(?:\.\d{1,2})?)\s*(FT|FEET|YDS?)',line)
        if m and name(m[1]):
            assign('opponent_putt',dict(owner=name(m[1]),feet=float(m[2])*(3 if m[3].startswith('YD') else 1)))
        m=re.fullmatch(r'PUTT CONCEDED\s*:\s*([A-Z0-9_]+)',line)
        if m and name(m[1]):
            assign('conceded_putt_owner',name(m[1]))
        m=re.fullmatch(r'SHOT STATS PLAYER\s*:\s*([A-Z0-9_]+)',line)
        if m and name(m[1]):
            assign('stats_owner',name(m[1]))
        m=re.fullmatch(r'LIE LOSS\s*:\s*(\d{1,2})\s*[-\u2013\u2014]\s*(\d{1,2})%',line)
        if m and 0<=int(m[1])<=int(m[2])<100:
            assign('lie_loss_percent',[int(m[1]),int(m[2])])
        m=re.fullmatch(r'CURRENT AIM (WATER|BUNKER)\s*:\s*(\d{1,3}(?:\.\d+)?)\s*[-–]\s*(\d{1,3}(?:\.\d+)?)\s*YDS?',line)
        if m and 0<float(m[2])<float(m[3])<=650:
            result.setdefault('hazards',[]).append(dict(kind=m[1],start=float(m[2]),end=float(m[3]),unit='yd',current_aim=True))
    unique=set(formats)
    if len(unique)==1:
        result['format']=unique.pop()
    if len(set(active))==1:
        result['active_player']=active[0]
    if players:
        result['players']=sorted(set(players))
    if teams:
        result['teams']=teams
    for key in conflicts:
        result.pop(key,None)
    return result


def combine_context(readings,now):
    """Conservative per-field agreement, with source confidence and age."""
    parsed=[parse_context(r['text']) for r in readings]
    result=dict(observed_at=now,confidence={},raw=readings)
    if len(readings)!=2 or any(r.get('confidence',0)<85 for r in readings):
        return result
    for key in set().union(*(p.keys() for p in parsed)):
        value=agreed([p.get(key) for p in parsed])
        if value is not None:
            result[key]=value
            result['confidence'][key]=min(r['confidence'] for r in readings)/100
    return result


def turn_permission(context,now,player=PLAYER,max_age=4):
    why=[]
    player=name(player)
    age=now-context.get('observed_at',float('-inf'))
    if not 0<=age<=max_age:
        why.append('turn evidence stale or absent')
    # Explicit SOLO mode is a separate, user-attested path. We cannot claim an
    # uncalibrated layout was calibrated; instead require live player OCR, live
    # swing/meter evidence, and absence of conflicting multiplayer indicators.
    # This never grants permission in multiplayer, nor on a stale/single frame.
    solo_evidence = context.get('solo_evidence') or {}
    solo_ready = (
        context.get('source') == 'solo_user_confirmed_live_hud'
        and context.get('solo_session_confirmed') is True
        and context.get('format') == 'SOLO_STROKE'
        and context.get('players') == [player]
        and context.get('active_player') == player
        and context.get('ball_owner') == player
        and context.get('ready_controls') is True
        and solo_evidence.get('player_header_visible') is True
        and solo_evidence.get('swing_and_meter_visible') is True
        and solo_evidence.get('multiplayer_conflicts_absent') is True
    )
    if context.get('layout_verified') is not True and not solo_ready:
        why.append('solo HUD unverified' if solo_evidence else 'WGT format/turn layout not calibrated')
    if context.get('stable_frames',0)<2:
        why.append('turn not stable across two captured frames')
    if context.get('turn_state') in ('WATCHING_TEAMMATE','WATCHING_OPPONENT'):
        why.append(context['turn_state'].lower().replace('_',' ')+'; our swing is forbidden')
    if context.get('source')=='wgt_team_hud_v1' and (
            context.get('turn_state')!='OWN_READY' or context.get('ready_controls') is not True):
        why.append('our ready swing controls are not verified in the multiplayer HUD')
    fmt=context.get('format')
    roster=context.get('players',[])
    active=context.get('active_player')
    teams=context.get('teams',{})
    required=['format','players','active_player']
    required+=['teams','ball_team'] if fmt=='ALTERNATE_SHOT' else ['ball_owner']
    if fmt in MULTIPLAYER:
        required.append('seconds_left')
    # A user-confirmed SOLO session has a different evidentiary basis from a
    # calibrated multiplayer OCR profile. Do not forge OCR confidence values.
    if not solo_ready and any(context.get('confidence',{}).get(key,0)<.85 for key in required):
        why.append('required turn facts lack verified OCR confidence')
    if fmt not in FORMATS:
        why.append('game format unverified')
    elif fmt not in LIVE_FORMATS:
        why.append('format rules not verified for automatic play')
    if player not in roster or len(roster)!=len(set(roster)):
        why.append('player roster unverified')
    if active!=player:
        why.append('active player is not verified as Tigerwould52')
    if fmt=='SOLO_STROKE' and roster!=[player]:
        why.append('solo roster contains another player')
    if fmt in MULTIPLAYER and len(roster)<2:
        why.append('multiplayer roster incomplete')
    if fmt=='MATCH_PLAY' and len(roster)!=2 or fmt in ('STROKE_PLAY','SKINS') and not 2<=len(roster)<=4:
        why.append('roster size disagrees with WGT format rules')
    if fmt in ('ALTERNATE_SHOT','BEST_BALL'):
        if (len(roster)!=4 or set(teams)!=set(roster) or len(set(teams.values()))!=2
                or sorted(Counter(teams.values()).values())!=[2,2]):
            why.append('two complete teams are not verified')
    if fmt=='ALTERNATE_SHOT':
        if not teams.get(player) or context.get('ball_team')!=teams.get(player):
            why.append('shared ball does not belong to our team')
    elif context.get('ball_owner')!=player:
        why.append('our ball ownership is unverified')
    if fmt in MULTIPLAYER and context.get('seconds_left') is None:
        why.append('turn timer unreadable')
    if fmt=='ALTERNATE_SHOT' and context.get('seconds_left') is not None:
        seconds=context['seconds_left']
        if (isinstance(seconds,bool) or not isinstance(seconds,(int,float))
                or not math.isfinite(seconds) or not 0<=seconds<=ALTERNATE_SHOT_TURN_SECONDS):
            why.append('timer conflicts with confirmed 60-second alternate-shot limit')
    return dict(allowed=not why,reasons=why,format=fmt,active_player=active,
                teammate=next((p for p in roster if p!=player and teams.get(p) and teams.get(p)==teams.get(player)),None),
                opponents=[p for p in roster if p!=player and (not teams or teams.get(p)!=teams.get(player))])


def swing_budget(context,now,stroke_seconds=12.,margin=3.):
    permission=turn_permission(context,now)
    if not permission['allowed']:
        return dict(allowed=False,reasons=permission['reasons'])
    remaining=context.get('seconds_left')
    if remaining is None:
        tracked=context.get('timer_deadline')
        if tracked is None:
            if context.get('turn_limit') is not None:
                return dict(allowed=False,deadline=None,remaining=None,
                            reasons=['shot clock is configured but time remaining is unverified'])
            return dict(allowed=True,deadline=None,remaining=None)
        deadline=tracked-margin
        usable=deadline-now
        return dict(allowed=usable>=stroke_seconds,deadline=deadline,remaining=usable,
                    reasons=[] if usable>=stroke_seconds else ['insufficient verified turn time for a complete stroke'])
    # OCR time consumes real time; never restart a countdown on each read.
    deadline=min(context['observed_at']+remaining,context.get('timer_deadline',float('inf')))-margin
    usable=deadline-now
    return dict(allowed=usable>=stroke_seconds,deadline=deadline,remaining=usable,
                reasons=[] if usable>=stroke_seconds else ['insufficient verified turn time for a complete stroke'])


def concession_decision(context,control,now,player=PLAYER,max_feet=2.0):
    why=[]
    try:
        limit=concession_limit(max_feet)
    except ValueError:
        limit=None
        why.append('invalid concession threshold')
    age=now-context.get('observed_at',float('-inf'))
    fmt=context.get('format')
    putt=context.get('opponent_putt') or {}
    opponent=putt.get('owner')
    roster=context.get('players',[])
    teams=context.get('teams',{})
    required=['format','players','active_player','ball_owner','opponent_putt','lie']
    if fmt=='ALTERNATE_SHOT':
        required+=['teams','ball_team']
    if any(context.get('confidence',{}).get(key,0)<.85 for key in required):
        why.append('required concession facts lack verified OCR confidence')
    if not 0<=age<=4:
        why.append('opponent evidence stale')
    if context.get('layout_verified') is not True:
        why.append('WGT concession/turn layout not calibrated')
    if context.get('stable_frames',0)<2:
        why.append('opponent putt not stable across two captured frames')
    if context.get('turn_state')=='WATCHING_TEAMMATE':
        why.append('watching teammate; no opponent-putt concession')
    if fmt not in CONCESSIONS:
        why.append('format does not verify putt concessions')
    if fmt=='MATCH_PLAY' and len(roster)!=2:
        why.append('match-play roster incomplete or conflicting')
    if opponent not in roster or name(player) not in roster or opponent==name(player):
        why.append('opposing ball owner unverified')
    if fmt=='ALTERNATE_SHOT':
        if (len(roster)!=4 or set(teams)!=set(roster) or sorted(Counter(teams.values()).values())!=[2,2] or not teams.get(opponent)
                or not teams.get(name(player)) or teams[opponent]==teams[name(player)]
                or context.get('ball_team')!=teams[opponent]):
            why.append('opposing team/shared ball unverified')
    if context.get('active_player')!=opponent or context.get('ball_owner')!=opponent:
        why.append('opponent active putt/ball ownership unverified')
    feet=putt.get('feet')
    if (isinstance(feet,bool) or not isinstance(feet,(int,float)) or not math.isfinite(feet)
            or limit is None or not 0<feet<=limit):
        why.append('opponent putt exceeds the configured threshold or distance is unverified')
    if limit==0:
        why.append('automatic concessions disabled by zero threshold')
    if context.get('conceded_putt_owner')==opponent and opponent is not None:
        why.append('this putt is already recorded as conceded')
    if context.get('lie')!='GREEN':
        why.append('ball on the putting green unverified')
    rect=control.get('rect',[]) if control else []
    if (not control or control.get('label')!='CONCEDE PUTT' or len(rect)!=4 or control.get('enabled') is not True
            or control.get('confidence',0)<.90 or not control.get('calibrated')):
        why.append('actual Concede Putt control uncalibrated or unreadable')
    return dict(eligible=not why,reasons=why,owner=opponent,feet=feet,threshold_feet=limit,
                action='concede_putt' if not why else 'wait',format=fmt)


def strategy(state):
    """Plan along the existing aim. Never infer water from a blue sky pixel."""
    result=dict(target=state.get('distance'),tee_full_power=False,aim_adjustment=False,
                hazards_verified=False,lie_verified=state.get('lie') is not None,reasons=[])
    tee=(state.get('shot')==1 and state.get('mode')=='FULL' and state.get('lie')=='TEE')
    result['tee_full_power']=tee
    hazards=state.get('verified_hazards',[])
    relevant=[h for h in hazards if h.get('unit')=='yd' and h.get('current_aim') is True and h.get('kind') in ('WATER','BUNKER')
              and isinstance(h.get('start'),(int,float)) and isinstance(h.get('end'),(int,float))
              and 0<h['start']<h['end']<=650]
    result['hazards_verified']=bool(relevant)
    if state.get('unit')=='yd' and relevant:
        for hazard in sorted(relevant,key=lambda h:h['start']):
            if hazard['start']<=state['distance']<=hazard['end']:
                result['target']=max(.5,hazard['start']-10.)
                result['reasons'].append('fixed-aim lay-up before verified '+hazard['kind'].lower())
                break
    if not relevant:
        result['reasons'].append('hazard corridor unverified; no hazard-avoidance claim')
    if tee:
        result['reasons'].append('user policy: 100% on a verified first tee shot')
    return result
