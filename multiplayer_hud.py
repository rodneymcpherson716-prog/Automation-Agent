"""Multiplayer HUD geometry and calibrated roster glyphs. No input APIs."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import re
import cv2
import numpy as np

BASE=(1288,728)


def runs(values,gap=2):
    return np.split(values,np.flatnonzero(np.diff(values)>gap)+1) if len(values) else []


def roster_rows(frame):
    image=cv2.resize(frame,BASE,interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    b,g,r=[image[:,:,i].astype(float) for i in range(3)]
    green=((hsv[:,:,0]>=40)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>=80)&(g>85)&(g>r*1.3)&(g>b*1.1))
    gold=((hsv[:,:,0]>=12)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=100)&(r>110)&(g>60)&(r>g*1.15)&(g>b*1.4))
    x1,x2=1125,1228
    y1,y2=41,157
    counts=(green|gold)[y1:y2,x1:x2].sum(axis=1)
    groups=[row+y1 for row in runs(np.flatnonzero(counts>=3)) if 4<=len(row)<=16]
    white=(hsv[:,:,1]<65)&(hsv[:,:,2]>190)
    white_counts=white[36:170,966:1256].sum(axis=1)
    highlights=[]
    for group in runs(np.flatnonzero(white_counts>=100)):
        if not 18<=len(group)<=40:
            continue
        top,bottom=int(group[0])+36,int(group[-1])+37
        if white[top:bottom,1110:1233].mean()>=.55:
            highlights.append((top,bottom))
    if len(highlights)!=1:
        return image,[],dict(reason='highlighted player panel missing or ambiguous',highlights=highlights)
    top,bottom=highlights[0]
    rows=[]
    for group in groups:
        lo,hi=int(group[0])-1,int(group[-1])+2
        if top<=lo<bottom and lo>top+17:
            continue  # Tier/sponsor marks below the active player's name.
        totals=[int(mask[lo:hi,x1:x2].sum()) for mask in (green,gold)]
        color=int(np.argmax(totals))
        if max(totals)<20 or max(totals)/max(1,sum(totals))<.80:
            continue
        mask=(green,gold)[color][lo:hi,x1:x2]
        xs=np.flatnonzero(mask.any(axis=0))
        if not len(xs):
            continue
        left,right=x1+int(xs[0]),x1+int(xs[-1])+1
        glyph=(green,gold)[color][lo:hi,left:right].astype(np.uint8)*255
        rows.append(dict(team='GREEN' if color==0 else 'GOLD',rect=[left,lo,right-left,hi-lo],
                         glyph=glyph,active=top<=float(group.mean())<bottom))
    if (len(rows)!=4 or sum(row['active'] for row in rows)!=1
            or sorted(row['team'] for row in rows)!=['GOLD','GOLD','GREEN','GREEN']):
        return image,[],dict(reason='complete four-player green/gold roster unverified',
                              candidates=[dict(team=row['team'],rect=row['rect'],active=row['active']) for row in rows])
    return image,rows,dict(highlight=[top,bottom],reason=None)


def normalized_glyph(glyph):
    ys,xs=np.nonzero(glyph)
    if len(xs)<15:
        return None
    tight=glyph[ys.min():ys.max()+1,xs.min():xs.max()+1]
    return cv2.resize(tight,(192,32),interpolation=cv2.INTER_NEAREST)


class TeamHudReader:
    """Read a reviewed green/gold roster; AS is score, never a format cue."""
    def __init__(self,root,spec,reader,ready_reader):
        self.root=Path(root).resolve()
        self.reader=reader
        self.ready_reader=ready_reader
        self.spec=spec
        self.templates=[]
        self.cache=OrderedDict()
        if spec.get('format')!='ALTERNATE_SHOT' or spec.get('reader')!='wgt_team_hud_v1':
            raise ValueError('Unknown multiplayer HUD calibration')
        for item in spec.get('roster_templates',[]):
            path=(self.root/item['file']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Roster calibration image missing')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Roster calibration image changed')
            glyph=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            normalized=normalized_glyph(glyph) if glyph is not None else None
            if normalized is None:
                raise ValueError('Roster glyph calibration is blank')
            self.templates.append(dict(item,image=normalized,ratio=glyph.shape[1]/glyph.shape[0]))
        members={item['player']:item['team'] for item in self.templates}
        if (len(members)!=4 or sorted(members.values())!=['GOLD','GOLD','GREEN','GREEN']
                or 'tigerwould52' not in members):
            raise ValueError('Complete reviewed four-player roster required')
        self.members=members
        self.shot_templates=[]
        for item in spec.get('shot_templates',[]):
            path=(self.root/item['file']).resolve()
            if (not path.is_relative_to(self.root) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']):
                raise ValueError('Shot-label calibration missing or changed')
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if (image is None or image.shape!=(25,30) or image.std()<8 or item['label'] not in
                    ('2ND SHOT','FOR BIRDIE','FOR PAR','FOR BOGEY')):
                raise ValueError('Unknown shot-label calibration')
            self.shot_templates.append((item['label'],image))

    def _ocr(self,image,psm):
        key=(image.shape,hashlib.sha256(image.tobytes()).digest(),psm)
        if key not in self.cache:
            result=self.reader(image,psm)
            if result.get('error'):
                return result
            self.cache[key]=result
        self.cache.move_to_end(key)
        while len(self.cache)>128:
            self.cache.popitem(last=False)
        return self.cache[key]

    def _identify(self,row):
        glyph=normalized_glyph(row['glyph'])
        if glyph is None:
            return None
        ratio=row['glyph'].shape[1]/row['glyph'].shape[0]
        scores={}
        for item in self.templates:
            if item['team']!=row['team'] or item['active']!=row['active']:
                continue
            if not .85<=ratio/item['ratio']<=1.15:
                continue
            score=float(cv2.matchTemplate(glyph,item['image'],cv2.TM_CCOEFF_NORMED)[0,0])
            scores[item['player']]=max(score,scores.get(item['player'],-1))
        ranked=sorted(scores.items(),key=lambda value:value[1],reverse=True)
        if not ranked or ranked[0][1]<.94 or len(ranked)>1 and ranked[0][1]-ranked[1][1]<.08:
            return None
        return dict(player=ranked[0][0],correlation=ranked[0][1])

    def _banner(self,image):
        patch=image[71:106,464:837]
        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(0,0,195),(179,55,255))
        variants=(patch,cv2.cvtColor(255-mask,cv2.COLOR_GRAY2BGR))
        results=[]
        raw=[]
        for candidate in variants:
            candidate=cv2.copyMakeBorder(candidate,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
            reading=self._ocr(candidate,7)
            text=' '.join(reading['text'].upper().split())
            raw.append(dict(text=text,confidence=reading['confidence']))
            results.append(text if reading['confidence']>=85 and text in ('WATCHING TEAMMATE','WATCHING OPPONENT') else None)
        return results[0].replace(' ','_') if results[0] is not None and results[0]==results[1] else None,raw

    @staticmethod
    def _dash(image):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(0,0,140),(179,70,255))
        ys,xs=np.nonzero(mask)
        if not len(xs) or len(xs)>20 or ys.max()-ys.min()>2:
            return False
        groups=runs(np.unique(xs),gap=1)
        return len(groups)==2 and all(1<=len(g)<=4 for g in groups)

    def _clock_and_label(self,image,rows,meta):
        top,bottom=meta['highlight']
        # Exclude the panel's left edge, colon and lower white border.
        clock=image[top+7:bottom-5,976:1013]
        hsv=cv2.cvtColor(clock,cv2.COLOR_BGR2HSV)
        white=((hsv[:,:,1]<65)&(hsv[:,:,2]>190)).mean()
        seconds=None
        raw=[]
        if white>=.6:
            for psm in (6,7):
                value=self._ocr(cv2.copyMakeBorder(clock,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255)),psm)
                text=value['text'].strip()
                match=re.fullmatch(r'[:;]?\s*(\d{1,2})',text)
                raw.append(dict(text=text,confidence=value['confidence']))
            values=[int(re.sub(r'[^0-9]','',value['text'])) if re.fullmatch(r'[:;]?\s*\d{1,2}',value['text'])
                    and value['confidence']>=85 else None for value in raw]
            if len(values)==2 and values[0] is not None and values[0]==values[1] and 0<=values[0]<=90:
                seconds=values[0]
        label=image[top+3:bottom-3,1024:1058]
        text_label=None
        label_raw=[]
        if self._dash(image[top+3:bottom-6,1028:1053]):
            inactive=[row for row in rows if not row['active']]
            if all(self._dash(image[max(0,row['rect'][1]-2):row['rect'][1]+12,1095:1107]) for row in inactive):
                text_label='NOT_STARTED'
        else:
            hsv=cv2.cvtColor(label,cv2.COLOR_BGR2HSV)
            label=cv2.cvtColor(255-cv2.inRange(hsv,(0,0,170),(179,70,255)),cv2.COLOR_GRAY2BGR)
            label=cv2.copyMakeBorder(label,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
            values=[]
            for psm in (6,11):
                value=self._ocr(label,psm)
                text=' '.join(value['text'].upper().split())
                label_raw.append(dict(text=text,confidence=value['confidence']))
                values.append(text if value['confidence']>=85 and re.fullmatch(r'(?:\d{1,2}(?:ST|ND|RD|TH) SHOT|FOR (?:EAGLE|BIRDIE|PAR|BOGEY|DOUBLE BOGEY|TRIPLE BOGEY))',text) else None)
            if values[0] is not None and values[0]==values[1]:
                text_label=values[0]
            if text_label is None:
                # Full reviewed label, including its ordinal suffix and SHOT.
                patch=image[top+3:top+28,1026:1056]
                mask=cv2.inRange(cv2.cvtColor(patch,cv2.COLOR_BGR2HSV),(0,0,170),(179,70,255))
                for name,template in self.shot_templates:
                    score=float(cv2.matchTemplate(mask,template,cv2.TM_CCOEFF_NORMED)[0,0])
                    if score>=.98 and cv2.absdiff(mask,template).mean()<=3:
                        text_label=name
                        label_raw.append(dict(template=name,correlation=score))
        return seconds,text_label,dict(clock=raw,label=label_raw)

    def read(self,frame):
        image,rows,meta=roster_rows(frame)
        if not rows:
            return dict(valid=False,reason=meta['reason'],geometry=meta)
        recognized=[]
        for row in rows:
            who=self._identify(row)
            if who is None:
                return dict(valid=False,reason='unrecognized roster glyph',geometry=meta)
            recognized.append(dict(who,team=row['team'],active=row['active'],rect=row['rect']))
        if len({row['player'] for row in recognized})!=4 or {row['player']:row['team'] for row in recognized}!=self.members:
            return dict(valid=False,reason='roster membership changed or ambiguous')
        active=next(row for row in recognized if row['active'])
        own_team=self.members['tigerwould52']
        watching,banner_raw=self._banner(image)
        if (watching=='WATCHING_TEAMMATE' and (active['team']!=own_team or active['player']=='tigerwould52')
                or watching=='WATCHING_OPPONENT' and active['team']==own_team):
            return dict(valid=False,reason='watching banner conflicts with highlighted player/team')
        ready=bool(self.ready_reader(frame)) if self.ready_reader is not None else False
        phase=watching or ('OWN_READY' if active['player']=='tigerwould52' and ready else
                           'OWN_SHOT_IN_PROGRESS' if active['player']=='tigerwould52' else 'OTHER_PLAYER_ACTIVE')
        seconds,label,readings=self._clock_and_label(image,rows,meta)
        score=min(row['correlation'] for row in recognized)
        return dict(valid=True,format='ALTERNATE_SHOT',players=sorted(self.members),teams=self.members,
                    active_player=active['player'],ball_owner=active['player'],ball_team=active['team'],
                    turn_state=phase,seconds_left=seconds,shot_label=label,roster=recognized,
                    ready_controls=ready,confidence=score,
                    raw=dict(banner=banner_raw,active_fields=readings,geometry=meta),
                    source='wgt_team_hud_v1')
