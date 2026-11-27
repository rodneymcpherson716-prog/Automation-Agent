"""Application conditions, calibration, visual motion evidence and session artifacts.

No Windows input API is used here. All observation images originate in MSS.
"""
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import uuid

import cv2
import numpy as np

from automation_core import consensus, plan_power
from terrain_handler import bunker_power_estimate, is_bunker_state
from shot_store import DATABASE, record_capture
from vision.regions import hud_crop


# Unit glyphs extracted from the user's saved tee/putting HUDs, normalized to
# a 61px sign. Only used when OCR abstains, with correlation and margin gates.
UNIT_GLYPHS={
 'YDS':['1001111101110','1101111111110','0111110111100','0110110111110','0110110110011','0110110110011','0110111101110'],
 'FT':['1111111','1100110','1100110','1110110','1100110','1100110','1000010'],
 'IN':['0111011','1111011','1111111','1111111','1111111','1111011','1111011'],
}

# Independently OCR-verified IN from the automatic run's +0.6 IN green badge.
# Keep its native 121px badge scale; this helps the blurred lower-screen HUD
# without substituting a unit based on shot mode or an earlier measurement.
UNIT_GLYPH_VARIANTS=[('IN',121,1.,[
 '11011100011','11011100011','11011100011','11011110011',
 '11001111011','11001111011','11001111011','11001101111',
 '11001001111','11001000111','11001000111','11001000011'])]


def unit_from_glyphs(patch,sign_width,*,elevation=False):
    if patch.size==0 or not 45<=sign_width<=160:
        return None
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(np.float32)
    high=gray-cv2.GaussianBlur(gray,(0,0),1.5)
    if gray.std()<2:
        return None
    scores={}
    variants=[(label,61,0.,rows) for label,rows in UNIT_GLYPHS.items()]
    if elevation and .85<=sign_width/121<=1.15:
        variants+=UNIT_GLYPH_VARIANTS
    for label,reference_width,blur,rows in variants:
        template=np.array([[255 if c=='1' else 0 for c in row] for row in rows],np.float32)
        scale=sign_width/reference_width
        template=cv2.resize(template,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
        if blur:
            template=cv2.GaussianBlur(template,(0,0),blur*scale)
        if template.shape[0]>gray.shape[0] or template.shape[1]>gray.shape[1]:
            continue
        edge=template-cv2.GaussianBlur(template,(0,0),1.5)
        correlation=cv2.matchTemplate(high,edge,cv2.TM_CCOEFF_NORMED)
        raw=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        correlation[raw<.45]=-1
        scores[label]=max(scores.get(label,-1),float(correlation.max()))
    scores=sorted(((score,label) for label,score in scores.items()),reverse=True)
    if not scores or scores[0][0]<.62 or (len(scores)>1 and scores[0][0]-scores[1][0]<.12):
        return None
    return scores[0][1]


def parse_wind(text):
    text = text.upper().replace('–','-').replace('—','-')
    m = re.search(r'(?<!\d)(\d{1,2})\s*-\s*(\d{1,2})\s*MPH\b',text)
    if m:
        lo,hi = int(m[1]),int(m[2])
        return (lo,hi) if 0<=lo<=hi<=50 else None
    m = re.search(r'(?<!\d)(\d{1,2})\s*MPH\b',text)
    return (int(m[1]),int(m[1])) if m and int(m[1])<=50 else None


def triangle_vector(patch, colour='blue'):
    """Find a triangular HUD arrow's apex; return screen x/right, y/down."""
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    low,high = ((85,40,100),(125,255,255)) if colour=='blue' else ((35,70,70),(90,255,255))
    mask = cv2.inRange(hsv,low,high)
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours,key=cv2.contourArea,reverse=True):
        if cv2.contourArea(c)<7:
            continue
        hull=cv2.convexHull(c)
        poly=cv2.approxPolyDP(hull,.06*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)
        if len(poly)!=3:
            continue
        angles=[]
        for i in range(3):
            a,b=poly[(i+1)%3]-poly[i],poly[(i+2)%3]-poly[i]
            angles.append(math.acos(float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
        order=np.argsort(angles)
        # Wind arrows have a pointed apex. Equilateral arrows do not establish
        # a unique heading; do not invent one from their bounding rectangle.
        if colour=='blue' and angles[order[1]]-angles[order[0]] < .10:
            continue
        tip=poly[order[0]]
        base=np.mean(np.delete(poly,order[0],axis=0),axis=0)
        vec=tip-base
        vec/=np.linalg.norm(vec)
        return tuple(float(v) for v in vec)
    return None


def elevation_arrow(patch):
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(35,70,70),(90,255,255))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    choices=[s for s in stats[1:] if s[4]>=7 and s[2]>=3 and s[3]>=3]
    if not choices:
        return None
    x,y,w,h,_=max(choices,key=lambda s:s[4])
    ys,_=np.where(mask[y:y+h,x:x+w]>0)
    # A downward triangle has its wide base above its apex.
    centre=(h-1)/2
    if abs(float(ys.mean())-centre)<.10:
        return None
    return -1 if ys.mean()<centre else 1


def read_putter_range(frame,ocr):
    """Separate integer/FT reads, with complete centred glyph evidence."""
    H,W=frame.shape[:2]
    patch=frame[int(.938*H):int(.966*H),int(.122*W):int(.172*W)]
    unit=frame[int(.966*H):int(.989*H),int(.125*W):int(.149*W)]
    if not patch.size or not unit.size:
        return None
    mask=cv2.inRange(patch,(100,)*3,(255,)*3)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    glyphs=[(i,s) for i,s in enumerate(stats[1:],1)
            if s[1]<.50*patch.shape[0] and s[3]>=.35*patch.shape[0] and s[4]>=5]
    if not 1<=len(glyphs)<=3:
        return None
    x=min(s[0] for _,s in glyphs);right=max(s[0]+s[2] for _,s in glyphs)
    y=min(s[1] for _,s in glyphs);bottom=max(s[1]+s[3] for _,s in glyphs)
    if x<2 or right>patch.shape[1]-2 or y<1 or bottom>patch.shape[0]-2:
        return None
    # Range text is centred on its fixed HUD column. An erased leading or
    # trailing digit shifts the remaining glyphs and cannot authorize power.
    if abs(int(.122*W)+(x+right)/2-.138*W)>1.6*W/1288:
        return None
    if max(s[3] for _,s in glyphs)-min(s[3] for _,s in glyphs)>2:
        return None
    selected=np.isin(labels,[i for i,_ in glyphs]).astype(np.uint8)*255
    gray=cv2.cvtColor(patch[y:bottom,x:right],cv2.COLOR_BGR2GRAY)
    binary=selected[y:bottom,x:right]
    texts=[]
    # The isolated numeral cluster is a single word. Line segmentation can
    # misread the printed 60 as 66 on the binary mask.
    for image,psm in ((gray,13),(binary,8)):
        image=cv2.resize(image,None,fx=10,fy=10,interpolation=cv2.INTER_CUBIC)
        image=cv2.copyMakeBorder(image,25,25,25,25,cv2.BORDER_CONSTANT,value=0)
        text=ocr(image,f'--psm {psm} -c tessedit_char_whitelist=0123456789').strip()
        if not re.fullmatch(r'\d{1,3}',text) or len(text)!=len(glyphs):
            return None
        texts.append(text)
    if texts[0]!=texts[1] or not 1<=int(texts[0])<=400:
        return None
    umask=cv2.inRange(unit,(100,)*3,(255,)*3)
    _,_,ustats,_=cv2.connectedComponentsWithStats(umask)
    letters=[s for s in ustats[1:] if s[3]>=.30*unit.shape[0] and s[4]>=5]
    if len(letters) not in (2,3) or any(s[0]<1 or s[0]+s[2]>unit.shape[1]-1 or
                               s[1]<1 or s[1]+s[3]>unit.shape[0]-1 for s in letters):
        return None
    label='FT' if len(letters)==2 else 'YDS'
    large=cv2.resize(umask,None,fx=10,fy=10,interpolation=cv2.INTER_NEAREST)
    large=cv2.copyMakeBorder(large,20,20,20,20,cv2.BORDER_CONSTANT,value=0)
    units=[ocr(large,f'--psm {psm} -c tessedit_char_whitelist={label}').strip() for psm in (6,7)]
    if units!=[label,label]:
        return None
    return dict(value=float(texts[0]),unit='ft' if label=='FT' else 'yd',number_reads=texts,unit_reads=units,
                glyph_count=len(glyphs),glyph_bounds=[int(v) for v in (x,y,right,bottom)])


def _read_complete_integer(patch,ocr,*,confirm_gray=False):
    """Read an integer only from complete, separate bright numerals."""
    hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY) if confirm_gray else None
    readings=[]
    for level in (175,200,225):
        mask=cv2.inRange(hsv,(0,0,level),(179,60,255))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        glyphs=sorted((s for s in stats[1:] if s[2]>=3 and s[3]>=.42*patch.shape[0] and s[4]>=8),
                      key=lambda s:s[0])
        if not 1<=len(glyphs)<=3:
            continue
        if any(y>.32*patch.shape[0] or x==0 or x+w>=patch.shape[1] or y+h>=patch.shape[0]
               for x,y,w,h,_ in glyphs):
            continue
        # A small mark at the digit baseline can be a decimal point. With
        # separate grayscale proof, ignore flag pixels below the text row.
        baseline=max(y+h for _,y,_,h,_ in glyphs)
        bottom_marks=[s[1]+s[3] for s in stats[1:] if s[4]>=2 and s[3]<.26*patch.shape[0]]
        if any(bottom>.68*patch.shape[0] and (not confirm_gray or bottom<=baseline+2)
               for bottom in bottom_marks):
            continue
        digits=[]
        for x,y,w,h,_ in glyphs:
            digit=255-mask[y:y+h,x:x+w]
            digit=cv2.copyMakeBorder(digit,5,5,5,5,cv2.BORDER_CONSTANT,value=255)
            digit=cv2.resize(digit,None,fx=7,fy=7,interpolation=cv2.INTER_CUBIC)
            text=ocr(digit,'--psm 13 -c tessedit_char_whitelist=0123456789').strip()
            if not re.fullmatch(r'\d',text):
                break
            if gray is not None:
                raw=cv2.copyMakeBorder(gray[y:y+h,x:x+w],5,5,5,5,cv2.BORDER_CONSTANT,value=130)
                raw=cv2.resize(raw,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
                if ocr(raw,'--psm 13 -c tessedit_char_whitelist=0123456789').strip()!=text:
                    break
            digits.append(text)
        if len(digits)==len(glyphs):
            readings.append(int(''.join(digits)))
    return readings[0] if len(readings)>=2 and len(set(readings))==1 else None


def read_fringe_putt_badge(frame,sign,ocr):
    """Verify the yard-based putting badge and return planner values in feet."""
    if sign is None:
        return None
    x,y,w,h=sign
    if not 90<=w<=145:
        return None
    if unit_from_glyphs(frame[y+int(.14*w):y+int(.40*w),x+int(.69*w):x+int(.99*w)],w)!='YDS':
        return None
    elevation_unit=unit_from_glyphs(frame[y+int(.54*h):y+int(.96*h),x+int(.65*w):x+w],w,elevation=True)
    if elevation_unit not in ('FT','IN'):
        return None
    direction=elevation_arrow(frame[y+int(.46*h):y+h,x:x+int(.32*w)])
    if direction is None:
        return None
    yards=_read_complete_integer(frame[y+int(.06*w):y+int(.40*w),x+int(.12*w):x+int(.74*w)],ocr)
    elevation=_read_complete_integer(frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.74*w)],ocr)
    if not yards or elevation is None:
        return None
    return dict(distance=3.*yards,unit='ft',elevation_ft=elevation*direction/(12 if elevation_unit=='IN' else 1))


def quick_putt_badge(frame,sign,ocr,*,elevation_text=None):
    """Read a badge, removing a verified two-pixel golfer stipple if needed."""
    result=_read_putt_badge(frame,sign,ocr,elevation_text=elevation_text)
    if result is not None or sign is None:
        return result
    x,y,w,h=map(int,sign)
    H,W=frame.shape[:2]
    if not (90<=w<=140 and 0<=x<x+w<=W and 0<=y<y+h<=H):
        return None
    # The golfer's transparency is an alternating black/clear pixel pattern,
    # unlike faded text or an absent label. Require that actual periodic signal
    # in the covered unit region before trying a tiny hole-closing operation.
    patch=frame[y+int(.54*h):y+int(.96*h),x+int(.65*w):x+w]
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(np.float32)
    adjacent=float(np.mean(np.abs(gray[:,1:]-gray[:,:-1])))
    repeated=float(np.mean(np.abs(gray[:,2:]-gray[:,:-2])))
    if adjacent<40 or repeated>=.5*adjacent:
        return None
    results=[]
    grayscale_confirmed=False
    for anchor in ((0,0),(1,1)):
        filtered=frame.copy()
        filtered[y:y+h,x:x+w]=cv2.morphologyEx(frame[y:y+h,x:x+w],cv2.MORPH_CLOSE,
                                             np.ones((2,2),np.uint8),anchor=anchor)
        decoded=_read_putt_badge(filtered,sign,ocr,require_gray=False,elevation_text=elevation_text)
        if decoded is None:
            return None
        results.append(decoded)
        grayscale_confirmed=grayscale_confirmed or _read_putt_badge(
            filtered,sign,ocr,elevation_text=elevation_text)==decoded
    # Keep all original unit correlations, numeric threshold votes, grayscale
    # agreement and arrow checks. Opposite filter alignments must also agree.
    return dict(results[0],recovery='verified_periodic_golfer_stipple') if grayscale_confirmed and results[0]==results[1] else None


def _visible_decimal_text(patch,digits):
    """Recover only the decimal's position from actual separate pixel evidence.

    The caller already read every digit in unthresholded grayscale. Require
    one complete component per digit and a distinct low dot between digits
    at two thresholds; never invent a dot from a prior measurement.
    """
    if not re.fullmatch(r'\d{2,5}',digits) or patch.size==0:
        return None
    positions=[]
    for level in (200,220):
        mask=cv2.inRange(patch,(level,)*3,(255,)*3)
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        parts=[tuple(int(v) for v in s) for s in stats[1:] if s[4]>=2]
        glyphs=sorted((s for s in parts if s[3]>=.35*patch.shape[0] and s[4]>=5),key=lambda s:s[0])
        if len(glyphs)!=len(digits) or len(parts)!=len(glyphs)+1:
            return None
        height=max(s[3] for s in glyphs)
        if min(s[3] for s in glyphs)<.75*height:
            return None
        if any(s[0]<1 or s[1]<1 or s[0]+s[2]>=patch.shape[1]-1
               or s[1]+s[3]>=patch.shape[0]-1 for s in parts):
            return None
        dots=[s for s in parts if s not in glyphs]
        if len(dots)!=1:
            return None
        dot=dots[0]
        if not (dot[3]<=.30*height and dot[2]<=.45*max(s[2] for s in glyphs)
                and dot[1]>=min(s[1] for s in glyphs)+.65*height):
            return None
        gap=[i for i in range(1,len(glyphs))
             if glyphs[i-1][0]+glyphs[i-1][2]<dot[0]
             and dot[0]+dot[2]<glyphs[i][0]]
        if len(gap)!=1:
            return None
        positions.append(gap[0])
    if positions[0]!=positions[1]:
        return None
    index=positions[0]
    return digits[:index]+'.'+digits[index:]


def read_locked_short_putt_distance(frame, sign, rounded_feet, ocr):
    """Decode a short locked badge without treating its rounded HUD as aim."""
    if sign is None or not 1 <= rounded_feet <= 5 or not visible_hole_lock(frame, sign):
        return None
    x,y,w,h = sign
    unit_patch = frame[y+int(.14*w):y+int(.40*w),x+int(.69*w):x+int(.99*w)]
    if unit_from_glyphs(unit_patch,w) != 'FT':
        return None
    # Exclude scene and stipple pixels at the left of the centered short label.
    patch = frame[y+int(.06*w):y+int(.40*w),x+int(.28*w):x+int(.74*w)]
    for threshold in (175,210):
        mask=cv2.inRange(patch,(threshold,)*3,(255,)*3)
        large=cv2.resize(mask,None,fx=7,fy=7,interpolation=cv2.INTER_NEAREST)
        large=cv2.copyMakeBorder(large,12,12,12,12,cv2.BORDER_CONSTANT,value=0)
        labels=[ocr(large,f'--psm {mode} -c tessedit_char_whitelist=0123456789.').strip()
                for mode in (6,7)]
        label=labels[0]
        if label!=labels[1] or not re.fullmatch(r'\d\.\d',label):
            continue
        value=float(label)
        if (value>0 and abs(value-rounded_feet)<=.5
                and _visible_decimal_text(patch,label.replace('.',''))==label):
            return value
    return None


def _read_segmented_decimal(patch, ocr):
    readings = []
    for level in (215, 225, 205):
        mask = cv2.inRange(patch, (level,)*3, (255,)*3)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        glyphs = sorted((s for s in stats[1:] if s[3] >= .42*patch.shape[0] and s[4] >= 7),
                        key=lambda s: s[0])
        if not 2 <= len(glyphs) <= 4:
            continue
        digits = []
        for x, y, w, h, _ in glyphs:
            digit = cv2.copyMakeBorder(255-mask[y:y+h, x:x+w], 5, 5, 5, 5,
                                       cv2.BORDER_CONSTANT, value=255)
            votes = [ocr(cv2.resize(digit, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
                         '--psm 13 -c tessedit_char_whitelist=0123456789').strip() for scale in (6, 8)]
            if votes[0] != votes[1] or not re.fullmatch(r'\d', votes[0]):
                break
            digits.append(votes[0])
        if len(digits) == len(glyphs):
            number = _visible_decimal_text(patch, ''.join(digits))
            if number is not None:
                readings.append(number)
        if len(readings) == 2:
            break
    return float(readings[0]) if len(readings) >= 2 and len(set(readings)) == 1 else None


def read_stippled_locked_putt(frame, sign, rounded_feet, ocr):
    """Read a locked putt through the translucent golfer when its digits remain visible."""
    if not sign or not 1 <= rounded_feet <= 30 or not visible_hole_lock(frame, sign):
        return None
    x, y, w, h = sign
    distance_patch = frame[y+round(.067*w):y+round(.367*w), x+round(.083*w):x+round(.742*w)]
    distance = _read_segmented_decimal(distance_patch, ocr)
    # The independent hole HUD supplies the rounded feet and unit. The
    # adjacent lock associates that printed measurement with this badge.
    if distance is None or abs(distance-rounded_feet) > .5:
        return None
    unit_patch = frame[y+int(.54*h):y+int(.96*h), x+int(.65*w):x+w]
    unit = unit_from_glyphs(unit_patch, w, elevation=True)
    if unit not in ('FT', 'IN'):
        return None
    elevation_patch = frame[y+round(.49*h):y+round(.935*h), x+round(.083*w):x+round(.742*w)]
    elevation = _read_segmented_decimal(elevation_patch, ocr)
    direction = elevation_arrow(frame[y+int(.46*h):y+h, x:x+int(.32*w)])
    if elevation is None or direction is None:
        return None
    return dict(distance=distance, unit='ft', elevation_ft=elevation*direction/(12 if unit=='IN' else 1))


def _read_putt_badge(frame,sign,ocr,require_gray=True,*,elevation_text=None):
    """Read visible feet/inch glyphs without scanning animated scenery."""
    if sign is None:
        return None
    x,y,w,h=map(int,sign)
    H,W=frame.shape[:2]
    if not (0<=x<x+w<=W and 0<=y<y+h<=H):
        return None
    unit=unit_from_glyphs(frame[y+int(.14*w):y+int(.40*w),x+int(.69*w):x+int(.99*w)],w)
    eu=unit_from_glyphs(frame[y+int(.54*h):y+int(.96*h),x+int(.65*w):x+w],w,elevation=True)
    if unit!='FT' or eu not in ('IN','FT'):
        return None
    def number(patch, raw_text=None):
        votes=[]
        labels=[]
        for level in (200,210,220):
            mask=cv2.inRange(patch,(level,)*3,(255,)*3)
            large=cv2.resize(mask,None,fx=6,fy=6,interpolation=cv2.INTER_LINEAR)
            large=cv2.copyMakeBorder(large,10,10,10,10,cv2.BORDER_CONSTANT,value=0)
            text=ocr(large,'--psm 7 -c tessedit_char_whitelist=0123456789.')
            if raw_text is not None:
                raw_text.append(text)
            text=text.strip()
            if not re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',text):
                return None
            votes.append(float(text))
            labels.append(text)
        label=labels[0]
        split_votes=len(set(votes))!=1
        if split_votes:
            if not require_gray:
                return None
            label=consensus(labels)
            if (label is None or '.' not in label
                    or _visible_decimal_text(patch,label.replace('.',''))!=label):
                return None
        if not require_gray:
            return votes[0]
        # Bright thresholds can turn a faded 5 into 9 in every mask. Require
        # an unthresholded read as independent evidence, including its decimal.
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
        padded=cv2.copyMakeBorder(gray,10,10,10,10,cv2.BORDER_CONSTANT,value=0)
        text=ocr(padded,'--psm 7 -c tessedit_char_whitelist=0123456789.')
        if raw_text is not None:
            raw_text.append(text)
        text=text.strip()
        if ('.' in labels[0] and all(label==labels[0] for label in labels)
                and text==labels[0].replace('.','')):
            # A flag/background can make whole-word grayscale OCR drop a
            # visible dot. Keep its independent digit reading and verify the
            # dot's actual component geometry instead of accepting a mismatch.
            restored=_visible_decimal_text(patch,text)
            if restored==labels[0]:
                text=restored
        if not text and '.' in label and _visible_decimal_text(patch,label.replace('.',''))==label:
            # Padding can turn a readable row into an empty OCR result.
            # Require both unpadded layouts and the visible decimal point.
            alternatives=[ocr(gray,f'--psm {mode} -c tessedit_char_whitelist=0123456789.')
                          for mode in (6,7)]
            if raw_text is not None:
                raw_text.extend(alternatives)
            if any(read.strip()!=label for read in alternatives):
                return None
        elif not re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',text) or float(text)!=float(label):
            return None
        elif split_votes:
            # A disagreement needs a second grayscale layout in addition to
            # the confirmed decimal geometry and the padded line read.
            unpadded=ocr(gray,'--psm 6 -c tessedit_char_whitelist=0123456789.')
            if raw_text is not None:
                raw_text.append(unpadded)
            if unpadded.strip()!=label:
                return None
        return float(label)
    distance=number(frame[y+int(.06*w):y+int(.40*w),x+int(.12*w):x+int(.74*w)])
    elevation_patch=frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)]
    elevation=number(elevation_patch,elevation_text)
    if distance is not None and elevation is None:
        # Padding the line can omit a narrow leading 1. Require complete
        # segmented digits and agreement with the unpadded grayscale row.
        integer=_read_complete_integer(elevation_patch,ocr)
        if integer is not None:
            gray=cv2.cvtColor(elevation_patch,cv2.COLOR_BGR2GRAY)
            large=cv2.resize(gray,None,fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
            texts=[ocr(large,f'--psm {mode}') for mode in (6,7)]
            if elevation_text is not None:
                elevation_text.extend(texts)
            if all(text.strip()==str(integer) for text in texts):
                elevation=integer
        if elevation is None and require_gray:
            # A moving flag can make the entire row read as a single 1.
            # Each repeated digit must agree in masked and grayscale crops.
            repeated=_read_complete_integer(elevation_patch,ocr,confirm_gray=True)
            if repeated in (11,111):
                elevation=repeated
    direction=elevation_arrow(frame[y+int(.46*h):y+h,x:x+int(.32*w)])
    if distance is None or elevation is None or (direction is None and elevation!=0):
        return None
    return dict(distance=distance,unit='ft',elevation_ft=elevation*(direction or 1)/(12 if eu=='IN' else 1))


def read_inch_elevation(frame, sign, ocr):
    """Read whole inches from bright numeral glyphs, away from the animated flag."""
    x, y, w, h = sign
    unit_patch = frame[y+int(.54*h):y+int(.96*h), x+int(.65*w):x+w]
    if unit_from_glyphs(unit_patch, w, elevation=True) != 'IN':
        return None
    patch = frame[y+int(.49*h):y+int(.94*h), x+int(.28*w):x+int(.74*w)]
    number = _read_complete_integer(patch, ocr)
    direction = elevation_arrow(frame[y+int(.46*h):y+h, x:x+int(.32*w)])
    if number is None or (number != 0 and direction is None):
        return None
    return number*(direction or 1)/12


def read_conditions(frame,sign,ocr,crop,verified_badge=None,numeric_reader=None,*,elevation_only=False,glyph_ocr=None):
    raw={}
    elevation=verified_badge['elevation_ft'] if verified_badge else None
    if sign and verified_badge is None and glyph_ocr is not None:
        elevation = read_inch_elevation(frame, sign, glyph_ocr)
        if elevation is not None:
            raw['elevation_method'] = 'isolated inch glyphs'
    if sign and verified_badge is None and elevation is None:
        x,y,w,h=sign
        texts=ocr(frame[y+int(.50*h):y+int(.96*h),x+int(.30*w):x+int(.97*w)],modes=(6,7),scale=7)
        raw['elevation']=texts
        values=[]
        for t in texts:
            m=re.search(r'(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*(FT|IN)\b',t.upper())
            if m:
                values.append((float(m[1]),m[2]))
        value=consensus(values)
        if value is None:
            lower=frame[y+int(.50*h):y+int(.96*h),x+int(.30*w):x+int(.97*w)]
            texts=ocr(cv2.inRange(lower,(175,175,175),(255,255,255)),modes=(6,7),scale=7)
            raw['elevation_white']=texts
            for t in texts:
                m=re.search(r'(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*(FT|IN)\b',t.upper())
                if m:
                    values.append((float(m[1]),m[2]))
            value=consensus(values)
        if value is None:
            number_texts=ocr(frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)],modes=(6,7),scale=8)
            unit_texts=ocr(frame[y+int(.57*h):y+int(.94*h),x+int(.74*w):x+int(.97*w)],modes=(6,7),scale=8)
            raw['elevation_number']=number_texts
            raw['elevation_unit']=unit_texts
            numbers=[];units=[]
            number_texts+=ocr(cv2.inRange(frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)],(175,175,175),(255,255,255)),modes=(6,7),scale=8)
            unit_texts+=ocr(cv2.inRange(frame[y+int(.57*h):y+int(.94*h),x+int(.74*w):x+int(.97*w)],(175,175,175),(255,255,255)),modes=(6,7),scale=8)
            for t in number_texts:
                clean=re.sub(r'[|\s]','',t)
                if re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',clean):
                    numbers.append(float(clean))
            for t in unit_texts:
                clean=re.sub('[^A-Z]','',t.upper())
                if clean=='IH':  # The narrow HUD N is consistently read as H.
                    clean='IN'
                if clean in ('FT','IN'):
                    units.append(clean)
            number,unit=consensus(numbers),consensus(units)
            if number is None and numeric_reader is not None:
                # Reuse the controller's separately segmented/thresholded
                # digit reader; no conversion of a bracket or T into a 1.
                number=numeric_reader(frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)])
            if number is None:
                number_patch=frame[y+int(.49*h):y+int(.95*h),x+int(.33*w):x+int(.73*w)]
                # A one-pixel translucent bottom border can join the digit.
                # Crop that border away rather than interpreting it as text.
                for trim in (1,2):
                    if number_patch.shape[0]<=trim:
                        continue
                    for t in ocr(number_patch[:-trim],modes=(6,7),scale=8):
                        clean=re.sub(r'[|\s]','',t)
                        if re.fullmatch(r'\d{1,3}(?:\.\d{1,2})?',clean):
                            numbers.append(float(clean))
                number=consensus(numbers)
            if unit is None:
                glyph_unit=unit_from_glyphs(frame[y+int(.54*h):y+int(.96*h),x+int(.65*w):x+w],w,elevation=True)
                if glyph_unit in ('FT','IN'):
                    unit=glyph_unit
                    raw['elevation_unit_method']='calibrated glyph correlation'
            if number is not None and unit is None and not units:
                # The badge's bright right border can merge with the unit.
                # Read the interior on a padded white background at lower scale.
                interior=frame[y+int(.57*h):y+int(.94*h),x+int(.74*w):x+int(.94*w)]
                mask=cv2.inRange(interior,(170,170,170),(255,255,255))
                padded=cv2.copyMakeBorder(255-mask,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
                raw['elevation_unit_interior']=ocr(padded,modes=(6,7),scale=4)
                labels=[text.strip().upper() for text in raw['elevation_unit_interior']]
                unit=consensus([label for label in labels if label in ('FT','IN')])
                if unit:
                    raw['elevation_unit_method']='padded interior OCR'
            if number is not None and unit:
                value=(number,unit)
        direction=elevation_arrow(frame[y+int(.46*h):y+h,x:x+int(.32*w)])
        if value and (direction is not None or value[0]==0):
            elevation=value[0]*(direction or 1)/(12 if value[1]=='IN' else 1)
    if elevation_only:
        return {'elevation_ft':elevation}
    wind_texts=ocr(hud_crop(frame,'wind',(.078,.063,.125,.095),crop),modes=(6,7),scale=6)
    raw['wind']=wind_texts
    speed=consensus([parse_wind(t) for t in wind_texts])
    if speed is None:
        # A wider speed range can crowd the small M. Reread the printed
        # unit independently before trusting agreeing digits from the line.
        unit_patch=hud_crop(frame,'wind',(.110,.065,.129,.100),crop)
        if unit_patch.size:
            units=ocr(unit_patch,modes=(6,7),scale=6)
            raw['wind_unit']=units
            if sum(text.strip().upper()=='MPH' for text in units)>=2:
                speed=consensus([parse_wind(re.sub(r'\s+(?:4?PH)$',' MPH',text.strip().upper()))
                                 for text in wind_texts])
    vector=triangle_vector(hud_crop(frame,'wind',(.061,.064,.078,.097),crop))
    wind=None
    if speed is not None and (vector is not None or speed[1]==0):
        dx,dy=vector or (0.,0.)
        wind=dict(low_mph=speed[0],high_mph=speed[1],mean_mph=sum(speed)/2,
                  screen_dx=round(dx,4),screen_dy=round(dy,4),
                  heading_degrees=round(math.degrees(math.atan2(dx,-dy))%360,1) if vector else None)
    club_texts=ocr(hud_crop(frame,'club',(.102,.938,.123,.968),crop),modes=(7,),scale=7)
    raw['club_name']=club_texts
    clubs=[]
    for text in club_texts:
        clean=re.sub('[^A-Z0-9]','',text.upper().replace('|','I').replace(']','I')).replace('1','I')
        if clean in ('DR','D','3W','5W','7W','PW','SW','LW','GW'):
            clubs.append('DR' if clean=='D' else clean)
        elif re.fullmatch('[2-9]I',clean):
            clubs.append(clean)
        elif re.fullmatch('[2-9]H(?:Y)?',clean):
            clubs.append(clean[0]+'HY')
        elif (m:=re.fullmatch(r'([357])W(?:D|O|OD|OOD)?',clean)):
            clubs.append(m[1]+'W')
    club_name=consensus(clubs)
    if club_name is None:
        # The tiny iron number can vanish when the brand row enters its crop.
        # Read the lower row independently, but only with a verified IRONS
        # category so a cropped driver loft cannot masquerade as an iron.
        category=ocr(hud_crop(frame,'club',(.094,.967,.132,.990),crop),modes=(6,7,13),scale=8)
        raw['club_category']=category
        if consensus(['IRONS' if re.sub('[^A-Z]','',t.upper())=='IRONS' else None for t in category]):
            patch=hud_crop(frame,'club',(.104,.943,.122,.965),crop)
            mask=cv2.inRange(patch,(100,100,100),(255,255,255))
            iron_texts=ocr(mask,modes=(6,7,13),scale=10)
            raw['club_iron_row']=iron_texts
            irons=[]
            for text in iron_texts:
                clean=re.sub('[^A-Z0-9]','',text.upper().replace('|','I').replace(']','I'))
                if re.fullmatch('[2-9][I1L]',clean):
                    irons.append(clean[0]+'I')
            club_name=consensus(irons)
            if club_name is None:
                # On the 9I row a tight white-on-black crop reads SI/SL.
                # Padding and inverted contrast preserve the visible digit;
                # retain the verified IRONS category and exact digit+I gate.
                padded_texts=[]
                gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
                for threshold in (145,180):
                    padded=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)[1]
                    padded=cv2.copyMakeBorder(padded,5,5,5,5,cv2.BORDER_CONSTANT,value=255)
                    padded_texts+=ocr(padded,modes=(7,),scale=8)
                raw['club_iron_padded']=padded_texts
                for text in padded_texts:
                    clean=re.sub('[^A-Z0-9]','',text.upper().replace('|','I').replace(']','I'))
                    if re.fullmatch('[2-9][I1L]',clean):
                        irons.append(clean[0]+'I')
                club_name=consensus(irons)
    if club_name is None:
        wider=ocr(hud_crop(frame,'club',(.073,.938,.130,.967),crop),modes=(6,7),scale=7)
        raw['club_name_wide']=wider
        drivers=[]
        for text in wider:
            match=re.search(r'(?<!\d)([789]|1[0-3])(?:[.,]([05]))?\s*D\b',text)
            if match:
                drivers.append(f'{match[1]}.{match[2] or "0"}D')
        club_name=consensus(drivers)
        if club_name is None:
            hybrids=[]
            for text in wider:
                match=re.search(r'(?<![A-Z0-9])([2-9])\s*H(?:Y)?(?![A-Z0-9])',text.upper())
                if match:
                    hybrids.append(match[1]+'HY')
            club_name=consensus(hybrids)
    green_texts=ocr(crop(frame,(.014,.144,.15,.17)),modes=(7,),scale=4)
    raw['green_speed']=green_texts
    speeds=[]
    for t in green_texts:
        m=re.search(r'\(\s*(\d{1,2}\.\d)\s*\)',t)
        if m and 4<=float(m[1])<=20:
            speeds.append(float(m[1]))
    if consensus(speeds) is None:
        green_numbers=ocr(crop(frame,(.094,.145,.119,.166)),modes=(6,7),scale=8)
        raw['green_speed_number']=green_numbers
        for t in green_numbers:
            m=re.search(r'(\d{1,2}\.\d)',t)
            if m and 4<=float(m[1])<=20:
                speeds.append(float(m[1]))
    return dict(elevation_ft=elevation,wind=wind,club_name=club_name,
                green_speed=consensus(speeds),conditions_raw=raw)


def visible_hole_lock(frame,sign):
    """Require the red lock's white circular HUD backing beside the badge."""
    if sign is None:
        return None
    H,W=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    red=cv2.inRange(hsv,(0,90,110),(12,255,255))|cv2.inRange(hsv,(170,90,110),(179,255,255))
    sx,sy,sw,sh=sign
    red[:max(int(.12*H),int(sy-.2*sh))]=0
    red[min(H,int(sy+.8*sh)):]=0
    red[:,:int(.30*W)]=0;red[:,int(.82*W):]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(red)
    sx,sy,sw,sh=sign
    for x,y,w,h,n in stats[1:]:
        if not (n>=25 and .008*W<w<.04*W and .015*H<h<.07*H):
            continue
        cx,cy=x+w/2,y+h/2
        if not (sx+sw<cx<sx+sw*2.2 and sy-.2*sh<cy<sy+.7*sh):
            continue
        zone=hsv[max(0,y-h//2):min(H,y+h+h//2),max(0,x-w):min(W,x+w*2)]
        if zone.size and ((zone[:,:,1]<55)&(zone[:,:,2]>190)).mean()>.25:
            return dict(x=int(cx),y=int(cy),source='visible_lock_beside_target')
    return None


def read_stats_patch(patch,ocr):
    """Reuse V41's numeric-row calibration, but require labels and OCR votes."""
    labels=' '.join(ocr(patch,modes=(6,),scale=3))
    if not all(word in labels for word in ('CARRY','ROLL','POWER')):
        return None
    h,w=patch.shape[:2]
    values=[]
    bands=((.238,.369),(.361,.492),(.475,.615),(.598,.738),(.721,.877))
    for index,(lo,hi) in enumerate(bands):
        image=patch[round(h*lo):round(h*hi),round(w*.405):round(w*.753)]
        maximum=(600,200,100,100,500)[index]
        unit=r'(?:YDS?|YOS)' if index<2 else '%' if index<4 else 'FT'
        def numbers(texts):
            row=[]
            for text in texts:
                if index in (2,3):
                    text=re.sub(r'(?<![A-Z0-9])O\s*%', '0%',text)
                match=re.search(r'(?<![\d.])(\d{1,4}(?:\.\d{1,2})?)\s*'+unit+r'\b' if unit!='%' else
                                r'(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*%',text)
                matches=[match[1]] if match else re.findall(r'(?<![\d.])\d{1,4}(?:\.\d{1,2})?(?![\d.])',text)
                # The distance/apex statistics display decimal places. An
                # OCR integer such as 37 must not outvote visible 1.37 after
                # dropping its small leading glyph and decimal point.
                if index in (0,1,4) and len(matches)==1 and '.' not in matches[0]:
                    continue
                if len(matches)==1 and 0<=float(matches[0])<=maximum:
                    row.append(float(matches[0]))
            return row
        row=numbers(ocr(image,modes=(7,),scale=8))
        if consensus(row) is None:
            row+=numbers(ocr(image,modes=(6,13),scale=8))
            neutral=cv2.inRange(image,(100,100,100),(255,255,255))
            row+=numbers(ocr(neutral,modes=(6,7,13),scale=8))
        value=consensus(row)
        if value is None:
            if index==4:
                values.append(None)  # Optional apex must never become a guessed integer.
                continue
            return None
        values.append(value)
    carry,roll,power,spin,apex=values
    if not (0<=carry<=600 and 0<=roll<=200 and 0<=power<=100 and 0<=spin<=100
            and (apex is None or 0<=apex<=500)):
        return None
    return dict(carry_yd=carry,roll_yd=roll,total_yd=round(carry+roll,2),power_pct=power,spin_pct=spin,apex_ft=apex)


class PowerCalibration:
    """Use contextual, verified landing observations; never fit unlabeled CSVs."""
    def __init__(self,records=(),legacy_count=0,bunker_reference=None):
        self.records=list(records)
        self.legacy_count=legacy_count
        self.bunker_reference=bunker_reference

    @classmethod
    def load(cls,path,legacy_csv=None):
        records=[]
        if Path(path).is_file():
            # Windows editors/PowerShell may prepend a UTF-8 BOM. Accept
            # either encoding without changing the saved calibration file.
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if data.get('schema')!=1 or not isinstance(data.get('observations'),list):
                raise ValueError('Invalid contextual power calibration file')
            records=data['observations']
        count=0
        if legacy_csv and Path(legacy_csv).is_file():
            import csv
            with Path(legacy_csv).open(newline='',encoding='utf-8-sig') as f:
                count=sum(1 for _ in csv.DictReader(f))
        reference_path=Path(path).parent/'bunker_reference_manual.json'
        reference=None
        if reference_path.is_file():
            reference=json.loads(reference_path.read_text(encoding='utf-8-sig'))
            if not isinstance(reference,dict) or reference.get('use_for_automatic_power') is not False:
                raise ValueError('Bunker manual reference must remain excluded from automatic power')
        return cls(records,count,bunker_reference=reference)

    def curve(self,state):
        # Sand must not inherit a fairway/full-shot curve or fit a single
        # manual hole-out. A future validated bunker fitter is separate work.
        if is_bunker_state(state):
            return None
        samples=[]
        identities=set()
        for r in self.records:
            if state.get('lie') is not None and r.get('lie')!=state['lie']:
                continue
            if 'game_context' in state and r.get('owner_verified') is not True:
                continue
            if state['mode'] in ('CHIP','PITCH','PUTT') and r.get('travel_basis')!='total':
                continue
            if (r.get('verified') is not True or r.get('mode')!=state['mode']
                    or r.get('unit')!=state['unit'] or r.get('club_name')!=state.get('club_name')
                    or r.get('club_range')!=state['club_range']
                    or abs(r.get('elevation_ft',999))>1 or (not state['putt'] and r.get('wind_mph',999)>3)
                    or (state['putt'] and r.get('green_speed')!=state.get('green_speed'))):
                continue
            identity=r.get('shot_id')
            if not identity or identity in identities:
                continue
            identities.add(identity)
            p,d=r.get('power'),r.get('travel')
            if isinstance(p,(int,float)) and isinstance(d,(int,float)) and math.isfinite(p) and math.isfinite(d) and .025<=p<=1 and d>0:
                samples.append((p,d))
        samples.sort()
        if len(samples)<2 or samples[-1][0]-samples[0][0]<.10:
            return None
        # Non-monotonic or same-power conflicting records are not a curve.
        if any(b[0]<=a[0] or b[1]<=a[1] for a,b in zip(samples,samples[1:])):
            return None
        return samples

    def record(self,path,state,stats,observed_power,shot_id):
        if not stats or not 0<stats['total_yd'] or not .025<=observed_power<=1:
            return False
        if any(r.get('shot_id')==shot_id for r in self.records):
            return False
        context=state.get('game_context')
        if context is not None:
            # A profile name does not attribute multiplayer statistics. The
            # runtime binds fresh results to an explicitly observed stats owner.
            if context.get('verified_stats_owner')!='tigerwould52' or not context.get('permission',{}).get('allowed'):
                return False
        short_game=state['mode'] in ('CHIP','PITCH')
        record=dict(shot_id=shot_id,verified=True,mode=state['mode'],unit=state['unit'],
                    club_name=state['club_name'],club_range=state['club_range'],
                    green_speed=state.get('green_speed'),elevation_ft=state['elevation_ft'],
                    wind_mph=state['wind']['mean_mph'],power=observed_power,
                    game_power=stats['power_pct']/100,
                    travel=stats['total_yd']*3 if state['putt'] else stats['total_yd'] if short_game else stats['carry_yd'],
                    travel_basis='total' if state['putt'] or short_game else 'carry',
                    source='fresh shot stats after MSS motion and stable stroke advance',stats=stats)
        # Preserve the conditions needed for future sand validation. Missing
        # historical lie/direction labels cannot be reconstructed from carry.
        record.update(lie=state.get('lie'),lie_percent=state.get('lie_percent'),
                      lie_loss_percent=state.get('lie_loss_percent'),
                      wind=dict(state['wind']),target_distance=state['distance'],
                      lie_percent_semantics='uninterpreted HUD percentage')
        if context is not None:
            record.update(owner='tigerwould52',owner_verified=True,lie=state.get('lie'),
                          format=context.get('format'),teams=context.get('teams',{}))
        self.records.append(record)
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(schema=1,observations=self.records),indent=2),encoding='utf-8')
        temporary.replace(path)
        return True

    def plan(self,state,for_selection=False):
        if is_bunker_state(state):
            result=bunker_power_estimate(state,self.bunker_reference)
            result['legacy_rows_not_fitted']=self.legacy_count
            return result
        plan_power(state,for_selection=for_selection)  # Units/mode/range stay mandatory.
        elevation=state.get('elevation_ft')
        wind=state.get('wind')
        if elevation is None or wind is None:
            raise ValueError('Live elevation and wind must be verified before a shot')
        if not math.isfinite(elevation) or not -300<=elevation<=300:
            raise ValueError('Invalid elevation reading')
        strategy=state.get('strategy',{})
        d=strategy.get('target',state['distance'])
        if not isinstance(d,(int,float)) or not math.isfinite(d) or d<=0:
            raise ValueError('Strategy target distance is invalid')
        # Same baseline elevation/wind conventions as existing helpers, now
        # fed by live readings. Coefficients are estimates, not measured course physics.
        if state['putt']:
            if state.get('green_speed') is None:
                raise ValueError('Putting green speed must be readable')
            effective=max(.1,d+elevation*12.)
            correction='putt baseline: one foot per uphill inch; no airborne wind correction'
        else:
            along=-wind['screen_dy']*wind['mean_mph']
            # Retain the existing short-shot coefficient (condition_distance
            # used 0.3% below 65 yd), distinct from full-flight estimation.
            wind_rate=.003 if state['mode'] in ('CHIP','PITCH') else .005
            effective=max(.5,(d+elevation/3.)*(1.-wind_rate*along))
            correction=f'{state["mode"].lower()} baseline: elevation ft/3; {wind_rate*100:.1f}% per mph along screen-top tailwind'
        loss=state.get('lie_loss_percent')
        if loss is not None and not state['putt']:
            if len(loss)!=2 or not 0<=loss[0]<=loss[1]<100:
                raise ValueError('Invalid verified lie-loss range')
            effective/=1.-sum(loss)/200.
            correction+='; displayed lie-loss midpoint estimate'
        samples=self.curve(state)
        model='selected-range baseline'
        if samples and samples[0][1]<=effective<=samples[-1][1]:
            p=float(np.interp(effective,[v[1] for v in samples],[v[0] for v in samples]))
            model=f'contextual empirical interpolation ({len(samples)} independent shots)'
        else:
            p=effective/state['club_range']
        if p<.025 and not for_selection:
            raise ValueError('Adjusted power too low for selected range')
        tee=(state.get('shot')==1 and state['mode']=='FULL' and state.get('lie')=='TEE')
        power=1. if tee and not for_selection else min(.99,p)
        return dict(power=power,unclipped_power=p,effective_distance=round(effective,3),model=model,
                    correction=correction,legacy_rows_not_fitted=self.legacy_count,
                    crosswind_mph=round(wind['screen_dx']*wind['mean_mph'],2),
                    aim_adjustment=False,calibration_points=len(samples or []),
                    clipped_to_full=(p>.99),tee_full_power=tee and not for_selection,
                    strategy_target=d,lie=state.get('lie'),
                    policy='100% verified first tee shot' if tee else 'distance/conditions')


class MotionEvidence:
    """Separate whole-scene animation from a tracked small bright ball candidate."""
    def __init__(self,before):
        self.previous=self.prepare(before)
        self.motion_frames=0
        self.peak_change=0.
        self.tracks=[]
        self.ball_track=False
        self.samples=0

    @staticmethod
    def prepare(frame):
        h,w=frame.shape[:2]
        roi=frame[int(.20*h):int(.90*h),int(.18*w):int(.86*w)]
        return cv2.resize(roi,(480,280))

    def update(self,frame):
        current=self.prepare(frame)
        delta=cv2.absdiff(current,self.previous)
        change=float((delta.max(axis=2)>30).mean())
        self.peak_change=max(self.peak_change,change)
        self.motion_frames+=int(change>.025)
        self.samples+=1
        hsv=cv2.cvtColor(current,cv2.COLOR_BGR2HSV)
        bright=cv2.inRange(hsv,(0,0,190),(179,65,255))
        moving=(delta.max(axis=2)>20).astype(np.uint8)*255
        mask=cv2.bitwise_and(bright,moving)
        n,_,stats,centres=cv2.connectedComponentsWithStats(mask)
        points=[tuple(centres[i]) for i in range(1,n)
                if 2<=stats[i,4]<=60 and 1<=stats[i,2]<=13 and 1<=stats[i,3]<=13
                and .35<=stats[i,2]/stats[i,3]<=2.8]
        new=[]
        for point in points[:40]:
            candidates=[t for t in self.tracks if 1<=np.linalg.norm(np.asarray(point)-t[-1])<=45]
            path=(max(candidates,key=len)+[point]) if candidates else [point]
            new.append(path[-12:])
            if len(path)>=4 and np.linalg.norm(np.asarray(path[-1])-path[0])>12:
                self.ball_track=True
        self.tracks=new
        self.previous=current

    def report(self):
        return dict(scene_motion_observed=self.motion_frames>=3,scene_motion_frames=self.motion_frames,
                    peak_changed_fraction=round(self.peak_change,4),samples=self.samples,
                    moving_ball_candidate_tracked=self.ball_track,
                    note='A bright candidate alone is not proof of a shot; require a stable game-state advance.')


class Session:
    def shot_id(self,number):
        return self.path.name+':'+getattr(self,'scope','')+str(number)

    def __init__(self,root):
        self.root=Path(root)
        self.path=self.root/'wgt_debug'/'runs'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True)
        self.events_file=(self.path/'events.jsonl').open('a',encoding='utf-8',buffering=1)
        self.console_file=(self.path/'console.log').open('a',encoding='utf-8',buffering=1)
        self.status='started'
        self.meter_frames=deque(maxlen=100)
        self.console_pending=[]
        self.event('session_started',path=str(self.path),input_sent=False)

    def event(self,event,**data):
        self.events_file.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(),event=event,**data),default=str)+'\n')

    def shot_stage(self, stage, *, label=None, **data):
        data=dict(screen_id=getattr(self,'screen_id',None),**data)
        self.event(stage,**data)
        if label:
            self.write(f'{label} {json.dumps(data,sort_keys=True)}\n')

    def shot_observation(self, state, timestamp, yards_text, elevation_text):
        evidence=dict(hole=state.get('hole'),shot=state.get('shot'),timestamp=timestamp,
                      observation_source='live_ui')
        yards=state.get('distance') if state.get('unit')=='yd' else None
        elevation=state.get('elevation_ft')
        yardage=dict(raw_yards_text=yards_text if state.get('unit')!='ft' else [],
                     parsed_yards=yards,distance=state.get('distance'),
                     distance_unit=state.get('unit'),distance_source=state.get('distance_source'))
        height=dict(raw_elevation_text=elevation_text,parsed_elevation_feet=elevation,
                    elevation_detected=elevation is not None)
        self.shot_stage('yardage_read',**evidence,**yardage)
        self.shot_stage('elevation_read',**evidence,**height)
        self.shot_stage('shot_telemetry',label='SHOT_TELEMETRY',**evidence,**yardage,**height)

    def write(self,text):
        # Queue console text in memory during input; flush at safe phase ends.
        self.console_pending.append(text)

    def flush_console(self):
        self.console_file.write(''.join(self.console_pending))
        self.console_pending.clear()
        self.console_file.flush()

    def image(self,name,image):
        scope=getattr(self,'scope','')
        if scope and not name.startswith(scope):
            name=scope+name
        path=self.path/(name+'.png')
        if image is not None and image.size and not cv2.imwrite(str(path),image):
            raise OSError(f'Cannot save diagnostic image: {path}')
        return str(path)

    def state(self,name,state,frame=None):
        scope=getattr(self,'scope','')
        if scope and not name.startswith(scope):
            name=scope+name
        (self.path/(name+'.json')).write_text(json.dumps(state,indent=2,default=str),encoding='utf-8')
        if frame is not None:
            self.image(name,frame)

    def save_shot_capture(self, number, capture):
        name = f"{getattr(self, 'scope', '')}shot_{number:02d}_capture"
        self.state(name, capture)
        if capture.get('verification', {}).get('before_tigerwould52_turn') is not True:
            return
        try:
            record_capture(self.root / DATABASE, self.path / f'{name}.json', capture)
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            self.event('sqlite_shot_write_failed', shot_id=capture.get('shot_id'), reason=str(exc))

    def meter(self,phase,image,value):
        # In-memory only until the stroke's time-critical input is finished.
        self.meter_frames.append((phase,image.copy(),value))

    def flush_meters(self,shot):
        frames=[]
        for phase,img,value in self.meter_frames:
            preview=cv2.resize(img,(900,60))
            cv2.putText(preview,f'{phase}: {value}',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            frames.append(preview)
        if frames:
            self.image(f'shot_{shot:02d}_meter_trace',cv2.vconcat(frames))
        self.meter_frames.clear()

    def close(self,status,**data):
        self.status=status
        try:
            self.event('session_closed',status=status,**data)
            result={**data,'status':status,'path':str(self.path)}
            (self.path/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            (self.root/'wgt_debug'/'latest_run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            self.flush_console()
        finally:
            self.events_file.close()
            self.console_file.close()
