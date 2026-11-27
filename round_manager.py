"""Verified scorecard and nine-hole sequencing. Native input stays in V48."""
import re
import time
import cv2


def scorecard(frame,ocr):
    h,w=frame.shape[:2]
    white=cv2.inRange(frame,(210,210,210),(255,255,255))
    contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        x,y,cw,ch=cv2.boundingRect(contour)
        if not (.35*w<cw<.65*w and .30*h<ch<.60*h and .18*w<x<.40*w and .20*h<y<.44*h):
            continue
        top=max(0,y-round(.055*h))
        header=' '.join(ocr(frame[top:y,x:x+cw],modes=(6,7),scale=3))
        if not all(re.search(r'\b'+word+r'\b',header) for word in ('SCORECARD','STATS','METER')):
            continue
        body=' '.join(ocr(frame[y:y+round(.55*ch),x:x+cw],modes=(6,),scale=3))
        if not all(re.search(r'\b'+word+r'\b',body) for word in ('PUTTS','PAR')):
            continue
        # A Continue control must actually be visible inside the result modal.
        area=frame[y+round(.70*ch):y+ch,x+round(.60*cw):x+cw]
        hsv=cv2.cvtColor(area,cv2.COLOR_BGR2HSV)
        gray=cv2.inRange(hsv,(0,0,60),(179,65,185))
        buttons,_=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for button in buttons:
            bx,by,bw,bh=cv2.boundingRect(button)
            if not (2<bw/max(bh,1)<5 and bw>.10*cw and bh>.025*ch):
                continue
            # Rounded button edges can OCR as arrows/letters. Read the text
            # interior while keeping the full verified rectangle for input.
            patch=area[by+round(.12*bh):by+bh-round(.12*bh),
                       bx+round(.10*bw):bx+bw-round(.10*bw)]
            words=[re.sub('[^A-Z]','',s.upper()) for s in ocr(patch,modes=(6,7),scale=4)]
            if len(words)<2 or any(s!='CONTINUE' for s in words):
                continue
            return dict(kind='completed_scorecard',rect=[x,top,cw,y+ch-top],
                        button=[x+round(.60*cw)+bx,y+round(.70*ch)+by,bw,bh],
                        header=header,body=body)
    return None


def hole_intro(frame, ocr, expected):
    """Identify the course-info panel for the exact next hole."""
    height, width = frame.shape[:2]
    white = cv2.inRange(frame, (235,)*3, (255,)*3)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.40*width < w < .55*width and .43*height < h < .60*height
                and .23*width < x < .35*width and .20*height < y < .32*height):
            continue
        banner = frame[y+round(.235*h):y+round(.30*h), x+round(.027*w):x+round(.44*w)]
        labels = ocr(banner, modes=(7,), scale=4)
        pattern = rf'\bHOLE\s+{expected}\s+PAR\s+[3-6]\b'
        if not any(re.search(pattern, label.upper()) for label in labels):
            continue
        bx, by = x+round(.79*w), y+round(.882*h)
        bw, bh = round(.17*w), round(.10*h)
        interior = frame[by+round(.28*bh):by+round(.78*bh), bx+round(.13*bw):bx+round(.87*bw)]
        words = [re.sub('[^A-Z]', '', label.upper()) for label in ocr(interior, modes=(6,7), scale=4)]
        if len(words) >= 2 and all(word == 'CONTINUE' for word in words):
            return {'kind': 'hole_intro', 'hole': expected, 'button': [bx, by, bw, bh]}
    return None


def next_hole(hole):
    if not isinstance(hole,int) or not 1<=hole<=18:
        raise RuntimeError('Current hole number is unverified')
    return hole%18+1


def continue_to_tee(live,safety,sct,rect,previous,timeout=90):
    """One guarded Continue click; never repeat it after an ambiguous outcome."""
    first=None
    for _ in range(2):
        safety.check()
        frame=live.capture(sct,rect)
        found=scorecard(frame,live.ocr_variants)
        if found is None or first is not None and found['button']!=first['button']:
            raise RuntimeError('Completed scorecard/Continue control could not be verified twice')
        first=found
        safety.wait(.25)
    _click_continue(live,safety,sct,rect,frame,first,'round_continue_verified',previous)
    return await_tee(live,safety,sct,rect,next_hole(previous),timeout)


def _click_continue(live,safety,sct,rect,frame,first,detection,hole):
    # Fresh pixel proof after OCR, before sending any input.
    safety.check()
    fresh=live.capture(sct,rect)
    x,y,w,h=first['button']
    if not (fresh[y:y+h,x:x+w]==frame[y:y+h,x:x+w]).all():
        raise RuntimeError('Continue button changed before input; no click')
    sx,sy=rect['left']+x+w//2,rect['top']+y+h//2
    if not live.point_in_wgt(safety.hwnd,sx,sy):
        raise RuntimeError('Continue control is occluded')
    live.diagnostic_state(detection,dict(hole=hole,panel=first),fresh)
    live.move_to(sx,sy)
    safety.check()
    safety.press()
    try:
        safety.wait(.035)
    finally:
        safety.release()


def await_tee(live,safety,sct,rect,expected,timeout=90,require_intro=False):
    started=time.perf_counter()
    intro_clicked=False
    while time.perf_counter()-started<timeout:
        safety.wait(.5)
        frame=live.capture(sct,rect)
        # Cheap gate: avoid full OCR during the loading/scorecard animation.
        if live.swing_button(frame,rect) is None or live.ready_meter(frame,rect) is None:
            if intro_clicked:
                continue
            panel=hole_intro(frame,live.ocr_variants,expected)
            if panel is not None:
                safety.check()
                confirm=live.capture(sct,rect)
                if hole_intro(confirm,live.ocr_variants,expected)!=panel:
                    continue  # A loading panel can settle without having received a click.
                _click_continue(live,safety,sct,rect,confirm,panel,'hole_intro_continue_verified',expected)
                intro_clicked=True
                print(f'Continue clicked for hole {expected}.',flush=True)
            continue
        if require_intro and not intro_clicked:
            raise RuntimeError('Expected a verified hole introduction before starting play')
        state,_=live.stable_state(safety,sct,rect,timeout=30,expected={'hole':expected,'shot':1})
        if state.get('hole')!=expected or state.get('shot')!=1:
            raise RuntimeError('Next tee identity disagrees; no swing')
        return state
    raise RuntimeError('Next tee not verified after Continue; no automatic retry')


def run_round(live,safety,sct,rect,reference,holes,max_shots):
    completed=[]
    for index in range(1,holes+1):
        live.DIAGNOSTICS.scope=f'hole_{index:02d}_'
        live.GAME.completed=False
        live.DIAGNOSTICS.event('round_hole_started',round_hole=index,requested_holes=holes)
        result=live.run_hole(safety,sct,rect,reference,max_shots)
        if result!='complete':
            raise RuntimeError('Hole completion is unverified; round cannot advance')
        current=getattr(live.GAME,'completed_hole',None)
        if not isinstance(current,int):
            raise RuntimeError('Completed hole number is unverified')
        if current in completed:
            raise RuntimeError('Duplicate completed hole; round cannot advance')
        completed.append(current)
        live.DIAGNOSTICS.state('round_progress',dict(requested=holes,completed_holes=completed))
        live.DIAGNOSTICS.event('round_hole_verified',round_hole=index,hole=current,completed_holes=completed)
        if index<holes:
            continue_to_tee(live,safety,sct,rect,current)
    return dict(outcome='round_complete',requested_holes=holes,completed_holes=completed)
