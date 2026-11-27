"""Offline-testable automation readers and decisions. No mouse, windows, or file writes."""
from collections import Counter, deque
from dataclasses import dataclass
import math
import re

import cv2
import numpy as np

from terrain_handler import BUNKER_BLOCK_REASON, is_bunker_state


@dataclass(frozen=True)
class Distance:
    value: float
    unit: str


def parse_distance(text):
    # Require a unit. Never let YDS become a trailing zero in a digit-only OCR.
    text = text.upper().replace(',', '.')
    match = re.search(r'(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*(YDS?|FT|FEET)\b', text)
    if not match:
        return None
    value = float(match[1])
    unit = 'ft' if match[2] in ('FT', 'FEET') else 'yd'
    if not math.isfinite(value) or not 0 < value <= (999 if unit == 'ft' else 650):
        return None
    return Distance(value, unit)


def consensus(values, minimum=2):
    values = [v for v in values if v is not None]
    if not values:
        return None
    ranked = Counter(values).most_common()
    if ranked[0][1] < minimum or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def parse_shot(text, par=None):
    match = re.search(r'\b(\d{1,2})\s*(?:ST|ND|RD|TH)\s*SHOT\b', text.upper())
    if match:
        return int(match[1])
    scores = {'EAGLE': -2, 'BIRDIE': -1, 'PAR': 0, 'BOGEY': 1,
              'DOUBLE BOGEY': 2, 'TRIPLE BOGEY': 3}
    if par is not None:
        match = re.search(r'\bFOR\s*(TRIPLE\s*BOGEY|DOUBLE\s*BOGEY|BOGEY|BIRDIE|EAGLE|PAR)\b', text.upper())
        if match:
            label = re.sub(r'(TRIPLE|DOUBLE)\s*BOGEY',r'\1 BOGEY',match[1])
            return par + scores[label]
    return None


def complete_text(text):
    # A permanent SCORECARD button or a FOR BOGEY label is not hole completion.
    return bool(re.search(r'\b(?:NEXT\s+HOLE|HOLE\s+COMPLETE|HOLED\s+OUT)\b', text.upper()))


@dataclass(frozen=True)
class StrokeReference:
    width: int
    height: int
    anchor: tuple
    pull: float
    drag_seconds: float
    accuracy_delay: float
    click_seconds: float
    samples: tuple

    @classmethod
    def from_data(cls, data):
        width, height = data['width'], data['height']
        if not isinstance(width, int) or not isinstance(height, int) or width < 600 or height < 400:
            raise ValueError('Invalid recording window geometry')
        seq = data['samples']
        if len(seq) < 4:
            raise ValueError('Recording is too short')
        transitions = []
        previous = False
        last_t = -1.0
        for i, entry in enumerate(seq):
            t, x, y, down = entry['t'], entry['x'], entry['y'], entry['down']
            if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (t, x, y))
                    or t < 0 or t <= last_t or type(down) is not bool
                    or not 0 <= x <= 1 or not 0 <= y <= 1):
                raise ValueError('Invalid, out-of-window or unordered stroke sample')
            if down != previous:
                transitions.append(i)
            previous, last_t = down, t
        if previous or len(transitions) != 4:
            raise ValueError('Recording needs exactly two complete down/up cycles')
        a, b, c, d = transitions
        start = seq[a]
        # The supplied recording starts on the bottom meter, not the aim flag.
        if not .17 <= start['x'] <= .89 or not .93 <= start['y'] <= .985:
            raise ValueError('Recorded press is not inside the bottom swing meter')
        pull = start['y'] - min(e['y'] for e in seq[a:b+1])
        drag = seq[b]['t'] - start['t']
        delay = seq[c]['t'] - seq[b]['t']
        if not .02 < pull < .80 or not .05 < drag < 15 or not .05 < delay < 8:
            raise ValueError('Recording has no valid upward drag and accuracy interval')
        return cls(width, height, (start['x'], start['y']), pull, drag, delay,
                   seq[d]['t'] - seq[c]['t'], tuple(seq[a:b+1]))


def meter_rect(rect):
    # Preserve the known working 12%/94% capture used by previous controllers.
    return dict(left=rect['left'] + int(rect['width']*.12),
                top=rect['top'] + int(rect['height']*.94),
                width=int(rect['width']*.82), height=max(24, int(rect['height']*.055)))


def _runs(xs, gap=1):
    if len(xs) == 0:
        return []
    return np.split(xs, np.flatnonzero(np.diff(xs) > gap) + 1)


def _band(image):
    if image is None or image.size == 0:
        return None
    h = image.shape[0]
    return image[int(h*.18):max(int(h*.94), 1)]


@dataclass(frozen=True)
class Meter:
    origin: int
    blue_left: int
    blue_right: int
    target: int
    width: int
    zero: int = None
    return_origin: int = None

    @property
    def green_origin(self):
        return self.return_origin if self.return_origin is not None else self.zero


def meter_outline(image):
    """Find paired long horizontal borders and their shared left endpoint.

    Club text can make tall neutral columns inside this capture. It cannot
    form the two long gauge borders, so never use column brightness alone.
    """
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    neutral=((gray>=55)&(hsv[:,:,1]<55)).astype(np.uint8)*255
    joined=cv2.morphologyEx(neutral,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
    kernel=max(21,int(.60*w))|1
    horizontal=cv2.morphologyEx(joined,cv2.MORPH_OPEN,np.ones((1,kernel),np.uint8))
    lines=[]
    for y in range(h):
        for run in _runs(np.flatnonzero(horizontal[y])):
            if len(run)>.65*w and .035*w<run[0]<.15*w:
                lines.append((int(run[0]),int(run[-1]),y))
    pairs=[]
    for a in lines:
        for b in lines:
            if not .25*h<=b[2]-a[2]<=.90*h:
                continue
            left_delta=abs(a[0]-b[0])
            right_delta=abs(a[1]-b[1])
            if left_delta>8 or right_delta>3:
                continue
            # Background pixels can extend one horizontal border. Recover
            # the inner corner only when both vertical sides are visible.
            extended=left_delta>3
            left=max(a[0],b[0]) if extended else min(a[0],b[0])
            vertical=neutral[a[2]:b[2]+1,max(0,left-1):left+2]
            if (vertical>0).any(axis=1).mean()<.55:
                continue
            if extended:
                right=min(a[1],b[1])
                vertical=neutral[a[2]:b[2]+1,right-1:right+2]
                if (vertical>0).any(axis=1).mean()<.55:
                    continue
            pairs.append(((a[1]-a[0])*(b[2]-a[2]),left,a[2],b[2]))
    return max(pairs)[1:] if pairs else None


def detect_meter(image, saved_target=None):
    band = _band(image)
    if band is None or not band.size:
        return None
    h, w = band.shape[:2]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (87, 32, 50), (130, 255, 255))
    counts = (blue > 0).sum(axis=0)
    xs = np.flatnonzero(counts >= max(2, int(h*.22)))
    groups = [g for g in _runs(xs, 7) if len(g) >= max(18, int(w*.12))
              and .30*w < g[0] < .85*w and g[-1] < .97*w]
    if not groups:
        return None
    bar = max(groups, key=len)
    left, right = int(bar[0]), int(bar[-1])
    target = int(round((left+right)/2))
    if saved_target is not None:
        if not left < saved_target < right or abs(saved_target-target) > max(8, .02*w):
            return None
        target = int(saved_target)
    # Detect both the full-power left limit and the neutral vertical marker.
    # Saved power_check frames show yellow expanding LEFT from neutral, not
    # growing right from the rectangle's left border.
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    neutral_border = (gray >= 55) & (hsv[:, :, 1] < 55)
    outline=meter_outline(image)
    if outline is None:
        return None
    origin = outline[0]
    if left-origin < .20*w:
        return None
    markers = np.flatnonzero(neutral_border.sum(axis=0) >= max(3, int(h*.40)))
    markers = markers[(markers > .30*w) & (markers < left-max(8,.015*w))]
    if not len(markers):
        return None
    zero = int(markers[-1])
    if zero-origin < .20*w:
        return None
    return Meter(origin, left, right, target, w, zero)


def yellow_power(image, meter):
    band = _band(image)
    if band is None or not band.size:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    # Reuse the red/green discrimination from the later putt power repair;
    # it separates yellow fill from green return and the blue target zone.
    b, g, r = [band[:, :, i].astype(float) for i in range(3)]
    mask = ((hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 43)
            & (hsv[:, :, 1] >= 65) & (r >= 34)
            & (r >= g*1.025) & (g >= b*1.45))
    filled = mask.sum(axis=0) >= max(2, int(band.shape[0]*.22))
    filled[:meter.origin+1] = False
    filled[meter.zero+1:] = False
    groups = [v for v in _runs(np.flatnonzero(filled), 7)
              if len(v) >= 3 and v[-1] >= meter.zero-max(12,.025*meter.width)]
    if not groups:
        return None  # Missing fill is unknown, not verified zero power.
    bar = max(groups, key=len)
    return float(np.clip((meter.zero-int(bar[0]))/(meter.zero-meter.origin), 0, 1))


def green_front(image, meter):
    band = _band(image)
    if band is None or not band.size:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 65, 45), (86, 255, 255))
    filled = ((green > 0).sum(axis=0) >= max(2, int(band.shape[0]*.23))).astype(np.uint8)
    filled = cv2.morphologyEx(filled[None, :], cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))[0]
    groups = [g for g in _runs(np.flatnonzero(filled))
              if abs(int(g[0])-meter.green_origin) <= max(10, .025*meter.width)
              and len(g) >= 5 and g[-1] < meter.blue_right]
    return int(max(groups, key=len)[-1]) if groups else None


class AccuracyTracker:
    """Require forward motion from the origin; predict one target crossing."""
    def __init__(self, meter, latency=.020):
        self.meter = meter
        self.latency = latency
        self.history = deque(maxlen=6)
        self.clicked = False
        self.last_prediction = None
        self.origin = meter.green_origin

    def update(self, now, edge):
        if self.clicked:
            return False
        if edge is None:
            if self.history and now-self.history[-1][0] > .15:
                self.history.clear()
            return False
        if not self.origin <= edge <= self.meter.blue_right:
            return False
        if not self.history:
            if edge > self.origin + .18*self.meter.width:
                return False  # Cannot join a late return with no observed origin.
        else:
            dt, dx = now-self.history[-1][0], edge-self.history[-1][1]
            if dt <= 0:
                return False
            if dt > .15 or dx < -3 or dx/dt > 2500:
                self.history.clear()
                return False
        self.history.append((now, edge))
        if len(self.history) < 3:
            return False
        first, last = self.history[0], self.history[-1]
        dt, dx = last[0]-first[0], last[1]-first[1]
        if dt < .008 or dx < 4:
            return False
        speed = dx/dt
        if not 20 <= speed <= 2500:
            return False
        # Predict up to one observed frame interval plus input latency. Cap
        # the lead to prevent a fast/noisy observation from clicking early.
        frame_dt = now-self.history[-2][0]
        lead = min(.05*self.meter.width, speed*(self.latency+frame_dt))
        prediction = edge+lead
        self.last_prediction = prediction
        if edge <= self.meter.target+5 and prediction >= self.meter.target:
            self.clicked = True
            return True
        return False


def plan_power(state, for_selection=False):
    # This generic ratio must never be an executable bunker fallback.
    if is_bunker_state(state):
        raise ValueError(BUNKER_BLOCK_REASON)
    distance, club = state.get('distance'), state.get('club_range')
    unit, club_unit = state.get('unit'), state.get('club_unit')
    if distance is None or club is None or distance <= 0 or club <= 0:
        raise ValueError('Distance and selected club range must both be verified')
    if unit != club_unit or unit not in ('yd', 'ft'):
        raise ValueError('Target and selected club units disagree')
    if state.get('putt') != (unit == 'ft'):
        raise ValueError('Putting mode and distance units disagree')
    # No special 100% rule for first strokes (par-3s and resumed putts exist).
    ratio = distance/club
    if ratio < .025 and not for_selection:
        raise ValueError('Selected range too large for a controlled short stroke')
    # Baseline only: actual carry/roll depends on lie, green speed and slope.
    return min(.99, ratio)


def resolve_target_distance(primary,hud,linked_to_hole=False,hud_resolution=1.):
    """Never confuse selected club range or an unrelated hole HUD with aim."""
    if primary is not None:
        return dict(distance=primary.value,unit=primary.unit,distance_source='floating_target',
                    distance_uncertainty=0.,distance_block_reason=None)
    if hud is not None and linked_to_hole:
        uncertainty=hud_resolution/2
        if uncertainty<=max(.15,hud.value*.05):
            return dict(distance=hud.value,unit=hud.unit,distance_source='verified_hole_hud',
                        distance_uncertainty=uncertainty,distance_block_reason=None)
        reason='Visible hole HUD is too rounded for this short target'
    else:
        reason='Target unreadable; no verified link between aim and the hole HUD'
    return dict(distance=None,unit=None,distance_source='floating_target',
                distance_uncertainty=None,distance_block_reason=reason)


def state_signature(state):
    return (state.get('hole'), state.get('shot'), state.get('distance'),
            state.get('unit'), state.get('putt'), state.get('club_range'), state.get('club_unit'))


class TransitionTracker:
    """Two stable ready observations after the shot; supports yd -> ft."""
    def __init__(self, before):
        self.before = before
        self.last = None
        self.count = 0

    def update(self, state, ready):
        if state.get('complete'):
            return 'complete'
        before = self.before
        hole_changed = (before.get('hole') is not None and state.get('hole') is not None
                        and before['hole'] != state['hole'])
        changed_shot = (before.get('shot') is not None and state.get('shot') is not None
                        and state['shot'] > before['shot'])
        # A readable stroke/hole change proves advancement even when the next
        # target badge is hidden. Distance-only changes can be aim changes and
        # must never authorize a repeat shot on the same stroke.
        valid = (ready and state.get('club_range') is not None
                 and (hole_changed or (changed_shot and state.get('hole')==before.get('hole')
                                      and state.get('hole') is not None)))
        signature = state_signature(state)
        if valid:
            self.count = self.count+1 if signature == self.last else 1
        else:
            self.count = 0
        self.last = signature
        if self.count >= 2:
            return 'complete' if hole_changed else 'next'
        return None
