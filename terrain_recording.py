"""Manual bunker evidence and Shot Stats storage. No desktop/input APIs.

These observations are deliberately separate from power_observations.json.
Field agreement verifies displayed numbers, not sand physics or impact-time
conditions. Attribution also requires the player's explicit one-shot attestation.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid


BEFORE_FIELDS = ('club', 'club_range_yd', 'target_distance_yd', 'mode', 'lie',
                 'lie_percent', 'wind_mph', 'wind_screen_vector', 'elevation_ft')
STAT_FIELDS = ('carry_yd', 'roll_yd', 'power_percent')


def numeric(value, low, high):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and low <= value <= high)


def same_number(value, low, high):
    return value if numeric(value, low, high) else None


def interval(value, low, high):
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(numeric(x, low, high) for x in value) and value[0] <= value[1]):
        return list(value)
    return None


def strict_agreement(values, minimum=2):
    """Abstain on any conflicting valid reading; never majority-vote decimals."""
    values = [v for v in values if v is not None]
    return values[0] if len(values) >= minimum and all(v == values[0] for v in values) else None


def parse_stat_row(label, label_reads, number_reads):
    """Require the row label and explicit units, including distance decimals."""
    if not row_label_verified(label, label_reads):
        return None
    pattern = (r'(\d{1,3}(?:\.\d{1,2})?)\s*%' if label == 'POWER' else
               r'(\d{1,3}\.\d{1,2})\s*YDS?')
    maximum = {'CARRY': 600, 'ROLL': 200, 'POWER': 100}[label]
    votes = []
    for text in number_reads:
        # Transparent panel edges add punctuation after an intact explicit
        # unit. Accept that decoration, never extra digits/letters/decimals.
        match = re.fullmatch(pattern + r'[^A-Z0-9.]*', text.strip().upper())
        if match and numeric(float(match[1]), 0, maximum):
            votes.append(float(match[1]))
    return strict_agreement(votes)


def row_label_verified(label, reads):
    return sum(bool(re.fullmatch(r'[^A-Z0-9]*' + label + r'[^A-Z0-9]*', t.strip().upper())) for t in reads) >= 2


def read_manual_stats_patch(patch, ocr):
    """Read carry, roll and actual game power independently; spin is optional.

    The row geometry is the existing V41/V48 Shot Stats layout. Unlike the
    general reader, this never substitutes bare integers or requires apex/spin.
    Every failed field stays null, with its raw OCR retained for review.
    """
    h, w = patch.shape[:2]
    header = ocr(patch[:round(h*.238)], modes=(6, 7, 13), scale=4)
    raw = dict(header=header)
    result = dict.fromkeys(STAT_FIELDS)
    title_verified = sum(bool(re.search(r'\bSHOT\s*STATS\b', t.upper())) for t in header) >= 2
    labelled_rows = []
    for label, field, (lo, hi) in (
            ('CARRY', 'carry_yd', (.238, .369)),
            ('ROLL', 'roll_yd', (.361, .492)),
            ('POWER', 'power_percent', (.475, .615))):
        row = patch[round(h*lo):round(h*hi)]
        labels = ocr(row[:, :round(w*.405)], modes=(6, 7), scale=6)
        numbers = ocr(row[:, round(w*.405):round(w*.753)], modes=(7, 6), scale=8)
        raw[field] = dict(labels=labels, numbers=numbers)
        labelled_rows.append(row_label_verified(label, labels))
        result[field] = parse_stat_row(label, labels, numbers)
    # The transparent title can be illegible while all three measured row
    # labels remain intact. That is independent panel evidence, not a guessed
    # title; each numeric field still needs its own exact label and units.
    panel_verified = title_verified or all(labelled_rows)
    if not panel_verified:
        result = dict.fromkeys(STAT_FIELDS)
    return dict(values=result, raw=raw, panel_verified=panel_verified,
                panel_basis='title' if title_verified else 'three exact row labels' if panel_verified else None)


def local_solo(snapshot):
    state, context = snapshot['state'], snapshot['context']
    evidence = context.get('solo_evidence') or {}
    return (state.get('player') is True and snapshot.get('ready') is True
            and context.get('format') == 'SOLO_STROKE'
            and context.get('source') == 'solo_user_confirmed_live_hud'
            and context.get('players') == ['tigerwould52']
            and context.get('active_player') == 'tigerwould52'
            and context.get('ball_owner') == 'tigerwould52'
            and context.get('turn_state') == 'OWN_READY'
            and context.get('ready_controls') is True
            and context.get('solo_session_confirmed') is True
            and not context.get('teams')
            and all(evidence.get(k) is True for k in
                    ('user_confirmed', 'player_header_visible', 'swing_and_meter_visible',
                     'multiplayer_conflicts_absent')))


def before_values(snapshot):
    s, c = snapshot['state'], snapshot['context']
    wind = s.get('wind') or {}
    speed = interval([wind.get('low_mph'), wind.get('high_mph')], 0, 100)
    if speed is None:
        # read_conditions keeps speed OCR even if the direction arrow fails.
        from vision_engine import parse_wind
        speed = strict_agreement([parse_wind(t) for t in (s.get('conditions_raw') or {}).get('wind', [])])
        speed = interval(speed, 0, 100)
    dx, dy = wind.get('screen_dx'), wind.get('screen_dy')
    vector = None
    if numeric(dx, -1, 1) and numeric(dy, -1, 1):
        norm = math.hypot(dx, dy)
        if .9 <= norm <= 1.05:
            vector = [dx, dy]
    confidence = c.get('confidence') or {}
    lie = c.get('lie') if numeric(confidence.get('lie'), .85, 1) else None
    percent = interval(c.get('lie_percent'), 1, 100) if numeric(confidence.get('lie_percent'), .85, 1) else None
    return dict(club=s.get('club_name') if s.get('club_name') == 'LW' else None,
                club_range_yd=same_number(s.get('club_range'), .01, 650) if s.get('club_unit') == 'yd' else None,
                target_distance_yd=same_number(s.get('distance'), .01, 650) if (
                    s.get('unit') == 'yd' and s.get('distance_source') == 'floating_target') else None,
                mode=s.get('mode') if s.get('mode') in ('FULL', 'CHIP', 'PITCH', 'FLOP', 'PUNCH') else None,
                lie=lie if lie == 'BUNKER' else None, lie_percent=percent,
                wind_mph=speed, wind_screen_vector=vector,
                elevation_ft=same_number(s.get('elevation_ft'), -300, 300))


def stats_values(snapshot):
    values = snapshot.get('stats', {}).get('values', {})
    return {key: same_number(values.get(key), 0, maximum)
            for key, maximum in (('carry_yd', 600), ('roll_yd', 200), ('power_percent', 100))}


def paired_values(pair, extractor):
    a, b = map(extractor, pair)
    return {key: a[key] if a[key] is not None and a[key] == b[key] else None for key in a}


def pair_identity(pair):
    result = {}
    for key, low, high in (('hole', 1, 18), ('shot', 1, 30)):
        a, b = [snapshot['state'].get(key) for snapshot in pair]
        result[key] = a if (numeric(a, low, high) and numeric(b, low, high)
                           and a == int(a) and b == int(b) and a == b) else None
    return result


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class ManualBunkerStore:
    def __init__(self, root, max_age=600):
        self.root = Path(root).resolve()
        self.folder = self.root/'wgt_calibration/manual_bunker'
        self.pending_path = self.folder/'pending.json'
        self.max_age = max_age

    def pending(self):
        if not self.pending_path.is_file():
            return None
        data = json.loads(self.pending_path.read_text(encoding='utf-8-sig'))
        if data.get('schema') != 1:
            raise ValueError('Unknown manual bunker pending schema')
        return data if data.get('status') == 'awaiting_after' else None

    def validate_pair(self, pair, now):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('Two independently captured frames are required')
        times, ids = [], []
        for snapshot in pair:
            stamp = snapshot.get('captured_at')
            if not numeric(stamp, now-self.max_age, now):
                raise ValueError('Capture is stale, future-dated or missing its time')
            evidence = snapshot.get('evidence') or {}
            path = (self.root / evidence.get('path', '')).resolve()
            allowed = self.root/'automation_debug/manual_bunker'
            if (not path.is_relative_to(allowed) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != evidence.get('sha256')):
                raise ValueError('Capture evidence missing, changed or outside the manual evidence folder')
            if not snapshot.get('capture_id') or not isinstance(snapshot.get('state'), dict) or not isinstance(snapshot.get('context'), dict):
                raise ValueError('Capture identity or OCR state missing')
            times.append(stamp)
            ids.append(snapshot['capture_id'])
        if ids[0] == ids[1] or not 0 < times[1]-times[0] <= 30:
            raise ValueError('Frames must be separate captures no more than 30 seconds apart')

    def stage(self, pair, solo_confirmed=False, now=None):
        now = time.time() if now is None else now
        if self.pending():
            raise ValueError('A BEFORE capture is pending; finish or discard it explicitly')
        self.validate_pair(pair, now)
        if solo_confirmed is not True or not all(local_solo(s) for s in pair):
            raise ValueError('User-confirmed solo play and independently visible local ready HUD are required')
        fields = paired_values(pair, before_values)
        identity = pair_identity(pair)
        if fields['club'] != 'LW' or fields['lie'] != 'BUNKER' or None in identity.values():
            raise ValueError('BEFORE must verify LW, BUNKER and the same hole/stroke on both frames')
        pending = dict(schema=1, status='awaiting_after', id=uuid.uuid4().hex,
                       created_at_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                       before=pair, before_fields=fields, before_identity=identity,
                       baseline_stats=paired_values(pair, stats_values),
                       solo_confirmed=True, use_for_automatic_power=False)
        atomic_json(self.pending_path, pending)
        return pending

    def finish(self, pair, one_manual_shot_confirmed=False, setup_unchanged_confirmed=False, now=None):
        now = time.time() if now is None else now
        pending = self.pending()
        if not pending:
            raise ValueError('No pending BEFORE capture; do not reconstruct a missing setup')
        result_path = self.folder/'records'/(pending['id'] + '.json')
        if result_path.is_file():  # Recover a crash after commit without making a duplicate.
            pending['status'] = 'recorded'
            atomic_json(self.pending_path, pending)
            return json.loads(result_path.read_text(encoding='utf-8'))
        self.validate_pair(pending['before'], now)
        self.validate_pair(pair, now)
        if len({s['capture_id'] for s in pending['before'] + pair}) != 4:
            raise ValueError('BEFORE and AFTER must use four separate captures')
        if pair[0]['captured_at'] <= pending['before'][-1]['captured_at']:
            raise ValueError('AFTER must be captured after BEFORE')
        fields = dict(paired_values(pending['before'], before_values), **paired_values(pair, stats_values))
        identity = pair_identity(pair)
        reasons = []
        if pending.get('solo_confirmed') is not True or not all(local_solo(s) for s in pending['before']):
            reasons.append('BEFORE local solo ready HUD/owner unverified')
        if one_manual_shot_confirmed is not True or setup_unchanged_confirmed is not True:
            reasons.append('exactly one manual shot with unchanged setup was not confirmed')
        if not all(local_solo(s) for s in pair):
            reasons.append('AFTER local solo ready HUD/owner unverified')
        before_identity = pair_identity(pending['before'])
        if None in before_identity.values():
            raise ValueError('Stored BEFORE stroke identity is invalid')
        if identity != dict(hole=before_identity['hole'], shot=before_identity['shot']+1):
            reasons.append('same hole and exactly one stroke increment unverified')
        baseline = paired_values(pending['before'], stats_values)
        after_stats = {key: fields[key] for key in STAT_FIELDS}
        if None in baseline.values() or None in after_stats.values():
            reasons.append('fresh Shot Stats comparison incomplete')
        elif baseline == after_stats:
            reasons.append('Shot Stats unchanged from BEFORE; possible stale result')
        attribution_verified = not reasons
        missing = [key for key, value in fields.items() if value is None]
        record = dict(schema=1, id=pending['id'], kind='manual_bunker_shot',
                      recorded_at_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                      status='verified' if attribution_verified and not missing else 'partial',
                      measurement_verified=attribution_verified and not missing,
                      attribution_verified=attribution_verified,
                      attribution_method='user attestation + two local HUD reads + one stroke increment + changed Shot Stats',
                      motion_evidence=None, owner='tigerwould52' if attribution_verified else None,
                      fields=fields, field_verified={k: v is not None for k, v in fields.items()},
                      missing_fields=missing, rejection_reasons=reasons,
                      total_yd=round(fields['carry_yd']+fields['roll_yd'], 2) if (
                          fields['carry_yd'] is not None and fields['roll_yd'] is not None) else None,
                      total_yd_source='carry plus roll, derived' if (
                          fields['carry_yd'] is not None and fields['roll_yd'] is not None) else None,
                      conditions_basis='observed BEFORE the manual shot; not measured at impact',
                      power_basis='WGT Shot Stats POWER, not requested power or observed meter fraction',
                      lie_percent_semantics='uninterpreted HUD percentage',
                      use_for_automatic_power=False, before=pending['before'], after=pair,
                      before_identity=before_identity, after_identity=identity,
                      manual_attestation=dict(one_shot=one_manual_shot_confirmed,
                                              setup_unchanged=setup_unchanged_confirmed))
        # The CLI holds ControllerLock across the entire session. Commit the
        # result before retiring pending state so an interrupted save can resume.
        atomic_json(result_path, record)
        pending.update(status='recorded', result_path=str(result_path.relative_to(self.root)))
        atomic_json(self.pending_path, pending)
        return record

    def discard(self):
        pending = self.pending()
        if pending:
            pending['status'] = 'discarded'
            archive = self.folder/'discarded'/(pending['id'] + '.json')
            atomic_json(archive, pending)
            atomic_json(self.pending_path, pending)
