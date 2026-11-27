"""Persistent, evaluated golf learning. All recommendations are SHADOW ONLY.

Only data/model artifacts are written. No source rewriting, Windows calls,
capture, mouse/keyboard input, network calls or live-model activation exist.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import uuid

import numpy as np

from learning_data import (automatic_record, baseline_descriptor, baseline_distance,
    compact_state, condition_features, digest, group_key, manual_record, number, verified_lie)
from terrain_recording import atomic_json

ALGORITHM = 'contextual-monotone-ridge-v1'
PROTOCOL = dict(version='session-split-v1', minimum_train=24, minimum_validation=8, minimum_audit=8,
                training_sessions=3, validation_sessions=2, audit_sessions=2,
                minimum_power_span=.20, minimum_baseline_improvement=.10, ridge=.001)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def split_for(session_id):
    bucket = int(digest(['golf-learning-split-v1', session_id])[:8], 16) % 10
    return 'train' if bucket < 6 else 'validation' if bucket < 8 else 'audit'


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _input(state, power):
    if not number(power, .025, 1):
        raise ValueError('actual power outside verified fitting range')
    conditions = condition_features(state)
    return dict(power=power, target_ratio=state['distance']/state['club_range'], **conditions)


def fit_model(rows):
    """Learn forward distance from action power and verified context, not target.

    y/range = p*(b0 + b1*p + sum(bj*condition_j)). Target distance is used
    only for support checks and inverse planning, never as a fitted outcome
    feature. Zero power has zero predicted travel. Ridge fits only training rows.
    """
    states = [r['state'] for r in rows]
    inputs = [_input(s, r['actual_game_power']) for s, r in zip(states, rows)]
    if len({tuple(sorted(x)) for x in inputs}) != 1:
        raise ValueError('feature schemas differ; contexts cannot be pooled')
    if max(x['power'] for x in inputs)-min(x['power'] for x in inputs) < PROTOCOL['minimum_power_span']:
        raise ValueError('insufficient verified power variation')
    conditions = [k for k in sorted(inputs[0]) if k not in ('power', 'target_ratio')
                  and max(x[k] for x in inputs)-min(x[k] for x in inputs) > 1e-7]
    features = ['power', 'power_squared'] + conditions
    if len(rows) < max(PROTOCOL['minimum_train'], len(features)*4):
        raise ValueError('too few independent training observations for feature count')
    matrix = np.asarray([[x['power'], x['power']**2] + [x['power']*x[k] for k in conditions] for x in inputs])
    if np.linalg.matrix_rank(matrix) < len(features):
        raise ValueError('conditions are confounded; independent variation is insufficient')
    y = np.asarray([r['actual']['travel']/s['club_range'] for r, s in zip(rows, states)])
    prior = np.zeros(len(features)); prior[0] = 1.
    ridge = PROTOCOL['ridge']
    coefficients = np.linalg.solve(matrix.T@matrix+ridge*np.eye(len(features)), matrix.T@y+ridge*prior)
    bounds = {k: [min(x[k] for x in inputs), max(x[k] for x in inputs)] for k in inputs[0]}
    # Analytic worst-case derivative over the entire supported context box.
    minimum_slope = coefficients[0] + min(2*coefficients[1]*p for p in bounds['power'])
    minimum_slope += sum(min(coefficients[i+2]*v for v in bounds[k]) for i, k in enumerate(conditions))
    if not np.isfinite(coefficients).all() or minimum_slope <= .01:
        raise ValueError('non-monotone or unstable distance response rejected')
    rolls = [r['actual']['roll_yd'] for r in rows if number(r['actual'].get('roll_yd'), 0, 200)]
    area_labels = [r['actual']['recovery_failed'] for r in rows
                   if r.get('verified', {}).get('intended_area') is True
                   and isinstance(r['actual'].get('recovery_failed'), bool)]
    return dict(algorithm=ALGORITHM, group=rows[0]['group'], features=features,
                coefficients=coefficients.tolist(), bounds=bounds, unit=states[0]['unit'],
                roll_envelope=float(np.quantile(rolls, .95)) if states[0]['mode'] in ('FULL', 'FLOP', 'PUNCH') and rolls else 0.,
                recovery_labels=len(area_labels),
                observed_recovery_failure_rate=sum(area_labels)/len(area_labels) if len(area_labels) >= 5 else None,
                minimum_slope=float(minimum_slope))


def forward(model, state, power, enforce_support=True):
    if group_key(state) != model['group']:
        raise ValueError('club/mode/lie/range/units differ from model')
    values = _input(state, power)
    if set(values) != set(model['bounds']):
        raise ValueError('unverified or incompatible context feature')
    if enforce_support and any(not lo-1e-8 <= values[k] <= hi+1e-8 for k, (lo, hi) in model['bounds'].items()):
        raise ValueError('outside measured condition/power/distance support; no extrapolation')
    p = values['power']
    vector = [p, p*p] + [p*values[k] for k in model['features'][2:]]
    result = float(np.dot(vector, model['coefficients'])*state['club_range'])
    if not number(result, 0, 5000):
        raise ValueError('invalid learned distance')
    return result


def inverse(model, state, target):
    if not number(target, .01, 1000):
        raise ValueError('invalid intended distance')
    query = dict(state, distance=target)
    low, high = model['bounds']['power']
    a, b = forward(model, query, low), forward(model, query, high)
    if not a <= target <= b:
        raise ValueError('target beyond verified power envelope')
    for _ in range(48):
        mid = (low+high)/2
        if forward(model, query, mid) < target:
            low = mid
        else:
            high = mid
    return (low+high)/2


def error_metrics(errors):
    values = np.asarray(errors, dtype=float)
    if not len(values):
        return None
    return dict(count=len(values), mae=float(np.mean(np.abs(values))), bias=float(np.mean(values)),
                p90=float(np.quantile(np.abs(values), .9)), worst=float(np.max(np.abs(values))))


def score(model, rows):
    errors, baselines = [], []
    for row in rows:
        if not number(row.get('baseline_at_actual_power'), 0, 5000):
            raise ValueError('held-out baseline prediction unavailable')
        value = forward(model, row['state'], row['actual_game_power'])
        actual = row['actual']['travel']
        errors.append(value-actual)
        baselines.append(row['baseline_at_actual_power']-actual)
    return dict(learned=error_metrics(errors), baseline=error_metrics(baselines))


def beats_baseline(metrics):
    a, b = metrics['learned'], metrics['baseline']
    return (b['mae'] > 1e-8 and a['mae'] <= b['mae']*(1-PROTOCOL['minimum_baseline_improvement'])
            and a['p90'] <= b['p90']+1e-8 and a['worst'] <= b['worst']+1e-8)


def no_worse(a, b):
    return all(a[k] <= b[k]+1e-8 for k in ('mae', 'p90', 'worst'))


class ShadowLearning:
    def __init__(self, root, session_id=None):
        self.root = Path(root).resolve()
        self.folder = self.root/'wgt_calibration/learning'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or 'offline-'+uuid.uuid4().hex
        self.db = sqlite3.connect(self.folder/'memory.sqlite3', timeout=5)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS shots(id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            partition TEXT NOT NULL, context_group TEXT, eligible INTEGER NOT NULL,
            collected_at TEXT NOT NULL, data TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS shot_context ON shots(context_group,eligible);
          CREATE TABLE IF NOT EXISTS predictions(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS models(version TEXT PRIMARY KEY, context_group TEXT NOT NULL,
            created_at TEXT NOT NULL, accepted_shadow INTEGER NOT NULL, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS champions(context_group TEXT PRIMARY KEY, version TEXT NOT NULL);
        ''')
        schema = self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if schema and schema[0] != '1':
            self.db.close()
            raise ValueError('Unknown shot-memory schema; no migration guessed')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema','1')")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('execution_mode','shadow_only')")
        self.options = defaultdict(dict)

    def close(self):
        self.db.close()

    def rows(self, group=None, eligible=False):
        where, params = [], []
        if group is not None:
            where.append('context_group=?'); params.append(group)
        if eligible:
            where.append('eligible=1')
        query = 'SELECT data FROM shots' + (' WHERE '+' AND '.join(where) if where else '') + ' ORDER BY collected_at,id'
        return [json.loads(r[0]) for r in self.db.execute(query, params)]

    def champion(self, group):
        row = self.db.execute('SELECT m.data FROM models m JOIN champions c ON c.version=m.version WHERE c.context_group=?',
                              (group,)).fetchone()
        return json.loads(row[0]) if row else None

    def remember(self, record, retrain=True):
        record = deepcopy(record)
        if not isinstance(record.get('id'), str) or not record['id'] or not record.get('session_id'):
            raise ValueError('shot and independent session identities are mandatory')
        record['partition'] = split_for(record['session_id'])
        record['use_for_live_control'] = False
        # Defense in depth: a caller cannot turn incomplete rows into training
        # samples by changing a saved boolean. Keep the reasons in shot memory.
        reasons = list(record.get('exclusions', []))
        try:
            condition_features(record['state'])
            _input(record['state'], record.get('actual_game_power'))
            if record.get('group') != group_key(record['state']):
                raise ValueError('context key does not match verified state')
            expected_basis = 'total' if record['state']['mode'] in ('CHIP', 'PITCH', 'PUTT') else 'carry'
            if (record.get('verified', {}).get('outcome_attributed') is not True
                    or record['actual'].get('travel_basis') != expected_basis
                    or record['actual'].get('unit') != record['state']['unit']
                    or not number(record['actual'].get('travel'), 0, 3000)):
                raise ValueError('actual distance/basis/owner attribution unverified')
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append(str(exc))
        record['exclusions'] = sorted(set(reasons))
        record['eligible'] = record.get('eligible') is True and not reasons
        with self.db:
            cursor = self.db.execute('INSERT OR IGNORE INTO shots VALUES (?,?,?,?,?,?,?)',
                (record['id'], record['session_id'], record['partition'], record.get('group'), int(record['eligible']),
                 utc_now(), _json(record)))
        if cursor.rowcount and retrain and record['eligible']:
            self.rebuild(record['group'])
        return bool(cursor.rowcount)

    def rebuild(self, group):
        rows = self.rows(group, eligible=True)
        version = digest([ALGORITHM, PROTOCOL, group, rows])[:24]
        existing = self.db.execute('SELECT data FROM models WHERE version=?', (version,)).fetchone()
        if existing:
            return json.loads(existing[0])
        partitions = {name: [r for r in rows if r['partition'] == name] for name in ('train', 'validation', 'audit')}
        report = dict(version=version, algorithm=ALGORITHM, protocol=PROTOCOL, group=group,
                      status='rejected', reasons=[], params=None, metrics={}, unit=None,
                      training_ids=[r['id'] for r in partitions['train']],
                      validation_ids=[r['id'] for r in partitions['validation']],
                      audit_ids=[r['id'] for r in partitions['audit']], live_control_allowed=False)
        try:
            for name, minimum, sessions in (('train', PROTOCOL['minimum_train'], PROTOCOL['training_sessions']),
                    ('validation', PROTOCOL['minimum_validation'], PROTOCOL['validation_sessions']),
                    ('audit', PROTOCOL['minimum_audit'], PROTOCOL['audit_sessions'])):
                if len(partitions[name]) < minimum or len({r['session_id'] for r in partitions[name]}) < sessions:
                    raise ValueError('insufficient '+name+' measurements/independent sessions')
            model = fit_model(partitions['train'])
            report.update(params=model, unit=model['unit'])
            incumbent = self.champion(group)
            for name in ('validation', 'audit'):
                report['metrics'][name] = score(model, partitions[name])
                if not beats_baseline(report['metrics'][name]):
                    raise ValueError(name+' did not improve baseline MAE by 10% without worsening tail/worst error')
                if incumbent:
                    try:
                        old_score = score(incumbent['params'], partitions[name])['learned']
                    except ValueError:
                        old_score = None  # Prior model no longer covers the expanded supported evidence.
                    if old_score and not no_worse(report['metrics'][name]['learned'], old_score):
                        raise ValueError(name+' less reliable than incumbent shadow model')
            report['status'] = 'accepted_shadow'
        except (ValueError, np.linalg.LinAlgError) as exc:
            report['reasons'].append(str(exc))
        with self.db:
            self.db.execute('INSERT INTO models VALUES (?,?,?,?,?)',
                            (version, group, utc_now(), int(report['status'] == 'accepted_shadow'), _json(report)))
            if report['status'] == 'accepted_shadow':
                self.db.execute('INSERT OR REPLACE INTO champions VALUES (?,?)', (group, version))
            else:
                incumbent = self.champion(group)
                if incumbent:
                    try:
                        reliable = all(beats_baseline(score(incumbent['params'], partitions[name]))
                                       for name in ('validation', 'audit'))
                    except (ValueError, TypeError):
                        reliable = False
                    if not reliable:
                        self.db.execute('DELETE FROM champions WHERE context_group=?', (group,))
        return report

    def note_club(self, state):
        if not verified_lie(state) or state.get('player') is not True:
            return
        group = group_key(state)
        if group:
            self.options[(state.get('hole'), state.get('shot'))][group] = compact_state(state)

    def recommend(self, state):
        """Rank only clubs actually observed in this shot's existing selection path."""
        target = (state.get('strategy') or {}).get('target', state.get('distance'))
        tee = state.get('shot') == 1 and state.get('mode') == 'FULL' and state.get('lie') == 'TEE'
        result = dict(mode='shadow_only', live_control_allowed=False, policy_preserved=True,
                      tee_policy_preserved=tee, selected=None, candidates=[], abstentions=[],
                      aim_adjustment=False, hazard_claim='unverified unless labelled current-aim corridor exists')
        if tee:
            result.update(selected=dict(club=state.get('club_name'), power=1., source='existing first-tee policy'))
            return result
        if not number(target, .01, 1000):
            result['abstentions'].append('target unverified')
            return result
        candidates = dict(self.options.get((state.get('hole'), state.get('shot')), {}))
        if group_key(state):
            candidates[group_key(state)] = compact_state(state)
        c = state.get('game_context') or {}
        hazards = state.get('verified_hazards', [])
        if (not number(c.get('confidence', {}).get('hazards'), .85, 1) or c.get('hazards') != hazards):
            hazards = []
        hazards = [h for h in hazards if h.get('unit') == state.get('unit') == 'yd' and h.get('current_aim') is True
                   and h.get('kind') in ('WATER', 'BUNKER') and number(h.get('start'), 0, 650)
                   and number(h.get('end'), 0, 650) and h['start'] < h['end']]
        if hazards:
            result['hazard_claim'] = 'verified labelled current-aim corridors; lateral geometry unknown'
        for group, option in candidates.items():
            if (option.get('mode') != state.get('mode') or option.get('lie') != state.get('lie')
                    or option.get('unit') != state.get('unit')):
                continue
            champion = self.champion(group)
            if not champion:
                result['abstentions'].append(dict(group=group, reason='no accepted, independently evaluated model'))
                continue
            try:
                # Do not reuse a wind/elevation reading taken at another club
                # selection time; apply current verified conditions to each option.
                option.update({k: deepcopy(state.get(k)) for k in
                               ('distance', 'wind', 'elevation_ft', 'lie_percent', 'lie_loss_percent', 'green_speed')})
                band = max(champion['metrics'][name]['learned']['p90'] for name in ('validation', 'audit'))
                roll_envelope = champion['params']['roll_envelope']
                hazard_envelope = band+roll_envelope
                safe_target = target
                for hazard in hazards:
                    if target+hazard_envelope >= hazard['start'] and target-hazard_envelope <= hazard['end']:
                        safe_target = min(safe_target, max(.01, hazard['start']-max(10., hazard_envelope)))
                power = inverse(champion['params'], option, safe_target)
                predicted = forward(champion['params'], dict(option, distance=safe_target), power)
                residual_hazard_risk = any(predicted+hazard_envelope >= h['start'] and predicted-hazard_envelope <= h['end'] for h in hazards)
                failure_rate = champion['params']['observed_recovery_failure_rate']
                item = dict(club=option['club_name'], club_range=option['club_range'], power=power,
                            target=safe_target, predicted_distance=predicted, unit=option['unit'],
                            model_version=champion['version'], empirical_error_band=band,
                            empirical_roll_envelope=roll_envelope, observed_recovery_failure_rate=failure_rate,
                            verified_recovery_labels=champion['params']['recovery_labels'],
                            error_band_is_guaranteed_coverage=False, hazard_overlap=residual_hazard_risk,
                            score=abs(predicted-safe_target)+band+(safe_target*failure_rate if failure_rate is not None else 0)
                                  +(1000 if residual_hazard_risk else 0))
                result['candidates'].append(item)
            except ValueError as exc:
                result['abstentions'].append(dict(group=group, reason=str(exc)))
        safe = [r for r in result['candidates'] if not r['hazard_overlap']]
        if safe:
            result['selected'] = min(safe, key=lambda r: r['score'])
        if state.get('lie') == 'BUNKER':
            result['bunker_execution'] = 'blocked unconditionally; independent manual evidence only'
        return result

    def begin_shot(self, shot_id, state, plan, calibration=None):
        existing = self.db.execute('SELECT data FROM predictions WHERE id=?', (shot_id,)).fetchone()
        if existing:
            return json.loads(existing[0])
        self.note_club(state)
        query = compact_state(state)
        if not verified_lie(state):
            query['lie'] = None
        recommendation = self.recommend(state) if verified_lie(state) else dict(
            mode='shadow_only', live_control_allowed=False, selected=None, abstentions=['lie unverified'])
        champion = self.champion(group_key(query)) if group_key(query) else None
        prediction = None
        if champion:
            try:
                prediction = forward(champion['params'], query, plan['power'])
            except (ValueError, KeyError):
                pass
        descriptor = baseline_descriptor(plan, calibration, state)
        snapshot = dict(id=shot_id, session_id=self.session_id, state=query,
                        captured_before=deepcopy(state),
                        selected_power=plan.get('power'), baseline_descriptor=descriptor,
                        baseline_version=digest(descriptor)[:16],
                        baseline_distance=baseline_distance(query, plan.get('power'), descriptor),
                        learned_model_version=champion['version'] if champion else None,
                        learned_distance=prediction, recommendation=recommendation, live_decision_changed=False)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO predictions VALUES (?,?,?)', (shot_id, utc_now(), _json(snapshot)))
        return snapshot

    def finish_shot(self, shot_id, before, after, observed_power, plan, evidence=None,
                    session_id=None, retrain=True, write_report=True):
        existing = self.db.execute('SELECT data FROM shots WHERE id=?', (shot_id,)).fetchone()
        if existing:
            return json.loads(existing[0])
        row = self.db.execute('SELECT data FROM predictions WHERE id=?', (shot_id,)).fetchone()
        frozen = json.loads(row[0]) if row else None
        record = automatic_record(shot_id, session_id or self.session_id, before, after, observed_power, plan, frozen, evidence)
        if frozen and frozen['session_id'] != record['session_id']:
            record['eligible'] = False
            record['exclusions'].append('prediction/result session identities disagree')
        actual = record['actual']['travel']
        record['prediction_errors'] = dict(baseline=None, learned=None, learned_at_actual_power=None)
        if actual is not None and frozen:
            for kind, key in (('baseline', 'baseline_distance'), ('learned', 'learned_distance')):
                if number(frozen.get(key), 0, 5000):
                    record['prediction_errors'][kind] = frozen[key]-actual
            version = frozen.get('learned_model_version')
            stored = self.db.execute('SELECT data FROM models WHERE version=?', (version,)).fetchone() if version else None
            if stored and number(record.get('actual_game_power'), .025, 1):
                try:
                    params = json.loads(stored[0])['params']
                    record['prediction_errors']['learned_at_actual_power'] = forward(
                        params, record['state'], record['actual_game_power'])-actual
                except ValueError:
                    pass
        self.remember(record, retrain=retrain)
        if write_report:
            self.write_report()
        return record

    def sync_existing(self):
        """Read existing session/manual files. No screenshots or game access."""
        added, issues = 0, []
        for path in sorted((self.root/'wgt_debug/runs').glob('*/*shot_*_result.json')):
            match = re.fullmatch(r'(?:(hole_\d+_))?shot_(\d+)_result\.json', path.name)
            if not match:
                continue
            shot_id = path.parent.name+':'+(match[1] or '')+str(int(match[2]))
            if self.db.execute('SELECT 1 FROM shots WHERE id=?', (shot_id,)).fetchone():
                continue
            try:
                payload = json.loads(path.read_text(encoding='utf-8-sig'))
                plan_path = path.with_name(path.name.replace('_result.json', '_plan.json'))
                planned = json.loads(plan_path.read_text(encoding='utf-8-sig')) if plan_path.is_file() else {}
                evidence=dict(result_path=str(path.relative_to(self.root)), result_sha256=digest(payload))
                # New logs archive their trace and capture together. Old rows
                # remain untouched; saved completeness booleans grant no trust.
                capture=payload.get('shot_capture')
                if (isinstance(capture,dict) and capture.get('schema')==1
                        and capture.get('kind')=='new_shot_capture' and capture.get('shot_id')==shot_id):
                    trace=payload.get('accuracy_trace')
                    if isinstance(trace,dict) and trace.get('shot_id')==shot_id:
                        evidence['accuracy_trace']=trace
                    from shot_capture import build_shot_capture
                    evidence['shot_capture']=build_shot_capture(shot_id,payload.get('before') or {},
                        payload.get('after'),payload.get('observed_power'),planned.get('plan') or {},
                        evidence.get('accuracy_trace'),payload.get('outcome'))
                self.finish_shot(shot_id, payload.get('before') or {}, payload.get('after'),
                    payload.get('observed_power'), planned.get('plan') or {}, session_id=path.parent.name,
                    evidence=evidence,
                    retrain=False, write_report=False)
                added += 1
            except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError) as exc:
                issues.append(dict(path=str(path.relative_to(self.root)), reason=str(exc)))
        for path in sorted((self.root/'wgt_calibration/manual_bunker/records').glob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8-sig'))
                if payload.get('kind') == 'manual_bunker_shot' and payload.get('measurement_verified') is not True:
                    excluded = dict(id='manual:'+payload['id'], session_id='manual-day:'+payload['recorded_at_utc'][:10],
                        source='manual partial, not promoted', state={}, group=None, eligible=False,
                        exclusions=['manual measurement incomplete or unattributed'], verified={}, actual={},
                        original=deepcopy(payload), use_for_live_control=False)
                    added += self.remember(excluded, retrain=False)
                    continue
                record = manual_record(self.root, payload)
                record['evidence']['record_path'] = str(path.relative_to(self.root))
                added += self.remember(record, retrain=False)
            except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError) as exc:
                issues.append(dict(path=str(path.relative_to(self.root)), reason=str(exc)))
        # Legacy calibration lacks sufficient context/provenance for this new
        # protocol. Preserve its facts in excluded memory rather than relabel it.
        legacy = self.root/'wgt_calibration/power_observations.json'
        if legacy.is_file():
            try:
                data = json.loads(legacy.read_text(encoding='utf-8-sig'))
                if data.get('schema') != 1 or not isinstance(data.get('observations'), list):
                    raise ValueError('legacy schema unrecognized')
                legacy_rows = data['observations']
            except (ValueError, AttributeError, OSError) as exc:
                issues.append(dict(path=str(legacy.relative_to(self.root)), reason=str(exc)))
                legacy_rows = []
            for item in legacy_rows:
                if not isinstance(item, dict):
                    continue
                identity = item.get('shot_id')
                if not identity:
                    continue
                record = dict(id='legacy:'+identity, session_id='legacy:'+identity.split(':')[0],
                              source='legacy calibration, not promoted', state={}, group=None, eligible=False,
                              exclusions=['legacy context/independent verification incomplete'], verified={}, actual={},
                              original=deepcopy(item), use_for_live_control=False)
                added += self.remember(record, retrain=False)
        for group, in self.db.execute('SELECT DISTINCT context_group FROM shots WHERE eligible=1').fetchall():
            self.rebuild(group)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('last_sync',?)", (_json(dict(added=added, issues=issues)),))
        return dict(added=added, issues=issues)

    def report(self):
        rows = self.rows()
        exclusions = Counter(reason for r in rows for reason in r.get('exclusions', []))
        champions = [json.loads(row[0]) for row in self.db.execute(
            'SELECT m.data FROM models m JOIN champions c ON c.version=m.version ORDER BY m.context_group')]
        trends = {}
        for group in sorted({r.get('group') for r in rows if r.get('group')}):
            group_rows = [r for r in rows if r.get('group') == group and r.get('eligible')]
            errors = [r.get('prediction_errors', {}).get('learned') for r in group_rows]
            errors = [x for x in errors if number(x)]
            trends[group] = dict(eligible_shots=len(group_rows), frozen_prediction_error=error_metrics(errors),
                                recent_20=error_metrics(errors[-20:]), earlier=error_metrics(errors[:-20]),
                                numeric_short=sum((r['actual'].get('numeric_travel_minus_target') or 0) < 0 for r in group_rows),
                                numeric_long=sum((r['actual'].get('numeric_travel_minus_target') or 0) > 0 for r in group_rows),
                                verified_area_results=sum(r.get('verified', {}).get('intended_area') is True for r in group_rows),
                                verified_recovery_failures=sum(r['actual'].get('recovery_failed') is True for r in group_rows))
        rejected = [json.loads(r[0]) for r in self.db.execute(
            'SELECT data FROM models WHERE accepted_shadow=0 ORDER BY created_at DESC LIMIT 30')]
        latest = [json.loads(r[0]) for r in self.db.execute('SELECT data FROM predictions ORDER BY created_at DESC LIMIT 20')]
        wind_telemetry = [r['wind_accuracy'] for r in rows if isinstance(r.get('wind_accuracy'), dict)]
        wind_summary = dict(observed_trigger_offsets=len(wind_telemetry),
                            validated_lateral_outcomes=0,
                            learned_wind_click_policy=False,
                            live_accuracy_offset_allowed=False,
                            status='telemetry only; meter offset has no measured lateral efficacy')
        sync = self.db.execute("SELECT value FROM meta WHERE key='last_sync'").fetchone()
        return dict(schema=1, mode='shadow_only', live_control_allowed=False, bunker_autoplay='blocked',
                    shots=len(rows), eligible=sum(r.get('eligible') is True for r in rows),
                    partitions=dict(Counter(r['partition'] for r in rows if r.get('eligible'))),
                    exclusions=dict(exclusions), accepted_shadow_models=champions, recent_rejections=rejected,
                    performance=trends, wind_accuracy=wind_summary,
                    protocol=PROTOCOL, latest_shadow_predictions=latest,
                    last_sync=json.loads(sync[0]) if sync else None,
                    limitations=['Aim/final hole distance/area success remain null unless explicitly verified.',
                      'No automatic promotion or source-code rewriting exists.',
                      'Repeated holdout checks are not a substitute for a fresh independent activation trial.',
                      'Current screen readers do not verify impact accuracy or full lateral hazard geometry.'])

    def write_report(self):
        report = self.report()
        atomic_json(self.folder/'learning_report.json', report)
        return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync', action='store_true', help='import existing local evidence and evaluate shadow models')
    parser.add_argument('--report', action='store_true', help='write the offline learning report')
    args = parser.parse_args(argv)
    engine = ShadowLearning(Path(__file__).resolve().parent)
    try:
        if args.sync:
            print(json.dumps(engine.sync_existing(), indent=2))
        report = engine.write_report()
        path = engine.folder/'learning_report.json'
        print(f'SHADOW ONLY: {report["shots"]} memory rows; {report["eligible"]} eligible; '
              f'{len(report["accepted_shadow_models"])} accepted shadow models. Bunker swings blocked.')
        print('Report:', path)
        return 0
    finally:
        engine.close()


if __name__ == '__main__':
    raise SystemExit(main())
