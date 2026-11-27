"""Statistical-learning and controller-boundary checks. All data is offline.

Synthetic training/evaluation records exist only inside TemporaryDirectory;
they are never mixed into the user's learning database or real calibration.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import execution_engine as live
from automation_core import Meter
from automation_learning import ShadowLearning, fit_model, forward, inverse, split_for
from learning_data import (automatic_record, baseline_descriptor, baseline_distance,
                               group_key, manual_record)

ROOT = Path(__file__).resolve().parent


def state(mode='FULL', lie='FAIRWAY', **changes):
    putt = mode == 'PUTT'
    percent = [40, 50] if lie in ('BUNKER', 'ROUGH', 'HEAVY ROUGH') else None
    context = dict(source='solo_user_confirmed_live_hud', format='SOLO_STROKE', players=['tigerwould52'],
        active_player='tigerwould52', ball_owner='tigerwould52', permission=dict(allowed=True),
        lie=lie, lie_percent=percent, confidence=dict(lie=.96, lie_percent=.96),
        solo_evidence=dict(player_header_visible=True, swing_and_meter_visible=True,
                           multiplayer_conflicts_absent=True, user_confirmed=True))
    result = dict(player=True, hole=10, shot=3, mode=mode, putt=putt, lie=lie, lie_percent=percent,
        lie_loss_percent=None, distance=50., unit='ft' if putt else 'yd',
        distance_source='floating_target', club_name='PUTTER' if putt else 'PW',
        club_range=100., club_unit='ft' if putt else 'yd', elevation_ft=0., green_speed=10.,
        wind=dict(low_mph=0, high_mph=0, mean_mph=0, screen_dx=0., screen_dy=0.),
        previous_stats=dict(carry_yd=126.22, roll_yd=7.87, power_pct=87), game_context=context)
    result.update(changes)
    return result


def result(before, carry=40., roll=2., power=.5, **changes):
    after = deepcopy(before)
    after.update(shot=before['shot']+1, motion=dict(scene_motion_observed=True),
                 movement_verification='video motion plus stable stroke/HUD advance',
                 fresh_stats=dict(carry_yd=carry, roll_yd=roll, power_pct=power*100))
    after.update(changes)
    return after


def sessions(partition, count):
    values = []
    i = 0
    while len(values) < count:
        name = 'independent-session-'+str(i)
        if split_for(name) == partition:
            values.append(name)
        i += 1
    return values


def synthetic_rows(mode='FULL', lie='FAIRWAY', club_range=100., club='PW', noise=0., gain=.78):
    records = []
    for partition, count in (('train', 4), ('validation', 2), ('audit', 2)):
        for session_index, session in enumerate(sessions(partition, count)):
            for j in range(8):
                power = .2+j*.1
                elevation = (-3., 0., 3.)[(j+session_index) % 3]
                wind_y = (-.5, 0., .5)[(j//2+session_index) % 3]
                mph = abs(wind_y)*20
                s = state(mode, lie, club_range=club_range, club_name=club,
                          distance=power*club_range, elevation_ft=elevation,
                          wind=dict(low_mph=mph, high_mph=mph, mean_mph=mph,
                                    screen_dx=0., screen_dy=0. if mph == 0 else (1. if wind_y > 0 else -1.)))
                e = elevation/club_range/(1 if mode == 'PUTT' else 3)
                expected = club_range*power*(gain+.12*power-.3*e-(0 if mode == 'PUTT' else .08*wind_y))
                expected += noise*((j % 2)*2-1)
                total_yd = expected/3 if mode == 'PUTT' else expected
                carry = total_yd*.8 if mode in ('CHIP', 'PITCH', 'PUTT') else expected
                roll = total_yd*.2 if mode in ('CHIP', 'PITCH', 'PUTT') else 2.
                a = result(s, carry=round(carry, 4), roll=round(roll, 4), power=power)
                record = automatic_record(f'{mode}:{lie}:{club}:{club_range}:{session}:{j}', session, s, a, power,
                                          dict(power=power, model='selected-range baseline'))
                record['source'] = 'synthetic test only'
                record['partition'] = partition
                records.append(record)
    return records


class DataEvidenceTests(unittest.TestCase):
    def test_automatic_memory_keeps_action_result_and_unknowns_separate(self):
        s = state()
        r = automatic_record('s:1', 's', s, result(s, power=.51), .51,
                             dict(power=.5, model='selected-range baseline'))
        self.assertTrue(r['eligible'], r['exclusions'])
        self.assertEqual(r['selected_power'], .5)
        self.assertEqual(r['actual_game_power'], .51)
        self.assertEqual(r['actual']['carry_yd'], 40.)
        self.assertEqual(r['actual']['roll_yd'], 2.)
        self.assertEqual(r['actual']['travel'], 40.)
        self.assertIsNone(r['aim'])
        self.assertIsNone(r['actual']['final_distance_to_hole'])
        self.assertIsNone(r['actual']['reached_intended_area'])
        self.assertFalse(r['use_for_live_control'])

    def test_missing_wrong_or_stale_results_never_train(self):
        s = state()
        for mutate in ('stale', 'owner', 'motion', 'skipped', 'lie', 'wind', 'power', 'incomplete'):
            b, a = deepcopy(s), result(s)
            if mutate == 'stale':
                a['fresh_stats'] = deepcopy(b['previous_stats'])
            elif mutate == 'owner':
                a['game_context']['players'] = ['opponent1']
            elif mutate == 'motion':
                a['motion'] = {}
            elif mutate == 'skipped':
                a['shot'] += 1
            elif mutate == 'lie':
                b['game_context']['confidence']['lie'] = .4
            elif mutate == 'wind':
                b['wind'] = None
            elif mutate == 'power':
                a['fresh_stats']['power_pct'] = 90
            else:
                del a['fresh_stats']['roll_yd']
            r = automatic_record('s:1', 's', b, a, .5, dict(power=.5))
            with self.subTest(mutate=mutate):
                self.assertFalse(r['eligible'])
                self.assertTrue(r['exclusions'])

    def test_putt_units_carry_and_roll_total_not_carry_only(self):
        for mode in ('FULL', 'CHIP', 'PITCH', 'PUTT'):
            s = state(mode, 'GREEN' if mode == 'PUTT' else 'FAIRWAY')
            r = automatic_record('a', 's', s, result(s, carry=4., roll=2.), .5, dict(power=.5))
            self.assertTrue(r['eligible'], r['exclusions'])
            self.assertEqual(r['actual']['travel'], 18. if mode == 'PUTT' else 6. if mode != 'FULL' else 4.)
            self.assertEqual(r['actual']['unit'], 'ft' if mode == 'PUTT' else 'yd')
        s = state('PUTT', 'GREEN')
        a = result(s); a.pop('fresh_stats')
        self.assertIsNone(automatic_record('x', 's', s, a, .5)['actual']['travel'])

    def test_final_hole_distance_requires_verified_hole_link_not_completion(self):
        s = state()
        a = result(s, complete=True, distance=6., unit='ft')
        self.assertIsNone(automatic_record('a', 's', s, a, .5)['actual']['final_distance_to_hole'])
        a['target_hole_link'] = dict(source='visible_lock_beside_target')
        distance = automatic_record('a', 's', s, a, .5)['actual']['final_distance_to_hole']
        self.assertEqual(distance['value'], 6.)
        self.assertEqual(distance['unit'], 'ft')

    def test_recovery_failure_and_area_success_need_declared_intent_and_observed_lie(self):
        s = state(lie='BUNKER', club_name='LW', club_range=52)
        a = result(s, carry=0., roll=0., power=.26)
        r = automatic_record('b', 's', s, a, .26, dict(power=.26, intended_area='OUTSIDE_BUNKER', intent_source='planner'))
        self.assertTrue(r['eligible'], r['exclusions'])
        self.assertTrue(r['actual']['recovery_failed'])
        self.assertEqual(r['actual']['travel'], 0.)
        self.assertIsNone(automatic_record('b', 's', s, a, .26)['actual']['recovery_failed'])
        a['lie'] = a['game_context']['lie'] = 'GREEN'
        r = automatic_record('b', 's', s, a, .26, dict(power=.26, intended_area='OUTSIDE_BUNKER', intent_source='planner'))
        self.assertTrue(r['actual']['reached_intended_area'])

    def test_explicit_aim_evidence_is_stored_and_missing_aim_is_not_reconstructed(self):
        s = state()
        s['aim_measurement'] = dict(verified=True, source='current_aim_marker_ocr', confidence=.96, x=.5, y=.3)
        r = automatic_record('a', 's', s, result(s), .5)
        self.assertEqual(r['aim'], s['aim_measurement'])
        s['aim_measurement']['confidence'] = .3
        self.assertIsNone(automatic_record('a', 's', s, result(s), .5)['aim'])

    def test_lie_display_is_raw_feature_and_baseline_does_not_treat_it_as_loss(self):
        s = state(lie='BUNKER', club_range=52, club_name='LW', distance=16)
        self.assertAlmostEqual(baseline_distance(s, .26), 52*.26)
        self.assertNotEqual(baseline_distance(s, .26), 16.)
        s['lie_loss_percent'] = [40, 50]
        self.assertAlmostEqual(baseline_distance(s, .26), 52*.26*.55)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = ShadowLearning(self.temp.name, 'test-session')
        self.addCleanup(self.engine.close)

    def populate(self, **kwargs):
        rows = synthetic_rows(**kwargs)
        self.assertTrue(all(r['eligible'] for r in rows), [r['exclusions'] for r in rows if not r['eligible']])
        for row in rows:
            self.engine.remember(row, retrain=False)
        report = self.engine.rebuild(rows[0]['group'])
        return rows, report

    def test_real_fitting_beats_baseline_on_separate_validation_and_audit_sessions(self):
        rows, report = self.populate()
        self.assertEqual(report['status'], 'accepted_shadow', report)
        for name in ('validation', 'audit'):
            metrics = report['metrics'][name]
            self.assertLess(metrics['learned']['mae'], metrics['baseline']['mae']*.1)
            self.assertLess(metrics['learned']['worst'], metrics['baseline']['worst'])
        self.assertFalse(report['live_control_allowed'])
        self.assertEqual(set(report['training_ids']) & set(report['validation_ids']), set())
        self.assertEqual(set(report['audit_ids']) & set(report['training_ids']), set())
        by_id = {r['id']: r for r in rows}
        sets = [{by_id[i]['session_id'] for i in report[key]} for key in ('training_ids', 'validation_ids', 'audit_ids')]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_nonlinear_inverse_changes_power_from_actual_experience(self):
        rows, report = self.populate()
        s = state(distance=50.)
        power = inverse(report['params'], s, 50.)
        self.assertGreater(power, .5)
        self.assertAlmostEqual(forward(report['params'], s, power), 50., places=6)
        self.assertNotEqual(power, .26)

    def test_models_are_separate_for_terrain_and_each_shot_mode(self):
        groups = set()
        for mode, lie, club in (('FULL', 'TEE', 'DR'), ('FULL', 'FAIRWAY', 'PW'),
                               ('FULL', 'ROUGH', 'PW'), ('FULL', 'BUNKER', 'LW'),
                               ('CHIP', 'FAIRWAY', 'PW'), ('PITCH', 'FAIRWAY', 'PW'), ('PUTT', 'GREEN', 'PUTTER')):
            rows, report = self.populate(mode=mode, lie=lie, club=club)
            self.assertEqual(report['status'], 'accepted_shadow', report)
            groups.add(rows[0]['group'])
        self.assertEqual(len(groups), 7)
        self.assertEqual(len(self.engine.report()['accepted_shadow_models']), 7)

    def test_insufficient_evidence_and_nonmonotone_response_are_rejected(self):
        rows = synthetic_rows()
        for row in rows[:8]:
            self.engine.remember(row, retrain=False)
        self.assertEqual(self.engine.rebuild(rows[0]['group'])['status'], 'rejected')
        train = [deepcopy(r) for r in rows if r['partition'] == 'train']
        for r in train:
            r['actual']['travel'] = 100*(1-r['actual_game_power'])
        with self.assertRaisesRegex(ValueError, 'non-monotone'):
            fit_model(train)
        for r in train:
            r['actual_game_power'] = .5
        with self.assertRaisesRegex(ValueError, 'variation'):
            fit_model(train)

    def test_a_good_training_fit_with_bad_holdout_is_never_accepted(self):
        rows = synthetic_rows()
        for r in rows:
            if r['partition'] != 'train':
                r['actual']['travel'] = r['baseline_at_actual_power']
            self.engine.remember(r, retrain=False)
        report = self.engine.rebuild(rows[0]['group'])
        self.assertEqual(report['status'], 'rejected')
        self.assertIsNone(self.engine.champion(rows[0]['group']))

    def test_no_extrapolation_to_unmeasured_wind_elevation_power_or_range(self):
        _, report = self.populate()
        for s, power in ((state(elevation_ft=10), .5), (state(club_range=52), .5),
                         (state(distance=99), .5), (state(), .99),
                         (state(wind=dict(low_mph=20, high_mph=20, screen_dx=1, screen_dy=0)), .5)):
            with self.assertRaises(ValueError):
                forward(report['params'], s, power)

    def test_duplicate_shots_and_session_partitions_survive_restart(self):
        rows, report = self.populate()
        self.assertFalse(self.engine.remember(rows[0]))
        other = ShadowLearning(self.temp.name)
        try:
            self.assertEqual(len(other.rows()), len(rows))
            self.assertEqual(other.champion(rows[0]['group'])['version'], report['version'])
            self.assertEqual({r['partition'] for r in other.rows() if r['session_id'] == rows[0]['session_id']}, {'train'})
        finally:
            other.close()

    def test_unverified_rows_remain_in_memory_with_reasons_but_never_fit(self):
        s = state(wind=None)
        row = automatic_record('bad:1', 'bad', s, result(s), .5)
        self.engine.remember(row)
        self.assertEqual(len(self.engine.rows()), 1)
        self.assertEqual(self.engine.rows(eligible=True), [])
        self.assertTrue(self.engine.rows()[0]['exclusions'])

    def test_frozen_pre_shot_model_versions_and_errors_are_not_rewritten_after_learning(self):
        _, report = self.populate()
        s = state(distance=60)
        plan = dict(power=.6, model='selected-range baseline')
        original_state, original_plan = deepcopy(s), deepcopy(plan)
        snap = self.engine.begin_shot('live:1', s, plan)
        self.assertEqual(snap['learned_model_version'], report['version'])
        self.assertNotEqual(snap['recommendation']['selected']['power'], plan['power'])
        actual = result(s, carry=51.12, roll=2, power=.6)
        record = self.engine.finish_shot('live:1', s, actual, .6, plan)
        self.assertEqual(record['prediction']['learned_model_version'], report['version'])
        self.assertAlmostEqual(record['prediction_errors']['learned'], snap['learned_distance']-51.12)
        self.assertEqual(s, original_state)
        self.assertEqual(plan, original_plan)
        self.assertTrue((self.engine.folder/'learning_report.json').is_file())

    def test_first_tee_policy_stays_at_full_power_even_in_shadow(self):
        s = state(lie='TEE', shot=1)
        r = self.engine.recommend(s)
        self.assertEqual(r['selected']['power'], 1.)
        self.assertTrue(r['tee_policy_preserved'])
        self.assertFalse(r['live_control_allowed'])

    def test_verified_hazards_use_learned_error_band_and_current_aim_only(self):
        self.populate()
        s = state(distance=60)
        hazard = dict(kind='BUNKER', unit='yd', current_aim=True, start=55., end=70.)
        s['verified_hazards'] = [hazard]
        s['game_context'].update(hazards=[hazard])
        s['game_context']['confidence']['hazards'] = .96
        recommendation = self.engine.recommend(s)
        self.assertLessEqual(recommendation['selected']['target'], 45.)
        self.assertFalse(recommendation['selected']['hazard_overlap'])
        s['game_context']['confidence']['hazards'] = .1
        self.assertEqual(self.engine.recommend(s)['selected']['target'], 60.)

    def test_club_ranking_uses_only_observed_options_and_past_error(self):
        self.populate(noise=1.)
        self.populate(club_range=80., club='SW', noise=0.)
        s = state(distance=50.)
        self.assertEqual(self.engine.recommend(s)['selected']['club'], 'PW')
        self.engine.note_club(state(club_name='SW', club_range=80., distance=50.))
        recommendation = self.engine.recommend(s)
        self.assertEqual(recommendation['selected']['club'], 'SW')
        self.assertFalse(recommendation['live_control_allowed'])

    def test_champion_is_retired_if_new_independent_outcomes_discredit_it(self):
        rows, accepted = self.populate()
        self.assertIsNotNone(self.engine.champion(rows[0]['group']))
        used = {r['session_id'] for r in rows}
        new_sessions = [s for s in sessions('validation', 20) if s not in used][:12]
        for session in new_sessions:
            for original in [r for r in rows if r['session_id'] == rows[32]['session_id']]:
                r = deepcopy(original)
                r.update(id=session+':'+r['id'], session_id=session)
                r['actual']['travel'] = r['baseline_at_actual_power']
                self.engine.remember(r, retrain=False)
        rejected = self.engine.rebuild(rows[0]['group'])
        self.assertEqual(rejected['status'], 'rejected')
        self.assertIsNone(self.engine.champion(rows[0]['group']))


class ImportAndSafetyTests(unittest.TestCase):
    def test_session_sync_is_read_only_idempotent_and_matches_live_shot_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root/'wgt_debug/runs/fixture-session'
            run.mkdir(parents=True)
            s = state()
            payload = dict(before=s, after=result(s), observed_power=.5)
            path = run/'hole_01_shot_01_result.json'
            path.write_text(json.dumps(payload))
            (run/'hole_01_shot_01_plan.json').write_text(json.dumps(dict(plan=dict(power=.5, model='selected-range baseline'))))
            source_bytes = path.read_bytes()
            engine = ShadowLearning(root)
            try:
                self.assertEqual(engine.sync_existing()['added'], 1)
                self.assertEqual(engine.rows()[0]['id'], 'fixture-session:hole_01_1')
                self.assertEqual(engine.sync_existing()['added'], 0)
                self.assertEqual(path.read_bytes(), source_bytes)
                self.assertEqual(len(engine.rows(eligible=True)), 1)
            finally:
                engine.close()

    def test_manual_import_rechecks_images_and_does_not_promote_the_contextual_reference(self):
        from test_manual_bunker_recording import snapshot
        from terrain_recording import ManualBunkerStore
        with tempfile.TemporaryDirectory() as folder:
            before = [snapshot(folder, 'b1', 100), snapshot(folder, 'b2', 101)]
            stats = dict(carry_yd=12.34, roll_yd=2.06, power_percent=26.)
            after = [snapshot(folder, 'a1', 120, shot=5, stats=stats), snapshot(folder, 'a2', 121, shot=5, stats=stats)]
            store = ManualBunkerStore(folder)
            store.stage(before, solo_confirmed=True, now=102)
            record = store.finish(after, one_manual_shot_confirmed=True, setup_unchanged_confirmed=True, now=122)
            unified = manual_record(folder, record)
            self.assertEqual(unified['actual']['carry_yd'], 12.34)
            self.assertIsNone(unified['selected_power'])
            self.assertTrue(unified['session_id'].startswith('manual-day:'))
            self.assertFalse(unified['use_for_live_control'])
            forged = deepcopy(record)
            forged['after'][0]['evidence']['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                manual_record(folder, forged)
            reference = json.loads((ROOT/'wgt_calibration/bunker_reference_manual.json').read_text(encoding='utf-8-sig'))
            with self.assertRaises(ValueError):
                manual_record(folder, reference)

    def test_legacy_verified_flags_alone_do_not_authorize_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'wgt_calibration/power_observations.json'
            path.parent.mkdir()
            original = json.dumps(dict(schema=1, observations=[dict(shot_id='old:1', verified=True, power=.26, travel=16.)]))
            path.write_text(original)
            engine = ShadowLearning(folder)
            try:
                engine.sync_existing()
                self.assertEqual(engine.report()['eligible'], 0)
                self.assertEqual(path.read_text(), original)
            finally:
                engine.close()

    def test_existing_swing_accuracy_f9_and_nine_hole_controls_are_preserved(self):
        backup = ROOT/'repair_backups/learning_shadow_20261113'
        # These two reviewed transition additions retain capture time and stop
        # a penalty-like counter jump. Preserve the rest of the function exactly.
        transition_updates = {
            ast.dump(node, include_attributes=False) for node in ast.parse("""
state['captured_at']=captured_at
if outcome=='next' and state['shot']!=before['shot']+1:
    diagnostic_state('unexpected_stroke_advance',dict(before=before,after=state,motion=report),frame)
    raise RuntimeError('Stroke counter advanced by more than one; possible penalty or external stroke. '
                       'Stop for a new plan; no automatic repeat shot')
""").body}
        class TransitionUpdates(ast.NodeTransformer):
            def visit(self,node):
                if isinstance(node,(ast.Assign,ast.If)) and ast.dump(node,include_attributes=False) in transition_updates:
                    return None
                return super().visit(node)
        def definitions(path):
            result={}
            for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                    if node.name=='wait_until_next':
                        node=TransitionUpdates().visit(node)
                    result[node.name]=ast.dump(node,include_attributes=False)
            return result
        a, b = definitions(ROOT/'execution_engine.py'), definitions(backup/'wgt_one_hole_v48.py')
        for name in ('setup', 'send', 'move_to', 'mouse_up', 'Safety', 'drag_swing', 'accuracy_click',
                     'ShotObserver', 'wait_until_next', 'activate_wgt', 'clicklock_disabled'):
            self.assertEqual(a[name], b[name], name)
        def normalized_source(path):
            source=path.read_text(encoding='utf-8-sig')
            return source.replace('terrain_handler', 'wgt_bunker')
        self.assertEqual(normalized_source(ROOT/'automation_core.py'),
                         normalized_source(backup/'wgt_core.py'), 'automation_core.py')
        def normalized_modules(tree):
            aliases={'automation_core':'wgt_core', 'context_manager':'wgt_context',
                     'vision_engine':'wgt_observation', 'terrain_handler':'wgt_bunker'}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in aliases:
                    node.module=aliases[node.module]
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in aliases:
                            alias.name=aliases[alias.name]
            return tree
        def unchanged_round_tree(path):
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='scorecard')]
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(unchanged_round_tree(ROOT/'round_manager.py'),unchanged_round_tree(backup/'round_manager.py'))
        def unchanged_observation_tree(path):
            tree=normalized_modules(ast.parse(path.read_text(encoding='utf-8-sig')))
            # Approved badge/range/conditions reader repairs have real-frame
            # and missing-glyph regressions. All other observation code,
            # including motion, persistence and power calibration, stays exact.
            readers={'quick_putt_badge','_read_putt_badge','_visible_decimal_text','read_putter_range','read_conditions'}
            tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in readers)]
            for node in tree.body:
                if isinstance(node,ast.ClassDef) and node.name=='Session':
                    node.body=[n for n in node.body if not (
                        isinstance(n,ast.FunctionDef) and n.name in ('shot_stage','shot_observation'))]
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(unchanged_observation_tree(ROOT/'vision_engine.py'),
                         unchanged_observation_tree(backup/'wgt_observation.py'))
        # Bunker/clock boundaries and reviewed solo freshness handling changed.
        # Preserve every other definition and module statement; behavior tests
        # in test_bunker_power and test_shot_clock cover the revised boundaries.
        def preserved_tree(path):
            tree = normalized_modules(ast.parse(path.read_text(encoding='utf-8-sig')))
            tree.body = [n for n in tree.body if not (
                isinstance(n, ast.ImportFrom) and n.module == 'copy' and n.level == 0
                and len(n.names) == 1 and n.names[0].name == 'deepcopy'
                and n.names[0].asname is None)]
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == 'game_state':
                    node.names = [n for n in node.names if not (
                        n.name == 'ALTERNATE_SHOT_TURN_SECONDS' and n.asname is None)]
            if path.name in ('wgt_bunker.py', 'terrain_handler.py'):
                tree.body = [n for n in tree.body if not (
                    isinstance(n, ast.FunctionDef) and n.name == 'bunker_power_estimate')]
            elif path.name == 'game_state.py':
                tree.body = [n for n in tree.body if not (
                    isinstance(n, ast.FunctionDef) and n.name == 'turn_permission') and not (
                    isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'ALTERNATE_SHOT_TURN_SECONDS'
                    and isinstance(n.value, ast.Constant) and n.value.value == 60)]
            else:
                tree.body=[n for n in tree.body if not (
                    isinstance(n,ast.FunctionDef) and n.name=='_solo_static_hud_unchanged')]
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == 'GameCoordinator':
                        for method in node.body:
                            if isinstance(method,ast.FunctionDef) and method.name=='__init__':
                                while method.args.args[-1].arg in ('read_state_from_source','elevation_reader'):
                                    name=method.args.args.pop().arg
                                    method.args.defaults.pop()
                                    extra=ast.dump(ast.parse(f'self.{name}={name}').body[0],include_attributes=False)
                                    method.body=[n for n in method.body if ast.dump(n,include_attributes=False)!=extra]
                        node.body = [n for n in node.body if not (
                            isinstance(n, ast.FunctionDef) and n.name in
                            ('arm_stroke','inspect_ready','require_turn','wait_for_turn','refresh_turn_frame',
                             'revalidate_putt'))]
                    if isinstance(node, ast.ClassDef) and node.name == 'ContextObserver':
                        node.body = [n for n in node.body if not (
                            isinstance(n, ast.FunctionDef) and n.name in ('__init__','observe'))]
            return ast.dump(tree, include_attributes=False)
        for current_name, archived_name in (
                ('context_manager.py', 'wgt_context.py'),
                ('terrain_handler.py', 'wgt_bunker.py'),
                ('game_state.py', 'game_state.py')):
            self.assertEqual(preserved_tree(ROOT/current_name), preserved_tree(backup/archived_name),
                             current_name)

    def test_learning_code_has_no_input_or_source_execution_apis(self):
        for name in ('automation_learning.py', 'learning_data.py'):
            tree = ast.parse((ROOT/name).read_text())
            calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                     for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
            self.assertFalse(calls & {'eval', 'exec', 'compile', 'SendInput', 'move_to', 'mouse_up', 'capture', 'Popen', 'startfile'})

    def test_shadow_exceptions_do_not_swallow_f9(self):
        with patch.object(live, 'LEARNING') as learner:
            learner.begin_shot.side_effect = KeyboardInterrupt('F9')
            with self.assertRaises(KeyboardInterrupt):
                live.learning_event('begin_shot', 'x', {}, {})

    def test_shadow_recommendation_never_becomes_the_live_swing_power(self):
        frame = np.zeros((728, 1288, 3), np.uint8)
        s = state(distance=50)
        meter = Meter(73, 598, 932, 767, 1056, 535)
        with patch.object(live, 'GAME') as game, patch.object(live, 'DIAGNOSTICS') as diagnostic, \
                patch.object(live, 'LEARNING') as learner, patch.object(live, 'POWER_MODEL') as baseline, \
                patch.object(live, 'park_on_meter'), patch.object(live, 'stable_state', return_value=(s, frame)), \
                patch.object(live, 'select_club', return_value=(s, frame)), patch.object(live, 'ready_meter', return_value=meter), \
                patch.object(live, 'shot_stats', return_value=None), patch.object(live, 'ShotObserver'), \
                patch.object(live, 'drag_swing', return_value=.5) as drag, patch.object(live, 'accuracy_click') as accuracy, \
                patch.object(live, 'wait_until_next', return_value=('complete', None)), \
                patch.object(Path, 'write_text'), patch.object(cv2_placeholder(), 'imwrite'), redirect_stdout(io.StringIO()):
            game.completed = False
            baseline.plan.return_value = dict(power=.5, model='existing baseline')
            learner.begin_shot.return_value = dict(recommendation=dict(selected=dict(power=.01)))
            self.assertEqual(live.run_hole(Mock(), None, {}, Mock()), 'complete')
        self.assertEqual(drag.call_args.args[-1], .5)
        accuracy.assert_called_once()
        learner.finish_shot.assert_called_once()


def cv2_placeholder():
    # Same imported module as the controller; no screen capture or input.
    import cv2
    return cv2


if __name__ == '__main__':
    unittest.main(verbosity=2)
