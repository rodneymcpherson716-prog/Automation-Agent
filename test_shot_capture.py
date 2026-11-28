"""New-shot capture tests. Synthetic data stays inside TemporaryDirectory."""
from copy import deepcopy
from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import execution_engine as live
from automation_core import Meter
from automation_learning import ShadowLearning
from learning_data import automatic_record
from vision_engine import PowerCalibration, Session
from shot_capture import build_shot_capture, ding_reading
from shot_store import DATABASE, import_captures
from test_learning import state, result


def trace(shot_id='offline:1'):
    return dict(shot_id=shot_id, clicked=True, target=120., samples=[
        dict(t=0., edge=100.), dict(t=.01, edge=110.), dict(t=.02, edge=121.)])


def after_state(s):
    a = result(s, carry=40., roll=2., power=.5)
    a.update(distance=6., unit='ft', mode='PUTT', putt=True, club_name='PUTTER',
             club_range=30., club_unit='ft', lie='GREEN',
             target_hole_link=dict(source='visible_lock_beside_target'))
    a['game_context'].update(lie='GREEN', lie_percent=None)
    return a


class ShotCaptureTests(unittest.TestCase):
    def capture(self, s=None, a=None, t=None):
        s = state() if s is None else s
        return build_shot_capture('offline:1', s, after_state(s) if a is None else a,
                                  .5, dict(power=.5), trace() if t is None else t, 'next')

    def test_complete_shot_contains_requested_facts_and_separate_power_measurements(self):
        s = state()
        p = self.capture(s)
        self.assertTrue(p['complete'], p['missing'])
        self.assertEqual(p['before']['club_name'], 'PW')
        self.assertEqual(p['before']['distance'], 50)
        self.assertEqual(p['before']['lie'], 'FAIRWAY')
        self.assertEqual(p['before']['wind'], s['wind'])
        self.assertEqual(p['before']['elevation_ft'], 0)
        self.assertEqual(p['swing']['requested_power_fraction'], .5)
        self.assertEqual(p['swing']['observed_meter_power_fraction'], .5)
        self.assertEqual(p['swing']['attributed_game_power_fraction'], .5)
        self.assertEqual(p['swing']['ding']['edge_minus_target_px'], 1)
        self.assertFalse(p['swing']['ding']['exact_mouse_down_verified'])
        self.assertEqual(p['result']['owner'], 'tigerwould52')
        self.assertEqual(p['result']['carry_yd'], 40)
        self.assertEqual(p['result']['roll_yd'], 2)
        self.assertEqual(p['result']['final_position']['distance_to_hole']['value'], 6)
        self.assertEqual(p['result']['final_position']['distance_to_hole']['unit'], 'ft')
        self.assertEqual(p['result']['final_position']['lie'], 'GREEN')
        self.assertIsNone(p['result']['final_position']['absolute_coordinates'])
        self.assertEqual(p['distance_training_eligible'],
                         automatic_record('offline:1', 's', s, after_state(s), .5)['eligible'])
        self.assertFalse(p['use_for_live_control'])
        json.dumps(p, allow_nan=False)

    def test_missing_before_conditions_and_baseline_are_disclosed_without_guesses(self):
        for field, check in (('club_name', 'before_club_and_units'), ('distance', 'before_distance'),
                             ('lie', 'before_lie'), ('wind', 'before_wind'),
                             ('elevation_ft', 'before_elevation'), ('previous_stats', 'before_stats_baseline')):
            s = state();s[field] = None
            with self.subTest(field=field):
                p = self.capture(s)
                self.assertFalse(p['complete'])
                self.assertIn(check, p['missing'])
                self.assertIsNone(p['before'][field])
                self.assertFalse(p['distance_training_eligible'])

    def test_stale_incomplete_or_foreign_stats_cannot_be_called_verified_results(self):
        for defect in ('stale', 'missing_roll', 'foreign', 'wrong_stroke', 'wrong_pre_player'):
            s = state();a = after_state(s)
            if defect == 'stale':
                a['fresh_stats'] = deepcopy(s['previous_stats'])
            elif defect == 'missing_roll':
                del a['fresh_stats']['roll_yd']
            elif defect == 'foreign':
                a['game_context']['stats_owner'] = 'opponent1'
                a['game_context']['source'] = 'wgt_team_hud_v1'
                a['game_context']['confidence']['stats_owner'] = .99
            elif defect == 'wrong_stroke':
                a['shot'] += 1
            else:
                s['game_context']['active_player'] = 'opponent1'
            with self.subTest(defect=defect):
                p = self.capture(s, a)
                self.assertFalse(p['complete'])
                self.assertFalse(p['verification']['result_belongs_to_tigerwould52'])
                self.assertIsNone(p['result']['owner'])
                self.assertIsNone(p['result']['carry_yd'])

    def test_conflicting_explicit_stats_owner_vetoes_a_solo_header(self):
        s = state();a = after_state(s)
        a['game_context']['stats_owner'] = 'opponent1'
        p = self.capture(s, a)
        self.assertFalse(p['verification']['result_belongs_to_tigerwould52'])
        self.assertFalse(p['complete'])
        self.assertFalse(p['distance_training_eligible'])

    def test_final_hole_distance_and_hole_out_are_not_inferred_from_travel(self):
        s = state();a = after_state(s)
        a.pop('target_hole_link')
        p = self.capture(s, a)
        self.assertFalse(p['complete'])
        self.assertIn('final_position', p['missing'])
        self.assertIsNone(p['result']['final_position']['distance_to_hole'])
        self.assertTrue(p['distance_training_eligible'])  # Existing distance-fit rules unchanged.
        p = self.capture(s, dict(complete=True, hole=s['hole'], motion=dict(scene_motion_observed=True)))
        self.assertFalse(p['complete'])
        self.assertTrue(p['result']['hole_completion_observed'])
        self.assertIsNone(p['result']['final_position']['distance_to_hole'])
        self.assertIsNone(p['result']['carry_yd'])

    def test_ding_rejects_other_shots_failed_clicks_and_nonfinite_or_unordered_samples(self):
        bad = [None, trace('other:1'), dict(trace(), clicked=False), dict(trace(), target=float('nan')),
               dict(trace(), samples=[dict(t=0., edge=100.)]),
               dict(trace(), samples=[dict(t=0., edge=100.), dict(t=.02, edge=110.), dict(t=.01, edge=121.)])]
        for t in bad:
            with self.subTest(trace=t):
                self.assertIsNone(ding_reading(t, 'offline:1'))
        p = self.capture(t=trace('other:1'))
        self.assertFalse(p['complete'])
        self.assertIn('ding_trace', p['missing'])
        self.assertIsNone(p['swing']['ding'])

    def test_snapshot_does_not_alias_before_after_or_ding_inputs(self):
        s = state();a = after_state(s);t = trace()
        p = self.capture(s, a, t)
        s['wind']['high_mph'] = 30
        a['fresh_stats']['carry_yd'] = 99
        t['samples'][-1]['edge'] = 400
        self.assertEqual(p['before']['wind']['high_mph'], 0)
        self.assertEqual(p['after']['fresh_stats']['carry_yd'], 40)
        self.assertEqual(p['swing']['ding']['samples'][-1]['edge'], 121)

    def test_before_snapshot_is_persisted_in_prediction_without_repairing_history(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = ShadowLearning(folder, session_id='offline')
            try:
                s = state()
                frozen = engine.begin_shot('offline:1', s, dict(power=.5))
                s['previous_stats'] = None
                stored = json.loads(engine.db.execute('SELECT data FROM predictions').fetchone()[0])
                self.assertEqual(stored['captured_before'], frozen['captured_before'])
                self.assertIsNotNone(stored['captured_before']['previous_stats'])
            finally:
                engine.close()

    def test_new_result_log_reimport_preserves_trace_and_does_not_promote_forged_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);run = root/'automation_debug/runs/offline'
            run.mkdir(parents=True)
            s = state();a = after_state(s);t = trace('offline:hole_01_1')
            packet = build_shot_capture(t['shot_id'], s, a, .5, dict(power=.5), t, 'next')
            # A tampered complete flag cannot replace absent source evidence.
            s['previous_stats'] = None
            payload = dict(before=s, after=a, observed_power=.5, accuracy_trace=t, shot_capture=packet)
            result_file = run/'hole_01_shot_01_result.json'
            result_file.write_text(json.dumps(payload))
            (run/'hole_01_shot_01_plan.json').write_text(json.dumps(dict(plan=dict(power=.5))))
            original = result_file.read_bytes()
            engine = ShadowLearning(root)
            try:
                self.assertEqual(engine.sync_existing()['added'], 1)
                row = engine.rows()[0]
                self.assertFalse(row['eligible'])
                self.assertFalse(row['use_for_live_control'])
                self.assertEqual(row['evidence']['accuracy_trace'], t)
                self.assertFalse(row['evidence']['shot_capture']['complete'])
                self.assertIn('before_stats_baseline',row['evidence']['shot_capture']['missing'])
                self.assertEqual(engine.sync_existing()['added'], 0)
                self.assertEqual(result_file.read_bytes(), original)
            finally:
                engine.close()

    def test_new_controller_shot_saves_full_capture_and_learning_record_after_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);session = Session(root);session.scope = 'hole_01_'
            engine = ShadowLearning(root, session_id=session.path.name)
            s = state();a = after_state(s)
            frame = np.zeros((728, 1288, 3), np.uint8)
            t = trace(session.shot_id(1));order = []
            model = PowerCalibration()
            def ding(*args):
                order.append('accuracy')
            def read_trace(*args):
                order.append('trace');return t
            def finished(*args, **kwargs):
                order.append('result')
                s['lie'] = 'ROUGH'  # A later read must not rewrite pre-shot evidence.
                return 'next', a
            try:
                with patch.object(live, 'ROOT', root), patch.object(live, 'DEBUG', root/'automation_debug'), \
                        patch.object(live, 'GAME', Mock(completed=False)), patch.object(live, 'DIAGNOSTICS', session), \
                        patch.object(live, 'LEARNING', engine), patch.object(live, 'POWER_MODEL', model), \
                        patch.object(model, 'record'), patch.object(live, 'park_on_meter'), \
                        patch.object(live, 'stable_state', return_value=(s, frame)), \
                        patch.object(live, 'select_club', return_value=(s, frame)), \
                        patch.object(live, 'shot_stats', return_value=s['previous_stats']), \
                        patch.object(live, 'ready_meter', return_value=Meter(73, 598, 932, 767, 1056, 535)), \
                        patch.object(live, 'ShotObserver'), patch.object(live, 'drag_swing', return_value=.5), \
                        patch.object(live, 'accuracy_click', side_effect=ding), \
                        patch.object(live, 'read_shot_accuracy_trace', side_effect=read_trace), \
                        patch.object(live, 'wait_until_next', side_effect=finished), redirect_stdout(io.StringIO()):
                    self.assertEqual(live.run_hole(Mock(), None, {}, Mock(), one_shot=True), 'one_shot')
                self.assertEqual(order, ['accuracy', 'trace', 'result'])
                archived = json.loads((session.path/'hole_01_shot_01_result.json').read_text())
                self.assertEqual(archived['before']['lie'], 'FAIRWAY')
                self.assertNotIn('verified_stats_owner', archived['before']['game_context'])
                self.assertEqual(archived['accuracy_trace'], t)
                self.assertTrue(archived['shot_capture']['complete'], archived['shot_capture']['missing'])
                self.assertTrue((session.path/'hole_01_shot_01_before_capture.json').is_file())
                self.assertTrue((session.path/'hole_01_shot_01_swing_capture.json').is_file())
                self.assertTrue((session.path/'hole_01_shot_01_capture.json').is_file())
                with closing(sqlite3.connect(root / DATABASE)) as database:
                    shot = database.execute(
                        'SELECT hole, stroke, outcome, wind_low_mph, wind_high_mph, '
                        'requested_power_fraction, observed_power_fraction, capture_json FROM shots'
                    ).fetchone()
                self.assertEqual(shot[:7], (s['hole'], s['shot'], 'next', 0, 0, .5, .5))
                self.assertEqual(json.loads(shot[7]), archived['shot_capture'])
                self.assertEqual(import_captures(root / DATABASE, session.path.parent),
                                 dict(inserted=0, already_present=1, skipped=0))
                rows = engine.rows()
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]['eligible'], rows[0]['exclusions'])
                self.assertTrue(rows[0]['evidence']['shot_capture']['complete'])
                self.assertFalse(rows[0]['use_for_live_control'])
            finally:
                engine.close()
                session.close('offline-test')

    def test_sqlite_failure_keeps_the_json_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Session(folder)
            try:
                capture = self.capture()
                capture['shot_id'] = session.shot_id(1)
                with patch('vision_engine.record_capture', side_effect=sqlite3.OperationalError('locked')):
                    session.save_shot_capture(1, capture)
                self.assertEqual(json.loads((session.path / 'shot_01_capture.json').read_text()), capture)
                events = [json.loads(line) for line in (session.path / 'events.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['event'], 'sqlite_shot_write_failed')
            finally:
                session.close('offline-test')

    def test_controller_foreign_stats_never_enter_live_power_calibration(self):
        s = state();a = after_state(s)
        a['game_context']['stats_owner'] = 'opponent1'
        a['game_context']['confidence']['stats_owner'] = .99
        frame = np.zeros((728, 1288, 3), np.uint8)
        game = Mock(completed=False);diagnostics = Mock();model = Mock();learner = Mock()
        diagnostics.shot_id.return_value = 'offline:1'
        model.plan.return_value = dict(power=.5, model='existing baseline')
        learner.begin_shot.return_value = dict(recommendation=dict(selected=None))
        with patch.object(live, 'GAME', game), patch.object(live, 'DIAGNOSTICS', diagnostics), \
                patch.object(live, 'LEARNING', learner), patch.object(live, 'POWER_MODEL', model), \
                patch.object(live, 'park_on_meter'), patch.object(live, 'stable_state', return_value=(s, frame)), \
                patch.object(live, 'select_club', return_value=(s, frame)), \
                patch.object(live, 'shot_stats', return_value=s['previous_stats']), \
                patch.object(live, 'ready_meter', return_value=Meter(73, 598, 932, 767, 1056, 535)), \
                patch.object(live, 'ShotObserver'), patch.object(live, 'drag_swing', return_value=.5), \
                patch.object(live, 'accuracy_click'), patch.object(live, 'read_shot_accuracy_trace', return_value=trace()), \
                patch.object(live, 'wait_until_next', return_value=('complete', a)), \
                patch.object(Path, 'write_text'), patch.object(live.cv2, 'imwrite'), redirect_stdout(io.StringIO()):
            self.assertEqual(live.run_hole(Mock(), None, {}, Mock()), 'complete')
        model.record.assert_not_called()
        capture = learner.finish_shot.call_args.kwargs['evidence']['shot_capture']
        self.assertFalse(capture['verification']['result_belongs_to_tigerwould52'])
        self.assertFalse(capture['distance_training_eligible'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
