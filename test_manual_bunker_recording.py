"""Manual recording regressions: saved images/synthetic captures, never WGT input."""
from contextlib import redirect_stdout
from copy import deepcopy
import ast
import hashlib
import io
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import manual_terrain_recorder as cli
import execution_engine as live
from terrain_recording import (ManualBunkerStore, atomic_json, parse_stat_row,
                                  read_manual_stats_patch, strict_agreement)
from vision_engine import PowerCalibration
from learning_data import solo_owner

ROOT = Path(__file__).resolve().parent


def snapshot(root, token, stamp, shot=4, stats=None, **state_changes):
    path = Path(root)/'wgt_debug/manual_bunker/synthetic'/f'{token}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))  # Synthetic evidence, never a WGT measurement.
    state = dict(player=True, hole=10, shot=shot, mode='FULL', distance=16., unit='yd',
                 distance_source='floating_target', club_name='LW', club_range=52., club_unit='yd',
                 elevation_ft=4., wind=dict(low_mph=13, high_mph=15, mean_mph=14,
                                           screen_dx=0., screen_dy=1.))
    state.update(state_changes)
    context = dict(format='SOLO_STROKE', source='solo_user_confirmed_live_hud',
                   players=['tigerwould52'], active_player='tigerwould52', ball_owner='tigerwould52',
                   turn_state='OWN_READY', ready_controls=True, solo_session_confirmed=True,
                   lie='BUNKER', lie_percent=[40, 50], confidence=dict(lie=.93, lie_percent=.93),
                   solo_evidence=dict(user_confirmed=True, player_header_visible=True,
                                      swing_and_meter_visible=True, multiplayer_conflicts_absent=True))
    return dict(capture_id=token, captured_at=stamp,
                evidence=dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                state=state, context=context, ready=True,
                stats=dict(values=stats or dict(carry_yd=126.22, roll_yd=7.87, power_percent=87.)))


class ManualStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ManualBunkerStore(self.root)
        self.before = [snapshot(self.root, 'before1', 100), snapshot(self.root, 'before2', 101)]
        stats = dict(carry_yd=12.34, roll_yd=2.06, power_percent=26.)
        self.after = [snapshot(self.root, 'after1', 120, shot=5, stats=stats),
                      snapshot(self.root, 'after2', 121, shot=5, stats=stats)]

    def stage(self):
        return self.store.stage(self.before, solo_confirmed=True, now=102)

    def finish(self, **changes):
        return self.store.finish(self.after, now=122, **dict(
            dict(one_manual_shot_confirmed=True, setup_unchanged_confirmed=True), **changes))

    def test_full_measurement_stores_observed_stats_conditions_and_provenance(self):
        pending = self.stage()
        result = self.finish()
        self.assertEqual(result['status'], 'verified')
        self.assertTrue(result['measurement_verified'])
        self.assertTrue(result['attribution_verified'])
        self.assertEqual(result['fields'], dict(club='LW', club_range_yd=52., target_distance_yd=16.,
                         mode='FULL', lie='BUNKER', lie_percent=[40, 50], wind_mph=[13, 15],
                         wind_screen_vector=[0., 1.], elevation_ft=4., carry_yd=12.34, roll_yd=2.06, power_percent=26.))
        self.assertEqual(result['total_yd'], 14.4)
        self.assertIn('derived', result['total_yd_source'])
        self.assertIn('BEFORE', result['conditions_basis'])
        self.assertIn('Shot Stats', result['power_basis'])
        self.assertIsNone(result['motion_evidence'])
        self.assertFalse(result['use_for_automatic_power'])
        self.assertEqual(result['before'], self.before)
        self.assertEqual(result['after'], self.after)
        saved = self.store.folder/'records'/(pending['id'] + '.json')
        self.assertEqual(json.loads(saved.read_text()), result)
        self.assertIsNone(self.store.pending())
        self.assertFalse((self.root/'wgt_calibration/power_observations.json').exists())

    def test_carry_does_not_come_from_target_or_26_percent_reference(self):
        for capture in self.after:
            capture['stats']['values'].update(carry_yd=.83, roll_yd=0., power_percent=39.)
        self.stage()
        result = self.finish()
        self.assertEqual(result['fields']['carry_yd'], .83)
        self.assertEqual(result['fields']['power_percent'], 39.)
        self.assertEqual(result['fields']['target_distance_yd'], 16.)

    def test_missing_conditions_are_null_and_verified_other_fields_are_kept(self):
        for s in self.before:
            s['state'].update(distance=None, elevation_ft=None, wind=None, club_range=None)
            s['context']['lie_percent'] = None
        self.stage()
        result = self.finish()
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['measurement_verified'])
        for field in ('target_distance_yd', 'elevation_ft', 'wind_mph', 'wind_screen_vector', 'club_range_yd', 'lie_percent'):
            self.assertIsNone(result['fields'][field])
            self.assertFalse(result['field_verified'][field])
        self.assertEqual(result['fields']['carry_yd'], 12.34)
        self.assertTrue(result['attribution_verified'])

    def test_missing_roll_does_not_erase_carry_or_invent_zero_roll_or_total(self):
        self.after[1]['stats']['values']['roll_yd'] = None
        self.stage()
        result = self.finish()
        self.assertIsNone(result['fields']['roll_yd'])
        self.assertEqual(result['fields']['carry_yd'], 12.34)
        self.assertIsNone(result['total_yd'])
        self.assertFalse(result['measurement_verified'])
        self.assertFalse(result['attribution_verified'])

    def test_changed_or_invalid_before_fields_abstain_without_copying_other_frame(self):
        self.before[1]['state'].update(club_range=64, distance=17, elevation_ft=True)
        self.before[1]['context']['lie_percent'] = [60, 70]
        self.stage()
        result = self.finish()
        for field in ('club_range_yd', 'target_distance_yd', 'elevation_ft', 'lie_percent'):
            self.assertIsNone(result['fields'][field])

    def test_wind_speed_can_be_kept_without_guessing_a_missing_direction(self):
        for s in self.before:
            s['state'].update(wind=None, conditions_raw=dict(wind=['13-15 MPH', '13-15 MPH']))
        self.stage()
        result = self.finish()
        self.assertEqual(result['fields']['wind_mph'], [13, 15])
        self.assertIsNone(result['fields']['wind_screen_vector'])
        self.assertFalse(result['measurement_verified'])

    def test_conflicting_wind_ranges_or_directions_are_not_averaged(self):
        self.before[1]['state']['wind'].update(high_mph=17, screen_dx=1, screen_dy=0)
        self.stage()
        result = self.finish()
        self.assertIsNone(result['fields']['wind_mph'])
        self.assertIsNone(result['fields']['wind_screen_vector'])

    def test_calm_speed_does_not_invent_an_unseen_direction_arrow(self):
        for s in self.before:
            s['state']['wind'] = dict(low_mph=0, high_mph=0, screen_dx=0., screen_dy=0.)
        self.stage()
        result = self.finish()
        self.assertEqual(result['fields']['wind_mph'], [0, 0])
        self.assertIsNone(result['fields']['wind_screen_vector'])

    def test_observed_manual_flop_mode_is_kept_without_enabling_that_mode(self):
        for s in self.before:
            s['state']['mode'] = 'FLOP'
        self.stage()
        result = self.finish()
        self.assertEqual(result['fields']['mode'], 'FLOP')
        self.assertFalse(result['use_for_automatic_power'])

    def test_target_must_have_verified_yard_units_and_floating_source(self):
        for s in self.before:
            s['state'].update(unit='ft', distance_source='verified_hole_hud')
        self.stage()
        self.assertIsNone(self.finish()['fields']['target_distance_yd'])

    def test_rejects_unconfirmed_solo_unknown_owner_nonbunker_or_wrong_club(self):
        with self.assertRaises(ValueError):
            self.store.stage(self.before, now=102)
        for key, value in (('club_name', 'SW'), ('player', False), ('shot', None)):
            bad = deepcopy(self.before)
            bad[0]['state'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.stage(bad, solo_confirmed=True, now=102)
        bad = deepcopy(self.before)
        bad[1]['context']['lie'] = 'FAIRWAY'
        with self.assertRaises(ValueError):
            self.store.stage(bad, solo_confirmed=True, now=102)

    def test_after_stroke_must_increment_exactly_once_on_same_hole(self):
        self.stage()
        for s in self.after:
            s['state'].update(hole=11, shot=6)
        result = self.finish()
        self.assertFalse(result['attribution_verified'])
        self.assertTrue(any('one stroke' in r for r in result['rejection_reasons']))
        self.assertEqual(result['status'], 'partial')

    def test_no_counter_advance_cannot_authorize_changed_stats(self):
        self.stage()
        for s in self.after:
            s['state']['shot'] = 4
        self.assertFalse(self.finish()['measurement_verified'])

    def test_other_player_stats_are_kept_only_as_unattributed_partial_evidence(self):
        self.stage()
        self.after[1]['context']['active_player'] = 'opponent1'
        result = self.finish()
        self.assertFalse(result['measurement_verified'])
        self.assertIsNone(result['owner'])

    def test_watching_banner_vetoes_contradictory_solo_flags(self):
        self.before[0]['context']['turn_state'] = 'WATCHING_OPPONENT'
        with self.assertRaises(ValueError):
            self.stage()

    def test_hole_out_without_a_next_ready_stroke_is_partial_not_inferred(self):
        self.stage()
        for s in self.after:
            s['state']['shot'] = None
            s['ready'] = False
        result = self.finish()
        self.assertFalse(result['measurement_verified'])
        self.assertEqual(result['fields']['carry_yd'], 12.34)
        self.assertNotEqual(result['fields']['carry_yd'], result['fields']['target_distance_yd'])

    def test_identical_stats_are_rejected_even_after_counter_advances(self):
        self.stage()
        for s in self.after:
            s['stats'] = deepcopy(self.before[0]['stats'])
        result = self.finish()
        self.assertFalse(result['attribution_verified'])
        self.assertTrue(any('stale' in r for r in result['rejection_reasons']))

    def test_missing_before_stats_cannot_be_called_a_fresh_measurement(self):
        self.before[0]['stats']['values']['power_percent'] = None
        self.stage()
        result = self.finish()
        self.assertFalse(result['attribution_verified'])
        self.assertEqual(result['fields']['power_percent'], 26.)

    def test_manual_attestation_is_required_and_does_not_replace_screen_evidence(self):
        self.stage()
        result = self.finish(setup_unchanged_confirmed=False)
        self.assertFalse(result['measurement_verified'])
        self.assertFalse(result['use_for_automatic_power'])

    def test_pending_setup_cannot_be_silently_overwritten(self):
        self.stage()
        original = self.store.pending_path.read_bytes()
        with self.assertRaises(ValueError):
            self.stage()
        self.assertEqual(self.store.pending_path.read_bytes(), original)

    def test_duplicate_finish_cannot_create_a_second_record(self):
        self.stage()
        result = self.finish()
        with self.assertRaises(ValueError):
            self.finish()
        self.assertEqual(len(list((self.store.folder/'records').glob('*.json'))), 1)
        self.assertEqual(result['status'], 'verified')

    def test_crash_after_record_commit_recovers_without_overwriting_result(self):
        self.stage()
        pending = self.store.pending_path.read_bytes()
        result = self.finish()
        path = self.store.folder/'records'/(result['id'] + '.json')
        original = path.read_bytes()
        self.store.pending_path.write_bytes(pending)
        self.assertEqual(self.finish(), result)
        self.assertEqual(path.read_bytes(), original)
        self.assertIsNone(self.store.pending())

    def test_discard_archives_setup_and_allows_a_new_capture(self):
        old = self.stage()
        self.store.discard()
        self.assertTrue((self.store.folder/'discarded'/(old['id'] + '.json')).is_file())
        self.assertIsNone(self.store.pending())
        self.assertNotEqual(self.stage()['id'], old['id'])

    def test_expired_or_future_pending_never_becomes_a_verified_record(self):
        self.stage()
        with self.assertRaises(ValueError):
            self.store.finish(self.after, one_manual_shot_confirmed=True, setup_unchanged_confirmed=True, now=1000)
        future = deepcopy(self.after)
        future[1]['captured_at'] = 150
        with self.assertRaises(ValueError):
            self.store.finish(future, now=122)

    def test_reused_frames_or_missing_changed_escaped_evidence_are_rejected(self):
        for mutate in ('same_id', 'same_time', 'hash', 'escape'):
            bad = deepcopy(self.before)
            if mutate == 'same_id':
                bad[1]['capture_id'] = bad[0]['capture_id']
            elif mutate == 'same_time':
                bad[1]['captured_at'] = bad[0]['captured_at']
            elif mutate == 'hash':
                bad[1]['evidence']['sha256'] = '0'*64
            else:
                bad[1]['evidence']['path'] = '../../outside.png'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.store.stage(bad, solo_confirmed=True, now=102)

    def test_loaded_measurement_does_not_enable_bunker_power(self):
        self.stage()
        record = self.finish()
        model = PowerCalibration([record])
        s = dict(self.before[0]['state'], lie='BUNKER', lie_percent=[40, 50], putt=False)
        self.assertIsNone(model.curve(s))
        self.assertIsNone(model.plan(s)['power'])
        self.assertFalse(model.plan(s)['automatic_swing_allowed'])


class ManualStatsTests(unittest.TestCase):
    def test_strict_units_decimals_and_conflict_rejection(self):
        labels = ['CARRY:', 'CARRY:']
        for texts, expected in ((['0.83 YDS', '0.83 YDS'], .83),
                                (['83 YDS', '83 YDS'], None),
                                (['0.83', '0.83'], None),
                                (['0.83 YDS', '1.83 YDS', '0.83 YDS'], None),
                                (['0.83 FT', '0.83 FT'], None),
                                (['601.00 YDS', '601.00 YDS'], None)):
            self.assertEqual(parse_stat_row('CARRY', labels, texts), expected)
        self.assertIsNone(parse_stat_row('CARRY', ['ROLL:', 'ROLL:'], ['0.83 YDS']*2))
        self.assertIsNone(parse_stat_row('POWER', ['POWER']*2, ['101%']*2))
        self.assertEqual(parse_stat_row('ROLL', ['ROLL']*2, ['0.00 YDS']*2), 0)
        self.assertEqual(parse_stat_row('POWER', ['POWER']*2, ['26%']*2), 26)
        self.assertEqual(parse_stat_row('POWER', ['POWER']*2, ['26% *', '26% -']), 26)
        self.assertIsNone(parse_stat_row('POWER', ['POWER']*2, ['26% 9']*2))

    def test_optional_spin_apex_and_failed_roll_do_not_erase_carry(self):
        responses = [ ['WGT SHOT STATS']*2, ['CARRY']*2, ['0.83 YDS']*2,
                      ['ROLL']*2, ['unreadable'], ['POWER']*2, ['26%']*2 ]
        result = read_manual_stats_patch(np.zeros((121, 205, 3), np.uint8), Mock(side_effect=responses))
        self.assertEqual(result['values'], dict(carry_yd=.83, roll_yd=None, power_percent=26))

    def test_hidden_or_wrong_panel_title_abstains(self):
        result = read_manual_stats_patch(np.zeros((121, 205, 3), np.uint8), Mock(return_value=['CARRY ROLL POWER']*2))
        self.assertEqual(list(result['values'].values()), [None]*3)

    def test_real_saved_stats_are_read_with_actual_units_and_no_invented_carry(self):
        live.setup()
        for name, expected in (('bunker_transition_fixture_20261110.png', (126.22, 7.87, 87)),
                               ('pitch_eleven_fixture_20261110.png', (.83, 0., 26))):
            frame = cv2.imread(str(ROOT/'wgt_debug'/name))
            self.assertIsNotNone(frame)
            result = read_manual_stats_patch(live.crop(frame, (955/1257, 115/718, 1160/1257, 236/718)), live.ocr_variants)
            self.assertEqual(tuple(result['values'][key] for key in ('carry_yd', 'roll_yd', 'power_percent')), expected, result)


class ManualRecorderBoundaryTests(unittest.TestCase):
    def test_actual_saved_before_frame_passes_capture_reader_and_storage_path_without_input(self):
        live.setup()
        frame = cv2.imread(str(ROOT/'wgt_debug/bunker_transition_fixture_20261110.png'))
        rect = dict(left=29, top=48, width=frame.shape[1], height=frame.shape[0])
        with tempfile.TemporaryDirectory() as folder, patch.object(live, 'setup'), \
                patch.object(live, 'GAME', None), patch.object(live, 'user32') as native, \
                patch.object(live, 'key_down', return_value=False), patch.object(live, 'find_wgt', return_value=10), \
                patch.object(live, 'rect_of', return_value=rect), patch.object(live, 'point_in_wgt', return_value=True), \
                patch.object(live, 'capture', side_effect=lambda *args: frame.copy()) as capture, \
                patch.object(cli.time, 'sleep'), redirect_stdout(io.StringIO()):
            native.GetForegroundWindow.return_value = 10
            native.IsIconic.return_value = False
            pair = cli.capture_pair(Path(folder), 'before', countdown=0)
            self.assertEqual(len(list((Path(folder)/'wgt_debug/manual_bunker').glob('*/capture.json'))), 1)
            pending = ManualBunkerStore(folder).stage(pair, solo_confirmed=True)
        self.assertEqual(capture.call_count, 2)
        self.assertEqual(pending['before_fields']['club'], 'LW')
        self.assertEqual(pending['before_fields']['club_range_yd'], 52.)
        self.assertEqual(pending['before_fields']['target_distance_yd'], 12.)
        self.assertEqual(pending['before_fields']['elevation_ft'], 3.)
        self.assertEqual(pending['before_fields']['lie_percent'], [40, 50])
        self.assertEqual(pending['baseline_stats']['carry_yd'], 126.22)
        native.SendInput.assert_not_called()
        native.SetForegroundWindow.assert_not_called()

    def test_console_before_manual_after_workflow_and_restart_safe_files(self):
        with tempfile.TemporaryDirectory() as folder:
            before = [snapshot(folder, 'b1', 100), snapshot(folder, 'b2', 101)]
            stats = dict(carry_yd=7.25, roll_yd=1.01, power_percent=31.)
            after = [snapshot(folder, 'a1', 120, shot=5, stats=stats), snapshot(folder, 'a2', 121, shot=5, stats=stats)]
            output = io.StringIO()
            with patch.object(cli, 'ROOT', Path(folder)), patch.object(cli, 'ControllerLock') as lock, \
                    patch.object(cli, 'capture_pair', side_effect=[before, after]) as capture, \
                    patch('builtins.input', side_effect=['1', 'SOLO', '2', 'ONE', 'Q']), \
                    patch('terrain_recording.time.time', return_value=122), redirect_stdout(output):
                self.assertEqual(cli.main([]), 0)
            records = list((Path(folder)/'wgt_calibration/manual_bunker/records').glob('*.json'))
            self.assertEqual(len(records), 1)
            self.assertTrue(json.loads(records[0].read_text())['measurement_verified'])
            self.assertIn('RESULT: VERIFIED', output.getvalue())
            self.assertEqual([c.args[1] for c in capture.call_args_list], ['before', 'after'])
            lock.return_value.close.assert_called_once()

    def test_after_without_before_and_unconfirmed_before_never_capture(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cli, 'ROOT', Path(folder)), \
                patch.object(cli, 'ControllerLock'), patch.object(cli, 'capture_pair') as capture, \
                patch('builtins.input', side_effect=['2', '1', 'NO', 'Q']), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]), 0)
        capture.assert_not_called()

    def test_existing_controller_lock_blocks_recorder_before_capture(self):
        with patch.object(cli, 'ControllerLock') as lock, patch.object(cli, 'capture_pair') as capture, \
                redirect_stdout(io.StringIO()):
            lock.return_value.acquire.side_effect = RuntimeError('already running')
            self.assertEqual(cli.main([]), 1)
        capture.assert_not_called()

    def test_foreground_geometry_and_coverage_checks_stop_before_capture(self):
        for failure in ('foreground', 'geometry', 'covered'):
            rect = dict(left=0, top=0, width=900 if failure == 'geometry' else 1288, height=728)
            with tempfile.TemporaryDirectory() as folder, patch.object(live, 'setup'), \
                    patch.object(live, 'user32') as native, patch.object(live, 'key_down', return_value=False), \
                    patch.object(live, 'find_wgt', return_value=10), patch.object(live, 'rect_of', return_value=rect), \
                    patch.object(live, 'point_in_wgt', return_value=failure != 'covered'), \
                    patch.object(live, 'capture') as capture, redirect_stdout(io.StringIO()):
                native.GetForegroundWindow.return_value = 99 if failure == 'foreground' else 10
                native.IsIconic.return_value = False
                with self.assertRaises(ValueError):
                    cli.capture_pair(Path(folder), 'before', countdown=0)
            capture.assert_not_called()
            native.SendInput.assert_not_called()

    def test_read_only_native_wrapper_blocks_input_and_activation_apis(self):
        native = Mock()
        wrapper = cli.ReadOnlyNative(native)
        self.assertIs(wrapper.GetForegroundWindow, native.GetForegroundWindow)
        for name in ('SendInput', 'mouse_event', 'keybd_event', 'SetCursorPos', 'SetForegroundWindow', 'PostMessageW'):
            with self.assertRaisesRegex(RuntimeError, 'forbids'):
                getattr(wrapper, name)
        native.SendInput.assert_not_called()

    def test_native_guard_is_restored_after_exception(self):
        native = Mock()
        holder = Mock(user32=native)
        with self.assertRaises(ValueError):
            with cli.read_only_native(holder):
                self.assertIsInstance(holder.user32, cli.ReadOnlyNative)
                raise ValueError('test')
        self.assertIs(holder.user32, native)

    def test_dependency_check_does_not_query_windows_or_capture(self):
        with patch.object(live, 'setup') as setup, patch.object(live, 'find_wgt') as find, \
                patch.object(cli, 'capture_pair') as capture, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--check']), 0)
        setup.assert_not_called()
        find.assert_not_called()
        capture.assert_not_called()

    def test_f9_stops_recording_without_any_release_or_input(self):
        with patch.object(live, 'key_down', return_value=True), patch.object(live, 'send') as send, \
                patch.object(live, 'mouse_up') as release:
            with self.assertRaises(KeyboardInterrupt):
                cli.check_stop(live)
        send.assert_not_called()
        release.assert_not_called()

    def test_recording_sources_never_call_control_or_game_launch_functions(self):
        tree = ast.parse((ROOT/'manual_terrain_recorder.py').read_text(encoding='utf-8'))
        forbidden = {'send', 'move_to', 'mouse_up', 'activate_wgt', 'run_hole', 'run_round',
                     'drag_swing', 'accuracy_click', 'press', 'release', 'Popen', 'startfile', 'system'}
        names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))}
        self.assertFalse(names & forbidden)
        launcher = (ROOT/'Record_Manual_Bunker.cmd').read_text().lower()
        self.assertIn('manual_terrain_recorder.py', launcher)
        self.assertNotIn('start_agent', launcher)
        self.assertNotIn('execution_engine.py', launcher)

    def assert_historical_calibration_preserved(self, original, current):
        self.assertEqual({k: v for k, v in current.items() if k != 'observations'},
                         {k: v for k, v in original.items() if k != 'observations'})
        old, rows = original['observations'], current['observations']
        self.assertGreaterEqual(len(rows), len(old))
        self.assertEqual(rows[:len(old)], old)  # No historical edits, deletion or reordering.
        identities = [r['shot_id'] for r in rows]
        self.assertTrue(all(isinstance(i, str) and i for i in identities))
        self.assertEqual(len(identities), len(set(identities)))

    def test_reference_unchanged_and_historical_calibration_preserved_with_verified_appends(self):
        backup = ROOT/'repair_backups/bunker_model_20261113'
        reference = 'bunker_reference_manual.json'
        self.assertEqual((ROOT/'wgt_calibration'/reference).read_bytes(), (backup/reference).read_bytes())
        original = json.loads((backup/'power_observations.json').read_text(encoding='utf-8-sig'))
        current = json.loads((ROOT/'wgt_calibration/power_observations.json').read_text(encoding='utf-8-sig'))
        self.assert_historical_calibration_preserved(original, current)
        # A live controller may append real shots. Each append must bind to its
        # saved result; merely setting verified=True cannot bypass this check.
        for row in current['observations'][len(original['observations']):]:
            with self.subTest(shot_id=row['shot_id']):
                self.assertIs(row['verified'], True)
                self.assertIs(row['owner_verified'], True)
                self.assertEqual(row['owner'], 'tigerwould52')
                match = re.fullmatch(r'(\d{8}_\d{6}_[a-f0-9]{8}):(?:(hole_\d{2}_))?(\d+)', row['shot_id'])
                self.assertIsNotNone(match)
                session, scope, number = match.groups()
                path = ROOT/'wgt_debug/runs'/session/f'{scope or ""}shot_{int(number):02d}_result.json'
                payload = json.loads(path.read_text(encoding='utf-8-sig'))
                before, after = payload['before'], payload['after']
                self.assertIs(before['player'], True)
                self.assertIs(before['game_context']['permission']['allowed'], True)
                self.assertEqual(before['game_context']['active_player'], 'tigerwould52')
                self.assertEqual(after['hole'], before['hole'])
                self.assertEqual(after['shot'], before['shot']+1)
                self.assertEqual(after['movement_verification'], 'video motion plus stable stroke/HUD advance')
                motion = after['motion']
                self.assertFalse(motion.get('capture_error'))
                self.assertTrue(motion.get('scene_motion_observed') or motion.get('moving_ball_candidate_tracked'))
                context = after['game_context']
                self.assertIn(context.get('stats_owner'), (None, 'tigerwould52'))
                self.assertTrue(solo_owner(after) or context.get('stats_owner') == 'tigerwould52'
                                and context.get('confidence', {}).get('stats_owner', 0) >= .85)
                self.assertEqual(row['stats'], after['fresh_stats'])
                self.assertEqual(row['power'], payload['observed_power'])
                self.assertEqual(row['game_power'], row['stats']['power_pct']/100)
                self.assertLessEqual(abs(row['power']-row['game_power']), .035)
                for key in ('mode', 'unit', 'club_name', 'club_range', 'lie', 'elevation_ft', 'wind'):
                    self.assertEqual(row[key], before[key], key)
                total = row['mode'] in ('CHIP', 'PITCH', 'PUTT')
                self.assertEqual(row['travel_basis'], 'total' if total else 'carry')
                expected = row['stats']['total_yd'] if total else row['stats']['carry_yd']
                self.assertEqual(row['travel'], expected*3 if row['mode'] == 'PUTT' else expected)

    def test_historical_preservation_check_rejects_tampering_deletion_reorder_and_duplicates(self):
        original = dict(schema=1, observations=[dict(shot_id='old:1', power=.2),
                                               dict(shot_id='old:2', power=.4)])
        for defect in ('edit', 'delete', 'reorder', 'duplicate', 'metadata'):
            current = deepcopy(original)
            if defect == 'edit':
                current['observations'][0]['power'] = .9
            elif defect == 'delete':
                current['observations'].pop()
            elif defect == 'reorder':
                current['observations'].reverse()
            elif defect == 'duplicate':
                current['observations'].append(deepcopy(current['observations'][0]))
            else:
                current['schema'] = 2
            with self.subTest(defect=defect), self.assertRaises(AssertionError):
                self.assert_historical_calibration_preserved(original, current)


if __name__ == '__main__':
    unittest.main(verbosity=2)
