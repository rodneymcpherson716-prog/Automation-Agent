"""Bunker model and input-boundary regressions using saved/synthetic data only."""
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from terrain_handler import bunker_calibration_plan, bunker_power_estimate, is_bunker_state
from context_manager import ContextObserver, GameCoordinator, verified_bunker_readings
from automation_core import plan_power
from game_state import combine_context, parse_context
from vision_engine import PowerCalibration
import execution_engine as live

ROOT = Path(__file__).resolve().parent


def state(**updates):
    return dict(dict(hole=1, shot=3, mode='FULL', putt=False, distance=16., unit='yd',
                     club_name='LW', club_range=52., club_unit='yd', lie='BUNKER',
                     lie_percent=[40, 50], elevation_ft=0.,
                     wind=dict(mean_mph=0., screen_dx=0., screen_dy=0.)), **updates)


class BunkerPowerTests(unittest.TestCase):
    def test_calibration_trial_changes_with_live_conditions(self):
        setup=state(player=True)
        first=bunker_calibration_plan(setup,1,3)
        uphill=bunker_calibration_plan(dict(setup,elevation_ft=3),1,3)
        self.assertTrue(first['automatic_swing_allowed'])
        self.assertGreater(uphill['power'],first['power'])
        for change in (dict(shot=4),dict(lie='FAIRWAY'),dict(lie_percent=[30,40]),dict(wind=None)):
            with self.assertRaises(ValueError):
                bunker_calibration_plan(dict(setup,**change),1,3)

    def test_calibration_trial_still_requires_verified_solo_ball_owner(self):
        setup=state(player=True)
        plan=bunker_calibration_plan(setup,1,3)
        game=GameCoordinator(Mock(),Mock(),Mock(),Mock(),target_visible=Mock())
        with patch.object(game,'require_turn',return_value={'format':'ALTERNATE_SHOT'}) as verify:
            with self.assertRaisesRegex(RuntimeError,'solo player and ball owner'):
                game.arm_stroke(Mock(),None,{},setup,Mock(),bunker_trial=plan)
        verify.assert_called_once()
        game.target_visible.assert_not_called()

    def test_measured_layup_caps_the_target_at_the_recorded_recovery(self):
        setup=state(player=True,distance=43,elevation_ft=6.)
        reference=dict(kind='single_bunker_calibration_measurement',club='LW',mode='FULL',
                       club_range_yd=52.,displayed_lie_percent=[40,50],after={'lie':'GREEN'},
                       shot_stats={'total_yd':23.46},elevation_ft=2/3,
                       wind={'screen_dy':-.7071,'mean_mph':16.},observed_meter_power=.6169)
        plan=bunker_calibration_plan(setup,1,3,reference=reference)
        self.assertEqual(plan['strategy_target'],23.46)
        self.assertGreater(plan['power'],reference['observed_meter_power'])
        with self.assertRaises(ValueError):
            bunker_calibration_plan(setup,1,3,reference=dict(reference,club_range_yd=51.))

    def test_real_saved_bunker_label_preserves_display_without_inferring_loss(self):
        live.setup()
        frame = cv2.imread(str(ROOT/'wgt_debug/bunker_transition_fixture_20261110.png'))
        self.assertIsNotNone(frame)
        result = verified_bunker_readings([dict(text='BUNKER')], frame)
        self.assertIsNotNone(result)
        self.assertEqual(result['lie'], 'BUNKER')
        self.assertEqual(result['lie_percent'], [40, 50])
        self.assertGreaterEqual(result['confidence'], .85)
        self.assertNotIn('lie_loss_percent', result)

    def test_displayed_lie_is_never_silently_renamed_to_loss(self):
        for dash in ('-', '\u2013', '\u2014'):
            for label in ('BUNKER', 'SAND'):
                parsed = parse_context(f'{label} | 40{dash}50% LIE')
                self.assertEqual(parsed['lie'], 'BUNKER')
                self.assertEqual(parsed['lie_percent'], [40, 50])
                self.assertNotIn('lie_loss_percent', parsed)
        parsed = parse_context('BUNKER | 40-50% LIE\nLIE LOSS: 10-20%')
        self.assertEqual(parsed['lie_percent'], [40, 50])
        self.assertEqual(parsed['lie_loss_percent'], [10, 20])

    def test_conflicting_lie_ranges_abstain(self):
        parsed = parse_context('BUNKER | 40-50% LIE\nBUNKER | 60-70% LIE')
        self.assertNotIn('lie_percent', parsed)
        readings = [dict(text=f'BUNKER | {p}% LIE', confidence=93) for p in ('40-50', '60-70')]
        self.assertNotIn('lie_percent', combine_context(readings, 100))

    def test_flat_and_region_context_raw_formats_distinguish_lie_from_aim_hazards(self):
        for raw in ([dict(text='LIE: BUNKER (40-50%)')],
                    dict(lie=[dict(text='BUNKER | unreadable')])):
            self.assertTrue(is_bunker_state(dict(game_context=dict(raw=raw))))
        raw = [dict(text='LIE: FAIRWAY\nCURRENT AIM BUNKER: 100-120 YD')]
        self.assertFalse(is_bunker_state(dict(lie='FAIRWAY', game_context=dict(raw=raw))))
        self.assertFalse(is_bunker_state(dict(game_context=dict(raw=[]))))

    def test_existing_bunker_ocr_contract_preserves_93_percent_label_confidence(self):
        data = dict(text=['BUNKER', '|', '40-50%', 'LIE'], conf=[93, 20, 94, 90])
        frame = np.zeros((728, 1288, 3), np.uint8)
        with patch('context_manager.pytesseract.image_to_data', return_value=data):
            result = verified_bunker_readings([dict(text='BUNKER')], frame)
        self.assertEqual(result, dict(lie='BUNKER', lie_percent=[40, 50], confidence=.93))
        for bad in (dict(data, conf=[84, 20, 94, 90]),
                    dict(data, text=['BUNKER', '|', '50-40%', 'LIE'])):
            with patch('context_manager.pytesseract.image_to_data', return_value=bad):
                self.assertIsNone(verified_bunker_readings([dict(text='BUNKER')], frame))

    def test_context_observer_to_coordinator_to_model_preserves_display(self):
        frame = np.zeros((728, 1288, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            observer = ContextObserver(folder, reader=lambda image, psm:
                                       dict(text='BUNKER | 40-50% LIE', confidence=93, words=[]))
            observer.regions = {'lie': observer.regions['lie']}
            with patch('context_manager.verified_bunker_readings', return_value=
                       dict(lie='BUNKER', lie_percent=[40, 50], confidence=.93)):
                s = state(lie=None, lie_percent=None, lie_loss_percent=[90, 95])
                game = GameCoordinator(observer, Mock(), Mock(), Mock())
                game.inspect(frame, s, captured_at=100)
        self.assertEqual(s['lie_percent'], [40, 50])
        self.assertIsNone(s['lie_loss_percent'])  # Clear stale previous-shot loss.
        result = PowerCalibration().plan(s)
        self.assertAlmostEqual(result['estimated_power'], 16/52)
        self.assertIsNone(result['power'])

    def test_context_stability_requires_matching_percentages_on_both_frames(self):
        frame = np.zeros((728, 1288, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            observer = ContextObserver(folder, reader=lambda image, psm:
                                       dict(text='BUNKER | 40-50% LIE', confidence=93, words=[]))
            observer.regions = {'lie': observer.regions['lie']}
            results = [dict(lie='BUNKER', lie_percent=p, confidence=.93)
                       for p in ([40, 50], [60, 70], [60, 70])]
            with patch('context_manager.verified_bunker_readings', side_effect=results):
                contexts = [observer.observe(frame, state(), now=t) for t in (100, 101, 102)]
        self.assertEqual([c['stable_frames'] for c in contexts], [1, 1, 2])

    def test_distance_and_club_range_scale_the_provisional_estimate(self):
        values = [bunker_power_estimate(state(distance=d))['estimated_power'] for d in (4, 8, 16, 32)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[2], 2*values[1])
        for club, yards in (('LW', 52), ('SW', 64), ('PW', 80)):
            p = bunker_power_estimate(state(club_name=club, club_range=yards))
            self.assertAlmostEqual(p['estimated_power'], 16/yards)
            self.assertFalse(p['automatic_swing_allowed'])

    def test_lie_scenarios_change_without_claiming_display_means_loss(self):
        for lie in ([20, 30], [40, 50], [70, 80], [100, 100]):
            p = bunker_power_estimate(state(lie_percent=lie))
            self.assertAlmostEqual(p['estimated_power'], 16/52)
            self.assertIsNone(p['lie_loss_percent'])
            self.assertIn('uninterpreted', p['lie_percent_semantics'])
            bounds = p['lie_scenarios']['if_display_means_retained_effectiveness']
            self.assertAlmostEqual(bounds[0], (16/52)/(lie[1]/100))
            self.assertFalse(p['scenario_ranges_are_confidence_intervals'])
        self.assertIsNone(p['lie_scenarios']['if_display_means_loss'][1])

    def test_explicit_loss_label_remains_a_separate_unvalidated_hypothesis(self):
        p = bunker_power_estimate(state(lie_loss_percent=[10, 20]))
        self.assertAlmostEqual(p['estimated_power'], 16/52)
        bounds = p['lie_scenarios']['explicit_loss_label_linear_hypothesis']
        self.assertAlmostEqual(bounds[0], (16/52)/.9)
        self.assertAlmostEqual(bounds[1], (16/52)/.8)
        self.assertEqual(p['lie_percent'], [40, 50])
        self.assertEqual(p['lie_loss_percent'], [10, 20])
        self.assertFalse(p['automatic_swing_allowed'])

    def test_combined_condition_matrix_stays_diagnostic_and_auditable(self):
        winds = ((0, 0, 0), (14, 0, 1), (14, 0, -1), (14, 1, 0), (30, -.6, .8))
        for distance, lie, elevation, (speed, dx, dy) in product(
                (4, 8, 16, 32), ([20, 30], [40, 50], [70, 80], [100, 100]),
                (-4, 0, 4), winds):
            s = state(distance=distance, lie_percent=lie, elevation_ft=elevation,
                      wind=dict(mean_mph=speed, screen_dx=dx, screen_dy=dy))
            before = deepcopy(s)
            with self.subTest(distance=distance, lie=lie, elevation=elevation, wind=s['wind']):
                p = bunker_power_estimate(s)
                self.assertIsNone(p['power'])
                self.assertFalse(p['automatic_swing_allowed'])
                self.assertEqual(p['calibration_points'], 0)
                self.assertAlmostEqual(p['estimated_power'],
                                       (distance+elevation/3)*(1+.003*dy*speed)/52)
                self.assertEqual(p['observed_inputs']['lie_percent'], lie)
                self.assertEqual(p['observed_inputs']['club_range_yd'], 52)
                self.assertEqual(p['observed_inputs']['shot_mode'], 'FULL')
                self.assertEqual(p['nominal_lie_conversion'], 'none')
                self.assertEqual(s, before)
                json.dumps(p, allow_nan=False)

    def test_diagnostic_output_cannot_mutate_state_or_manual_reference(self):
        s = state()
        reference = json.loads((ROOT/'wgt_calibration/bunker_reference_manual.json').read_text(
            encoding='utf-8-sig'))
        before = deepcopy(reference)
        p = bunker_power_estimate(s, reference)
        p['contextual_reference']['reported_facts']['lie_percent'][0] = 1
        p['observed_inputs']['wind']['mean_mph'] = 99
        p['observed_inputs']['lie_percent'][0] = 1
        self.assertEqual(reference, before)
        self.assertEqual(s, state())

    def test_head_tail_crosswind_and_gust_range(self):
        plans = {}
        for name, dx, dy in (('head', 0, 1), ('tail', 0, -1), ('cross', 1, 0)):
            plans[name] = bunker_power_estimate(state(wind=dict(
                mean_mph=14, low_mph=13, high_mph=15, screen_dx=dx, screen_dy=dy)))
        calm = bunker_power_estimate(state())['estimated_power']
        self.assertGreater(plans['head']['estimated_power'], calm)
        self.assertLess(plans['tail']['estimated_power'], calm)
        self.assertAlmostEqual(plans['cross']['estimated_power'], calm)
        self.assertEqual(plans['cross']['crosswind_mph'], 14)
        for p in plans.values():
            lo, hi = p['lie_scenarios']['no_lie_conversion']
            self.assertLessEqual(lo, p['estimated_power'])
            self.assertGreaterEqual(hi, p['estimated_power'])

    def test_elevation_in_feet_and_strategy_target_are_used(self):
        values = [bunker_power_estimate(state(elevation_ft=e))['estimated_power'] for e in (-4, 0, 4)]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        p = bunker_power_estimate(state(elevation_ft=4, strategy=dict(target=12)))
        self.assertAlmostEqual(p['estimated_power'], (12+4/3)/52)

    def test_manual_reference_is_read_only_context_not_a_carry_or_a_fit(self):
        path = ROOT/'wgt_calibration/bunker_reference_manual.json'
        original = path.read_bytes()
        ref = json.loads(original.decode('utf-8-sig'))
        model = PowerCalibration.load(ROOT/'wgt_calibration/power_observations.json')
        for distance in (8, 16, 32):
            p = model.plan(state(distance=distance, elevation_ft=4, wind=dict(
                mean_mph=14, low_mph=13, high_mph=15, screen_dx=0, screen_dy=1)))
            r = p['contextual_reference']
            self.assertEqual(r['reported_facts'], ref)
            self.assertEqual(r['reported_facts']['power_percent'], 26)
            self.assertIsNone(r['inferred_carry_yd'])
            self.assertFalse(r['used_for_fit'])
            self.assertNotEqual(p['estimated_power'], .26)
            self.assertEqual(p['calibration_points'], 0)
        self.assertIsNone(ref['carry_distance_yd'])
        self.assertFalse(ref['use_for_automatic_power'])
        self.assertEqual(path.read_bytes(), original)

    def test_zero_verified_bunker_records_and_old_lw_rows_cannot_fit_sand(self):
        model = PowerCalibration.load(ROOT/'wgt_calibration/power_observations.json')
        self.assertEqual(sum(r.get('verified') is True and r.get('lie') == 'BUNKER'
                             for r in model.records), 0)
        self.assertIsNone(model.curve(state()))
        points = [dict(verified=True, shot_id=str(i), lie='BUNKER', mode='FULL', unit='yd',
                       club_name='LW', club_range=52, elevation_ft=0, wind_mph=0,
                       power=p, travel=d) for i, (p, d) in enumerate(((.2, 8), (.6, 30)))]
        self.assertIsNone(PowerCalibration(points).curve(state()))
        self.assertEqual(PowerCalibration(points).plan(state())['calibration_points'], 0)

    def test_future_records_preserve_lie_wind_and_real_stats_without_inferring_travel(self):
        s = state(elevation_ft=4, wind=dict(mean_mph=14, screen_dx=1, screen_dy=0))
        stats = dict(carry_yd=7., roll_yd=2., total_yd=9., power_pct=26., spin_pct=0., apex_ft=4.)
        with tempfile.TemporaryDirectory() as folder:
            model = PowerCalibration()
            self.assertTrue(model.record(Path(folder)/'observations.json', s, stats, .26, 'offline:1'))
        r = model.records[0]
        self.assertEqual(r['lie_percent'], [40, 50])
        self.assertIsNone(r['lie_loss_percent'])
        self.assertEqual(r['wind'], s['wind'])
        self.assertEqual(r['travel'], 7.)
        self.assertEqual(r['target_distance'], 16.)
        self.assertIsNone(model.curve(s))

    def test_invalid_and_missing_measurements_fail_closed(self):
        for changes in (dict(distance=0), dict(distance=float('nan')), dict(distance=True),
                        dict(club_range=-1), dict(unit='ft'), dict(club_unit='ft'),
                        dict(putt=True), dict(mode='PUTT'), dict(club_name='7I'),
                        dict(lie_percent=None), dict(lie_percent=[50, 40]),
                        dict(lie_percent=[0, 40]), dict(lie_percent=[40, float('inf')]),
                        dict(lie_loss_percent=[40, 100]), dict(elevation_ft=None),
                        dict(elevation_ft=-100), dict(wind=None),
                        dict(wind=dict(mean_mph=-1, screen_dx=0, screen_dy=1)),
                        dict(wind=dict(mean_mph=14, screen_dx=0, screen_dy=0)),
                        dict(wind=dict(mean_mph=14, low_mph=15, high_mph=13, screen_dx=0, screen_dy=1)),
                        dict(wind=dict(mean_mph=14, screen_dx=float('nan'), screen_dy=1))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bunker_power_estimate(state(**changes))

    def test_extreme_distances_do_not_hide_out_of_range_estimates(self):
        for distance in (.1, 100):
            p = bunker_power_estimate(state(distance=distance))
            self.assertTrue(p['outside_controlled_power_range'])
            self.assertAlmostEqual(p['estimated_power'], distance/52)
            self.assertIsNone(p['power'])
            json.dumps(p, allow_nan=False)

    def test_generic_fallback_and_selection_cannot_authorize_sand(self):
        for s in (state(), state(lie='SAND'), state(lie=None, game_context=dict(
                raw=dict(lie=[dict(text='BUNKER | unreadable')])))):
            for selection in (True, False):
                with self.assertRaisesRegex(ValueError, 'bunker swings blocked'):
                    plan_power(s, for_selection=selection)
                p = PowerCalibration().plan(s, for_selection=selection)
                self.assertIsNone(p['power'])
                self.assertFalse(p['automatic_swing_allowed'])

    def test_final_input_guard_blocks_verified_and_unverified_bunker(self):
        for s in (state(), state(lie=None, lie_percent=None, game_context=dict(
                raw=dict(lie=[dict(text='BUNKER | 40-50% LIE')])))):
            observer = Mock()
            game = GameCoordinator(observer, Mock(), Mock(), Mock(), target_visible=Mock())
            safety = Mock(deadline=None)
            with patch.object(game, 'require_turn', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'no swing'):
                    game.arm_stroke(safety, None, {}, s, Mock())
            safety.press.assert_not_called()
            game.move_to.assert_not_called()
            game.target_visible.assert_not_called()

    def test_controller_stops_before_club_selection_drag_or_accuracy(self):
        frame = np.zeros((728, 1288, 3), np.uint8)
        for model in (None, PowerCalibration()):
            game = Mock(completed=False)
            with patch.object(live, 'GAME', game), patch.object(live, 'POWER_MODEL', model), \
                    patch.object(live, 'DIAGNOSTICS', None), patch.object(live, 'park_on_meter'), \
                    patch.object(live, 'stable_state', return_value=(state(), frame)), \
                    patch.object(live, 'select_club') as select, patch.object(live, 'send') as send, \
                    patch.object(live, 'drag_swing') as drag, patch.object(live, 'accuracy_click') as accuracy, \
                    patch.object(live, 'diagnostic_state') as diagnostic, patch.object(live.cv2, 'imwrite'):
                with self.assertRaisesRegex(RuntimeError, 'bunker swings blocked'):
                    live.run_hole(Mock(), None, {}, Mock())
            select.assert_not_called()
            drag.assert_not_called()
            accuracy.assert_not_called()
            send.assert_not_called()
            self.assertIsNone(diagnostic.call_args.args[1]['plan']['power'])

    def test_direct_club_selector_blocks_before_pointer_or_wheel_input(self):
        with patch.object(live, 'move_to') as move, patch.object(live, 'send') as send:
            with self.assertRaisesRegex(RuntimeError, 'bunker swings blocked'):
                live.select_club(Mock(), None, {}, state(), None)
        move.assert_not_called()
        send.assert_not_called()

    def test_bunker_newly_detected_after_club_selection_still_blocks_before_swing(self):
        frame = np.zeros((728, 1288, 3), np.uint8)
        game = Mock(completed=False)
        s = state(lie='FAIRWAY', lie_percent=None)
        def reread(safety, sct, rect, observed):
            if game.require_turn.call_count == 2:
                observed.update(lie='BUNKER', lie_percent=[40, 50])
        game.require_turn.side_effect = reread
        with patch.object(live, 'GAME', game), patch.object(live, 'POWER_MODEL', PowerCalibration()), \
                patch.object(live, 'DIAGNOSTICS', None), patch.object(live, 'park_on_meter'), \
                patch.object(live, 'stable_state', return_value=(s, frame)), \
                patch.object(live, 'select_club', return_value=(s, frame)), \
                patch.object(live, 'drag_swing') as drag, patch.object(live, 'accuracy_click') as accuracy, \
                patch.object(live, 'diagnostic_state'), patch.object(live.cv2, 'imwrite'):
            with self.assertRaisesRegex(RuntimeError, 'bunker swings blocked'):
                live.run_hole(Mock(), None, {}, Mock())
        drag.assert_not_called()
        accuracy.assert_not_called()

    def test_percentage_change_during_final_reread_requires_replanning(self):
        s = state(lie='ROUGH')
        game = GameCoordinator(Mock(), Mock(), Mock(), Mock())
        def reread(*args):
            s['lie_percent'] = [60, 70]
            return {}
        with patch.object(game, 'require_turn', side_effect=reread):
            with self.assertRaisesRegex(RuntimeError, 'Lie or strategy changed'):
                game.arm_stroke(Mock(), None, {}, s, Mock())

    def test_in_place_lie_or_strategy_change_requires_replanning(self):
        for field in ('lie_percent', 'lie_loss_percent', 'strategy'):
            s = state(lie='ROUGH', lie_loss_percent=[10, 20], strategy=dict(target=16))
            game = GameCoordinator(Mock(), Mock(), Mock(), Mock(), target_visible=Mock(return_value=True))
            def reread(*args):
                if field == 'strategy':
                    s[field]['target'] = 12
                else:
                    s[field][:] = [60, 70]
                return {}
            with self.subTest(field=field), patch.object(game, 'require_turn', side_effect=reread), \
                    patch('context_manager.swing_budget', return_value=dict(allowed=True, deadline=None)):
                with self.assertRaisesRegex(RuntimeError, 'Lie or strategy changed'):
                    game.arm_stroke(Mock(deadline=None), None,
                                    dict(left=0, top=0, width=1288, height=728), s,
                                    Mock(drag_seconds=1., accuracy_delay=1., anchor=(.5, .5)))
            game.target_visible.assert_not_called()

    def test_final_guard_keeps_bunker_evidence_even_if_reread_would_erase_it(self):
        s = state(lie=None, lie_percent=None, game_context=dict(
            raw=dict(lie=[dict(text='BUNKER | unreadable')])))
        game = GameCoordinator(Mock(), Mock(), Mock(), Mock(), target_visible=Mock(return_value=True))
        def reread(*args):
            s['game_context'] = dict(raw={})
            return {}
        with patch.object(game, 'require_turn', side_effect=reread) as require, \
                patch('context_manager.swing_budget', return_value=dict(allowed=True, deadline=None)):
            with self.assertRaisesRegex(RuntimeError, 'bunker swings blocked'):
                game.arm_stroke(Mock(deadline=None), None,
                                dict(left=0, top=0, width=1288, height=728), s,
                                Mock(drag_seconds=1., accuracy_delay=1., anchor=(.5, .5)))
        require.assert_not_called()
        game.target_visible.assert_not_called()

    def test_controller_keeps_bunker_evidence_before_context_reread(self):
        frame = np.zeros((728, 1288, 3), np.uint8)
        s = state(lie=None, lie_percent=None, game_context=dict(
            raw=dict(lie=[dict(text='BUNKER | unreadable')])))
        game = Mock(completed=False)
        def reread(safety, sct, rect, observed):
            observed['game_context'] = dict(raw={})
        game.require_turn.side_effect = reread
        with patch.object(live, 'GAME', game), patch.object(live, 'POWER_MODEL', None), \
                patch.object(live, 'DIAGNOSTICS', None), patch.object(live, 'park_on_meter'), \
                patch.object(live, 'stable_state', return_value=(s, frame)), \
                patch.object(live, 'select_club', side_effect=AssertionError('Bunker reached club selection')) as select, \
                patch.object(live, 'drag_swing') as drag, patch.object(live, 'accuracy_click') as accuracy, \
                patch.object(live, 'diagnostic_state'), patch.object(live.cv2, 'imwrite'):
            with self.assertRaisesRegex(RuntimeError, 'bunker swings blocked'):
                live.run_hole(Mock(), None, {}, Mock())
        game.require_turn.assert_not_called()
        select.assert_not_called()
        drag.assert_not_called()
        accuracy.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
