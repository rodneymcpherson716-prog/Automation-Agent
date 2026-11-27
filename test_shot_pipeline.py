"""Verify the decision cutoff and the power-to-accuracy handoff offline."""

from dataclasses import dataclass
import unittest
from unittest.mock import Mock

from agent.shot_pipeline import ShotTiming, build_plan, execute_stroke


@dataclass
class Meter:
    origin: int = 20
    zero: int = 100
    return_origin: int = 20


class ShotPipelineTests(unittest.TestCase):
    def test_decision_deadline_blocks_before_swing(self):
        clock = Mock(side_effect=[0, 60])
        timing = ShotTiming(clock)
        with self.assertRaisesRegex(RuntimeError, 'stopped before swing'):
            timing.before_swing()

    def test_observed_power_sets_the_accuracy_origin_once(self):
        clock = Mock(side_effect=range(7))
        timing = ShotTiming(clock)
        timing.before_swing()
        drag, click = Mock(return_value=.5), Mock()
        actual = execute_stroke(None, None, {}, None, Meter(), .6, drag, click, timing)
        self.assertEqual(actual, .5)
        click.assert_called_once_with(None, None, {}, Meter(return_origin=60), None)
        self.assertEqual(timing.summary()['stage_seconds'], {'set_power': 1, 'accuracy': 1, 'execution': 4})

    def test_rough_policy_keeps_the_calibrated_power(self):
        state = dict(lie='ROUGH', par=5, club_name='7I')
        model = Mock()
        model.plan.return_value = {'power': .87}
        plan = build_plan(state, model)
        self.assertEqual((plan['risk_level'], plan['objective'], plan['power']), ('LOW', 'RECOVERY', .87))
        self.assertIsNone(plan['expected_score'])

    def test_spin_advice_uses_the_approach_conditions(self):
        model = Mock()
        model.plan.return_value = {'power': .45}
        state = dict(lie='FAIRWAY',mode='PITCH',wind={'mean_mph': 10},target_hole_link=True)
        downhill = build_plan(dict(state,green_speed=10,elevation_ft=-2),model)
        uphill = build_plan(dict(state,green_speed=7,elevation_ft=2),model)
        unknown = build_plan(dict(state,green_speed=10),model)
        self.assertEqual(downhill['spin_advice']['direction'], 'BACKSPIN')
        self.assertEqual(uphill['spin_advice']['direction'], 'TOPSPIN')
        self.assertIsNone(downhill['spin'])
        self.assertIsNone(unknown['spin_advice'])
