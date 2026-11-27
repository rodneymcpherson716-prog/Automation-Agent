"""Verify the manual caddie's yardage rules and CLI boundaries."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from caddie.recommendation_engine import recommend_club
from main import main


class RecommendationTests(unittest.TestCase):
    def test_signed_wind_can_change_the_selected_club(self):
        setup = dict(distance=165, unit='yd', mode='FULL')
        for direction, expected in ((1, '6 Iron'), (-1, '7 Iron'), (0, '7 Iron')):
            with self.subTest(direction=direction):
                wind = dict(mean_mph=12, screen_dy=direction)
                self.assertEqual(recommend_club(dict(setup, wind=wind))['club'], expected)

    def test_tee_power_requires_a_first_full_swing_from_the_tee(self):
        setup = dict(distance=282, unit='yd', mode='FULL', wind=dict(mean_mph=9, screen_dy=1))
        tee = recommend_club(dict(setup, shot=1, lie='TEE'))
        self.assertEqual((tee['club'], tee['power']), ('Driver', 100.0))
        self.assertIsNone(recommend_club(dict(setup, shot=2, lie='FAIRWAY'))['power'])

    def test_unreadable_wind_unsupported_lie_and_short_game_abstain(self):
        setup = dict(distance=156, unit='yd', mode='FULL', wind=dict(mean_mph=12, screen_dy=1))
        for changes in (dict(wind=None), dict(wind=dict(mean_mph=12, screen_dy=None)),
                        dict(lie='BUNKER'), dict(mode='PUTT'), dict(distance=90)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recommend_club(dict(setup, **changes))


class ManualCaddieCliTests(unittest.TestCase):
    def test_default_sample_needs_neither_screen_nor_database(self):
        output = io.StringIO()
        with redirect_stdout(output), patch('main.capture_window') as capture, \
                patch('main.make_session_factory') as session_factory:
            self.assertEqual(main([]), 0)
        self.assertIn('7 Iron (159.6 adjusted yd)', output.getvalue())
        capture.assert_not_called()
        session_factory.assert_not_called()

    def test_declining_to_record_never_opens_sqlite(self):
        state = dict(distance=156, unit='yd', mode='FULL', club_name='7 Iron',
                     wind=dict(mean_mph=12, screen_dy=1))
        with redirect_stdout(io.StringIO()), patch('main.load_image') as load, \
                patch('main.read_shot', return_value=state), patch('main.make_session_factory') as session_factory, \
                patch('main.find_spec', return_value=True), patch('builtins.input', return_value='n'):
            self.assertEqual(main(['--image', 'before.png', '--record']), 0)
        load.assert_called_once_with(Path('before.png'))
        session_factory.assert_not_called()
