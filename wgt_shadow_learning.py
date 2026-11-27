"""Read-only WGT shadow learner: learns from verified results, never clicks or swings.

Save beside the WGT controller. Run: python wgt_shadow_learning.py --self-test
or: python wgt_shadow_learning.py --report

The live controller is NOT modified or activated by this module.
User hypothesis: 13 mph headwind with a slight-right component may benefit from
a small LEFT-of-DING meter click. This is UNVERIFIED and not applied live.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest


def finite(value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    if not math.isfinite(value):
        return False
    return ((minimum is None or value >= minimum)
            and (maximum is None or value <= maximum))


def verified_shot(row):
    """Existing PowerCalibration observation; unmeasured shots cannot train."""
    if not isinstance(row, dict):
        return None
    if row.get('verified') is not True or row.get('owner_verified') is not True:
        return None
    if not row.get('shot_id') or not row.get('club_name'):
        return None
    mode = row.get('mode')
    if mode not in ('FULL', 'CHIP', 'PITCH', 'PUTT'):
        return None
    if not all((finite(row.get('club_range'), 1, 600),
                finite(row.get('power'), .025, 1),
                finite(row.get('travel'), .01, 900),
                finite(row.get('wind_mph'), 0, 50),
                finite(row.get('elevation_ft'), -300, 300))):
        return None
    basis = 'total' if mode in ('CHIP', 'PITCH', 'PUTT') else 'carry'
    if row.get('travel_basis') != basis:
        return None
    if mode == 'PUTT' and not finite(row.get('green_speed'), 4, 20):
        return None
    if not isinstance(row.get('lie'), str) or not row['lie']:
        return None
    # Existing records contain lie name but not necessarily lie percentage.
    # Keep those records distinct; NEVER infer a percentage from the label.
    loss = row.get('lie_loss_percent')
    if loss is not None:
        if (not isinstance(loss, (list, tuple)) or len(loss) != 2
                or not all(finite(v, 0, 99) for v in loss) or loss[0] > loss[1]):
            return None
        loss = tuple(loss)
    return {'shot_id': str(row['shot_id']), 'club': row['club_name'],
            'range': float(row['club_range']), 'power': float(row['power']),
            'travel': float(row['travel']), 'wind': float(row['wind_mph']),
            'elevation': float(row['elevation_ft']), 'lie': row['lie'],
            'lie_loss': loss, 'mode': mode, 'basis': basis,
            'unit': row.get('unit'), 'green_speed': row.get('green_speed')}


def group_key(shot):
    # Different lies, ranges, measurement bases, and lie-loss bands cannot mix.
    return (shot['mode'], shot['club'], shot['range'], shot['lie'],
            shot['lie_loss'], shot['basis'], shot['unit'],
            shot['green_speed'] if shot['mode'] == 'PUTT' else None)


def train_validation_split(shots):
    """Deterministic, shot-ID based split keeps repeat evaluations consistent."""
    train, holdout = [], []
    for shot in shots:
        bucket = int(hashlib.sha256(shot['shot_id'].encode()).hexdigest()[:8], 16) % 5
        (holdout if bucket == 0 else train).append(shot)
    return train, holdout


def interpolate(shots, power, wind, elevation):
    """Strict same-condition interpolation. No extrapolation or wind guesses."""
    compatible = [s for s in shots if abs(s['wind'] - wind) <= 2
                  and abs(s['elevation'] - elevation) <= 3]
    if len(compatible) < 4:
        return None
    by_power = defaultdict(list)
    for shot in compatible:
        by_power[shot['power']].append(shot['travel'])
    points = sorted((p, statistics.median(ds)) for p, ds in by_power.items())
    if len(points) < 3 or points[-1][0] - points[0][0] < .12:
        return None
    if any(b[1] <= a[1] for a, b in zip(points, points[1:])):
        return None
    for (p0, d0), (p1, d1) in zip(points, points[1:]):
        if p0 <= power <= p1:
            return d0 + (power - p0) / (p1 - p0) * (d1 - d0)
    return None


def evaluate_group(group):
    train, holdout = train_validation_split(group)
    pairs = [(s, interpolate(train, s['power'], s['wind'], s['elevation']))
             for s in holdout]
    pairs = [(s, pred) for s, pred in pairs if pred is not None]
    if len(pairs) < 3:
        return {'status': 'NEEDS_MORE_VERIFIED_DATA', 'train': len(train),
                'holdout': len(holdout), 'evaluated': len(pairs)}
    # This baseline is ONLY a reporting comparator, not the existing controller.
    baseline = statistics.mean(abs(s['power'] * s['range'] - s['travel'])
                               for s, _ in pairs)
    learned = statistics.mean(abs(pred - s['travel']) for s, pred in pairs)
    improvement = (baseline - learned) / baseline if baseline > 0 else 0
    return {'status': 'SHADOW_PROMISING' if improvement >= .15 else 'SHADOW_NOT_BETTER',
            'train': len(train), 'holdout': len(holdout),
            'evaluated': len(pairs), 'shadow_mae': round(learned, 3),
            'simple_comparator_mae': round(baseline, 3)}


def load_calibration(path):
    if not path.is_file():
        return [], 'Calibration file not found'
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 1 or not isinstance(data.get('observations'), list):
        raise ValueError('Unexpected calibration schema')
    seen, shots = set(), []
    for row in data['observations']:
        shot = verified_shot(row)
        if shot and shot['shot_id'] not in seen:
            shots.append(shot)
            seen.add(shot['shot_id'])
    return shots, f"{len(data['observations'])} source records, {len(shots)} independently verified and usable"


def verified_wind_click(row):
    """Future meter-offset outcomes require actual measured lateral error."""
    if not isinstance(row, dict):
        return False
    if row.get('verified') is not True or row.get('owner_verified') is not True:
        return False
    if not row.get('shot_id') or row.get('actual_offset_px') is None:
        return False
    fields = {'wind_dx_mph': (-50, 50), 'wind_dy_mph': (-50, 50),
              'actual_offset_px': (-100, 100), 'lateral_miss_yd': (-200, 200)}
    return all(finite(row.get(k), *bounds) for k, bounds in fields.items())


def wind_report(path):
    if not path.is_file():
        return '0 verified meter-offset outcomes; CENTER CLICK UNCHANGED'
    accepted, seen = 0, set()
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if verified_wind_click(row) and row['shot_id'] not in seen:
            accepted += 1
            seen.add(row['shot_id'])
    return f'{accepted} verified meter-offset outcomes; CENTER CLICK UNCHANGED'


class OfflineTests(unittest.TestCase):
    def sample(self, **changes):
        row = dict(shot_id='unique', verified=True, owner_verified=True,
                   mode='FULL', club_name='LW', club_range=52, power=.3,
                   travel=15, wind_mph=2, elevation_ft=0, lie='BUNKER',
                   travel_basis='carry', unit='yd')
        row.update(changes)
        return row

    def test_reject_unverified(self):
        self.assertIsNone(verified_shot(self.sample(owner_verified=False)))
        self.assertIsNone(verified_shot(self.sample(verified=False)))

    def test_reject_missing_measurement(self):
        self.assertIsNone(verified_shot(self.sample(travel=None)))
        self.assertIsNone(verified_shot(self.sample(lie='')))

    def test_reject_unverified_bunker_percent(self):
        self.assertIsNone(verified_shot(self.sample(lie_loss_percent=[40, None])))
        self.assertIsNone(verified_shot(self.sample(lie_loss_percent=[60, 40])))

    def test_lie_percent_not_inferred(self):
        self.assertIsNone(verified_shot(self.sample())['lie_loss'])

    def test_condition_groups_separate(self):
        a = verified_shot(self.sample())
        b = verified_shot(self.sample(lie='FAIRWAY'))
        self.assertNotEqual(group_key(a), group_key(b))

    def test_interpolation(self):
        shots = [verified_shot(self.sample(shot_id=str(i), power=p, travel=d))
                 for i, (p, d) in enumerate(((.2, 9), (.3, 14), (.4, 19), (.5, 25)))]
        self.assertAlmostEqual(interpolate(shots, .35, 2, 0), 16.5)
        self.assertIsNone(interpolate(shots, .6, 2, 0))
        self.assertIsNone(interpolate(shots, .35, 15, 0))

    def test_shadow_evaluation_with_independent_holdouts(self):
        examples = [verified_shot(self.sample(shot_id='trial-'+str(i),
                     power=.1+i*.025, travel=5+i*1.9)) for i in range(31)]
        status = evaluate_group(examples)
        self.assertEqual(status['status'], 'SHADOW_PROMISING')
        self.assertGreaterEqual(status['evaluated'], 3)

    def test_no_wind_click_training_without_outcome(self):
        self.assertFalse(verified_wind_click({'verified': True,
                                              'owner_verified': True,
                                              'shot_id': 'a', 'actual_offset_px': -2}))

    def test_no_mutating_game_imports(self):
        import sys
        self.assertNotIn('pyautogui', sys.modules)
        self.assertNotIn('pydirectinput', sys.modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--report', action='store_true')
    opts = parser.parse_args()
    if opts.self_test:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineTests)
        result = unittest.TextTestRunner(verbosity=1).run(tests)
        if not result.wasSuccessful():
            raise SystemExit(1)
        print('OFFLINE PASS. NO GAME INPUT. NO SOURCE FILES MODIFIED.')
    elif opts.report:
        root = Path(__file__).resolve().parent
        shots, note = load_calibration(root / 'wgt_calibration' / 'power_observations.json')
        print('SHOT DATA:', note)
        groups = defaultdict(list)
        for shot in shots:
            groups[group_key(shot)].append(shot)
        if not groups:
            print('No validated groups yet; learner abstains.')
        for key, group in sorted(groups.items(), key=lambda item: -len(item[1])):
            print(f'{key[1]} {key[0]} {key[3]} ({len(group)} shots): {evaluate_group(group)}')
        print('WIND:', wind_report(root / 'wgt_calibration' / 'wind_click_verified.jsonl'))
        print('SHADOW MODE ONLY: no control or automatic wind offsets enabled.')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
