"""Store verified shot captures in SQLite and import existing run artifacts.

Usage: python -m shot_store
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from learning_data import number


DATABASE = Path('wgt_shot_logs') / 'shots.sqlite3'
CAPTURE_NAME = re.compile(r'(?:hole_\d+_)?shot_\d+_capture\.json')
SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY NOT NULL,
    run_id TEXT NOT NULL,
    artifact TEXT NOT NULL,
    hole INTEGER NOT NULL CHECK (hole BETWEEN 1 AND 18),
    stroke INTEGER NOT NULL CHECK (stroke > 0),
    outcome TEXT NOT NULL CHECK (outcome IN ('next', 'complete')),
    mode TEXT,
    club TEXT,
    target_distance REAL,
    target_unit TEXT CHECK (target_unit IN ('yd', 'ft')),
    elevation_ft REAL,
    wind_low_mph REAL,
    wind_high_mph REAL,
    wind_screen_dx REAL,
    wind_screen_dy REAL,
    requested_power_fraction REAL,
    observed_power_fraction REAL,
    capture_json TEXT NOT NULL
)
"""
INSERT = """
INSERT INTO shots (
    shot_id, run_id, artifact, hole, stroke, outcome, mode, club,
    target_distance, target_unit, elevation_ft,
    wind_low_mph, wind_high_mph, wind_screen_dx, wind_screen_dy,
    requested_power_fraction, observed_power_fraction, capture_json
) VALUES (
    :shot_id, :run_id, :artifact, :hole, :stroke, :outcome, :mode, :club,
    :target_distance, :target_unit, :elevation_ft,
    :wind_low_mph, :wind_high_mph, :wind_screen_dx, :wind_screen_dy,
    :requested_power_fraction, :observed_power_fraction, :capture_json
) ON CONFLICT(shot_id) DO NOTHING
"""


def _insert(connection, artifact, capture):
    if not isinstance(capture, dict):
        raise ValueError('Capture must be a JSON object')
    run_id = artifact.parent.name
    shot_id, outcome = capture.get('shot_id'), capture.get('outcome')
    before, checks = capture.get('before'), capture.get('verification')
    if (capture.get('schema') != 1 or capture.get('kind') != 'new_shot_capture'
            or not isinstance(shot_id, str) or not shot_id.startswith(run_id + ':')
            or not shot_id[len(run_id) + 1:] or outcome not in ('next', 'complete')
            or not isinstance(before, dict) or not isinstance(checks, dict)
            or checks.get('before_tigerwould52_turn') is not True):
        raise ValueError('Capture lacks a verified Tigerwould52 shot and outcome')
    hole, stroke = before.get('hole'), before.get('shot')
    if type(hole) is not int or not 1 <= hole <= 18 or type(stroke) is not int or stroke < 1:
        raise ValueError('Capture has an invalid hole or stroke number')

    unit = before.get('unit')
    distance = before.get('distance')
    if checks.get('before_distance') is not True or unit not in ('yd', 'ft') or not number(distance):
        distance, unit = None, None
    elevation = before.get('elevation_ft') if checks.get('before_elevation') is True else None
    wind = before.get('wind') if checks.get('before_wind') is True else None
    wind = wind if isinstance(wind, dict) else {}
    swing = capture.get('swing')
    swing = swing if isinstance(swing, dict) else {}
    requested = swing.get('requested_power_fraction')
    observed = swing.get('observed_meter_power_fraction')
    row = dict(
        shot_id=shot_id, run_id=run_id, artifact=f'{run_id}/{artifact.name}',
        hole=hole, stroke=stroke, outcome=outcome,
        mode=before.get('mode'), club=before.get('club_name'),
        target_distance=distance, target_unit=unit, elevation_ft=elevation,
        wind_low_mph=wind.get('low_mph'), wind_high_mph=wind.get('high_mph'),
        wind_screen_dx=wind.get('screen_dx'), wind_screen_dy=wind.get('screen_dy'),
        requested_power_fraction=requested if number(requested, 0, 1) else None,
        observed_power_fraction=observed if checks.get('actual_meter_power') is True else None,
        capture_json=json.dumps(capture, default=str, allow_nan=False),
    )
    return connection.execute(INSERT, row).rowcount == 1


def _connect(database):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(database, timeout=.25)


def _prepare(connection):
    connection.execute(SCHEMA)
    connection.execute('CREATE INDEX IF NOT EXISTS shots_run_hole ON shots (run_id, hole, stroke)')


def record_capture(database, artifact, capture):
    """Store one verified shot after the controller writes its JSON artifact."""
    with closing(_connect(database)) as connection, connection:
        _prepare(connection)
        return _insert(connection, Path(artifact), capture)


def import_captures(database, runs):
    """Index existing JSON without rewriting any run files."""
    runs = Path(runs)
    if not runs.is_dir():
        raise FileNotFoundError(f'Run directory does not exist: {runs}')
    counts = dict(inserted=0, already_present=0, skipped=0)
    with closing(_connect(database)) as connection, connection:
        _prepare(connection)
        for artifact in sorted(runs.glob('*/*shot_*_capture.json')):
            if not CAPTURE_NAME.fullmatch(artifact.name):
                continue
            try:
                capture = json.loads(artifact.read_text(encoding='utf-8-sig'))
                inserted = _insert(connection, artifact, capture)
            except (OSError, TypeError, ValueError, sqlite3.InterfaceError, sqlite3.ProgrammingError):
                counts['skipped'] += 1
            else:
                counts['inserted' if inserted else 'already_present'] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DATABASE)
    parser.add_argument('--runs', type=Path, default=Path('automation_debug/runs'))
    args = parser.parse_args()
    try:
        print(json.dumps(import_captures(args.database, args.runs), indent=2))
    except (OSError, sqlite3.Error) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
