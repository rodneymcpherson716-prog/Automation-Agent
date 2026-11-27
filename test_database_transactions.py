"""Verify service boundaries and atomic SQLite writes without starting WGT."""

from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from database.database import make_session_factory
from database.services import CaddieService, ShotService, StatsService
from database.session_scope import session_scope
from database.unit_of_work import UnitOfWork
from exceptions.database_exceptions import CreateError, DeleteError, RepositoryError, UpdateError


class ShotServiceTests(unittest.TestCase):
    def test_accepts_a_unit_of_work_without_sqlalchemy(self):
        shots, results = Mock(), Mock()
        shots.log_shot.return_value = SimpleNamespace(id=8)
        results.record_result.return_value = SimpleNamespace(shot_id=8)
        shots.as_dict.return_value = {'id': 8}
        results.as_dict.return_value = {'shot_id': 8}
        uow_factory = Mock(return_value=nullcontext(SimpleNamespace(shots=shots, results=results)))

        saved = ShotService(uow_factory=uow_factory).save_shot_result(
            {'session_id': 1, 'distance': 154}, {'surface': 'GREEN'},
        )

        uow_factory.assert_called_once_with()
        shots.log_shot.assert_called_once_with(session_id=1, distance=154)
        results.record_result.assert_called_once_with(shot_id=8, surface='GREEN')
        self.assertEqual(saved, ({'id': 8}, {'shot_id': 8}))


@unittest.skipUnless(find_spec('sqlalchemy'), 'Requires database/requirements.txt')
class DatabaseTransactionTests(unittest.TestCase):
    def setUp(self):
        from database.models import PlaySession

        folder = self.enterContext(TemporaryDirectory())
        self.sessions = make_session_factory(Path(folder) / 'test.db')
        self.engine = self.sessions.kw['bind']
        self.addCleanup(self.engine.dispose)
        with UnitOfWork(self.sessions) as uow:
            course = uow.courses.create_if_missing('Test Course')
            hole = uow.holes.create(course_id=course.id, hole_number=1, par=4)
            club = uow.clubs.create(name='8 Iron', category='IRON')
            play = PlaySession(session_uuid='test-round')
            uow.session.add(play)
            uow.session.flush()
            self.course_id = course.id
            self.shot_data = dict(session_id=play.id, hole_id=hole.id, club_id=club.id, distance=154)
        self.service = ShotService(self.sessions)

    def test_successful_services_share_the_committed_records(self):
        shot, result = self.service.save_shot_result(
            self.shot_data, {'surface': 'GREEN', 'carry': 147, 'distance_to_pin': 6},
        )
        self.assertEqual(result['shot_id'], shot['id'])
        self.assertEqual(self.service.recent_shots(), [shot])
        caddie = CaddieService(self.sessions)
        recommendation = caddie.save_recommendation(shot['id'], '8 Iron', confidence=.9)
        self.assertEqual(caddie.recommendations_for_shot(shot['id']), [recommendation])
        self.assertEqual(caddie.high_confidence_recommendations(), [recommendation])
        stats = StatsService(self.sessions)
        self.assertEqual(stats.average_distance_by_club(), [
            {'club': '8 Iron', 'average_carry_yards': 147.0, 'samples': 1},
        ])
        self.assertEqual(stats.clubs_by_proximity(150, 160), [
            {'club': '8 Iron', 'average_pin_feet': 6.0, 'samples': 1},
        ])
        self.assertEqual(self.engine.pool.checkedout(), 0)

    def test_failed_result_rolls_back_shot_and_logs_context(self):
        from sqlalchemy.exc import IntegrityError

        with self.assertLogs('database.services.shot_service', level='ERROR') as log:
            with self.assertRaises(CreateError) as failure:
                self.service.save_shot_result(self.shot_data, {'distance_to_pin': -1})

        self.assertIsInstance(failure.exception.__cause__, IntegrityError)
        record, = log.records
        self.assertEqual(record.getMessage(), 'Shot recording failed')
        self.assertEqual((record.hole, record.club, record.distance), (
            self.shot_data['hole_id'], self.shot_data['club_id'], 154,
        ))
        self.assertIs(record.exc_info[1], failure.exception)
        self.assertEqual(self.service.recent_shots(), [])
        self.assertEqual(self.engine.pool.checkedout(), 0)

    def test_foreign_keys_on_multiple_connections_and_orphan_rejection(self):
        from sqlalchemy.exc import IntegrityError

        with self.engine.connect() as first, self.engine.connect() as second:
            for connection in (first, second):
                self.assertEqual(connection.exec_driver_sql('PRAGMA foreign_keys').scalar_one(), 1)

        with self.assertRaises(CreateError) as failure, UnitOfWork(self.sessions) as uow:
            uow.results.record_result(shot_id=999999, surface='GREEN')
        self.assertIsInstance(failure.exception.__cause__, IntegrityError)
        with UnitOfWork(self.sessions) as uow:
            self.assertEqual(uow.results.get_all(), [])

    def test_update_and_delete_errors_keep_the_original_records(self):
        from sqlalchemy.exc import IntegrityError

        with self.assertRaises(UpdateError) as failure, UnitOfWork(self.sessions) as uow:
            uow.holes.update(self.shot_data['hole_id'], par=2)
        self.assertIsInstance(failure.exception.__cause__, IntegrityError)

        with self.assertRaises(DeleteError) as failure, UnitOfWork(self.sessions) as uow:
            uow.courses.delete(self.course_id)
        self.assertIsInstance(failure.exception.__cause__, IntegrityError)

        with UnitOfWork(self.sessions) as uow:
            self.assertEqual(uow.holes.get_by_id(self.shot_data['hole_id']).par, 4)
            self.assertEqual(uow.courses.get_by_id(self.course_id).name, 'Test Course')

    def test_non_database_exception_rolls_back_without_reclassification(self):
        uow = UnitOfWork(self.sessions)
        error = RuntimeError('vision detection failed')
        with self.assertRaises(RuntimeError) as failure, uow:
            uow.shots.log_shot(**self.shot_data)
            raise error
        self.assertIs(failure.exception, error)
        with uow:
            self.assertEqual(uow.shots.get_all(), [])

    def test_deferred_constraint_at_commit_is_translated_and_rolled_back(self):
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with self.assertRaises(RepositoryError) as failure:
            with session_scope(self.sessions) as session:
                session.execute(text('PRAGMA defer_foreign_keys = ON'))
                session.execute(text('INSERT INTO shot_results (shot_id) VALUES (999999)'))
        self.assertIs(type(failure.exception), RepositoryError)
        self.assertIsInstance(failure.exception.__cause__, IntegrityError)
        with UnitOfWork(self.sessions) as uow:
            self.assertEqual(uow.results.get_all(), [])
        self.assertEqual(self.engine.pool.checkedout(), 0)

    def test_manual_record_distinguishes_played_and_suggested_clubs(self):
        state = dict(distance=156, unit='yd', mode='FULL', hole=1, shot=2, lie='FAIRWAY',
                     wind=dict(mean_mph=12, low_mph=11, high_mph=13,
                               screen_dy=1, screen_dx=0, heading_degrees=180))
        recommendation = dict(club='7 Iron', power=None)
        shot, result = self.service.save_manual_shot(
            state, recommendation, dict(outcome='next', final_lie='GREEN', distance_to_pin=2, distance_unit='yd'),
            club_name='8 Iron', power=94, artifact='before.png',
        )
        self.assertEqual((shot['club_id'], shot['power_percent']), (self.shot_data['club_id'], 94))
        self.assertFalse(shot['recommended'])
        self.assertIsNone(result['landing_surface'])
        self.assertEqual(CaddieService(self.sessions).recommendations_for_shot(shot['id'])[0]['recommended_club'],
                         '7 Iron')
        self.service.save_shot_result(self.shot_data, dict(distance_to_pin=12, distance_unit='ft'))
        self.service.save_shot_result(self.shot_data, dict(outcome='next'))
        self.assertEqual(StatsService(self.sessions).average_distance_to_pin(),
                         {'average_pin_feet': 9.0, 'samples': 2})
