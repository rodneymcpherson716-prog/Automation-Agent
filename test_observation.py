import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import vision_engine as obs
import execution_engine as live
from automation_core import detect_meter, meter_rect, yellow_power, green_front

FIXTURES=Path(r'Automation-Agent project root')
LIVE_FIXTURE=Path(__file__).with_name('live_detection_fixture.png')
if not LIVE_FIXTURE.is_file():
    LIVE_FIXTURE=FIXTURES/'wgt_debug/detection_fixture_20261110.png'
APPROACH_FIXTURE=Path(__file__).with_name('approach_detection_fixture.png')
if not APPROACH_FIXTURE.is_file():
    APPROACH_FIXTURE=FIXTURES/'wgt_debug/approach_detection_fixture_20261110.png'
PITCH_FIXTURE=Path(__file__).with_name('pitch_detection_fixture.png')
if not PITCH_FIXTURE.is_file():
    PITCH_FIXTURE=FIXTURES/'wgt_debug/pitch_detection_fixture_20261110.png'


class CurrentScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()

    def test_live_meter_full_power_limit_is_outline_not_club_digits(self):
        frame=cv2.imread(str(LIVE_FIXTURE))
        self.assertIsNotNone(frame)
        h,w=frame.shape[:2]
        rect=dict(left=37,top=46,width=w,height=h)
        r=meter_rect(rect)
        x,y=r['left']-rect['left'],r['top']-rect['top']
        band=frame[y:y+r['height'],x:x+r['width']]
        meter=detect_meter(band,767)
        self.assertIsNotNone(meter)
        self.assertEqual(meter.origin,73)
        self.assertEqual(meter.zero,535)
        self.assertEqual(meter.target,767)
        self.assertIsNone(yellow_power(band,meter))
        self.assertIsNone(green_front(band,meter))

    def test_session_close_keeps_authoritative_path_if_report_also_has_path(self):
        with tempfile.TemporaryDirectory() as folder:
            s=obs.Session(folder)
            expected=str(s.path)
            s.close('detection_probe_only',path='duplicate metadata path',input_sent=False)
            result=json.loads((Path(folder)/'wgt_debug/latest_run.json').read_text())
            self.assertEqual(result['path'],expected)
            self.assertEqual(result['status'],'detection_probe_only')

    def test_all_five_current_screen_detectors_and_coordinate_spaces(self):
        frame=cv2.imread(str(LIVE_FIXTURE))
        h,w=frame.shape[:2]
        rect=dict(left=37,top=46,width=w,height=h)
        with patch.object(live,'CALIBRATION',FIXTURES/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',FIXTURES/'swing_button.png'), \
                patch.object(live,'send') as send:
            reading=live.inspect_screen(frame,rect)
        self.assertTrue(reading['verified'],reading)
        self.assertEqual(reading['state']['distance'],282)
        self.assertEqual(reading['state']['club_name'],'3I')
        self.assertEqual(reading['state']['club_range'],187)
        self.assertEqual((reading['state']['wind']['low_mph'],reading['state']['wind']['high_mph']),(11,13))
        self.assertAlmostEqual(reading['swing_button_window'][0],1152,delta=3)
        self.assertAlmostEqual(reading['swing_button_window'][1],529,delta=3)
        self.assertEqual(reading['swing_button_desktop'][0],rect['left']+reading['swing_button_window'][0])
        self.assertEqual(reading['meter']['full_power_x_window'],227)
        self.assertEqual(reading['meter']['neutral_x_window'],689)
        self.assertEqual(reading['meter']['accuracy_x_window'],921)
        self.assertTrue(live.detections_agree(reading,reading))
        wrong=json.loads(json.dumps(reading))
        wrong['meter']['origin']=40
        self.assertFalse(live.detections_agree(reading,wrong))
        self.assertFalse(reading['input_sent'])
        send.assert_not_called()

    def test_club_like_neutral_digit_does_not_shift_gauge_outline(self):
        from test_wgt import band
        image=band()
        cv2.line(image,(40,3),(40,35),(200,200,200),3)
        meter=detect_meter(image)
        self.assertIsNotNone(meter)
        self.assertEqual(meter.origin,70)

    def test_missing_outline_abstains_despite_text_and_blue_zone(self):
        from test_wgt import band
        image=band()
        # Erase the horizontal boundaries and leave the blue target/ticks.
        image[5,:]=0
        image[32,:]=0
        self.assertIsNone(detect_meter(image))

    def test_translucent_approach_sign_and_wood_club_label(self):
        frame=cv2.imread(str(APPROACH_FIXTURE))
        self.assertIsNotNone(frame)
        with patch.object(live,'CALIBRATION',FIXTURES/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',FIXTURES/'swing_button.png'):
            h,w=frame.shape[:2]
            reading=live.inspect_screen(frame,dict(left=37,top=46,width=w,height=h))
        self.assertEqual(reading['state']['distance'],200,reading)
        self.assertEqual(reading['state']['unit'],'yd')
        self.assertEqual(reading['state']['club_name'],'3W',reading)
        self.assertEqual(reading['state']['club_range'],205)
        self.assertEqual(reading['state']['elevation_ft'],-3,reading)
        self.assertTrue(reading['verified'],reading)

    def test_second_shot_badge_recovers_56_yards_from_leading_zero_ocr(self):
        saved=FIXTURES/'wgt_debug/runs/20261122_035206_b02b5cf1/stopped_frame.png'
        frame=cv2.imread(str(saved))
        self.assertIsNotNone(frame)
        sign=(573,42,107,80)  # Coordinates from the live misread.
        self.assertEqual(live.read_sign_distance(frame,sign),live.Distance(56.,'yd'))
        with patch.object(live,'read_segmented_distance',return_value=None):
            self.assertIsNone(live.read_sign_distance(frame,sign))

    def test_second_shot_badge_above_previous_search_boundary(self):
        saved=FIXTURES/'wgt_debug/runs/20261122_035806_aeadfc97/detection_read_1.png'
        frame=cv2.imread(str(saved))
        self.assertIsNotNone(frame)
        sign=live.find_distance_sign(frame)
        self.assertIsNotNone(sign)
        self.assertEqual(live.read_sign_distance(frame,sign),live.Distance(56.,'yd'))

    def test_pitch_badge_keeps_both_narrow_digits(self):
        run=FIXTURES/'wgt_debug/runs/20261122_220123_73ad5459'
        for name in ('hole_01_ready_wait_1.png','hole_01_stopped_frame.png'):
            with self.subTest(name=name):
                frame=cv2.imread(str(run/name))
                self.assertIsNotNone(frame)
                sign=live.find_distance_sign(frame)
                self.assertIsNotNone(sign)
                self.assertEqual(live.read_sign_distance(frame,sign),live.Distance(11.,'yd'))

    def test_unit_glyph_fallback_rejects_uniform_and_missing_labels(self):
        self.assertIsNone(obs.unit_from_glyphs(np.zeros((16,18,3),np.uint8),61))
        self.assertIsNone(obs.unit_from_glyphs(np.full((16,18,3),150,np.uint8),61))
        frame=cv2.imread(str(APPROACH_FIXTURE))
        self.assertEqual(obs.unit_from_glyphs(frame[182:198,752:770],61),'YDS')
        self.assertEqual(obs.unit_from_glyphs(frame[198:217,750:771],61),'FT')

    def test_occluded_pitch_target_is_not_replaced_with_hole_distance(self):
        frame=cv2.imread(str(PITCH_FIXTURE))
        with patch.object(live,'CALIBRATION',FIXTURES/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',FIXTURES/'swing_button.png'),patch.object(live,'send') as send:
            h,w=frame.shape[:2]
            reading=live.inspect_screen(frame,dict(left=37,top=46,width=w,height=h))
        state=reading['state']
        self.assertEqual(state['mode'],'PITCH',state)
        self.assertEqual(state['club_name'],'PW',state)
        self.assertEqual(state['club_range'],32,state)
        self.assertEqual(state['club_unit'],'yd',state)
        self.assertEqual(state['distance_source'],'floating_target')
        self.assertIsNone(state['distance'],state)
        self.assertFalse(reading['verified'])
        self.assertTrue(reading['checks']['club'])
        self.assertTrue(reading['checks']['swing_button'])
        self.assertTrue(reading['checks']['meter'])
        send.assert_not_called()


class ConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.setup()

    def test_wind_parser_requires_units_and_ordered_range(self):
        self.assertEqual(obs.parse_wind('11-13 MPH'),(11,13))
        self.assertEqual(obs.parse_wind('0 MPH'),(0,0))
        self.assertIsNone(obs.parse_wind('13-11 MPH'))
        self.assertIsNone(obs.parse_wind('11 13'))

    def test_saved_tee_conditions(self):
        frame=cv2.imread(str(FIXTURES/'wgt_debug/v48_before_shot_1.png'))
        state=obs.read_conditions(frame,live.find_distance_sign(frame),live.ocr_variants,live.crop)
        self.assertEqual(state['elevation_ft'],-10,state)
        self.assertIsNotNone(state['wind'],state)
        self.assertEqual((state['wind']['low_mph'],state['wind']['high_mph']),(11,13),state)
        self.assertTrue(10<state['wind']['heading_degrees']<80,state)
        self.assertEqual(state['club_name'],'3I',state)
        self.assertEqual(state['green_speed'],10,state)

    def test_saved_putt_conditions(self):
        frame=cv2.imread(str(FIXTURES/'wgt_debug/v47_before_shot_1.png'))
        state=obs.read_conditions(frame,live.find_distance_sign(frame),live.ocr_variants,live.crop)
        self.assertIsNotNone(state['elevation_ft'],state)
        self.assertAlmostEqual(state['elevation_ft'],-.05,msg=str(state))
        self.assertEqual((state['wind']['low_mph'],state['wind']['high_mph']),(12,14),state)
        self.assertEqual(state['green_speed'],10,state)

    def test_second_shot_elevation_requires_visible_ft_glyph(self):
        run=FIXTURES/'wgt_debug/runs/20261118_151019_e9af04a8'
        for name in ('hole_01_ready_wait_52.png','hole_01_ready_wait_99.png'):
            with self.subTest(frame=name):
                frame=cv2.imread(str(run/name))
                self.assertIsNotNone(frame)
                sign=live.find_distance_sign(frame)
                state=obs.read_conditions(frame,sign,live.ocr_variants,live.crop)
                self.assertEqual(state['elevation_ft'],13,state)
                self.assertEqual(state['conditions_raw']['elevation_unit_method'],
                                 'padded interior OCR')
        x,y,_,_=sign
        frame[y+52:y+66,x+75:x+90]=frame[y+47,x+83].copy()
        state=obs.read_conditions(frame,sign,live.ocr_variants,live.crop)
        self.assertIsNone(state['elevation_ft'],state)

    def test_live_hud_telemetry_keeps_ocr_and_parsed_measurements_together(self):
        frame=cv2.imread(str(FIXTURES/'wgt_debug/runs/20261118_151019_e9af04a8/hole_01_ready_wait_99.png'))
        self.assertIsNotNone(frame)
        with tempfile.TemporaryDirectory() as folder:
            session=obs.Session(folder)
            session.screen_id=1
            try:
                with patch.object(live,'DIAGNOSTICS',session),patch.object(live,'GAME',None):
                    live.shot_state(frame)
                    live.shot_state(frame)  # Cached OCR must retain the same raw evidence.
            finally:
                session.close('observed_only')
            events=[json.loads(line) for line in (session.path/'events.jsonl').read_text().splitlines()]
            expected=['observation_started','yardage_read','elevation_read','shot_telemetry']
            stages=[event['event'] for event in events if event['event'] in expected]
            self.assertEqual(stages,expected*2)
            console=(session.path/'console.log').read_text().splitlines()
            self.assertEqual(len(console),2)
            for line,event in zip(console,(e for e in events if e['event']=='shot_telemetry')):
                self.assertTrue(line.startswith('SHOT_TELEMETRY '))
                payload=json.loads(line.removeprefix('SHOT_TELEMETRY '))
                self.assertEqual((payload['hole'],payload['shot'],payload['screen_id']),(1,2,1))
                self.assertEqual((payload['parsed_yards'],payload['parsed_elevation_feet']),(154,13))
                self.assertIn('154',{text.strip() for text in payload['raw_yards_text']})
                self.assertIn('FT',{text.strip() for text in payload['raw_elevation_text']})
                self.assertEqual(payload['timestamp'],event['timestamp'])

    def test_dim_tee_badge_reads_from_lower_edge_and_requires_visible_badge(self):
        run=FIXTURES/'wgt_debug/runs/20261120_060836_41a532b5'
        for name in ('hole_01_ready_wait_1.png','hole_01_ready_wait_62.png'):
            with self.subTest(frame=name):
                frame=cv2.imread(str(run/name))
                self.assertIsNotNone(frame)
                sign=live.find_distance_sign(frame)
                self.assertEqual(sign,(826,95,62,46))
                distance=live.read_sign_distance(frame,sign)
                self.assertEqual((distance.value,distance.unit),(282,'yd'))
                conditions=obs.read_conditions(frame,sign,live.ocr_variants,live.crop,
                                               numeric_reader=live.read_segmented_distance)
                self.assertEqual(conditions['elevation_ft'],15)
        frame[92:144,824:889]=frame[92:144,748:813].copy()
        self.assertIsNone(live.find_distance_sign(frame))


class ModelTests(unittest.TestCase):
    def test_power_calibration_accepts_windows_utf8_bom_without_rewriting(self):
        data=dict(schema=1,observations=[])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'power_observations.json'
            path.write_text(json.dumps(data),encoding='utf-8-sig')
            original=path.read_bytes()
            loaded=obs.PowerCalibration.load(path)
            self.assertEqual(loaded.records,[])
            self.assertEqual(path.read_bytes(),original)

    def state(self,**updates):
        s=dict(distance=50,unit='yd',club_range=100,club_unit='yd',putt=False,mode='FULL',
               club_name='PW',elevation_ft=0,wind=dict(mean_mph=0,screen_dx=0,screen_dy=0),green_speed=10)
        return dict(s,**updates)

    def test_missing_live_conditions_block_power(self):
        with self.assertRaises(ValueError):
            obs.PowerCalibration().plan(self.state(wind=None))
        with self.assertRaises(ValueError):
            obs.PowerCalibration().plan(self.state(elevation_ft=None))

    def test_condition_adjustments_and_legacy_rows_not_fitted(self):
        model=obs.PowerCalibration(legacy_count=7)
        flat=model.plan(self.state())
        uphill=model.plan(self.state(elevation_ft=9))
        head=model.plan(self.state(wind=dict(mean_mph=10,screen_dx=0,screen_dy=1)))
        self.assertEqual(flat['power'],.5)
        self.assertGreater(uphill['power'],flat['power'])
        self.assertGreater(head['power'],flat['power'])
        self.assertEqual(flat['legacy_rows_not_fitted'],7)
        self.assertIn('baseline',flat['model'])

    def test_contextual_calibration_only_matching_verified_independent_samples(self):
        s=self.state()
        points=[dict(shot_id=str(i),verified=True,mode='FULL',unit='yd',club_name='PW',club_range=100,
                     elevation_ft=0,wind_mph=0,power=p,travel=d) for i,(p,d) in enumerate(((.3,25),(.7,65)))]
        model=obs.PowerCalibration(points)
        self.assertAlmostEqual(model.plan(s)['power'],.55)
        self.assertIn('empirical',model.plan(s)['model'])
        self.assertIsNone(model.curve(self.state(club_name='SW')))
        self.assertIsNone(obs.PowerCalibration([points[0],points[0]]).curve(s))
        self.assertIsNone(obs.PowerCalibration([dict(p,verified=False) for p in points]).curve(s))

    def test_putt_uses_feet_inches_and_green_speed(self):
        state=self.state(distance=5.9,unit='ft',club_range=30,club_unit='ft',putt=True,mode='PUTT',
                         club_name='PUTTER',elevation_ft=-.05)
        plan=obs.PowerCalibration().plan(state)
        self.assertAlmostEqual(plan['effective_distance'],5.3)
        self.assertAlmostEqual(plan['power'],5.3/30)
        with self.assertRaises(ValueError):
            obs.PowerCalibration().plan(dict(state,green_speed=None))

    def test_only_fresh_contextual_stats_are_persisted_and_reloaded(self):
        model=obs.PowerCalibration()
        stats=dict(carry_yd=25.,roll_yd=2.,total_yd=27.,power_pct=30.,spin_pct=0.,apex_ft=10.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'power_observations.json'
            self.assertTrue(model.record(path,self.state(),stats,.3,'session:1'))
            self.assertFalse(model.record(path,self.state(),stats,.3,'session:1'))
            loaded=obs.PowerCalibration.load(path)
            self.assertEqual(len(loaded.records),1)
            self.assertEqual(loaded.records[0]['travel'],25)


class EvidenceTests(unittest.TestCase):
    def test_ocr_timeout_abstains_and_does_not_cache_failure(self):
        live.OCR_CACHE.clear()
        with patch.object(live.pytesseract,'image_to_string',side_effect=[RuntimeError('Tesseract process timeout'),'282 YDS']):
            image=np.ones((5,5),np.uint8)
            self.assertEqual(live.ocr_text(image,'--psm 7'),'')
            self.assertEqual(live.ocr_text(image,'--psm 7'),'282 YDS')
        live.OCR_CACHE.clear()

    def test_club_selection_learns_scroll_direction_and_verifies_maximum_range(self):
        from unittest.mock import Mock
        s=dict(distance=282,club_range=187,mode='FULL',unit='yd',putt=False)
        values=[dict(s,club_range=175),dict(s,club_range=187),dict(s,club_range=220),dict(s,club_range=277),dict(s,club_range=277)]
        with patch.object(live,'move_to'),patch.object(live,'send') as send, \
                patch.object(live,'stable_state',side_effect=[(v,None) for v in values]):
            result,_=live.select_club(Mock(),None,dict(left=0,top=0,width=1288,height=728),s,None)
            self.assertEqual(result['club_range'],277)
            self.assertEqual(send.call_count,5)

    def test_stationary_scene_has_no_ball_or_scene_motion(self):
        frame=np.zeros((728,1288,3),np.uint8)
        evidence=obs.MotionEvidence(frame)
        for _ in range(8):
            evidence.update(frame)
        self.assertFalse(evidence.report()['scene_motion_observed'])
        self.assertFalse(evidence.report()['moving_ball_candidate_tracked'])

    def test_moving_small_ball_is_distinguished_from_stationary_white_pixels(self):
        frame=np.zeros((728,1288,3),np.uint8)
        evidence=obs.MotionEvidence(frame)
        for i in range(8):
            current=frame.copy()
            cv2.circle(current,(600+i*12,500-i*8),4,(255,255,255),-1)
            evidence.update(current)
        self.assertTrue(evidence.report()['moving_ball_candidate_tracked'])

    def test_artifacts_have_unique_session_and_explicit_unverified_status(self):
        with tempfile.TemporaryDirectory() as folder:
            first=obs.Session(folder)
            second=obs.Session(folder)
            self.assertNotEqual(first.path,second.path)
            first.close('stopped_unverified',reason='test')
            second.close('observed_only',input_sent=False)
            result=json.loads((Path(folder)/'wgt_debug/latest_run.json').read_text())
            self.assertEqual(result['status'],'observed_only')
            self.assertFalse(result['input_sent'])

    def test_capture_failure_still_saves_stop_result_without_input(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(live,'ROOT',Path(folder)), \
                patch.object(live,'RECORD',FIXTURES/'wgt_calibration/last_manual_stroke.json'), \
                patch.object(live,'CALIBRATION',FIXTURES/'automation_accuracy.json'), \
                patch.object(live,'setup'),patch.object(live.pytesseract,'get_tesseract_version',return_value='5'), \
                patch.object(live,'find_wgt',return_value=10), \
                patch.object(live,'rect_of',return_value=dict(left=0,top=0,width=1288,height=728)), \
                patch.object(live,'capture',side_effect=OSError('MSS capture failed')), \
                patch.object(live,'send') as send,redirect_stdout(io.StringIO()):
            self.assertEqual(live.main(['--observe']),1)
            result=json.loads((Path(folder)/'wgt_debug/latest_run.json').read_text())
            self.assertEqual(result['status'],'stopped_unverified')
            self.assertIn('MSS capture failed',result['reason'])
            send.assert_not_called()

    def test_mss_observer_owns_its_capture_context_and_stops(self):
        import time
        baseline=np.zeros((728,1288,3),np.uint8)
        observer=live.ShotObserver({},baseline)
        with patch.object(live,'capture',return_value=baseline) as capture:
            observer.start()
            observer.arm()
            deadline=time.perf_counter()+.6
            while observer.report()['samples']<2 and time.perf_counter()<deadline:
                time.sleep(.01)
            observer.close()
            self.assertGreaterEqual(capture.call_count,2)
            self.assertFalse(observer.thread.is_alive())

    def test_hole_controller_arms_video_only_after_accuracy_input(self):
        from unittest.mock import Mock
        state=dict(hole=1,shot=2,distance=154,unit='yd',elevation_ft=13,mode='FULL',putt=False,
                   club_name='8I',club_range=170,club_unit='yd')
        frame=np.zeros((728,1288,3),np.uint8)
        from automation_core import Meter
        meter=Meter(73,598,932,767,1056,535)
        order=[]
        observer=Mock()
        observer.start.side_effect=lambda:order.append('start')
        observer.arm.side_effect=lambda:order.append('arm')
        diagnostics=Mock()
        power_model=Mock()
        power_model.plan.return_value=dict(power=1.)
        with patch.object(live,'park_on_meter'),patch.object(live,'GAME') as game,patch.object(live,'DIAGNOSTICS',diagnostics),patch.object(live,'POWER_MODEL',power_model), \
                patch.object(live,'ShotObserver',return_value=observer), \
                patch.object(live,'stable_state',return_value=(state,frame)), \
                patch.object(live,'select_club',return_value=(state,frame)), \
                patch.object(live,'shot_stats',return_value=None), \
                patch.object(live,'ready_meter',return_value=meter), \
                patch.object(live,'drag_swing',side_effect=lambda *args:order.append('drag') or .9) as drag, \
                patch.object(live,'accuracy_click',side_effect=lambda *args:order.append('accuracy') or True), \
                patch.object(live,'wait_until_next',return_value=('complete',None)), \
                patch.object(live,'DEBUG',Path(__file__).parent),patch.object(Path,'write_text'), \
                patch.object(cv2,'imwrite'),redirect_stdout(io.StringIO()):
            game.completed=False
            self.assertEqual(live.run_hole(Mock(),None,{},Mock()),'complete')
            self.assertEqual(order,['start','drag','accuracy','arm'])
            observer.close.assert_called_once()
            stages={call.args[0]:call.kwargs for call in diagnostics.shot_stage.call_args_list}
            self.assertEqual(list(stages),['observation_validation_passed','club_selected',
                                          'shot_parameters_calculated','swing_initiated',
                                          'post_shot_state_detected'])
            for name in ('shot_parameters_calculated','swing_initiated'):
                self.assertEqual((stages[name]['yards'],stages[name]['elevation_feet'],
                                  stages[name]['club'],stages[name]['power_pct']),(154,13,'8I',100.))
            self.assertEqual(stages['swing_initiated']['label'],'SHOT_EXECUTION')
            self.assertTrue(stages['swing_initiated']['measurements_match_plan'])

            def change_elevation(safety,sct,rect,current,reference):
                current['elevation_ft']=12
            game.arm_stroke.side_effect=change_elevation
            with self.assertRaisesRegex(RuntimeError,'Shot parameters changed after planning'):
                live.run_hole(Mock(),None,{},Mock())
            drag.assert_called_once()
            diagnostics.shot_stage.assert_called_with(
                'shot_parameters_changed',planned=live.shot_parameters(dict(state,elevation_ft=13)),
                before_swing=live.shot_parameters(state))


if __name__=='__main__':
    unittest.main(verbosity=2)
