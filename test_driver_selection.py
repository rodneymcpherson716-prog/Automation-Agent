"""Regression for the real 282-yard driver selection failure. No live input."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import execution_engine as live
from vision_engine import PowerCalibration

FIXTURE=Path(__file__).with_name('driver_selection_fixture.json')
if not FIXTURE.is_file():
    FIXTURE=Path(r'Automation-Agent project root\wgt_debug\driver_selection_fixture_20261110.json')

class DriverSelectionTests(unittest.TestCase):
    def test_actual_282_yard_driver_is_usable_without_wheel_input(self):
        s=json.loads(FIXTURE.read_text())['observed']
        model=PowerCalibration()
        self.assertGreater(model.plan(s,for_selection=True)['unclipped_power'],.92)
        with patch.object(live,'POWER_MODEL',model),patch.object(live,'send') as send, \
                patch.object(live,'move_to') as move,patch.object(live,'stable_state') as ready:
            selected,frame=live.select_club(Mock(),None,{},s,'saved frame')
        self.assertEqual(selected['club_name'],'9.0D')
        self.assertEqual(selected['club_range'],282)
        self.assertEqual(selected['distance'],282)
        self.assertEqual(frame,'saved frame')
        send.assert_not_called()
        move.assert_not_called()
        ready.assert_not_called()

    def test_driver_can_be_reached_from_an_iron_for_full_power_layup(self):
        driver=json.loads(FIXTURE.read_text())['observed']
        iron=dict(driver,club_name='3I',club_range=187)
        with patch.object(live,'POWER_MODEL',PowerCalibration()),patch.object(live,'move_to'), \
                patch.object(live,'send') as send,patch.object(live,'stable_state',return_value=(driver,None)), \
                patch.object(live,'diagnostic_state') as diagnostic:
            selected,_=live.select_club(Mock(),None,dict(left=0,top=0,width=1288,height=728),iron,None)
        self.assertEqual(selected['club_range'],282)
        self.assertEqual(send.call_count,1)
        self.assertIn('step_01',diagnostic.call_args.args[0])

    def test_short_driver_shot_still_selects_a_smaller_club(self):
        driver=json.loads(FIXTURE.read_text())['observed']
        s=dict(driver,distance=38)
        wedge=dict(s,club_name='LW',club_range=60)
        with patch.object(live,'POWER_MODEL',PowerCalibration()),patch.object(live,'move_to'), \
                patch.object(live,'send') as send,patch.object(live,'stable_state',return_value=(wedge,None)):
            selected,_=live.select_club(Mock(),None,dict(left=0,top=0,width=1288,height=728),s,None)
        self.assertEqual(selected['club_name'],'LW')
        self.assertEqual(send.call_count,1)

    def test_cyclic_non_driver_ranges_stop_before_repeating_input(self):
        base=dict(distance=300,club_name='3I',club_range=180,mode='FULL',putt=False,unit='yd',hole=10,shot=1)
        states=[dict(base,club_name=c,club_range=r) for c,r in [('2I',200),('3I',180),('2I',200),('3I',180)]]
        with patch.object(live,'POWER_MODEL',None),patch.object(live,'move_to'), \
                patch.object(live,'send') as send,patch.object(live,'stable_state',side_effect=[(s,None) for s in states]):
            with self.assertRaisesRegex(RuntimeError,'repeated a verified club'):
                live.select_club(Mock(),None,dict(left=0,top=0,width=1288,height=728),base,None)
        self.assertEqual(send.call_count,4)

if __name__=='__main__':
    unittest.main()
