"""Replay the real club-hover obstruction; all native input is mocked."""
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import cv2
import execution_engine as live
from automation_core import meter_rect
from vision_engine import PowerCalibration

ROOT=Path(r'Automation-Agent project root')
RUN=ROOT/'wgt_debug/runs/20261112_015125_4dcc7340'
RECT=dict(left=37,top=46,width=1288,height=728)

class ClubHoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):live.setup()

    def test_actual_open_club_menu_hides_swing_then_closed_frame_is_ready(self):
        opened=cv2.imread(str(RUN/'context_00021.png'))
        closed=cv2.imread(str(RUN/'stopped_frame.png'))
        with patch.object(live,'CALIBRATION',ROOT/'automation_accuracy.json'),patch.object(live,'SWING_TEMPLATE',ROOT/'swing_button.png'),patch.object(live,'send') as send:
            self.assertIsNone(live.swing_button(opened,RECT))
            self.assertIsNotNone(live.swing_button(closed,RECT))
            self.assertIsNotNone(live.ready_meter(opened,RECT))
            self.assertIsNotNone(live.ready_meter(closed,RECT))
        send.assert_not_called()

    def test_pointer_parks_inside_meter_without_buttons_or_wheel(self):
        with patch.object(live,'key_down',return_value=False),patch.object(live,'move_to') as move,patch.object(live,'send') as send:
            live.park_on_meter(Mock(),RECT)
        x,y=move.call_args.args
        meter=meter_rect(RECT)
        self.assertTrue(meter['left']<x<meter['left']+meter['width'])
        self.assertTrue(meter['top']<y<meter['top']+meter['height'])
        send.assert_not_called()

    def test_held_left_button_never_moves_pointer(self):
        with patch.object(live,'key_down',return_value=True),patch.object(live,'move_to') as move:
            with self.assertRaisesRegex(RuntimeError,'Left button held'):
                live.park_on_meter(Mock(),RECT)
        move.assert_not_called()

    def test_f9_or_focus_abort_prevents_pointer_move(self):
        for error in (KeyboardInterrupt('F9'),RuntimeError('WGT lost focus')):
            with self.subTest(error=str(error)):
                safety=Mock();safety.check.side_effect=error
                with patch.object(live,'move_to') as move:
                    with self.assertRaises(type(error)):live.park_on_meter(safety,RECT)
                move.assert_not_called()

    def test_scroll_leaves_club_hover_before_waiting_for_ready_hud(self):
        before=dict(distance=282,unit='yd',mode='FULL',putt=False,hole=10,shot=1,
            club_name='3W',club_range=205,club_unit='yd',elevation_ft=-3,
            wind=dict(mean_mph=12,screen_dx=1,screen_dy=0))
        after=dict(before,club_name='9.0D',club_range=282)
        history=[];hover=[False]
        def move(x,y):
            hover[0]=x<RECT['left']+RECT['width']*.2
            history.append('hover_club' if hover[0] else 'park_meter')
        def send(flags,**kwargs):
            self.assertEqual(flags,live.WHEEL);self.assertTrue(hover[0])
            history.append('wheel')
        def ready(*args,**kwargs):
            self.assertFalse(hover[0],'CLUB SELECTION still hides Swing')
            history.append('verify');return after,None
        with patch.object(live,'GAME',None),patch.object(live,'POWER_MODEL',PowerCalibration()),patch.object(live,'DIAGNOSTICS',None),patch.object(live,'key_down',return_value=False),patch.object(live,'move_to',side_effect=move),patch.object(live,'send',side_effect=send),patch.object(live,'stable_state',side_effect=ready):
            selected,_=live.select_club(Mock(),None,RECT,before,None)
        self.assertEqual(selected['club_name'],'9.0D')
        self.assertEqual(history,['hover_club','wheel','park_meter','verify'])

if __name__=='__main__':unittest.main()
