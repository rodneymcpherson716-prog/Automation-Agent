"""Actual 8I/130 tee HUD regression; no input."""
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import execution_engine as live

PROJECT=Path(r'Automation-Agent project root')

class IronReadingTests(unittest.TestCase):
    def test_actual_nine_iron_range_is_verified_during_tee_club_selection(self):
        live.setup()
        f=cv2.imread(str(PROJECT/'automation_debug/nine_iron_fixture_20261111.png'))
        self.assertIsNotNone(f)
        h,w=f.shape[:2]
        rect=dict(left=37,top=46,width=w,height=h)
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            r=live.inspect_screen(f,rect)
            problems=live.readiness_problems(r['state'],f,rect,expected=dict(hole=10,shot=1))
        self.assertEqual(r['state']['club_name'],'9I',r)
        self.assertEqual(r['state']['club_range'],115,r)
        self.assertEqual(r['state']['club_unit'],'yd',r)
        self.assertEqual(r['state']['distance'],282,r)
        self.assertTrue(r['verified'],r)
        self.assertEqual(problems,[])
        send.assert_not_called()

    def test_actual_eight_iron_is_verified_before_club_selection(self):
        live.setup()
        path=Path(__file__).with_name('iron_tee.png')
        if not path.is_file():
            path=PROJECT/'automation_debug/iron_tee_fixture_20261110.png'
        f=cv2.imread(str(path))
        self.assertIsNotNone(f)
        h,w=f.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            r=live.inspect_screen(f,dict(left=37,top=46,width=w,height=h))
        s=r['state']
        self.assertEqual(s['club_name'],'8I',s)
        self.assertEqual(s['club_range'],130,s)
        self.assertEqual(s['distance'],282,s)
        self.assertEqual(s['hole_distance'],527,s)
        self.assertEqual(s['shot'],1,s)
        self.assertTrue(r['verified'],r)
        send.assert_not_called()

    def test_actual_seven_iron_is_read_while_keyboard_blocks_swing(self):
        live.setup()
        path=Path(__file__).with_name('seven_iron.png')
        if not path.is_file():
            path=PROJECT/'automation_debug/seven_iron_fixture_20261110.png'
        f=cv2.imread(str(path))
        self.assertIsNotNone(f)
        h,w=f.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'), \
                patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            r=live.inspect_screen(f,dict(left=37,top=46,width=w,height=h))
        self.assertEqual(r['state']['club_name'],'7I',r)
        self.assertEqual(r['state']['club_range'],143,r)
        self.assertFalse(r['checks']['swing_button'])
        self.assertFalse(r['verified'])
        send.assert_not_called()

if __name__=='__main__':
    unittest.main()
