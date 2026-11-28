from pathlib import Path
import unittest
from unittest.mock import patch
import cv2,numpy as np
import execution_engine as live

PROJECT=Path(r'Automation-Agent project root')

class PitchElevenTests(unittest.TestCase):
    def test_visible_eleven_yards_and_partly_overlapped_swing_circle(self):
        live.setup()
        path=Path(__file__).with_name('pitch_eleven.png')
        if not path.is_file():
            path=PROJECT/'automation_debug/pitch_eleven_fixture_20261110.png'
        f=cv2.imread(str(path));self.assertIsNotNone(f)
        h,w=f.shape[:2]
        with patch.object(live,'CALIBRATION',PROJECT/'automation_accuracy.json'),patch.object(live,'SWING_TEMPLATE',PROJECT/'swing_button.png'),patch.object(live,'send') as send:
            r=live.inspect_screen(f,dict(left=29,top=48,width=w,height=h))
            problems=live.readiness_problems(r['state'],f,r['rect'],expected=dict(hole=10,shot=5))
        s=r['state']
        self.assertEqual(s['distance'],11,s)
        self.assertEqual(s['unit'],'yd',s)
        self.assertEqual(s['mode'],'PITCH',s)
        self.assertEqual(s['shot'],5,s)
        self.assertEqual(s['club_name'],'SW',s)
        self.assertEqual(s['club_range'],23,s)
        self.assertAlmostEqual(s['elevation_ft'],19/12,msg=str(s))
        self.assertTrue(r['verified'],r)
        self.assertEqual(problems,[])
        send.assert_not_called()

    def test_segmentation_never_drops_decimal_dot_or_reads_blank(self):
        blank=np.zeros((45,100,3),np.uint8)
        self.assertIsNone(live.read_segmented_distance(blank))
        decimal=blank.copy()
        cv2.rectangle(decimal,(20,5),(26,35),(255,255,255),-1)
        cv2.rectangle(decimal,(60,5),(66,35),(255,255,255),-1)
        cv2.circle(decimal,(44,33),2,(255,255,255),-1)
        self.assertIsNone(live.read_segmented_distance(decimal))

    def test_uncertain_short_apex_is_unknown_not_truncated_37(self):
        live.setup()
        path=Path(__file__).with_name('pitch_eleven.png')
        if not path.is_file():
            path=PROJECT/'automation_debug/pitch_eleven_fixture_20261110.png'
        stats=live.shot_stats(cv2.imread(str(path)))
        self.assertIsNotNone(stats)
        self.assertEqual(stats['carry_yd'],.83)
        self.assertEqual(stats['roll_yd'],0)
        self.assertIsNone(stats['apex_ft'])

if __name__=='__main__':
    unittest.main()
