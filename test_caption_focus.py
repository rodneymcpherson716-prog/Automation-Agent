"""Caption-only foreground fallback; no actual Windows input."""
import ctypes
import unittest
from unittest.mock import patch
import execution_engine as live

class CaptionFocusTests(unittest.TestCase):
    def titlebar(self,hwnd,pointer):
        info=ctypes.cast(pointer,ctypes.POINTER(live.TITLEBARINFO)).contents
        info.rcTitleBar=live.wt.RECT(40,50,1300,77)
        return True

    def test_verified_caption_click_stays_above_game_and_always_releases(self):
        bounds=dict(left=37,top=46,width=1288,height=728)
        with patch.object(live,'user32') as api,patch.object(live,'rect_of',return_value=bounds), \
                patch.object(live,'key_down',return_value=False),patch.object(live,'move_to') as move, \
                patch.object(live,'atomic_caption_click') as click,patch.object(live,'mouse_up') as up:
            api.GetTitleBarInfo.side_effect=self.titlebar
            api.WindowFromPoint.return_value=10
            live.click_wgt_titlebar(10)
            move.assert_called_once_with(160,63)
            click.assert_called_once_with(160,63)
            up.assert_called_once()

    def test_covered_caption_sends_no_input(self):
        with patch.object(live,'user32') as api,patch.object(live,'rect_of',return_value=dict(left=37,top=46,width=1288,height=728)), \
                patch.object(live,'key_down',return_value=False),patch.object(live,'send') as send,patch.object(live,'move_to') as move:
            api.GetTitleBarInfo.side_effect=self.titlebar
            api.WindowFromPoint.return_value=20
            with self.assertRaisesRegex(RuntimeError,'covers'):
                live.click_wgt_titlebar(10)
            send.assert_not_called()
            move.assert_not_called()

    def test_f9_prevents_caption_click(self):
        with patch.object(live,'key_down',return_value=True),patch.object(live,'send') as send:
            with self.assertRaises(KeyboardInterrupt):
                live.click_wgt_titlebar(10)
            send.assert_not_called()

    def test_f9_during_caption_click_still_releases(self):
        values=iter([False,False,False,True])
        with patch.object(live,'user32') as api,patch.object(live,'rect_of',return_value=dict(left=37,top=46,width=1288,height=728)), \
                patch.object(live,'key_down',side_effect=lambda key:next(values)),patch.object(live,'move_to'), \
                patch.object(live,'atomic_caption_click') as click,patch.object(live,'mouse_up') as up:
            api.GetTitleBarInfo.side_effect=self.titlebar
            api.WindowFromPoint.return_value=10
            with self.assertRaises(KeyboardInterrupt):
                live.click_wgt_titlebar(10)
            click.assert_called_once_with(160,63)
            up.assert_called_once()

    def test_normal_focus_request_still_has_no_caption_click(self):
        with patch.object(live,'user32') as api,patch.object(live,'key_down',return_value=False), \
                patch.object(live,'click_wgt_titlebar') as caption:
            api.IsIconic.return_value=False
            api.GetForegroundWindow.side_effect=[20,10]
            live.activate_wgt(10)
            caption.assert_not_called()

    def test_refused_focus_uses_caption_then_requires_verified_foreground(self):
        with patch.object(live,'user32') as api,patch.object(live,'key_down',return_value=False), \
                patch.object(live,'click_wgt_titlebar') as caption, \
                patch.object(live.time,'perf_counter',side_effect=[0,3,4,4.1]):
            api.IsIconic.return_value=False
            api.GetForegroundWindow.side_effect=[20,10]
            live.activate_wgt(10)
            caption.assert_called_once_with(10)

    def test_caption_press_and_release_are_one_absolute_batch(self):
        with patch.object(live,'user32') as api:
            api.GetSystemMetrics.side_effect=lambda n:{76:0,77:0,78:3840,79:2160}[n]
            api.SendInput.return_value=2
            live.atomic_caption_click(160,63)
            count,events,size=api.SendInput.call_args.args
            self.assertEqual(count,2)
            self.assertEqual(size,ctypes.sizeof(live.INPUT))
            self.assertEqual(events[0].data.mi.dwFlags,live.MOVE|live.ABSOLUTE|live.VIRTUAL|live.DOWN)
            self.assertEqual(events[1].data.mi.dwFlags,live.MOVE|live.ABSOLUTE|live.VIRTUAL|live.UP)
            self.assertEqual((events[0].data.mi.dx,events[0].data.mi.dy),(events[1].data.mi.dx,events[1].data.mi.dy))

if __name__=='__main__':
    unittest.main()
