"""Verify saved MSS captures and save annotated evidence. Never send input."""
import json
from pathlib import Path

import cv2
import execution_engine as wgt
from vision_engine import Session


def main():
    wgt.setup()
    root=Path(__file__).resolve().parent
    session=Session(root)
    def no_input(*args,**kwargs):
        raise RuntimeError('Input is forbidden in saved-screen verification')
    wgt.send=no_input
    results={}
    try:
        cases={'tee':'detection_fixture_20261110.png',
               'approach':'approach_detection_fixture_20261110.png',
               'occluded_pitch':'pitch_detection_fixture_20261110.png'}
        for name,filename in cases.items():
            path=root/'automation_debug'/filename
            frame=cv2.imread(str(path))
            if frame is None:
                raise RuntimeError('Missing saved MSS fixture: '+str(path))
            h,w=frame.shape[:2]
            rect=dict(left=37,top=46,width=w,height=h)
            report=wgt.inspect_screen(frame,rect)
            report['source_kind']='saved MSS capture; no new live observation'
            report['source_file']=str(path)
            session.state(name,report)
            session.image(name+'_annotated',wgt.detection_overlay(frame,rect,report))
            state=report['state']
            results[name]=dict(distance=state['distance'],unit=state['unit'],mode=state['mode'],
                               club=state['club_name'],club_range=state['club_range'],
                               wind=state['wind'],checks=report['checks'],verified=report['verified'])
        correct=(results['tee']['verified'] and results['tee']['distance']==282
                 and results['approach']['verified'] and results['approach']['distance']==200
                 and results['occluded_pitch']['distance'] is None
                 and results['occluded_pitch']['club']=='PW' and results['occluded_pitch']['club_range']==32
                 and not results['occluded_pitch']['verified'])
        session.close('offline_screen_verification',input_sent=False,all_expected_results=correct,results=results)
        print(json.dumps(dict(path=str(session.path),all_expected_results=correct,input_sent=False,results=results),indent=2))
        return 0 if correct else 1
    except BaseException as exc:
        session.close('offline_screen_verification_failed',input_sent=False,reason=str(exc))
        raise


if __name__=='__main__':
    raise SystemExit(main())
