#!/usr/bin/env python3
"""Whole-video AP/FPS via the running web service, with a reproducible request.

Single GPU job at a time, every input frame saved; frozen-subset exploration is
kept separate. A visible browser may simultaneously display the normal preview.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def get(url):
    with urllib.request.urlopen(url,timeout=10) as f:return json.load(f)


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8765')
    p.add_argument('--videos',nargs='+',type=Path,required=True)
    p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',default='current');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);records=[];boot=get(a.url+'/api/bootstrap')
    for video in a.videos:
        for profile in a.profiles:
            data=dict(source=str(video.resolve()),output=str(a.output.resolve()),mode='coco',profile=profile)
            request=urllib.request.Request(a.url+'/api/start',data=json.dumps(data).encode(),
                    headers={'Content-Type':'application/json','X-Studio-Token':boot['token']})
            with urllib.request.urlopen(request,timeout=10) as f:start=json.load(f)
            deadline=time.monotonic()+600;last=0
            while time.monotonic()<deadline:
                state=get(a.url+'/api/status')
                if state['id']!=start['id']:raise RuntimeError('Another job replaced this measurement')
                if state['status'] in {'completed','failed','cancelled'}:break
                if time.monotonic()-last>15:print(a.label,video.name,profile,state['status'],state['frame'],flush=True);last=time.monotonic()
                time.sleep(.3)
            if state['status']!='completed':raise RuntimeError(str(state))
            folder=Path(state['output_dir']);report=json.loads((folder/'pipeline.json').read_text())
            if report['frames']!=state['total'] or report.get('detected_frames')!=state['total'] or not 'NVENC' in report['encoder']:raise RuntimeError('Incomplete or wrong encoder')
            rows=dict(label=a.label,profile=profile,video=str(video.resolve()),state=state,pipeline=report,
                      profile_config=boot['profiles']['coco'][profile])
            records.append(rows);(a.output/(a.label+'_comparison.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
            print(a.label,video.name,profile,round(report['processing_fps'],2),'FPS',round(state['evaluation']['map50']*100,2),'AP50',flush=True)


if __name__=='__main__':main()
