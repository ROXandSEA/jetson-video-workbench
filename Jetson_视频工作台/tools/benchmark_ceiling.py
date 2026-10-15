#!/usr/bin/env python3
"""Sequential whole-video experiments; never benchmark concurrent GPU tasks."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,default=ROOT/'videos/MOT20/MOT20-01-raw.mp4')
    p.add_argument('--output-root',type=Path,default=ROOT/'.state'/('ceiling_bench_'+time.strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=False)
    runs=[(f'{profile}_{repeat}',profile,{}) for repeat in range(a.repeats) for profile in ['balanced','fast_motion','fast_static','speed']]
    for label,profile,settings in runs:
        output=a.output_root/label
        if output.exists():raise RuntimeError('Use a new run name rather than overwrite evidence: '+label)
        command=[sys.executable,str(ROOT/'tools/benchmark_pipeline.py'),'--video',str(a.video.resolve()),
                 '--profile',profile,'--output',str(output),'--label',label,'--cfg-overrides',json.dumps(settings)]
        with (output.parent/(label+'.log')).open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/'pipeline.json').read_text())
        print(label,round(report['processing_fps'],2),'FPS','AP50',round(report['evaluation']['map50']*100,2),flush=True)
if __name__=='__main__':main()
