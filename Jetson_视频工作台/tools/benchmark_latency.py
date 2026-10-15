#!/usr/bin/env python3
"""Alternate engines to check a surprising INT8 slowdown under dynamic clocks."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime.backend import TRTDetector,sha256
records=[]
for repeat in range(3):
    order=['fp16','int8'] if repeat%2==0 else ['int8','fp16']
    for precision in order:
        engine=ROOT/f'models/coco_yolov12n_640_{precision}.engine';detector=TRTDetector('coco',engine=engine)
        tensor=np.zeros((1,3,640,640),dtype=detector.dtype)
        for _ in range(60):detector.raw(tensor)
        times=[]
        for _ in range(250):
            tick=time.perf_counter();detector.raw(tensor);times.append((time.perf_counter()-tick)*1000)
        detector.close()
        r=dict(repeat=repeat,precision=precision,engine_sha256=sha256(engine),input='all-zero fixed shape; engine has data-independent execution graph',mean_ms=float(np.mean(times)),p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)))
        records.append(r);print(r,flush=True)
summary={p:dict(mean_ms=float(np.mean([r['mean_ms'] for r in records if r['precision']==p])),min_run_mean_ms=min(r['mean_ms'] for r in records if r['precision']==p),max_run_mean_ms=max(r['mean_ms'] for r in records if r['precision']==p)) for p in ['fp16','int8']}
output=dict(warmup_per_run=60,measure_per_run=250,alternating_order=True,hardware='Orin NX 16GB / 40W; clocks not locked; desktop and remote desktop remain running',runs=records,summary=summary)
(ROOT/'reports/benchmarks/alternating_latency.json').write_text(json.dumps(output,indent=2)+'\n');print('Summary',summary,flush=True)
