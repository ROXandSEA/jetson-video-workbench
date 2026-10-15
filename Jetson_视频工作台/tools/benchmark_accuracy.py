#!/usr/bin/env python3
"""Paired frozen 60-frame AP and isolated detector latency; no PyTorch required."""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'app')]
from runtime.backend import TRTDetector,preprocess,postprocess,sha256
from evaluation import evaluate_detections

def audit(video,engine,count,output):
    gt=json.loads(video.with_suffix('.gt.json').read_text());selected=np.linspace(1,gt['info']['frame_count'],min(count,gt['info']['frame_count']),dtype=int).tolist()
    wanted=set(selected);gt['images']=[i for i in gt['images'] if i['id'] in wanted];gt['annotations']=[a for a in gt['annotations'] if a['image_id'] in wanted]
    detector=TRTDetector('coco',engine=engine);mapping={k:c['id'] for k,n in detector.names.items() for c in gt['categories'] if c['name']==n}
    capture=cv2.VideoCapture(str(video));ok,first=capture.read();capture.set(cv2.CAP_PROP_POS_FRAMES,0)
    tensor,_=preprocess(first,detector.size,dtype=detector.dtype)
    for _ in range(60):detector.raw(tensor)
    latency=[]
    for _ in range(200):
        start=time.perf_counter();detector.raw(tensor);latency.append((time.perf_counter()-start)*1000)
    predictions={.05:[],.001:[]};timings=[];index=0;middle=selected[len(selected)//2]
    while True:
        ok,frame=capture.read()
        if not ok:break
        index+=1
        if index not in wanted:continue
        start=time.perf_counter();tensor,geometry=preprocess(frame,detector.size,dtype=detector.dtype);raw=detector.raw(tensor);timings.append((time.perf_counter()-start)*1000)
        for conf in predictions:
            detections=postprocess(raw,geometry,len(detector.names),conf,.7,300)
            for x1,y1,x2,y2,score,cls in detections:
                if int(cls) in mapping:predictions[conf].append(dict(image_id=index,category_id=mapping[int(cls)],bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
            if index==middle and conf==.05:
                np.savez_compressed(output/f'{video.stem}_{engine.stem}_visual.npz',frame=frame,detections=detections,frame_id=index)
        if index==selected[0]:np.savez_compressed(output/f'{video.stem}_{engine.stem}_raw_sample.npz',raw=raw.copy())
    capture.release();detector.close();records=[]
    for conf,items in predictions.items():
        name=f'{video.stem}_{engine.stem}_conf{conf}'
        with (output/(name+'.log')).open('w') as stream,contextlib.redirect_stdout(stream):scores=evaluate_detections(gt,items)
        record=dict(video=video.name,video_sha256=sha256(video),engine=engine.name,engine_sha256=sha256(engine),size=detector.size,confidence=conf,
                    sampling='uniform frozen 60-frame subset, not full-video AP',frame_ids=selected,metrics=scores,
                    raw_inference_with_H2D_D2H_mean_ms=float(np.mean(latency)),raw_inference_p50_ms=float(np.median(latency)),raw_inference_p95_ms=float(np.percentile(latency,95)),
                    preprocess_and_inference_mean_ms=float(np.mean(timings)),prediction_count=len(items),
                    hardware='Orin NX 16GB / 40W; clocks not locked; no GPU builds/jobs during measurement')
        (output/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');records.append(record)
        print(name,'AP50',scores['map50'],'AP50:95',scores['map50_95'],'raw ms',record['raw_inference_with_H2D_D2H_mean_ms'],flush=True)
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engines',nargs='+',required=True,type=Path);p.add_argument('--frames',type=int,default=60);p.add_argument('--output',type=Path,default=ROOT/'reports/benchmarks')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=[]
    for engine in a.engines:
        for video in [ROOT/'videos/MOT20/MOT20-01-raw.mp4',ROOT/'videos/VisDrone/VisDrone-val-uav0000137_00458_v-raw.mp4']:
            records.extend(audit(video,engine,a.frames,a.output))
        (a.output/'paired_benchmark.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
