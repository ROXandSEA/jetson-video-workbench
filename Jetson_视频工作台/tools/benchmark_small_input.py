#!/usr/bin/env python3
"""Developer exploration of small-input accuracy on a frozen video subset.

All variants use the same FP16 512x320 engine, AP=.001 and identical GT.
Single-pass image/NMS variants and explicit extra-view variants are measured
separately; detector time is not reported as whole-pipeline FPS.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'app')]
from runtime import backend as rt
from evaluation import evaluate_detections


def merge(views,iou=.65,cap=300):
    rows=np.concatenate(views) if views else np.empty((0,6),np.float32)
    if not len(rows):return rows
    keep=rt.frame_ops.nms(rows[:,:4],rows[:,4],rows[:,5].astype(np.int32),iou,cap)
    if keep is None:
        chosen=[]
        for cls in np.unique(rows[:,5]):
            ids=np.flatnonzero(rows[:,5]==cls);boxes=rows[ids,:4].copy();boxes[:,2:]-=boxes[:,:2]
            k=cv2.dnn.NMSBoxes(boxes.tolist(),rows[ids,4].tolist(),0,iou)
            chosen.extend(ids[np.asarray(k).reshape(-1)].tolist())
        keep=np.asarray(chosen,int)
    keep=np.asarray(keep,int);keep=keep[np.argsort(-rows[keep,4],kind='stable')[:cap]]
    return rows[keep]


def roi(frame,dets,fraction=.65):
    h,w=frame.shape[:2];rw,rh=round(w*fraction),round(h*fraction)
    boxes=dets[(dets[:,5]==0)&(dets[:,4]>=.05)]
    r=min(320/h,512/w)
    boxes=boxes[np.minimum(boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1])*r<24]
    centers=(boxes[:,:2]+boxes[:,2:4])*.5
    best=(-1,(round((w-rw)/2),round((h-rh)/2)))
    for cy in [.3,.5,.7]:
        for cx in [.3,.5,.7]:
            x=max(0,min(w-rw,round(cx*w-rw/2)));y=max(0,min(h-rh,round(cy*h-rh/2)))
            inside=(centers[:,0]>=x)&(centers[:,0]<x+rw)&(centers[:,1]>=y)&(centers[:,1]<y+rh)
            score=float(np.sqrt(boxes[inside,4]).sum())
            if score>best[0]:best=(score,(x,y))
    x,y=best[1];return x,y,rw,rh


def infer_crop(detector,frame,rect):
    x,y,w,h=rect;crop=frame[y:y+h,x:x+w]
    tensor,geometry=rt.preprocess(crop,detector.input_size,detector.dtype)
    raw=detector.raw(tensor)
    _,dets=rt.postprocess_compact_pair(raw,geometry,80,.05,.001,.7,300)
    # Discard truncations touching an internal tile edge; original full-frame
    # detections still cover those pixels. No GT is used for selection/filtering.
    valid=np.ones(len(dets),bool)
    if x>0:valid &= dets[:,0]>2
    if y>0:valid &= dets[:,1]>2
    if x+w<frame.shape[1]:valid &= dets[:,2]<w-2
    if y+h<frame.shape[0]:valid &= dets[:,3]<h-2
    dets=dets[valid].copy();dets[:,[0,2]]+=x;dets[:,[1,3]]+=y
    return dets


def main():
    p=argparse.ArgumentParser();p.add_argument('--videos',nargs='+',type=Path,required=True)
    p.add_argument('--frames',type=int,default=60);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preset',choices=['explore','refine','confirm'],default='explore')
    p.add_argument('--engine',type=Path,default=ROOT/'models/coco_yolov12n_512x320_fp16.engine')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with urllib.request.urlopen('http://127.0.0.1:8765/api/status') as s:
        if json.load(s)['status'] in {'loading','running','encoding','evaluating','stopping'}:raise RuntimeError('Wait for current web task')
    variants=['linear_nms070','linear_nms060','linear_nms080','linear_nms090',
              'area','lanczos','unsharp','gamma','clahe','dense_roi055','dense_roi065','dense_roi075','tiles4']
    if a.preset=='refine':
        variants=['linear_nms045','linear_nms050','linear_nms055','linear_nms060','linear_nms070',
                  'tiles_top2','tiles_best2','tiles4_050','tiles4_055','tiles4_065']
    elif a.preset=='confirm':variants=['linear_nms050','linear_nms055','linear_nms060','linear_nms070']
    records=[]
    engine=a.engine
    detector=rt.TRTDetector('coco',engine=engine,gpu_postprocess=True)
    try:
        blank=np.zeros(detector.buffers[detector.input_name].shape,np.float32)
        for _ in range(30):detector.raw(blank)
        detector.enable_cuda_graph()
        lut=np.round((np.arange(256)/255.)**.85*255).astype(np.uint8)
        for video in a.videos:
            full=json.loads(video.with_suffix('.gt.json').read_text());total=full['info']['frame_count']
            ids=np.linspace(1,total,min(a.frames,total),dtype=int).tolist();wanted=set(ids)
            gt={**full,'images':[x for x in full['images'] if x['id'] in wanted],
                'annotations':[x for x in full['annotations'] if x['image_id'] in wanted]}
            del full
            mapping={k:c['id'] for k,n in detector.names.items() for c in gt['categories'] if c['name']==n}
            predictions={v:[] for v in variants};elapsed={v:[] for v in variants}
            capture=cv2.VideoCapture(str(video));index=0;middle=ids[len(ids)//2]
            try:
                while True:
                    ok,frame=capture.read()
                    if not ok:break
                    index+=1
                    if index not in wanted:continue
                    start=time.perf_counter();tensor,geometry=rt.preprocess(frame,detector.input_size,detector.dtype);raw=detector.raw(tensor).copy()
                    base_t=time.perf_counter()-start
                    outputs={}
                    for threshold in [int(v[-3:])/100 for v in variants if v.startswith('linear_nms')]:
                        begin=time.perf_counter();_,ds=rt.postprocess_compact_pair(raw,geometry,80,.05,.001,threshold,300)
                        key=f'linear_nms{int(threshold*100):03d}';outputs[key]=ds;elapsed[key].append(base_t+time.perf_counter()-begin)
                    for variant in ['area','lanczos','unsharp','gamma','clahe']:
                        if variant not in variants:continue
                        begin=time.perf_counter();image=frame
                        if variant in ['area','lanczos']:
                            ratio=geometry[0];image=cv2.resize(frame,(round(frame.shape[1]*ratio),round(frame.shape[0]*ratio)),interpolation=cv2.INTER_AREA if variant=='area' else cv2.INTER_LANCZOS4)
                        elif variant=='unsharp':image=cv2.addWeighted(frame,1.35,cv2.GaussianBlur(frame,(0,0),1.),-.35,0)
                        elif variant=='gamma':image=cv2.LUT(frame,lut)
                        else:
                            lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0]);image=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
                        tensor,g=rt.preprocess(image,detector.input_size,detector.dtype)
                        _,ds=rt.postprocess_compact_pair(detector.raw(tensor),g,80,.05,.001,.7,300)
                        if variant in ['area','lanczos']:ds[:,:4]/=geometry[0]
                        outputs[variant]=ds;elapsed[variant].append(time.perf_counter()-begin)
                    if any(v.startswith('dense_roi') for v in variants):
                        begin=time.perf_counter();local=infer_crop(detector,frame,roi(frame,outputs['linear_nms070']))
                        cost=time.perf_counter()-begin
                        for threshold in [.55,.65,.75]:
                            key=f'dense_roi{int(threshold*100):03d}';begin=time.perf_counter()
                            outputs[key]=merge([outputs['linear_nms070'],local],threshold)
                            elapsed[key].append(elapsed['linear_nms070'][-1]+cost+time.perf_counter()-begin)
                    if any(v.startswith('tiles') for v in variants):
                        fh,fw=frame.shape[:2];cw,ch=round(fw*.6),round(fh*.6)
                        rectangles=[(x,y,cw,ch) for y in [0,fh-ch] for x in [0,fw-cw]]
                        tiles=[];costs=[]
                        for rect in rectangles:
                            begin=time.perf_counter();tiles.append(infer_crop(detector,frame,rect));costs.append(time.perf_counter()-begin)
                        base=outputs['linear_nms070'];centers=(base[:,:2]+base[:,2:4])*.5
                        small=(base[:,5]==0)&(base[:,4]>=.05)&(np.minimum(base[:,2]-base[:,0],base[:,3]-base[:,1])*geometry[0]<24)
                        density=[]
                        for x,y,w,h in rectangles:
                            inside=small&(centers[:,0]>=x)&(centers[:,0]<x+w)&(centers[:,1]>=y)&(centers[:,1]<y+h)
                            density.append(float(np.sqrt(base[inside,4]).sum()))
                        for variant in variants:
                            if not variant.startswith('tiles'):continue
                            selected=[0,1] if variant=='tiles_top2' else np.argsort(-np.array(density),kind='stable')[:2].tolist() if variant=='tiles_best2' else list(range(4))
                            threshold=int(variant[-3:])/100 if variant.startswith('tiles4_') else .55 if variant in ['tiles_top2','tiles_best2'] else .65
                            begin=time.perf_counter();outputs[variant]=merge([base,*[tiles[i] for i in selected]],threshold)
                            elapsed[variant].append(elapsed['linear_nms070'][-1]+sum(costs[i] for i in selected)+time.perf_counter()-begin)
                    for variant,ds in outputs.items():
                        for x1,y1,x2,y2,score,cls in ds:
                            if int(cls) in mapping:predictions[variant].append(dict(image_id=index,category_id=mapping[int(cls)],bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
                    if index==middle:
                        np.savez_compressed(a.output/(video.stem+'_visual.npz'),frame=frame,frame_id=index,**outputs)
                    if index%500==0:print(video.name,'frame',index,flush=True)
            finally:capture.release()
            for variant in variants:
                with (a.output/(video.stem+'_'+variant+'.log')).open('w') as log,contextlib.redirect_stdout(log):scores=evaluate_detections(gt,predictions[variant])
                record=dict(video=str(video),video_sha256=rt.sha256(video),engine_sha256=rt.sha256(engine),
                    variant=variant,input_shape=list(detector.buffers[detector.input_name].shape),frame_ids=ids,metrics=scores,
                    mean_detector_prepost_ms=float(np.mean(elapsed[variant]))*1000,
                    views=5 if variant.startswith('tiles4') else 3 if variant.startswith('tiles') else 2 if variant.startswith('dense_roi') else 1,
                    note='Frozen subset, not full-video AP; isolated detector/pre/post time is not pipeline FPS')
                records.append(record);print(video.name,variant,round(scores['map50']*100,2),round(scores['map50_95']*100,2),'AP;',round(record['mean_detector_prepost_ms'],2),'ms',flush=True)
            (a.output/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    finally:detector.close()


if __name__=='__main__':main()
