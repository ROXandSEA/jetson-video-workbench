#!/usr/bin/env python3
"""Auxiliary MOT identity comparison, using official IDs at preview resolution.

Requires developer-only motmetrics (not a web/runtime dependency). This is a
single-sequence comparison with an explicit ignore policy, not an official
MOTChallenge submission score.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if (ROOT/'.state/dev_modules').is_dir():sys.path.insert(0,str(ROOT/'.state/dev_modules'))
from runtime.backend import sha256

def main():
    import motmetrics as mm
    p=argparse.ArgumentParser();p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();gt=np.loadtxt(a.gt,delimiter=',',ndmin=2)
    if sha256(a.gt)!='89fd0196d67a5eb6011a470dc2a49b02255403b49e8848031cdf99add8a36d9c':
        raise ValueError('This auxiliary probe requires the verified official MOT20-01 GT')
    results=[]
    for folder in a.runs:
        report=json.loads((folder/'pipeline.json').read_text())
        sx=report['source_size'][0]/1920;sy=report['source_size'][1]/1080
        if report['source_size']!=[960,540] or report['frames']!=429:raise ValueError('This probe is configured for MOT20-01 preview')
        tracks={}
        with (folder/'tracks.csv').open() as f:
            for row in csv.DictReader(f):
                if int(row['class_id'])!=0:continue
                tracks.setdefault(int(row['frame']),[]).append([int(row['track_id']),*[float(row[k]) for k in ['x','y','width','height']]])
        acc=mm.MOTAccumulator(auto_id=True);ignored=0
        for frame in range(1,report['frames']+1):
            source=gt[gt[:,0]==frame];boxes=source[:,2:6].copy()
            boxes[:,:2]-=1;boxes[:,[0,2]]*=sx;boxes[:,[1,3]]*=sy
            # Match the application's border clipping used by its GT converter.
            far=boxes[:,:2]+boxes[:,2:];boxes[:,:2]=np.maximum(boxes[:,:2],0)
            far=np.minimum(far,report['source_size']);boxes[:,2:]=np.maximum(far-boxes[:,:2],0)
            active=(source[:,6]==1)&(source[:,7]==1)&(boxes[:,2]>0)&(boxes[:,3]>0)
            ignore=((source[:,6]==0)&(source[:,7]==1))|np.isin(source[:,7],[2,7,8,12])
            expected=boxes[active];ids=source[active,1].astype(int)
            rows=np.array(tracks.get(frame,[]),float).reshape(-1,5)
            predicted=rows[:,1:];pid=rows[:,0].astype(int)
            distance=mm.distances.iou_matrix(expected,predicted,max_iou=.5)
            keep=np.ones(len(pid),bool)
            # Ignore only otherwise-unmatchable predictions overlapping ignored GT.
            for index,box in enumerate(predicted):
                if np.isfinite(distance[:,index]).any():continue
                overlap=np.maximum(0,np.minimum(box[:2]+box[2:],boxes[ignore,:2]+boxes[ignore,2:])-np.maximum(box[:2],boxes[ignore,:2]))
                if len(overlap) and np.max(np.prod(overlap,axis=1))/max(np.prod(box[2:]),1e-12)>=.5:
                    keep[index]=False;ignored+=1
            acc.update(ids.tolist(),pid[keep].tolist(),distance[:,keep])
        metrics=['idf1','idp','idr','mota','num_switches','num_objects','num_false_positives','num_misses','num_predictions']
        scores=mm.metrics.create().compute(acc,metrics=metrics,name='run').iloc[0].to_dict()
        results.append(dict(run=folder.name,profile=report['profile'],frames=report['frames'],ignored_predictions=ignored,metrics=scores,
                            track_sha256=sha256(folder/'tracks.csv')))
        print(folder.name,scores,flush=True)
    value=dict(dataset='MOT20-01 official train GT / 960x540 preview',gt_sha256=sha256(a.gt),
               method='motmetrics 1.4.0, IoU >= 0.5, marked pedestrians, all visibility; unmatched predictions with >=50% area on ignore regions excluded',
               limitation='Auxiliary comparison of these configurations only; not the official MOTChallenge evaluation or whole-dataset score',runs=results)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
