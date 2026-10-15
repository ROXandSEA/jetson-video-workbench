"""Optional full-frame + four overlapping views, using the SAME TRT engine.

Each source frame is inferred five times. This is an explicit accuracy option,
never a claim of single-pass hundred-FPS processing. No GT selects the crops.
"""
import time
import cv2
import numpy as np


def tile_rectangles(shape, fraction=.6):
    if not .5 <= fraction <= .8:
        raise ValueError('tile_fraction must be between 0.5 and 0.8')
    h,w=shape[:2]
    cw,ch=max((w+1)//2,round(w*fraction)),max((h+1)//2,round(h*fraction))
    return [(x,y,cw,ch) for y in (0,h-ch) for x in (0,w-cw)]


def restore_crop_boxes(detections,rect,shape,margin=2):
    """Reject truncations at internal seams, preserve outer image borders."""
    x,y,w,h=rect
    valid=np.isfinite(detections).all(axis=1)
    if x>0:valid &= detections[:,0]>margin
    if y>0:valid &= detections[:,1]>margin
    if x+w<shape[1]:valid &= detections[:,2]<w-margin
    if y+h<shape[0]:valid &= detections[:,3]<h-margin
    rows=detections[valid].copy()
    rows[:,[0,2]]+=x;rows[:,[1,3]]+=y
    rows[:,[0,2]]=np.clip(rows[:,[0,2]],0,shape[1])
    rows[:,[1,3]]=np.clip(rows[:,[1,3]],0,shape[0])
    return rows[(rows[:,2]>rows[:,0])&(rows[:,3]>rows[:,1])]


def merge_views(views,frame_ops,iou,cap):
    """Class-constrained global NMS; one shared cap after all views merge."""
    rows=np.concatenate(views)
    if not len(rows):return np.empty((0,6),np.float32)
    keep=frame_ops.nms(rows[:,:4],rows[:,4],rows[:,5].astype(np.int32),iou,cap)
    if keep is None:
        selected=[]
        for cls in np.unique(rows[:,5]):
            ids=np.flatnonzero(rows[:,5]==cls);boxes=rows[ids,:4].copy();boxes[:,2:]-=boxes[:,:2]
            k=cv2.dnn.NMSBoxes(boxes.tolist(),rows[ids,4].tolist(),0,iou)
            selected.extend(ids[np.asarray(k).reshape(-1)].tolist())
        keep=np.asarray(selected,int)
    keep=np.asarray(keep,int)
    keep=keep[np.argsort(-rows[keep,4],kind='stable')[:cap]]
    return rows[keep].astype(np.float32,copy=False)


def extra_views(frame,full,detector,cfg,rt,cancel):
    start=time.perf_counter();prepare_ms=infer_ms=0.;calls=0;views=[full]
    postprocess=rt.postprocess_compact_pair if getattr(detector,'reduce_lib',None) else rt.postprocess_pair
    for rect in tile_rectangles(frame.shape,float(cfg.get('tile_fraction',.6))):
        if cancel.is_set():break
        x,y,w,h=rect
        begin=time.perf_counter()
        tensor,geometry=rt.preprocess(frame[y:y+h,x:x+w],detector.input_size,detector.dtype)
        prepared=time.perf_counter();prepare_ms+=(prepared-begin)*1000
        raw=detector.raw(tensor);inferred=time.perf_counter();infer_ms+=(inferred-prepared)*1000;calls+=1
        minimum=min(cfg['track_low_thresh'],cfg.get('evaluation_conf',.001))
        _,rows=postprocess(raw,geometry,len(detector.names),minimum,
                          minimum,cfg['nms_iou'],cfg['max_det'])
        views.append(restore_crop_boxes(rows,rect,frame.shape))
    rows=merge_views(views,rt.frame_ops,float(cfg.get('tile_merge_iou',.6)),cfg['max_det'])
    # Attribute the extra GPU calls to inference, rather than hiding them in NMS.
    post_ms=max(0,(time.perf_counter()-start)*1000-prepare_ms-infer_ms)
    return rows,calls,np.array([prepare_ms,infer_ms,post_ms])
