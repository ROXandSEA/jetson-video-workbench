"""Independent TensorRT / NumPy / BoT-SORT video pipeline."""
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from collections import deque


def process_video(request, publish, cancel):
    from pipeline import load_names, resolve
    from pipeline_workers import FrameProducer, VideoOutput
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import backend as rt
    import numpy as np
    import cv2
    import tensorrt as trt
    from tracker.mc_bot_sort import BoTSORT
    cfg = request['config']
    sys.setswitchinterval(max(.0005,float(cfg.get('gil_interval_ms',5))/1000))
    cv2.setNumThreads(max(1,int(cfg.get('opencv_threads',2))))
    mode = request['mode']
    if mode == 'coco' and cfg.get('with_reid'):
        raise RuntimeError('COCO 模式不能启用移交的无人机 ReID。')
    if not 0 <= cfg['track_low_thresh'] <= cfg['track_high_thresh'] <= 1:
        raise RuntimeError('跟踪阈值无效。')
    output = Path(request['output'])
    capture = detector = tracker = producer = video_output = None
    frame_count = track_rows = 0
    timing_window = deque(maxlen=120)
    stage_total = np.zeros(6)
    started = time.perf_counter()
    report = None
    try:
        if cancel.is_set(): return {}
        engine = resolve(request['config_dir'],cfg['engine'])
        # TRTDetector checks sidecar hash, mode and fixed shape before loading.
        metadata = json.loads(engine.with_suffix('.meta.json').read_text())
        detector = rt.TRTDetector(mode,metadata['size'],str(engine),gpu_postprocess=cfg.get('gpu_postprocess',False))
        names = load_names(request['config_dir'],cfg,mode)
        if names != [detector.names[i] for i in range(len(detector.names))]:
            raise RuntimeError('网页标签与引擎元数据不匹配。')
        blank = np.zeros(detector.buffers[detector.input_name].shape,dtype=detector.dtype)
        for _ in range(30):
            if cancel.is_set():return {}
            detector.raw(blank)
        if cfg.get('cuda_graph',False):
            detector.enable_cuda_graph()
            for _ in range(5):detector.raw(blank)
        capture = cv2.VideoCapture(request['source'])
        if not capture.isOpened():raise RuntimeError('OpenCV 无法打开输入视频。')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:fps=30
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = [int(capture.get(k)) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT]]
        if min(width,height)<=0:raise RuntimeError('无法读取视频尺寸。')
        # Direct H.264 + detection logs; no MJPEG intermediate.
        estimate=max(512*2**20,total*min(width*height*.15,200_000)) if total>0 else 512*2**20
        if shutil.disk_usage(output).free<estimate:
            raise RuntimeError(f'导出空间不足：预计至少需要 {estimate/2**30:.2f} GiB。')
        args=rt.tracker_args(cfg.get('with_reid',False),cfg.get('cmc_method','sparseOptFlow'))
        for key in ['track_high_thresh','track_low_thresh','new_track_thresh','track_buffer','match_thresh']:
            if key in cfg:setattr(args,key,cfg[key])
        tracker=BoTSORT(args,frame_rate=fps)
        if tracker.gmc.method=='sparseOptFlow':
            tracker.gmc.downscale=max(1,int(cfg.get('gmc_downscale',2)))
            tracker.gmc.feature_params['maxCorners']=int(cfg.get('gmc_max_corners',1000))
        run_start=time.perf_counter()
        video_output=VideoOutput(output,width,height,fps,cfg,publish,cfg.get('preview_width',request['preview_width']),cfg.get('preview_fps',request['preview_fps']))
        async_gmc=cfg.get('async_camera_motion',True) and tracker.gmc.method=='sparseOptFlow'
        producer=FrameProducer(capture,detector,cfg,cancel,rt,gmc=tracker.gmc if async_gmc else None)
        stage_names=['read_ms','preprocess_ms','infer_transfer_ms','postprocess_ms','tracker_ms','output_queue_ms']
        publish(status='running',message='正在检测与追踪…',total=total)
        with (output/'timings.csv').open('w',newline='') as timing:
            tw=csv.writer(timing);tw.writerow(['frame',*stage_names,'cycle_ms'])
            while not cancel.is_set():
                tick=time.perf_counter()
                item=producer.next()
                if item is None:break
                frame,dets,eval_dets,detection_times,features=item
                if frame.shape[:2]!=(height,width):raise RuntimeError('视频中途分辨率变化。')
                tracking_start=time.perf_counter()
                prepared=features.result() if features is not None else None
                online,_=tracker.update(dets,frame,gmc_prepared=prepared)
                frame_count+=1
                rows=[]
                for track in online:
                    if not track.is_activated or track.frame_id!=tracker.frame_id:continue
                    x,y,w,h=rt.clipped_track_box(track,frame.shape)
                    if w*h<cfg.get('min_box_area',1):continue
                    cls=int(track.cls);tid=int(track.track_id)
                    rows.append((tid,cls,names[cls],x,y,w,h,float(track.score)))
                track_rows+=len(rows)
                tracking_end=time.perf_counter()
                video_output.submit((frame_count,frame,rows,dets,eval_dets))
                now=time.perf_counter()
                stages=np.r_[detection_times,(tracking_end-tracking_start)*1000,(now-tracking_end)*1000]
                stage_total+=stages;timing_window.append((now-tick)*1000)
                tw.writerow([frame_count,*stages,timing_window[-1]])
                elapsed=now-run_start
                publish(frame=frame_count,total=total,targets=len(rows),progress=min(99,frame_count/total*100) if total>0 else None,
                        fps=round(frame_count/max(elapsed,.001),1),latency_ms=round(sum(timing_window)/len(timing_window),1),elapsed=round(now-started,1))
                if frame_count%100==0 and shutil.disk_usage(output).free<256*2**20:
                    raise RuntimeError('剩余磁盘空间低于 256 MiB，已停止写入。')
        producer.close();inference_calls=producer.inference_calls;producer=None
        publish(message='正在保存视频末尾和索引…')
        video_output.close()
        elapsed=time.perf_counter()-run_start
        if not frame_count and not cancel.is_set():raise RuntimeError('输入没有可解码帧。')
        if total>0 and frame_count!=total and not cancel.is_set():raise RuntimeError(f'输入提前结束：解码 {frame_count}/{total} 帧。')
        if video_output.frames!=frame_count:raise RuntimeError('输出帧数与处理帧数不一致。')
        report=dict(mode=mode,profile=request.get('profile'),evaluation_confidence=cfg.get('evaluation_conf',.001),tracking_confidence=cfg['track_low_thresh'],frames=frame_count,track_rows=track_rows,source_fps=fps,source_size=[width,height],input_shape=list(detector.buffers[detector.input_name].shape),
                    output_shape=list(detector.buffers[detector.output_name].shape),engine=str(engine),engine_metadata=metadata,
                    tensorrt_version=trt.__version__,reid=args.with_reid,processing_seconds=elapsed,processing_fps=frame_count/max(elapsed,.001),
                    stages_mean_ms=(stage_total/max(frame_count,1)).tolist(),stage_names=stage_names,
                    output_stage_names=['draw_logs_ms','encode_submit_ms','preview_ms'],
                    output_stages_mean_ms=[v/max(frame_count,1) for v in video_output.stage_total],
                    encoder=video_output.encoder_kind,encoding_seconds=video_output.close_seconds,cuda_graph=detector.graph_status,
                    async_camera_motion=async_gmc,
                    camera_motion_method=tracker.gmc.method,gmc_downscale=tracker.gmc.downscale,
                    native_frame_ops=rt.frame_ops.LIB is not None,
                    gpu_channel_reduction=detector.reduce_lib is not None,
                    native_nms=rt.frame_ops.NMS_LIB is not None,detected_frames=frame_count,detection_interval=1,
                    detector_inferences=inference_calls,detail_views=cfg.get('detail_views','full'),
                    nms_iou=cfg['nms_iou'],tile_merge_iou=cfg.get('tile_merge_iou') if cfg.get('detail_views')=='tiles4' else None,
                    draw_style=cfg.get('draw_style','antialiased'),
                    gil_interval_ms=sys.getswitchinterval()*1000,
                    pipeline='bounded decode/TRT + BoT-SORT + draw/NVENC; 2-frame queues; no frame skipping',
                    detector_code_sha256=rt.sha256(Path(rt.__file__)),tracker_code_sha256=rt.sha256(rt.BOT/'tracker/mc_bot_sort.py'),
                    detections_sha256=rt.sha256(output/'detections.jsonl'),cancelled=cancel.is_set(),
                    note='FPS=实际帧数/处理墙钟时间，含队列排空、全部帧编码、MP4索引；不含加载、预热与mAP。各线程时间重叠，不能相加计算FPS。')
        (output/'pipeline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if cancel.is_set():return {}
        publish(fps=round(report['processing_fps'],1),elapsed=round(time.perf_counter()-started,1))
        result=str(output/'result.mp4')
        if result and not cancel.is_set():
            from evaluation import evaluate_video
            publish(status='evaluating', message='正在核对标注并计算检测 mAP…', progress=99)
            evaluation_start=time.perf_counter()
            evaluation=evaluate_video(request['source'],mode,frame_count,[width,height],names,
                                      output/'evaluation_detections.jsonl',output,
                                      explicit=request.get('annotations',''),
                                      confidence=cfg.get('evaluation_conf',.001),cancel=cancel)
            report['evaluation_seconds']=time.perf_counter()-evaluation_start
            report['evaluation']=evaluation
            publish(evaluation=evaluation)
        (output/'pipeline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if cancel.is_set():return {}
        if result:publish(progress=100)
        return {'result_file':result}
    finally:
        active_error=sys.exc_info()[0]
        cleanup_error=None
        callbacks=[]
        if producer is not None:callbacks.append(producer.close)
        if video_output is not None and video_output.thread.is_alive():callbacks.append(video_output.close)
        if capture is not None:callbacks.append(capture.release)
        if detector is not None:callbacks.append(detector.close)
        for close in callbacks:
            try:close()
            except BaseException as error:cleanup_error=cleanup_error or error
        tracker=None
        if cleanup_error is not None and active_error is None:raise cleanup_error
