"""Bounded producer/output stages. Every processed frame is encoded, in order."""
import csv
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class FrameProducer:
    """Single owner of capture/TRT; GPU work overlaps CPU tracking/output."""
    def __init__(self, capture, detector, cfg, cancel, rt, gmc=None):
        self.capture, self.detector, self.cfg = capture, detector, cfg
        self.cancel, self.rt = cancel, rt
        self.queue = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.error = None
        self.inference_calls = 0
        self.gmc = gmc
        self.feature_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-features') if gmc else None
        self.thread = threading.Thread(target=self.run, name='decode-trt', daemon=True)
        self.thread.start()

    def put(self, item):
        while not self.stop.is_set() and not self.cancel.is_set():
            try:
                self.queue.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def run(self):
        try:
            import numpy as np
            def prepare():
                begin = time.perf_counter()
                ok, frame = self.capture.read()
                read = time.perf_counter()
                if not ok:
                    return None
                features = self.feature_pool.submit(self.gmc.prepareSparseOptFlow, frame) if self.feature_pool else None
                tensor, geometry = self.rt.preprocess(frame, getattr(self.detector,'input_size',self.detector.size), dtype=self.detector.dtype)
                return frame, tensor, geometry, (read-begin)*1000, (time.perf_counter()-read)*1000, features
            current = prepare()
            while not self.stop.is_set() and not self.cancel.is_set():
                if current is None:
                    break
                frame, tensor, geometry, read_ms, preprocess_ms, features = current
                begin=time.perf_counter();self.detector.submit(tensor);self.inference_calls += 1
                # One original frame of lookahead; CPU decoding/letterbox overlaps GPU.
                following = prepare()
                raw = self.detector.synchronize()
                inferred = time.perf_counter()
                postprocess=self.rt.postprocess_compact_pair if getattr(self.detector,'reduce_lib',None) else self.rt.postprocess_pair
                dets, eval_dets = postprocess(
                    raw, geometry, len(self.detector.names), self.cfg['track_low_thresh'],
                    self.cfg.get('evaluation_conf', .001), self.cfg['nms_iou'], self.cfg['max_det'])
                timings = np.array([read_ms, preprocess_ms, (inferred-begin)*1000,
                                    (time.perf_counter()-inferred)*1000])
                if self.cfg.get('detail_views') == 'tiles4':
                    from runtime.spatial_detail import extra_views
                    full=eval_dets if self.cfg.get('evaluation_conf',.001)<=self.cfg['track_low_thresh'] else dets
                    merged,calls,extra_times=extra_views(frame,full,self.detector,self.cfg,self.rt,self.cancel)
                    self.inference_calls += calls
                    timings[1:] += extra_times
                    dets=merged[merged[:,4]>=self.cfg['track_low_thresh']]
                    eval_dets=merged[merged[:,4]>=self.cfg.get('evaluation_conf',.001)]
                if not self.put((frame, dets, eval_dets, timings, features)):
                    break
                current = following
        except BaseException as error:
            self.error = error
        finally:
            self.put(None)

    def next(self):
        while not self.cancel.is_set():
            try:
                item = self.queue.get(timeout=.1)
                if item is None and self.error:
                    raise self.error
                return item
            except queue.Empty:
                if not self.thread.is_alive():
                    if self.error:
                        raise self.error
                    return None
        return None

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.feature_pool:
            self.feature_pool.shutdown(wait=True, cancel_futures=True)


class VideoEncoder:
    """NVENC direct MP4 on JetPack; portable direct FFmpeg fallback."""
    def __init__(self, output, width, height, fps, cfg):
        import cv2
        self.partial = Path(output) / 'result.partial.mp4'
        self.final = Path(output) / 'result.mp4'
        self.writer = self.process = self.log = None
        self.frames = 0
        self.input_width = width
        self.padding_right = 0
        self.kind = 'FFmpeg libx264'
        hardware = cfg.get('hardware_encoder', True)
        if hardware and width % 2 == 0 and height % 2 == 0 and 'GStreamer' in cv2.getBuildInformation():
            # A real encoder device is required; plugin presence alone is insufficient.
            if any(Path(p).exists() for p in ('/dev/nvhost-msenc', '/dev/v4l2-nvenc')):
                bitrate = max(4_000_000, min(40_000_000, round(width * height * fps * .16)))
                # OpenCV's GStreamer BGR writer assumes a four-byte row stride.
                # Widths like MOT20-05's 826 otherwise shift/corrupt every row.
                # Pad the CPU BGR input, then crop BGRx (always aligned) before
                # nvvidconv; the saved video keeps the original dimensions.
                self.input_width = (width + 3) // 4 * 4
                self.padding_right = self.input_width - width
                crop = f'videocrop right={self.padding_right} ! ' if self.padding_right else ''
                location = str(self.partial).replace('\\', '\\\\').replace('"', '\\"')
                pipeline = ('appsrc ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
                            'videoconvert n-threads=2 ! video/x-raw,format=BGRx ! '+crop+'nvvidconv ! '
                            'video/x-raw(memory:NVMM),format=NV12 ! '
                            f'nvv4l2h264enc bitrate={bitrate} maxperf-enable=true preset-level=1 '
                            f'iframeinterval={max(1, round(fps))} insert-sps-pps=true ! '
                            f'h264parse ! qtmux faststart=true ! filesink location="{location}" sync=false')
                try:
                    candidate = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (self.input_width, height))
                except cv2.error:
                    candidate = None
                if candidate is not None and candidate.isOpened():
                    self.writer = candidate
                    self.kind = 'Jetson NVENC H.264 (GStreamer)'
                elif candidate is not None:
                    candidate.release()
        if self.writer is None:
            executable = shutil.which('ffmpeg')
            if not executable:
                raise RuntimeError('缺少 ffmpeg，无法保存结果视频。')
            self.log = (Path(output) / 'ffmpeg.log').open('wb')
            command = [executable, '-hide_banner', '-nostdin', '-y', '-f', 'rawvideo',
                       '-pixel_format', 'bgr24', '-video_size', f'{width}x{height}',
                       '-framerate', str(fps), '-i', 'pipe:0', '-an', '-vf',
                       'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-threads', '2',
                       '-preset', 'ultrafast', '-crf', '20', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart', str(self.partial)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.log,
                                            stderr=subprocess.STDOUT, bufsize=0)

    def write(self, frame):
        if self.writer is not None:
            if self.padding_right:
                import cv2
                frame=cv2.copyMakeBorder(frame,0,0,0,self.padding_right,cv2.BORDER_REPLICATE)
            self.writer.write(frame)
        else:
            # Raw pipe writes can be partial. Never silently truncate a frame.
            data = memoryview(frame).cast('B')
            while data:
                count = self.process.stdin.write(data)
                if not count:
                    raise RuntimeError('视频编码器提前退出，请查看 ffmpeg.log。')
                data = data[count:]
        self.frames += 1

    def close(self):
        try:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            if self.process is not None:
                self.process.stdin.close()
                try:
                    code = self.process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                    raise RuntimeError('视频编码器结束超时。')
                self.process = None
                if code:
                    raise RuntimeError('视频编码失败，请查看 ffmpeg.log。')
            if self.frames:
                probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                        '-show_entries', 'stream=codec_name,nb_frames', '-of', 'json',
                                        str(self.partial)], capture_output=True, text=True, timeout=15)
                streams = json.loads(probe.stdout or '{}').get('streams', [])
                if probe.returncode or not streams or streams[0].get('codec_name') != 'h264':
                    raise RuntimeError('结果视频未正确编码，已保留 partial 文件。')
                if int(streams[0].get('nb_frames', -1)) != self.frames:
                    raise RuntimeError('编码视频帧数不符，已保留 partial 文件。')
                self.partial.replace(self.final)
        finally:
            if self.log is not None:
                self.log.close()


class VideoOutput:
    """Draw, ordered logs and encode off the inference/tracking critical path."""
    def __init__(self, output, width, height, fps, cfg, publish, preview_width, preview_fps):
        self.output = Path(output)
        self.width, self.height, self.fps = width, height, fps
        self.cfg, self.publish = cfg, publish
        self.preview_width, self.preview_fps = preview_width, preview_fps
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.frames = 0
        self.stage_total = [0., 0., 0.]
        self.encoder_kind = None
        self.close_seconds = 0
        self.thread = threading.Thread(target=self.run, name='draw-encode', daemon=True)
        self.thread.start()

    def submit(self, item):
        while self.thread.is_alive():
            if self.error:
                raise self.error
            try:
                self.queue.put(item, timeout=.1)
                return
            except queue.Full:
                pass
        raise self.error or RuntimeError('视频输出线程提前结束。')

    def run(self):
        import cv2
        from runtime.frame_ops import DetectionJSON
        detection_json=DetectionJSON()
        encoder = None
        try:
            encoder = VideoEncoder(self.output, self.width, self.height, self.fps, self.cfg)
            self.encoder_kind = encoder.kind
            last_preview = 0
            with (self.output / 'tracks.csv').open('w', newline='') as tf, \
                 (self.output / 'detections.jsonl').open('wb') as df, \
                 (self.output / 'evaluation_detections.jsonl').open('wb') as ef, \
                 (self.output / 'output_timings.csv').open('w', newline='') as timing:
                cw = csv.writer(tf)
                cw.writerow(['frame','track_id','class_id','class_name','x','y','width','height','score'])
                tw = csv.writer(timing)
                tw.writerow(['frame', 'draw_logs_ms', 'encode_submit_ms', 'preview_ms'])
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    frame_id, frame, tracks, detections, eval_detections = item
                    begin = time.perf_counter()
                    for stream,values in [(df,detections),(ef,eval_detections)]:
                        encoded=detection_json.encode(frame_id,values)
                        if encoded is None:encoded=(json.dumps({'frame':frame_id,'detections':values.tolist()})+'\n').encode()
                        stream.write(encoded)
                    for row in tracks:
                        tid, cls, name, x, y, w, h, score = row
                        cw.writerow([frame_id, *row])
                        color = ((tid*71)%160+95, (tid*47)%160+95, (tid*29)%160+95)
                        cv2.rectangle(frame, (round(x), round(y)), (round(x+w), round(y+h)), color, 2)
                        label = f'#{tid} {name} {score:.2f}'
                        origin = (round(x), max(20, round(y)-8))
                        if self.cfg.get('draw_style')=='compact':
                            text_size,_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
                            cv2.rectangle(frame,(origin[0],origin[1]-text_size[1]-2),
                                          (origin[0]+text_size[0]+2,origin[1]+2),(16,16,16),-1)
                            cv2.putText(frame,label,origin,cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_8)
                        else:
                            cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 3, cv2.LINE_AA)
                            cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
                    drawn = time.perf_counter()
                    encoder.write(frame)
                    written = time.perf_counter()
                    if written-last_preview >= 1/max(1, float(self.preview_fps)):
                        scale = min(1, int(self.preview_width)/self.width)
                        preview = cv2.resize(frame, (round(self.width*scale), round(self.height*scale))) if scale < 1 else frame
                        ok, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
                        if ok:
                            self.publish(jpeg=jpeg.tobytes())
                        last_preview = written
                    end = time.perf_counter()
                    times = [(drawn-begin)*1000, (written-drawn)*1000, (end-written)*1000]
                    self.stage_total = [a+b for a,b in zip(self.stage_total, times)]
                    self.frames += 1
                    tw.writerow([frame_id, *times])
        except BaseException as error:
            self.error = error
        finally:
            start = time.perf_counter()
            if encoder is not None:
                try:
                    encoder.close()
                except BaseException as error:
                    self.error = self.error or error
            self.close_seconds = time.perf_counter()-start

    def close(self):
        if self.thread.is_alive() and self.error is None:
            try:
                self.submit(None)
            except BaseException:
                # Still join the encoder's finally block before propagating its error.
                if self.error is None:
                    raise
        self.thread.join()
        if self.error:
            raise self.error
