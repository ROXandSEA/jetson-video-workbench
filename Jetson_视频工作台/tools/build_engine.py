#!/usr/bin/env python3
"""Build FP16 or calibrated INT8 locally, without PyTorch. No random INT8 scales."""
import argparse
import ctypes
import datetime
import json
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import tensorrt as trt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.backend import CUDA, preprocess, sha256

class Progress(trt.IProgressMonitor):
    def __init__(self):
        super().__init__(); self.lock=threading.Lock(); self.steps={}; self.last=time.monotonic()
    def phase_start(self,phase_name,parent_phase,num_steps):
        with self.lock:
            self.steps[phase_name]=num_steps
            if parent_phase is None:print(f'构建阶段：{phase_name}（{num_steps} 步）',flush=True)
    def step_complete(self,phase_name,step):
        with self.lock:
            if time.monotonic()-self.last>=30:
                print(f'仍在构建：{phase_name} {step+1}/{self.steps.get(phase_name,"?")}',flush=True);self.last=time.monotonic()
        return True
    def phase_finish(self,phase_name):
        with self.lock:self.steps.pop(phase_name,None)

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, videos, size, count, cache):
        super().__init__()
        self.size=size; self.cache=cache; self.cuda=CUDA(); self.device=self.cuda.allocate(3*size*size*4)
        self.samples=[]; self.index=0; self.capture=None; self.current=None
        each=max(1,count//len(videos))
        for video in videos:
            cap=cv2.VideoCapture(str(video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            if total<=0:raise RuntimeError(f'Calibration video unreadable: {video}')
            for frame in np.linspace(0,total-1,min(each,total),dtype=int):
                self.samples.append((video,int(frame)))
        self.provenance={'algorithm':'TensorRT entropy calibration 2','size':size,'count':len(self.samples),
                         'videos':[{'name':p.name,'sha256':sha256(p)} for p in videos],
                         'frames':[{'video':p.name,'frame_1based':f+1} for p,f in self.samples],
                         'preprocess':'RGB square letterbox 114 /255; identical inference preprocessing'}
    def get_batch_size(self):return 1
    def get_batch(self,names):
        if self.index==len(self.samples):return None
        path,index=self.samples[self.index]
        if path!=self.current:
            if self.capture:self.capture.release()
            self.capture=cv2.VideoCapture(str(path)); self.current=path
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=self.capture.read()
        if not ok:raise RuntimeError(f'Calibration decode failed {path}:{index}')
        tensor,_=preprocess(frame,self.size)
        self.cuda.check(self.cuda.lib.cudaMemcpy(self.device,tensor.ctypes.data,tensor.nbytes,1))
        self.index+=1
        if self.index%20==0:print(f'Calibrating {self.index}/{len(self.samples)}',flush=True)
        return [self.device.value]
    def read_calibration_cache(self):return None  # Always regenerate from declared frames.
    def write_calibration_cache(self,cache):self.cache.write_bytes(cache)
    def close(self):
        if self.capture:self.capture.release()
        if self.device:self.cuda.lib.cudaFree(self.device);self.device=None

def build(mode,size,precision,videos=None,count=160,height=None,device='gpu',dla_core=0,gpu_fallback=False):
    shape_label=f'{size}x{height}' if height else str(size)
    onnx=ROOT/f'models/{mode}_yolov12n_{shape_label}.onnx'
    meta=json.loads(onnx.with_suffix('.json').read_text())
    if sha256(onnx)!=meta['onnx_sha256']:raise RuntimeError('ONNX hash mismatch')
    device_suffix=f'_dla{dla_core}' if device=='dla' else ''
    output=onnx.with_name(onnx.stem+'_'+precision+device_suffix+'.engine')
    logger=trt.Logger(trt.Logger.INFO); builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx.read_bytes()):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2048*2**20)
    config.clear_flag(trt.BuilderFlag.TF32);config.set_flag(trt.BuilderFlag.FP16)
    if device=='dla':
        if not 0<=dla_core<builder.num_DLA_cores:raise ValueError('DLA core unavailable')
        config.default_device_type=trt.DeviceType.DLA;config.DLA_core=dla_core
        if gpu_fallback:config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    cache_file=ROOT/'models/build_timing.cache'
    try:
        timing_cache=config.create_timing_cache(cache_file.read_bytes() if cache_file.exists() else b'')
        if not config.set_timing_cache(timing_cache,False):raise RuntimeError('Timing cache mismatch')
    except Exception:
        print('旧构建缓存与本机不兼容，使用空缓存。',flush=True)
        config.set_timing_cache(config.create_timing_cache(b''),False)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    monitor=Progress(); config.progress_monitor=monitor
    calibrator=None
    try:
        if precision=='int8':
            if height:raise ValueError('Rectangular INT8 calibration is not implemented')
            if not videos:raise ValueError('INT8 requires representative calibration videos')
            calibrator=Calibrator(videos,size,count,output.with_suffix('.calibration.cache'))
            config.set_flag(trt.BuilderFlag.INT8);config.int8_calibrator=calibrator
        serialized=builder.build_serialized_network(network,config)
        if serialized is None:raise RuntimeError('TensorRT engine build failed')
        output.with_suffix('.engine.tmp').write_bytes(serialized)
        output.with_suffix('.engine.tmp').replace(output)
        cache_file.write_bytes(config.get_timing_cache().serialize())
        runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized)
        inspector=engine.create_engine_inspector()
        detail=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        output.with_suffix('.layers.json').write_text(detail)
        io=[dict(name=engine.get_tensor_name(i),shape=list(engine.get_tensor_shape(engine.get_tensor_name(i))),
                 dtype=str(engine.get_tensor_dtype(engine.get_tensor_name(i))),mode=str(engine.get_tensor_mode(engine.get_tensor_name(i))))
            for i in range(engine.num_io_tensors)]
        meta.update(engine_sha256=sha256(output),tensorrt=trt.__version__,precision=precision,
                    built_at=datetime.datetime.now().isoformat(),io=io,
                    l4t=Path('/etc/nv_tegra_release').read_text(),format='raw TensorRT plan',
                    calibration=calibrator.provenance if calibrator else None,
                    device=device,dla_core=dla_core if device=='dla' else None,gpu_fallback=gpu_fallback,
                    precision_note='INT8 with FP16/FP32 fallback; inspect layers JSON' if calibrator else 'FP16 with FP32 fallback')
        output.with_suffix('.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        print('Built',output,flush=True)
    finally:
        if calibrator:calibrator.close()
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['coco','uav'],required=True);p.add_argument('--size',type=int,default=640)
    p.add_argument('--precision',choices=['fp16','int8'],default='fp16');p.add_argument('--calibration',nargs='+',type=Path);p.add_argument('--frames',type=int,default=160)
    p.add_argument('--height',type=int);p.add_argument('--device',choices=['gpu','dla'],default='gpu')
    p.add_argument('--dla-core',type=int,default=0);p.add_argument('--gpu-fallback',action='store_true')
    a=p.parse_args();build(a.mode,a.size,a.precision,a.calibration,a.frames,a.height,a.device,a.dla_core,a.gpu_fallback)
