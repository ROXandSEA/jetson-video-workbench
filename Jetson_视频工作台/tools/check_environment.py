#!/usr/bin/env python3
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def check():
    if platform.machine()!='aarch64' or sys.version_info[:2]!=(3,10):
        raise RuntimeError('该离线包面向 JetPack 6 / aarch64 / Python 3.10 的 Jetson。')
    versions={}
    for name in ['numpy','cv2','tensorrt','scipy','lap','cython_bbox','yaml','pycocotools']:
        module=importlib.import_module(name);versions[name]=getattr(module,'__version__','installed')
    if int(versions['tensorrt'].split('.')[0])!=10:raise RuntimeError('本包要求 TensorRT 10.x，建议 JetPack 6.1/6.2；不能直接用于 TensorRT 8.x。')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):raise RuntimeError('缺少 ffmpeg/ffprobe，请执行 sudo apt install ffmpeg python3-tk')
    probe=subprocess.run(['ffmpeg','-hide_banner','-encoders'],capture_output=True,text=True)
    if 'libx264' not in probe.stdout:raise RuntimeError('ffmpeg 缺少 libx264 编码器')
    from runtime.backend import TRTDetector,preprocess,GPUResourceError
    import numpy as np
    cfg=json.loads((ROOT/'app/config.json').read_text())
    checked=set()
    for mode, profiles in cfg['profiles'].items():
        for profile,c in profiles.items():
            engine=(ROOT/'app'/c['engine']).resolve()
            if profile=='int8':
                print('INT8 为精度退化的可选实验，不在普通启动时重建；详见 reports/精度排查与交付报告.md',flush=True)
                continue
            if engine in checked:continue
            checked.add(engine)
            try:
                detector=TRTDetector(mode,engine=engine)
            except GPUResourceError:
                # Rebuilding on OOM uses still more memory and cannot repair it.
                # Preserve the valid engine and surface the actionable error.
                raise
            except Exception as exc:
                print(f'{mode}/{profile} 引擎需在本机重建：{exc}',flush=True)
                meta=json.loads(engine.with_suffix('.meta.json').read_text())
                precision=meta.get('precision','fp16')
                command=[sys.executable,str(ROOT/'tools/build_engine.py'),'--mode',mode,'--size',str(meta['size']),'--precision',precision]
                if meta['input_shape'][2]!=meta['input_shape'][3]:
                    command+=['--height',str(meta['input_shape'][2])]
                if precision=='int8':
                    command+=['--calibration']+[str(ROOT/'videos'/folder/name) for folder,name in [
                        ('MOT17','MOT17-04-DPM-raw.mp4'),('MOT17','MOT17-05-DPM-raw.mp4'),('MOT20','MOT20-02-raw.mp4'),
                        ('VisDrone','VisDrone-val-uav0000086_00000_v-raw.mp4'),('VisDrone','VisDrone-val-uav0000117_02622_v-raw.mp4')]]
                subprocess.run(command,check=True)
                detector=TRTDetector(mode,engine=engine)
            raw=detector.raw(np.zeros(detector.buffers[detector.input_name].shape,dtype=detector.dtype))
            if not np.isfinite(raw).all():raise RuntimeError('引擎输出非有限值')
            detector.close()
    sys.path.insert(0,str(ROOT/'app'))
    from fast_coco import LIB
    print(json.dumps({'platform':platform.platform(),'modules':versions,'GPU_engines':'passed',
                      'mAP_native_matching_abi':LIB.coco_match_abi() if LIB is not None else 'reference fallback',
                      'NVENC_device_present':any(Path(p).exists() for p in ['/dev/v4l2-nvenc','/dev/nvhost-msenc']),
                      'encoder_note':'实际编码器在每次任务 pipeline.json 中记录；硬件编码不可用时回退 FFmpeg'},ensure_ascii=False,indent=2))

if __name__=='__main__':check()
