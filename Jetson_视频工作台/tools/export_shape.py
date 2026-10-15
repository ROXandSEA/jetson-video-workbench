#!/usr/bin/env python3
"""Developer-only export using the original verified training project/PyTorch."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--mode',choices=['coco','uav'],default='coco')
    p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.project_root.resolve()/'deployment'))
    from common import load_model,sha256,WEIGHTS,torch
    from ultralytics.nn.modules.head import Detect
    import onnx
    model,names=load_model(a.mode,device='cpu')
    for m in model.modules():
        if isinstance(m,Detect):m.export=True;m.format='onnx';m.dynamic=False
    x=torch.zeros(1,3,a.height,a.width)
    label=str(a.width) if a.height==a.width else f'{a.width}x{a.height}'
    dest=ROOT/f'models/{a.mode}_yolov12n_{label}.onnx'
    with torch.inference_mode():
        y=model(x);print('Input/output',x.shape,y.shape,flush=True)
        torch.onnx.export(model,x,str(dest),opset_version=17,input_names=['images'],output_names=['output0'],dynamic_axes=None,dynamo=False)
    onnx.checker.check_model(onnx.load(str(dest)))
    info=dict(mode=a.mode,size=max(a.width,a.height),input_size=[a.height,a.width],names=names,
              weights_sha256=sha256(WEIGHTS[a.mode]),onnx_sha256=sha256(dest),opset=17,
              input_shape=list(x.shape),output_shape=list(y.shape),nms=False,input_dtype='float32',
              output_semantics='xywh + class probabilities, no objectness',
              preprocess='fixed letterbox 114, BGR to RGB, NCHW, /255')
    dest.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print('ONNX checker passed',dest,flush=True)
if __name__=='__main__':main()
