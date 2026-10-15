#!/usr/bin/env python3
"""Physical GPU parity including invalid values and first-class score ties."""
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime.backend import CUDA,postprocess_pair,postprocess_compact_pair

def main():
    cuda=CUDA();lib=ctypes.CDLL(str(ROOT/'runtime/native/yolo_reduce.so'))
    lib.yolo_reduce.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    lib.yolo_reduce.restype=ctypes.c_int
    rng=np.random.default_rng(671);checks=0
    for nc in [1,3,80]:
        n=8400;raw=rng.uniform(0,1,(1,4+nc,n)).astype(np.float32)
        raw[0,:2]*=700;raw[0,2:4]*=150;raw[0,4:]=rng.choice([.001,.05,.3,.9],(nc,n))
        raw[0,:,8]=raw[0,:,7];raw[0,0,0]=np.nan;raw[0,4,9]=np.inf;raw[0,2,12]=0
        out=np.empty((1,6,n),np.float32)
        x=cuda.allocate(raw.nbytes);y=cuda.allocate(out.nbytes);stream=ctypes.c_void_p()
        try:
            cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(stream)))
            cuda.check(cuda.lib.cudaMemcpy(x,raw.ctypes.data,raw.nbytes,1))
            cuda.check(lib.yolo_reduce(x,y,n,nc,stream));cuda.check(cuda.lib.cudaStreamSynchronize(stream))
            cuda.check(cuda.lib.cudaMemcpy(out.ctypes.data,y,out.nbytes,2))
            for cap in [1,300]:
                for track,eval_conf in [(.05,.001),(.001,.05),(.05,.05)]:
                    args=((1.,0,0,640,640),nc,track,eval_conf,.7,cap)
                    ref=postprocess_pair(raw,*args);got=postprocess_compact_pair(out,*args)
                    for expected,actual in zip(ref,got):np.testing.assert_array_equal(expected,actual);checks+=1
        finally:
            cuda.lib.cudaFree(x);cuda.lib.cudaFree(y);cuda.lib.cudaStreamDestroy(stream)
    record=dict(passed=True,array_equal_checks=checks,classes=[1,3,80],candidates_per_case=8400,
                cases='ties, invalid coordinates, nonfinite class scores, reversed thresholds, global caps')
    (ROOT/'reports/validation/gpu_reduction_fixtures.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
if __name__=='__main__':main()
