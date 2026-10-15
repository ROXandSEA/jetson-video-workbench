"""Independent fallback comparison for native packing and channel reduction."""
import sys
from pathlib import Path
import unittest
import json
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime import backend,frame_ops

class NativeFrameTests(unittest.TestCase):
    def test_detection_json_preserves_float64_values_and_resizes_buffer(self):
        if frame_ops.LIB is None:self.skipTest('Native library unavailable')
        rng=np.random.default_rng(187)
        rows=rng.integers(0,2**32,(300,6),dtype=np.uint32).view(np.float32)
        rows[~np.isfinite(rows)]=0
        codec=frame_ops.DetectionJSON()
        for values in [rows[:0],rows[:1],rows,rows[:2]]:
            raw=codec.encode(429,values)
            self.assertEqual(json.loads(raw),{'frame':429,'detections':values.tolist()})

    def setUp(self):
        if frame_ops.LIB is None:self.skipTest('Native library unavailable')
    def test_packing_matches_numpy_for_portrait_landscape_and_rectangular_inputs(self):
        rng=np.random.default_rng(472)
        for h,w in [(90,160),(160,90),(23,29)]:
            frame=rng.integers(0,256,(h,w,3),np.uint8)
            for size in [320,640,(384,640),(320,512)]:
                with patch.object(frame_ops,'LIB',None):reference,geometry=backend.preprocess(frame,size)
                actual,other=backend.preprocess(frame,size)
                np.testing.assert_array_equal(reference,actual);self.assertEqual(geometry,other)
    def test_nms_matches_reference_including_ties_nonfinite_and_thresholds(self):
        rng=np.random.default_rng(914)
        for nc in [1,3,80]:
            raw=rng.uniform(0,1,(1,4+nc,1500)).astype(np.float32)
            raw[0,:2]*=700;raw[0,2:4]*=120
            raw[0,4:]=rng.choice([.001,.05,.3,.9],(nc,1500))
            raw[0,:,20]=raw[0,:,19]
            raw[0,0,0]=np.nan;raw[0,4,-1]=np.inf;raw[0,2,7]=0
            for threshold in [.001,.05,.3]:
                for cap in [1,300]:
                    args=(raw,(1.,0,0,640,640),nc,threshold,.7,cap)
                    with patch.object(frame_ops,'LIB',None),patch.object(frame_ops,'NMS_LIB',None):ref=backend.postprocess(*args)
                    np.testing.assert_array_equal(ref,backend.postprocess(*args))
    def test_missing_library_uses_reference(self):
        frame=np.zeros((40,60,3),np.uint8)
        with patch.object(frame_ops,'LIB',None):
            tensor,_=backend.preprocess(frame,(32,64))
            self.assertEqual(tensor.shape,(1,3,32,64))
if __name__=='__main__':unittest.main()
