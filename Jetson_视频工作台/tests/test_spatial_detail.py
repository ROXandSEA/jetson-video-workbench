"""Seam/border geometry, class-safe merging, cancellation and threshold behavior."""
import sys
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime import frame_ops
from runtime.spatial_detail import tile_rectangles,restore_crop_boxes,merge_views,extra_views


class DetailTests(unittest.TestCase):
    def test_every_pixel_covered_for_landscape_portrait_and_odd_sizes(self):
        for shape in [(540,827,3),(827,540,3),(5,7,3)]:
            for fraction in [.5,.6,.8]:
                pixels=np.zeros(shape[:2],bool)
                for x,y,w,h in tile_rectangles(shape,fraction):pixels[y:y+h,x:x+w]=True
                self.assertTrue(pixels.all())
        with self.assertRaises(ValueError):tile_rectangles((10,20,3),.3)

    def test_internal_seam_truncation_rejected_outer_border_preserved(self):
        boxes=np.array([[0,4,12,20,.8,0],[8,4,18,20,.7,0],[15,4,29,20,.6,0]],np.float32)
        # x=30..60: both are interior image seams, so touching x=0/x=30 drops.
        restored=restore_crop_boxes(boxes,(30,0,30,30),(40,100,3))
        np.testing.assert_array_equal(restored,np.array([[38,4,48,20,.7,0]],np.float32))
        # Same local left edge is now the original image edge, retained.
        restored=restore_crop_boxes(boxes[:1],(0,0,30,30),(40,100,3))
        np.testing.assert_array_equal(restored,boxes[:1])
        np.testing.assert_array_equal(boxes[0,:4],[0,4,12,20])

    def test_global_merge_removes_duplicates_preserves_other_classes_and_cap(self):
        first=np.array([[0,0,20,20,.9,0],[40,0,50,20,.7,0]],np.float32)
        second=np.array([[1,0,21,20,.8,0],[0,0,20,20,.85,1]],np.float32)
        merged=merge_views([first,second],frame_ops,.6,300)
        self.assertEqual(len(merged),3);self.assertEqual(merged[:,5].tolist(),[0,1,0])
        with patch.object(frame_ops,'NMS_LIB',None):reference=merge_views([first,second],frame_ops,.6,300)
        np.testing.assert_array_equal(merged,reference)
        self.assertEqual(len(merge_views([first,second],frame_ops,.6,1)),1)

    def test_cancel_prevents_extra_gpu_calls_and_preserves_full_view(self):
        cancelled=threading.Event();cancelled.set()
        full=np.array([[10,10,20,20,.8,0]],np.float32)
        rt=SimpleNamespace(frame_ops=frame_ops,postprocess_pair=lambda *a:None)
        result,calls,_=extra_views(np.zeros((80,100,3),np.uint8),full,SimpleNamespace(),
                                   {'max_det':300},rt,cancelled)
        self.assertEqual(calls,0);np.testing.assert_array_equal(result,full)


if __name__=='__main__':unittest.main()
