"""Concurrency, ordered output, fallback encoder, and NMS equivalence checks."""
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'app')]
from pipeline_workers import FrameProducer, VideoEncoder, VideoOutput
from runtime.backend import postprocess, postprocess_pair


class NMSReuseTests(unittest.TestCase):
    def test_two_thresholds_equal_separate_nms_with_ties_and_cap(self):
        rng = np.random.default_rng(2718)
        raw = np.zeros((1, 7, 1800), dtype=np.float32)
        raw[0, :2] = rng.uniform(-10, 650, (2, 1800))
        raw[0, 2:4] = rng.uniform(1, 140, (2, 1800))
        raw[0, 4:] = rng.choice([.001, .02, .05, .1, .3, .9], (3, 1800))
        raw[0, :, 12] = raw[0, :, 11]  # Same-box score ties.
        raw[0, 0, 0] = np.nan
        raw[0, 5, 1] = np.inf
        geometry = (1., 0, 0, 640, 640)
        for cap in (1, 30, 300):
            for track_conf, eval_conf in ((.05, .001), (.001, .05), (.05, .05)):
                tracking, evaluation = postprocess_pair(raw, geometry, 3, track_conf, eval_conf, .7, cap)
                np.testing.assert_array_equal(tracking, postprocess(raw, geometry, 3, track_conf, .7, cap))
                np.testing.assert_array_equal(evaluation, postprocess(raw, geometry, 3, eval_conf, .7, cap))

    def test_camera_preparation_preserves_flow_exactly(self):
        from tracker.gmc import GMC
        ordinary, prepared = GMC(), GMC()
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        for index in range(5):
            frame = np.roll(image, index*2, axis=1)
            first = ordinary.apply(frame)
            second = prepared.apply(frame, prepared=prepared.prepareSparseOptFlow(frame))
            np.testing.assert_array_equal(first, second)


class ProducerTests(unittest.TestCase):
    def test_finite_producer_preserves_every_frame_in_order(self):
        class Capture:
            index=0
            def read(self):
                self.index+=1
                return (True,np.full((8,8,3),self.index,np.uint8)) if self.index<=4 else (False,None)
        raw=np.zeros((1,6,1),np.float32)
        def submit(tensor):raw.fill(float(tensor[0,0,0]))
        detector=SimpleNamespace(size=8,dtype=np.float32,names=['object'],reduce_lib=True,
                                  submit=submit,synchronize=lambda:raw)
        rt=SimpleNamespace(preprocess=lambda f,s,dtype:(f,None),
                           postprocess_compact_pair=lambda r,*a:(r.copy(),r.copy()))
        cfg=dict(track_low_thresh=.05,nms_iou=.7,max_det=300)
        producer=FrameProducer(Capture(),detector,cfg,threading.Event(),rt)
        values=[]
        try:
            while True:
                item=producer.next()
                if item is None:break
                values.append((int(item[0][0,0,0]),int(item[1][0,0,0])))
        finally:producer.close()
        self.assertEqual(values,[(1,1),(2,2),(3,3),(4,4)])

    def producer(self, fail=False):
        class Capture:
            def __init__(self): self.index = 0
            def read(self):
                self.index += 1
                return True, np.full((8, 8, 3), self.index % 255, np.uint8)
        def raw(tensor):
            if fail: raise RuntimeError('inference fixture failure')
            return tensor
        rt = SimpleNamespace(preprocess=lambda f, s, dtype: (f, None),
                             postprocess_pair=lambda *a: (np.empty((0,6)),np.empty((0,6))))
        detector = SimpleNamespace(size=8,dtype=np.float32,names=['object'],
                                   submit=raw,synchronize=lambda:None)
        return FrameProducer(Capture(),detector,dict(track_low_thresh=.05,nms_iou=.7,max_det=300),threading.Event(),rt)

    def test_cancel_full_queue_never_deadlocks(self):
        producer = self.producer()
        time.sleep(.1)
        self.assertEqual(producer.queue.qsize(), 2)
        producer.close()
        self.assertFalse(producer.thread.is_alive())

    def test_producer_error_reaches_consumer(self):
        producer = self.producer(fail=True)
        try:
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                producer.next()
        finally:
            producer.close()


class OutputTests(unittest.TestCase):
    def test_hardware_bgr_input_is_aligned_then_cropped_to_original_width(self):
        writer=mock.Mock();writer.isOpened.return_value=True
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch('pipeline_workers.Path.exists',return_value=True), \
             mock.patch('cv2.getBuildInformation',return_value='GStreamer'), \
             mock.patch('cv2.VideoWriter',return_value=writer) as factory:
            encoder=VideoEncoder(Path(directory),826,540,25,{'hardware_encoder':True})
            args=factory.call_args.args
            self.assertEqual(args[-1],(828,540))
            self.assertIn('videocrop right=2',args[0])
            frame=np.full((540,826,3),89,np.uint8);encoder.write(frame)
            padded=writer.write.call_args.args[0]
            self.assertEqual(padded.shape,(540,828,3))
            np.testing.assert_array_equal(padded[:,:826],frame)
            np.testing.assert_array_equal(padded[:,-1],frame[:,-1])
            self.assertEqual(frame.shape,(540,826,3))

    def test_fallback_encodes_all_frames_and_preserves_log_order(self):
        with tempfile.TemporaryDirectory(prefix='视频输出测试 ') as directory:
            folder = Path(directory)
            output = VideoOutput(folder, 96, 64, 25, {'hardware_encoder':False}, lambda **k:None, 96, 8)
            for index in range(1, 13):
                frame = np.full((64, 96, 3), index*15, np.uint8)
                output.submit((index, frame, [], np.empty((0,6)), np.empty((0,6))))
            output.close()
            self.assertEqual(output.frames, 12)
            self.assertEqual(output.encoder_kind, 'FFmpeg libx264')
            capture = cv2.VideoCapture(str(folder/'result.mp4'))
            means = []
            while True:
                ok, frame = capture.read()
                if not ok:break
                means.append(frame.mean())
            capture.release()
            self.assertEqual(len(means), 12)
            self.assertTrue(all(a < b for a,b in zip(means,means[1:])))
            self.assertEqual(len((folder/'evaluation_detections.jsonl').read_text().splitlines()), 12)
            self.assertFalse((folder/'result.partial.mp4').exists())

    def test_encoder_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('pipeline_workers.VideoEncoder', side_effect=RuntimeError('encoder fixture failure')):
                output = VideoOutput(Path(directory),96,64,25,{},lambda **k:None,96,8)
                with self.assertRaisesRegex(RuntimeError,'encoder fixture failure'):
                    output.close()

    def test_write_failure_waits_for_encoder_cleanup(self):
        closed = threading.Event()
        class BrokenEncoder:
            kind = 'fixture'
            def write(self, frame):raise RuntimeError('write fixture failure')
            def close(self):
                time.sleep(.03)
                closed.set()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('pipeline_workers.VideoEncoder', return_value=BrokenEncoder()):
                output = VideoOutput(Path(directory),96,64,25,{},lambda **k:None,96,8)
                output.submit((1,np.zeros((64,96,3),np.uint8),[],np.empty((0,6)),np.empty((0,6))))
                with self.assertRaisesRegex(RuntimeError,'write fixture failure'):
                    output.close()
                self.assertTrue(closed.is_set())
                self.assertFalse(output.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
