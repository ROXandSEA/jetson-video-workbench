"""Paired ground-truth audit: identical held-out frames, preprocessing and NMS."""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

DEPLOY = Path(__file__).resolve().parents[1]
ROOT = DEPLOY.parent
sys.path.insert(0, str(DEPLOY))
sys.path.insert(0, str(ROOT / 'Jetson_Web_Handoff/Jetson_UI'))
from common import TorchDetector, TRTDetector, preprocess, postprocess, sha256
from evaluation import evaluate_detections

VIDEOS = [ROOT / '原始视频/MOT20/MOT20-01-raw.mp4',
          ROOT / '原始视频/VisDrone/VisDrone-val-uav0000137_00458_v-raw.mp4']


def audit(video, backend, size, count, confidences, engine=None):
    gt = json.loads(video.with_suffix('.gt.json').read_text())
    frame_count = gt['info']['frame_count']
    selected = np.linspace(1, frame_count, min(count, frame_count), dtype=int).tolist()
    wanted = set(selected)
    gt['images'] = [image for image in gt['images'] if image['id'] in wanted]
    gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] in wanted]
    names = json.loads((DEPLOY / 'artifacts/coco_yolov12n_640_fp16.meta.json').read_text())['names']
    mapping = {int(k): c['id'] for k, value in names.items() for c in gt['categories'] if c['name'] == value}
    detector = TorchDetector('coco', size) if backend == 'torch' else TRTDetector('coco', size, engine)
    blank = torch.zeros(1, 3, size, size, device='cuda', dtype=detector.dtype)
    for _ in range(30):
        detector.raw(blank)
    torch.cuda.synchronize()
    capture = cv2.VideoCapture(str(video))
    predictions = {threshold: [] for threshold in confidences}
    timings = []
    numerical = None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index not in wanted:
            continue
        started = time.perf_counter()
        tensor, geometry = preprocess(frame, size, dtype=detector.dtype)
        raw = detector.raw(tensor)
        torch.cuda.synchronize()
        inference_done = time.perf_counter()
        if numerical is None:
            numerical = raw.detach().cpu().numpy().copy()
        for confidence in confidences:
            detections = postprocess(raw, geometry, 80, confidence, .7, max_det=300)
            for x1, y1, x2, y2, score, category in detections:
                cls = int(category)
                if cls in mapping:
                    predictions[confidence].append({'image_id': index, 'category_id': mapping[cls],
                                                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                                    'score': float(score)})
        timings.append((inference_done - started) * 1000)
    capture.release()
    output = ROOT / '精度与INT8排查'
    results = []
    for confidence, predictions_list in predictions.items():
        label = f'{video.stem}_{backend}_{size}_conf{confidence}'
        with (output / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log):
            metrics = evaluate_detections(gt, predictions_list)
        # Describe tiny objects in network pixels, not only native video pixels.
        ratios = {i['id']: min(size / i['width'], size / i['height']) for i in gt['images']}
        shortest = [min(a['bbox'][2:]) * ratios[a['image_id']] for a in gt['annotations'] if not a.get('iscrowd', 0)]
        record = {'video': str(video.relative_to(ROOT)), 'video_sha256': sha256(video),
                  'backend': backend, 'size': size, 'confidence': confidence, 'iou_nms': .7,
                  'max_detection_global': 300, 'AP_max_detections_per_image_per_class': 100,
                  'sampling': 'uniform deterministic subset, not full-video AP', 'frame_ids': selected,
                  'metrics': metrics, 'preprocess_and_inference_mean_ms': float(np.mean(timings)),
                  'object_short_side_median_network_pixels': float(np.median(shortest)),
                  'objects_short_side_under_8px_fraction': float(np.mean(np.asarray(shortest) < 8)),
                  'prediction_count': len(predictions_list)}
        (output / (label + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(label, metrics['map50'], metrics['map50_95'], flush=True)
        results.append(record)
    # Store only a representative raw tensor for a paired numerical backend comparison.
    np.savez_compressed(output / f'{video.stem}_{backend}_{size}_raw_sample.npz', raw=numerical)
    del detector, blank, raw, tensor
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['torch', 'trt'], required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--engine')
    parser.add_argument('--confidence', nargs='+', type=float, default=[.05, .001])
    args = parser.parse_args()
    results = []
    for source in VIDEOS:
        results.extend(audit(source, args.backend, args.size, args.frames, args.confidence, args.engine))
    name = f'audit_{args.backend}_{args.size}' + ('_' + Path(args.engine).stem if args.engine else '')
    (ROOT / '精度与INT8排查' / (name + '.json')).write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
