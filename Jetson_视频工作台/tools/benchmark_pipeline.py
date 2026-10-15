#!/usr/bin/env python3
"""Run a complete original video with production thresholds/output/evaluation."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

PACKAGE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--code-root', type=Path, default=PACKAGE)
parser.add_argument('--video', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', default='coco')
parser.add_argument('--profile', default='quality')
parser.add_argument('--label', default='optimized')
parser.add_argument('--cuda-graph', choices=['on', 'off'])
parser.add_argument('--cfg-overrides', type=json.loads, default={}, help='JSON object with experimental settings')
args = parser.parse_args()
root = args.code_root.resolve()
sys.path[:0] = [str(root / 'app'), str(root)]
from jetson_runtime import process_video
config = json.loads((PACKAGE / 'app/config.json').read_text())
cfg = dict(config['modes'][args.mode])
cfg.update(config['profiles'][args.mode][args.profile])
cfg.update(args.cfg_overrides)
if args.cuda_graph is not None:
    cfg['cuda_graph'] = args.cuda_graph == 'on'
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
request = dict(source=str(args.video.resolve()), output=str(output), mode=args.mode,
               profile=args.profile, config=cfg, config_dir=str(PACKAGE / 'app'),
               annotations='', preview_fps=config.get('preview_fps', 8),
               preview_width=config.get('preview_width', 1280))
state = {}
last_print = 0
previews = 0
def publish(jpeg=None, **changes):
    global last_print, previews
    state.update(changes)
    if jpeg is not None:
        previews += 1
    now = time.monotonic()
    if now-last_print > 15 or 'status' in changes:
        print(json.dumps(state, ensure_ascii=False), flush=True)
        last_print = now
start = time.perf_counter()
result = process_video(request, publish, threading.Event())
record = dict(label=args.label, code_root=str(root), request=request, state=state,
              wall_seconds=time.perf_counter()-start, preview_count=previews, result=result)
(output / 'benchmark.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
report = json.loads((output / 'pipeline.json').read_text())
print(json.dumps(dict(label=args.label, wall_seconds=record['wall_seconds'],
                     fps=report['processing_fps'], stages_mean_ms=report['stages_mean_ms'],
                     evaluation=report.get('evaluation')), ensure_ascii=False), flush=True)
