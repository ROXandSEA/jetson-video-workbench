#!/usr/bin/env python3
"""One-time workspace housekeeping; keep the live product directory in place.

Default is an inventory. --execute performs the explicitly listed operations,
checks duplicates by SHA256, and records every removed file and moved path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

PRODUCT = Path(__file__).resolve().parents[1]
ROOT = PRODUCT.parent
RECORD = PRODUCT / 'reports/validation/目录整理_20261015.json'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def probe(p):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(p)])
    return json.loads(out)['streams'][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if RECORD.exists():
        print('Already completed; audit:', RECORD)
        return
    if not args.execute:
        print('Planned: archive historical projects, remove verified transcode AVI '
              'and an extracted source ZIP, deduplicate identical large files. '
              'Live product, outputs, venv, browser profile stay in place.')
        return
    import urllib.request
    state = json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/status'))
    if state['status'] in {'loading', 'running', 'encoding', 'evaluating', 'stopping'}:
        raise RuntimeError('Wait for the active web task to finish')
    start_free = shutil.disk_usage(ROOT).free
    record = dict(date='2026-10-15', moves=[], deleted=[], deduplicated=[],
                  preserved='Live product, its venv/state/outputs; original video/GT, '
                            'unique source JPEGs, training weights/source, historical measurements')
    # Persist after each operation so interruption cannot hide destructive work.
    def save():
        RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    def remove(p, reason, **extra):
        info = dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                    sha256=digest(p), reason=reason, **extra)
        p.unlink()
        record['deleted'].append(info)
        save()

    # These files are full, sequential encodes; compare dimensions, FPS, and
    # decoded frame counts rather than assuming every historical job succeeded.
    for avi in sorted((ROOT/'Jetson_Web_Handoff/outputs').glob('*/tracking.avi')):
        mp4 = avi.with_name('result.mp4')
        if not mp4.is_file():
            continue
        a, b = probe(avi), probe(mp4)
        if a != b or int(a.get('nb_read_frames', '0')) <= 0:
            continue
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(mp4),
                        '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        remove(avi, 'MJPEG intermediate; retained H264 MP4 fully decoded with '
               'equal frame count, dimensions and playback FPS', retained_mp4=str(mp4.relative_to(ROOT)),
               retained_sha256=digest(mp4), video=b)

    # Preserve the exact original JPEGs, annotations, and cache as extracted;
    # the download archive is redundant only if EVERY member agrees.
    srczip = ROOT/'datasets/VisDrone/visdrone_validation_3_sequences.zip'
    if srczip.is_file():
        with zipfile.ZipFile(srczip) as z:
            count = 0
            for info in z.infolist():
                if info.is_dir():
                    continue
                target = srczip.parent/info.filename
                if not target.is_file() or target.stat().st_size != info.file_size:
                    raise RuntimeError(f'Unpreserved archive member: {info.filename}')
                h = hashlib.sha256()
                with z.open(info) as f:
                    for chunk in iter(lambda: f.read(2**20), b''):
                        h.update(chunk)
                if h.hexdigest() != digest(target):
                    raise RuntimeError(f'Archive member mismatch: {info.filename}')
                count += 1
        remove(srczip, 'All archive files SHA256-identical to preserved extracted files',
               verified_members=count)

    oldroot = ROOT/'开发归档/早期部署工程'
    oldroot.mkdir(parents=True, exist_ok=True)
    group = ['deployment', 'YOLOv12-BoT-SORT-ReID-main', 'COCO_multiclass',
             '.venv-jetson-uav', 'MOT_yolov12n.pt', 'SOT_yolov12l.pt',
             'model_0016.pth', 'config.yaml', 'SHA256SUMS', 'BoT-SORT-main.zip',
             'ByteTrack-main.zip', 'YOLOv12-BoT-SORT-ReID-main.zip',
             '给Jetson端AI的部署移交文档.md', 'Jetson部署检查清单.md', '部署性能报告']
    moves = [(ROOT/name, oldroot/name) for name in group]
    moves += [(ROOT/'Jetson_Web_Handoff', ROOT/'开发归档/旧网页移交版'),
              (ROOT/'Jetson_Web_Handoff.zip', ROOT/'开发归档/Jetson_Web_Handoff_原始移交.zip'),
              (ROOT/'datasets', ROOT/'开发归档/数据准备'),
              (ROOT/'精度与INT8排查', ROOT/'开发归档/精度与INT8排查')]
    for source, dest in moves:
        if not source.exists():
            continue
        if dest.exists():
            raise RuntimeError(f'Refuse to overwrite {dest}')
        source.rename(dest)
        record['moves'].append(dict(source=str(source.relative_to(ROOT)),
                                    destination=str(dest.relative_to(ROOT))))
        save()
    # Relative data references in the historical research scripts keep working.
    for name, target in [('视频结果', '../../视频结果'), ('原始视频', '../../原始视频'),
                         ('datasets', '../数据准备')]:
        (oldroot/name).symlink_to(target, target_is_directory=True)
    # The explicit .venv/bin/python already relocates; repair generated text
    # entrypoints and activate scripts, which otherwise retain the old prefix.
    oldprefix = str(ROOT/'.venv-jetson-uav').encode()
    newprefix = str(oldroot/'.venv-jetson-uav').encode()
    repaired = []
    for p in (oldroot/'.venv-jetson-uav/bin').iterdir():
        if p.is_symlink() or not p.is_file():
            continue
        raw = p.read_bytes()
        if b'\0' not in raw and oldprefix in raw:
            p.write_bytes(raw.replace(oldprefix, newprefix))
            repaired.append(str(p.relative_to(ROOT)))
    record['relocated_venv_entrypoints'] = repaired

    # Hardlink immutable duplicate inputs/results after checking every byte by
    # SHA256. Keep both user-facing paths; ZIP packaging remains self-contained.
    candidates = []
    for base in [ROOT/'原始视频', PRODUCT/'videos', ROOT/'视频结果', ROOT/'开发归档']:
        candidates.extend(p for p in base.rglob('*') if p.is_file()
                          and not p.is_symlink() and p.stat().st_size >= 2**20
                          and '.venv-jetson-uav' not in p.parts)
    sizes = {}
    for p in candidates:
        sizes.setdefault(p.stat().st_size, []).append(p)
    for size, paths in sizes.items():
        if len(paths) < 2:
            continue
        hashes = {}
        for p in paths:
            sha = digest(p)
            if sha not in hashes:
                hashes[sha] = p
                continue
            canonical = hashes[sha]
            before = p.stat()
            if before.st_ino == canonical.stat().st_ino:
                continue
            tmp = p.with_name(p.name+'.dedup-tmp')
            os.link(canonical, tmp)
            os.replace(tmp, p)
            record['deduplicated'].append(dict(path=str(p.relative_to(ROOT)),
                canonical=str(canonical.relative_to(ROOT)), sha256=sha, bytes=size,
                freed_if_last_link=before.st_nlink == 1))
    # Generated bytecode in inactive archived projects is safe to regenerate.
    for p in (ROOT/'开发归档').rglob('*.pyc'):
        if '.venv-jetson-uav' not in p.parts and p.is_file():
            remove(p, 'Regenerable Python bytecode in inactive archive')
    record['deleted_bytes'] = sum(x['bytes'] for x in record['deleted'])
    record['dedup_released_bytes'] = sum(x['bytes'] for x in record['deduplicated']
                                        if x['freed_if_last_link'])
    record['estimated_released_bytes'] = record['deleted_bytes'] + record['dedup_released_bytes']
    record['filesystem_free_delta_bytes'] = shutil.disk_usage(ROOT).free-start_free
    record['completed'] = True
    save()
    print(json.dumps({k: v for k, v in record.items()
                      if k not in {'deleted', 'deduplicated', 'moves'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
