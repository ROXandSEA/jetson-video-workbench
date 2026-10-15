#!/usr/bin/env python3
"""Remove only named, regenerable experiment duplicates after archiving evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1]

def size(path):
    items=[path] if path.is_file() else path.rglob('*')
    return sum(p.lstat().st_blocks*512 for p in items if p.is_file() and not p.is_symlink())

def busy(path):
    prefix=str(path.resolve())+'/'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            cwd=str((proc/'cwd').resolve());cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if cwd+'/'==prefix or cwd.startswith(prefix) or prefix in cmd:return int(proc.name)
            for fd in (proc/'fd').iterdir():
                target=os.readlink(fd)
                if target.startswith(prefix):return int(proc.name)
        except (OSError,PermissionError):pass
    return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--include-ceiling',action='store_true');a=p.parse_args()
    targets=[ROOT/'.state/old-venv-1791987425',ROOT/'.state/performance_runs',
             ROOT/'.state/performance_browser_outputs',Path('/tmp/Jetson 加速版ZIP验收')]
    if a.include_ceiling:targets += [ROOT/'.state/ceiling_runs',ROOT/'.state/ceiling_browser_outputs',ROOT/'.state/performance_baseline_20261015']
    entries=[dict(path=str(t),allocated_bytes=size(t),busy_pid=busy(t)) for t in targets if t.exists()]
    if a.execute:
        archive=ROOT/'reports/benchmarks'/('百帧实验记录_'+str(time.time_ns())+'.tar.gz')
        with tarfile.open(archive,'w:gz') as out:
            for folder in targets:
                if not str(folder).startswith(str(ROOT)):continue
                for f in folder.rglob('*') if folder.exists() else []:
                    if f.is_file() and f.suffix in {'.json','.jsonl','.csv','.log'} and 'venv' not in str(f):
                        out.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
        for item in entries:
            if item['busy_pid'] is not None:item['result']='skipped: active process';continue
            path=Path(item['path'])
            if path.is_symlink():raise RuntimeError('Refusing a symlink target')
            shutil.rmtree(path);item['result']='removed'
    report=dict(executed=a.execute,at=time.strftime('%Y-%m-%d %H:%M:%S'),entries=entries,
                released_allocated_bytes=sum(e['allocated_bytes'] for e in entries if e.get('result')=='removed'),
                retained='原始视频、标注、模型、当前环境、正式 outputs、单列结果视频；历史实验 JSON/CSV/log 已归档')
    if a.execute:
        previous_path=ROOT/'reports/validation/cleanup.json'
        previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
        report['entries']=previous.get('entries',[])+report['entries']
        report['released_allocated_bytes']+=previous.get('released_allocated_bytes',0)
        report['evidence_archives']=previous.get('evidence_archives',['reports/benchmarks/旧性能实验记录_20261015.tar.gz'])+[str(archive.relative_to(ROOT))]
        previous_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
