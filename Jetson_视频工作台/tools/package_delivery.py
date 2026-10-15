#!/usr/bin/env python3
"""Produce a self-contained ZIP, preserving file permissions, excluding local state."""
import hashlib
import json
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.venv','.state','__pycache__'}
def files():
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(part in EXCLUDE for part in rel.parts) or p.suffix=='.pyc':continue
        if rel.parts[0]=='outputs' and len(rel.parts)>1 and rel.parts[1]!='README_结果在哪里.md':continue
        if rel.parts[0]=='reports' and any(part=='benchmarks' for part in rel.parts) and p.suffix=='.npz':continue
        if rel.parts[0]=='reports' and 'benchmarks' in rel.parts and p.suffix=='.gz' and p.stat().st_size>2**20:continue
        if p.name in {'SHA256SUMS','package.log'}:continue
        yield p

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
if __name__=='__main__':
    entries=list(files());manifest=ROOT/'SHA256SUMS'
    manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in entries))
    archive=ROOT.parent/(ROOT.name+'_可转发.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in entries+[manifest]:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
        z.writestr(ROOT.name+'/outputs/',b'')
        invalid=z.testzip()
        if invalid:raise RuntimeError('Archive CRC failed '+invalid)
    record=dict(file=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),entries=len(entries)+2,exclude=['.venv','.state','outputs/*','cache','benchmark NPZ intermediates','large raw experiment archives (retained locally)'])
    archive.with_suffix('.sha256').write_text(f"{record['sha256']}  {archive.name}\n")
    archive.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
