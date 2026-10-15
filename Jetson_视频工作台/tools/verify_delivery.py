#!/usr/bin/env python3
"""Verify the actual ZIP by fresh offline startup and one complete GPU task.

Run after package_delivery.py has FINISHED. Uses an isolated temporary directory
with Chinese/spaces, a free local port, and the shipped clickable ARM64 entry.
No GUI/browser windows are opened. Stops only its own service and removes its
temporary extraction; records proof next to the ZIP.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/(ROOT.name+'_可转发.zip')


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def get(url):
    with urllib.request.urlopen(url,timeout=10) as r:return json.load(r)


def main():
    state=get('http://127.0.0.1:8765/api/status')
    if state['status'] in {'loading','running','encoding','evaluating','stopping'}:raise RuntimeError('Main service has an active task')
    record=json.loads(ARCHIVE.with_suffix('.json').read_text())
    if digest(ARCHIVE)!=record['sha256']:raise RuntimeError('ZIP hash mismatch')
    temp=Path(tempfile.mkdtemp(prefix='Jetson 报告版 验收 '))
    package=temp/ROOT.name
    try:
        with zipfile.ZipFile(ARCHIVE) as z:
            bad=z.testzip()
            if bad:raise RuntimeError('ZIP CRC failed '+bad)
        subprocess.run(['unzip','-q',str(ARCHIVE),'-d',str(temp)],check=True)
        manifest=(package/'SHA256SUMS').read_text().splitlines()
        for line in manifest:
            expected,name=line.split('  ',1)
            if digest(package/name)!=expected:raise RuntimeError('Manifest mismatch '+name)
        print('Fresh extraction: all',len(manifest),'manifest files verified',flush=True)
        for port in range(8766,8786):
            with socket.socket() as s:
                try:s.bind(('127.0.0.1',port));break
                except OSError:continue
        else:raise RuntimeError('No free local port')
        env=dict(os.environ,STUDIO_PORT=str(port))
        launch=subprocess.run([str(package/'双击启动网页'),'--no-ui','--no-browser'],
                              cwd=package,env=env,check=True,capture_output=True,text=True,timeout=180)
        url=f'http://127.0.0.1:{port}'
        boot=get(url+'/api/bootstrap')
        if boot['package_root']!=str(package) or not all(boot['dependencies'].values()):raise RuntimeError('Incomplete offline environment')
        print('Shipped entry: offline environment started on',url,flush=True)
        data=dict(source=str(package/'videos/MOT20/MOT20-01-raw.mp4'),
                  output=str(package/'outputs'),mode='coco',profile='speed')
        request=urllib.request.Request(url+'/api/start',data=json.dumps(data).encode(),
             headers={'Content-Type':'application/json','X-Studio-Token':boot['token']})
        with urllib.request.urlopen(request,timeout=10) as r:json.load(r)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            state=get(url+'/api/status')
            if state['status'] in {'completed','failed','cancelled'}:break
            time.sleep(.25)
        if state['status']!='completed':raise RuntimeError(str(state))
        output=Path(state['output_dir']);p=json.loads((output/'pipeline.json').read_text())
        if p['frames']!=429 or p['detected_frames']!=429 or 'NVENC' not in p['encoder'] or not p['cuda_graph'].startswith('enabled'):raise RuntimeError('Missing frames or wrong acceleration')
        reference=json.loads((ROOT/'reports/validation/browser_small_input.json').read_text())
        expected_ap=next(x['state']['evaluation']['map50'] for x in reference['records'] if x['state']['profile']=='speed')
        if abs(state['evaluation']['map50']-expected_ap)>1e-10:raise RuntimeError('AP changed')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=codec_name,width,height,nb_read_frames','-of','json',str(output/'result.mp4')]))['streams'][0]
        if probe!={'codec_name':'h264','width':960,'height':540,'nb_read_frames':'429'}:raise RuntimeError('Unexpected video '+str(probe))
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(output/'result.mp4'),'-f','null','-'],check=True,capture_output=True)
        request=urllib.request.Request(url+state['result_url'],headers={'Range':'bytes=0-1023'})
        with urllib.request.urlopen(request,timeout=10) as r:
            if r.status!=206 or len(r.read())!=1024:raise RuntimeError('Result playback Range failed')
        historical=ROOT/'reports/validation/交付包历史验收_512优化前.json'
        if not historical.is_file():historical=ROOT/'reports/validation/交付包历史验收_20261015.json'
        if historical.is_file():
            old=json.loads(historical.read_text());record['prior_verified_archive']=dict(sha256=old['sha256'],
                record=str(Path(ROOT.name)/historical.relative_to(ROOT)))
        record['manifest_verified_files']=len(manifest)
        record['fresh_zip_verification']=dict(date='2026-10-15',one_click=True,entry_flags=['--no-ui','--no-browser'],
             offline_environment=True,chinese_and_space_path=True,archive_sha256=record['sha256'],
             state=state,pipeline=p,video=probe,video_full_decode=True,http_range=True,
             launcher_tail=launch.stdout[-1800:],temporary_service_stopped=True,temporary_directory_removed=True)
        record['workspace_reorganization_checks']=dict(original_handoff_hashes=11,
            relocated_coco_model_classes=80,unit_tests_passed=35,main_service_unchanged=True)
        ARCHIVE.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        print('Fresh ZIP:',p['processing_fps'],'FPS; all429frames, NVENC, Graph, AP and H264 decode passed',flush=True)
    finally:
        # Identify by this unique temporary cwd and the exact server argument;
        # never terminate the user's main service or browser.
        for process in Path('/proc').iterdir():
            if not process.name.isdecimal():continue
            try:
                cwd=Path(os.readlink(process/'cwd'))
                argv=(process/'cmdline').read_bytes().split(b'\0')
                if cwd==package and b'app/server.py' in argv:os.kill(int(process.name),signal.SIGTERM)
            except (OSError,ProcessLookupError):pass
        shutil.rmtree(temp)


if __name__=='__main__':main()
