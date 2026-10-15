#!/usr/bin/python3
"""Clickable entry: system Python only, detached server, relocatable package."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.state'
LOG = STATE / 'service.log'


def bootstrap(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/bootstrap', timeout=0.7) as response:
            return json.load(response)
    except Exception:
        return None


def choose_port():
    requested = os.environ.get('STUDIO_PORT')
    if requested:
        ports = [int(requested)]
    else:
        try:
            previous = int((STATE / 'gui_port').read_text())
        except (OSError, ValueError):
            previous = 8765
        ports = list(dict.fromkeys([previous, *range(8765, 8786)]))
    for port in ports:
        data = bootstrap(port)
        if data and data.get('package_root') == str(ROOT):
            return port, True
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
                return port, False
            except OSError:
                pass
    raise RuntimeError('网页端口均已占用，请关闭另一份工作台后重试。')


def open_browser(url):
    browser = ROOT / 'browser/chrome-linux-arm64/chrome'
    if browser.is_file():
        browser.chmod(browser.stat().st_mode | 0o111)
        subprocess.Popen([str(browser), '--no-first-run',
                          '--user-data-dir=' + str(STATE / 'browser-profile'), url],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    else:
        webbrowser.open(url)


def launch(progress, browser=True):
    STATE.mkdir(exist_ok=True)
    # Serialize overlapping clicks, including the first offline environment setup.
    with (STATE / 'launcher.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                progress('另一启动窗口正在准备环境，请稍候…')
                time.sleep(0.5)
        port, existing = choose_port()
        url = f'http://127.0.0.1:{port}'
        if not existing:
            progress('正在配置离线依赖并检查 TensorRT 环境…')
            env = dict(os.environ, STUDIO_PORT=str(port), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
            with LOG.open('w') as log:
                process = subprocess.Popen(['/bin/bash', str(ROOT / '启动网页.sh'), '--no-browser'],
                                           cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                data = bootstrap(port)
                if data and data.get('package_root') == str(ROOT):
                    break
                if process.poll() is not None:
                    raise RuntimeError(f'启动失败（退出码 {process.returncode}）。请点击“查看日志”。')
                try:
                    # Read only the tail; engine build logs can grow substantially.
                    with LOG.open('rb') as log:
                        log.seek(max(0, LOG.stat().st_size - 3000))
                        lines = log.read().decode('utf-8', errors='replace').splitlines()
                    if lines:
                        progress(lines[-1][-160:])
                except OSError:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError('准备时间超过 1 小时。后台进程仍在运行，请查看日志了解进度。')
        (STATE / 'gui_port').write_text(str(port))
        if browser:
            progress('网页已就绪，正在打开浏览器…')
            open_browser(url)
        return dict(url=url, reused=existing, package_root=str(ROOT), log=str(LOG))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-ui', action='store_true', help='For automated verification')
    parser.add_argument('--no-browser', action='store_true', help='For automated verification')
    args = parser.parse_args()
    if args.no_ui:
        try:
            result = launch(lambda line: print(line, flush=True), not args.no_browser)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 0
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        subprocess.run(['zenity', '--error', '--text=缺少图形启动组件 python3-tk，请先由设备管理员安装。'])
        return 1
    window = tk.Tk()
    window.title('Jetson 视频工作台 · 一键启动')
    window.geometry('620x250')
    window.resizable(False, False)
    ttk.Label(window, text='正在启动视频工作台', font=('', 17)).pack(pady=(24, 12))
    status = tk.StringVar(value='首次启动会自动安装随包依赖，请稍候。')
    ttk.Label(window, textvariable=status, wraplength=565).pack(padx=20, pady=8)
    bar = ttk.Progressbar(window, mode='indeterminate', length=540)
    bar.pack(pady=12)
    bar.start()
    ttk.Label(window, text='服务在后台运行。网页就绪后会自动打开。').pack()
    actions = ttk.Frame(window)
    actions.pack(pady=12)
    ttk.Button(actions, text='查看日志', command=lambda: subprocess.Popen(
        ['xdg-open', str(LOG)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)).pack(side='left', padx=6)
    ttk.Button(actions, text='关闭此窗口', command=window.destroy).pack(side='left', padx=6)
    events = queue.Queue()

    def worker():
        try:
            result = launch(lambda line: events.put(('progress', line)), not args.no_browser)
            events.put(('done', result))
        except Exception as exc:
            try:
                with LOG.open('a') as log:
                    log.write('\n启动器错误：' + str(exc) + '\n')
            except OSError:
                pass
            events.put(('error', str(exc)))

    # Non-daemon: closing the progress window does not interrupt first-time setup.
    threading.Thread(target=worker, daemon=False).start()

    def poll():
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == 'done':
                    window.destroy()
                    return
                status.set(value)
                if kind == 'error':
                    bar.stop()
        except queue.Empty:
            pass
        window.after(150, poll)

    window.after(150, poll)
    window.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
