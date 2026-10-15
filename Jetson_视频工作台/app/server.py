"""Offline local UI. Python standard library only until a real job is started."""
import argparse
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from jobs import JobManager

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"


def parse_range(header, size):
    if not header:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or size == 0:
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        if not last or int(last) <= 0:
            raise ValueError("Invalid range")
        start, end = max(0, size - int(last)), size - 1
    else:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("Invalid range")
    return start, end, True


class Studio:
    def __init__(self, config_path, runner=None):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.jobs = JobManager(self.config, self.config_path.parent, runner=runner)
        self.token = secrets.token_urlsafe(32)
        self.sources = {}
        self.source_lock = threading.Lock()
        self.dialog_lock = threading.Lock()

    def source(self, raw):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError("请输入本机视频的绝对路径。")
        path = path.resolve()
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}:
            raise ValueError("视频不存在或格式不支持。")
        with self.source_lock:
            key = secrets.token_urlsafe(16)
            # Keep only a bounded set of authorized previews.
            if len(self.sources) >= 16:
                self.sources.pop(next(iter(self.sources)))
            self.sources[key] = path
        return dict(path=str(path), name=path.name, bytes=path.stat().st_size, url="/api/source/" + key)

    def native_dialog(self, kind):
        if not self.dialog_lock.acquire(blocking=False):
            raise RuntimeError("已有文件选择窗口打开。")
        try:
            command = [sys.executable, str(HERE / "picker.py"), kind]
            initial = str(self.jobs.resolve(self.config.get("video_library" if kind == "video" else "default_output", "")))
            if initial and Path(initial).is_dir():
                command.append(initial)
            result = subprocess.run(command,
                                    capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError("文件窗口无法打开。请确认 Jetson 有桌面会话及 python3-tk，或直接填写绝对路径。")
            path = result.stdout.decode("utf-8").strip()
            if not path:
                return dict(cancelled=True)
            return self.source(path) if kind == "video" else dict(path=path)
        finally:
            self.dialog_lock.release()

    def bootstrap(self):
        checks = []
        for mode, cfg in self.config["modes"].items():
            engine = self.jobs.resolve(cfg["engine"])
            checks.append(dict(mode=mode, exists=engine.is_file(), path=str(engine)))
        modules = {name: importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "tensorrt", "lap", "cython_bbox", "pycocotools")}
        return dict(token=self.token, state=self.jobs.snapshot(), engines=checks, dependencies=modules,
                    ffmpeg=bool(shutil.which("ffmpeg")),
                    default_output=str(self.jobs.resolve(self.config.get("default_output", "../results"))),
                    package_root=str(HERE.parent), profiles=self.config.get("profiles", {}), default_profiles=self.config.get("default_profiles", {}))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Seeking/closing a browser may cancel an HTTP keep-alive connection.
            pass

    def log_message(self, *_):
        pass

    @property
    def studio(self):
        return self.server.studio

    def headers_safe(self, mime, length=None):
        self.send_header("Content-Type", mime)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")

    def json_response(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.headers_safe("application/json; charset=utf-8", len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def local_host(self):
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def allowed_write(self):
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        return (self.local_host() and
                (origin is None or origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}) and
                secrets.compare_digest(self.headers.get("X-Studio-Token", ""), self.studio.token))

    def do_GET(self):
        if not self.local_host():
            return self.json_response(dict(error="仅允许本机访问。"), 403)
        path = urlparse(self.path).path
        try:
            if path == "/api/bootstrap":
                return self.json_response(self.studio.bootstrap())
            if path == "/api/status":
                return self.json_response(self.studio.jobs.snapshot())
            if path.startswith("/api/frame/"):
                jpeg = self.studio.jobs.frame(path.split("/")[-1])
                if jpeg is None:
                    return self.json_response(dict(error="尚无处理帧。"), 404)
                self.send_response(200)
                self.headers_safe("image/jpeg", len(jpeg))
                self.end_headers()
                return self.wfile.write(jpeg)
            if path.startswith("/api/source/"):
                with self.studio.source_lock:
                    media = self.studio.sources.get(path.split("/")[-1])
                return self.send_file(media)
            if path.startswith("/api/result/"):
                return self.send_file(self.studio.jobs.result(path.split("/")[-1]))
            assets = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css",
                      "/app.js": "app.js", "/mark.svg": "mark.svg", "/studio-font.woff2": "studio-font.woff2"}
            if path in assets:
                return self.send_file(WEB / assets[path])
            return self.json_response(dict(error="页面不存在。"), 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self.json_response(dict(error=str(exc)), 500)

    def do_POST(self):
        if not self.allowed_write():
            # Consume small unauthorized bodies before closing: Windows otherwise
            # resets the socket before the client receives the 403 response.
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if 0 < length <= 16384:
                    self.connection.settimeout(5)
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
            self.close_connection = True
            return self.json_response(dict(error="请求校验失败，请刷新本机页面。"), 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 16384:
                self.close_connection = True
                return self.json_response(dict(error="请求过大。"), 413)
            data = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/pick/video":
                result = self.studio.native_dialog("video")
            elif path == "/api/pick/output":
                result = self.studio.native_dialog("output")
            elif path == "/api/source":
                result = self.studio.source(str(data.get("path", "")))
            elif path == "/api/start":
                result = self.studio.jobs.start(str(data.get("source", "")), str(data.get("output", "")), str(data.get("mode", "")), str(data.get("annotations", "")), str(data.get("profile", "")))
            elif path == "/api/stop":
                result = self.studio.jobs.stop()
            elif path == "/api/open-output":
                directory = self.studio.jobs.output_dir
                if not directory or not directory.is_dir():
                    raise ValueError("尚无导出结果。")
                if os.name == "nt":
                    os.startfile(str(directory))
                else:
                    subprocess.Popen(["xdg-open", str(directory)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                result = dict(ok=True)
            else:
                return self.json_response(dict(error="接口不存在。"), 404)
            self.json_response(result)
        except (ValueError, KeyError, TypeError) as exc:
            self.json_response(dict(error=str(exc)), 400)
        except RuntimeError as exc:
            self.json_response(dict(error=str(exc)), 409)
        except Exception as exc:
            self.json_response(dict(error=str(exc)), 500)

    def send_file(self, path):
        if not path or not Path(path).is_file():
            return self.json_response(dict(error="文件不存在。"), 404)
        path = Path(path)
        size = path.stat().st_size
        try:
            start, end, partial = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(416)
            self.headers_safe("text/plain", 0)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        mime = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(path.suffix)
        mime = mime or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(206 if partial else 200)
        self.headers_safe(mime, max(0, end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = stream.read(min(262144, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)


def create_server(studio, port):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.studio = studio
    return server


def open_browser(url):
    browser = HERE.parent / "browser/chrome-linux-arm64/chrome"
    if browser.is_file():
        subprocess.Popen([str(browser), "--no-first-run",
                          "--user-data-dir=" + str(HERE.parent / ".state/browser-profile"), url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="TRACK / FORM 本地视频跟踪工作台")
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = create_server(Studio(args.config), args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"TRACK / FORM  {url}\nCtrl+C 退出。请在 Jetson 本机浏览器打开。", flush=True)
    if not args.no_browser:
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.studio.jobs.stop()
        if server.studio.jobs.thread:
            server.studio.jobs.thread.join(timeout=15)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
