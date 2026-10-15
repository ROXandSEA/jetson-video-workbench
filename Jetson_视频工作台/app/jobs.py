"""One bounded video-processing job at a time. No GPU imports at server startup."""
import copy
import json
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ACTIVE = {"loading", "running", "encoding", "evaluating", "stopping"}


class JobManager:
    def __init__(self, config, config_dir, runner=None):
        self.config = config
        self.config_dir = Path(config_dir).resolve()
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.runner = runner
        self.jpeg = None
        self.result_path = None
        self.output_dir = None
        self.state = self.idle_state()

    @staticmethod
    def idle_state():
        return dict(id=None, status="idle", message="选择一个视频，开始你的追踪。", mode=None,
                    frame=0, total=0, progress=None, fps=None, latency_ms=None, targets=0,
                    elapsed=0, preview_seq=0, result_url=None, output_dir=None, error=None,
                    evaluation=None, source_path=None)

    def resolve(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.config_dir / path).resolve()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def publish(self, jpeg=None, **changes):
        with self.lock:
            if jpeg is not None:
                self.jpeg = bytes(jpeg)
                self.state["preview_seq"] += 1
            # Once stop is requested, updates may refresh metrics but cannot revert its state.
            if self.cancel.is_set() and changes.get("status") in {"running", "encoding", "evaluating"}:
                changes["status"] = "stopping"
                changes["message"] = "正在停止并保存已处理内容…"
            self.state.update(changes)

    def start(self, source, output, mode, annotations="", profile=""):
        if mode not in self.config["modes"]:
            raise ValueError("请选择无人机或 COCO 模式。")
        source = Path(source).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}:
            raise ValueError("输入视频不存在或格式不支持。")
        if not output.strip():
            raise ValueError("请设置导出目录。")
        output = Path(output).expanduser()
        if not output.is_absolute():
            raise ValueError("导出目录必须是本机的绝对路径。")
        output = output.resolve()
        with self.lock:
            # Thread liveness also prevents a new run during final GPU/resource cleanup.
            if self.state["status"] in ACTIVE or (self.thread and self.thread.is_alive()):
                raise RuntimeError("当前任务尚未结束，请等待它完成。")
            mode_config = copy.deepcopy(self.config["modes"][mode])
            profiles = self.config.get("profiles", {}).get(mode, {})
            profile = profile or self.config.get("default_profiles", {}).get(mode, "balanced")
            if profiles:
                if profile not in profiles: raise ValueError("该模式不支持所选运行方案。")
                mode_config.update(profiles[profile])
            engine = self.resolve(mode_config["engine"])
            if self.runner is None and (not engine.is_file() or engine.suffix != ".engine"):
                raise ValueError("该模式的 TensorRT 引擎尚未配置。请先在 Jetson 构建 Engine，并修改 config.json。")
            output.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex
            task_dir = output / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + mode + "_" + job_id[:8])
            task_dir.mkdir(exist_ok=False)
            self.cancel = threading.Event()
            self.jpeg = None
            self.result_path = None
            self.output_dir = task_dir
            self.state = self.idle_state()
            self.state.update(id=job_id, status="loading", message="正在加载模型与跟踪器…",
                              mode=mode, profile=profile, output_dir=str(task_dir), source_path=str(source))
            request = dict(source=str(source), output=str(task_dir), mode=mode,
                           annotations=annotations, profile=profile,
                           config=mode_config, config_dir=str(self.config_dir),
                           preview_fps=self.config.get("preview_fps", 8),
                           preview_width=self.config.get("preview_width", 1280))
            self.thread = threading.Thread(target=self._run, args=(request,), daemon=True, name="tracking-worker")
            self.thread.start()
            return self.snapshot()

    def stop(self):
        with self.lock:
            if self.state["status"] in ACTIVE:
                self.cancel.set()
                self.state.update(status="stopping", message="正在停止并保存已处理内容…")
            return self.snapshot()

    def _run(self, request):
        started = time.perf_counter()
        try:
            if self.runner:
                result = self.runner(request, self.publish, self.cancel)
            else:
                from pipeline import process_video
                result = process_video(request, self.publish, self.cancel)
            with self.lock:
                result_file = result.get("result_file")
                self.result_path = Path(result_file) if result_file else None
                cancelled = self.cancel.is_set()
                self.state.update(status="cancelled" if cancelled else "completed",
                                  message="已停止，已处理内容保存在导出目录。" if cancelled else "处理完成，结果已保存。",
                                  elapsed=round(time.perf_counter() - started, 2),
                                  result_url=("/api/result/" + self.state["id"]) if self.result_path else None)
        except Exception as exc:
            traceback.print_exc()
            log = self.output_dir / "error.log"
            log.write_text(traceback.format_exc(), encoding="utf-8")
            self.publish(status="failed", message="处理未完成，请查看错误详情。", error=str(exc),
                         elapsed=round(time.perf_counter() - started, 2))
        finally:
            report = dict(request=request, state=self.snapshot())
            (self.output_dir / "job.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def frame(self, job_id):
        with self.lock:
            return self.jpeg if self.state["id"] == job_id else None

    def result(self, job_id):
        with self.lock:
            return self.result_path if self.state["id"] == job_id else None
