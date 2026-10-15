# mAP 评估与代码说明

网页下方增加 mAP@0.50、mAP@0.50:0.95、评估帧数、有效标注目标数、评估类别和各类 AP。

默认自动读取输入视频同名 `.gt.json`，也支持手填绝对路径。原始视频库已有 12 段配套标注，MOT20-08 没有公开 GT。模型选择、素材来源和转换口径见`videos/README_素材选择.md`。

评估在视频编码后运行，状态为 `evaluating`，期间不允许换模式或重复启动，可停止；评估耗时与检测 FPS 分开记录。评估失败不会冒出假分数，也不会丢掉已经保存的视频。使用独立的 `evaluation_detections.jsonl`（置信度 ≥0.001），跟踪检测另存 `detections.jsonl`（≥0.05），不用 BoT-SORT 轨迹框或轨迹置信度。

新增代码：

- `evaluation.py`：视频/标注身份与尺寸验证、原始检测日志读取、参考 COCOeval、各类别 AP。
- `jobs.py`：传递 annotations 路径，发布 evaluation 和 source_path，加入 evaluating 活动状态。
- `jetson_runtime.py`：编码后评估、写入 pipeline.json，独立记录 evaluation_seconds。
- `web/index.html`、`app.js`、`style.css`：黑绿评估区、空白/等待/不可评估状态、视频和模式切换时清除旧分数。
- `config.json`、`server.py`、`picker.py`：选择视频窗口默认打开原始视频素材库。
- `tests/test_evaluation.py`：已知 AP、无检测、假阳性、严格 IoU、忽略区域、错误视频/尺寸和缺帧测试。
- `tests/browser_map_real.cjs`、`browser_visdrone_map.cjs`：实际 GPU + 浏览器验证，不模拟模型结果。

推理环境依赖 `pycocotools==2.0.11`，由包内离线依赖安装到 `.venv`；复用 JetPack 的 TensorRT。网页启动本身不需要加载 GPU 模型。

每次评估写 `evaluation.json`；成功时额外保存 `ground_truth.json` 标注快照和 `evaluation.log` 参考 evaluator 输出。`job.json` 含视频/配置/最终评估；`pipeline.json` 含评估结果和耗时。

## 自己的标注格式

使用 COCO bbox JSON，`images.id` 是从 1 开始的连续帧号，包含全部帧，哪怕这一帧没有目标。标注类别名必须与模型标签完全一致；类别 ID 可自定义。框使用原视频像素坐标 `[x,y,width,height]`，不是归一化坐标。

```json
{
  "info": {
    "mode": "coco",
    "video_sha256": "输入视频文件的真实 SHA256",
    "frame_count": 1,
    "description": "自己的视频；person 单类"
  },
  "images": [{"id": 1, "width": 1280, "height": 720}],
  "categories": [{"id": 0, "name": "person"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                   "bbox": [100, 200, 50, 120], "area": 6000, "iscrowd": 0}]
}
```

这里的坐标只是格式例子，不是自动生成的真实 GT。可以用 CVAT/Label Studio 等工具人工标注后转换。用 `sha256sum '/绝对路径/自己的视频.mp4'` 获取绑定哈希。标注超过 256 MiB 时请拆分视频并重新绑定。

复测命令：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

评估采用类别 argmax、按类别 NMS 0.7、检测全局最多 300 个框、COCOeval 每帧每类最多 100 个框。与 Ultralytics val 的多标签输出及官方 MOT/VisDrone 评测定义有区别；本包比较固定相同口径。只对有有效 GT 的类别取平均。
