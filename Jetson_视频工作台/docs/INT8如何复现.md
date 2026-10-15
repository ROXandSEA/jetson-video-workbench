# INT8 实验如何复现

本包已经提供校准过的 COCO 640 INT8 引擎，网页可选“实验 · 640 INT8（实测不推荐）”。它没有达到更快且维持精度的目的，所以默认不启用；完整 AP/延迟记录见报告。

在其他 TensorRT 10.x 的 Jetson 上，要手动重建这版 INT8，先正常启动一次网页以创建 `.venv`，退出网页服务，然后在包目录执行：

```bash
PYTHONNOUSERSITE=1 .venv/bin/python tools/build_engine.py \
  --mode coco --size 640 --precision int8 --frames 160 \
  --calibration videos/MOT17/MOT17-04-DPM-raw.mp4 \
  videos/MOT17/MOT17-05-DPM-raw.mp4 videos/MOT20/MOT20-02-raw.mp4 \
  videos/VisDrone/VisDrone-val-uav0000086_00000_v-raw.mp4 \
  videos/VisDrone/VisDrone-val-uav0000117_02622_v-raw.mp4
```

这 5 段各均匀取 32 帧，预处理与推理一致，测试用 MOT20-01 和 VisDrone uav0000137 不用于校准。生成 engine、meta.json、calibration.cache、layers.json。首次构建可能需数十分钟，正常进度会打印。引擎支持混合精度回退，不能因为设置 INT8 标志就声称所有算子 INT8。

精度与延迟复测：

```bash
PYTHONNOUSERSITE=1 .venv/bin/python tools/benchmark_accuracy.py \
  --engines models/coco_yolov12n_640_fp16.engine \
  models/coco_yolov12n_1280_fp16.engine models/coco_yolov12n_640_int8.engine
PYTHONNOUSERSITE=1 .venv/bin/python tools/benchmark_latency.py
```

测时请先停止网页任务与其他 GPU 模型，不要边构建边测性能。默认没有锁频，保留每轮分布。复测结果写 reports/benchmarks；不要把 60 帧抽样 AP 当作整段或整个数据集成绩。用户换校准视频后产生的是另一版量化实验，应重新核对精度与速度。

这版结果不能证明 Jetson 的 INT8 永远不快；只能证明当前 YOLOv12n 的这套 PTQ 校准与混合精度执行不值得设为默认。后续可考虑校准分布、敏感层保留浮点或 QAT/显式 QDQ，但必须再测。无人机模型没有做对应 INT8 校准，不提供 UAV INT8 选项。
