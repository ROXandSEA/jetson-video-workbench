# GitHub 源码版部署指南

本指南针对从 GitHub 克隆的源码与 ONNX 模型。完整离线包的双击入口、随包浏览器、Python wheel、TensorRT 引擎和原生 `.so` 不在仓库中，因此源码版采用在线安装依赖、在 Jetson 上构建引擎、直接启动 Python 服务的流程。

以下命令根据当前源码接口整理，需要在目标 Jetson 上执行；硬件推理、编码和性能应以实际设备验证结果为准。

## 1. 准备 Jetson 系统环境

目标环境为 Ubuntu 22.04、aarch64、Python 3.10、JetPack 6.1/6.2 和 TensorRT 10.x。先完成 JetPack 系统安装，确保系统 Python 可以使用 CUDA/TensorRT 相关组件。

安装辅助系统工具：

```bash
sudo apt update
sudo apt install git python3-venv python3-dev python3-tk python3-opencv \
  ffmpeg build-essential
```

这里的 apt 命令不包含 JetPack、CUDA 或 TensorRT 的安装。检查系统 Python 与编码工具：

```bash
uname -m
/usr/bin/python3 --version
/usr/bin/python3 -c 'import tensorrt, cv2; print("TensorRT:", tensorrt.__version__); print("OpenCV:", cv2.__version__)'
ffmpeg -hide_banner -encoders
```

架构应为 `aarch64`，Python 应为 3.10，TensorRT 主版本应为 10，FFmpeg 编码器列表应包含 `libx264`。如果系统 Python 无法导入 TensorRT，应先修复 JetPack 的 Python 绑定。

## 2. 克隆源码并创建 Python 环境

```bash
git clone https://github.com/ROXandSEA/jetson-video-workbench.git
cd jetson-video-workbench/Jetson_视频工作台
/usr/bin/python3 -m venv --system-site-packages .venv
```

`--system-site-packages` 用于复用 JetPack 安装的 TensorRT 和系统 OpenCV。以下命令使用 `PYTHONNOUSERSITE=1`，避免个人 Python 包目录干扰环境：

```bash
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --ignore-installed \
  'numpy==1.26.4' 'Cython==3.2.4'
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install --no-build-isolation \
  -r environment/requirements.txt
```

在线安装需要可访问的 Python 包源；若无匹配的 aarch64 wheel，部分依赖会在本机编译。版本以 [requirements.txt](../environment/requirements.txt) 为准。不要用其他平台的 wheel，也不要为解决依赖问题随意用 pip 版 OpenCV 覆盖系统 OpenCV，否则可能失去 GStreamer 编码支持。

## 3. 检查环境并生成 FP16 引擎

在上述项目目录执行：

```bash
PYTHONNOUSERSITE=1 .venv/bin/python tools/check_environment.py
```

脚本检查架构、Python、依赖、TensorRT、FFmpeg 和配置中的非 INT8 引擎。引擎缺失或不兼容时，会根据已提交的 ONNX 和元数据调用构建工具生成本机 FP16 引擎，并执行一次输出检查。共检查五个不同的 FP16 引擎，第一次可能需要较长时间；发生 GPU 内存不足时会报错，不会把它当成版本不兼容反复重建。

需要单独构建某个模型时，例如：

```bash
# COCO 默认 1280 方形模型
PYTHONNOUSERSITE=1 .venv/bin/python tools/build_engine.py \
  --mode coco --size 1280 --precision fp16

# COCO 640×384 横向模型：size 表示宽度，height 表示高度
PYTHONNOUSERSITE=1 .venv/bin/python tools/build_engine.py \
  --mode coco --size 640 --height 384 --precision fp16

# 无人机 640 方形模型
PYTHONNOUSERSITE=1 .venv/bin/python tools/build_engine.py \
  --mode uav --size 640 --precision fp16
```

构建产物写入 `models/`。引擎、缓存与编译产物已被 Git 忽略；构建工具也可能更新已跟踪的模型元数据，提交前应检查 `git diff`。更换 ONNX 时必须同步更新对应的 JSON 元数据与 SHA256，构建工具会校验模型哈希。

INT8 不在普通环境检查时自动构建。此实验需要自备校准视频，详情见 [INT8 如何复现](INT8如何复现.md)。

## 4. 可选：编译原生加速库

这些库缺失时，程序使用参考实现。若需要复现报告中的性能，应在目标设备编译，并检查实际任务的 `pipeline.json`，确认相应加速路径启用。

在项目目录执行 CPU 内核编译：

```bash
gcc -O3 -std=c11 -fPIC -shared runtime/native/frame_ops.c \
  -o runtime/native/frame_ops.so
g++ -O3 -std=c++14 -ffp-contract=off -fPIC -shared runtime/native/nms_bridge.cpp \
  -o runtime/native/nms_bridge.so
gcc -O3 -fPIC -shared runtime/native/coco_match.c \
  -o runtime/native/coco_match.so
```

安装了 nvcc 的 Orin 设备可另外编译 GPU 类别归约内核：

```bash
nvcc -O3 -std=c++14 -arch=sm_87 --shared -Xcompiler=-fPIC \
  runtime/native/yolo_reduce.cu -o runtime/native/yolo_reduce.so
```

`sm_87` 对应这里的 Orin 目标。不要添加 `fast-math`，也不要在正在运行的任务中覆盖动态库。实现、回退条件和第三方许可见 [原生内核说明](../runtime/native/README.md)。

## 5. 启动本机网页

```bash
PYTHONNOUSERSITE=1 .venv/bin/python app/server.py --port 8765
```

在 Jetson 本机浏览器访问 <http://127.0.0.1:8765>。源码版没有随包浏览器，服务会尝试打开系统默认浏览器，也可以手动访问。

指定其他端口，或只启动服务而不自动打开浏览器：

```bash
PYTHONNOUSERSITE=1 .venv/bin/python app/server.py --port 8766 --no-browser
```

服务面向本机访问；不要把远程电脑的 `127.0.0.1` 当成 Jetson 地址。终端直接启动的服务可用 `Ctrl+C` 退出。`STUDIO_PORT` 属于离线启动脚本的参数方式，直接启动 `app/server.py` 时使用 `--port`。

**只有克隆的源码时，不直接执行 `bash 启动网页.sh`。** 该脚本使用 `--no-index --find-links environment/wheels` 安装离线依赖，并对随包 Chrome 文件执行权限检查；这些文件没有发布到 GitHub。完整离线包的用法保留在 [原运行包说明](../README_先看这里.md)。

## 6. 准备视频与人工标注

自备原始视频，可以放在 `videos/`，也可以通过文件选择器打开本机其他目录。网页接受 MP4、AVI、MKV、MOV、M4V、WebM 文件，是否能解码取决于系统 OpenCV/FFmpeg。

- MOT17/MOT20 行人视频与 VisDrone 地面人车视频使用 COCO 模式。
- 无人机模式用于检测画面中的无人机目标。
- 配套标注采用同名 `.gt.json`，需包含所有帧、原视频尺寸、类别与 SHA256；见 [标注格式](mAP标注格式.md)。
- 没有人工标注仍可追踪和导出视频，但不能计算有效 AP。不要将其他视频的 GT 直接改名复用。

源码仓库只包含素材来源说明与下载清单，不包含原始视频和 GT。请遵守 [素材说明](../videos/README_素材选择.md) 中的数据来源与使用条款。

## 7. 验证与排查

在依赖准备完成后，可运行现有单元测试：

```bash
PYTHONNOUSERSITE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

单元测试不能代替真实 GPU、编码器和网页验证。`tests/validate_gpu_reduction.py` 用于 GPU 路径对照；`tests/browser_*.cjs` 属于浏览器开发验证，需要另外准备相应 JavaScript 环境与素材。

| 现象 | 检查方向 |
| --- | --- |
| 找不到 TensorRT / CUDA | 确认 JetPack 已安装对应系统组件，系统 Python 可导入 TensorRT，虚拟环境启用了系统包 |
| 找不到引擎或反序列化失败 | 使用 `tools/check_environment.py` 在目标 Jetson 上检查与构建；不要跨版本直接复制引擎 |
| 第一次检查耗时较长 | 正在构建多个 TensorRT 引擎；观察构建进度，避免同时启动其他 GPU 任务 |
| GPU 内存不足 | 先结束其他 GPU 任务并释放内存，再重试；不要反复构建引擎来处理资源不足 |
| 编码回退到 libx264 | 检查硬件编码器设备、GStreamer 插件和系统 OpenCV 支持；`pipeline.json` 记录实际编码器 |
| 速度明显低于报告 | 核对输入尺寸、目标密度、功耗/频率、桌面负载、编码器及是否编译原生内核 |
| 网页没有 AP | 核对 GT 是否存在，以及视频哈希、全部帧、尺寸和标签是否匹配 |
| 端口已占用 | 更换 `--port`，并在浏览器访问相同端口 |

运行结果默认保存在 `outputs/` 的独立任务子目录中，详见 [结果说明](../outputs/README_结果在哪里.md)。分享日志或结果前，请检查视频内容、本机路径和个人信息。
