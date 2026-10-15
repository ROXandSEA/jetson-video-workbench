# Jetson 视频工作台

面向 NVIDIA Jetson Orin 的视频检测与 BoT-SORT 追踪工作台，包含网页界面、TensorRT 推理、性能优化工具、评测代码与技术文档。

项目代码位于 [`Jetson_视频工作台/`](Jetson_视频工作台/)，运行包说明见 [README_先看这里.md](Jetson_视频工作台/README_先看这里.md)，源码结构见 [代码导航](Jetson_视频工作台/docs/代码导航.md)。目标运行环境为 JetPack 6.1/6.2、Ubuntu 22.04、aarch64、Python 3.10 与 TensorRT 10.x；Windows 仅用于源码管理。

## 仓库与离线运行包

此仓库保留源码、网页资源、ONNX 模型、模型元数据、配置、测试和可检查的 Markdown 文档。原有文件夹中的完整离线运行包仍保留在本机。

以下内容不纳入 Git：

- API 密钥、私有配置、账号凭据、环境文件、私钥和运行状态。
- 原始视频、数据集标注、包含本机路径的视频清单及原始验收记录。
- 随包浏览器、离线 wheel、TensorRT 引擎、编译后的原生库和启动程序。
- 日志、生成的 PDF/Word/HTML 报告、截图、历史源码压缩包及离线包校验清单。

因此，克隆仓库不会得到完整的即用离线包；部分历史报告中的图片、生成报告和原始数据链接也需要本机离线包才能打开。请先按原运行说明准备 JetPack 系统组件，并安装 `Jetson_视频工作台/environment/requirements.txt` 中的 Python 依赖。启动与编译入口保留在 `启动网页.sh`、`tools/` 和 `runtime/native/README.md` 中，模型引擎需要在兼容的 Jetson 环境中生成。克隆后可使用 `bash 启动网页.sh`，不包含原离线包的双击二进制入口。

## 隐私与提交

根目录 `.gitignore` 排除常见凭据、运行状态和本机产物。上传前已对实际提交文件执行 Gitleaks 扫描及本机路径检查。后续加入新文件或使用 `git add -f` 前仍应检查内容；忽略规则不能识别任意文件中新增的密钥。

## 来源与许可证

上游许可证见 [LICENSE-上游.txt](Jetson_视频工作台/LICENSE-上游.txt)，第三方代码、模型与素材来源见 [来源与许可证](Jetson_视频工作台/docs/来源与许可证.md)。保留原作者署名及许可证声明。
