# 原始视频素材库

本文件夹只放未叠加检测框或跟踪 ID 的输入视频、配套标注及来源记录。网页的“选择输入视频”默认打开本运行包 videos 目录。

**这里三类素材都建议选择网页的“多类别 / COCO 80”模式。** MOT17/MOT20 拍的是行人，VisDrone 拍的是航拍地面的人、车、自行车等；VisDrone 不等于画面中有无人机。当前“无人机”模型检测的是无人机目标。

| 子文件夹 | 视频 | 配套 mAP 标注 |
| --- | --- | --- |
| MOT20 | 01、02、03、05、08，共 5 段 | 01/02/03/05 有；08 是测试集，没有公开 GT |
| MOT17 | 02、04、05、09、11，共 5 段 | 5 段都有；只下载各场景的一份 DPM 命名原片，不把 DPM/FRCNN/SDP 三个重复原片算三段 |
| VisDrone | uav0000086_00000_v、uav0000117_02622_v、uav0000137_00458_v，共 3 段 | 3 段都有，转换为与 COCO 兼容的类别子集 |

建议先试短片：`MOT20/MOT20-01-raw.mp4` 或 `VisDrone/VisDrone-val-uav0000137_00458_v-raw.mp4`。MOT20-03、02、05 较长、场景更密集，处理时间会较久。

操作：打开 http://127.0.0.1:8765 → 选择 COCO → 左下选本目录中的 MP4 → 选择导出目录 → 启动追踪。完整任务结束后，网页下方自动显示 mAP。

同名 `.gt.json` 是标注，例如 `MOT20-01-raw.mp4` 对应 `MOT20-01-raw.gt.json`。不要选 JSON 当视频，也不要把标注用于另一段视频。默认不需要填写网页的“配套标注”栏；手动指定时填写绝对路径。

## 素材来源与转换

- MOT20：https://motchallenge.net/data/MOT20/
- MOT17：https://motchallenge.net/data/MOT17/
- 视频来自网站的 `sequenceVideos/*-raw.mp4`，是官方未画框预览视频，通常缩小了分辨率，不是完整原始 JPEG 训练集。标注来自官方 `MOT17Labels.zip` / `MOT20Labels.zip`。
- VisDrone 官方：https://github.com/VisDrone/VisDrone-Dataset 。Google Drive 在本机网络不可达，使用公开 Kaggle 镜像 `enansari/visdrone2019-mot-custom` 的三段验证序列。镜像原名 1/2/3，通过另一份验证集镜像 `juanpedrovv/unziped` 对照标注全文和首张 JPEG，字节完全相同后恢复官方序列名。
- VisDrone 保留连续原始 JPEG 的全部帧和原尺寸，以 30 FPS、H.264 CRF18 组装为可选 MP4，没有画框。这一 FPS 是组装设置；逐帧标注按顺序对应。

`MOT_download_manifest.json`、`VisDrone_download_manifest.json` 记录 URL、尺寸、帧数、文件大小和 SHA256。`MOT_annotation_manifest.json`、`VisDrone_annotation_manifest.json` 记录标注状态。本机源JPEG和身份校验记录保存在工作区 `开发归档/数据准备/VisDrone/`，MOT官方GT保存在工作区 `原始视频/标注/`。本独立包已包含各视频转换后的GT和来源清单，日常运行无需归档数据。

## 分数怎么读

网页显示当前完整视频的原始检测框 AP，使用参考 `pycocotools`：101 点 recall，IoU 0.50:0.05:0.95，评估每帧每类最多 100 个候选框，检测全局最多 300 个；跟踪置信度 ≥0.05，评估单独保留置信度 ≥0.001 的检测。mAP@0.50 条件宽松，mAP@0.50:0.95 对框的位置更严格。

MOT 只标注 person，因此显示 **person 单类 AP**，不能称为 COCO 80 类的整体 mAP。官方缩放预览视频的帧数与 FPS 必须匹配 seqinfo，坐标由原尺寸同步缩放；MOT 一基坐标转换为零基坐标。未计分行人及 person-on-vehicle/static-person/distractor/reflection 映射为 COCO crowd 忽略区域。该口径不是 MOT 榜单的 MOTA/IDF1。

VisDrone 合并 pedestrian/people → person、car/van → car；其余对应 bicycle、motorcycle、bus、truck。未兼容的三轮车、其他类别和 score=0 区域在评估类别内作为 crowd 忽略区域。结果是 **映射后的兼容 COCO 类别子集 AP**，不是官方 VisDrone 全类别榜单分数。当前视频没有有效标注的类别会显示“—”，不参与 AP 平均。

没有标注显示“无法评估”；标注错误、视频 SHA256/帧数/尺寸/类别不对应显示“标注不适用”，分数保持空白。裁剪、缩放或改编码会改变视频哈希，需要重新生成配套标注。切换视频或模式会清空旧分数。

mAP 不衡量跟踪 ID 是否稳定；提高置信度也不等于提高准确率。

另外 UAV/MultiUAV-003.mp4 用无人机模式，没有配套人工检测标注，因此不显示 mAP。
