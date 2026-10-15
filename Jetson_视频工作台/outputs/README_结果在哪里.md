# 结果在哪里

网页每启动一次任务，都会在此处新建 `时间_模式_任务编号/` 子文件夹。

- `result.mp4`：最终 H.264 跟踪视频，网页可回放。
- `tracks.csv`：目标轨迹与 ID。
- `detections.jsonl`：跟踪使用的原始检测框，门槛 0.05。
- `evaluation_detections.jsonl`：单独用于 AP 的检测框，门槛 0.001。
- `evaluation.json`：配套标注评估结果；没有标注则记录无法评估。
- `pipeline.json` / `timings.csv` / `output_timings.csv`：模型、方案、编码器、运行时长和各线程阶段耗时。
- `job.json`：输入、配置、最终状态。
- `error.log`：出错时的详细原因。

2026-10-15 起直接输出 H.264 MP4，不再生成 `tracking.avi`；旧任务的 AVI 保留。处理 FPS 包含全部输出帧编码、队列排空和 MP4 索引写入；不含模型加载、预热和 mAP 计算。线程间有重叠，阶段时间不能简单相加来计算 FPS。

停止任务时保留已处理的 CSV、检测日志和可播放的部分 `result.mp4`，不显示不完整视频的 AP。编码异常时可能保留 `result.partial.mp4`，详细原因见错误日志；它不能当作完整成功结果。

交付 ZIP 不打包本机历史输出。解压后这里为空，首次运行会生成。
