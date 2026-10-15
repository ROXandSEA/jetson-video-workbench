#!/usr/bin/env python3
"""Developer-only report renderer. Runtime does not depend on matplotlib/markdown."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports'
data = json.loads((REPORT/'validation/performance_comparison.json').read_text())
rows = data['paired_runs']
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
labels = [r['label'] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for axis, key, title in ((axes[0],'processing_fps','视频处理速度（帧/秒，越高越好）'),
                         (axes[1],'wall_seconds','完整任务耗时（秒，越低越好）')):
    x = list(range(len(rows)))
    before = [r['before'][key] for r in rows]
    after = [r['after'][key] for r in rows]
    for shift, values, color, label in ((-.18,before,'#7a8897','优化前'),(.18,after,'#168f78','优化后')):
        bars = axis.bar([i+shift for i in x],values,.34,color=color,label=label)
        axis.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_title(title,fontsize=11)
    axis.set_ylim(0,max(before+after)*1.2)
    axis.spines[['top','right']].set_visible(False)
    axis.legend(frameon=False,fontsize=9)
fig.tight_layout()
figure = REPORT/'figures/20261015_性能优化对照.png'
fig.savefig(figure,dpi=180,bbox_inches='tight')
plt.close(fig)

table = ['| 同一原视频与输入方案 | 帧数 | 优化前 FPS | 优化后 FPS | 倍数 | 完整任务耗时（前 → 后） |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
ap = ['| 视频/方案 | AP50（前 = 后） | AP50:95（前 = 后） | 检测、评估候选、轨迹日志 |',
      '| --- | ---: | ---: | --- |']
for r in rows:
    a,b = r['before'],r['after']
    table.append(f"| {r['label']} | {a['frames']} | {a['processing_fps']:.2f} | {b['processing_fps']:.2f} | {b['processing_fps']/a['processing_fps']:.2f}× | {a['wall_seconds']:.1f} → {b['wall_seconds']:.1f} 秒 |")
    ap.append(f"| {r['label']} | {b['evaluation']['map50']*100:.2f}% | {b['evaluation']['map50_95']*100:.2f}% | 三份文件逐字节一致 |")
browser_lines = []
for record in data['browser_runs']:
    state, pipeline = record['state'],record['pipeline']
    browser_lines.append(f"| {Path(state['source_path']).name} / {state['profile']} | {pipeline['frames']} | {pipeline['processing_fps']:.2f} | 原尺寸 H.264、网页回放通过 |")
vis = rows[0]
before = vis['before']
stage_ms = before['stages_mean_ms']
paragraph = f"以 VisDrone 2688×1512、1280 FP16 为例，旧版每帧约 {sum(stage_ms):.0f} ms：读帧/预处理 {sum(stage_ms[:2]):.0f}、推理与拷贝 {stage_ms[2]:.0f}、后处理 {stage_ms[3]:.0f}、跟踪 {stage_ms[4]:.0f}、画框/日志/MJPEG/预览 {stage_ms[5]:.0f} ms。最后还要把 AVI 转成 MP4，再计算 mAP。TensorRT 加速了检测器，但没有解决 CPU 输出和串行等待。"
native = data['native_evaluation']
native_text = '；'.join(f"{r['scene']}：{r['reference_seconds']:.1f} → {r['native_seconds']:.1f} 秒" for r in native)
text = f'''# Jetson 网页性能优化报告

2026-10-15 · Orin NX 16GB · JetPack 6 / TensorRT 10.7 · 40W 动态调频

## 1. 改完后快了多少

保留原模型、FP16、输入尺寸、阈值、原始帧数和输出分辨率。默认 1280 小目标方案继续保留；没有用跳帧或降低检测输入尺寸换速度。

{chr(10).join(table)}

![同视频优化前后对照](figures/20261015_性能优化对照.png)

FPS 是实际帧数除以视频处理墙钟时间。旧版不含最终 MP4 转码，新版包含直接编码、队列排空和 MP4 索引，因此新版口径更完整。完整任务耗时另含模块/模型加载、预热、编码和 mAP，使用各自 benchmark.json 的实际墙钟记录。以上为顺序执行的单次完整视频对照；同为 40W、未锁频，没有与另一项 GPU 视频任务并行。桌面负载未隔离，速度会波动。

## 2. 原来慢在哪里，改了什么

{paragraph}

| 改动 | 具体做法 | 保留的行为 |
| --- | --- | --- |
| 硬件直接编码 | 用 NVENC 直接保存 H.264 MP4，删除 MJPEG 中间文件和二次转码；不可用时回退 FFmpeg 直接编码 | 每个输出帧仍保存，原尺寸和 FPS 不变 |
| 有界并行流水线 | 检测、BoT-SORT 关联、画框/日志/输出分线程；队列最多两帧，跟不上就等待 | 顺序、帧数、停止和错误处理 |
| CPU/GPU 重叠 | 提交异步推理后，CPU 提前读入并预处理下一帧，再同步取结果 | 同样 letterbox、RGB、归一化和模型输出 |
| 相机特征提前计算 | 原样保留灰度、缩放、角点、光流和 RANSAC 参数，只把帧内特征工作并行准备 | 相机运动补偿和跟踪算法 |
| 合并重复后处理 | 在低阈值执行一次类别内 NMS，按阈值分别提供跟踪和评估候选 | 0.05 跟踪、0.001 评估、IoU 0.7、max_det=300 |
| 加速 mAP 匹配循环 | double 精度 C 循环替代 Python 三层循环，其他步骤继续用 pycocotools | IoU 阈值、稳定排序、crowd、面积、101 点和 maxDets |

这些改动有重叠收益，不能把单项比例相加。探索记录中，VisDrone 从旧版 2.36 FPS，到直编码/并行版 9.94、提前预处理版 11.41、相机特征并行版 13.52；这是顺序开发中的单次观测，用来解释方向，不当作严格隔离的单项收益。

## 3. 精度和跟踪有没有变化

{chr(10).join(ap)}

上表覆盖全部原视频帧，不是只抽几张图。比较 detections.jsonl、evaluation_detections.jsonl 和 tracks.csv 的 SHA256 与完整内容，三份均一致。原视频 SHA256、引擎 SHA256、两份候选阈值和方案也保留在验证 JSON 中。

mAP 内核独立验证：{native_text}。AP50、AP50:95、AP75、召回率、分类 AP 与参考实现完全一致。另有随机框、并列分数、crowd、空类别/空帧、完美/错误框和面积范围等单元对照。无法加载原生库时自动回退原 pycocotools，不改变分数定义。

当前 ReID 仍关闭。BoT-SORT 负责关联 ID，TensorRT 负责检测加速；此次提速不宣称提升了模型精度或 ReID 能力。编码器改变会使结果 MP4 的压缩字节不同，逐字节一致的是检测和轨迹数据。

## 4. 网页实测与故障检查

以下为真实浏览器启动的完整任务，同时生成预览并核对播放器；与上面的独立基准分开列出。

| 素材 / 方案 | 完整帧数 | 网页处理 FPS | 验证 |
| --- | ---: | ---: | --- |
{chr(10).join(browser_lines)}

{data['unit_tests']} 项单元测试通过。真实停止任务保留可回放的部分 MP4，不发布不完整 AP；重复任务被拒绝；错误 GT 不显示假分数，但保留检测视频。所有验收 MP4 另作完整解码核对。网页开着、远程桌面和其他应用会占用 CPU/GPU/内存带宽，不能保证所有视频达到相同 FPS。

## 5. 测了但没有默认开启的选项

- CUDA Graph：1280 固定输入的隔离推理约 48.16 → 48.31 ms；完整第一版流水线 9.94 → 9.82 FPS，没有稳定收益。保留可选实现，默认关闭。
- 锁定最高频率：在原 40W 模式内临时测试，VisDrone 约 13.52 → 14.01 FPS，单次观测收益较小。测试后已恢复原动态调频，没有修改风扇和开机功耗设置。
- INT8：前次实测更慢且精度下降，继续作为明确标记的实验方案；不替换正式 FP16。
- 更大的模型、切片和更高分辨率：可能改善小目标精度，但会增加计算。需要另外做精度/速度对照，此次没有未经验证就换模型。

## 6. 怎么使用与复现

继续双击桌面“Jetson 视频工作台”，或运行包根目录“ 双击启动网页 ”。新任务自动使用新版；旧结果文件保留。

代码入口：app/jetson_runtime.py、app/pipeline_workers.py、runtime/backend.py；mAP 加速在 app/fast_coco.py、runtime/native/coco_match.c。完整导航见 docs/代码导航.md。生产运行不需要 Torch，也不需要管理员密码。

验证记录在 reports/validation/performance_comparison.json、browser_performance.json、native_coco_equivalence.json、optimization_unit_tests.log；完整对照可用 tools/benchmark_pipeline.py 复测。旧版最小源码快照在 reports/benchmarks/性能优化前源码_20261015.tar.gz，供开发者解压到单独目录后传给 --code-root；模型不重复打包。原生匹配源码、编译命令与许可证见 runtime/native/README.md。

本机对照视频单列到工作区“视频结果/性能优化_20261015/”。交付 ZIP 不塞入历史输出，使用者运行后仍写入自己的 outputs/。
'''
(REPORT/'性能优化报告_20261015.md').write_text(text)
style = "body{font-family:'Noto Sans CJK SC',sans-serif;max-width:1100px;margin:30px auto;padding:0 25px;line-height:1.65;color:#20323c}h1,h2{color:#126e61}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #c9d4d8;padding:6px}th{background:#e8f4f0}h2{break-after:avoid}table,img{break-inside:avoid}@media print{body{font-size:12px;margin:0;padding:0}@page{size:A4;margin:13mm}}"
(REPORT/'性能优化报告_20261015.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Jetson 性能优化报告</title><style>'+style+'</style>'+markdown.markdown(text,extensions=['tables'])+'</html>')
print(REPORT/'性能优化报告_20261015.md')
