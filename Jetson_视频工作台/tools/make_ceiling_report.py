#!/usr/bin/env python3
"""Render the measured web/whole-pipeline/DLA comparison, with real figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports'

def main():
    data=json.loads((REPORT/'validation/ceiling_comparison.json').read_text())
    rows=data['web_mot20'];before=data['web_before_tuning']
    plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
    key=['balanced','fast_static','speed'];names=['640 标准','640×384 固定机位','512×320 极速']
    final={r['state']['profile']:r for r in rows}
    old={r['state']['profile']:r for r in before if Path(r['state']['source_path']).name=='MOT20-01-raw.mp4'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for delta,records,color,label in [(-.18,old,'#818996','网页调度优化前'),(.18,final,'#168b70','网页调度优化后')]:
        bars=axes[0].bar([i+delta for i in range(3)],[records[k]['pipeline']['processing_fps'] for k in key],.36,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    axes[0].axhline(100,color='#be5e36',linestyle='--',linewidth=1,label='100 FPS 目标')
    axes[0].set_xticks(range(3),names,fontsize=9);axes[0].set_title('真实网页：每帧检测、跟踪、原尺寸保存');axes[0].set_ylabel('帧/秒');axes[0].legend(fontsize=8,frameon=False)
    for k,name in zip(key,names):
        r=final[k]['pipeline'];x=r['processing_fps'];y=r['evaluation']['map50']*100
        axes[1].scatter(x,y,s=85,color='#168b70');axes[1].annotate(f'{name}\n{y:.2f}%',(x,y),xytext=(4,8),textcoords='offset points',fontsize=9)
    axes[1].set_xlabel('真实网页处理 FPS');axes[1].set_ylabel('MOT20-01 mAP@50 (%)');axes[1].set_title('速度与检测准确率的取舍');axes[1].set_ylim(45,65)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(REPORT/'figures/20261015_百帧网页与精度.png',dpi=180,bbox_inches='tight');plt.close(fig)
    latency=data['dla']['latency']['runs'];gpu=sum(r['mean_ms'] for r in latency if r['label'].startswith('gpu_'))/2;dla=next(r['mean_ms'] for r in latency if r['label']=='dla_gpu')
    fig,ax=plt.subplots(figsize=(6.5,2.7));bars=ax.barh(['GPU FP16','DLA + GPU 回退'],[gpu,dla],color=['#168b70','#818996'])
    ax.bar_label(bars,fmt='%.1f ms',padding=5);ax.set_xlim(0,dla*1.2);ax.set_xlabel('640 方形模型：隔离推理及传输耗时，越低越好');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(REPORT/'figures/20261015_DLA实测.png',dpi=180,bbox_inches='tight');plt.close(fig)
    table=['| 网页方案 | 全部帧 | 处理 FPS | 总任务耗时 | mAP@50 | mAP@50:95 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    labels={'quality':'1280 小目标优先','balanced':'640 标准','fast_motion':'640×384 移动镜头','fast_static':'640×384 固定机位','speed':'512×320 极速固定机位'}
    for item in rows:
        r=item['pipeline'];e=r['evaluation'];table.append(f"| {labels[item['state']['profile']]} | {r['frames']} | {r['processing_fps']:.2f} | {item['state']['elapsed']:.2f} 秒 | {e['map50']*100:.2f}% | {e['map50_95']*100:.2f}% |")
    tracking=['| 辅助跟踪对照 | IDF1 | ID 切换次数 |','| --- | ---: | ---: |']
    trlabels={'gpu_balanced':'640 标准','rect640_motion':'640×384 轻量 GMC','rect640_graph':'640×384 固定机位','rect512_graph':'512×320 固定机位'}
    for item in data['tracking']['runs']:
        m=item['metrics'];tracking.append(f"| {trlabels[item['run']]} | {m['idf1']*100:.2f}% | {int(m['num_switches'])} |")
    fastest=final['speed']['pipeline'];fixed=final['fast_static']['pipeline'];standard=final['balanced']['pipeline']
    other=data['other_web_videos'];other_text='；'.join(f"{Path(r['state']['source_path']).name} / {r['state']['profile']}：{r['pipeline']['processing_fps']:.2f} FPS" for r in other)
    released=data['cleanup']['released_allocated_bytes']/1e9
    text=f'''# Jetson 百帧优化与 DLA 验证报告

2026-10-15 · Orin NX 16GB · JetPack 6 / TensorRT 10.7 / CUDA 12.6 · 40W

## 1. 实际达到多少

同一段 MOT20-01 原始视频，全部 429 帧，真实网页预览、BoT-SORT 跟踪和原尺寸 H.264 保存全部开启：**固定机位 640×384 为 {fixed['processing_fps']:.2f} FPS，极速 512×320 为 {fastest['processing_fps']:.2f} FPS**。没有跳帧，也没有把纯模型推理速度当成网页速度。较快方案已作为网页可选项，原来的 1280 和 640 标准方案保留。

{chr(10).join(table)}

![网页速度与准确率](figures/20261015_百帧网页与精度.png)

处理 FPS 是全部帧数除以视频处理墙钟时间，包含解码、预处理、检测、跟踪、绘制、日志、编码、队列排空和 MP4 索引。总任务耗时另加进程内模型加载、预热和完整 mAP 评估，因此不能拿处理 FPS 倒推用户从点击到结束的全部等待时间。视频仍按原来的 25 FPS 播放，提速体现在更快生成结果。

其他素材网页实测：{other_text}。这些数字只代表对应视频和方案；目标密度、画面尺寸、镜头运动以及桌面其他程序都会影响速度。测试保留桌面、远程桌面和现有应用，没有通过关闭用户程序制造高分。

## 2. 为什么这些改动有效

**TensorRT FP16 与 CUDA Graph。** 检测网络仍是同一个 YOLOv12n。FP16 使用 GPU 的 Tensor Core 计算；CUDA Graph 重放固定形状的 CUDA 操作，减少每帧重复提交算子的 CPU 开销。上一轮 1280 方案中 Graph 没有稳定收益，本轮矩形小输入实测有效，故只在速度方案默认启用。640×384 独立完整流水线从 88.90 到 104.14 FPS；512×320 从约 99 到 110.49 FPS。相同输入、Graph 开关对照的检测与轨迹文件逐字节一致。

**减少填充计算。** 宽屏画面进入 640×640 时有较多上下填充。640×384 使用相同约 640 像素的横向缩放宽度，减少 40% 的输入面积；512×320 再减少实际图像分辨率。YOLOv12 的注意力和特征会随输入形状变化，因此不能保证矩形输入的检测完全相同。MOT20-01 的 mAP@50：640 标准 57.12%，640×384 56.33%，512×320 52.96%。这分别是约 0.79 和 4.16 个百分点的下降。

**减少传输与 Python 竞争。** COCO 原始输出每个候选框有 84 个数，在 GPU 上先选出最大类别分数，再传回 6 个数，传输量减少 92.86%。保留所有候选框及原来的 NMS 和阈值；未通过减少评估框数抬高 FPS。RGB 排布、类别筛选、逐类 NMS 和检测日志格式化移到 C/C++，相机光流有效点筛选改为数组操作。日志仍逐帧保存相同浮点值，解析后与 Python 参考完全一致。

初次真实网页测试明显慢于独立程序，说明线程竞争也是瓶颈。检测日志格式化释放 Python GIL，线程切换间隔调整为 1 毫秒；速度方案的预览改为 640 宽、4 次/秒。预览设置只影响网页刷新，检测次数和保存视频帧数不变。图中的“网页调度优化前/后”来自实际浏览器，两组结果均保留。

**BoT-SORT 策略。** 640 标准和 1280 方案保留原相机运动补偿。移动镜头速度方案使用缩小四倍的灰度图和最多 300 个角点；固定机位方案关闭相机运动补偿。固定机位的光流补偿可省去不少 CPU 工作，但移动镜头应选择移动镜头方案。速度方案使用简洁文字绘制，全部类别、置信度和 ID 仍显示。ReID 仍关闭，没有把它作为此次提速来源；之前的 ReID 对照报告继续保留。

## 3. 两个 DLA / NPU 为什么没有默认启用

TensorRT 实际查询到 **2 个 DLA 核心**。纯 DLA 构建失败，日志报告 `/model.2/Shape` 不支持。允许 GPU 回退后构建成功，但引擎被拆成 16 个 DLA 子图，另有 GPU 矩阵运算、Shuffle、PointWise 和 58 个 Reformat 层；日志还报告每个 DLA 核心最多支持 16 个子图。

![DLA 实测延迟](figures/20261015_DLA实测.png)

同一个 640 模型，GPU 前后交叉测试平均约 {gpu:.2f} 毫秒，DLA/GPU 混合约 {dla:.2f} 毫秒；完整视频混合方案仅 20.48 FPS，mAP@50 为 57.16%。频繁的设备切换、格式转换和当前注意力模型的支持限制抵消了 DLA 收益。两个核心不能直接与 GPU 的速度相加。此次保留可复现构建命令和层信息，不让较慢的 DLA 方案成为默认。

INT8 的既有实测也不支持启用：640 FP16 约 8.92 毫秒，INT8 约 10.36 毫秒，MOT20 mAP@50 从 56.72% 降到 47.90%。INT8 仍保留为实验选项，不把更低精度自动等同于更快。

## 4. 跟踪质量也有取舍

{chr(10).join(tracking)}

这是 MOT20-01 官方训练 ID 标注在预览分辨率上的辅助对照，使用 motmetrics 1.4、IoU≥0.5，保留全部可见度，按报告中明确的忽略区域规则处理。不是 MOTChallenge 官方提交分数。极速输入会漏掉更多远处目标，IDF1 因此下降；ID 切换次数减少也可能只是成功跟踪的目标变少，不能单独宣称跟踪变好。检测 mAP 与身份 IDF1 是两个指标。

## 5. 如何使用及复现

双击根目录“启用高性能模式”，按系统授权窗口输入管理员密码；此动作只把当前功耗模式下的 CPU、GPU、EMC 锁至最高频率，本次开机有效，不保存密码，不修改开机策略或风扇设置。再双击“启动网页”入口，选择 COCO 视频及运行方案。重视远处小目标选 1280；兼顾精度与速度的固定横向画面选 640×384；追求吞吐选 512×320。竖向画面不一定适合横向矩形模型。

运行配置在 `app/config.json`；TensorRT、CUDA 与预处理在 `runtime/backend.py`；GPU 类别归约和原生 CPU 内核在 `runtime/native/`；跟踪和有界流水线在 `app/jetson_runtime.py`、`app/pipeline_workers.py`。每次结果的 `pipeline.json` 记录实际输入形状、Graph、GMC、编码器、完整帧数和耗时。详细职责见 `docs/代码导航.md`。

开发复现：`tools/benchmark_ceiling.py` 顺序比较四种方案，避免并发 GPU 任务；`tests/browser_ceiling.cjs` 验证真实网页的八个完整任务、回放、停止、重复启动和错误 GT；`tests/validate_gpu_reduction.py` 做物理 GPU 数值对照。新模型形状的 ONNX 导出工具为 `tools/export_shape.py`，只在开发时依赖原训练项目和 PyTorch；交付网页运行和 TensorRT 重建均不需要 PyTorch。随包已有 ONNX、Engine、原生库和源码。

## 6. 本次验证与清理

单元测试通过；物理 GPU 类别归约完成 36 组数组一致性检查，另与真实视频样本对照。标准 640 的原生后处理与 GPU 归约，在日志格式化改动前已通过全部 429 帧检测、评估和轨迹逐字节一致性验证；原生 JSON 格式化另验证解析后的数值一致。真实网页完成所有原始帧、保持原尺寸，并使用 NVENC 编码和 H.264 回放；取消任务不计算完整 mAP，错误 GT 不生成假分数。

已清理 {released:.2f} GB：旧环境备份、重复中间 AVI/MP4、历史浏览器测试输出和已停用的 ZIP 解压副本。原始视频、人工标注、模型、当前环境和正式结果保留。实验 JSON/CSV/log 和完整检测日志已压缩归档，原始归档留在本机 `reports/benchmarks/`，转发包提供指标摘要与验证代码。上一轮归档位于 `reports/benchmarks/旧性能实验记录_20261015.tar.gz`。记录和哈希见 `reports/validation/ceiling_comparison.json`、`cleanup.json`、`gpu_pipeline_equivalence.json`、`browser_ceiling.json`。

本轮结果视频单列到工作区的 `视频结果/百帧优化_20261015/`，其中包含标准 640、640×384 和 512×320，便于直接观看。
'''
    name='百帧优化与DLA验证报告_20261015'
    (REPORT/(name+'.md')).write_text(text)
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    css='''@font-face{font-family:Studio;src:url(../app/web/studio-font.woff2)}body{font:14px/1.65 Studio,sans-serif;max-width:980px;margin:35px auto;color:#25342f}h1{font-size:27px}h2{font-size:20px;color:#116b56;margin-top:28px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #d1dcd6;padding:6px}th{background:#eaf1ec}img{max-width:100%}code{background:#f0f3f1;font-size:11px}h2,table,img{break-inside:avoid}@page{size:A4;margin:15mm}@media print{body{margin:0;font-size:11px;line-height:1.6}h1{font-size:22px}h2{font-size:16px}table{font-size:9px}}'''
    (REPORT/(name+'.html')).write_text('<!doctype html><meta charset="utf-8"><title>'+name+'</title><style>'+css+'</style>'+body)
    print('Rendered',name)
if __name__=='__main__':main()
