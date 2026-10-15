#!/usr/bin/env python3
"""Developer report generation; matplotlib/markdown are optional, not runtime dependencies."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports';FIG=REPORT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
bench=json.loads((REPORT/'benchmarks/paired_benchmark.json').read_text())
fp32=json.loads((REPORT/'audit_torch_640.json').read_text());ref1280=json.loads((REPORT/'audit_torch_1280.json').read_text())
rows=[r for r in bench if r['confidence']==.001]
scenes=['MOT20-01-raw.mp4','VisDrone-val-uav0000137_00458_v-raw.mp4']
profiles=['coco_yolov12n_640_fp16.engine','coco_yolov12n_1280_fp16.engine','coco_yolov12n_640_int8.engine']
labels=['640 FP16','1280 FP16','640 INT8 混合']
colors=['#3183b9','#13a678','#de8a30']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,video,title in zip(axes,scenes,['MOT20-01：person AP','VisDrone：兼容类别子集 AP']):
 x=np.arange(3);rs=[next(r for r in rows if r['video']==video and r['engine']==e) for e in profiles]
 for offset,key,hatch,name in [(-.18,'map50',None,'AP50'),(.18,'map50_95','//','AP50:95')]:
  values=[100*r['metrics'][key] for r in rs]
  bars=ax.bar(x+offset,values,.34,color=colors,hatch=hatch,label=name,alpha=1 if hatch is None else .6)
  for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.8,f'{v:.2f}',ha='center',fontsize=10)
 ax.set_xticks(x,labels);ax.set_ylim(0,80);ax.set_ylabel('AP（%）');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
fig.suptitle('同一批 60 帧 / 评估门槛 0.001 / 相同 NMS 与 GT；不是全视频成绩',fontsize=12);fig.tight_layout();fig.savefig(FIG/'精度对照.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));raw=[next(r for r in rows if r['video']==scenes[0] and r['engine']==e)['raw_inference_with_H2D_D2H_mean_ms'] for e in profiles]
bars=ax.bar(labels,raw,color=colors)
for bar,v in zip(bars,raw):ax.text(bar.get_x()+bar.get_width()/2,v+.3,f'{v:.2f} ms',ha='center')
ax.set_ylabel('单帧检测器执行（ms，越小越快）');ax.set_title('包含 H2D + TensorRT + D2H；不等于整条视频流水线 FPS');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(FIG/'推理耗时对照.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
visrs=[next(r for r in rows if r['video']==scenes[1] and r['engine']==e) for e in profiles[:2]]
classes=[c['name'] for c in visrs[0]['metrics']['per_class'] if c['map50_95'] is not None]
x=np.arange(len(classes))
for k,r in enumerate(visrs):
    lookup={c['name']:c['map50_95'] for c in r['metrics']['per_class']}
    bars=ax.bar(x+(k-.5)*.36,[lookup[n]*100 for n in classes],.36,label=labels[k],color=colors[k])
    for bar,n in zip(bars,classes):ax.text(bar.get_x()+bar.get_width()/2,lookup[n]*100+1,f'{lookup[n]*100:.1f}',ha='center',fontsize=10)
ax.set_xticks(x,classes);ax.set_ylabel('AP50:95（%）');ax.set_ylim(0,65);ax.set_title('VisDrone 各类严格 AP：平均数会被自行车、摩托车拉低');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(FIG/'VisDrone分类精度.png',dpi=170);plt.close(fig)
# Honest view of source GT versus threshold-.05 detections, with the same native crop.
video=scenes[1];source=ROOT/'videos/VisDrone'/video;gt=json.loads(source.with_suffix('.gt.json').read_text())
archives=[np.load(REPORT/'benchmarks'/f'{source.stem}_{Path(e).stem}_visual.npz') for e in profiles[:2]]
frame=archives[0]['frame'];index=int(archives[0]['frame_id']);annotations=[a for a in gt['annotations'] if a['image_id']==index and not a['iscrowd']]
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(font_path,26)
x0,y0,w,h=600,450,1300,780
if frame.shape[1]<x0+w or frame.shape[0]<y0+h:x0=y0=0;h,w=frame.shape[:2]
panels=[]
for panel,title in [(0,'人工 GT（绿色，所有计分类别）'),(1,'640 FP16 检测（蓝色，≥0.30）'),(2,'1280 FP16 检测（蓝色，≥0.30）')]:
 im=Image.fromarray(frame[:,:,::-1]).crop((x0,y0,x0+w,y0+h));draw=ImageDraw.Draw(im)
 boxes=[a['bbox'] for a in annotations] if panel==0 else [[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])] for d in archives[panel-1]['detections'] if int(d[5]) in {0,1,2,3,5,7} and float(d[4])>=.3]
 count=0
 for x,y,bw,bh in boxes:
  if x+bw>x0 and x<x0+w and y+bh>y0 and y<y0+h:
   draw.rectangle([x-x0,y-y0,x+bw-x0,y+bh-y0],outline='#31d889' if panel==0 else '#63b9ff',width=3);count+=1
 im=im.resize((780,round(780*h/w)));canvas=Image.new('RGB',(780,im.height+90),'#161c24');canvas.paste(im,(0,90));d=ImageDraw.Draw(canvas);d.text((16,10),title,font=font,fill='white');d.text((16,47),f'同一帧 {index} / 此区域框数 {count}',font=font,fill='#cbd5e1');panels.append(canvas)
canvas=Image.new('RGB',(sum(i.width for i in panels),panels[0].height),'white');offset=0
for im in panels:canvas.paste(im,(offset,0));offset+=im.width
canvas.save(FIG/'小目标同帧对照.png')
# Verified INT8 format coverage (layer output count, not FLOP coverage).
layers=json.loads((ROOT/'models/coco_yolov12n_640_int8.layers.json').read_text())['Layers']
int8_layers=[l for l in layers if 'Int8' in json.dumps(l) or 'INT8' in json.dumps(l)]
coverage={'total_engine_layers':len(layers),'layers_with_INT8_format':len(int8_layers),'note':'Format presence in inspector layer I/O, not MAC/FLOP coverage; mixed precision'}
(REPORT/'INT8层精度核查.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
validation=REPORT/'validation/browser_delivery.json';actual=json.loads(validation.read_text()) if validation.exists() else None
lines=['# 精度排查、INT8 对照与独立交付报告','',
'本次确认了两项可以修正的原因：评估错误地使用了跟踪门槛 0.05，以及 640 输入对小目标不够友好。TensorRT FP16 与 FP32 的差异很小，不能把之前偏低的 AP 归因于 Jetson 算力不足或 FP16 明显破坏精度。运行代码已整理成独立文件夹，COCO 默认使用 1280 FP16，仍可切换 640 FP16 和实验 INT8。','',
'## 1. 先看结果，再看原因','',
'所有下面的对照表均为 MOT20-01、VisDrone uav0000137_00458_v 各均匀抽取固定 60 帧。测试帧与 INT8 校准帧分开。AP 的百分点差是绝对差，不是速度提升比例，也不代表完整数据集分数。','',
'![精度对照](figures/精度对照.png)','',
'| 输入与后端 | MOT AP50 | MOT AP50:95 | VisDrone AP50 | VisDrone AP50:95 | 检测器耗时 |','| --- | ---: | ---: | ---: | ---: | ---: |']
for name,records in [('640 PyTorch FP32',[r for r in fp32 if r['confidence']==.001]),*[(label,[next(r for r in rows if r['video']==v and r['engine']==e) for v in scenes]) for label,e in zip(labels,profiles)]]:
 a,b=records;latency='参考审计，未混作隔离延迟' if 'engine' not in a else f"{a['raw_inference_with_H2D_D2H_mean_ms']:.2f} ms"
 lines.append(f"| {name} | {a['metrics']['map50']*100:.2f}% | {a['metrics']['map50_95']*100:.2f}% | {b['metrics']['map50']*100:.2f}% | {b['metrics']['map50_95']*100:.2f}% | {latency} |")
base=[next(r for r in rows if r['video']==v and r['engine']==profiles[0]) for v in scenes]
quality=[next(r for r in rows if r['video']==v and r['engine']==profiles[1]) for v in scenes]
quant=[next(r for r in rows if r['video']==v and r['engine']==profiles[2]) for v in scenes]
speed=raw[0]/raw[2];lines+=['',f"1280 FP16 相对 640 FP16 的 AP50：MOT 提升 {(quality[0]['metrics']['map50']-base[0]['metrics']['map50'])*100:.2f} 个百分点，VisDrone 提升 {(quality[1]['metrics']['map50']-base[1]['metrics']['map50'])*100:.2f} 个百分点。代价是检测器延迟更大，网页提供方案选择。",'',
'## 2. 找到了什么实际问题','',
'**门槛混用。** 跟踪的低分关联门槛 0.05 是为 BoT-SORT 设置的；计算 AP 通常需要保留到 0.001。此前将两者共用，遗漏低分候选，影响召回。FP32 同样 60 帧上，MOT AP50 从 47.84% 到 56.85%（+9.01 个百分点）；VisDrone 从 30.80% 到 34.39%（+3.59 个百分点）。这部分是评估口径修正，不能宣称模型能力提高了同样多。现在跟踪日志 ≥0.05，评估日志 ≥0.001，来自同一次模型输出。','',
'**输入尺寸。** VisDrone 原片 2688×1512，送入 640 时缩小到约 640×360，剩余填充。固定标注帧目标短边中位数从原图缩到 14.05 像素，9.34% 的目标短边不足 8 像素。1280 输入短边中位数为 28.10 像素，不足 8 像素比例降到 0.13%。提高输入尺寸确实改善了 AP。MOT 标注短边在 640 时中位数 23.33 像素，拥挤遮挡与边界定位也会限制严格 AP。','',
'![小目标同帧对照](figures/小目标同帧对照.png)','',
'上图绿色是人工标注，蓝色是同一帧检测结果；框数只帮助看差异，不能当作精度。两种模型在这张插图中统一显示分数 ≥0.30 的检测（高分关联门槛），与跟踪低分候选门槛 0.05 和 AP 门槛 0.001 分开。提高分辨率也可能增加低分误检或跨类别重叠框；不能只看框数量判断效果。','',
'**模型与任务。** 当前是 YOLOv12n 小模型，不是更大容量模型，也没有证据表明 COCO 权重针对 VisDrone 航拍密集人车做过微调。Jetson 等级提供速度和可用内存，准确性取决于模型、训练分布、输入信息和标注口径。VisDrone 用 COCO 模式；UAV 模式只找画面中的无人机。后续更高精度可考虑航拍数据微调、更大模型或切片推理，需重新对照，不在本次结果中虚构收益。','',
'**平均方式也是视觉观感与 mAP 的差异来源。** 640 FP16 在固定 VisDrone 帧上，car 的 AP50:95 约 44.29%，person 17.56%，bicycle 1.38%，motorcycle 4.25%，四类等权平均只有 16.87%。画面里车辆框看起来不错，不代表自行车、摩托车也同样好。该结果提示航拍类别和遮挡目标确实需要模型改进，不能只靠硬件或换 INT8。','',
'![分类精度](figures/VisDrone分类精度.png)','',
'## 3. 排除了哪些部署错误','',
'FP32 与原 TensorRT FP16 在相同 640、同门槛的 AP50 差异都小于 0.14 个百分点。代表帧输出类别概率的平均绝对差约 1.4e-6～1.8e-6。FP16 的确存在小数值误差，但不足以解释原来的低分。','',
'已核对 BGR→RGB、NCHW、/255、填充 114、缩放回原图、80 类名字与类别 ID、无 objectness 的输出解释、输入/输出形状、模型/ONNX/引擎哈希。新 NumPy 预处理与原 FP32 预处理逐元素一致；NMS 改用 OpenCV，极低分相同分值的候选排序可能不同，因此另用固定帧 AP 对照验证，而不宣称所有日志逐字节一致。','',
'MOT 官方预览片缩小了画面：已核对帧数、FPS，同步缩放 GT；一基坐标转零基。VisDrone 源序列与镜像的标注全文、首 JPEG 相同，全部连续帧编码，GT 按帧匹配。网页再校验视频 SHA256、帧数、尺寸与类别，杜绝误评另一段视频。图示 GT 和检测框使用同一原视频像素坐标。','',
'**评测边界。** MOT 是 person 单类 AP；VisDrone 是映射后的兼容 COCO 类别子集 AP，当前片段实际计分 4 类，bus/truck 无有效 GT 不参与平均。crowd 用作忽略区域。检测全局上限 300，COCOeval 每帧每类最多 100；101 个召回点，IoU .50:.05:.95。这不是官方 MOT 的 MOTA/IDF1，不是官方 VisDrone 全类别榜单，也不是全部 COCO 80 类测试集成绩。类别 argmax 与 Ultralytics val 多标签输出也有区别；所有本次对照保持相同口径。','',
'## 4. INT8 做了什么，值不值得','',
'使用 TensorRT EntropyCalibration2，5 段非测试视频各抽 32 帧，共 160 帧：MOT17-04、MOT17-05、MOT20-02、VisDrone uav0000086、uav0000117。校准预处理与推理一致。没有使用随机动态范围；保存了视频哈希、具体帧号、校准缓存与层信息。两段测试视频未用于校准；这是留出视频实验，不是完整训练/验证集泛化结论。','',
f"引擎层检查发现 {len(layers)} 个执行层中，{len(int8_layers)} 个层的输入或输出格式出现 INT8；其他部分保留 FP16/FP32。因此称为 INT8 混合精度，不能称为全网络纯 INT8。这是格式计数，不是计算量覆盖比例。",'',
'INT8 层检查还看到 69 个 Reformat 层，部分矩阵运算、softmax 和输出仍需浮点计算。格式转换和回退会消耗时间，Orin 上 INT8 理论吞吐更高，不保证小模型实际更快。精度下降是这版 PTQ 量化的实测结果；可进一步尝试保留敏感层浮点、显式 Q/DQ 或 QAT，但本次未虚构这些方法的收益。','',
'![推理耗时](figures/推理耗时对照.png)','',
f"本机 640 INT8 相对 640 FP16 检测器速度为 {speed:.2f} 倍，平均延迟增加 {(raw[2]/raw[0]-1)*100:.1f}%；AP50 变化为 MOT {(quant[0]['metrics']['map50']-base[0]['metrics']['map50'])*100:+.2f}、VisDrone {(quant[1]['metrics']['map50']-base[1]['metrics']['map50'])*100:+.2f} 个百分点。",'',
'交替复测 FP16 / INT8 共 3 轮，每次预热 60 次、测量 250 次：FP16 平均 8.92 ms，INT8 平均 10.36 ms，INT8 延迟高约 16.0%。每轮 INT8 都更慢，排除了单次测量顺序造成的偶然反转；该复测使用固定零输入，原表使用真实视频首帧。动态频率未锁定，保留各轮范围与 P50/P95 原始数据。','',
'默认保留 FP16：本次 INT8 同时损失精度和速度，不推荐日常使用。用户当前重点是小目标检测精度，1280 FP16 有明确改善；INT8 仅留作可复现实验，不自动覆盖 FP16 引擎，普通启动不重建它。其他 INT8 实现是否值得用，需要在自己的视频上重新核对速度和 AP；本次这版的结论是不值得替换 FP16。只缩减检测器耗时无法按同样比例加速解码、跟踪、画图和最终编码。','',
'测时在 Orin NX 16GB 的 40W 模式、未锁频情况下进行：每段先预热 60 次，隔离重复执行 200 次，包含主机/设备拷贝及同步，排除预处理、NMS、跟踪、视频解码与编码；引擎构建及视频任务不与测时并行。系统桌面正常运行，结果会受动态频率影响。','',
'## 5. 网页与交付如何使用','',
'![运行方案](figures/网页运行方案.png)','',
'目录 `Jetson_视频工作台` 是独立运行包。根目录启动脚本创建可重建虚拟环境，离线安装锁定依赖，检查 JetPack 库与引擎，打开随包 ARM64 浏览器。生产推理不导入 PyTorch 或 Ultralytics，使用 TensorRT + CUDA 原生缓冲和 NumPy/OpenCV；减少对原工程及 GPU Torch 安装的绑定。原始视频与输出分目录存放。','',
'网页与后端启用 TensorRT、BoT-SORT 和 sparseOptFlow 相机运动补偿；ReID 关闭。检测 mAP 评估原始检测框，不评估轨迹 ID，不能把 mAP 提升归功于 BoT-SORT/ReID。历史 ReID 结果另存报告供参考，不混进本次量化测量。','',
'接收方需要 Jetson Orin + JetPack 6.1/6.2 + TensorRT 10.x + Python 3.10，以及系统 TensorRT/CUDA/OpenCV、FFmpeg、python3-venv/tk。包内提供 Python wheel、ONNX/引擎、浏览器与源码，双击根目录 `双击启动网页` 即可；进度窗口自动准备环境，网页就绪后打开浏览器，服务在后台运行，再次双击复用已有服务。`启动网页.sh` 保留供终端调试。不能保证普通 Windows 电脑只打开 HTML 就能运行 GPU 检测。版本不匹配会本机重建正式 FP16 引擎；INT8 也保留真实校准素材和手动构建工具，因实测不推荐而不在普通启动时重建。','',
'## 6. 实际验收与原始证据','']
if actual:
 lines+=['已在真实浏览器完成以下整段任务，结果 MP4 可播放，下方 AP 对应当前视频与运行方案。与前面的 60 帧抽样表分开报告：','',
 '| 任务 | 完整帧数 | AP50 | AP50:95 | 视频流水线 FPS |','| --- | ---: | ---: | ---: | ---: |']
 for state in actual['records']:
  ev=state['evaluation'];pipeline=json.loads((Path(state['output_dir'])/'pipeline.json').read_text())
  score=lambda k: f"{ev[k]*100:.2f}%" if ev['status']=='completed' else '无 GT'
  lines.append(f"| {Path(state['source_path']).name} / {state['profile']} | {state['frame']} | {score('map50')} | {score('map50_95')} | {pipeline['processing_fps']:.2f} |")
 lines+=['','视频流水线 FPS 包含解码、预处理、检测、两份 NMS、BoT-SORT、画图、AVI、预览和日志，排除预热、最终 H.264 编码及 COCOeval；最终编码和评估秒数另存 pipeline.json。']
lines+=['','14 项评估/配置单元测试通过；真实 GPU 浏览器任务、重复任务拒绝、停止保留部分输出、错误标注不出假分数，以及目录复制更名后的离线安装与两模式验证另见 `validation/`。所有素材视频已有完整解码与哈希校验，UAV 例片额外随包提供。','',
'- `benchmarks/paired_benchmark.json`：冻结帧号、每配置 AP、延迟、引擎与视频哈希。',
'- `audit_torch_640.json`、`audit_trt_640.json`、`audit_torch_1280.json`：原参考后端审计。',
'- `INT8层精度核查.json`、`../models/*int8.layers.json`：量化执行层格式。',
'- `validation/browser_delivery.json`：真实任务记录；`validation/portability.json`：迁移路径测试。',
'- `../docs/代码导航.md`：代码入口；`../docs/mAP标注格式.md`：自己的视频怎么评估。',
'- `../videos/README_素材选择.md`：素材来源与选择；`../SHA256SUMS`：交付文件校验。','']
lines.insert(2, '> 2026-10-15 已更新并行流水线、NVENC 直接编码和 mAP 原生匹配加速。本文中的旧流水线耗时保留作历史记录，最新速度及精度不变验证见同目录《性能优化报告_20261015》。')
text='\n'.join(lines);(REPORT/'精度排查与交付报告.md').write_text(text)
import markdown
html='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>Jetson 精度排查与交付</title><style>body{font-family:'Noto Sans CJK SC',sans-serif;max-width:1100px;margin:35px auto;line-height:1.65;color:#192b35;padding:0 25px}h1,h2{color:#126f61}h2{margin-top:30px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #cad5da;padding:7px}th{background:#e8f4f0}code{background:#edf0f2;padding:2px 4px}@media print{body{font-size:13px;margin:0}h2{break-after:avoid}img,table{break-inside:avoid}@page{size:A4;margin:14mm}}</style>'''+markdown.markdown(text,extensions=['tables','fenced_code'])+'</html>'
(REPORT/'精度排查与交付报告.html').write_text(html);print(REPORT/'精度排查与交付报告.md')
