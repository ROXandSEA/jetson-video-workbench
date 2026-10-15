#!/usr/bin/env python3
"""Summarize verified full-video 512 input experiments and actual output frames.

Run after the corrected encoder runs and browser_small_input.cjs have finished.
Snapshots the measurements; outputs a short PDF/HTML/Markdown, a scatter plot,
real annotated-frame comparison and separate videos for the user to inspect.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import markdown

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/512密集人群优化'
REPORT.mkdir(parents=True,exist_ok=True)
FIG=REPORT/'插图';FIG.mkdir(exist_ok=True)
RAW=ROOT/'.state/small_input_corrected'
VIDEO=ROOT.parent/'视频结果/512密集人群优化_20261015'
TITLE='512密集人群精度优化报告'


def main():
    baseline=RAW/'original_mot05'
    b=json.loads((baseline/'pipeline.json').read_text())
    br=json.loads((baseline/'benchmark.json').read_text())
    old=dict(label='原512 / NMS0.7',profile='speed_original',pipeline=b,
             state=dict(output_dir=str(baseline),elapsed=br['wall_seconds'],evaluation=b['evaluation']))
    current=json.loads((RAW/'optimized/optimized_comparison.json').read_text())
    rows=[old,*current]
    for x in rows:
        assert x['pipeline']['frames']==x['state']['evaluation']['frames']
        assert 'NVENC' in x['pipeline']['encoder']
    (ROOT/'reports/validation/small_input_corrected.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (REPORT/'实测数据.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    mot05=[x for x in rows if x['pipeline']['frames']==3315]
    mot01=[x for x in current if x['pipeline']['frames']==429]
    labels=['原512','512去重优化','512分区增强']
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.3))
    for ax,key,title in [(axes[0],'map50','MOT20-05全3315帧 · AP50'),(axes[1],'map50_95','MOT20-05全3315帧 · AP50:95')]:
        bars=ax.bar(labels,[x['pipeline']['evaluation'][key]*100 for x in mot05],color=['#8997aa','#159982','#2683b5'])
        ax.bar_label(bars,fmt='%.2f%%',padding=4);ax.set_ylim(0,44 if key=='map50' else 20);ax.set_title(title);ax.set_ylabel('AP（%）');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(FIG/'01_512精度对照.png',dpi=170);plt.close(fig)
    # Read the actual encoded images, not re-drawn detections that might hide an
    # encoder error. Non-multiple-of-four width is explicitly checked here.
    panels=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',26)
    for x,label in zip(mot05,labels):
        folder=Path(x['state']['output_dir']);movie=folder/'result.mp4'
        cap=cv2.VideoCapture(str(movie));cap.set(cv2.CAP_PROP_POS_FRAMES,1699);ok,image=cap.read();cap.release()
        if not ok or image.shape!=(540,826,3) or image.std()<15:raise RuntimeError('Output image invalid '+str(movie))
        rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));panel=Image.new('RGB',(826,620),'#182b38');panel.paste(rgb,(0,80))
        draw=ImageDraw.Draw(panel);draw.text((18,8),label+f" / AP50 {x['pipeline']['evaluation']['map50']*100:.2f}%",font=font,fill='white')
        draw.text((18,43),'同一源帧1700 / 实际BoT-SORT输出 / 原尺寸',font=font.font_variant(size=20),fill='#cbd9df');panels.append(panel)
    canvas=Image.new('RGB',(826*3,620),'#182b38')
    for i,panel in enumerate(panels):canvas.paste(panel,(i*826,0))
    canvas.save(FIG/'02_512真实结果同帧.png')
    evidence=[]
    proof=ROOT/'reports/validation/small_input_video_decode.json'
    prior=json.loads(proof.read_text()) if proof.is_file() else []
    decoded_hashes={item['sha256'] for item in prior if item.get('full_decode') and item.get('frames')==3315}
    # Keep the full three MOT20-05 videos and metrics in a user-facing folder.
    # Hardlinks avoid another copy of large immutable files on the same disk.
    for i,(x,label) in enumerate(zip(mot05,labels),1):
        source=Path(x['state']['output_dir']);dest=VIDEO/f'{i:02d}_{label}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['result.mp4','pipeline.json','evaluation.json','tracks.csv']:
            src=source/name;target=dest/name
            if not target.exists():
                try:target.hardlink_to(src)
                except OSError:shutil.copy2(src,target)
        digest=hashlib.sha256((dest/'result.mp4').read_bytes()).hexdigest()
        if digest not in decoded_hashes:
            subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(dest/'result.mp4'),'-f','null','-'],check=True,capture_output=True)
        evidence.append(dict(path=str(dest/'result.mp4'),sha256=digest,frames=3315,full_decode=True))
    # Save a compact sequential record of full-video decoded content checks.
    (ROOT/'reports/validation/small_input_video_decode.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    def table():
        lines=['| MOT20-05全3315帧 | AP50 | AP50:95 | 处理FPS | 总耗时秒 |','| --- | ---: | ---: | ---: | ---: |']
        for x,label in zip(mot05,labels):
            p=x['pipeline'];e=p['evaluation'];lines.append(f"| {label} | {e['map50']*100:.2f}% | {e['map50_95']*100:.2f}% | {p['processing_fps']:.2f} | {x['state']['elapsed']:.2f} |")
        return '\n'.join(lines)
    one=next(x for x in mot05 if x['profile']=='speed');detail=next(x for x in mot05 if x['profile']=='speed_detail')
    gain=(one['pipeline']['evaluation']['map50']-b['evaluation']['map50'])*100
    gain2=(detail['pipeline']['evaluation']['map50']-b['evaluation']['map50'])*100
    sections=[f'''# 512×320 密集人群精度优化

2026-10-15。本次重点是MOT20密集人群。**同一个512×320 FP16引擎，原全图一次检测AP50为{b['evaluation']['map50']*100:.2f}%；去重优化后{one['pipeline']['evaluation']['map50']*100:.2f}%（+{gain:.2f}个百分点）；全图＋四分区增强后{detail['pipeline']['evaluation']['map50']*100:.2f}%（+{gain2:.2f}个百分点）。** 完整3315帧，没有跳帧。

![精度对照](插图/01_512精度对照.png)

{table()}

处理FPS包含解码、每帧检测、BoT-SORT、绘图日志和NVENC保存，排除加载预热和最后AP。总耗时包含最后评估，MOT20-05约64.6万个标注行人会增加评估等待。原版以同一生产流水线的基准脚本运行，新方案经正常网页接口运行；真实浏览器预览的单独验证见后文，不把运行界面负载混作受控速度收益。

**第一项：去重优化，已更新“极速512”。** NMS IoU从0.7改为0.55，抑制小输入下错位重叠的重复框，不增加检测次数。阈值不能无限降低，否则会压掉相邻真实行人。MOT20-01、05用于探索，另用MOT20-02和VisDrone固定40帧交叉检查。0.45虽然在05更高，却损失01；最终选0.55，没有按单段最高分取值。降低NMS阈值不会保证每个场景都更好，标准640/1280的NMS仍是0.7。

**第二项：新增“人群细节512（分区增强）”。** 每帧全图一次，再以60%宽高的四个重叠区域各推理一次，小目标得到约1.67倍的线性像素尺度；过滤内部裁剪边缘截断框，恢复原图坐标，按类合并去重后只更新一次BoT-SORT。仍是512×320引擎，但每帧五次推理，明确用更多计算换细节。四个区域完全覆盖画面，不依赖人工GT或手动圈定目标。
''',f'''## 实际画面、限制与使用方法

![实际输出同帧](插图/02_512真实结果同帧.png)

上图来自三份实际H.264结果的同一帧1700，未用重绘检测代替真实输出。原单次512在此帧的person最高分约0.23，很多框低于原轨迹建立门槛；分区检测能找回更多小人。增强选项还把高分关联/新建阈值从0.30/0.40调为0.15/0.20，低分门槛仍为0.05。更容易显示低分轨迹也可能增加误轨迹，尚未测本方案IDF1；**AP提升来自检测改变，不是改跟踪阈值或ReID。**

两种优化方案的完整MOT20-01分别为AP50 {next(x for x in mot01 if x['profile']=='speed')['pipeline']['evaluation']['map50']*100:.2f}%与{next(x for x in mot01 if x['profile']=='speed_detail')['pipeline']['evaluation']['map50']*100:.2f}%，原512历史为52.96%。极速优化的AP50:95在01有轻微下降，不能声称所有指标都提升。曝光/亮度、锐化、CLAHE和插值试验收益很小，因此未加入正式方案。

**保持相同评估规则。** AP置信度0.001，COCOeval 101点召回，IoU .50:.05:.95，每帧每类maxDets100；GT、帧数、类别、源视频不变。MOT20-05平均每帧约195个有效行人，maxDets100会限制密集场景最大召回，但本次没有放宽它来抬分。这是单段person AP，不是整个MOT20或COCO80成绩。

**同时修复结果视频。** 826像素宽的BGR行没有对齐4字节时，旧OpenCV/GStreamer入口可能输出黑屏或花屏。现先补到828宽，转为BGRx后裁回826，再用NVENC编码；没有缩放或裁掉原画面。真实30帧回环的平均像素绝对误差约2.16/255，且三段完整3315帧视频均完整解码通过。修复前视频不作为本报告可交付结果，所有表中速度和视频均重跑。

**网页直接选择。** 刷新网页，COCO模式下选“极速512（去重优化）”保留速度；选“人群细节512（分区增强）”优先看拥挤人群。固定机位优先；移动镜头需要GMC或标准方案。分区增强的实测速度见表，不宣称还能百帧；要兼顾更高FPS可用原640×384固定方案，要更高精度仍可选1280。

**验证与文件。** 35项单元测试通过；真实浏览器完成两种方案的全429帧、原尺寸H.264回放、NVENC、明确1/5次检测计数、分区中途取消及拒绝并发检查。浏览器实测数据见 `../validation/browser_small_input.json`，各时刻后台负载影响FPS；不把单次速度差全部归因于去重。原生C/CUDA代码及标准方案保持相同语义。

代码：`runtime/spatial_detail.py`（裁剪/坐标恢复/合并）、`app/pipeline_workers.py`（调度/编码对齐）、`app/config.json`（方案）、`tools/benchmark_small_input.py`（冻结样本探索）、`benchmark_small_web.py`（整段接口）、`tests/browser_small_input.cjs`（浏览器验收）。原始结果在 `实测数据.json`、`../benchmarks/small_input_20261015/`，GT与模型哈希在各记录内。单列完整结果在工作区 `视频结果/512密集人群优化_20261015/`。
''']
    text='\n\n'.join(sections);(REPORT/(TITLE+'.md')).write_text(text)
    css="@page{size:A4;margin:14mm}body{font:11pt/1.62 'Noto Sans CJK JP',sans-serif;color:#203344;margin:0}article{break-after:page}article:last-child{break-after:auto}h1,h2{font-size:22pt;color:#087c74;line-height:1.35}img{max-width:100%;max-height:230px;object-fit:contain}table{border-collapse:collapse;width:100%;font-size:9.5pt}td,th{border:1px solid #cfdee3;padding:6px}th{background:#e9f5f2}code{font-size:9pt;overflow-wrap:anywhere}strong{color:#126b68}p:has(img),table{break-inside:avoid}@media screen{body{max-width:1000px;margin:35px auto}}"
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>512人群精度优化</title><style>'+css+'</style>'+''.join('<article>'+markdown.markdown(s,extensions=['tables','fenced_code'])+'</article>' for s in sections)+'</html>'
    (REPORT/(TITLE+'.html')).write_text(html)
    subprocess.run([str(ROOT/'browser/chrome-linux-arm64/chrome'),'--headless','--no-sandbox','--disable-gpu','--no-pdf-header-footer',f'--print-to-pdf={REPORT/(TITLE+".pdf")}',(REPORT/(TITLE+'.html')).as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (VIDEO/'README_对照说明.md').write_text('# 512密集人群优化对照\n\n三段均为完整MOT20-05：3315帧、原826×540、25FPS播放。只看视频播放速度不能判断处理吞吐。\n\n'+table()+'\n\n01原512；02仍每帧一次检测，只调整NMS去重；03全图＋4分区，每帧5次推理，合并后跟踪一次，并降低新轨迹门槛。\n\n全部结果使用修复后的NVENC行对齐输出，已完整解码；标注AP使用同一规则。报告见正式包 reports/512密集人群优化/。\n')
    (REPORT/'来源校验.json').write_text(json.dumps(dict(new_inference=False,whole_video_data_sha256=hashlib.sha256((REPORT/'实测数据.json').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(REPORT.glob(TITLE+'.*'))]),ensure_ascii=False,indent=2)+'\n')
    print(REPORT/(TITLE+'.pdf'))


if __name__=='__main__':main()
