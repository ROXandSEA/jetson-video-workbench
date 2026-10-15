# 原生性能内核

`frame_ops.c` / `runtime/frame_ops.py` 加速 RGB letterbox 排布和连续通道类别筛选；FP32 输入与 NumPy 参考逐元素一致；检测 JSON 在 C 中格式化并释放 GIL，17 位有效数字保证 float32 转 double 后解析值一致。编译：`gcc -O3 -std=c11 -fPIC -shared runtime/native/frame_ops.c -o runtime/native/frame_ops.so`。不要增加 fast-math。ABI 3。

`yolo_reduce.cu` 在 GPU 上对每个候选框选最大类别分数，传回 xywh、分数和类别。候选框不截断，NaN/Inf 按原规则排除，平分保留最先类别。COCO 的传输从 84 个数降到 6 个数，约减少 92.86%；无人机仅一个类别不使用此优化。编译：`nvcc -O3 -std=c++14 -arch=sm_87 --shared -Xcompiler=-fPIC runtime/native/yolo_reduce.cu -o runtime/native/yolo_reduce.so`。普通启动不需要 nvcc；缺库自动回退原始 TensorRT 输出。物理 GPU 验证脚本是 `tests/validate_gpu_reduction.py`。

`nms_bridge.cpp` 独立实现当前 OpenCV 4.5.4 的逐类贪心 NMS，保留同分稳定排序、double IoU 与 float 距离/重叠的舍入顺序；不依赖 OpenCV C++ 库。编译：`g++ -O3 -std=c++14 -ffp-contract=off -fPIC -shared runtime/native/nms_bridge.cpp -o runtime/native/nms_bridge.so`。Python OpenCV 版本变化或缺库时回退 `cv2.dnn.NMSBoxes`。最终全局排序、max_det 截断和坐标还原由同一份 Python 函数完成。

重编译使用临时 `.so` 再原子替换，避免截断其他进程正在映射的库。随包提供 aarch64 / Orin SM87 构建产物。

## COCO 匹配内核

`coco_match.c` 把 COCOeval 中按 IoU 阈值、检测、人工标注逐项匹配的循环移到原生 C；使用 double 精度，未使用 fast-math。Python 仍使用 pycocotools 的 IoU、稳定排序、面积范围、crowd 忽略、101 点 precision/recall 和标准汇总。没有调整 maxDets、标注数或评测帧数。

编译：`gcc -O3 -fPIC -shared runtime/native/coco_match.c -o runtime/native/coco_match.so`。随包提供 Jetson aarch64 构建产物，无额外 Python 依赖。共享库缺失、架构不匹配或 ABI 不匹配时，自动使用原版 Python COCOeval。源代码和编译方法全部保留。

Python 单图评估接口适配自 pycocotools COCOeval，遵循其 BSD-2-Clause 许可：

Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
