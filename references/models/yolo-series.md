# YOLO Series Inference (v8 / v9 / v10 / v11)

> Ultralytics 谱系：anchor-free、单阶段、TRT/ONNX 生态最成熟的检测网络。
> 推理调优套路高度一致 —— channels_last + FP16/INT8 + 视频解码片上化。

## 版本与尺寸矩阵

参数与 FLOPs 为 COCO 640×640 输入下、Ultralytics 官方 model card 数字（取最常用规模档）。

### YOLOv8 (2023)

| 档 | Params | GFLOPs (640²) | COCO mAP50-95 | 推理目标硬件 |
|---|---|---|---|---|
| YOLOv8n | 3.2 M | 8.7 | 37.3 | T4 / 300V Pro / Jetson |
| YOLOv8s | 11.2 M | 28.6 | 44.9 | T4 / 300V Pro |
| YOLOv8m | 25.9 M | 78.9 | 50.2 | T4 / 910B / H100 |
| YOLOv8l | 43.7 M | 165.2 | 52.9 | 910B / H100 |
| YOLOv8x | 68.2 M | 257.8 | 53.9 | 910B / H100 |

特性：anchor-free、decoupled head、C2f 模块、CIoU + DFL loss。

### YOLOv9 (2024)

| 档 | Params | GFLOPs | COCO mAP50-95 |
|---|---|---|---|
| YOLOv9t | 2.0 M | 7.7 | 38.3 |
| YOLOv9s | 7.1 M | 26.4 | 46.8 |
| YOLOv9m | 20.0 M | 76.3 | 51.4 |
| YOLOv9c | 25.3 M | 102.1 | 53.0 |
| YOLOv9e | 57.3 M | 189.0 | 55.6 |

特性：**PGI** (Programmable Gradient Information) + **GELAN** 主干，相同参数比 v8 提精度。

### YOLOv10 (2024)

| 档 | Params | GFLOPs | COCO mAP50-95 |
|---|---|---|---|
| YOLOv10n | 2.3 M | 6.7 | 39.5 |
| YOLOv10s | 7.2 M | 21.6 | 46.8 |
| YOLOv10m | 15.4 M | 59.1 | 51.3 |
| YOLOv10b | 19.1 M | 92.0 | 52.7 |
| YOLOv10l | 24.4 M | 120.3 | 53.4 |
| YOLOv10x | 29.5 M | 160.4 | 54.4 |

**关键变化**：**NMS-free** 端到端（一致 + 一对多双 head 训练，推理时端到端无需 NMS 后处理）。

### YOLOv11 (2024 Q4, Ultralytics)

| 档 | Params | GFLOPs | COCO mAP50-95 |
|---|---|---|---|
| YOLOv11n | 2.6 M | 6.5 | 39.5 |
| YOLOv11s | 9.4 M | 21.5 | 47.0 |
| YOLOv11m | 20.1 M | 68.0 | 51.5 |
| YOLOv11l | 25.3 M | 86.9 | 53.4 |
| YOLOv11x | 56.9 M | 194.9 | 54.7 |

特性：**C3K2** 模块替换 C2f、改进 SPPF（→ C2PSA）、推理延迟显著降低（同 mAP 下比 v8 快约 20-25%）。

### 任务覆盖（Ultralytics 通用）

每个版本都有：检测 (-)、分割 (-seg)、姿态 (-pose)、分类 (-cls)、OBB (-obb)、追踪集成（BoT-SORT/ByteTrack）。
推理优化要点基本一致；分割多一个 mask head 后处理，姿态多关键点回归 head。

## 显存预算（推理）

```
Forward 激活 + weights ≈ 1 GB (v8n FP16) ~ 4 GB (v8x FP16) at batch=1, 640²
```

640×640 输入 + batch=8 FP16：
- v8n: ~1.2 GB
- v8m: ~2.5 GB
- v8x: ~4.5 GB
- v11x: ~3.5 GB

实际 TRT/ATC engine 还有 1-4 GB workspace；T4 16GB 上 v8x batch ≤ 32 是安全线。

## 量化与精度损失（参考）

| 量化方案 | 损失 mAP50-95 | 工具 | 备注 |
|---|---|---|---|
| FP16 | < 0.1 | TRT `--fp16` | 几乎无损 |
| BF16 | < 0.1 | TRT `--bf16` (Ampere+) | 数值更稳，A100/H100 |
| INT8 PTQ (Entropy) | 0.3-1.0 | TRT `--int8 --calib` | 标准方案 |
| INT8 + QAT | < 0.3 | NVIDIA pytorch-quantization | 精度敏感场景 |
| FP8 (Hopper+) | < 0.3 | TRT-LLM / ModelOpt | 速度 ≈ INT8 |
| 2:4 sparsity | 0.5-1.5 | torch.sparse + cuSPARSELt | 配合 INT8/FP16 |
| Ascend INT8 (ATC) | 0.3-1.0 | ATC `--precision_mode=allow_mix_precision` | 类似 NV |

> 注：以上是公开 benchmark 通常范围；落地数据集差异大，自家数据上务必跑回归。

## NVIDIA TRT 推理样板

```bash
# 1. PyTorch → ONNX (Ultralytics CLI 已封装)
yolo export model=yolov11l.pt format=onnx opset=17 dynamic=True simplify=True

# 2. TRT INT8 engine (T4/A100/H100 通用)
trtexec --onnx=yolov11l.onnx \
    --int8 --fp16 \
    --calib=calib.cache \
    --saveEngine=yolov11l_int8.plan \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:16x3x640x640 \
    --maxShapes=images:64x3x640x640 \
    --useCudaGraph \
    --builderOptimizationLevel=5

# 3. 跑基线
trtexec --loadEngine=yolov11l_int8.plan \
    --useCudaGraph --warmUp=2000 --duration=10 --avgRuns=200 \
    --shapes=images:16x3x640x640
```

参考数字（同模型 batch=16, 640² INT8）：
- T4：~600-900 FPS (v8n/v8s)，~150-220 FPS (v8x)
- A100：~3500 FPS (v8n)，~800 FPS (v8x)
- H100：~5000+ FPS (v8n)，~1300 FPS (v8x)
- 910B：与 H100 同档位粗略 50-70%（INT8 path）

## Triton + DeepStream + NVDEC 组合（视频推理生产）

```bash
# DeepStream 7.x pipeline (简化示例)
[primary-gie]
nvinfer-config-file=yolov8m_int8.txt   # TRT engine + label
batch-size=8

[application]
gpu-id=0
```

DeepStream 把 NVDEC → 预处理 → TRT 推理 → tracker → osd 全部在 GPU 上串成 GStreamer pipeline，**CPU 几乎只做 I/O 调度**。

## YOLOv10 NMS-Free 的特殊处理

v10 训练时双 head（一对多 / 一对一），推理时**只用 one-to-one head，没有 NMS**。
- TRT 导出时不要保留 NMS plugin（v8/v9/v11 通常带 `EfficientNMS_TRT`）
- 客户端不要再做 NMS 后处理
- 优势：端到端 latency 降 10-30%，特别在小目标多的场景
- 劣势：少数情况下小目标召回略低于 v8/v11 + 调优 NMS

## Ascend (ATC + DVPP + AIPP) 推理样板

```bash
# 1. PyTorch → ONNX (同上)
yolo export model=yolov11m.pt format=onnx opset=17

# 2. ATC 编译（Atlas 300V Pro / 910B 都适用）
atc --model=yolov11m.onnx \
    --framework=5 \
    --soc_version=Ascend310P3 \
    --output=yolov11m_om \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --insert_op_conf=aipp_yuv2rgb.cfg \
    --precision_mode=allow_mix_precision

# 3. AIPP：DVPP 解码 YUV → RGB+normalize 全片上
# aipp_yuv2rgb.cfg 见 huawei-atlas-300v-pro.md 样例
```

视频推理：用 **MindX SDK VideoAnalysis** 或自写 DVPP pipeline，把视频解码 + AIPP + 推理 + 后处理串起来。

## 输入预处理（CV 端常见错）

YOLO 的标准预处理：
1. **letterbox**（保持长宽比，pad 到 640×640，填灰 114）
2. **HWC→CHW**
3. **BGR→RGB**
4. **uint8/255 → fp16/32**

实战坑：
- 直接 `cv2.resize` 不 letterbox → mAP 大幅下降
- BGR 没转 RGB → 几乎所有指标全崩
- 不除以 255 → 模型输入数值范围错

TRT 上推荐把预处理写进 ONNX（用 Ultralytics 的 `nms=True` 内嵌）或用 DALI/DeepStream。

## 调优要点速查

1. **channels_last (NV) / 默认 NCHW + NZ (Ascend)** — 卷积优化关键
2. **video 走 NVDEC / DVPP** — 永远不要 CPU 解码
3. **batch 32-64 是 sweet spot** — 太大 mAP 不升但显存 + cache 压力涨
4. **INT8 calibration ≥ 500 张 COCO 风格样本** — 不够 mAP 掉 1%+
5. **NMS 留在 TRT plugin 里**（v8/v9/v11）—— 客户端别再做
6. **dynamic batch**：minShapes=1, optShapes=训练 batch, maxShapes=2×optShapes
7. **640² 是黄金分辨率**：1280 慢 4×、416 快 2× 但精度掉
8. **多模型流水线**：YOLO 检测 + ReID/CLIP 二阶分类 → 共享 NVDEC frame
9. **追踪 (BoT-SORT/ByteTrack)** 通常 CPU 跑足够；ByteTrack GPU 实现仅在 > 100 FPS × 多路时有意义

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| TRT INT8 calibration 数据集与线上分布不一致 | mAP 大跳 | 用线上抽样样本 calibrate |
| Ultralytics PyTorch eager 跑 → 觉得"原生最快" | 实测远慢于 TRT | 必须导 TRT/ONNX 才上生产 |
| Dynamic input 不固定 → 频繁触发 cudnn benchmark | p99 抖 | 用 `--minShapes/--optShapes` 限定区间 |
| YOLOv10 当 v8 用做了 NMS 后处理 | 重复 NMS 慢且漏检 | 看 v10 文档，端到端无 NMS |
| 视频分辨率不是 32 倍数 | letterbox 后右下灰带 | 网络 stride=32，输入必须 32 倍数 |
| Ascend 上 ONNX `EfficientNMS_TRT` 不支持 | ATC 编译失败 | 导出时 `nms=False`，在客户端/CPU 做 NMS |
| TRT 8.6 → 10 升级后 plugin 名变 | engine deserialize 失败 | 重新 build engine |
| FP16 在小目标 mAP50:95 掉 0.5+ | 大头预测梯度小，FP16 精度不够 | 关键层留 FP32 (TRT `setPrecision`) |
| 多模型共 NVDEC frame 还 D2H 再 H2D | PCIe 瓶颈 | DeepStream/CUDA shared device buffer |

## 服务化建议

- **Triton Inference Server** + TRT backend + dynamic batching
- 单卡塞 N 路视频 = min(NVDEC 容量, GPU 算力, 显存) 取最小
  - T4: 36 路 1080p YOLOv8s INT8 ≈ 实际上限
  - H100: 80+ 路 v11l INT8 不难
  - Atlas 300V Pro: **264 路 1080p 解码 IP** + 多 v8s 实例 = 视频 AI 性价比之王
- 同一 Triton 实例放多模型（检测 → ReID → 属性识别）做 ensemble，跨步 GPU 内传

## 参考

- Ultralytics docs: docs.ultralytics.com
- YOLOv10 paper（NMS-free 设计）
- DeepStream SDK 文档（视频管线）
- 本仓库 [`../../skills/nv-infer-cnn-vit/SKILL.md`](../../skills/nv-infer-cnn-vit/SKILL.md)
- 本仓库 [`../../skills/nv-infer-io-data/SKILL.md`](../../skills/nv-infer-io-data/SKILL.md)
