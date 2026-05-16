---
name: nv-infer-cnn-vit
description: NVIDIA GPU 上 CNN 与 ViT 推理的专项优化——channels_last (NHWC)、cuDNN benchmark / heuristic、Conv-BN-ReLU 融合、im2col vs implicit GEMM vs Winograd vs FFT、stride 卷积、depthwise/group conv、patch embed、windowed attention (Swin)、relative position bias、detection / segmentation head 优化、TensorRT 卷积 tactic 选择、INT8 卷积 calibration、动态分辨率。当用户问"CNN 推理"、"ResNet/EfficientNet/YOLO"、"ViT"、"Swin"、"DETR"、"channels_last"、"NHWC"、"cuDNN"、"卷积慢"、"patch embed"、"为什么 INT8 没快"时使用。
---

# CNN / ViT Inference Optimization

> 这一类模型在 NVIDIA 上的优化范式相对成熟，几乎全可以靠 **channels_last + TensorRT/torch.compile + 量化** 拿到接近极限的吞吐。

## 通用 checklist（先一次做完）

1. `model = model.to(memory_format=torch.channels_last)`
2. 输入也 `.to(memory_format=torch.channels_last)`
3. 用 `torch.compile(mode="reduce-overhead")` 或 TensorRT
4. 量化到 FP16/BF16；目标卡 INT8 calibration
5. 固定输入 shape（或 dynamic batch + 静态 H,W）
6. 关 autograd：`torch.inference_mode()`
7. `cudnn.benchmark = True`（固定输入下）；`cudnn.deterministic = False`

```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = model.eval().half().to('cuda').to(memory_format=torch.channels_last)
with torch.inference_mode():
    out = model(x.half().to('cuda', memory_format=torch.channels_last))
```

## 1. channels_last (NHWC) — 必须

cuDNN 在 NHWC 路径下 Tensor Core kernel 选择更多，FP16 conv2d 通常 **1.3-2×** NCHW。

要求**全链路 NHWC**，否则反复 transpose 会更慢：
- conv2d ✅
- BatchNorm2d ✅ (PyTorch 1.7+)
- pooling ✅
- ReLU / GELU / 激活 ✅（elementwise，layout 无关）
- residual add ✅
- 自定义层 ❓ → 自查 `is_contiguous(memory_format=torch.channels_last)`

ViT 的 patch embed 是 `Conv2d(in, dim, kernel=patch, stride=patch)`，同样吃 NHWC 加速。

**验证**：
```python
def check_layout(model, x):
    x = x.to(memory_format=torch.channels_last)
    handles = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            def hook(mod, inp, out, name=n):
                if not out.is_contiguous(memory_format=torch.channels_last):
                    print(f"NCHW fallback at {name}")
            handles.append(m.register_forward_hook(hook))
    model(x); [h.remove() for h in handles]
```

## 2. cuDNN benchmark vs heuristic

```python
torch.backends.cudnn.benchmark = True   # 第一次每个 unique shape 跑 algo 选最佳
```

- **固定 shape**：开 benchmark，几个 warmup step 后稳定
- **shape 变**：benchmark 会反复 retune，p99 飞起 → 关掉，依赖 heuristic

cuDNN v9 的 heuristic 通常已经接近 benchmark 90%+ 性能。

## 3. Conv 算法实现的 trade-off

| 算法 | 时机 | 显存 | 速度 |
|---|---|---|---|
| **Implicit GEMM** | 默认，大多数 conv | 低 | 高 |
| **Winograd (F(2,3)/F(4,3))** | 3×3 stride=1，filter 多 | 中 | 极高（FP16 上 1.5×） |
| **FFT** | 大 kernel（≥ 7×7） | 高 | 视情 |
| **Direct** | 小 in/out channel | 低 | 一般 |

cuDNN benchmark 自动选；手动控制：
```python
import torch.backends.cudnn as cudnn
cudnn.conv.allow_tf32 = True
# 强制 algo（不推荐）
# torch.use_deterministic_algorithms 会限制可选 algo
```

TensorRT 的 `--builderOptimizationLevel=5` 让 builder 试更多 tactic。

## 4. Conv-BN-ReLU 融合

- TensorRT 自动 fuse（永远会做）
- torch.compile 在 `reduce-overhead` 下做 fuse
- eager mode：手动 `torch.nn.utils.fuse_conv_bn_eval(conv, bn)`

```python
def fuse_model(m):
    for name, module in m.named_children():
        if isinstance(module, torch.nn.Sequential):
            modules = list(module.named_children())
            for i in range(len(modules) - 1):
                if isinstance(modules[i][1], torch.nn.Conv2d) and isinstance(modules[i+1][1], torch.nn.BatchNorm2d):
                    fused = torch.nn.utils.fuse_conv_bn_eval(modules[i][1], modules[i+1][1])
                    setattr(module, modules[i][0], fused)
                    setattr(module, modules[i+1][0], torch.nn.Identity())
        else:
            fuse_model(module)
```

## 5. 模型族特定建议

### ResNet / EfficientNet
- channels_last 必开
- INT8 calibration 后 FPS × 2 几乎无损（top-1 掉 < 0.5%）
- TensorRT `--int8 --fp16` 自动混合精度处理 outlier 层

### MobileNet / EfficientNet（depthwise）
- depthwise conv 在 cuDNN 不如其他 conv 高效（low arithmetic intensity）
- **batch 拉大** 才有 Tensor Core 利用
- 或考虑替换为 grouped conv with groups=channels（部分 cuDNN 路径更优）
- 边缘场景：TensorRT 8.6+ 对 depthwise 有专门 plugin

### YOLO 系列
- NMS 在 GPU 上做（TensorRT `BatchedNMSPlugin` / `EfficientNMSPlugin`）
- 后处理 anchor decode 可融合进图（导出时一起写 ONNX）
- INT8 在 detection head 容易掉 mAP → 那部分留 FP16

### ViT / DeiT / BEiT
- patch embed + global avg pool + MLP head 主体 ≈ MHSA + MLP
- MHSA 用 `F.scaled_dot_product_attention`（自动 FA）
- 小 batch（1-8）下 GEMM 不饱和 → 多请求 dynamic batching
- ViT-L 以上 FP8（H100）有戏，TRT-LLM 已支持 vision encoder

### Swin Transformer
- Windowed attention 用自定义 mask，**很可能禁用 FlashAttention** 后端
- 解法：
  - PyTorch 2.4+ 的 `flex_attention` 支持窗口 attention 高效路径
  - 或 xFormers 的 `memory_efficient_attention` + 自定义 attn_bias
- Cyclic shift 的 `roll` 操作在 NHWC 下注意 layout
- Relative position bias 当 INT8 量化敏感层

### DETR / DINO / RT-DETR
- 后端 Transformer 同 ViT
- 前端 ResNet backbone 同 CNN 套路
- Hungarian matcher 在 CPU 比 GPU 快（量小，迁移成本高）

### Segmentation (DeepLab / Mask R-CNN)
- 高分辨率 → 显存大头是激活而非权重 → activation checkpointing 仅对训练有意义；推理用更小 batch 即可
- Dilation conv：cuDNN 有专门 algo，benchmark 开就行
- 后处理 (upsample + softmax) 在 GPU 上做完再 D2H，比 D2H 后 CPU 上做快得多

## 6. INT8 calibration for CNN

```python
# pytorch-quantization (NVIDIA) PTQ 流程
import pytorch_quantization.quant_modules as qm
qm.initialize()
model_q = build_model()
load_pretrained(model_q)

# Collect calibration stats
for name, module in model_q.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        module.enable_calib(); module.disable_quant()

with torch.no_grad():
    for img in calib_loader:        # 500-2000 张
        model_q(img.cuda())

for name, module in model_q.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
        module.load_calib_amax(method='percentile', percentile=99.99)
        module.disable_calib(); module.enable_quant()

# Export to ONNX with QDQ nodes, then TRT --int8
```

精度校验：
- ImageNet top-1 不应掉 > 0.5%
- COCO mAP 不应掉 > 1.0

掉得多 → 找罪魁层（`polygraphy debug precision`），第一/最后/检测头留 FP16。

## 7. 动态分辨率

CNN 服务化常见：输入 H,W 可变（如 OCR、检测）。

- TRT：用 `--minShapes/--optShapes/--maxShapes` 定义 profile
- 多 profile：不同尺度组各一份 engine，根据输入选 profile（cost: 显存翻倍）
- 或者 **强制 padding**：所有输入 padding 到最大尺寸 + mask（推理逻辑里忽略 padding 区）—— 最实用

```python
# 简单 padding 到 32 的倍数（YOLO 风格）
H_pad = (H + 31) // 32 * 32
W_pad = (W + 31) // 32 * 32
x_pad = torch.nn.functional.pad(x, (0, W_pad-W, 0, H_pad-H))
```

## 8. ViT-specific 加速

```python
# torch.compile 配合 SDPA + channels_last
model = torch.compile(vit.to(memory_format=torch.channels_last),
                      mode="reduce-overhead", fullgraph=True)

# 验证用 FlashAttention
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    out = model(x)
```

Token merging / token pruning（推理时合并相似 token）：
- ToMe / EViT，2× 加速精度损失 1-2%
- 适合 ViT-L/ViT-H，小模型收益小

## 9. CV 数据 pipeline 提示

CNN/ViT 推理服务的 H2D 经常成为瓶颈（图片 decode + resize）：
- **DALI** 在 GPU 上做 JPEG 解码 + resize + normalize
- nvJPEG / nvImageCodec：GPU JPEG decode
- 输入端尽量传二进制 bytes，server 内部用 GPU decode

详见 `nv-infer-io-data`。

## 10. 实测命令

```bash
# 基线 latency / fps
python -c "
import torch, time
model = ... .to('cuda', memory_format=torch.channels_last).half().eval()
x = torch.randn(32,3,224,224, device='cuda').to(memory_format=torch.channels_last).half()
with torch.inference_mode():
    for _ in range(20): model(x)   # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): model(x)
    torch.cuda.synchronize()
    print('ms/iter', (time.perf_counter()-t0)/200*1000)
"

# TRT 对比
trtexec --onnx=resnet50.onnx --fp16 --useCudaGraph \
    --shapes=x:32x3x224x224 \
    --warmUp=2000 --duration=10
```

## 反模式

- channels_last 设了 model 没设 input → 入口 transpose，慢
- depthwise conv 用 batch=1 跑还期待 Tensor Core 利用
- cudnn.benchmark 在动态分辨率下开 → autotune 风暴
- Swin / windowed attention 仍期望 FlashAttention 自动启用
- ViT 上 INT8 整网量化没回 FP16 → patch embed 数值崩
- 服务化时 H2D 走 CPU JPEG decode → GPU 干等
- 检测 NMS 在 client/Python 端 → 多 100ms RTT

## See also

- 模型族细节：
  - [`references/models/yolo-series.md`](../../references/models/yolo-series.md) — YOLOv8/9/10/11 全系
  - [`references/models/clip-siglip.md`](../../references/models/clip-siglip.md) — CLIP / SigLIP
- 各卡适配（T4 INT8、L4/L40S、H100、910B、300V Pro）：[`references/hardware/`](../../references/hardware/)
- Ascend 上 CNN/ViT（DVPP+AIPP 全片上 pipeline）：[`ascend-infer-cnn-vit`](../ascend-infer-cnn-vit/SKILL.md)
