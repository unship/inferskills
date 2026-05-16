# NVIDIA T4 (Tesla T4)

> Turing 时代的低功耗推理工马。2018 发布，2024 起逐步退役但仍大量在用。
> 你买它就是冲着 **70W 被动 + 16GB + INT8 Tensor Core** 三件套，主打 CNN/ViT 与小 NLP 推理。

## 一句话画像

```
Turing TU104 · sm_75 · 16GB GDDR6 · 320 GB/s · 70W 被动 · PCIe Gen3 x16 · 无 NVLink
INT8 Tensor: 130 TOPS · FP16 Tensor: 65 TFLOPS · 无 BF16/FP8/TF32 Tensor Core
```

## 核心规格

| 项 | 值 |
|---|---|
| 架构 | Turing TU104 (sm_75) |
| Compute Capability | 7.5 |
| SM 数 | 40 |
| CUDA Cores | 2560 |
| Tensor Cores | 320（2nd gen, FP16/INT8/INT4/INT1） |
| RT Cores | 40 |
| L2 Cache | 4 MB |
| 显存 | 16 GB GDDR6, 256-bit, 10 Gbps |
| 显存带宽 | 320 GB/s |
| 时钟 (boost) | 1590 MHz |
| FP32 (vector) | 8.1 TFLOPS |
| FP16 (vector) | 16.2 TFLOPS |
| **FP16 Tensor** | **65 TFLOPS** |
| **INT8 Tensor** | **130 TOPS** |
| **INT4 Tensor** | **260 TOPS** |
| INT1 Tensor | 1040 TOPS (二值，研究向) |
| FP64 | 254 GFLOPS（很慢，不要做 FP64） |
| PCIe | Gen3 x16 (~ 16 GB/s 单向) |
| NVLink | **无** |
| NVENC | 1 (Turing 7th gen) |
| NVDEC | 2 (Turing 7th gen) |
| TDP | 70W |
| 冷却 | 被动 (服务器风道供气) |
| 形态 | HHHL，单插槽 |

## 软件栈

| 组件 | 最低版本 | 推荐 |
|---|---|---|
| Driver | R450+ (LTS) | R535+ |
| CUDA Runtime | 10.0+ | 12.x (向后兼容) |
| cuDNN | 7.6+ | 9.x |
| TensorRT | 5.0+ | 10.x (注意 Turing 仍支持，TRT-LLM 也支持) |
| PyTorch | 1.x | 2.3+ |

**重要**：从 CUDA 13 起 Turing 可能进入 deprecated 名单（NVIDIA 公告窗口期），生产升级 CUDA/PyTorch 之前先验证 sm_75 仍在 build 列表。

## 推理实用特性

- ✅ **INT8 Tensor Core** —— TRT 的 INT8 PTQ 路径主战场
- ✅ **NVDEC × 2** —— 视频流推理（H.264/H.265/VP9）成本极低
- ✅ **小卡型** —— 1U 服务器塞 4-8 张
- ❌ 没有 BF16 / TF32 / FP8 / FlashAttention v2/v3 的硬件加速
- ❌ 没有 NVLink，多 T4 走 PCIe Gen3 = 16 GB/s 单向（TP 几乎不可行）
- ❌ 没有 MIG（A100+ 才有）
- ❌ 没有 Confidential Compute
- ❌ Tensor Core 不支持 BF16，FP16 训练 / 推理要小心 overflow

## 适合 / 不适合的 workload

| 类型 | T4 上 |
|---|---|
| ResNet-50 / EfficientNet INT8 推理 | ✅ 性价比扛打 |
| ViT-B/16 FP16 推理 | ✅ 但 batch 拉不大 (16GB 限制) |
| YOLO 系列 + NVDEC 视频 | ✅ 经典组合 |
| BERT-Base / DistilBERT FP16 | ✅ |
| Whisper-base/small | ✅ |
| Stable Diffusion 1.5 INT8/FP16 | ⚠️ 慢但可用，batch=1 出图 3-5s |
| LLM 7B FP16 | ⚠️ 16GB 装不下，必须 INT4 |
| LLM 7B INT4 (AWQ/GPTQ) | ⚠️ Turing 上 Marlin / GPTQ 部分 kernel 没优化路径，性能远不如 Ampere |
| LLM 13B+ | ❌ 显存装不下 |
| 训练 | ❌ 显存与 BW 都不够 |
| FlashAttention | ❌ 官方 FA v2/v3 不支持 sm_75 (FA v1 有支持) |

## TensorRT 推理样板（CNN / ViT）

```bash
# 构建 engine
trtexec --onnx=resnet50.onnx \
    --int8 --fp16 \
    --calib=calib.cache \
    --saveEngine=resnet50_t4_int8.plan \
    --shapes=x:32x3x224x224 \
    --useCudaGraph \
    --builderOptimizationLevel=5 \
    --memPoolSize=workspace:4096

# 跑基线
trtexec --loadEngine=resnet50_t4_int8.plan \
    --useCudaGraph --warmUp=2000 --duration=10 --avgRuns=200
```

**典型 T4 数字**（参考）：
- ResNet-50 INT8 + TRT + bs=32：~6500 img/s
- YOLOv8-N INT8 + TRT + bs=1：~3 ms/frame
- BERT-Base FP16 + TRT，seq=128：~3 ms/req

## 调优要点

1. **必开**：`-pm 1` 持久化、固定时钟、TRT INT8 + CUDA Graph
2. **batch 拉满到显存边界** —— 16GB 是硬约束，留 1-2GB workspace
3. **避免**：自定义 CUDA kernel 用 Hopper 才有的 PTX（`mma.sync` 形态都不一样）
4. **不要**：在 T4 上跑 vLLM/TRT-LLM 大模型期待与 H100 同等行为
5. **NVDEC 与推理 overlap**：用 `DeepStream` 或 Triton DALI backend，让 video decode 走 NVDEC，CUDA 干 inference
6. **多 T4 节点**：每张卡跑独立服务实例 (DP)，不要试 TP

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| `BF16` op 在 T4 上 fallback | 速度跳水 5-10× | 用 FP16 / INT8 |
| `torch.compile` mode="max-autotune" 在 sm_75 慢 | autotune 选不到好 tile | 用 "reduce-overhead" |
| `flash_attn` import 报 sm_75 unsupported | crash | 退回 `F.scaled_dot_product_attention`，PyTorch 选 mem-efficient backend |
| TRT INT8 calibration 后掉 1%+ | scale 选错 | 增 calibration 数据 + percentile 99.99 |
| 多 T4 跑 NCCL all-reduce 极慢 | PCIe Gen3 瓶颈 | 改 DP 部署，避免 collective |
| `cudnn.benchmark=True` 在 dynamic shape 下风暴 | p99 飙 | 关掉或固定 shape |
| NVDEC + inference 抢 PCIe | 性能跌 | NVDEC frame 留在 GPU 不下回 CPU |

## 实测命令（验证基线）

```bash
# 卡况
nvidia-smi -q -i 0 | head -40
nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu --format=csv -l 1

# 持久化 + 锁时钟
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 5000,1590    # mem,gfx (T4 推荐值)

# CUDA / TRT 自检
nvidia-smi --query-gpu=compute_cap --format=csv
trtexec --help | head

# Microbenchmark
# INT8 GEMM 峰值 (cublasLt int8 path)
# 用 NVIDIA 的 cuBLAS benchmark 或 nccl-tests 单卡 sanity
```

## 变体

T4 只有一个主 SKU（Tesla T4 16GB）。
**注意区分**：
- **T4**（Tesla T4，HHHL 70W 被动）：本文件主角，数据中心
- **T4G**（极少见，标识相同的 OEM 变体）
- **TeslaT10** 等是不同代，别混
- 千万别把 RTX 系列 Turing 卡（RTX 20xx）当 T4 部署 —— 它们有 NVENC 限制、被动散热不支持、驱动分支不同

## 退役 / 替换路径

NVIDIA 后续低功耗推理卡推荐路径：
- **L4** (Ada, 2023) —— T4 直接接班，72W，24GB，FP8 起步 (Ada 引入 FP8)
- **L40 / L40S** (Ada) —— 高端推理 + 训练（300W）
- 国产化路径：Atlas 300V Pro (本目录另一文件)

## 进一步阅读

- NVIDIA T4 产品页（官方 datasheet）
- TensorRT Best Practices: INT8 Calibration
- 本仓库 [`../../skills/nv-infer-precision/SKILL.md`](../../skills/nv-infer-precision/SKILL.md) — INT8 量化 + calibration
- 本仓库 [`../../skills/nv-infer-cnn-vit/SKILL.md`](../../skills/nv-infer-cnn-vit/SKILL.md) — T4 上最常见的 workload 调优
