---
name: nv-infer-precision
description: NVIDIA GPU 推理的数值精度优化——FP16/BF16/TF32/FP8 (E4M3/E5M2)/INT8/INT4 weight-only、2:4 structured sparsity、SmoothQuant/AWQ/GPTQ/RTN、QAT vs PTQ、calibration、KV cache 量化、精度校验流程。当用户问"量化"、"INT8/INT4/FP8"、"AWQ/GPTQ"、"SmoothQuant"、"mixed precision"、"BF16 vs FP16"、"Tensor Core"、"精度损失"、"sparsity"、"2:4 稀疏" 时使用。
---

# Precision & Quantization for Inference

> 在 NVIDIA 上，**精度=速度+显存**。同一模型从 FP16 → FP8 通常再快 1.6-2×，再省 KV 显存 50%。

## 精度选项一览

| 格式 | bits | 用途 | 支持卡 |
|---|---|---|---|
| FP32 | 32 | 调试基线 / 个别 sensitive 层 | 全系 |
| TF32 | 19 | A100+ 训练默认；推理少用 | A100+ |
| BF16 | 16 | LLM 训练 / 推理默认 | A100+ |
| FP16 | 16 | CNN/ViT 推理默认 | Volta+ |
| FP8 E4M3 | 8 | activation / weights，主流 | Hopper+ |
| FP8 E5M2 | 8 | gradient / 反向 | Hopper+ |
| MXFP8/MXFP4 | 8/4 | Blackwell micro-scaling | Blackwell |
| INT8 (W8A8) | 8 | CNN/ViT/LLM 推理 | Turing+ |
| INT4 weight-only | 4 (W) 16 (A) | LLM 内存压缩 | Ampere+ (Marlin/Machete) |
| 2:4 structured sparse | — | 与 FP16/BF16/FP8/INT8 叠加 | Ampere+ |

## 通用决策树

```
单卡推理目标：
  ├─ 模型 < HBM 一半，追极致 latency → FP8 (H100+) / FP16
  ├─ 模型 ≈ HBM 大小，吞吐为先     → FP8 / INT8 W8A8
  ├─ 模型 > HBM                     → INT4 weight-only (AWQ/GPTQ) + FP16 act
  └─ 长上下文 KV 爆显存             → KV cache 量化到 FP8/INT8（独立于权重）
```

## CNN / ViT 量化

### FP16 / BF16（最快上手）

```python
model = model.half().cuda()                 # FP16
# 或
model = model.bfloat16().cuda()             # BF16（数值范围更稳）
with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
    out = model(x.half())
```

BF16 vs FP16 选择：
- CNN/ViT：FP16 通常够用（动态范围小）
- LLM：BF16 更稳（避免 attention softmax 下溢）
- Hopper：两个都走 Tensor Core，速度等同

### INT8（PTQ，推荐 TensorRT 路径）

```bash
# 1. 导出 ONNX
# 2. 准备 ~500-1000 张 calibration 图片
# 3. trtexec int8 build
trtexec --onnx=model.onnx --int8 --fp16 \
        --calib=calib.cache \
        --saveEngine=m_int8.plan
```

精度校验：
```bash
polygraphy run model.onnx m_int8.plan --trt --validate \
    --abs 1e-2 --rel 1e-2 --check-error-stat mean
```

CNN/ViT 上 INT8 通常 **掉 < 1% top-1**；ResNet/EfficientNet/ViT-B 都能稳定通过。
难量化：dynamic range 极大的层（如 Swin 的 relative position bias）—— 设为 FP16 例外。

### 2:4 Structured Sparsity（叠加用）

权重每 4 个元素中 2 个为 0，Tensor Core 跳过它们，吞吐 **理论 2×**。

```python
from torch.sparse import to_sparse_semi_structured
weight_sp = to_sparse_semi_structured(weight)
```

实际增益 1.2-1.5×（受 epilogue/IO 限制），CNN/ViT 上叠加 FP16/INT8 都可。
需要 fine-tune 恢复精度（直接 prune 会掉 1-3%）。NVIDIA `apex/contrib/sparsity` 有现成工具。

## LLM 量化

### W8A8 SmoothQuant

把激活 outlier 的 scale 转移到权重，让 INT8 都好量化。

```python
# 用 llm-compressor / autosmooth / TensorRT-LLM 自带
trtllm-quantize --model_dir llama3_8b/ \
    --output_dir quant/ \
    --qformat int8_sq \
    --calib_size 512
```

### W4A16 (AWQ / GPTQ)

权重 4-bit（per-group scale），激活 FP16/BF16。LLM 推理的"显存压缩王者"。

```bash
# AutoAWQ
python -m awq.entry --model_path llama3_70b/ --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq.pt
python -m awq.entry --load_awq awq.pt --q_backend real --dump_quant model_awq.pt

# TensorRT-LLM int4_awq
trtllm-quantize --model_dir llama3_70b/ --qformat int4_awq --awq_block_size 128 \
    --output_dir quant_awq/
```

Marlin / Machete kernel 让 W4A16 在 Ampere/Hopper 上达 **1.5-2×** decode 吞吐（相比 FP16）。
vLLM 0.6+、TRT-LLM 0.10+ 都已集成。

### FP8 (E4M3) — Hopper / Blackwell 推荐

```python
# Transformer Engine
import transformer_engine.pytorch as te
fp8_recipe = te.recipe.DelayedScaling(fp8_format=te.recipe.Format.E4M3,
                                       amax_history_len=16, amax_compute_algo="max")
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(x)
```

```bash
# TRT-LLM
trtllm-build --use_fp8 --use_fp8_context_fmha enable ...
trtllm-quantize --qformat fp8 ...
```

FP8 在 LLM 上典型：
- 速度 ≈ INT8，精度比 INT8 好（特别在小 batch / 短 prompt）
- KV cache FP8 几乎免费 50% 显存节省
- 注意：amax history 需要在 calibration / warmup 中收集稳定

### KV cache 量化（独立于权重）

```bash
# TRT-LLM
trtllm-build --kv_cache_type=fp8 ...

# vLLM
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --quantization awq \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales
```

KV cache 通常占 LLM 显存 30-60%，FP8 KV 让你同时多塞 ~2× 并发 batch，吞吐直接翻倍。

## QAT vs PTQ

| | PTQ (post-training) | QAT (quantization-aware training) |
|---|---|---|
| 成本 | 几小时 calibration | 重新 fine-tune 几天 |
| 精度 | 通常掉 0-2% | 几乎无损 |
| 何时用 | 大部分情况 | INT4/INT2、稀有精度敏感模型 |

LLM 实操几乎全部 PTQ：AWQ/GPTQ/SmoothQuant/FP8。
CNN/ViT 上对精度极敏感（医疗、低误检 SLA）才上 QAT。

## Calibration 数据

- **数量**：512-2048 样本通常够（更多边际收益小）
- **分布**：贴近线上分布！否则量化 scale 错，精度大跳
- **激活统计**：per-tensor max / per-channel max / percentile (99.99%)
- 对 LLM：用代表性 prompt（含长短、多领域）

## 精度回归验证流程

```
1. 跑 FP32 / FP16 baseline，保存 N 个代表样本输出
2. 量化后跑同样输入
3. 计算指标：
   - CNN: top-1/top-5、mAP、DICE
   - LLM: perplexity（WikiText, C4）、benchmark 任务（MMLU, GSM8K）
4. 阈值（可调）：CNN top-1 掉 < 0.5%、LLM PPL 升 < 1%、MMLU 掉 < 0.5
5. 失败时：找罪魁层 → 该层留高精度（mixed precision）
```

工具：
- TRT：`polygraphy debug precision --check abs-rel ...`
- LLM：`lm-evaluation-harness` 直接跑 MMLU/HellaSwag/HumanEval

## 混合精度策略（mixed precision）

如果整模型 INT8/FP8 掉太多：
- **敏感层留 FP16/BF16**：通常是第一层、最后一层、attention softmax、layernorm
- TRT：`setPrecision(layer, kFLOAT)` 单独标
- TRT-LLM：`--strongly_typed=False` + 用 fallback
- Transformer Engine：`fp8_autocast` 自动跳过 norm 等

## 反模式

- 量化完不跑精度回归 → 上线后用户投诉
- 用一张图 calibration → scale 完全失真
- INT4 不开 group quant (group_size=128) → 精度崩盘
- FP8 不做 amax warmup → 第一 batch 数值爆
- KV cache FP8 但 calibration 阶段 KV scale 没收 → decode 早期 garbage
- BF16 在 Volta 跑（不支持）→ fallback 到 FP32，反而慢
- vocab/hidden 不对齐到 16/64 → INT8/FP8 Tensor Core 走不顺

## 验证命令

```python
# 看模型 dtype 分布
for n, p in model.named_parameters():
    print(n, p.dtype, p.shape)

# 看实际跑的 kernel 是不是 INT8/FP8（ncu）
# 名字含 'i8'、'imma'、'fp8' / 'qmma' / 'hgmma' 才对
```

```bash
# DCGM 看 INT/FP8 Tensor Active%
dcgmi dmon -e 1004,1009 -c 30
```

## See also

- 各卡支持的精度档（T4 INT8 only / H100 FP8 / B200 FP4）：[`references/hardware/`](../../references/hardware/)
- 模型量化损失参考：
  - LLM 上 FP8 / INT4：[`references/models/qwen3-8b.md`](../../references/models/qwen3-8b.md)
  - VLM 上量化与 OCR 损失：[`references/models/minicpm-v.md`](../../references/models/minicpm-v.md)
  - CNN INT8 上 mAP 损失：[`references/models/yolo-series.md`](../../references/models/yolo-series.md)
- Ascend (无 FP8/FP4) 量化路径：[`ascend-infer-precision`](../ascend-infer-precision/SKILL.md)
