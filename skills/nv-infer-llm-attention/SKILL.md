---
name: nv-infer-llm-attention
description: LLM 推理的 attention / KV cache 专项优化——FlashAttention v2/v3、PagedAttention、KV cache 显存预算与量化、prefill/decode disaggregation、chunked prefill、sliding window、speculative decoding、MQA/GQA、推测解码 (EAGLE/Medusa/draft model)、长上下文 (RoPE scaling)、KV offload。当用户问"FlashAttention"、"PagedAttention"、"KV cache"、"prefill 慢"、"decode 慢"、"长上下文"、"speculative decoding"、"MQA/GQA"、"sliding window"、"chunked prefill" 时使用。
---

# LLM Attention & KV Cache Optimization

> LLM 推理 = prefill (compute-bound) + decode (memory-bound)。
> Decode 阶段每个 token 都重读全部 KV cache —— 这是大部分 LLM 慢的根因。

## 核心模型：prefill vs decode

| 阶段 | 计算特征 | 关键瓶颈 | 优化方向 |
|---|---|---|---|
| Prefill (process prompt) | GEMM 大矩阵，compute-bound | TFLOPS | FlashAttention、FP8、TP |
| Decode (generate token) | 全部 KV 读一遍，memory-bound | HBM BW | KV 量化、PagedAttn、speculative、batch |

**指标**：
- TTFT (Time To First Token) = prefill 时间
- TPOT / ITL (Inter-Token Latency) = 平均 decode 时间
- TTLT (Total Latency) = TTFT + TPOT × out_tokens

## 1. FlashAttention — 必开

FlashAttn 2/3 通过 tiling + 在线 softmax，把 attention 从 O(N²) 显存降到 O(N)，同时 1.5-3× 提速。

PyTorch 2.0+：
```python
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# 自动选 FlashAttention backend（Hopper/Ampere）
```

强制 backend / 检查实际跑的：
```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

FA-3 (H100)：
- 用 TMA + WGMMA
- FP8 attention（端到端）
- 比 FA-2 再快 1.5-2×
- pip: `pip install flash-attn --no-build-isolation`（构建久）

哪些情况 FA 不会被用：
- 非 causal 且 mask 复杂（如 Swin window attention 自定义 mask）
- head_dim 不是 32/64/96/128/192/256
- dtype 是 FP32
- 老 PyTorch < 2.2

## 2. PagedAttention（vLLM 风格）

把 KV cache 切成固定大小 block（典型 16 token/block），按需分配，几乎消除碎片：

- 显存利用率从 ~70% 提到 **> 95%**
- 同显存可塞 **2-3×** batch
- 支持 prefix caching（共享 system prompt 的 KV）

实操：直接用 vLLM / TensorRT-LLM 的 paged_kv_cache。**不要自己写**。

```bash
# vLLM 起服务
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --block-size 16

# TRT-LLM
trtllm-build --paged_kv_cache enable \
             --use_paged_context_fmha enable \
             --tokens_per_block 64 \
             ...
```

## 3. KV cache 显存预算与压缩

```
KV_bytes = 2 × n_layer × n_kv_head × head_dim × dtype × seq × batch
```

LLaMA-3-70B (n_layer=80, n_kv_head=8, head_dim=128, FP16, seq=8K, batch=32) ≈ **80 GB** —— 单卡 80GB 装不下，必须 TP 或 KV 量化。

降 KV 方案（按 ROI 排序）：
1. **FP8 KV cache**（H100+）：50% 显存，几乎无精度损失
   ```bash
   --kv-cache-dtype fp8_e4m3       # vLLM
   --kv_cache_type=fp8             # TRT-LLM
   ```
2. **INT8 KV cache**（A100）：50% 显存
   ```bash
   --kv-cache-dtype int8 --calculate-kv-scales
   ```
3. **GQA/MQA**：n_kv_head 直接小（模型设计）。LLaMA-3、Mistral 已经 GQA。
4. **Sliding window attention**：Mistral 4K window，长 prompt 也只保 4K KV
5. **KV cache offload**：长不活跃的 sequence 的 KV 移到 CPU（NVLink-C2C / PCIe）
6. **KV cache eviction / compression**：H2O、Scissorhands、StreamingLLM（学术方案，慎用）

## 4. Prefill / Decode Disaggregation

两个阶段计算特征完全不同：
- Prefill: 高 FLOPS 利用，长 batch 来回搞干扰 decode
- Decode: 低 FLOPS，频繁打断 prefill 也会让 prefill TTFT 飙

**解法**：**分到不同 GPU/实例**。
- vLLM：`--disable-async-output-proc=false`、`--enable-chunked-prefill`
- TRT-LLM + Triton: 用 disaggregated server mode
- NVIDIA Dynamo (NIXL)：分布式 KV 传输，把 prefill 的 KV 直接 NVLink/IB 推到 decode 节点

**Chunked Prefill**（更简单的折中）：
把长 prompt 切成小 chunk，每个 step 一边推一点 prefill、一边继续 decode：
```bash
vllm serve ... --enable-chunked-prefill --max-num-batched-tokens 4096
```
代价：单请求 TTFT 略变长，但整体吞吐显著上升。

## 5. Continuous Batching（持续批处理）

老式静态 batch：8 个请求一个 batch，最长那个不结束所有人都等。
Continuous batching：每个 step 结束后把 done 的 sequence 替换为新请求，**所有人 GPU 永不空转**。

vLLM、TRT-LLM、TGI、SGLang 都默认开。直接用现成 server，不要自己写。

```bash
vllm serve ... --max-num-seqs 256 --max-num-batched-tokens 16384
```

## 6. Speculative Decoding

用小 draft model 一次猜 k 个 token，target model 一次验证 k 个。decode 时 HBM 带宽不变但产出 k× token。

实测加速 **1.5-3×**，依赖 draft 命中率。

```bash
# vLLM
vllm serve meta-llama/Llama-3-70B-Instruct \
    --speculative-model meta-llama/Llama-3-8B-Instruct \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1

# TRT-LLM 内置 Medusa / EAGLE
trtllm-build --medusa_num_heads 4 --medusa_num_layers 1 ...
```

变体：
- **Draft model**：另一个小模型（同 family 更准）
- **Medusa**：在 target model 上加多个 prediction head
- **EAGLE / EAGLE-2**：训练 hidden state 上的小网络猜 next token，**通常最好**
- **n-gram / lookup**：完全无模型，结构化任务有效（代码补全）

## 7. 长上下文（128K+）

挑战：KV 爆显存 + attention 二次复杂度。

工具：
- **RoPE scaling / YaRN**：把 8K 模型外推到 128K（推理时）
- **Chunked prefill**：避免一次塞 128K token 引爆 KV
- **Paged KV + prefix caching**：相同 system prompt 共享
- **Ring attention / Context parallelism (CP)**：长 seq 切到多卡
- **KV offload**：超长 inactive context 放 Grace CPU（GH200）

```bash
# vLLM long-context flags
vllm serve ... \
    --rope-scaling '{"type": "linear", "factor": 4.0}' \
    --max-model-len 131072 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192
```

## 8. Attention 变体兼容性

| 变体 | FlashAttn 支持 | 备注 |
|---|---|---|
| Causal | ✅ | 默认 |
| Bidirectional | ✅ | encoder |
| Sliding window | ✅ FA-2.4+ | Mistral |
| ALiBi | ✅ FA-2.3+ | BLOOM |
| Soft cap (Gemma) | ✅ FA-2.5+ | logit cap |
| Custom mask (Swin) | ❌ | 退化到 SDPA math |
| Multi-query / GQA | ✅ | 自动处理 |

## 9. 精度叠加策略

| 模型规模 | 单卡 | 多卡 | 推荐栈 |
|---|---|---|---|
| 7B / 8B | 80GB OK | — | FP16/BF16 + FA + paged KV + FP8 KV |
| 13B | 80GB OK | — | FP8 全栈 / W4A16 + FP8 KV |
| 30B-34B | 80GB FP16 紧 | TP=2 | FP8 / W4A16-AWQ |
| 70B | OOM | TP=4 (FP16) / TP=2 (W4A16) | TP=2 W4A16 + FP8 KV + speculative |
| 405B | — | TP=8 (FP8) / TP=4 (W4A16) | FP8 / W4A16 + EAGLE-2 |

## 10. 验证命令

```bash
# vLLM benchmark
python benchmarks/benchmark_serving.py \
    --backend vllm --model meta-llama/Llama-3-70B-Instruct \
    --dataset-name sharegpt --num-prompts 1000 \
    --request-rate 10

# 关键指标
# - mean TTFT  (prefill)
# - mean TPOT  (decode per-token)
# - p99 latency
# - successful requests / s
# - throughput (output tokens/s)
```

```bash
# nsys 看 prefill 与 decode 的 kernel 分布
nsys profile -t cuda,nvtx -o vllm.qdrep python -m vllm.entrypoints.openai.api_server ...
# 在 timeline 上找 'flash_fwd_kernel' / 'paged_attn'
```

## 反模式

- 用 PyTorch eager + KV list of tensor → 每次 cat 重分配，慢且爆
- prefill 和 decode 用同一个 batcher 不分离 → TTFT 与 TPOT 互相打架
- KV cache FP8 但 head_dim 不是 16 的倍数 → kernel fallback 慢
- speculative 用毫不相关的 draft model → 命中率 < 30%，反而慢
- 在 attention 内部加 if 切分支 → 自动 fuse 失效
- 长上下文不 chunked prefill → 单请求把 GPU 占住 5 秒，p99 崩
- 没 prefix caching → 同 system prompt 重复编码千万次

## See also

- LLM 推荐配置与显存预算：
  - [`references/models/qwen3-8b.md`](../../references/models/qwen3-8b.md) — Qwen3-8B 配置模板
  - [`references/models/minicpm-v.md`](../../references/models/minicpm-v.md) — VLM (MiniCPM-V) 视觉前端 + LLM
- 各卡 HBM/NVLink 决定 KV 上限：[`references/hardware/nvidia-h100.md`](../../references/hardware/nvidia-h100.md) / [`nvidia-b200.md`](../../references/hardware/nvidia-b200.md)
- Ascend 上 LLM 等价栈（MindIE-LLM、FlashAttentionScore、INT8 KV）：[`ascend-infer-llm-stack`](../ascend-infer-llm-stack/SKILL.md)
