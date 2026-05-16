# Qwen3-8B Inference (并兼容 Qwen2.5-7B / 后续 Qwen3.x 同族)

> 阿里 2025-04 发布的 Qwen3 dense 系列中的 8B 档。Apache 2.0。
> **Hybrid thinking** —— 同一权重切换 *thinking mode* 与 *non-thinking mode*，是这一代的招牌。
> Qwen2.5-7B 与 Qwen3-8B 架构同族（decoder-only + GQA），本文调优要点对两者通用。

> 关于版本：截至本文撰写，公开发布的 Qwen 系列有 Qwen2.5（2024-09）与 Qwen3（2025-04）；若你手上是 "Qwen 3.5 8B" 或更新版本，**架构很可能仍是 GQA decoder-only + RoPE**，下面的推理 / 量化 / 部署套路通用，只需对齐 vocab / hidden_dim / layer count 数字。

## 架构速记

| 项 | Qwen3-8B | Qwen2.5-7B |
|---|---|---|
| Params | 8.19 B | 7.62 B |
| Layers | 36 | 28 |
| Hidden | 4096 | 3584 |
| FFN intermediate | 12288 | 18944 |
| Heads (Q) | 32 | 28 |
| KV heads (GQA) | 8 | 4 |
| Head dim | 128 | 128 |
| Vocab | 151936 | 152064 |
| RoPE theta | 1,000,000 | 1,000,000 |
| Context (native) | 32K | 32K (128K with YaRN) |
| Activation | SwiGLU | SwiGLU |
| Normalization | RMSNorm | RMSNorm |
| Tying weights | No | No |
| Modes | think + non-think | non-think only |

GQA ratio：8 KV head 对 32 Q head（4:1）—— 推理时 KV cache 比 LLaMA-3-8B（同 4:1 ratio）略大，因为 layer 数更多（36 vs 32）。

## 显存预算

### 权重
- FP16 / BF16：**16.4 GB**
- INT8 W8A8：~8.2 GB
- INT4 W4A16 (AWQ/GPTQ)：~4.5 GB
- FP8 (Hopper)：~8.2 GB
- FP4 (Blackwell)：~4.3 GB

### KV cache (per token, GQA)
```
KV/token = 2 × 36 × 8 × 128 × dtype = 73,728 × dtype bytes
        = 144 KB (FP16) / 72 KB (FP8/INT8) / 36 KB (FP4)
```

8K context × batch=32 × FP16 KV = **36 GB** —— 单 H100 80GB 上 batch=16 + FP8 KV 才够舒服跑 32K context。

### 推荐配置（KV FP8）
| 卡 | 权重 dtype | KV dtype | max ctx × batch |
|---|---|---|---|
| T4 16GB | INT4 W4A16 | INT8 | 4K × 4 (紧) |
| L40S 48GB | FP16 | FP8 | 16K × 16 |
| A100 80GB | FP16 | FP8 | 32K × 32 |
| H100 80GB | FP8 | FP8 | 32K × 64 |
| H200 141GB | FP8 | FP8 | 128K × 32 (YaRN) |
| B200 192GB | FP4 | FP4 | 128K × 128 |
| Atlas 910B 64GB | W4A16 | INT8 | 16K × 16 |

## Hybrid Thinking 模式

Qwen3 独有：单权重双模式。
- **Thinking mode**：模型先在 `<think>...</think>` 内做 CoT 推理，再输出最终答案
- **Non-thinking mode**：直接出答案

切换：
```python
# Transformers
text = tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=True)   # 或 False

# vLLM (OpenAI API 扩展)
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
```

**推理影响**：
- thinking mode 平均输出 token 数 2-10× non-thinking
- 单请求 TTLT 显著长，但准确率（reasoning / 数学 / 代码）大幅上升
- 服务化需：增大 `max_new_tokens`、调高 max-num-batched-tokens（continuous batching 下不会拖死 batch）

## 量化与精度损失（参考 MMLU / HumanEval / GSM8K）

| 方案 | 工具 | MMLU 损失 | GSM8K 损失 | 备注 |
|---|---|---|---|---|
| FP16/BF16 | baseline | 0 | 0 | |
| FP8 E4M3 | TE / TRT-LLM `--use_fp8` | -0.2 ~ -0.5 | -0.5 ~ -1.0 | H100+ |
| INT8 W8A8 (SmoothQuant) | llmcompressor | -0.5 ~ -1.0 | -1.0 ~ -2.0 | |
| INT4 W4A16 GPTQ | autoGPTQ | -0.5 ~ -1.5 | -1.5 ~ -3.0 | group_size=128 |
| INT4 W4A16 AWQ | autoAWQ | -0.3 ~ -1.0 | -1.0 ~ -2.0 | 略好于 GPTQ |
| INT4 (Ascend MindIE) | MindIE quantizer | -0.5 ~ -1.5 | -1.5 ~ -3.0 | |
| FP4 (NVFP4/MXFP4) | nvidia-modelopt | -0.5 ~ -1.5 | -1.5 ~ -3.5 | Blackwell |

**Thinking mode 下量化损失更敏感** —— CoT 链每步精度损失累计，复杂数学题量化后掉幅更大。建议生产环境跑同 prompt 集做精度回归。

## 部署 — vLLM (主流)

```bash
pip install vllm>=0.7

vllm serve Qwen/Qwen3-8B \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes
```

`--reasoning-parser qwen3` 让 vLLM 把 `<think>...</think>` 单独 stream 出来（OpenAI 兼容 `reasoning_content` 字段）。

### 量化版本

```bash
# W4A16 AWQ
vllm serve Qwen/Qwen3-8B-AWQ \
    --quantization awq \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 32768
```

### 长上下文（128K + YaRN）

```bash
vllm serve Qwen/Qwen3-8B \
    --rope-scaling '{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max-model-len 131072 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8_e4m3
```

## 部署 — TensorRT-LLM (H100/B200 极致延迟)

```bash
# Checkpoint
python convert_checkpoint.py \
    --model_dir Qwen3-8B/ \
    --output_dir trt_ckpt/ \
    --dtype bfloat16

# Engine
trtllm-build --checkpoint_dir trt_ckpt/ \
    --output_dir trt_engine/ \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --use_fused_mlp enable \
    --use_paged_context_fmha enable \
    --paged_kv_cache enable --remove_input_padding enable \
    --kv_cache_type=fp8 \
    --max_batch_size 256 --max_input_len 32768 --max_output_len 4096 \
    --max_num_tokens 16384

# FP8 quantize
python ../quantization/quantize.py --model_dir Qwen3-8B/ \
    --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 \
    --output_dir trt_ckpt_fp8/ --calib_size 512
```

H100 上 Qwen3-8B FP8 单卡 **decode 吞吐 4000-6000 tok/s（batch 拉满）**，单流约 150-220 tok/s。

## 部署 — SGLang (agent / 工具调用)

```bash
python -m sglang.launch_server --model Qwen/Qwen3-8B \
    --port 30000 --tp 1 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype fp8_e5m2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen
```

SGLang 的 RadixAttention 对 agent / ReAct 场景（同前缀多轮）显著快。

## 部署 — Ascend 910B (MindIE-LLM)

```bash
mindie-cli convert --model_path Qwen3-8B/ \
    --output_path qwen3_8b_om/ \
    --soc_version Ascend910B3 \
    --max_batch_size 32 --max_seq_len 32768 \
    --enable_paged_kv true \
    --quant w8a8        # 或 w4a16

mindieservice_daemon --config qwen3_service.json
```

`qwen3_service.json` 关键项：
```json
{
  "modelName": "Qwen3-8B",
  "tp": 1,
  "pagedAttention": true,
  "continuousBatching": true,
  "maxBatchSize": 32,
  "maxSeqLen": 32768,
  "kvCacheDtype": "int8",
  "enablePrefixCache": true
}
```

性能参考（910B3 单卡，W8A8）：decode 单流 80-130 tok/s，batch 16 总吞吐 1500-2500 tok/s。

## Speculative Decoding 配方

Qwen3-8B 自身较小，**用更小的 draft model 加速**：

```bash
# vLLM
vllm serve Qwen/Qwen3-8B \
    --speculative-model Qwen/Qwen3-0.6B \
    --num-speculative-tokens 5 \
    --speculative-draft-tensor-parallel-size 1 \
    --kv-cache-dtype fp8_e4m3
```

Qwen3-0.6B 作 draft 通常 **1.5-2× 加速**（命中率 60-75%）。

Qwen3 系列内 draft / target 配对建议：
- Qwen3-8B target ← Qwen3-0.6B / Qwen3-1.7B draft
- Qwen3-32B target ← Qwen3-4B / Qwen3-8B draft

## Tool Calling

Qwen3 native 支持 OpenAI 风格 tool calling：
```bash
vllm serve Qwen/Qwen3-8B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}
        }, "required": ["city"]}
    }
}]
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "北京天气？"}],
    tools=tools
)
```

## 调优要点

1. **vocab 大 (151936) → embedding/lm_head 是显存大头**：FP8 KV 后 GEMM 也吃 vocab 维度，pad 到 152064 (÷64=2376) 让 Tensor Core 走顺
2. **GQA 4:1 但 KV head 8 较小**：FP8 KV 加 paged 几乎是必做
3. **thinking mode 输出长 → continuous batching 必开**：否则长 reasoning 会阻塞 batch
4. **chunked prefill 必开**：长 prompt（含历史/RAG 上下文）能平滑 TTFT
5. **prefix cache 配 system prompt**：agent 场景常见同 system，命中率 > 80%
6. **shape 桶化**：`max-num-batched-tokens` 设为 8192/16384，让动态 batch 不抖
7. **YaRN scaling 触发条件**：context > 32K 才需要 `--rope-scaling`，短 context 时反而轻微掉点
8. **Tokenizer**：`use_fast=True`（默认 fast）；高 QPS 启 `--tokenizer-pool-size 8`
9. **dtype 选择**：BF16 > FP16（Qwen3 训练用 BF16，FP16 偶有数值边界问题）

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 用 transformers `generate()` 单流跑生产 | 慢、无 batching | 改 vLLM / TRT-LLM / MindIE |
| FP16 → BF16 没切，长 prompt 出现 NaN | softmax 数值 | dtype=bfloat16 全程 |
| 没启 reasoning parser，thinking 内容混进输出 | 客户端看到 `<think>` 标签 | `--reasoning-parser qwen3` |
| max_new_tokens=128 跑 thinking | CoT 被截断，答案错误 | thinking 场景 ≥ 2048 |
| YaRN factor 设过大 | 短 prompt 精度掉 | factor 按实际 context / native 算 |
| KV FP8 但 calibration 没收 KV scale | decode 早期数值乱 | warmup 200+ step 让 scale 稳 |
| W4A16 AWQ + Marlin kernel sm_75 fallback | T4 上慢 | T4 上用 GPTQ 或换 Ampere+ |
| Ascend 上自家 op 缺 | 转换失败 | CANN 8.0+ 已覆盖 Qwen2/3；老 CANN 升一下 |
| Tool calling 解析失败 | JSON 不闭合 | `--tool-call-parser hermes` 配 Qwen3 |
| 多轮 thinking 累计 context | 几轮后 context 爆 | 客户端清掉历史 `<think>` 块 |
| vocab pad 不对齐 16 | FP8 GEMM 走慢 path | pad 到 152064 |
| BF16 模型 INT4 AWQ 后 thinking 链质量降 | reasoning 任务掉 3-5% | thinking 关键场景考虑 FP8 不走 INT4 |

## 性能参考（社区 + 实测，量级）

| 卡 | dtype/quant | KV | 单流 tok/s | batch=32 总 tok/s |
|---|---|---|---|---|
| T4 | INT4 GPTQ (sm_75 慢 path) | INT8 | 15-25 | ~150 |
| L4 | FP16 (24GB OK) | FP8 | 80-120 | 1200-1800 |
| A100 80G | BF16 | FP8 | 130-180 | 3000-4500 |
| H100 80G | FP8 | FP8 | 200-260 | 5000-7000 |
| H200 141G | FP8 + long ctx | FP8 | 200-260 | 5000-7000 (ctx 长更显优势) |
| B200 192G | FP4 | FP4 | 350-500 | 10000+ |
| Atlas 910B3 | W8A8 (MindIE) | INT8 | 100-150 | 2000-3000 |

Thinking mode 下 throughput 视用户输入分布而定，CoT 长度通常占输出 50-80%。

## 与本仓库 skill 联动

- `nv-infer-llm-attention` —— KV cache 预算、FlashAttention、paged KV
- `nv-infer-serving` —— vLLM / TRT-LLM 部署
- `nv-infer-precision` —— FP8/INT4 量化路线
- `ascend-infer-llm-stack` —— Ascend 910B 上的 MindIE-LLM 部署
- `references/hardware/nvidia-h100.md` / `nvidia-b200.md` / `huawei-atlas-910b.md` —— 选卡

## 参考

- Qwen 官方博客（架构细节、benchmark）
- Hugging Face Qwen3-8B model card
- vLLM Qwen3 集成文档
- TensorRT-LLM Qwen2/3 examples
- MindIE-LLM model zoo (Qwen2/3 支持)
