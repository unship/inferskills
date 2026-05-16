---
name: ascend-infer-llm-stack
description: Ascend NPU 上 LLM 推理专项——MindIE-LLM、atb_llm、paged KV、continuous batching、FlashAttentionScore、prefix cache、chunked prefill、speculative decoding on Ascend、TP/PP for LLM、长上下文 / YaRN、Qwen / LLaMA / MiniCPM / DeepSeek 在 910B 上跑。当用户问"昇腾 LLM 推理"、"MindIE-LLM"、"atb_llm"、"NPU 上跑大模型"、"910B vLLM 等价"、"FlashAttentionScore" 时使用。
---

# Ascend LLM Inference Stack

> 与 nv-infer-llm-attention + nv-infer-serving 合并的 Ascend 版本。
> 主战场：**MindIE-LLM**（含 paged KV + continuous batching + 量化），底层 **atb (Ascend Transformer Boost)** 算子库。

## 决策树

```
LLM 在 910B 上要部署 → 几乎 100% 走 MindIE-LLM
  ├─ 模型在 MindIE model zoo 里? → 直接 mindie-cli convert + serve
  ├─ HF 模型未官方支持?           → 看是否兼容 transformers 架构 (Llama/Qwen/Bloom...) 通常可以
  └─ 全新自家架构                 → 写适配 + 可能写 Ascend C op
```

torch_npu + transformers `generate()` 仅用于 debug / 快速验证；生产**必须** MindIE-LLM。

## MindIE-LLM 全流程

### 1. 模型转换

```bash
# Qwen3-8B 示例
mindie-cli convert \
    --model_path /models/Qwen3-8B/ \
    --output_path /models/Qwen3-8B-om/ \
    --soc_version Ascend910B3 \
    --max_batch_size 32 \
    --max_seq_len 32768 \
    --max_prefill_tokens 8192 \
    --enable_paged_kv true \
    --kv_cache_dtype int8 \
    --quant w8a8                 # 或 w4a16 / fp16
```

转换内部走 ATC + atb 算子，**自动用 FlashAttentionScore + paged KV**。

### 2. Service 配置

`/etc/mindie/qwen3_service.json`：
```json
{
  "Version": "1.0.0",
  "LogConfig": { "logLevel": "Info" },
  "ServerConfig": {
    "ipAddress": "0.0.0.0",
    "managementPort": 1026,
    "httpsEnabled": false,
    "maxLinkNum": 1000
  },
  "BackendConfig": {
    "backendName": "mindieservice_llm_engine",
    "modelInstanceNumber": 1,
    "npuDeviceIds": [[0]],
    "ModelDeployConfig": {
      "engineName": "mindieservice_llm_engine",
      "ModelConfig": [{
        "modelName": "Qwen3-8B",
        "modelWeightPath": "/models/Qwen3-8B-om/",
        "worldSize": 1,
        "cpuMemSize": 5,
        "npuMemSize": 60,
        "backendType": "atb"
      }]
    },
    "ScheduleConfig": {
      "maxPrefillBatchSize": 16,
      "maxPrefillTokens": 8192,
      "prefillTimeMsPerReq": 150,
      "maxBatchSize": 32,
      "maxIterTimes": 4096,
      "maxPreemptCount": 0,
      "supportSelectBatch": true,
      "maxQueueDelayMicroseconds": 5000
    }
  }
}
```

### 3. 启动

```bash
mindieservice_daemon --config /etc/mindie/qwen3_service.json
```

OpenAI 兼容 API：
```bash
curl http://localhost:1025/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 512
  }'
```

## atb_llm 直接调用（更灵活）

`atb_llm` 是 atb 算子库 + LLM 调度的 Python/C++ 入口，比 MindIE Service 更"白盒"：

```bash
cd /usr/local/Ascend/atb-models/
python -m examples.run_pa --model_path Qwen3-8B-om/ \
    --input_text "讲个笑话" --max_output_length 256 \
    --kv_cache_dtype int8
```

适合：调研 / benchmark / 自家定制调度逻辑。

## FlashAttentionScore（CANN 7.0+）

Ascend 上的 FlashAttention 等价：
- **aclnnFlashAttentionScore** —— forward
- **aclnnFlashAttentionScoreGrad** —— backward（训练用）
- 自动用于 MindIE-LLM 内部

直接调（自写代码）：
```python
import torch
import torch_npu
from torch_npu.npu.aclnn import flash_attention_score

q = torch.randn(B, S, H, D, device='npu', dtype=torch.float16)
k = torch.randn(B, S, H, D, device='npu', dtype=torch.float16)
v = torch.randn(B, S, H, D, device='npu', dtype=torch.float16)
out = flash_attention_score(q, k, v, head_num=H, scale_value=1.0/(D**0.5),
                            input_layout="BSND", sparse_mode=0)
```

支持模式（CANN 8.0+）：
- causal mask
- sliding window
- GQA / MQA
- ALiBi（部分）
- soft cap（部分，Gemma 类）

不支持（截至 CANN 8.0）：
- 复杂自定义 mask（Swin window attention）→ fallback math path

## Paged KV Cache

MindIE-LLM 内置 PagedAttention 等价实现：
- block_size 默认 128（NV vLLM 用 16，Ascend 块更大）
- 显存利用率从 ~70% 提到 > 95%
- 支持 KV INT8 量化

配置（service.json）：
```json
"ScheduleConfig": {
  "cacheBlockSize": 128,
  "enableKvScalesCalibration": true,
  "maxPreemptCount": 0
}
```

## Continuous Batching

默认开启。调度参数：

```json
{
  "maxBatchSize": 32,
  "maxPrefillBatchSize": 16,
  "maxPrefillTokens": 8192,
  "supportSelectBatch": true       // 智能选择 batch 组合，类似 vLLM 的 batch packing
}
```

## Prefix Cache

```json
{
  "enablePrefixCache": true,
  "prefixCacheGB": 8                // 给 prefix cache 留 8 GB HBM
}
```

适合 agent / chatbot（同 system prompt 重复）—— 命中后 TTFT 降 50-90%。

## Chunked Prefill

```json
{
  "enableChunkedPrefill": true,
  "chunkSize": 2048
}
```

长 prompt（≥ 4K）分块送入，平滑 TTFT/TPOT。

## Speculative Decoding

CANN 8.0+ MindIE-LLM 支持 draft-target spec：
```json
{
  "speculativeConfig": {
    "enabled": true,
    "draftModelPath": "/models/Qwen3-0.6B-om/",
    "numSpeculativeTokens": 5
  }
}
```

加速 1.5-2×，与 NV 类似。

## TP / PP 多卡

### TP

```json
{
  "worldSize": 8,
  "npuDeviceIds": [[0,1,2,3,4,5,6,7]]
}
```

HCCS 拓扑必须先 OK（见 `ascend-infer-multinpu`）。LLaMA-3-70B 在 8 张 910B 上 TP=8 + W8A8 是典型配置。

### PP

跨节点用 PP：
```json
{
  "pipelineParallelSize": 2,
  "tensorParallelSize": 8
}
```

## 长上下文（YaRN / RoPE scaling）

转换时启用：
```bash
mindie-cli convert ... \
    --rope_scaling_type yarn \
    --rope_scaling_factor 4.0 \
    --max_seq_len 131072
```

配合 chunked prefill + KV INT8，128K context 在 8×910B 上可行。

## 显存预算（实战，Qwen3-8B）

| 配置 | 单卡显存占用 | 备注 |
|---|---|---|
| FP16 / BF16 + FP16 KV | ~16.5 GB (weights) + KV (varies) | 单卡推理 OK |
| W8A8 + INT8 KV | ~8.5 GB (weights) + KV/2 | 节省一半，batch 翻倍 |
| W4A16 + INT8 KV | ~4.5 GB (weights) + KV/2 | 单卡 64GB 跑超大 batch / 长 ctx |

实测吞吐参考（910B3，W8A8，KV INT8，max_seq=4K, batch=16）：
- 单流 decode：100-150 tok/s
- 总吞吐：2000-3500 tok/s

## 不同模型族的注意点

### Qwen2 / Qwen3 系列
- CANN 8.0+ 全支持，MindIE-LLM 含官方配置
- Qwen3 **thinking mode** 在 MindIE-LLM 也支持，注意 max_iter_times 拉到 4096+

### LLaMA-3 系列
- 8B / 70B / 405B 全支持
- GQA 自动走优化 path
- 70B 推荐 TP=4 W8A8；405B TP=8 W4A16

### MiniCPM-V / Qwen2-VL
- 视觉前端 + LLM 联合，MindIE-LLM 支持多模态 (CANN 8.0+)
- 推理参数加 `imagePreprocessConfig`

### DeepSeek-V2 / V3 (MoE)
- DeepSeek-V3 上 910B 通过 MindIE-LLM 已有适配（社区贡献）
- EP (Expert Parallel) 通过 `moeExpertParallel: true` 配置
- 显存占用大（236B params 即使 INT4 也 > 100GB）→ 必须多卡

## 调优速记

1. **必开**：paged KV、continuous batching、KV INT8、prefix cache（agent 场景）
2. **prefill / decode 桶化**：`maxPrefillTokens` 与 `maxBatchSize` 按线上分布调
3. **量化**：W8A8 几乎默认；70B+ 上 W4A16
4. **HCCL 环境变量**：HCCL_BUFFSIZE=512 / HCCL_OVER_OFI=0（单节点）
5. **AI Core 数感知**：910B3 有 25 个 AI Core，batch 不能太小（< 16 时 Core 闲）

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| eager + transformers generate | 慢 5-10× | 切 MindIE-LLM |
| KV INT8 没收 scale | decode 早期数值乱 | enableKvScalesCalibration + warmup 100+ step |
| 自家模型架构没在 atb_llm 适配 | 转换报 op 缺 | 升 CANN / 提交适配 PR / 写 Ascend C |
| chunked prefill 没开长 prompt | 单大 prompt 拖死 batch | enableChunkedPrefill: true |
| TP=8 但 HCCS 没建好 | 性能塌 | `npu-smi info -t topo` 检查 |
| Qwen3 thinking 输出 token 过短 | CoT 截断 | maxIterTimes ≥ 4096 |
| 转换后精度跳水 | 量化 scale 不准 | calibration 数据贴近线上 |
| YaRN factor 设错 | 短 prompt 精度掉 | factor 按 (target_len / native_len) |
| MindIE 升级前不读 release notes | 配置项变名 | 看版本兼容矩阵 |

## 与 NV 对应

| NV | Ascend |
|---|---|
| vLLM | MindIE-LLM Service |
| TensorRT-LLM | MindIE-LLM 转换 (mindie-cli) |
| FlashAttention v2/v3 | aclnnFlashAttentionScore |
| PagedAttention | MindIE paged KV (cacheBlockSize=128) |
| Continuous batching | MindIE 默认 |
| Prefix caching | enablePrefixCache |
| Chunked prefill | enableChunkedPrefill |
| Speculative (vLLM `--speculative-model`) | MindIE speculativeConfig |
| TP/PP/EP | worldSize / pipelineParallelSize / moeExpertParallel |
| KV cache FP8 | KV INT8（**无 FP8**） |
| W4A16 AWQ + Marlin | W4A16 (msmodelslim + MindIE) |

## 参考

- MindIE-LLM 用户指南（CANN 文档）
- atb-models GitHub
- `references/models/qwen3-8b.md` —— Qwen3 在 910B 性能参考
- `references/models/minicpm-v.md` —— VLM
- `references/hardware/huawei-atlas-910b.md`
