# Model-Specific Inference Reference

> 每个常用模型族的：架构速记 / 显存与 KV 预算 / 量化选项 / 推荐硬件搭配 / 调优要点 / 已知坑。
> 用于：选卡时算够不够装、选量化时知道精度损失大致区间、调优时直接看本族要点。

## 覆盖模型

| 模型 | 文件 | 类型 | 典型场景 |
|---|---|---|---|
| **YOLO** v8 / v9 / v10 / v11 | [`yolo-series.md`](yolo-series.md) | CNN 检测 | 视频流分析、工业质检、安防、自动驾驶 |
| **CLIP / SigLIP / SigLIP2** | [`clip-siglip.md`](clip-siglip.md) | ViT + 文本 encoder | 图文检索、零样本分类、VLM 视觉前端 |
| **MiniCPM-V 2.6 / 4.0 / o-2.6** | [`minicpm-v.md`](minicpm-v.md) | 多模态 LLM (VLM) | 端侧/边缘多模态对话、OCR、文档理解 |
| **Qwen3-8B** (及 Qwen2.5-7B / Qwen3.5 同族) | [`qwen3-8b.md`](qwen3-8b.md) | Decoder-only LLM | 通用对话、code、agent、reasoning |

## 选模型 × 选硬件矩阵

按"模型 + 你手上的卡 + 推荐量化"快速选：

| 模型 | T4 | H100 (80GB) | B200 (192GB) | Atlas 300V Pro | Atlas 910B |
|---|---|---|---|---|---|
| YOLOv8n/s/m | ✅ TRT INT8 | ✅ 大材小用 | ✅ 大材小用 | ✅ ATC INT8 + DVPP | ✅ ATC INT8 |
| YOLOv8l/x, v9-C/E, v10-l/x, v11-l/x | ✅ TRT INT8/FP16 | ✅ | ✅ | ✅ INT8 | ✅ |
| CLIP ViT-B/16 | ✅ FP16/INT8 | ✅ batch 拉满 | ✅ | ✅ FP16 | ✅ |
| CLIP ViT-L/14 | ⚠️ batch 受限 (16GB) | ✅ | ✅ | ✅ | ✅ |
| CLIP ViT-H/14 | ⚠️ 大 batch OOM | ✅ | ✅ | ⚠️ LPDDR 慢 | ✅ |
| MiniCPM-V 2.6 (8B) | ⚠️ INT4 才装得下 | ✅ FP16/FP8 | ✅ | ⚠️ 限 INT4，吞吐受限 | ✅ FP16/INT8 |
| MiniCPM-V 4.0 / o-2.6 | ❌ 过大 | ✅ | ✅ | ❌ | ✅ INT8/W4A16 |
| Qwen3-8B FP16 | ❌ 16GB 装不下 (需 INT4) | ✅ 单卡 | ✅ 单卡 | ❌ | ✅ 单卡 |
| Qwen3-8B INT4 (AWQ/GPTQ) | ⚠️ Marlin sm_75 fallback 慢 | ✅ 极快 | ✅ | ⚠️ 通过 MindIE-LLM W4A16 | ✅ |
| Qwen3-32B FP16 (66GB+) | ❌ | TP=2 OK | 单卡 OK | ❌ | TP=2 |
| Qwen3-235B-A22B (MoE) | ❌ | TP=8+EP | TP=4-8 OK | ❌ | TP=8+EP |

> "OK"代表能跑且性能合理；"⚠️"代表能跑但有显著限制；"❌"代表显存/算力/生态不支持。

## 显存粗算（推理时 KV cache）

```
KV_bytes = 2 (K,V) × n_layer × n_kv_head × head_dim × dtype_bytes × seq × batch
```

热点模型单 token KV（dtype=FP16）：

| 模型 | n_layer | n_kv_head | head_dim | 单 token KV |
|---|---|---|---|---|
| LLaMA-3-8B (GQA) | 32 | 8 | 128 | 128 KB |
| Qwen3-8B (GQA) | 36 | 8 | 128 | 144 KB |
| LLaMA-3-70B (GQA) | 80 | 8 | 128 | 320 KB |
| Qwen3-32B | 64 | 8 | 128 | 256 KB |
| MiniCPM-V 2.6 (Qwen2-7B backbone) | 28 | 4 | 128 | 56 KB |

FP8 KV 减半，INT8 减半，FP4 KV (B200) 减 75%。

## 调优要点共通模式

不管你用哪个模型，到具体卡上几乎一定要做：

1. **量化到目标卡的最优精度档** —— 见每个模型文件的量化表
2. **shape 对齐 16/32/64** —— vocab / hidden_dim / batch / seq 都对齐
3. **batch 桶化** —— 减少 dynamic shape 触发的 recompile
4. **数据布局对齐**：CNN/ViT 用 channels_last (NV) 或 NZ (Ascend)
5. **KV cache 用 paged + 量化**（LLM/VLM）
6. **FlashAttention / FlashAttentionScore** —— ViT 与 LLM 都受益
7. **dynamic batching / continuous batching** —— 提服务吞吐

## 模型族 + skill 路径

| 模型族 | 主要走的 skill |
|---|---|
| YOLO 全系 | nv-infer-cnn-vit / nv-infer-precision / nv-infer-io-data / ascend-infer-cnn-vit |
| CLIP / SigLIP | nv-infer-cnn-vit (vision encoder) + nv-infer-compile-stack + ascend-infer-cnn-vit |
| MiniCPM-V | nv-infer-llm-attention + nv-infer-cnn-vit (vision encoder) + nv-infer-precision |
| Qwen3 | nv-infer-llm-attention + nv-infer-serving + nv-infer-precision + ascend-infer-llm-stack |

## 文件结构惯例

每个模型文件包含：
- 一句话画像 + 完整 SKU/尺寸表
- 架构关键点（影响推理调优的部分）
- 显存预算公式与典型数字
- 量化优先级与精度损失参考
- 各卡上的推荐配置（命令模板）
- 已知坑

## 后续可扩展

待补充模型：
- Llama-3.1/3.2/3.3 全系
- DeepSeek-V3 / V3.5 (MoE)
- Mixtral-8x7B / 8x22B
- Gemma-2 / Gemma-3
- Stable Diffusion 3.5 / FLUX.1
- Whisper Large v3
- Mamba / Hyena / Jamba（状态空间）
- DiT / CogVideoX（视频生成）

PR 欢迎，按本目录文件格式即可。
