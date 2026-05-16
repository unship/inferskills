# MiniCPM-V Series Inference

> OpenBMB 出品的端侧/边缘多模态 LLM (VLM) 系列。卖点：**小尺寸 (~8B) 跑出 GPT-4V 级多模态能力**，并且对**长 OCR / 高分辨率 / 视频**专门优化。

## 版本谱

| 版本 | 年 | 参数 | LLM backbone | Vision encoder | 主要新特性 |
|---|---|---|---|---|---|
| MiniCPM-V 1.0 | 2024 Q1 | 2.4 B | MiniCPM-2.4B | SigLIP-400M | 端侧 VLM 起步 |
| MiniCPM-V 2.0 | 2024 Q2 | 2.8 B | MiniCPM-2.4B | SigLIP-400M | OCR 强化、448 分辨率 |
| MiniCPM-Llama3-V 2.5 | 2024 Q2 | 8 B | Llama-3-8B | SigLIP-400M | 8B 档发力 |
| **MiniCPM-V 2.6** | 2024 Q3 | 8 B | Qwen2-7B | SigLIP-400M | **多图 + 视频 + 高分辨率** OCR |
| MiniCPM-o 2.6 | 2024 Q4 | 8 B | Qwen2-7B | SigLIP + Whisper-medium | **omnimodal**：视觉 + 音频 + 语音生成 |
| **MiniCPM-V 4.0 / 4.5** | 2025 | 8 B (V), MoE 变体 | Qwen2.5 backbone | SigLIP2 | 视频理解 + 实时流 + 工具调用 |

## 核心架构（以 V 2.6 / 4.0 为代表）

```
image → SigLIP-400M ViT (patch 14, 448×448 切片) → projector (MLP/resampler)
                                                       ↓
text prompt → tokenizer → → → → → → → → → → → → → LLM (Qwen2-7B 28 layers)
                                                       ↓
                                                output tokens
```

**关键设计：自适应视觉编码 (Adaptive Visual Encoding)**
- 输入图像不强制 resize 到固定分辨率
- 按"切片"策略划分多个 448×448 patch，每片独立过 SigLIP
- 一张 1344×896 图 → 6 个切片 + 1 个全局缩略 = **7×(patch 序列)** 给 LLM
- 高分辨率/长 OCR 因此显著强，但**推理时 image token 数可变**（≈ 64 - 1280 tokens 视图像复杂度）

## 显存预算

LLM 端（Qwen2-7B 32 层，GQA 4 KV head）：
- KV per token = 2 × 32 × 4 × 128 × 2 (FP16) = **64 KB**
- 8K context × batch=8 ≈ 4 GB KV

Vision 端单图：
- 7 切片 × (448/14)² ≈ 7 × 1024 = **7168 image tokens** 最坏（高复杂度）
- 加上 text，prompt 总 token 容易 8K+

实战：MiniCPM-V 2.6 单卡推理（FP16）需 ≥ **22 GB 显存**舒适，T4 16GB 必须 INT4 才能跑。

## 量化与精度

| 方案 | 模型 size | 精度损失 (OpenCompass MM) | 推荐场景 |
|---|---|---|---|
| FP16 | ~16 GB | baseline | 24GB+ 卡 |
| INT8 W8A8 | ~9 GB | ~0.5-1.0 | 16GB+ 卡 |
| INT4 W4A16 (GPTQ/AWQ) | ~5 GB | 1.0-2.5 | T4 16GB / 边缘 |
| GGUF Q4_K_M | ~5 GB | 1.0-2.5 | llama.cpp / 端侧 |
| FP8 (Hopper) | ~9 GB | <0.5 | H100/L40S/B200 |

OCR/Doc 类任务对量化**比一般 VLM 敏感**：从 FP16 到 INT4 通常掉 2-4% 字符识别率。calibration 数据要含 OCR 样本。

## 推理实战 — vLLM (推荐 NV 主流路径)

```bash
pip install vllm>=0.6.3 transformers>=4.45

vllm serve openbmb/MiniCPM-V-2_6 \
    --trust-remote-code \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype float16 \
    --limit-mm-per-prompt image=10,video=1
```

API 调用（OpenAI 兼容）：
```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
resp = client.chat.completions.create(
    model="openbmb/MiniCPM-V-2_6",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "图里有什么？"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)
```

## SGLang 路径（多模态 prefill 更快）

```bash
pip install sglang>=0.3
python -m sglang.launch_server --model openbmb/MiniCPM-V-2_6 \
    --port 30000 --tp 1 \
    --mem-fraction-static 0.85 \
    --enable-mixed-chunk \
    --kv-cache-dtype fp8_e5m2
```

SGLang 的 **RadixAttention** 对"共享同前缀 image tokens"的批量推理收益明显（同图多问题、agent 流）。

## llama.cpp 端侧（GGUF）

MiniCPM-V 2.6 / V 4.0 都有官方 GGUF；T4 / 笔记本 GPU / Apple Silicon 都能跑。

```bash
# 编译开启 CUDA
LLAMA_CUBLAS=1 make -j

./llama-minicpmv-cli -m minicpm-v-2_6-Q4_K_M.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image test.jpg \
    -p "描述这张图" \
    -ngl 99 --temp 0.7
```

`--mmproj` 是 vision encoder + projector 的 GGUF；主模型与之分开管理。

## TensorRT-LLM 路径

TRT-LLM 0.13+ 支持 MiniCPM-V 2.6 (Qwen2 backbone)。视觉部分需要单独编译：

```bash
# 1. Build vision encoder (TRT)
python build_vision_engine.py --model_dir minicpmv2_6/visual/ \
    --output_dir engines/visual/ --max_batch_size 8

# 2. Build LLM (TRT-LLM)
trtllm-build --checkpoint_dir minicpmv2_6/llm_ckpt/ \
    --output_dir engines/llm/ \
    --gemm_plugin float16 --gpt_attention_plugin float16 \
    --use_fp8_context_fmha enable --kv_cache_type=fp8 \
    --paged_kv_cache enable --remove_input_padding enable \
    --max_batch_size 32 --max_input_len 8192 --max_output_len 2048
```

## Ascend (910B + MindIE-LLM) 路径

MindIE-LLM 7.0+ 已收录 MiniCPM-V / Qwen2-VL 系列：

```bash
# 模型转换（HF → MindIE）
mindie-cli convert --model_path MiniCPM-V-2_6/ \
    --output_path minicpmv26_om/ \
    --soc_version Ascend910B3 \
    --max_batch_size 16 --max_seq_len 8192 \
    --enable_paged_kv true \
    --quant w8a8

mindieservice_daemon --config minicpmv26_service.json
```

300V Pro 上跑 2.6 (8B) 较吃力 —— LPDDR 带宽 + 24GB 限制下，必须 W4A16 量化并接受较低吞吐（典型 < 5 tok/s）。生产建议 V 2.0 (2.4B) 档放 300V Pro，V 2.6+ 上 910B。

## 调优要点（MiniCPM-V 专属）

1. **image token 数不固定** — 推理时按图像复杂度切片
   - 平均 600-1000 tokens/图（含切片+全局）
   - 多图聊天 → 几千 image tokens + 文本 → context 长度规划
2. **chunked prefill 必开** — vision 输出注入后 LLM prefill 阶段易长
3. **prefix cache 多图场景受限** — 不同图 patch 不同，前缀复用率低；同图多问对话才受益
4. **batch=1 单流即可** — 8B 模型 + 高分辨率图，batch 拉大显存压力急升
5. **FP8 KV** 比 INT8 KV 更适合 — 长文档 OCR 数值范围大，FP8 更稳
6. **video 推理 (V 2.6+)**：按帧采样（典型 1 帧/秒），多帧拼接到 prompt，注意总 image tokens 上限

## 视频流处理（V 2.6+）

```python
import cv2
def sample_frames(video_path, fps=1, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(orig_fps / fps)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    return frames

frames = sample_frames("clip.mp4", fps=1, max_frames=16)
# 16 帧 × ~600 tokens/帧 ≈ 10K image tokens → 注意 max_model_len 配够
```

注意：16 帧 1 fps 通常足够；时间分辨率不够时再加密。

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 用 transformers `generate()` 跑生产 | 慢且无 batching | 改 vLLM/SGLang/MindIE |
| `--max-model-len` 设小（如 4096）配高分辨率 | image tokens 超长被截，OCR 漏字 | 设 ≥ 8192，必要时 16K |
| 量化时 calibration 用纯文本 / 自然图 | OCR 任务掉点 | calibration 含 OCR/文档样本 |
| vision encoder 重复算 | 同图多 prompt 时未缓存 | 自行缓存 vision features（多数 server 已自动） |
| FP16 → BF16 mismatch | OCR/中文数字错乱 | backbone 是 Qwen2 → BF16 更稳 |
| llama.cpp 视频 prompt 长度爆 | OOM | 减 max_frames 或上 GPU 全卸载 |
| 多图聊天历史不裁剪 | context 越用越长 | 客户端做窗口管理 |
| Ascend 转换报 op 缺失 | resampler 自定义 op | 升 CANN 8.0+ / 用 MindIE 官方模型仓 |
| T4 上 INT4 慢 | Marlin / GPTQ kernel sm_75 fallback 慢 | 接受 ≤ 10 tok/s 或换 L4/A10 |

## 性能参考（社区公开 + 实测，仅作量级参考）

| 卡 | 模型 | 量化 | 单流 decode tok/s |
|---|---|---|---|
| T4 | MiniCPM-V 2.6 | INT4 GGUF | 8-12 |
| A100 80G | V 2.6 | FP16 | 60-80 |
| H100 80G | V 2.6 | FP8 + paged KV | 90-130 |
| L40S | V 2.6 | INT8 | 50-70 |
| Atlas 910B3 | V 2.6 (MindIE-LLM) | W8A8 | 40-60 (估) |
| Atlas 300V Pro | V 2.0 (2.4B) | INT8 | 10-15 (估) |

吞吐取决于 image token 数 + max-num-seqs + max-num-batched-tokens 配置。

## 参考

- OpenBMB MiniCPM-V GitHub 与 HF 模型卡
- 本仓库 [`../../skills/nv-infer-llm-attention/SKILL.md`](../../skills/nv-infer-llm-attention/SKILL.md)
- 本仓库 [`../../skills/nv-infer-serving/SKILL.md`](../../skills/nv-infer-serving/SKILL.md)
- 本仓库 [`clip-siglip.md`](clip-siglip.md) — 视觉前端
- 本仓库 [`qwen3-8b.md`](qwen3-8b.md) — LLM backbone 调优共通
