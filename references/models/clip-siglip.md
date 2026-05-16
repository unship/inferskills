# CLIP / SigLIP / SigLIP2 Inference

> 图文对比学习 encoder 家族。推理时通常**只用 encoder**（vision 或 text 端），输出 embedding 给下游用：检索、零样本分类、VLM 视觉前端、RAG。

## 模型族对比

| 模型 | 厂 | 年 | Loss | 输入分辨率 | Patch | 主要用途 |
|---|---|---|---|---|---|---|
| **CLIP** (ViT-B/16, L/14, H/14) | OpenAI | 2021 | InfoNCE (softmax) | 224 / 336 | 16 / 14 | 通用图文 |
| **OpenCLIP** | LAION | 2022 | 同上，更大数据 | 224 / 336 | 14 / 16 | 开源等效 |
| **EVA-CLIP** | BAAI | 2023 | InfoNCE + MIM 初始化 | 224 / 336 / 448 | 14 | 高精度通用 |
| **SigLIP** | Google | 2023 | **Sigmoid** (pairwise) | 224 / 256 / 384 | 14 / 16 | 小 batch 也能训，VLM 前端首选 |
| **SigLIP-SO400M** | Google | 2024 | Sigmoid | 384 | 14 | "Shape Optimized" 400M ViT |
| **SigLIP2** | Google | 2025 | Sigmoid + locca + 自蒸馏 | 256 / 384 / 512 | 14 / 16 | SigLIP 改进版，多模态预训练 |

## 尺寸 / 参数 / 计算量（vision encoder 主导）

CLIP / OpenCLIP 主要档：

| 模型 | Vision params | Text params | 输入 | Image GFLOPs | embed dim |
|---|---|---|---|---|---|
| ViT-B/32 | 88 M | 63 M | 224 | 8.8 | 512 |
| ViT-B/16 | 86 M | 63 M | 224 | 35.1 | 512 |
| ViT-L/14 | 304 M | 124 M | 224 | 162 | 768 |
| ViT-L/14@336 | 304 M | 124 M | **336** | 365 | 768 |
| ViT-H/14 | 632 M | 354 M | 224 | 381 | 1024 |
| ViT-bigG/14 | 1.84 B | 695 M | 224 | 1018 | 1280 |
| EVA-02-E/14+ | 4.7 B | — | 224 | 2362 | 1024 |
| SigLIP-SO400M/14 | 400 M | 877 M | 384 | ~280 | 1152 |

VLM 中常见组合（"X 模型用什么视觉前端"）：
- LLaVA-1.5：CLIP ViT-L/14@336
- LLaVA-OneVision：SigLIP-SO400M/14@384
- Qwen2-VL / Qwen2.5-VL：自家 ViT (动态分辨率)
- MiniCPM-V 2.6：SigLIP-SO400M/14
- InternVL：InternViT-300M / 6B
- DeepSeek-VL：SigLIP

> CLIP 类 encoder 推理只算 vision side 的 forward；text 一次性 encode 后缓存（常用于离线 embedding）。

## 显存预算

```
forward 激活 ≈ batch × seq × hidden × 4 × layers × bytes(dtype)
seq = (H/patch)² + 1 (CLS token)
```

ViT-L/14@336 FP16，batch=64：
- seq = (336/14)² + 1 = 577
- 中间激活 ≈ 64 × 577 × 1024 × 4 × 24 × 2 ≈ **30 GB**（不含 attn 中间矩阵）
- 实际带 FlashAttention 后 attn 临时 buffer 不需要 O(N²) 显存
- TRT/torch.compile 后激活进一步合并，单卡 24GB+ batch=128 可行

## 推理调优要点

1. **channels_last (NV)**：patch embed = stride 卷积，吃 NHWC 加速；ViT block 内部 token 维度 + hidden 是 1D 流，layout 无关
2. **`scaled_dot_product_attention` 自动 FlashAttention**：必走 FA-2/FA-3 后端
   ```python
   from torch.nn.attention import SDPBackend, sdpa_kernel
   with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
       feats = vision(x)
   ```
3. **batch 拉大**：vision encoder 是 compute-bound，吞吐随 batch 线性涨直到 SM 饱和；GPU 上 batch=128/256 常见
4. **量化**：
   - FP16/BF16 几乎无损（embedding 余弦相似度 > 0.999）
   - INT8 PTQ：top-1 retrieval recall 掉 < 0.5%，calibration 数据集要覆盖目标 domain
   - FP8 (Hopper+)：与 FP16 极接近，TRT-LLM/ModelOpt 支持
5. **固定分辨率**：不要用 dynamic shape，否则 cudnn 反复 benchmark；推荐 224 / 336 / 384 / 448 几个固定值各 build 一个 engine
6. **CLS token only 优化**：如果只取 `pooler_output`（CLS），可裁剪部分末层算非 CLS token 的输出 —— TRT 不会自动做，需要导 ONNX 时切图
7. **text encoder 缓存**：离线场景，把常用 label/prompt 的 text embedding 算一次存好，**推理时只跑 vision side**

## NVIDIA 推理样板（vision encoder）

```python
# PyTorch 2.4+ 一键
import torch
from open_clip import create_model_from_pretrained

model, _ = create_model_from_pretrained("ViT-L-14", pretrained="openai")
model = model.visual.eval().half().to('cuda').to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

x = torch.randn(64, 3, 224, 224, device='cuda').half().to(memory_format=torch.channels_last)
with torch.inference_mode():
    feats = model(x)   # (64, 768)
```

```bash
# TRT INT8 path (生产)
trtexec --onnx=clip_vit_l14_visual.onnx --int8 --fp16 \
    --calib=clip_calib.cache --useCudaGraph \
    --shapes=images:64x3x224x224 \
    --saveEngine=clip_vit_l14_int8.plan
```

## Ascend 推理样板

```bash
atc --model=clip_vit_l14_visual.onnx \
    --framework=5 \
    --soc_version=Ascend310P3 \
    --output=clip_vit_l14_om \
    --input_format=NCHW \
    --input_shape="images:64,3,224,224" \
    --precision_mode=allow_mix_precision
```

> Ascend 上 attention 走 `aclnnFlashAttentionScore` (CANN 7.0+)，不需要自己写。SigLIP 的 sigmoid loss 仅训练时使用，推理 forward 与 CLIP 完全等价。

## CLIP 与 SigLIP 推理时的区别

实际推理：vision encoder forward 计算图**几乎一样**（同是 ViT），区别在：
- pooling: CLIP 用 CLS token；SigLIP 默认用 mean pool 或 attention pool
- normalization: CLIP 输出 L2 normalize 后做余弦相似度；SigLIP 直接 dot product + sigmoid（推理时如果做 zero-shot 分类，公式不一样）
- 训练 loss 影响精度/泛化，但**推理图没区别**

## 常见集成场景

### 1. 图文检索 / 零样本分类
```python
# 离线一次：text embeddings
text_feats = text_encoder(tokenize(labels))      # (N, D)
text_feats = text_feats / text_feats.norm(-1, keepdim=True)

# 在线：vision embedding + 余弦
img_feats = vision(image_batch)
img_feats = img_feats / img_feats.norm(-1, keepdim=True)
logits = img_feats @ text_feats.T                # (B, N)
pred = logits.argmax(-1)
```

### 2. VLM 视觉前端
```
image → SigLIP vision → MLP projector → 拼到 LLM 输入 token 序列
```
MiniCPM-V / LLaVA / InternVL / Qwen-VL 全是此结构。推理时 vision 和 LLM 分两阶段：
- **Phase 1**: SigLIP forward 一次拿 image tokens
- **Phase 2**: LLM prefill (image tokens + text prompt) → decode

VLM 整体调优见 `minicpm-v.md` 与 `nv-infer-llm-attention`。

### 3. RAG / 多模态检索
- 把图库一次 encode 存 vector DB (FAISS / Milvus)
- 查询时 text → text encoder → top-K 检索

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 输入未做 OpenAI 官方 normalize | 检索结果差 | 用 `_transform = preprocess` 而不是自己 resize |
| 224 input 用 ViT-L/14@336 模型 | 输出形状不对 | 必须按训练分辨率 |
| `bfloat16` 在 T4 fallback FP32 | 慢 5× | 用 FP16 |
| SigLIP 沿用 CLIP 的 normalize 公式 | logits 量级不对 | SigLIP 不要 L2 norm + sigmoid，逻辑不同 |
| OpenCLIP / OpenAI weights 输入 mean/std 不同 | 数值偏差 | check 官方 preprocess |
| ViT-H/14 在 16GB 卡上 batch>32 OOM | 显存 | 降 batch，或 INT8 |
| TRT INT8 calibration 用 ImageNet → 推理领域是医疗 | recall 跳水 | 领域内数据 calibration |
| 多个 prompt embedding 不缓存 | 重复算 text encoder | 离线 build 缓存 |

## 服务化建议

- **embedding 服务**：单卡部署 vision encoder，HTTP/gRPC 输入图片返回 embedding；Triton + TRT backend
- **VLM 视觉前端**：与 LLM 同进程（共享显存）vs 独立 embedding 服务（独立扩缩）；MiniCPM-V / Qwen-VL 都在同进程的 vLLM/MindIE 中跑
- **大并发 retrieval**：vision encoder 独立服务，batch 拉满；text embedding 预计算

## 参考

- OpenAI CLIP paper, OpenCLIP/EVA-CLIP repos
- SigLIP paper (Google 2023)
- 本仓库 [`../../skills/nv-infer-cnn-vit/SKILL.md`](../../skills/nv-infer-cnn-vit/SKILL.md)
- 本仓库 [`../../skills/nv-infer-precision/SKILL.md`](../../skills/nv-infer-precision/SKILL.md)
