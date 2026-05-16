---
name: nv-infer-io-data
description: NVIDIA GPU 推理的数据 / IO pipeline 优化——DataLoader 与 worker / pinned memory / prefetch、NVIDIA DALI、nvJPEG / nvImageCodec、GPU 端预处理 (resize/normalize)、tokenizer 加速、二进制/memmap 数据格式、对象存储 (S3/GCS) 读取并发、GPUDirect Storage、checkpoint / model load 加速、镜像预热。当用户问"GPU 利用率 50%"、"CPU 喂不动"、"DataLoader"、"DALI"、"JPEG decode 慢"、"模型加载慢"、"S3 读图慢"、"tokenizer 瓶颈" 时使用。
---

# I/O & Data Pipeline for Inference

> 推理时 GPU 闲着 = 数据没到。这一层不调好，所有其他优化都浪费。

## 触发场景速判

| 现象 | 大概率原因 |
|---|---|
| GPU util < 70%，CPU 100% | 预处理 / decode 在 CPU 太慢 |
| GPU util 锯齿状 | DataLoader prefetch 不够 / batch 间断 |
| 第一请求 200ms+ | tokenizer / model load 冷启动 |
| 服务启动几分钟 | 大模型 weight 从磁盘/S3 加载 |
| LLM TTFT 高但 model 在 GPU | tokenizer 是 Python 慢实现 |
| dcgmi PCIe RX 持续高 | H2D 是瓶颈 |

## CNN / ViT 数据 pipeline

### 1. PyTorch DataLoader 黄金配置

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,                  # = CPU 核数的 50-100%
    pin_memory=True,                # H2D 异步必需
    persistent_workers=True,        # 不每个 epoch 重启 worker
    prefetch_factor=4,              # 每个 worker 预取 4 个 batch
    drop_last=False,
    collate_fn=fast_collate,
)
```

```python
# 推理 loop：必须 non_blocking
for x, _ in loader:
    x = x.to('cuda', non_blocking=True).to(memory_format=torch.channels_last)
    with torch.inference_mode(): out = model(x)
```

### 2. DALI（强烈推荐 CNN/ViT 服务化）

NVIDIA DALI 在 GPU 上做 JPEG decode + resize + normalize，CPU 几乎 0 占用。

```python
import nvidia.dali as dali
from nvidia.dali import pipeline_def, fn

@pipeline_def(batch_size=64, num_threads=4, device_id=0)
def pipe():
    files, _ = fn.readers.file(file_root="/data/imgs")
    images = fn.decoders.image(files, device="mixed", output_type=dali.types.RGB)
    images = fn.resize(images, resize_x=224, resize_y=224, device="gpu")
    images = fn.crop_mirror_normalize(
        images, dtype=dali.types.FLOAT16,
        output_layout="HWC",         # channels_last
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255])
    return images

p = pipe(); p.build()
for _ in range(N): (gpu_batch,) = p.run()
```

Triton 直接支持 DALI backend（`backend: "dali"`），把整个 preprocessing 移进 server。

### 3. nvJPEG / nvImageCodec 独立用

只要 JPEG decode 加速，不要整个 DALI：

```python
import nvidia.nvimagecodec as ico
decoder = ico.Decoder()
img = decoder.read("/path/to.jpg", cuda_stream=stream)  # 直接到 GPU
```

适合 server 自己控制 batching 的场景。

### 4. 模型权重加载加速

70B FP16 = 140 GB，从普通磁盘读 5-10 分钟。

加速手段：
- **SafeTensors** 替代 pickle：mmap + zero-copy，加载 ×2-5
- **多线程 shard load**：HF `safetensors` 默认并行
- **GPUDirect Storage (GDS)**：NVMe → GPU 显存直传，跳过 CPU bounce buffer
  ```python
  import kvikio
  with kvikio.CuFile("weights.bin", "r") as f:
      f.read(gpu_tensor.data_ptr(), nbytes)
  ```
- **tensorizer**（CoreWeave）：流式加载，第一层就绪即可开始 prefill
- **Run:ai Model Streamer**：S3 → GPU 流式

vLLM 0.6+ 集成 tensorizer：
```bash
vllm serve ... --load-format tensorizer --model-loader-extra-config '{"tensorizer_uri": "s3://..."}'
```

### 5. S3 / GCS 读图

并发是关键：
- `boto3` 单线程 ≈ 50 MB/s；用 `s3transfer` 多线程或 `aioboto3` + `asyncio` ≈ 1+ GB/s
- 用 **mountpoint-s3** 或 **goofys** 让 S3 看起来像本地文件
- 边缘 cache：服务节点 NVMe 挂 cache，热数据本地

```python
# aioboto3 并发读
import asyncio, aioboto3
async def fetch(key):
    async with session.client('s3') as s3:
        r = await s3.get_object(Bucket=B, Key=key)
        return await r['Body'].read()
results = await asyncio.gather(*[fetch(k) for k in keys])
```

## LLM 数据 pipeline

### 1. Tokenizer 加速

```python
# 用 fast 版本（Rust 实现），慢 10×
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path, use_fast=True)

# 批量 tokenize
batch = tok(["t1", "t2", ...], padding=True, return_tensors="pt")
```

Python 端 tokenize 仍可能占 5-10% 端到端 latency。生产建议：
- tokenizer 放 vLLM/Triton server 内（已默认）
- 高 QPS 场景：tokenizer 多 worker（vLLM `--tokenizer-pool-size 8`）

### 2. Prompt 长度桶化

变长 prompt 会让 prefill kernel 多次重选，p99 抖。
策略：按 seq_len padding 到 128 / 256 / 512 / 1K / 2K / 4K / 8K 几个桶。
TRT-LLM `--remove_input_padding enable` + paged FMHA 让变长 batch 不浪费 token slot，是更现代的解。

### 3. 长 context 输入

128K context 客户端要传 1-5 MB JSON：
- 用 gRPC + proto 或 binary tokens，**不要 JSON**
- 把已 tokenize 的 `prompt_token_ids` 直接传，server 跳过 tokenize
- 巨长 prompt 考虑 server 端缓存 token IDs

### 4. Embedding / RAG

embedding 模型通常是 BERT 类，吞吐瓶颈 = decode 端 + tokenize：
- batch 拉大（128-512）
- channels_last 不适用（NLP 是 [B,S,H]）
- 多 worker DataLoader 并发 tokenize
- 用 Triton ensemble：tokenize → embedding → postproc

## H2D / D2H 高效写法

```python
# ✅ pinned + non_blocking
batch = torch.empty(shape, pin_memory=True)
batch.copy_(some_data)
batch_gpu = batch.to('cuda', non_blocking=True)

# ❌ 不 pinned，sync 拷贝
batch_gpu = torch.tensor(some_data).cuda()

# ✅ 大 batch 一次拷
batch = torch.stack(items, out=preallocated)
batch_gpu = batch.to('cuda', non_blocking=True)

# ❌ 循环里 100 次小 H2D
for x in items: ...to('cuda')...
```

## Checkpoint / 周期性 IO

推理服务一般无 ckpt 写入。但有：
- Prometheus metrics 落盘 → 异步、独立线程
- Request log → 采样写、不要 sync flush
- Trace（nsys/PyTorch profiler）→ 离线开关，生产关掉

## 镜像 / 启动预热

容器镜像 30-50 GB 拉取 5-10 分钟：
- 用 **lazy pulling**（stargz / nydus）：按需拉层
- Node-local image cache（K8s `imagePullPolicy=IfNotPresent`）
- 模型权重单独放 PV/PVC，不打进镜像
- **预热 daemon**：节点空闲时 prefetch 常用模型到本地 NVMe

## 验证

```bash
# 1. CPU 是否瓶颈
top -H -p $(pgrep -f serve.py)    # 看 worker 线程占用
# 如果某线程持续 100% CPU → 那是瓶颈

# 2. PCIe 是否瓶颈
dcgmi dmon -e 1010,1011 -c 30      # PCIe RX/TX 字节速率
# 持续 > 80% 链路带宽 → H2D 瓶颈

# 3. GPU 是否真在等
nsys profile -t cuda,osrt python serve.py
# 看 GPU 时间线是否有大段 idle 配 CPU 忙
```

## 反模式

- `pin_memory=False` 配 `non_blocking=True` → non_blocking 没生效
- `num_workers=0` → 主线程做预处理，GPU 等
- `num_workers` 等于 vCPU 数全占 → 系统响应卡顿，影响 server 主循环
- `persistent_workers=False` + 短任务 → 反复 fork worker 浪费时间
- DALI 把所有 op 设 `device="cpu"` → 比 PyTorch 还慢
- LLM tokenizer 用 slow 版本（Python） → 端到端慢 30%
- 大文件 sync 写日志 → tail latency 周期性抖
- 模型权重从远端对象存储边读边推 → 第一次极慢，没 cache
- 容器 `--shm-size=64m`（默认） → DataLoader worker IPC 阻塞
- 高 QPS 服务的 tokenizer 是单线程 → CPU 单核 100% 限速
