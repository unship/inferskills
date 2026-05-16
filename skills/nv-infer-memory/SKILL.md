---
name: nv-infer-memory
description: NVIDIA GPU 推理的内存层优化——GPU 内存层级(HBM/L2/L1/Smem/Reg)、global memory coalescing、pinned (page-locked) memory、async H2D/D2H、内存对齐、channels_last / NHWC 布局、PyTorch caching allocator 调优、KV cache 显存预算与碎片化、unified memory。当用户问"HBM 带宽"、"memory bound"、"pinned memory"、"channels_last"、"OOM 但 free"、"碎片化"、"allocator"、"KV cache 太大"、"内存对齐"、"unified memory" 时使用。
---

# Memory Optimization for Inference

> 现代 GPU 上，"算得快"的前提是"喂得够"。HBM 与 host 都喂不上，SM 就在 stall。

## GPU 内存层级速记

| 层级 | 大小 (H100) | 带宽 | 延迟 (cycles) | 典型用途 |
|---|---|---|---|---|
| Register | 256 KB/SM | — | 1 | 线程局部 |
| Shared / L1 | 228 KB/SM | ~19 TB/s | ~30 | block 内复用 |
| L2 | 50 MB | ~5 TB/s | ~250 | 跨 SM 复用 |
| HBM (Global) | 80/96 GB | 3.35 TB/s | ~500 | 模型权重 / 激活 / KV |
| PCIe Gen5 x16 | — | 64 GB/s | μs | host ↔ device |
| NVLink 4 | — | 900 GB/s (单卡聚合) | ns | GPU ↔ GPU |

设计原则：**计算 → register / smem，复用 → smem / L2，无可奈何 → global / host**。

## 1. Memory Coalescing（写 kernel 才管，调框架可跳）

一个 warp 32 线程要访问连续 128B（或 32 个 4B）才能合并为一次 transaction。
读 `nv-infer-kernels` 的 occupancy / divergence 一节。

判定：`ncu --section MemoryWorkloadAnalysis` 看 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request`，理想是 4（128B/32B sector）。

## 2. Pinned (page-locked) Host Memory + async transfer

**没 pin 的 host 内存不能 DMA**，要先到 staging buffer，慢且阻塞。

### PyTorch

```python
# DataLoader 自动 pin
loader = DataLoader(..., pin_memory=True, num_workers=8,
                    persistent_workers=True, prefetch_factor=4)

# 自己分配
x = torch.empty(shape, pin_memory=True)
x_gpu = x.to('cuda', non_blocking=True)   # 必须 non_blocking 才能 overlap
```

**陷阱**：`non_blocking=True` 只在源是 pinned memory 时生效。`torch.tensor(np.array(...)).cuda()` 不是 pinned。

### 验证 overlap

跑 nsys 看时间线，应该看到：
```
GPU stream0:  [kernel_A........][kernel_B........]
GPU memcpy :     [H2D buf1]   [H2D buf2]    [H2D buf3]
```
完全错位才算 overlap 成功。

## 3. CUDA Streams + Double Buffering

让计算与拷贝并行：

```python
stream_compute = torch.cuda.Stream()
stream_copy    = torch.cuda.Stream()

bufs = [torch.empty(shape, device='cuda') for _ in range(2)]
for i, batch in enumerate(host_batches):
    cur = i & 1
    with torch.cuda.stream(stream_copy):
        bufs[cur].copy_(batch, non_blocking=True)
    stream_compute.wait_stream(stream_copy)
    with torch.cuda.stream(stream_compute):
        out = model(bufs[cur])
```

**多 stream 的并发上限**：H100 ≈ 32 个 hardware queues，通过 `CUDA_DEVICE_MAX_CONNECTIONS=32` 解锁（默认 8）。

## 4. PyTorch Caching Allocator 调优

PyTorch 不立刻还显存给 driver，而是自己管理一块 cache。问题来源：碎片化与 retry。

```bash
# 启用 expandable segments（PyTorch 2.1+），大幅减少 OOM-but-free
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 大请求直接走 cudaMallocAsync（CUDA 11.4+）
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True

# Debug 碎片
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
```

### 检查碎片

```python
import torch
print(torch.cuda.memory_summary())
print(torch.cuda.memory_stats()['allocation.all.allocated'])
print(torch.cuda.memory_stats()['reserved_bytes.all.current'])
```

`reserved - allocated` 很大且持续增长 → 碎片严重。

### 显式释放

长跑的 LLM 服务里，**KV cache 一致大**问题不大；如果做 prefill/decode disagg、batch 切换大，每周期 `torch.cuda.empty_cache()` 一次（昂贵，别每步都做）。

## 5. 张量布局：channels_last for CNN/ViT

NCHW → NHWC（PyTorch 叫 `channels_last`）。Tensor Core 在 NHWC 路径效率更高（H100 cuDNN9 上 conv2d 提速 **1.3-2×**）。

```python
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)
```

要全链路 channels_last，包括：
- conv 输入/输出
- BN
- pooling
- residual add

任何一个回退 NCHW，整张图触发 layout transpose，反而变慢。
用 `torch.profiler` 看是否有 `to(memory_format=...)` op 出现。

ViT 里 patch embed 是 conv2d（kernel = patch_size，stride = patch_size），同样吃 channels_last 加速。

## 6. Tensor 对齐

- FP16/BF16 要 8-byte 对齐才能用 LDG.128 加载
- INT8 要 16-byte 对齐 / dim 是 16 的倍数才能走 INT8 Tensor Core
- TensorRT 自动 padding，PyTorch 手写 kernel 要小心

实务：把 `hidden_dim`、`num_heads`、`vocab_size` 都 **round up 到 8 / 16 / 64 的倍数**。
例：`vocab_size=50257` pad 到 50304（÷64=786）能让 logits matmul 在 H100 FP8 上多吃 5-10%。

## 7. LLM KV Cache 显存预算

```
KV_bytes = 2 (K+V) × n_layer × n_kv_head × head_dim × dtype × seq × batch
```
LLaMA-3-70B（n_layer=80，n_kv_head=8，head_dim=128，FP16）：
- 单 token KV = 2 × 80 × 8 × 128 × 2 = **327 KB**
- 4K context × 32 batch = ~40 GB（已经接近 1 张 80GB H100 显存上限）

降 KV 显存的工具：
1. **PagedAttention**（vLLM） — 物理分页减碎片，让 batch 拉满
2. **FP8 / INT8 KV cache** — TensorRT-LLM `--kv_cache_type=fp8`，约省 50%
3. **GQA / MQA** — 已在模型设计阶段决定，n_kv_head 直接小
4. **Sliding window / chunked prefill** — 长上下文截断
5. **KV cache offload to CPU** — Hopper 上 NVLink-C2C，Grace 体系下几乎免费

详细见 `nv-infer-llm-attention`。

## 8. Unified Memory / GH200 / GB200 superchip

Grace Hopper / Grace Blackwell 上 CPU 和 GPU 共享物理地址空间：

```python
# 直接分配在 host，GPU 走 NVLink-C2C 访问（900 GB/s）
x = torch.empty(shape, device='cuda', pin_memory=False)  # GH200 unified path
```

KV cache 溢出到 Grace CPU 的 LPDDR5X 上，对长上下文 LLM 是关键优化。
**Note**：GH200 上 `cudaMallocManaged` 几乎无 page fault 开销，不像 PCIe 系统。

## 9. 减少中间 allocation

eager mode 推理里，每个 op 都新分配输出。优化：

- `torch.compile`（自动减少）
- `out=` 参数：`torch.matmul(a, b, out=preallocated)`
- 用 `torch.no_grad()` + `torch.inference_mode()` 关 autograd 增加 view 复用
- 自己 preallocate 大 buffer，op 间复用

## 10. Memcpy 反模式速查

| 反模式 | 现象 | 修复 |
|---|---|---|
| `tensor.cpu()` 再 `.cuda()` | sync + 慢 | 留在 device |
| `.item()` 在 hot loop | strong sync | 改 boolean cuda tensor |
| `numpy(); torch.from_numpy().cuda()` | 没 pin | `pin_memory=True` |
| DataLoader 默认 `pin_memory=False` | H2D 阻塞 | 开 pin_memory |
| 多次小 H2D | overhead 高 | 一次大 batch 拷 |
| 模型权重一直 reload | 占 PCIe | 启动一次 load 到 GPU，常驻 |

## 验证清单

```python
# 1. 看是否 channels_last
print(model[0].weight.is_contiguous(memory_format=torch.channels_last))

# 2. 看 allocator 状态
print(torch.cuda.memory_summary(abbreviated=True))

# 3. 看 PCIe 是否在搬数据（dcgmi 1010/1011 应该接近 0，运行稳态时）
```

## 反模式总结

- `pin_memory=True` 但忘了 `non_blocking=True` → 还是 sync 拷贝
- 在循环里反复 `to(memory_format=channels_last)` → 反复 transpose
- 把 KV cache 放 CPU 但走 PCIe（非 GH200）→ 比直接拒绝请求还慢
- 用 `cudaMallocManaged` 在普通 H100 上当主存 → page migration 风暴
