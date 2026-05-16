---
name: ascend-infer-memory
description: Ascend NPU 推理内存层优化——HBM/LPDDR、Unified Buffer (UB)、L0/L1 buffer、NZ format 数据布局、torch_npu allocator (PYTORCH_NPU_ALLOC_CONF)、pinned host memory、aclrtMemcpy、KV cache 显存预算与碎片化。当用户问"NZ format"、"UB"、"NPU 显存"、"碎片化"、"OOM but free"、"allocator"、"pinned memory NPU"时使用。
---

# Ascend Memory Optimization

> 与 nv-infer-memory 平行。重点在 **NZ format**（Ascend 内部张量布局）、**Unified Buffer**（类比 shared memory）、**torch_npu allocator**。

## Ascend 内存层级（910B / 310P）

| 层级 | 容量 | 带宽 | 用途 |
|---|---|---|---|
| Cube/Vector Register | KB 级 | — | 单线程局部 |
| **L0A / L0B** (Cube 输入 buffer) | 64 KB | 极高 | Cube GEMM 输入分片 |
| **L0C** (Cube 累加 buffer) | 256 KB | 极高 | Cube GEMM 累加 |
| **L1 Buffer** | 1 MB | 高 | 跨 Cube 复用、片上缓存 |
| **Unified Buffer (UB)** | 192-256 KB / AI Core | 高 | Vector unit 工作区 |
| **HBM** (910B) / **LPDDR4X** (310P) | 64 GB / 24 GB | 1.6 TB/s / 204 GB/s | weights / activations / KV |
| Host DDR | — | PCIe Gen4 ~32 GB/s | host buffer |

设计原则：**Cube 算 → L0；Vector 算 → UB；跨 op 复用 → L1；不得已 → HBM**。

ATC/GE 自动安排（Cube/Vector dispatch + UB/L1 staging），自定义 Ascend C 算子才需手动管。

## NZ Format（关键）

Ascend 内部张量不是常见的 NCHW / NHWC，而是 **NZ format**（Native-Z），一种 5D 分块布局：

```
N C H W (原)
   ↓
N (C1) H W (C0)      其中 C = C1 * C0, C0 = 16 (FP16) / 32 (INT8)
   ↓ 进一步 tile
N C1 H1 W1 H0 W0 C0  （内部 6D）
```

- **C0 维度对齐 16/32**：让 Cube 16×16×16 GEMM 直接读
- **不齐的 channel/dim 会 padding**（占额外 5-15% HBM）

**实际意义**：
- 用 ATC/MindIE 自动转换 → 不用关心
- 手写 Ascend C kernel → 必须按 NZ 写，否则性能崩
- 跨框架/工具传 weight → 中间用 ND（regular layout），最后 entry 时转 NZ

### 影响排查

```bash
# msprof 中查看 op 类型，如果有大量 TransData (ND ↔ NZ 转换)，说明 layout 反复转
grep TransData prof_out/PROF*/device_*/data/op_summary*.csv
```

理想情况：模型 entry 一次 ND→NZ，exit 一次 NZ→ND，中间不再转。

## torch_npu Caching Allocator

类比 PyTorch CUDA caching allocator：

```bash
# 启用 expandable segments（必开，大幅减少 OOM-but-free）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 大请求走 aclrtMallocAsync
export PYTORCH_NPU_ALLOC_CONF=backend:aclrtMallocAsync,expandable_segments:True

# Debug 碎片
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512
```

查询：
```python
import torch_npu
print(torch.npu.memory_summary())
print(torch.npu.memory_stats()['allocation.all.allocated'])
print(torch.npu.memory_stats()['reserved_bytes.all.current'])
```

碎片严重（`reserved - allocated` 大且增长）→ 检查是否大小张量频繁交替分配。

## Pinned Host Memory + Async H2D

torch_npu 支持 pin_memory：

```python
loader = DataLoader(..., pin_memory=True, num_workers=8,
                    persistent_workers=True, prefetch_factor=4)

# DataLoader 内部自动选 npu pin
x = batch.to('npu', non_blocking=True)
```

或手动：

```python
x = torch.empty(shape, pin_memory=True)
x_npu = x.to('npu', non_blocking=True)
```

**非 pinned 内存 + non_blocking=True 实际仍同步**（同 NV 一样）。

## CUDA Stream → aclrtStream

PyTorch on NPU 有等价的 stream 概念：

```python
s_copy = torch.npu.Stream()
s_compute = torch.npu.Stream()

with torch.npu.stream(s_copy):
    x_npu.copy_(x_cpu, non_blocking=True)
torch.npu.current_stream().wait_stream(s_copy)
with torch.npu.stream(s_compute):
    out = model(x_npu)
```

底层是 `aclrtStream`。

## KV Cache 显存预算（LLM）

公式同 NV：
```
KV_bytes = 2 × n_layer × n_kv_head × head_dim × dtype × seq × batch
```

Ascend 上没有 FP8 KV，最多到 INT8 KV（约省 50%）：

| 量化 | KV 字节/token (Qwen3-8B) | 备注 |
|---|---|---|
| FP16 | 144 KB | 基线 |
| BF16 | 144 KB | 同 |
| **INT8** | 72 KB | MindIE-LLM 内置 |

详见 `ascend-infer-llm-stack`。

## 减少 OM 显存占用

ATC 编译选项影响显存：

```bash
atc ... \
    --buffer_optimize=l2_buffer_optimize \    # L2 cache 复用更激进
    --enable_small_channel=1 \                # 小 channel 优化
    --optypelist_for_implmode=...             # 部分 op 选低显存 impl
```

`--buffer_optimize` 在固定 batch 下提升显存复用，**通常 +10-20% 可用 batch**。

## Memcpy 反模式

| 反模式 | 现象 | 修复 |
|---|---|---|
| `tensor.cpu()` 再 `.npu()` | 慢 + sync | 留在 device |
| `.item()` 在 hot loop | strong sync | 改 npu tensor |
| `numpy().npu()` | 没 pin | pin_memory=True |
| DataLoader pin=False | H2D 阻塞 | 开 pin_memory |
| 多次小 H2D | overhead 高 | 一次大 batch 拷 |
| 模型权重反复 reload | 占 PCIe | 启动一次 load 到 NPU |
| eager 模式每个 op 新分配 | 中间 tensor 满天飞 | 用 TorchAir / MindIE 编译 |

## 验证

```python
# 看 allocator 状态
import torch, torch_npu
print(torch.npu.memory_summary(abbreviated=True))

# 显存峰值
print(torch.npu.max_memory_allocated() / 1e9, "GB")
```

```bash
# 看实际 NPU 显存（从 npu-smi）
npu-smi info -t common -i 0 -c 0 | grep -i memory
```

## 与 NVIDIA 对应

| NV | Ascend |
|---|---|
| HBM3 (80/192 GB) | HBM2e (64 GB on 910B) / LPDDR4X (24 GB on 310P) |
| L2 cache (50 MB) | L1 buffer (1 MB) — 容量小很多，复用窗口窄 |
| Shared Memory (228 KB/SM) | Unified Buffer (192-256 KB/AI Core) |
| Registers | Cube/Vector registers |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| pin_memory=True | pin_memory=True (torch_npu) |
| non_blocking=True | non_blocking=True |
| channels_last (NHWC) | NZ format (自动) |
| cudaMallocAsync | aclrtMallocAsync |
| cudaStream | aclrtStream / torch.npu.Stream |

## 反模式

- 期望 NPU 上 channels_last 加速 —— Ascend 自有 NZ，跟 channels_last 无关
- 在自定义 op 里假设 ND layout 输入 —— 会被自动 TransData 包夹，性能塌
- 跨 op 反复 `.cpu()` 看中间结果（debug） —— 不要在生产代码里留
- 没设 `PYTORCH_NPU_ALLOC_CONF=expandable_segments` —— 经常 OOM but free
- 单个超大 tensor 反复 alloc/free —— 用 preallocated buffer + `out=` 风格

## 参考

- CANN 文档"内存管理与优化"
- AscendCL API：`aclrtMalloc / aclrtFree / aclrtMallocHost / aclrtMemcpy`
- `references/hardware/huawei-atlas-910b.md`
- `nv-infer-memory` —— 对照 NV 侧概念
