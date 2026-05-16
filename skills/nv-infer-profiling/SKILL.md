---
name: nv-infer-profiling
description: 用 Nsight Systems/Compute、NVTX、framework profiler 找 NVIDIA GPU 推理的真实瓶颈。当用户说"为什么这么慢"、"GPU 利用率低"、"找瓶颈"、"profile 一下"、"想看 kernel 时间分布"、"延迟尾巴 (p99)"、"warp divergence"、"register spilling"、"SM occupancy" 时使用。覆盖 nsys/ncu/dcgmi/PyTorch profiler 的实战命令与读图方法。
---

# Profiling, Debugging & Monitoring — find the real bottleneck

## 触发场景
- "GPU 利用率只有 30%，为什么？"
- "推理慢，但我不知道慢在哪里"
- "p99 飙到 200ms，p50 才 30ms"
- 要写 perf regression CI

## 黄金顺序：从粗到细

1. **`nvidia-smi dmon` / `dcgmi`** — 1 分钟看 SM/HBM/功耗/温度宏观
2. **Nsight Systems (`nsys`)** — 时间线，看 host/CUDA/NCCL/网卡的 overlap
3. **Nsight Compute (`ncu`)** — 单个 kernel 的微观指标（occupancy、divergence、stalls）
4. **Framework profiler (PyTorch / TF)** — 算子归属、Python overhead

跳级是常见错误：**没看 nsys 就先 ncu 是浪费时间** —— ncu 只告诉你某 kernel 不够好，nsys 告诉你瓶颈是不是 kernel。

## Step 1: 宏观状态（30 秒）

```bash
# SM 使用率、HBM 利用率、功耗、温度
nvidia-smi dmon -s pucvmet -c 30

# DCGM 更精细（DRAM Active%、Tensor Active%、PCIe RX/TX、NVLink）
dcgmi dmon -e 1001,1002,1003,1004,1005,1009,1010,1011 -c 30
# 1001 = GR Active, 1002 = SM Active, 1003 = SM Occupancy
# 1004 = Tensor Active, 1005 = DRAM Active
# 1009 = FP64/FP32/FP16 Active, 1010 = PCIe RX, 1011 = PCIe TX
```

判定方向：
- GR Active 高、Tensor Active 低 → 没走 Tensor Core，去 `nv-infer-precision`
- DRAM Active > 70% → memory bound
- SM Occupancy < 25% → 看 `nv-infer-kernels`（register/smem pressure）
- PCIe RX 持续高 → H2D 拷贝慢，去 `nv-infer-io-data`

## Step 2: Nsight Systems — 时间线

### 基本采集

```bash
nsys profile \
  -t cuda,nvtx,cudnn,cublas,osrt \
  -o infer_baseline \
  --capture-range=cudaProfilerApi \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  python serve.py
```

让 profile 范围可控：

```python
import torch
# 跳过 warmup，只采集 5 步
for i, batch in enumerate(loader):
    if i == 5: torch.cuda.cudart().cudaProfilerStart()
    if i == 10: torch.cuda.cudart().cudaProfilerStop(); break
    run(batch)
```

### NVTX 标注（强烈建议）

PyTorch 自动插了部分 NVTX，自定义阶段加上：

```python
import torch.cuda.nvtx as nvtx
nvtx.range_push("prefill")
out = model.prefill(...)
nvtx.range_pop()
nvtx.range_push("decode_step")
...
```

LLM serving 至少标 `prefill / decode / kv_alloc / scheduler / send_response`。

### 读 nsys 的优先级

1. **GPU idle gaps** > 5% → host 在阻塞（Python、同步 `.item()`、`torch.cuda.synchronize`）
2. **Memcpy HtoD/DtoH 跟 kernel 没 overlap** → 未用 pinned mem / 没分 stream，去 `nv-infer-memory`
3. **NCCL kernels 串行 / 卡在 AllReduce** → 通信瓶颈，去 `nv-infer-multigpu`
4. **某个 kernel 占总时间 > 30%** → 该 kernel 去 ncu

```bash
# 直接生成报告统计
nsys stats --report cuda_gpu_kern_sum infer_baseline.nsys-rep | head -30
```

## Step 3: Nsight Compute — 单 kernel 深挖

```bash
# 只 profile 名字匹配的 kernel，否则会爆慢
ncu --target-processes all \
    --kernel-name regex:"flash_attn|gemm|conv" \
    --launch-skip 50 --launch-count 5 \
    --set full \
    -o kernel_detail \
    python serve.py
```

关键 metrics 速查（Roofline / Speed of Light）：

| metric | 阈值 | 含义 / 下一步 |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | > 80% 好 | 已接近 compute SOL |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | > 80% 好 | 已接近 memory SOL |
| `smsp__warps_active.avg.per_cycle_active` / 64 | > 0.5 | occupancy，低则 register/smem 太多 |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | > 95% | divergence 少 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second` | — | global load 量，配合 HBM BW |
| `launch__registers_per_thread` | <= 64 通常 | 高了限制 occupancy |
| `launch__shared_mem_per_block` | — | 看是不是 smem 太满 |

ncu 自带 sections：
```bash
ncu --section SpeedOfLight --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis --section Occupancy \
    --section WarpStateStats ...
```

### Roofline 判断

`Arithmetic Intensity = FLOPs / Bytes`，落在 roofline 哪一段：
- 左侧 (低 AI) → memory bound → 降精度 / kernel fusion / 提 cache reuse
- 右侧 (高 AI) → compute bound → 用 Tensor Core / FP8 / sparsity

## Step 4: Framework Profiler

### PyTorch

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_trace"),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(loader):
        run(batch)
        prof.step()
```

`record_shapes=True` 让你看到每个 op 的输入 shape（找 dynamic shape 抖动）。
`with_stack=True` 让你看 Python 调用栈（找 host overhead 来源）。

### TensorRT / vLLM

- TensorRT：用 `trtexec --profilingVerbosity=detailed --dumpProfile`
- vLLM：`--enable-prefix-caching` 后跑 metrics endpoint，看 `vllm:time_to_first_token`、`vllm:time_per_output_token`
- TensorRT-LLM：内置 `--gather_all_token_logits` / `gpt_log` 看 prefill vs decode 分布

## Python overhead 的诊断

如果 nsys 里 GPU 行有大量空隙，看 CPU 行那个时候在干啥：
- 大量 `aten::*` 但 GPU idle → 没 batch / 没 fuse / `.item()` 强同步
- 大量 `cudaStreamSynchronize` / `cudaMemcpyAsync` → 看 `nv-infer-memory` 的 pinned/stream 部分
- 大量 Python frame → 用 `torch.compile` 或 CUDA Graphs（见 `nv-infer-compile-stack`）

## Tail latency (p99) 专题

p50 好、p99 差的常见原因：
1. **Cold-start / 首次 kernel autotune** — warmup 至少 50 step
2. **cuDNN benchmark 反复重选** — `torch.backends.cudnn.benchmark=True` + 固定 shape
3. **PyTorch allocator 触发 cudaFree** — 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. **GC 暂停** — Python `gc.disable()` 关键路径
5. **ECC 错误 / 时钟降频** — 看 `nvidia-smi -q -d CLOCK,TEMPERATURE`，温度 > 83℃ 会降频
6. **NUMA miss** — 见 `nv-infer-system-setup`
7. **Spot/Preempted instance / 共享 GPU 邻居噪音** — MIG/MPS 隔离

针对 p99 跑 1000+ requests 做直方图，不要只看均值。

## CI / 防回归

```bash
# 每个 PR 跑同 batch / seq / model 的 nsys，对比 stats
nsys profile -o pr.qdrep --force-overwrite=true python bench.py
nsys stats --report gputrace --format csv pr.qdrep > pr.csv
python ci/compare_perf.py baseline.csv pr.csv --tolerance 3%
```

## 反模式

- 直接看 `nvidia-smi --query-gpu=utilization.gpu` 判好坏 — 这个数字只表示某个 SM 在跑 kernel，不代表跑得好
- 用 `time` 测延迟没做 warmup — 第一次包含 JIT/cuDNN 选 kernel/分配
- profile 时还在跑别的负载 — 隔离环境再说
- 关 NVTX 之后 profile —— 没标注的 timeline 找原因像盲拆
