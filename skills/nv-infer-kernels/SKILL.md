---
name: nv-infer-kernels
description: NVIDIA GPU 推理的 CUDA kernel 层调优——kernel fusion、CUDA streams 并发、CUDA Graphs、occupancy / register / shared memory 平衡、warp divergence、cooperative groups、warp-level primitives、dynamic parallelism、thread block clusters (Hopper)、launch overhead、PTX/SASS 检查。当用户问"自定义 CUDA kernel"、"kernel 融合"、"occupancy"、"register spilling"、"warp divergence"、"CUDA Graphs"、"PTX/SASS"、"launch overhead 太高"、"小 kernel 太多"时使用。
---

# CUDA Kernel Tuning for Inference

> 自定义/调内核之前先检查：torch.compile / TensorRT 是否已经做了？多数情况下框架已自动 fuse。
> 仅当 profiler 显示某 kernel 单独占总时间 > 20%，且不是 cublas/cudnn 的标准算子时，才动手。

## 决策树

```
nsys 显示瓶颈在 kernel  → 看 kernel 名
  ├─ 是 cublas/cudnn/cutlass 的标准 GEMM/Conv? 
  │      → 调精度 / layout / 选 algo (cudnn benchmark)，不要重写
  ├─ 是 framework 自动生成的 elementwise + reduce 串?
  │      → torch.compile / inductor fusion (见 nv-infer-compile-stack)
  ├─ 是 attention / softmax / layernorm?
  │      → FlashAttention / TransformerEngine / FA-3 (nv-infer-llm-attention)
  └─ 自家 kernel：进入下面流程
```

## Step 1: 用 ncu 拿瓶颈

```bash
ncu --kernel-name regex:"my_kernel" --launch-skip 50 --launch-count 5 --set full \
    -o k.ncu-rep python infer.py
```

读这几个 section：
- **SpeedOfLight**：compute% vs memory%，谁高谁是瓶颈
- **Occupancy**：theoretical vs achieved，差距大说明被 register/smem 限制
- **WarpStateStats**：哪种 stall 占主（NoInst, Wait, LongScoreboard…）
- **MemoryWorkloadAnalysis**：L1/L2/HBM 各自的吞吐与 hit rate

## Step 2: 常见瓶颈与处方

### A. Memory bound + 低 L2 hit

- shared memory 复用：把多次访问的小 tile 搬进 smem
- 改 tile size，让 working set 装进 smem
- 减少 dtype 大小（FP16/BF16/INT8）
- 调 access pattern 让 coalesce（连续 thread 访连续地址）

### B. Compute bound 但没用 Tensor Core

- 检查 dtype 是否 FP16/BF16/TF32/FP8/INT8
- 检查 M/N/K 维度对齐（FP16 要 8 的倍数；FP8 要 16）
- 用 cublasLt / cuBLAS，让其自动选 Tensor Core kernel
- 或者上 CUTLASS / cuBLASDx 写定制 epilogue

### C. Low occupancy

- ncu 看 `launch__registers_per_thread`：> 64 通常限制 occupancy
  - 用 `__launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM)` 告诉编译器你的目标
  - 减少局部数组、避免编译器把变量当寄存器堆
  - `nvcc -maxrregcount=N`
- `launch__shared_mem_per_block` 太大 → 减 tile
- block size 太大 → 一个 SM 装不下多个 block

### D. Warp divergence

- 减少 `if (threadIdx.x % 2 == 0)` 这类按 thread index 分支
- 把 mask 数据预排序，让一个 warp 内的活/死分支一致
- 用 `__ballot_sync` / `__shfl_sync` 做 warp 内合并
- ncu `smsp__sass_average_branch_targets_threads_uniform.pct` 应 > 95%

### E. Stall on long scoreboard

- global load 太远，pipeline 太短
- 用 `cp.async`（Ampere+）发起异步 load，再 `cp.async.wait` 同步
- 或更大 batch / unroll，给 latency hiding 留余地

## Kernel Fusion

eager mode 的 `x.add_(a).mul_(b).relu_()` 跑成三个 kernel + 多次 HBM 来回。

| 工具 | 何时用 |
|---|---|
| `torch.compile` (Inductor) | 默认首选，自动 fuse elementwise + reduce |
| Triton | 想写自定义 fused kernel、原 PyTorch 没覆盖 |
| TensorRT / TRT-LLM | 整图编译，fuse 范围最大 |
| CUTLASS / cuBLASDx + epilogue | GEMM 末尾接 bias/act/quantize |

简单的 elementwise fusion，**先 torch.compile**：

```python
@torch.compile(mode="reduce-overhead")
def fused_norm_act(x, w, b):
    x = torch.nn.functional.layer_norm(x, x.shape[-1:], w, b)
    return torch.nn.functional.gelu(x)
```

需要专家手写的场景：
- Attention 变体（FA 不支持的 mask、ALiBi、特殊位置编码）
- 复杂 epilogue（GEMM + bias + dequant + RMSNorm）→ 用 CUTLASS epilogue
- 小 batch decode kernel（cublas 选错 algo）

## CUDA Streams：并发的基础

每个 stream 内顺序，跨 stream 可并发（受 hardware queue 限制）。

```cpp
cudaStream_t s_compute, s_copy;
cudaStreamCreate(&s_compute);
cudaStreamCreate(&s_copy);
kernel<<<g,b,0,s_compute>>>(...);
cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, s_copy);
cudaEventRecord(ev, s_copy);
cudaStreamWaitEvent(s_compute, ev, 0);
```

PyTorch：
```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
with torch.cuda.stream(s1): a = layer1(x)
with torch.cuda.stream(s2): b = layer2(y)
torch.cuda.current_stream().wait_stream(s1)
torch.cuda.current_stream().wait_stream(s2)
```

环境变量解锁更多 HW queues：
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32     # 默认 8
```

## CUDA Graphs — 消灭 launch overhead

每次 launch kernel host 端要花 ~5-10 μs。decode 阶段一个 token 几十个 kernel，host 都喘不过气。

```python
# torch.compile mode="reduce-overhead" 自动包 CUDA Graph
opt = torch.compile(model, mode="reduce-overhead", fullgraph=True)

# 手动方式
g = torch.cuda.CUDAGraph()
static_in  = torch.zeros_like(sample_in)
static_out = torch.zeros_like(sample_out)
# warmup
for _ in range(3): model(static_in)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    static_out.copy_(model(static_in))

# 推理时
static_in.copy_(real_in)
g.replay()
real_out = static_out.clone()
```

**适合**：
- 固定 batch / seq 的 decode step
- ViT 固定输入
- CNN 固定输入

**不适合**（或要分桶）：
- prefill 阶段（seq 不定）
- dynamic batch size
- 含 if/else 的控制流

LLM 实战：把 decode step 分成几个 fixed batch size 桶，每个桶一张 Graph（vLLM/TRT-LLM 都这么做）。

## Hopper 专项

### Thread Block Clusters (CGA)

H100 引入 cluster，多个 block 共享 distributed shared memory：
```cuda
__cluster_dims__(2, 2, 1) __global__ void k(...) {
    auto cluster = cooperative_groups::this_cluster();
    int rank = cluster.block_rank();
    ...
}
```
适合大 GEMM tile、跨 block 的 collective。CUTLASS 3.x 大量用。

### TMA (Tensor Memory Accelerator)

Hopper 的 TMA 让全块异步加载/存储 tile，不再用 LDG/STG 手填：
- 通过 CUTLASS / cuBLASDx 间接用
- 写裸 CUDA：用 `cuda::barrier` + `cuda::memcpy_async` API
- Triton 3.0+ 自动用 TMA

### WGMMA

Warp-Group MMA：8 个 warp 协同发起一个大 GEMM tile。比 mma.sync 大、占 register 少。
不要自己写，调 CUTLASS。

## Cooperative Groups & warp primitives

```cuda
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
auto warp  = cg::tiled_partition<32>(block);

// warp 内求和
float v = ...;
for (int o = 16; o > 0; o /= 2) v += warp.shfl_down(v, o);
// v 在 lane 0
```

`__shfl_sync` / `__ballot_sync` / `__match_any_sync` 是写高性能 reduce/scan/histogram 的关键。

## Library 优先（永远先试）

不要重写：
- GEMM → cuBLAS / cuBLASLt
- Conv → cuDNN (cudnn benchmark 自动选 algo)
- FFT → cuFFT
- Sort/Scan → CUB / Thrust
- Sparse → cuSPARSELt
- 量化 GEMM → cuBLASLt INT8 / FP8 path、Marlin、Machete
- Attention → FlashAttention 2/3、xFormers、TransformerEngine

## PTX / SASS 检查（高阶）

当你觉得 nvcc 没生成最优指令：

```bash
# 生成 PTX
nvcc -ptx my.cu -o my.ptx
# 反汇编 cubin
cuobjdump --dump-sass my.cubin > my.sass
# 看具体 SM arch
nvcc -gencode=arch=compute_90a,code=sm_90a ...
```

关注：
- 是否生成 `mma.sync` / `wgmma` Tensor Core 指令
- 是否有大量 `LDG.E.64` 而非 `LDG.E.128`（说明对齐不好）
- 是否 `STL`/`LDL` 频繁（register spill）

## 验证

| 改动 | 测什么 |
|---|---|
| 加 fusion | nsys 看 kernel 数减少 + 单步时间下降 |
| 上 CUDA Graphs | nsys 看 host 端 gap 消失 + p99 改善 |
| Shared memory tiling | ncu MemoryWorkloadAnalysis L1 hit 上升、HBM BW 下降 |
| 改 dtype | ncu Tensor Active% 上升 |
| 减 register | ncu Achieved Occupancy 上升 |

## 反模式

- 上来就写自定义 CUDA，没试 torch.compile / TRT
- CUDA Graph 在 dynamic shape 下用 → 频繁 capture 反而更慢
- Stream 创建一堆但都 sync 到默认 stream → 没真正并发
- `__syncthreads()` 在分支内 → undefined behavior
- 用 `__shfl` 不传 mask → 跨 warp lane 行为未定义
- 自己写 softmax 不做 max-subtraction → 数值不稳
