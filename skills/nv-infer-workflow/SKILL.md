---
name: nv-infer-workflow
description: NVIDIA GPU 推理性能优化的总入口与流程。当用户提到 CNN/ViT/LLM 推理调优、想"提升推理吞吐/降延迟"、问"从哪里下手"、不确定优化方向时使用。给出 80/20 优先级、阶段化决策树，并指向更专门的 nv-infer-* skills。改编自 Chris Fregly《AI Systems Performance Engineering》Appendix 的 175+ 项 checklist，聚焦推理场景。
---

# NVIDIA Inference Tuning Workflow (CNN / ViT / LLM)

## 何时用这条 skill
- 用户说"我的模型推理慢/吞吐低/成本高"，但还没说明卡在哪一层
- 用户问"应该先做什么"、"有没有 checklist"
- 准备一次完整的 inference 调优 sprint，需要总体节奏

不要在这条 skill 里给具体内核代码 — 先帮用户**定位**，再分发到下面的子 skill。

## 阶段化优化顺序（80/20）

每个阶段先做"测量 → 实施 → 再测量"。**不测量的优化等于没做。**

```
Phase 0  Baseline   ──► 记录基线吞吐/p50/p99/SM 利用率/HBM BW/能耗
Phase 1  System     ──► 驱动/CUDA/MIG/MPS/NUMA/clocks/ECC          → nv-infer-system-setup
Phase 2  Profile    ──► Nsight Systems 找首要瓶颈                  → nv-infer-profiling
Phase 3  Stack      ──► torch.compile / TensorRT(-LLM) / Graphs    → nv-infer-compile-stack
Phase 4  Precision  ──► FP16/BF16/FP8/INT8/INT4 + 2:4 sparsity     → nv-infer-precision
Phase 5  LLM Attn   ──► FlashAttn / PagedAttn / KV / prefill split → nv-infer-llm-attention
Phase 6  Batching   ──► continuous batch / dynamic batch / 调度    → nv-infer-serving
Phase 7  Memory     ──► layout / pinned / coalescing / channels_last → nv-infer-memory
Phase 8  Kernels    ──► fusion / streams / CUDA Graphs / occupancy → nv-infer-kernels
Phase 9  Multi-GPU  ──► TP/PP/EP, NCCL, NVLink                     → nv-infer-multigpu
Phase 10 IO/Data    ──► DALI / pinned / prefetch (CNN/ViT 重要)    → nv-infer-io-data
Phase 11 Model      ──► CNN/ViT 专项                              → nv-infer-cnn-vit
```

通常 **Phase 0–4 足以给单卡推理 2-4× 提升**；多卡 LLM 再加 5,6,8,9。

## 第一步：建立基线（永远做这件事）

无论用户接下来想优化什么，**先要这些数字**：

```bash
# 实时 GPU 状态（先 nvidia-smi dmon 跑 30s）
nvidia-smi dmon -s pucvmet -c 30

# 关键基线指标（让用户自己跑当前的推理脚本，并记录）
# - throughput: tokens/s 或 images/s 或 req/s
# - latency:    p50 / p95 / p99
# - SM util:    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv
# - HBM BW:     dcgmi dmon -e 1005,1004 -c 30   (DRAM active%, Gr Active%)
# - Power:      nvidia-smi --query-gpu=power.draw,temperature.gpu --format=csv -l 1
```

把基线数字记到 git 里（建议 `bench/baseline.md`），后续每次优化都更新对比。

## 80/20 路径：从基线指标推断瓶颈

| 现象 (看 nsys / nvidia-smi) | 最可能瓶颈 | 跳到哪个 skill |
|---|---|---|
| GPU util 持续 < 70%，CPU/IO 忙 | 数据 pipeline / H2D 慢 | `nv-infer-io-data` |
| SM Active 高但 HBM BW > 80% | memory bound | `nv-infer-memory` + 量化 |
| SM Active 高且 HBM BW < 30% | compute bound | `nv-infer-precision` (低精度) |
| LLM decode 阶段慢，prefill 还行 | KV cache / attention | `nv-infer-llm-attention` |
| batch 拉满但延迟扛不住 | 调度策略 | `nv-infer-serving` |
| 多卡 scaling < 0.8x/卡 | 通信 / 拓扑 | `nv-infer-multigpu` |
| Python 占比 > 20%（看 nsys CPU 行） | host overhead | `nv-infer-compile-stack` (CUDA Graphs) |
| ViT/CNN，FP16 还慢 | 卷积布局 / 算子 | `nv-infer-cnn-vit` |

## CNN / ViT / LLM 三类的差异化重点

**CNN 推理（ResNet, EfficientNet, YOLO, …）**
- 永远开 `channels_last` (NHWC)：cuDNN 在 NHWC 下 Tensor Core 路径更顺
- 卷积 + BN + ReLU 融合（TensorRT 自动，eager mode 用 `torch.compile`）
- INT8 calibration 通常 2× 提速、精度损失 < 1%；FP8 在 Hopper/Blackwell 起飞
- 大 batch + dynamic shape 时检查是否被 cuDNN benchmark 反复重选 kernel

**ViT 推理**
- patch embed 是 stride 卷积，cuDNN 算法不一定最优 — TensorRT/torch.compile 通常更快
- MHSA 永远走 FlashAttention（SDPA 已经自动选择，但需 PyTorch ≥ 2.0 + Hopper/Ampere）
- 小 seq_len (197/577) 下 prefer 单卡，TP 收益低
- Swin/DeiT 等带 windowed attention 的，自定义 mask 容易禁用 FlashAttn 后端

**LLM 推理**
- prefill 和 decode 是两种 workload：prefill 是 compute-bound、decode 是 memory-bound
- KV cache 是头号显存吃货 — 算 `2 × n_layer × n_kv_head × head_dim × dtype × seq × batch`
- 用 PagedAttention（vLLM）或 TensorRT-LLM 的 paged KV，几乎一定要
- 长上下文 + 高并发：prefill/decode disaggregation（不同卡/不同实例）
- 量化优先级：INT8 W8A8 → FP8 → INT4 weight-only (AWQ/GPTQ)

## 完整 175+ checklist 在子 skills 里如何分布

| 书中章节 | 对应 skill |
|---|---|
| Performance Tuning & Cost Mindset | 本 skill（流程） |
| Reproducibility & Documentation | 本 skill（基线 & CI） |
| System Architecture & Hardware Planning | `nv-infer-system-setup` |
| Multi-GPU Scaling & Interconnect | `nv-infer-multigpu` |
| OS / Driver / Kernel Tuning | `nv-infer-system-setup` |
| GPU Resource Management & Scheduling | `nv-infer-system-setup` |
| I/O Optimization | `nv-infer-io-data` |
| Data Processing Pipelines | `nv-infer-io-data` |
| Profiling / Debugging / Monitoring | `nv-infer-profiling` |
| GPU Programming & CUDA Tuning | `nv-infer-kernels` |
| Compiler Stack (torch.compile, TRT) | `nv-infer-compile-stack` |
| Arithmetic / Mixed Precision | `nv-infer-precision` |
| Algorithmic & Sparsity (FlashAttn) | `nv-infer-llm-attention` + `nv-infer-precision` |
| Networking (NCCL/RDMA) | `nv-infer-multigpu` |
| Efficient Inference & Serving | `nv-infer-serving` |
| Multinode Inference & Serving | `nv-infer-serving` + `nv-infer-multigpu` |
| Power / Thermals / Energy | `nv-infer-system-setup` |

## 一次完整 sprint 的样板（粘给用户）

```
Day 1  跑基线 + nsys profile + 填上面的瓶颈对照表
Day 2  系统/驱动层快赢（持久化模式、MIG/MPS、NUMA pin、clock lock）
Day 3  Phase 3 编译栈（torch.compile or TRT-LLM/vLLM）
Day 4  Phase 4 精度（FP16→FP8/INT8，跑精度 regression）
Day 5  LLM：KV/Attention；CNN/ViT：channels_last + conv 融合
Day 6  调度策略（continuous batching / dynamic batch）
Day 7  再 profile，对比基线，写 report，加 CI bench 防回归
```

## 反模式（别这样）

- "我先重写一个 CUDA kernel" — 99% 情况下，编译栈和精度先做完再说
- 没基线就上 TensorRT-LLM 重构 — 不知道是不是已经够快
- 同时改五个变量 — 一次只改一个，便于二分
- 关掉 ECC 抢吞吐 — 数据校验失败比慢一点贵得多
- 用 `--no-verify` 绕过 perf CI — 看 `nv-infer-profiling` 怎么定位真正回归

## 复盘 / 防回归

- 把基线和每个阶段后的 metrics 提交进 `bench/`
- 在 CI 加 perf 守门（pytest-benchmark 或自写 nsys 脚本），允许 ±3% 波动
- 文档放在 repo wiki/README，**记录每个 env var 与版本号**（驱动、CUDA、cuDNN、TRT、torch、vLLM）

> 想深入某一阶段，直接问 "用 nv-infer-profiling" 或 "查 nv-infer-precision"。

## See also

- 硬件细节（T4 / H100 / B200 各 SKU 差异、TDP、互联）：[`references/hardware/`](../../references/hardware/)
- 模型族特定数字（YOLO / CLIP / MiniCPM-V / Qwen3-8B）：[`references/models/`](../../references/models/)
- 同栈 Ascend (910B / 300V Pro) 调优：[`ascend-infer-workflow`](../ascend-infer-workflow/SKILL.md)；命令对照见 [`tooling-cheatsheet`](../../references/hardware/tooling-cheatsheet.md)
