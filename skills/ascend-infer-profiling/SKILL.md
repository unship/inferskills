---
name: ascend-infer-profiling
description: Ascend NPU 推理性能分析——msprof、MindStudio Profiler、torch_npu profiler、aic-metrics、AI Core / Vector / Scalar / MTE pipeline 利用率、AICPU fallback 检测、HCCL 性能、内存访问分析。当用户问"昇腾 profile"、"msprof"、"MindStudio"、"AI Core 利用率低"、"AICPU fallback"、"npu 慢"、"NPU 利用率"时使用。
---

# Ascend Profiling & Diagnostics

> 与 nv-infer-profiling 平行。Ascend 的"看时间线"工具是 **msprof**（CLI）+ **MindStudio Profiler**（IDE 可视化），与 NVIDIA 的 nsys + Nsight 关系类似。

## 黄金顺序

1. **`npu-smi info watch`** — 宏观利用率、HBM、功耗 30 秒看一眼
2. **`msprof`** — 时间线，看 op 分布、AICPU fallback、HCCL 占比
3. **MindStudio Profiler** — 单算子微观（Cube/Vector/Scalar/MTE pipeline 利用率）
4. **`torch_npu.profiler`** — 框架层算子归属 + Python overhead

跳级是常见错误：没看 msprof 就先深挖单算子，浪费时间。

## Step 1: 宏观状态

```bash
# 实时监控（类似 nvidia-smi dmon）
npu-smi info watch -i 0 -d 1
# 关注: AI Core(%), Memory(%), Power(W), Temp(℃), Freq

# 单字段查询
npu-smi info -t common -i 0 -c 0
```

判定方向（910B 上）：
- AI Core 持续 < 60% & CPU 忙 → 数据 pipeline 瓶颈（看 `ascend-infer-io-data`）
- AI Core 高 + Memory > 70% → memory bound（降精度、调 layout）
- AI Core 抖动剧烈 → batch 不齐 / dynamic shape / 大量小 op

## Step 2: msprof 时间线采集

最常用命令：

```bash
msprof --output=./prof_out \
       --application="python infer.py" \
       --aic-metrics=PipeUtilization,ArithmeticUtilization,MemoryAccess \
       --aicpu=on \
       --runtime-api=on \
       --task-time=on \
       --hccl=on
```

关键参数：
- `--aic-metrics`：AI Core 指标，常用：
  - `PipeUtilization` — Cube/Vector/Scalar/MTE 流水线占比
  - `ArithmeticUtilization` — 计算单元利用率
  - `MemoryAccess` — HBM/UB 访存统计
  - `Memory` — 总线带宽
  - `MemoryL0` / `MemoryL1` / `MemoryUB`
- `--aicpu=on`：AICPU（NPU 内嵌的通用 CPU）算子捕获 —— **发现 fallback 关键**
- `--runtime-api=on`：acl* runtime 调用
- `--hccl=on`：集合通信
- `--task-time=on`：精细的任务时间戳

### 限定 profile 范围

不要 profile 整个进程；profile 5-10 个稳态 step：

```python
import torch_npu

# 跑 warmup，然后开 profile
for i, batch in enumerate(loader):
    if i == 5:
        torch_npu.npu.profiler.start()
    if i == 10:
        torch_npu.npu.profiler.stop()
        break
    run(batch)
```

或者用 `msprof.range_push / range_pop` 给阶段打点（类比 nvtx）：

```python
import mindspore.profiler as msp     # MindSpore
# 或者 torch_npu 的 profiler 上下文
```

## Step 3: 用 MindStudio 打开 .prof 目录

```
prof_out/
└── PROF_xxxx_xxx/
    └── device_*/
        └── data/
            ├── op_summary.csv
            ├── aicpu_xxx.csv
            ├── hccl_*.csv
            └── ...
```

MindStudio 加载后看：
- **Timeline View** —— host / device / HCCL / AICPU 时间线
- **Op Summary** —— 算子耗时排序
- **Pipe Utilization** —— Cube/Vector/Scalar/MTE 占比
- **Memory** —— HBM、UB、L1 各层带宽

### CLI 直接看 op_summary

```bash
# top-20 耗时算子
awk -F',' 'NR>1 {print $4, $0}' prof_out/PROF*/device_*/data/op_summary*.csv \
  | sort -rn | head -20
```

## Step 4: torch_npu Profiler

```python
import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_npu"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        run(batch)
        prof.step()
```

跑完 `tensorboard --logdir tb_npu` 看 PyTorch profiler tab。

## 关键现象诊断

### A. AICPU fallback（最常见）

某个 op 不在 AI Core OpAPI 里，被丢给 AICPU 跑 —— 通常**慢 10-100×**。

```bash
# msprof 加 --aicpu=on 后，aicpu_*.csv 列出所有 AICPU 调用
grep -v "^#" prof_out/PROF*/device_*/data/aicpu_*.csv | head
```

修法：
1. 升 CANN（每个 release 都补算子）
2. 改写模型避免该 op（如 unsupported argsort 改成 topk）
3. 写 Ascend C 自定义算子注册（高成本，看 `ascend-infer-kernels`）

### B. 大量小算子 / launch 串行化

类比 NV 的 launch overhead 问题。Ascend 上同样：
- 用 ATC/GE 的 graph mode（自动 fusion）
- TorchAir（PyTorch graph 模式 on Ascend）替代 eager
- 详见 `ascend-infer-compile-stack`

### C. Pipe Utilization 不均（Cube 忙、Vector 闲，反之亦然）

- Cube 单元低利用：GEMM 不饱和 → 增 batch / pad shape 对齐 16
- Vector 高、Cube 低：模型主要是 elementwise（如 LN/激活）→ ATC fusion 不够
- MTE (Memory Transfer Engine) 高 → 内存搬运瓶颈，看 `ascend-infer-memory`

### D. HCCL 占比过高

```bash
# msprof 输出中查看 hccl_*.csv
# 每个 collective 的耗时
```

> 30% 在 HCCL → 通信瓶颈，调拓扑 / 减 TP / 改算法（见 `ascend-infer-multinpu`）

## 推理特定：Tail latency 分析

p99 抖的常见 Ascend 因素：
1. **首次算子编译 / op 缓存未命中** → warmup 充分
2. **动态 shape**：每次新 shape 触发 op 再编译，p99 飙
   - 用 `dynamic_dims` 预枚举
3. **AICPU 算子串行化**：某个 op fallback 阻塞 pipeline → 升 CANN
4. **温度降频**：长跑温度 > 80℃ → 散热
5. **HCCL 抖动**：跨节点 RoCE PFC/ECN 配错

## CI / 防回归

```bash
# 每个 PR 跑同 batch / model 的 msprof，对比 op_summary
msprof --output=./pr_prof --application="python bench.py" --task-time=on
python ci/compare_ascend_perf.py baseline/op_summary.csv pr_prof/op_summary.csv --tol 3%
```

## 常用查询脚本

```bash
# 查看 op 类型分布
awk -F',' 'NR>1 {print $2}' prof_out/PROF*/device_*/data/op_summary*.csv \
  | sort | uniq -c | sort -rn | head -20

# 查看 AICPU 占比
total=$(awk -F',' 'NR>1 {s+=$4} END {print s}' op_summary.csv)
aicpu=$(awk -F',' 'NR>1 {s+=$4} END {print s}' aicpu_*.csv)
echo "scale=2; $aicpu/$total*100" | bc
```

## 与 NVIDIA 工具对应

| NV | Ascend |
|---|---|
| `nsys profile -t cuda,nvtx` | `msprof --task-time=on --runtime-api=on` |
| `ncu --set full` | `msprof --aic-metrics=PipeUtilization,ArithmeticUtilization,MemoryAccess` |
| `nvtx.range_push` | `mindspore.profiler` range / `aclrtProfilerPush` |
| Nsight Systems UI | MindStudio Profiler |
| Nsight Compute UI | MindStudio AI Core Analysis |
| `nvidia-smi dmon` | `npu-smi info watch` |
| `dcgmi dmon -e ...` | `npu-smi info -t ...` |

## 反模式

- 直接拿 `npu-smi info` 的 "AI Core %" 当衡量标准 —— 这是平均利用率，实际 Cube 利用率可能很低
- profile 时还在跑别的负载 —— 隔离环境
- profile 完不 stop —— `.prof` 文件持续写入，磁盘爆
- warmup 不够就 profile —— 第一 step 含编译/cache miss，数据不可信
- 用 transformers `generate()` profile LLM —— eager 路径上 op 数爆炸，看不出真实瓶颈，应 profile MindIE 服务

## 参考

- CANN 文档中"性能调优指南"章节
- MindStudio 用户手册
- `references/hardware/tooling-cheatsheet.md`
