---
name: nv-infer-multigpu
description: NVIDIA 多 GPU / 多节点推理优化——Tensor Parallel (TP) / Pipeline Parallel (PP) / Expert Parallel (EP) / Context Parallel (CP) 选型、NCCL 调优、NVLink/NVSwitch/IB 拓扑、all-reduce / all-to-all、p2p、网络环境变量、RDMA、GPUDirect、NIXL / Dynamo 分布式 KV 传输、跨节点延迟。当用户问"TP/PP/EP/CP"、"NCCL"、"all-reduce"、"NVLink"、"NVSwitch"、"InfiniBand"、"GPUDirect"、"多卡 scaling 不行"、"多节点慢"、"为什么 8 卡不是 1 卡的 8 倍"时使用。
---

# Multi-GPU / Multi-Node Inference

> 多卡 scaling 不到 0.8×/卡基本说明你被通信卡住了。先调通信和拓扑，再谈算法。

## 并行策略选型（推理）

| 策略 | 切什么 | 通信 | 何时用 |
|---|---|---|---|
| **TP** (Tensor Parallel) | 单层 weight 横切（如 GEMM 切 M 维） | 每层 all-reduce | LLM 单卡装不下，且 NVLink/NVSwitch 充足 |
| **PP** (Pipeline Parallel) | 切 layer 到不同 GPU | 层间 P2P | 跨节点（IB），TP 之后还要扩 |
| **EP** (Expert Parallel) | MoE 的 expert 分到不同 GPU | all-to-all | DeepSeek/Mixtral 类 MoE 推理 |
| **CP** (Context Parallel) | seq 维度切到多卡 | ring all-reduce | 超长上下文（128K+） |
| **DP** (Data Parallel) | 不同 request 到不同卡 | 无（独立实例） | 多副本提升吞吐（最简单） |

**典型组合**：
- 70B LLaMA：TP=4, PP=1, EP=1（单节点 NVLink）
- 405B LLaMA：TP=8, PP=2，跨节点 IB
- DeepSeek-V3 MoE：TP=4, EP=8（NVLink+IB）
- 长上下文 128K：TP=4 + CP=2

## TP / PP / EP 速记

### TP
- All-reduce 在每层激活上，频次高 → **NVLink/NVSwitch 必需**
- TP=8 跨节点（IB）几乎不可行（带宽差几十倍）
- 切 head_dim 的 attention TP 注意 head 数能整除 TP size

### PP
- 通信少（仅层间 P2P）→ 可跨节点
- 推理 PP 不能像训练那样填满 micro-batch，**bubble 很大** → 通常 PP <= 2 才划算
- 配合 in-flight batching：vLLM/TRT-LLM 用 1F1B 风格

### EP
- 每个 token 只走 k 个 expert，每层一次 all-to-all
- expert 不均衡 → 慢的 expert 拖累整体 → 需 router 负载均衡
- DeepEP（DeepSeek）/ NVIDIA TensorRT-LLM EP plugin

### CP
- ring attention：每卡持一段 KV，沿 seq 维 rotate
- 适合 prefill 长 prompt；decode 阶段 CP 收益小

## NCCL 调优

### 必设环境变量

```bash
# 让 NCCL 自动选拓扑（多数情况够）
export NCCL_DEBUG=WARN

# 单节点内强制走 NVLink/NVSwitch（不要 PCIe fallback）
export NCCL_P2P_LEVEL=NVL

# 多节点跨主机时禁用某些不稳定路径
export NCCL_IB_DISABLE=0          # 用 IB
export NCCL_IB_HCA=mlx5_0,mlx5_1  # 指定卡
export NCCL_IB_GID_INDEX=3        # RoCE 用
export NCCL_NET_GDR_LEVEL=PHB     # 启用 GPUDirect RDMA

# 显式选 SHARP / Tree / Ring 算法（默认自动）
export NCCL_ALGO=Tree             # 大 message 试 Ring
export NCCL_PROTO=Simple,LL,LL128

# 减少 CPU thread 抢占
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# 多 stream 并发上限（影响 overlap）
export CUDA_DEVICE_MAX_CONNECTIONS=32
```

### Debug 拓扑

```bash
# Topology dump
NCCL_DEBUG=INFO NCCL_TOPO_DUMP_FILE=topo.xml mpirun ...

# 带宽测试
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests && make MPI=1
mpirun -np 8 ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
# H100 NVSwitch 全互联 8 卡 all-reduce ≥ 350 GB/s 算好
# 跨 8 节点 IB 200Gb 全互联 ≥ 20 GB/s 算好
```

低于预期 50% 的 → 拓扑配错 / Fabric Manager 没起 / 进程 NUMA 没 pin。

## NVLink / NVSwitch

```bash
# 看链路状态
nvidia-smi nvlink -s
nvidia-smi nvlink --status -i 0

# 看链路利用（dcgmi）
dcgmi dmon -e 449,450  # NVLink RX/TX bytes
```

H100 SXM：每卡 18 条 NVLink 4，总 900 GB/s 双向。NVSwitch 让 8 卡两两全速。
没 NVSwitch 的 PCIe 服务器（如 H100 PCIe）：跨卡 P2P 走 PCIe Gen5 = 64 GB/s/方向，**TP 必死**。

## InfiniBand / GPUDirect RDMA

跨节点：
- IB 200/400 Gb/s（NDR / XDR），单端口 ≈ 25/50 GB/s
- RoCE v2 over Ethernet 也行，但要严格配 PFC/ECN
- **GPUDirect RDMA** 让网卡直接 DMA GPU 显存，绕过 host bounce buffer
  - 要求：MOFED + nvidia_peermem.ko 加载 + 同 NUMA 上 GPU 和 NIC
  - 验证：`lsmod | grep nvidia_peermem` + `NCCL_DEBUG=INFO` log 里有 `using GPUDirect RDMA`

```bash
# IB 测带宽
ib_write_bw -d mlx5_0 -F --report_gbits
# GPU 跨节点测
mpirun -np 16 -hostfile hosts ./build/all_reduce_perf -b 1M -e 1G -g 1
```

PCIe ↔ NIC ↔ GPU 拓扑：
```bash
nvidia-smi topo -m
# 看 NIC mlx5_X 和 GPU 的关系：
# NV# = NVLink#  PIX = same PCIe switch  PHB = same NUMA  SYS = cross-NUMA
# 选 PIX/PHB 的 NIC，不要 SYS
```

K8s 里通过 `nvidia.com/mlnx_sriov_rdma` resource + topology manager 绑同 NUMA。

## All-Reduce 拓扑选择

NCCL 自动按 message size 选 Ring / Tree / SHARP：
- **小 msg（< 64KB）**：LL128 protocol + Tree
- **大 msg（> 1MB）**：Simple + Ring
- **SHARP（NVSwitch + Quantum-2/3 IB）**：硬件 in-network reduce，**大 msg 快 1.5-2×**
  - 验证：`NCCL_DEBUG=INFO` log 出现 `using SHARP`

## NIXL / Dynamo（分布式 KV）

NVIDIA Dynamo (NIXL = NVIDIA Inference Xfer Library) 让多节点 LLM 推理时 KV cache 在 GPU 之间直接通过 NVLink + IB 传输，绕过 CPU/PCIe：

用途：
- prefill/decode disaggregation 跨节点
- KV cache 跨 worker 迁移（rebalance）

实操：vLLM 0.6+ / TRT-LLM 0.13+ 内置 NIXL 后端，配置：
```bash
vllm serve ... --kv-transfer-config '{"kv_connector": "NixlConnector"}'
```

## Overlap：computation + communication

LLM TP 推理的核心 trick：把 `all-reduce` 与下一个 GEMM 部分 overlap。
- TRT-LLM：`--use_custom_all_reduce`（小 size 用 fast path）
- DeepSpeed-Inference / Megatron-LM 推理：内置 async all-reduce
- 自家代码：用 NCCL `commLaunchAsync` + 多 stream

## Scaling 评估

```
efficiency = (N卡吞吐) / (N × 单卡吞吐)
```
理想 1.0。低于 0.8 → 通信瓶颈；低于 0.5 → 拓扑或软件配错。

定位：
```bash
# 看 NCCL 占总时间多少
nsys profile -t cuda,nvtx,nccl -o multi.qdrep mpirun -np 8 python serve.py
nsys stats --report nccl_sum multi.qdrep
```

>30% 在 NCCL 而非 compute → 切 PP / 减 TP / 上更大 NVLink 域。

## 反模式

- TP=8 跨两台 4 GPU 节点 → PCIe/IB 把 NVLink 速度拖到地板
- 没起 Fabric Manager → NVSwitch 退化 PCIe
- NIC 与 GPU 不同 NUMA → GPUDirect 没用上，吞吐折半
- 每个 worker 用 default stream 通信 → 串行化，无 overlap
- DP 副本数 = GPU 数但每副本要装下全模型 → 浪费显存，应 TP
- 跨节点用 TP → 几乎一定慢于单节点 TP + 跨节点 PP/DP
- 同时设 `NCCL_ALGO=Ring NCCL_PROTO=Simple` 又开 SHARP → 互斥，看 log
- 调 NCCL 不重启进程 → env 没生效
- MoE EP 不做 expert balance → 长尾 expert 拖死整 step
