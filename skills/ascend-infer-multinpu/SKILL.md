---
name: ascend-infer-multinpu
description: Ascend 多 NPU / 多节点推理优化——TP/PP/EP/DP、HCCL (Huawei Collective Communication Library)、HCCS (节点内 8 卡互联)、HCCN (跨节点 RoCE/IB)、拓扑感知、all-reduce / all-to-all、Atlas 800 集群、跨节点延迟与带宽。当用户问"昇腾多卡"、"NPU TP"、"HCCL"、"HCCS"、"HCCN"、"NPU 集群"、"Ascend RoCE"、"8 卡 scaling 不行" 时使用。
---

# Ascend Multi-NPU / Multi-Node Inference

> 与 nv-infer-multigpu 平行。Ascend 集群拓扑：**节点内 HCCS（8 卡全互联）+ 节点间 HCCN（RoCE/IB）**。
> 多 NPU scaling < 0.8×/卡 几乎一定是通信问题，先调拓扑再谈算法。

## 并行策略选型（推理）

| 策略 | 切什么 | 通信 | 何时用 |
|---|---|---|---|
| **TP** (Tensor Parallel) | 单层 weight 切 | 每层 all-reduce | LLM 单卡装不下，节点内 HCCS 充足 |
| **PP** (Pipeline Parallel) | 切 layer | 层间 P2P | 跨节点 |
| **EP** (Expert Parallel) | MoE 的 expert 切 | all-to-all | DeepSeek/Mixtral 类 |
| **DP** (Data Parallel) | 不同 req 到不同卡 | 无 | 多副本提吞吐 |

典型组合（910B 8 卡节点）：
- Qwen3-8B：DP=8（每卡一份）或 TP=1
- Qwen3-32B：TP=2，DP=4
- LLaMA-3-70B：TP=4 + DP=2 或 TP=8
- DeepSeek-V3 (236B MoE)：跨 4 节点 TP=8 + EP=8

## HCCL 关键环境变量

```bash
# 时间限制
export HCCL_CONNECT_TIMEOUT=1200            # 连接超时秒
export HCCL_EXEC_TIMEOUT=1200               # 执行超时秒

# Buffer / 性能
export HCCL_BUFFSIZE=512                    # 单消息 buffer MB，影响大消息吞吐
export HCCL_RDMA_TC=...                     # RDMA Traffic Class
export HCCL_RDMA_SL=...                     # Service Level

# 路径
export HCCL_OVER_OFI=0                      # 单节点不要走 OFI，跨节点才需要
export HCCL_WHITELIST_DISABLE=1             # 关闭白名单（自家集群）

# Debug / 日志
export ASCEND_GLOBAL_LOG_LEVEL=3            # 关日志（生产）
export HCCL_DEBUG=INFO                      # debug 时用
```

## 节点内 HCCS（8 卡全互联）

910B 节点内 8 卡通过 **HCCS** 全互联（类比 NVSwitch），总带宽 392 GB/s 双向聚合。

### 拓扑验证

```bash
npu-smi info -t topo
# 期望输出：每对卡之间显示 HCCS 链路
# 输出格式类似:
#       NPU0    NPU1    NPU2    ...
# NPU0  X       HCCS    HCCS    ...
# NPU1  HCCS    X       HCCS    ...
```

如果显示 `PCIe` 而非 `HCCS` —— 拓扑没建好（驱动/固件问题），TP 性能将塌。

### HCCL 带宽 micro-bench

```bash
# CANN 自带 hccl_test
cd /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test
./bin/all_reduce_test -b 8 -e 1G -f 2 -d fp16 -p 8

# 期望（单节点 8 卡 HCCS, 1GB msg）：
# - all_reduce: ≥ 200 GB/s aggregate
# - all_gather: ≥ 150 GB/s
# - reduce_scatter: ≥ 200 GB/s
```

< 50% 期望值 → 拓扑配错 / 进程 NUMA pin 错。

## 跨节点 HCCN（RoCE / IB）

Atlas 800 节点通常配 **8 个 200 Gb RoCE NIC**（每卡对应一个）。

### 配置

```bash
# 给每个 HCCN NIC 配 IP（同子网内不同 IP）
hccn_tool -i 0 -ip -s address 192.168.10.10 netmask 255.255.255.0
hccn_tool -i 1 -ip -s address 192.168.10.11 netmask 255.255.255.0
...
hccn_tool -i 7 -ip -s address 192.168.10.17 netmask 255.255.255.0

# 验证 link
hccn_tool -i 0 -link -g                      # 链路状态
hccn_tool -i 0 -net_health -g                # 网络健康
hccn_tool -i 0 -tls -s enable 0              # TLS 关，性能优先
```

集群所有节点 IP 写到 `ranktable.json`：
```json
{
  "version": "1.0",
  "server_count": "2",
  "server_list": [
    {
      "server_id": "node0",
      "device": [
        {"device_id": "0", "device_ip": "192.168.10.10", "rank_id": "0"},
        {"device_id": "1", "device_ip": "192.168.10.11", "rank_id": "1"},
        ...
      ]
    },
    { "server_id": "node1", ... }
  ]
}
```

启动时传 `RANK_TABLE_FILE=ranktable.json`。

### RDMA 性能

```bash
# IB 风格测试
ib_write_bw -d mlx5_0 -F --report_gbits

# 期望（200 Gb RoCE 单 NIC）：~22-24 GB/s
# 8 NIC 聚合：> 150 GB/s
```

NIC ↔ NPU 同 NUMA 才能走最优路径。

## NCCL → HCCL 心智映射

| NCCL 概念 | HCCL 等价 | 备注 |
|---|---|---|
| `NCCL_P2P_LEVEL=NVL` | （自动用 HCCS，无需 env） | |
| `NCCL_IB_HCA=mlx5_X` | `hccn_tool` 配 NIC IP | |
| `NCCL_DEBUG=INFO` | `HCCL_DEBUG=INFO` | |
| `NCCL_NSOCKS_PERTHREAD` | （无直接对等） | |
| `NCCL_ALGO=Ring/Tree` | HCCL 自动选 | |
| `NCCL_NVLS_ENABLE` | （HCCL 在 NVSwitch SHARP 等价上较少暴露） | |
| `ncclAllReduce` | `HcclAllReduce` | |
| `nccl-tests` | `hccl_test` | |
| GPUDirect RDMA | 同等的 NPU Direct RDMA（自动启用，配套 hccn_tool） | |

## Overlap：computation + communication

LLM TP 推理的核心 trick —— 把 all-reduce 与下一 GEMM 部分重叠。
- MindIE-LLM 内部已实现（不用关心）
- 自家代码：用 `aclrtStream` 多流 + HCCL async 接口

## EP（MoE Expert Parallel）

MindIE-LLM 配置：
```json
{
  "moeExpertParallel": true,
  "moeEpSize": 8,
  "moeTpSize": 1
}
```

DeepSeek-V3 类 MoE 在 8 卡 910B + EP=8 上吞吐良好；跨节点 EP 走 HCCN，性能阶跃下降，调度需让 expert 分布优先节点内。

## 多节点部署样板

```bash
# 节点 0 / 节点 1 各自跑（环境一致）
export RANK_TABLE_FILE=/etc/ranktable.json
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_BUFFSIZE=512

mindieservice_daemon --config /etc/mindie/llama405b_service.json
```

`service.json` 中：
```json
{
  "ModelConfig": [{
    "modelName": "Llama-3-405B",
    "worldSize": 16,
    "tensorParallelSize": 8,
    "pipelineParallelSize": 2,
    "npuDeviceIds": [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]]
  }]
}
```

## Scaling 评估

```
efficiency = (N卡吞吐) / (N × 单卡吞吐)
```
理想 1.0；低于 0.8 → 通信瓶颈；低于 0.5 → 拓扑或软件配错。

```bash
# 看 HCCL 占总时间
msprof --hccl=on --output=./prof --application="..."
# 解析 hccl_*.csv，对比 op 总时长
```

> 30% 在 HCCL（非计算）→ 切 PP / 减 TP / 改拓扑

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 拓扑显示 PCIe 而非 HCCS | TP 慢 5-10× | 重启 driver / 检查固件 |
| HCCN IP 未配 | 跨节点 init 卡死 | hccn_tool 给每 NIC 配 IP |
| `ranktable.json` device_ip 写错 | init 失败 / 通信卡 | 严格按 hccn_tool 实际 IP |
| 跨节点 TP（不推荐） | 远慢于节点内 TP | 节点内 TP + 节点间 PP/DP |
| HCCL_BUFFSIZE 太小 | 大消息切片多，慢 | 设 512 MB |
| RDMA TLS 没关 | 加密开销大 | hccn_tool -tls -s enable 0 |
| Fabric Manager-like 服务缺 | 链路不全 | Ascend driver 自带，重启 driver |
| Spot / 节点重启 | rank 顺序变 | ranktable 用 server_id 锚 |
| TP 配在不同节点的 8 卡 | scaling 崩 | 严格节点内全互联才上 TP |
| MoE EP 不均衡 | 长尾 expert 拖死 step | 路由 + balance loss / 提前 batch 平滑 |

## 监控

```bash
# HCCL 流量
npu-smi info -t link -i 0 -c 0    # 链路统计

# RoCE 网卡（如果可见）
ethtool -S enp0s0
```

## 与 NV 对应

| NV | Ascend |
|---|---|
| NCCL | HCCL |
| NVLink + NVSwitch | HCCS（节点内 8 卡） |
| InfiniBand / RoCE + GPUDirect RDMA | HCCN (RoCE) + NPU Direct RDMA |
| nccl-tests | hccl_test |
| `NCCL_*` env | `HCCL_*` env |
| Fabric Manager | （Ascend driver 内置） |
| `nvidia-smi nvlink -s` | `npu-smi info -t link -i 0 -c 0` |
| `nvidia-smi topo -m` | `npu-smi info -t topo` |

## 反模式

- TP=16 跨两节点 → 绝大多数情况下慢于节点内 TP=8 + 节点间 PP=2
- HCCN 没配 IP 直接跑跨节点 → init 阶段卡死或 fallback 慢路径
- 多节点共用一个 server_id → ranktable 不识别
- 调 HCCL env 后不重启进程 → 不生效
- MoE EP 没做 router balancing → 长尾 expert 拖死

## 参考

- CANN HCCL 用户指南
- Atlas 800 集群部署手册
- `references/hardware/huawei-atlas-910b.md`
- `references/hardware/tooling-cheatsheet.md`
- `ascend-infer-llm-stack` —— LLM TP/PP/EP 配置
