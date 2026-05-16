---
name: nv-infer-system-setup
description: NVIDIA GPU 推理的系统层调优——驱动/CUDA 栈版本、NUMA pin、Transparent Hugepages、locked memory、GPU persistence、clock lock、ECC、MIG/MPS、cgroup/Docker/K8s 配置、能耗与散热。当用户问"机器需要怎么配"、"OS/driver 怎么调"、"NUMA"、"MIG"、"MPS"、"持久化模式"、"功耗墙"、"GPU 降频"、"hugepages"、"为什么 GPU 性能时好时坏" 时使用。
---

# System Setup — Foundation Tuning

> 这一层不调好，上层做再多优化也漏血。OS、驱动、调度器、电源、拓扑是性能的"底板"。

## 检查清单（先一次跑完）

```bash
# 1. GPU 与驱动信息
nvidia-smi -q | head -40
nvcc --version
cat /proc/driver/nvidia/version

# 2. 持久化模式（消除每次 cudaInit 的延迟，几百 ms）
nvidia-smi -pm 1

# 3. 当前时钟与功耗墙
nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.limit,power.draw,temperature.gpu --format=csv

# 4. NUMA 拓扑
nvidia-smi topo -m
numactl -H
lscpu | grep -i numa

# 5. Hugepages / 锁页限制
cat /sys/kernel/mm/transparent_hugepage/enabled
ulimit -l            # locked memory，应该 unlimited
ulimit -n            # file descriptors，推理服务建议 65535+

# 6. ECC 状态（推理生产环境必开）
nvidia-smi -q -d ECC | grep -E "Current|Pending"

# 7. MIG / MPS 状态
nvidia-smi mig -lgi
echo $CUDA_MPS_PIPE_DIRECTORY
```

## 驱动 / CUDA 栈版本对齐

- 升级到最近的稳定 **R550+ / CUDA 12.4+**（Hopper FP8 + cudnn 9 + cublas 12.4）
- 对于 Blackwell (B100/B200)，至少 **R570 / CUDA 12.8+**
- 框架要对齐：PyTorch ≥ 2.3 才有完整 FlashAttention v2/v3，vLLM 0.6+，TensorRT-LLM 0.10+
- **容器镜像统一用 NGC**（`nvcr.io/nvidia/pytorch:24.10-py3` 之类），避免 host 驱动与 cuda-toolkit 版本错配

```bash
# 验证驱动 / 运行时 / 应用都能跑
docker run --gpus all --rm nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## NUMA & CPU 亲和

GPU 通过 PCIe 挂在某个 NUMA node 上。用错 socket 的 CPU 喂数据，**带宽下降 30%+**。

```bash
# 看每个 GPU 对应的 NUMA node
nvidia-smi topo -m
cat /sys/class/pci_bus/0000:??/device/numa_node      # 替换实际 BDF

# 把推理进程 pin 到正确的 NUMA node + CPU
numactl --cpunodebind=0 --membind=0 python serve.py --gpu 0
```

K8s/Slurm 环境下用 `nvidia-container-toolkit` 的 `--numa` 或 device plugin 的 topology manager。

PyTorch 数据加载线程也要 pin：
```python
import psutil, os
p = psutil.Process(os.getpid())
p.cpu_affinity([0,1,2,3,4,5,6,7])   # 让 worker 留在同 socket
```

## Transparent Hugepages

绝大多数 inference workload **开 madvise 比 always/never 好**：

```bash
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

LLM serving 进程内显式 `madvise(MADV_HUGEPAGE)` host 端权重内存可以减 TLB miss。

## Locked memory & swap

```bash
# /etc/security/limits.conf
*   soft   memlock   unlimited
*   hard   memlock   unlimited
*   soft   nofile    1048576
*   hard   nofile    1048576

# 关 swap（推理服务一旦走 swap 必崩 SLO）
sudo swapoff -a
```

Docker 启动用 `--ulimit memlock=-1 --shm-size=16g`，否则 pinned 内存分配会失败。

## GPU 时钟、功耗、ECC

### 时钟锁定（生产推理服务推荐）

让 GPU 时钟不要因为温度抖动，p99 才稳：

```bash
# 查询合法 application clocks
nvidia-smi -q -d SUPPORTED_CLOCKS | head -40

# 锁到接近最高的 P-state
sudo nvidia-smi -ac 1593,1980      # mem,graphics (示例 H100)
# 关 boost 抖动
sudo nvidia-smi --auto-boost-default=DISABLED
```

### 功耗墙

按机房散热实际能力设，**不要盲目拉到 700W** 还散不掉热：

```bash
# H100 SXM 默认 700W，降到 600W 性能损失 5-10% 但温度好很多
sudo nvidia-smi -pl 600
```

DCGM `power.draw` 长期 > 95% TDP 且温度 > 80℃ → 散热瓶颈，会触发 clock throttling，p99 飙。

### ECC

**推理生产**：开 ECC（默认开）。掉吞吐 ~3-5%，换数据可靠性。
**离线 benchmark**：可以关来榨数据，但记得是不可发布的 setting。

```bash
sudo nvidia-smi -e 1   # 开
sudo nvidia-smi -e 0   # 关（需要重启）
```

## MIG vs MPS — 一卡多租户怎么选

| 维度 | MIG (Multi-Instance GPU) | MPS (Multi-Process Service) |
|---|---|---|
| 隔离 | 硬件级（独立 SM、L2、HBM 分区） | 软件复用上下文 |
| 适用卡 | A100 / H100 / B100/B200 | 所有 |
| 适合场景 | 多租户、SLA 严格、需独立 fail domain | 小模型并发、自家服务、追极致吞吐 |
| 显存隔离 | 是 | 否 |
| 切换开销 | 重配置需 stop 工作 | 即时 |
| 性能 | 子实例独立，无 cross talk | 共享，可能互相抢 SM |

### MIG 配置示例（H100 80GB）

```bash
# 7 个 1g.10gb 实例（最大切分）
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C
# Profile 19 = 1g.10gb；其他 profile 见 nvidia-smi mig -lgip

# 给某实例独占
export CUDA_VISIBLE_DEVICES=MIG-<uuid>
```

MIG 实例可看作独立 GPU，但**单实例 SM/HBM/NVLink 都是减半/分片的**，单 query 延迟会变差，适合 **多并发小模型**（如多个 ViT/CNN 服务共用一卡）。

### MPS 配置示例

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
sudo -E nvidia-cuda-mps-control -d
echo "start_server -uid $(id -u)" | nvidia-cuda-mps-control
# 然后多个进程共享同一 GPU，命令保持不变
```

H100+ 支持 **MPS percent limit**：
```bash
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50    # 每进程最多吃 50% SM
```

## Job 调度 / 拓扑感知

多 GPU 节点调度时让任务感知 NVLink/NVSwitch 拓扑：

```bash
# Slurm
srun --gpu-bind=closest --gres=gpu:8 ...

# K8s nvidia-device-plugin
# values.yaml: gfd 启用 topology label
nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3
nvidia.com/gpu.nvlink=true
```

NVSwitch 集群必须跑 **Fabric Manager**：
```bash
sudo systemctl enable nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```
没起 fabricmanager 的 NVSwitch 系统，多 GPU 通信会退化到 PCIe，**LLM TP/EP 立即慢一档**。

## Docker / Kubernetes

`--gpus all` 是最低要求，但生产环境还要：

```bash
docker run --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  --ipc=host \
  --cap-add=SYS_NICE \
  --cpuset-cpus=0-15 \
  --cpuset-mems=0 \
  ...
```

K8s 关键：
- `topologyManager` policy `single-numa-node`
- `cpuManager` policy `static`
- `--cpu-manager-reconcile-period=10s`
- pod 加 `nvidia.com/gpu` request + `requests.cpu == limits.cpu` 才能拿独占 CPU

## 故障排查清单

| 症状 | 检查 |
|---|---|
| 启动慢、首请求 200ms+ | 持久化模式没开 (`nvidia-smi -pm 1`) |
| p99 抖 | 时钟没锁、ECC 报错、温度超 80℃、邻居噪声 |
| nccl bandwidth 远低于 NVLink 标称 | Fabric Manager 没起 / 拓扑错位 |
| pinned 内存分配 OOM | `ulimit -l` 不是 unlimited |
| `RuntimeError: out of memory` 但 nvidia-smi 显示空闲 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 或 fragmentation |
| 多进程跑挂 | MPS 没起，或 `CUDA_MPS_*` 没传给容器 |
| GPU util 跳变 | CPU 喂不动数据，看 `nv-infer-io-data` |

## 一行启动模板

单卡推理服务推荐起手：

```bash
sudo nvidia-smi -pm 1 && \
sudo nvidia-smi -ac 1593,1980 && \
sudo nvidia-smi --auto-boost-default=DISABLED && \
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled && \
sudo swapoff -a && \
ulimit -l unlimited && ulimit -n 65535 && \
numactl --cpunodebind=0 --membind=0 \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_DEVICE_MAX_CONNECTIONS=32 \
  python serve.py
```

## 反模式

- 用 host 上 `nvidia-smi` 的版本搭 container 里更新的 CUDA 应用 → driver 不兼容
- ECC 关了忘记打回来上线
- 把 `MIG_PARTED` 配置文件忘了入 git → 节点换机后丢失分区
- MPS server 当前用户 != 应用用户 → 进程互访失败
- `--shm-size` 默认 64MB → DataLoader worker IPC 慢成幻灯片
