---
name: ascend-infer-system-setup
description: Huawei Ascend NPU 系统层调优——driver / firmware / CANN 版本对齐、npu-smi 配置、电源/频率模式、vNPU 切分、HCCN 网络（RoCE/IB）、Ascend Docker Runtime、K8s device plugin、Atlas 800I/T A2 服务器配置。当用户问"昇腾配置"、"npu-smi"、"CANN 版本"、"vNPU"、"HCCN"、"Atlas 800"、"Ascend Docker"、"ARM 鲲鹏" 时使用。
---

# Ascend System Setup

> 与 nv-infer-system-setup 平行。Ascend 的"系统层"比 NV 更需要严守版本矩阵 —— driver / firmware / CANN / torch_npu / MindIE 必须整体一致。

## 启动前一次性检查

```bash
# 1. 卡基本信息 & 版本
npu-smi info
npu-smi info -t board -i 0
npu-smi info -t product -i 0          # 确认 910B / 910B3 / 310P3 等具体型号
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
cat /usr/local/Ascend/firmware/version.info 2>/dev/null

# 2. python 侧
python -c "import torch_npu; print(torch_npu.__version__); print(torch_npu.npu.is_available())"

# 3. 拓扑（多卡必查）
npu-smi info -t topo                  # HCCS 拓扑

# 4. NUMA / CPU
numactl -H
lscpu | grep -i numa

# 5. 网络（HCCN，多节点必查）
hccn_tool -i 0 -ip -g                 # 看每个 NIC 的 IP
hccn_tool -i 0 -link -g               # 看 link 状态

# 6. 文件描述符 / 锁页
ulimit -l
ulimit -n
```

## 版本矩阵（关键）

CANN / driver / firmware / torch_npu / MindIE 必须按官方 release matrix 整体对齐，**不能任选**。常见生产组合：

| 阶段 | driver | CANN | torch_npu | MindIE | 适用 |
|---|---|---|---|---|---|
| 老生产 | 23.0.RC* | 6.3.RC | 2.0.* | 1.0.RC | 910A/910B 早期 |
| 当前主流 | 24.1.RC* | **7.0.0 / 7.0.RC1** | 2.1.0.post* | 1.0.0 | 910B + LLM |
| 推荐 (2025) | 24.1+ | **8.0.RC* / 8.0.0** | 2.1.0+ / 2.3 | 2.0+ | 全栈最新 |

升级顺序：firmware → driver → CANN toolkit → torch_npu → MindIE。**反顺序会让旧 driver 装不下新 firmware 的 binary。**

```bash
# 升级 CANN toolkit
chmod +x Ascend-cann-toolkit_*_linux-aarch64.run
./Ascend-cann-toolkit_*_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

容器化（推荐）：
```bash
docker pull ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.RC2-arm64
```
让镜像与 host 的 driver / firmware 解耦（driver 在 host，CANN/toolkit 在 image）。

## npu-smi 关键设置

### 持久化 / 电源模式

```bash
# 设高性能模式（频率不降）
npu-smi set -t power-mode -i 0 -c 0 -p 1    # 1=高性能, 0=平衡

# 查询
npu-smi info -t power -i 0 -c 0
npu-smi info -t freq -i 0 -c 0
```

### 时钟锁定

某些 SKU 支持手动锁频：
```bash
npu-smi set -t aicore-freq -i 0 -c 0 -f <MHz>     # 视固件支持
```

### ECC

910B 默认开 ECC。查询：
```bash
npu-smi info -t ecc -i 0 -c 0
```

不要在生产关 ECC；离线 benchmark 可在固件层关，但要重启。

### 温度 / 功耗墙

```bash
npu-smi info -t temp -i 0 -c 0
npu-smi info -t power -i 0 -c 0
# 长期 > 80℃ → 散热问题，会降频
```

## vNPU（虚拟切分，类比 MIG）

把一颗 910B 切成多个独立 NPU 实例，多服务隔离。

```bash
# 1. 开启 vNPU 模式
npu-smi set -t vnpu -i 0 -c 0 -v 1

# 2. 查看可用 profile
npu-smi info -t vnpu -i 0 -c 0

# 3. 创建切分实例
npu-smi set -t vnpu -i 0 -c 0 -v <profile>

# 4. 容器/进程通过 ASCEND_RT_VISIBLE_DEVICES 选实例
export ASCEND_RT_VISIBLE_DEVICES=0
```

300V Pro 也支持 vNPU（最多 4 切分）；910B 最多 8 切分（具体看 SKU）。

**注意**：切了 vNPU 后单实例算力 = 总算力 / N，单实例 HBM = 总 HBM / N（且不可跨实例共享）。适合多模型/多租户；单大模型场景不要切。

## HCCN 网络（多节点）

跨节点 NPU 通信走 HCCN（Huawei Compute Communication Network），底层 RoCE/IB。

```bash
# 配置 HCCN IP（Atlas 800 节点通常有 8 个 NIC 对应 8 卡）
hccn_tool -i 0 -ip -s address 192.168.10.10 netmask 255.255.255.0
hccn_tool -i 1 -ip -s address 192.168.10.11 netmask 255.255.255.0
...

# 验证链路
hccn_tool -i 0 -link -g
hccn_tool -i 0 -net_health -g

# RoCE 性能配置（PFC/ECN）
hccn_tool -i 0 -tls -s enable 0    # TLS off（性能优先）
```

Atlas 800T A2 / 800I A2 整机出厂已配好 HCCN；自组机需手动配。

## Docker / 容器

最小可用启动参数：

```bash
docker run -it --rm \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  ... (every davinciN you need)
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  --ipc=host \
  ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.RC2-arm64
```

或者用 **Ascend Docker Runtime**（推荐）：

```bash
docker run -it --rm --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3 \
  ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.RC2-arm64
```

## K8s

Huawei `ascend-device-plugin` 暴露资源：

```yaml
resources:
  limits:
    huawei.com/Ascend910: 4         # 或 huawei.com/Ascend310P
```

拓扑感知调度：用 `ascend-scheduler-plugin`，在 pod 调度时根据 HCCS topology 把同一 job 的多卡尽量分配到同一 NPU group（NV 上的 NVSwitch domain 对等概念）。

## 环境变量（推理常用）

```bash
# 设备可见
export ASCEND_VISIBLE_DEVICES=0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3   # runtime 层（更新）

# CANN 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 日志级别
export ASCEND_GLOBAL_LOG_LEVEL=3            # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# 内存分配器（torch_npu）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# HCCL（多卡）
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_BUFFSIZE=512
export HCCL_OVER_OFI=0                       # 单节点
```

## ARM (鲲鹏 920) vs x86

Atlas 800T A2 / 800I A2 主流是**鲲鹏 920 ARM CPU + Ascend NPU**。注意：

- 镜像要 ARM64 版本（aarch64）
- PyPI wheel 多数 ARM 上要自编译；用 NGC 风格的 ascendhub 镜像最稳
- numactl / cgroup / NUMA 调优同 x86

x86 + Ascend 也支持，但 Huawei 主推 ARM 配套。

## 故障排查清单

| 症状 | 检查 |
|---|---|
| `import torch_npu` 报版本不匹配 | torch_npu / CANN / firmware / driver release matrix 对齐 |
| `npu-smi info` 卡死 | driver service 异常 → `sudo systemctl restart` ascend driver |
| 进程 OOM 但 `npu-smi` 显示 free | 显存碎片，加 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| TP 通信慢 | `npu-smi info -t topo` 验证 HCCS，环境变量 HCCL_* 检查 |
| ATC 编译报某 op 缺失 | 升 CANN / 看支持算子列表 / 写 Ascend C |
| 容器内 npu-smi 不可用 | mount /dev/davinci_manager / /dev/devmm_svm / /usr/local/Ascend/driver |
| K8s pod 拉起但无 NPU | device-plugin 没 ready / pod 没声明 resource |
| 编码 ASCII 乱码（中文日志） | locale → `LANG=en_US.UTF-8` |

## 一行启动模板（推理服务）

```bash
npu-smi set -t power-mode -i 0 -c 0 -p 1 && \
ulimit -n 65535 && \
source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
export HCCL_CONNECT_TIMEOUT=1200 HCCL_EXEC_TIMEOUT=1200 \
       HCCL_BUFFSIZE=512 HCCL_OVER_OFI=0 \
       ASCEND_GLOBAL_LOG_LEVEL=3 \
       PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
       ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && \
numactl --cpunodebind=0 --membind=0 \
  python serve.py
```

## 反模式

- driver 升了但忘了升 firmware → 启动报错且日志在 `/var/log/ascend_seclog/`
- 把 CANN tool 装在 root 但服务跑非 root，source set_env 失败
- 多个不同 CANN 版本共存 → 切换不彻底，"latest" symlink 没动
- 镜像架构混（x86 镜像跑在鲲鹏 ARM 上）
- 多容器抢同 vNPU 实例 → 加 `ASCEND_RT_VISIBLE_DEVICES` 隔离
- 不配 HCCN IP 直接跑 TP → fall back PCIe，慢 5-10×

## 参考

- `references/hardware/huawei-atlas-910b.md`
- `references/hardware/huawei-atlas-300v-pro.md`
- `references/hardware/tooling-cheatsheet.md` — npu-smi 完整命令对照
- Huawei Ascend 文档中心 — CANN 安装与升级、Atlas 800 服务器手册
