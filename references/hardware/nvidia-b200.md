# NVIDIA B200 (Blackwell)

> 2024 末-2025 量产，下一代 LLM/MoE 推理主力。卖点三件套：**192 GB HBM3e + FP4 Tensor Core + NVLink5 / NVL72 全互联**。
> 软件栈要新：CUDA 12.8+、TRT-LLM 0.13+、vLLM 0.6+。老镜像 100% 跑不起。

## SKU 一览

| SKU | 形态 | 显存 | BW | NVLink | TDP | 备注 |
|---|---|---|---|---|---|---|
| **B200 SXM** | OAM/SXM6 | 192 GB HBM3e | ~8 TB/s | 5th gen, 1.8 TB/s 双向 | 1000 W | 主力推理/训练卡 |
| **B100 SXM** | OAM | 192 GB HBM3e | ~8 TB/s | 5th gen, 1.8 TB/s | 700 W | 风冷版，频率/性能 ~80% B200 |
| **GB200 Superchip** | 1× Grace ARM + 2× B200 (via NVLink-C2C) | 2 × 192 GB HBM3e + 480 GB LPDDR5X (Grace) | 8 TB/s (HBM) + 900 GB/s (C2C) | 5th gen | 1200 W (per Superchip) | 单 socket 上跑大模型/长 context KV offload |
| **GB200 NVL72** | 1 rack = 36 GB200 = 72 B200 + 36 Grace | 13.8 TB HBM3e (72 × 192 GB) | aggregate ~576 TB/s HBM | NVSwitch4 全互联 130 TB/s 双向聚合 | ~120 kW/rack | 单柜跑 trillion-参数 LLM |
| **B200 NVL** (PCIe 形态，预计 2025) | PCIe + NVLink bridge | TBD | TBD | 桥接 | TBD | 期待，规格未完全公开 |

## 核心规格（B200 SXM，公开数据）

| 项 | 值 |
|---|---|
| 架构 | Blackwell GB100，**双 die 通过 NVLink-C2C 互联** |
| Compute Capability | sm_100 (Blackwell architecture，cuda 12.8+ 支持) |
| 显存 | **192 GB HBM3e** (8 stacks × 24 GB) |
| 显存带宽 | **~8 TB/s** (HBM3e @ 8+ Gbps × 8192-bit) |
| **FP64 Tensor** | 40 TFLOPS |
| **TF32 Tensor** | 2.2 PFLOPS (dense) |
| **BF16/FP16 Tensor** | **2.25 PFLOPS (dense), 4.5 PFLOPS (sparse)** |
| **FP8 Tensor** | **4.5 PFLOPS (dense), 9 PFLOPS (sparse)** |
| **FP6 Tensor** | 4.5 PFLOPS (dense) |
| **FP4 / MXFP4 Tensor** | **9 PFLOPS (dense), 18 PFLOPS (sparse)** |
| INT8 Tensor | 4.5 POPS (dense) |
| PCIe | Gen6 x16 (256 GB/s 双向，需主板支持) |
| NVLink | **5th gen**, 18 link/卡, **1.8 TB/s 双向聚合** |
| NVSwitch | v4，支持 NVL72 域内全互联 |
| Decompression Engine | 800 GB/s（专为 LLM weight/KV 解压） |
| Confidential Compute | v2 |
| RAS engine | 硬件级故障检测 / 修复 |
| TDP | **1000 W (B200 SXM)** / 700 W (B100) |
| 冷却 | **液冷必需 (B200 SXM)** / 风冷 (B100, 频率降) |

NVIDIA 在 GTC 上披露的两个 die（B200 内部）通过 10 TB/s C2C 缝合，OS 看到一颗 GPU。GB100 die 单个上有更多 SM，且时钟更高，总 SM 数比 H100 翻倍量级。

## Blackwell 新特性

| 特性 | 用途 |
|---|---|
| **2nd-gen Transformer Engine** | 自动管理 FP4 / FP6 / FP8 / MXFP scaling，**block-wise scaling** (微张量) |
| **MXFP8 / MXFP6 / MXFP4 (OCP micro-scaling)** | 每 32 元素一个 scale，比 per-tensor FP8 精度更稳 |
| **NVFP4** | NVIDIA 私有 FP4 格式，**2× block size** 适配 |
| **FP4 Tensor Core** | 大幅提升 LLM decode 吞吐（已量化模型） |
| **第二代 NVLink Switch / NVLink Network** | NVL72 域内全互联，跨节点 IB 之上的"新一层" |
| **Decompression Engine** | 800 GB/s 解压 weight/KV，与 PCIe Gen6 配合，**提升从 SSD/host 流式加载速度** |
| **RAS Engine** | 长跑可靠性，预测性维护信号 |
| **Confidential Compute v2** | 性能损失更小 |

## 软件栈（**2025 实际要求**）

| 组件 | 最低 | 推荐 |
|---|---|---|
| Driver | R555 | **R570+** |
| CUDA | 12.8 | **12.9+** |
| cuDNN | 9.5 | **9.7+** |
| cuBLAS | 12.8 | 12.9+ |
| TensorRT | 10.4 | **10.6+** |
| TensorRT-LLM | 0.13 | **0.14+** (FP4 完整) |
| PyTorch | 2.5 | **2.6+** (含 SM_100 wheel) |
| Transformer Engine | 1.10 | **1.13+** |
| FlashAttention | v3.0 | **v3.1+** |
| vLLM | 0.6.3 | **0.7+** (FP4 + MX) |
| Triton (lang) | 3.1 | **3.2+** |
| NCCL | 2.22 | **2.24+** (NVL72 拓扑感知) |

容器：`nvcr.io/nvidia/pytorch:25.01-py3` 起为 B200 ready。

## 适合 workload

| workload | 适合度 |
|---|---|
| 405B / 671B (DeepSeek-V3) LLM 推理 | ✅✅✅ NVL72 一柜搞定 |
| MoE 模型（DeepSeek-V3、Mixtral、Qwen-MoE）EP+TP | ✅✅✅ |
| 长上下文 128K-1M | ✅✅✅ 192GB HBM 给 KV |
| FP4 推理（已 INT4/FP4 量化模型） | ✅✅✅ 独家硬件加速 |
| 70B 单卡 FP8/INT4 | ✅✅ 大材小用，看价格 |
| 训练 405B+ | ✅✅✅ |
| 7B 小 LLM | ⚠️ 大材小用 |
| CNN/ViT 推理 | ⚠️ 大材小用，除非超大 batch 离线 |
| 边缘 / 低功耗 | ❌ |

## 推理配置模板

### B200 单卡，70B / 405B（W4A16 + FP4 KV）

```bash
# vLLM 0.7+
vllm serve meta-llama/Llama-3.1-405B-Instruct \
    --quantization fp4 \
    --kv-cache-dtype fp4_e2m1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enable-prefix-caching --enable-chunked-prefill
```

### GB200 NVL72，超大 MoE (DeepSeek-V3 671B)

```bash
# TensorRT-LLM with EP + TP
trtllm-build --checkpoint_dir dsv3_ckpt/ \
    --output_dir dsv3_engine/ \
    --quantization fp4 \
    --use_fp8_context_fmha enable \
    --paged_kv_cache enable \
    --kv_cache_type=fp8 \
    --moe_tp_size 8 --moe_ep_size 8 \
    --tp_size 8 --pp_size 1 \
    --max_batch_size 256 --max_input_len 8192 --max_output_len 4096

# 72 卡部署，跨 NVSwitch4
export NCCL_NVLS_ENABLE=1     # 新拓扑
mpirun -n 72 ... 
```

### 长 context 128K (B200)

```bash
vllm serve some-128k-model \
    --max-model-len 131072 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp4_e2m1 \
    --gpu-memory-utilization 0.92
# 192GB HBM 让长 context KV 充裕
```

## FP4 实战要点

- **NVFP4** (NVIDIA 私有) vs **MXFP4** (OCP 标准)：TRT-LLM 0.13+ 都支持
- 一般 NVFP4 精度略好，MXFP4 跨厂商通用
- 模型必须 **重新 calibration / quantize**（FP4 不能从 FP8/FP16 直接 cast）
- 工具：
  - NVIDIA AMMO/ModelOpt (`nvidia-modelopt`)
  - llmcompressor (vLLM 生态)
  - TensorRT-LLM 自带 quantize 脚本
- KV cache 可独立选 FP4/FP8/INT8
- 精度回归：MMLU/HumanEval 上掉幅通常 < 1.5%，对 reasoning（GSM8K/MATH）可能更敏感

## NVL72 拓扑要点

- 72 卡通过 18 个 NVSwitch4 全互联，**任意两卡 1.8 TB/s P2P**
- 跨柜走 IB/Spectrum-X，BW 阶跃下降 → 算法上**尽量把 collective 锁在柜内**
- 调度：NIXL / Dynamo 在柜内 NVLink 上做 prefill→decode KV 传输，跨柜降级
- 一个 rack ≈ 13.8 TB HBM3e，单 rack 跑 trillion-参数 LLM (FP8)

## 调优速记

1. **必开**：
   - `nvidia-smi -pm 1`
   - Fabric Manager (NVL72 必须)
   - `NCCL_NVLS_ENABLE=1`（NVLink Sharp，NCCL 2.24+）
   - `CUDA_DEVICE_MAX_CONNECTIONS=32`
2. **B200 内部 2-die NUMA 感知**：
   - 极端 latency 场景，注意单 kernel 跨 die 的 C2C 开销（~10% 带宽折损）
   - cuDNN/cuBLAS 已自动处理；自定义 kernel 用 CUTLASS 3.5+
3. **量化优先级**：FP4 > FP8 > INT4 (W4A16) > FP16
4. **KV cache**：默认 FP8，长 context + FP4 模型可以 FP4 KV
5. **不要**：拿老 PyTorch / TRT 在 B200 上跑 → fallback 到 BF16 path，FP4 全用不上

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 老镜像（24.x） | sm_100 wheel 缺失，编译失败 | 升 25.x 镜像 |
| FP4 calibration 集太小 | 精度大跳 | ≥ 1024 样本，分布贴近线上 |
| GB200 NVL72 跑 IB collective | 跨柜带宽不对齐预期 | 算法切分让 collective 锁柜内 |
| `flash-attn` 老版本 | 编译报 sm_100 unsupported | 装 v3.1+ |
| 跨 die kernel 频繁同步 | 性能掉 | 使用 CUTLASS 3.5+，自动 cluster-aware |
| 风冷机房塞 B200 | 过热降频 | 必须液冷，或买 B100 |
| 双 die 显存"假装"是 192GB 一片 | 大 tensor 分配跨 die NUMA | PyTorch 2.6+ 加 die-aware allocator hint |
| MIG 切分 | 早期 driver 仅部分 profile 可用 | 看 release notes |

## 实测命令

```bash
# 验证 FP4 path
python -c "
import torch
print(torch.cuda.get_device_capability())  # 应 (10,0)
print(torch.cuda.is_fp8_supported(), 
      hasattr(torch, 'float4_e2m1'))         # FP4 dtype
"

# NVL72 NCCL 带宽
./all_reduce_perf -b 1M -e 4G -f 2 -g 72
# 期望 1G msg ≥ 4 TB/s aggregate

# HBM3e 带宽
./bandwidthTest --memory=pinned --mode=range \
    --start=2147483648 --end=17179869184 --increment=2147483648
# 期望 ≥ 6.5 TB/s
```

## DCGM 指标新增

Blackwell 上 dcgmi 加了：
- FP4 / FP6 / MXFP 类型 Tensor Active%
- Decompression Engine 利用率
- 跨 die NVLink-C2C 流量

```bash
dcgmi dmon -e 1002,1004,1005,1009,...   # 具体 field id 看 R570+ 版本 dcgmi --field-list
```

## 部署建议

- 单柜电力 ≥ 120 kW，液冷必备
- 电源/网络/冷却基建升级先行（NVL72 不是普通机柜）
- 容器：用 NGC 镜像，**不要**自己拼 CUDA 12.8 base
- 24/7 服务必须开 RAS 监控、ECC 必开

## 参考

- NVIDIA Blackwell Architecture Whitepaper
- GB200 NVL72 Reference Design
- TensorRT-LLM Blackwell tuning notes
- 本仓库 [`../../skills/nv-infer-precision/SKILL.md`](../../skills/nv-infer-precision/SKILL.md) — FP8/FP4 量化
- 本仓库 [`../../skills/nv-infer-multigpu/SKILL.md`](../../skills/nv-infer-multigpu/SKILL.md) — NVL72 拓扑
