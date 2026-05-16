# Huawei Atlas 910B (Ascend 910B)

> 华为目前的训练/推理顶配，2023 量产，2024-2025 大规模铺货。国产化合规 + 7B-405B LLM 推理可行。
> 注意：**910B 是一个 SKU 家族**（910B / B1 / B2 / B3 / B4），实际算力差异显著，先认清自己的版本。

## 一句话画像

```
Ascend 910B (Da Vinci Max) · 64 GB HBM2e · ~1.6 TB/s · ~400W
FP16: ~320 TFLOPS · INT8: ~640 TOPS · BF16: ~320 TFLOPS · 无 FP8/FP4 硬件
HCCS 392 GB/s + HCCN (RoCE) 跨节点 · 单卡推理 7B-13B 顺畅，TP=8 跑 70B
```

## SKU 家族（务必对齐）

| 版本 | 主要差异 | 显存 | FP16 算力 (估) | TDP |
|---|---|---|---|---|
| Atlas 910A (910) | 初代，2019 | 32 GB HBM2 | 256-320 TFLOPS | ~310 W |
| **Atlas 910B** | 2023 量产主力 | 64 GB HBM2e | ~280-320 TFLOPS | ~400 W |
| **Atlas 910B1 / B2** | 算力档 | 64 GB | ~280-320 TFLOPS | ~400 W |
| **Atlas 910B3** | 中高档（部分公开测试见过 ~313 TFLOPS） | 64 GB | ~313 TFLOPS | ~400 W |
| **Atlas 910B4** | 高档 / 优化版 | 64 GB | 接近 380 TFLOPS 不等 | ~400 W |
| **Atlas 910C** (传闻 / 国产化下一代) | 2025+ | TBD | TBD | TBD |

> 上述数字主要来自合作伙伴公开 benchmark 与媒体披露，**华为官方没有公开完整 spec sheet**，以你机器实际跑 `npu-smi info -t product` 与 micro-bench 为准。

确认你手上是哪一档：
```bash
npu-smi info -t product -i 0
npu-smi info -t board -i 0
# 看 Model / Type 字段，例如 "Atlas 910B3"
```

## 核心规格（910B 主流档）

| 项 | 值 |
|---|---|
| SoC | Huawei Ascend 910B (Da Vinci Max) |
| 架构 | Da Vinci（含 Cube/Vector/Scalar/MTE 子单元） |
| AI Core 数 | 25 (910B3 公开数据；其他档可能 24/30) |
| 显存 | **64 GB HBM2e** |
| 显存带宽 | **~1.6 TB/s** |
| **FP16 (Cube)** | **~320 TFLOPS** (B3) |
| **BF16** | **~320 TFLOPS** （910B 起支持，910A 无 BF16） |
| FP32 (Vector) | ~80 TFLOPS（粗略，主要走 Cube FP16） |
| **INT8** | **~640 TOPS** |
| INT4 | 软件层支持（MindIE-LLM 的 W4A16），无独立 Tensor Core |
| **FP8 / FP4** | **不支持**（硬件层无） |
| 卡间互联 | **HCCS** (Huawei Cache Coherent System) 7 链路 × 56 GB/s = **392 GB/s 双向聚合** |
| 节点内互联拓扑 | 8 卡全互联（类比 NVSwitch），全速 P2P |
| 跨节点 | **HCCN** (Huawei Compute Communication Network)，200 Gbps RoCE/IB |
| Host PCIe | Gen4 x16（部分 Gen5） |
| 加速 IP | Cube (matmul) / Vector / Scalar / MTE (内存搬运) / AICPU |
| 切分 | vNPU 支持（最多 8 切分） |
| Confidential | 部分（与国密合规集成） |
| TDP | ~400 W |
| 形态 | 模组（NPU module）+ Atlas 800T A2 / 800I A2 服务器整机 |
| 冷却 | 风冷 (800T/800I) / 液冷 |

## 软件栈（Ascend AI 全栈）

| 组件 | 角色 | 类比 NVIDIA |
|---|---|---|
| **CANN** 7.0+ / 8.0 | 全栈基础 | CUDA + cuDNN + cuBLAS |
| **AscendCL (acl)** | Runtime API | CUDA Runtime |
| **ATC** | 静态图编译器 | TensorRT builder |
| **GE** (Graph Engine) | 图优化与执行 | XLA / TRT runtime |
| **TBE / Ascend C** | 自定义算子开发 | CUDA C / CUTLASS / Triton |
| **MindIE-LLM** | LLM 推理引擎（含 paged KV、continuous batching） | TensorRT-LLM |
| **MindIE-Service** | 推理服务化 | Triton Inference Server |
| **MindIE-Torch / TorchAir** | PyTorch 图模式接入 | torch.compile + TRT EP |
| **torch_npu** | PyTorch eager mode on NPU | PyTorch CUDA backend |
| **HCCL** | 集合通信 | NCCL |
| **MindSpore** | 华为深度学习框架 | PyTorch (训练侧) |
| **MindCluster / MindX DL** | 集群管理 | NeMo / NVIDIA AI Enterprise |
| **MindStudio** | IDE + Profiler | Nsight Systems/Compute |
| **msprof** | CLI profiler | nsys/ncu |
| **npu-smi** | 卡管理 | nvidia-smi |

### CANN 版本与生态对齐

| CANN | torch_npu | MindIE | 备注 |
|---|---|---|---|
| 6.3.RC | torch_npu 2.0 | MindIE 1.0 | 老生产基线 |
| 7.0.RC1 / 7.0.0 | 2.1.0.post* | MindIE 1.0.RC* | 主流生产 (2024) |
| **8.0.RC1+** | **2.1.0.post* / 2.3** | **MindIE 1.0/2.0** | **LLM 推理推荐** (paged KV、FlashAttn 等齐) |

升 CANN 时 driver/firmware 必须同步升级，**不能混搭**。

## 适合 workload

| workload | 适合度 |
|---|---|
| LLM 7B-13B 单卡推理 | ✅✅✅ |
| LLM 70B 8 卡 TP 推理 | ✅✅ (MindIE-LLM 已支持 paged KV + continuous batch) |
| LLM 405B+ 多节点 | ✅ (需 HCCN + 大集群调优) |
| MoE (Mixtral / 国产 MoE) | ✅ (EP 通过 MindIE/MindSpore) |
| CNN/ViT 大 batch 推理 | ✅✅✅ |
| BERT/T5 类 | ✅✅✅ |
| 文生图 (SD / DiT) | ✅ (MindIE-SD 适配) |
| Whisper / TTS | ✅ |
| 训练（LLM/ViT） | ✅✅✅ (这是 910B 的设计目标) |
| **FP8 / FP4 极致量化** | **❌** 硬件不支持 |
| 视频解码 + AI（端到端） | ⚠️ 910B 视频 IP 弱于 310P；视频前端配 Atlas 300V Pro 更优 |
| 边缘 / 低功耗 | ❌ 400 W |

## 推理实用配置

### 单卡 910B，LLaMA-3-8B (MindIE-LLM)

```bash
# 模型转换（HF → MindIE 兼容格式，通常用 atb_llm.py / mindie-cli）
mindie-cli convert --model_path llama3_8b/ \
    --output_path llama3_8b_om/ \
    --soc_version Ascend910B3 \
    --precision_mode fp16 \
    --max_batch_size 32 --max_seq_len 8192

# 启动 MindIE Service
mindieservice_daemon --config service_config.json
# config 含: ports / max_batch / paged_kv / continuous_batching 等
```

### 8 卡 910B TP 跑 70B

```bash
# torch_npu + DeepSpeed-NPU / MindIE 内置 TP
# 关键 HCCL env
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_BUFFSIZE=512
export HCCL_OVER_OFI=0          # 单节点不要走 OFI
export ASCEND_GLOBAL_LOG_LEVEL=3

mindieservice_daemon \
    --tp 8 --pp 1 \
    --model llama3_70b_om/ \
    --paged_kv_cache true \
    --max_batch_size 64 --max_seq_len 8192
```

### PyTorch 推理（torch_npu）

```python
import torch
import torch_npu                              # 只需 import 就 patch 进 PyTorch
import torch_npu.contrib.transfer_to_npu      # 自动 cuda → npu

model = AutoModel.from_pretrained("llama3_8b").half().npu().eval()
x = tokenizer("hi", return_tensors='pt').input_ids.npu()
with torch.no_grad():
    out = model.generate(x, max_new_tokens=128)
```

主要 op 已 ONNX 兼容；少数特殊 op 需要：
- 升 CANN 8.0+（覆盖更广）
- 或手写 Ascend C kernel + 注册到 torch_npu

## 调优速记

1. **必开**：
   - `npu-smi set -t power-mode -i 0 -c 0 -p 1` 高性能模式
   - HCCL `HCCL_BUFFSIZE=512`、`HCCL_CONNECT_TIMEOUT=1200`
   - MindIE-LLM 默认开 paged KV + continuous batching
2. **NZ format**：
   - Ascend 内部 weights 存为 **NZ (Native-Z)** 格式（4D 分块 5D 实存），不是 NCHW
   - 从 PyTorch checkpoint 转 OM 时 ATC/TorchAir 自动处理
   - 自定义 op 必须按 NZ 写
3. **量化优先级**：
   - FP16 / BF16（基线）
   - INT8 W8A8（CNN/ViT 必试）
   - W4A16 (AWQ/GPTQ)（LLM 大模型推荐）
   - **没有 FP8** —— 不要找 FP8 路径
4. **shape 友好**：
   - 维度对齐 16 / 32（Cube 单元自然 tile）
   - vocab/hidden 不齐 → 内部 padding，浪费算力
   - 动态 shape：用 `dynamic_dims` 提前枚举几个桶
5. **HCCL 调优**：
   - 单节点 8 卡走 HCCS，**绝不要回退 PCIe**
   - `npu-smi info -t topo` 验证拓扑
   - 跨节点 RoCE 配 PFC/ECN

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| CANN/torch_npu/driver/firmware 版本不齐 | import 报版本不匹配 | 严格按 release matrix 对齐 |
| 模型转换 OM 报某 op 缺失 | atc 编译失败 | 升 CANN / 写 Ascend C / 替换 op (HF 模型偶尔需 patch) |
| `attention_mask` 形状奇怪 | FlashAttentionScore 退化慢 path | 用 MindIE 的 GQA-friendly attention API |
| 推理速度远低于标称 | shape 不齐 + 通用 path | check msprof，确认 Cube 利用率 > 50% |
| 8 卡 TP 慢 | HCCS 未启用或拓扑错位 | `npu-smi info -t topo`；检查 HCCN init log |
| OOM 但 npu-smi 显示 free | 显存碎片 | aclrt allocator 调参 / 重启进程 |
| 训练同代码 fork 推理慢 | 没切 inference 模式 | `model.eval()` + 关 autograd |
| BF16 mask scale 不精确 | 长 seq 上 NaN | FA Score 内部已处理；自写 attention 注意 |
| W4A16 AWQ 转 Ascend | 量化 scheme 与 NV 略异 | 用 MindIE 自带 quantizer 重做，不要直接搬 NV 的 awq.pt |
| 中文文档 ≫ 英文文档 | 海外团队上手慢 | 翻译 / 关注 huawei-ascend GitHub |
| 镜像 ARM/x86 区分 | 拉错跑不起 | 确认 `uname -m`，鲲鹏节点用 arm64 镜像 |
| Confidential Compute 与性能 | 部分功能开启后性能掉 | 按需启用 |

## 实测命令

```bash
# 卡况
npu-smi info
npu-smi info -t board -i 0
npu-smi info -t common -i 0
npu-smi info -t topo                 # HCCS 拓扑
npu-smi info watch -i 0              # dmon

# 温度 / 功耗 / 频率
npu-smi info -t temp -i 0 -c 0
npu-smi info -t power -i 0 -c 0
npu-smi info -t freq -i 0 -c 0

# vNPU 切分
npu-smi set -t vnpu -i 0 -c 0 -v 1   # 启用 vNPU
npu-smi info -t vnpu -i 0 -c 0

# HCCL 单节点 8 卡带宽
# 用 CANN 自带的 hccl_test
# 路径通常在 /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test
./bin/all_reduce_test -b 8 -e 1G -f 2 -d fp16 -p 8
# 期望：单节点 8 卡 HCCS，1GB msg ≥ 200 GB/s aggregate

# msprof
msprof --output=./prof --aic-metrics=PipeUtilization,ArithmeticUtilization,MemoryAccess \
       --application="python infer.py" --task-time=on --aicpu=on
```

## 监控指标（生产）

`npu-smi` 暴露的关键指标（可通过 SNMP / Prometheus exporter 拉到 Grafana）：

| 指标 | 类比 NVIDIA |
|---|---|
| AI Core utilization (%) | SM Active |
| HBM utilization (%) | DRAM Active |
| HBM Used (GB) | memory.used |
| Aicore Frequency (MHz) | clocks.gr |
| Power (W) | power.draw |
| Temperature (℃) | temperature.gpu |
| HCCS link status | NVLink link state |
| ECC errors | ECC counters |

## 部署建议

- 训练/推理服务器：**Atlas 800T A2** (训练) / **Atlas 800I A2** (推理)，8 卡一节点
- K8s：华为 `Ascend Device Plugin`，`huawei.com/Ascend910` resource
- 镜像：`ascendhub.huawei.com/public-ascendhub/ascend-mindspore` 或 `ascend-pytorch`
- ARM 节点（鲲鹏 920）+ Ascend：原生组合，性能最稳
- 与 NVIDIA 共存：建议服务层（OpenAI 兼容 HTTP API）抽象，业务无感切换

## 与本仓库 skill 的对应

| nv-infer-* skill | 在 910B 上的迁移 |
|---|---|
| `nv-infer-workflow` | 流程通用，**Phase 4 没有 FP8** 这一项 |
| `nv-infer-profiling` | nsys/ncu → msprof / MindStudio |
| `nv-infer-system-setup` | nvidia-smi → npu-smi；MIG → vNPU |
| `nv-infer-memory` | channels_last → NZ format；pinned mem 仍然概念适用 |
| `nv-infer-kernels` | CUDA → Ascend C；fusion 通过 ATC/GE 自动；WGMMA/TMA 无 |
| `nv-infer-compile-stack` | torch.compile + TRT-LLM → torch_npu graph + MindIE-LLM |
| `nv-infer-precision` | **FP16/BF16/INT8/W4A16**，**无 FP8/FP4** |
| `nv-infer-llm-attention` | FlashAttention → FlashAttentionScore (CANN 7.0+)；PagedAttention → MindIE 内部 paged KV |
| `nv-infer-serving` | Triton/vLLM/TRT-LLM → MindIE Service |
| `nv-infer-multigpu` | NCCL → HCCL；NVLink → HCCS；IB/RoCE → HCCN |
| `nv-infer-cnn-vit` | channels_last → NZ；cuDNN → CANN OpAPI；多数概念可类比 |
| `nv-infer-io-data` | DALI → DVPP (910B 视频 IP 弱)；主要数据 pipeline 仍由 host CPU + 多线程驱动 |

详细命令对照见 [`tooling-cheatsheet.md`](tooling-cheatsheet.md)。

## 参考

- Huawei Ascend 文档中心（CANN 文档套）
- Atlas 800T A2 / 800I A2 服务器手册
- MindIE-LLM 使用指南（推荐：CANN 8.0+ 配套版本）
- Huawei Ascend GitHub（torch_npu、mindspore、modelzoo）
- 国内合作伙伴公开 benchmark（如 InfiniGAI、信通院 AIBench 等）
