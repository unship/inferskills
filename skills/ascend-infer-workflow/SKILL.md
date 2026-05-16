---
name: ascend-infer-workflow
description: Huawei Ascend NPU (Atlas 300V Pro / 300I Pro / 910B 等) 上 CNN / ViT / LLM 推理性能优化的总入口与流程。当用户提到 Ascend / 昇腾 / 910B / 310P / Atlas 300V Pro / Atlas 800 / CANN / MindIE / npu-smi / msprof / ATC，或在 NPU 上调优 CNN/ViT/LLM 推理性能时使用。给出阶段化决策树并分发到 ascend-infer-* 子 skill。
---

# Ascend NPU Inference Tuning Workflow

> 与 NVIDIA 的 nv-infer-workflow 平行，针对昇腾全栈（CANN / MindIE / torch_npu / Ascend C）。
> 心智模型映射与命令对照见 `references/hardware/tooling-cheatsheet.md`。
> 硬件参数：300V Pro 见 [`references/hardware/huawei-atlas-300v-pro.md`](../../references/hardware/huawei-atlas-300v-pro.md)，910B 见 [`references/hardware/huawei-atlas-910b.md`](../../references/hardware/huawei-atlas-910b.md)。

## 何时用本 skill

- 用户的目标硬件是 Ascend 310P / 910B 系列
- 出现"昇腾/Ascend/CANN/MindIE/npu-smi/ATC/msprof"等关键词
- 调优、迁移、对标 NVIDIA 方案在 Ascend 上的落地

## 阶段化流程

```
Phase 0  Baseline    ── 基线吞吐 / 时延 / AI Core 利用率 / HBM 带宽 / 功耗
Phase 1  System      ── CANN/driver/firmware 版本、npu-smi 设置、vNPU、HCCN  → ascend-infer-system-setup
Phase 2  Profile     ── msprof / MindStudio 时间线 + AI Core metrics       → ascend-infer-profiling
Phase 3  Compile     ── ATC / TorchAir / MindIE 编译                        → ascend-infer-compile-stack
Phase 4  Precision   ── FP16/BF16/INT8/W4A16（无 FP8/FP4）                  → ascend-infer-precision
Phase 5  LLM stack   ── MindIE-LLM / paged KV / FlashAttentionScore         → ascend-infer-llm-stack
Phase 6  Serving     ── MindIE Service / 持续批处理 / 调度                    → ascend-infer-serving
Phase 7  Memory      ── NZ format / UB / HBM allocator                     → ascend-infer-memory
Phase 8  Kernels     ── Ascend C 自定义算子（仅必要时）                       → ascend-infer-kernels
Phase 9  Multi-NPU   ── HCCL / HCCS / HCCN / TP-PP-EP                       → ascend-infer-multinpu
Phase 10 IO/Data     ── DVPP / AIPP / DataLoader                            → ascend-infer-io-data
Phase 11 Model       ── CNN/ViT 专项                                        → ascend-infer-cnn-vit
```

Phase 1-4 通常占 ROI 的 80%。Phase 8 (Ascend C) 是最后手段。

## 第一步：建立基线

```bash
# 1. 卡况
npu-smi info
npu-smi info -t board -i 0
npu-smi info -t topo

# 2. 监控
npu-smi info watch -i 0 -d 1     # 类似 nvidia-smi dmon
# 关注: AI Core util% / HBM util% / 频率 / 功耗 / 温度

# 3. 版本对齐
cat /usr/local/Ascend/driver/version.info
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
python -c "import torch_npu; print(torch_npu.__version__)"
```

把基线数字（throughput、p50/p95/p99、AI Core 利用率、HBM 利用率）写到 `bench/baseline_ascend.md`。

## 瓶颈对照表

| 现象 | 最可能瓶颈 | 跳到 |
|---|---|---|
| AI Core util < 60%，CPU 100% | 数据 pipeline / DVPP 没用上 | `ascend-infer-io-data` |
| AI Core util 高，HBM util > 80% | memory bound | `ascend-infer-memory` + 量化 |
| AI Core util 高，HBM util < 30% | compute bound | `ascend-infer-precision`（INT8/W4A16） |
| LLM decode 卡 prefill 快 | KV / attention | `ascend-infer-llm-stack` |
| TP=8 慢于 1 卡 ×4 | HCCS 没起 / 拓扑错 | `ascend-infer-multinpu` |
| 大量 AICPU 算子 fallback | 通用 path | `ascend-infer-compile-stack`（升 CANN） |
| op 报缺失编译失败 | 算子不在 CANN OpAPI 里 | `ascend-infer-kernels`（写 Ascend C） |
| 模型转换 OM 后精度大跳 | 量化校准 | `ascend-infer-precision` |

## CNN / ViT / LLM 三类在 Ascend 上的重点

**CNN 推理（YOLO、ResNet、EfficientNet）**
- DVPP 视频解码 + AIPP 预处理全片上化 —— 不要 CPU 跑 OpenCV
- ATC INT8 calibration → `.om`，与 NV 的 TRT 套路一致
- 300V Pro 视频通道多，910B 算力高
- NZ format 自动处理（ATC/GE 内部），但自写 op 要意识到

**ViT / CLIP / SigLIP**
- attention 走 `aclnnFlashAttentionScore`（CANN 7.0+）
- patch embed stride 卷积 ATC 自动选合适 algo
- 输入分辨率桶化（动态 shape 用 `--dynamic_dims` 枚举）
- ViT-L/H 在 910B 上吞吐良好；300V Pro 上 ViT-L 受 LPDDR 带宽限制

**LLM (Qwen / LLaMA / MiniCPM 等)**
- 走 MindIE-LLM（推荐）：内建 paged KV、continuous batching、INT8/W4A16 量化
- 单流走 torch_npu + transformers 仅作 debug
- 7-13B 单卡，70B 用 TP=4 或 TP=8 跨 HCCS
- 没有 FP8 / FP4 硬件，量化最高到 INT8 W8A8 / W4A16
- 长上下文：KV INT8 + paged + chunked prefill
- 详见 `ascend-infer-llm-stack`

## 一次 Ascend sprint 模板

```
Day 1  跑基线 + msprof + 瓶颈对照表
Day 2  CANN/driver/firmware 升到最新稳定 + npu-smi 配置 + HCCN init
Day 3  ATC/MindIE 编译 + precision_mode=allow_mix_precision
Day 4  INT8 calibration（CNN/ViT）或 W4A16/W8A8（LLM）
Day 5  KV INT8 + paged + continuous batching
Day 6  TP/PP 调优 + HCCL 环境变量
Day 7  msprof 复测，写报告，加 CI bench 防回归
```

## 反模式

- 试图找 FP8 path —— 910B/310P 硬件不支持
- 把 NV 的 AWQ 量化文件直接 load 到 MindIE —— quantization scheme 不同，重做
- 让 CPU 跑 OpenCV 预处理 —— Ascend 卡的 DVPP+AIPP 是核心卖点之一
- 期待 PyTorch eager + torch_npu 达到 MindIE 性能 —— eager 仅 debug，生产走 OM/MindIE
- 不验证 `npu-smi info -t topo` 就上 TP —— HCCS 没建立时直接走 PCIe
- 把 driver 升了但 CANN/firmware/torch_npu 没跟上 —— 必须按 release matrix 整体对齐

## 与 NV skill 的对照（迁移项目时用）

| nv-infer-* | ascend-infer-* | 主要差异 |
|---|---|---|
| nv-infer-workflow | **本 skill** | 流程相似，工具栈不同 |
| nv-infer-profiling | ascend-infer-profiling | nsys/ncu → msprof/MindStudio |
| nv-infer-system-setup | ascend-infer-system-setup | nvidia-smi → npu-smi；MIG → vNPU |
| nv-infer-memory | ascend-infer-memory | channels_last → NZ；HBM allocator |
| nv-infer-kernels | ascend-infer-kernels | CUDA C → Ascend C |
| nv-infer-compile-stack | ascend-infer-compile-stack | torch.compile/TRT → torch_npu graph/ATC/MindIE |
| nv-infer-precision | ascend-infer-precision | 无 FP8/FP4 |
| nv-infer-llm-attention | ascend-infer-llm-stack | FlashAttn → FlashAttentionScore；MindIE-LLM 整合 |
| nv-infer-serving | ascend-infer-serving | Triton/vLLM → MindIE Service |
| nv-infer-multigpu | ascend-infer-multinpu | NCCL → HCCL；NVLink → HCCS |
| nv-infer-cnn-vit | ascend-infer-cnn-vit | DVPP+AIPP 重头戏 |
| nv-infer-io-data | ascend-infer-io-data | DVPP 替代 DALI |

## 进一步参考

- `references/hardware/tooling-cheatsheet.md` —— 命令级 NV↔Ascend 对照
- `references/hardware/huawei-atlas-910b.md` —— 910B SKU 与拓扑细节
- `references/hardware/huawei-atlas-300v-pro.md` —— 300V Pro 与 DVPP/AIPP
- Huawei Ascend 文档中心、hiascend.com 开发者社区
