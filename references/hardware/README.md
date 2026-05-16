# Hardware Reference for `inferskills`

每颗你常用的推理加速卡的规格、互联、软件栈、推理调优要点。
所有 `nv-infer-*` 与未来 `ascend-infer-*` skill 都会引用本目录。

## 你的硬件清单

| 卡 | 角色 | 软件栈 | 详细文件 |
|---|---|---|---|
| **NVIDIA T4** | 旧代低功耗推理 / 视频分析 / 小模型 | CUDA + TensorRT | [`nvidia-t4.md`](nvidia-t4.md) |
| **NVIDIA H100** | 主力 LLM 推理 / 高性能 CNN/ViT | CUDA 12.4+ + TRT-LLM + vLLM | [`nvidia-h100.md`](nvidia-h100.md) |
| **NVIDIA B200** | 下一代 LLM / FP4 / MoE / 超长上下文 | CUDA 12.8+ + TRT-LLM 0.13+ + vLLM 0.6+ | [`nvidia-b200.md`](nvidia-b200.md) |
| **Huawei Atlas 300V Pro** (Ascend 310P3) | 视频推理 / 边缘 / 小 CNN-ViT | CANN + ATC + MindIE | [`huawei-atlas-300v-pro.md`](huawei-atlas-300v-pro.md) |
| **Huawei Atlas 910B** (Ascend 910B) | 大模型训练 / LLM 推理 (国产化) | CANN 7.0+ + MindIE-LLM + torch_npu | [`huawei-atlas-910b.md`](huawei-atlas-910b.md) |

工具对照（nvidia-smi ↔ npu-smi、nsys ↔ msprof、TensorRT ↔ ATC/MindIE 等）见 [`tooling-cheatsheet.md`](tooling-cheatsheet.md)。

## 高层规格对比

数字均按"主流单卡"标注；变体（PCIe vs SXM、910B1 vs 910B3 等）见各自详细文件。
TFLOPS 在 NVIDIA 侧标 **dense**（无稀疏加成），Tensor Core 单位。

| 维度 | T4 | H100 SXM5 80G | B200 SXM | Atlas 300V Pro | Atlas 910B |
|---|---|---|---|---|---|
| 架构 | Turing (sm_75) | Hopper (sm_90a) | Blackwell (sm_100) | Ascend 310P3 (Da Vinci) | Ascend 910B (Da Vinci Max) |
| 发布年 | 2018 | 2022 | 2024 | 2022 | 2023 |
| 工艺 | 12nm | 4N TSMC | 4NP TSMC, 2-die | 7nm+ | 7nm+ (估) |
| 显存 | 16 GB GDDR6 | 80 GB HBM3 | 192 GB HBM3e | 24 GB LPDDR4X | 64 GB HBM2e (主流) |
| 显存带宽 | 320 GB/s | 3.35 TB/s | ~8 TB/s | ~204 GB/s | ~1.6 TB/s |
| FP64 Tensor | — | 67 TFLOPS | 40 TFLOPS | — | — |
| TF32 Tensor | — | 989 TFLOPS | 2.2 PFLOPS | — | — |
| FP16/BF16 Tensor | 65 TFLOPS (FP16) | 989 TFLOPS | 2.25 PFLOPS | 70 TFLOPS | ~320 TFLOPS |
| FP8 Tensor | — | 1979 TFLOPS | 4.5 PFLOPS | — | — |
| FP4 / MXFP4 | — | — | 9 PFLOPS | — | — |
| INT8 Tensor | 130 TOPS | 1979 TOPS | 4.5 POPS | 140 TOPS | ~640 TOPS |
| INT4 | 260 TOPS | — (软件层) | — (软件层) | — | — |
| 卡间互联 | 无 | NVLink4 900 GB/s + NVSwitch3 | NVLink5 1.8 TB/s + NVSwitch4 (NVL72) | 无 | HCCS 392 GB/s |
| 主机互联 | PCIe Gen3 x16 | PCIe Gen5 x16 (128 GB/s) | PCIe Gen6 x16 (256 GB/s) | PCIe Gen4 x16 | PCIe Gen5 x16 |
| TDP | 70 W | 700 W | 1000 W | 72 W | ~400 W |
| 冷却 | 被动 | 液冷 / 风冷 | 液冷 (B200 SXM) | 被动 | 风冷/液冷 |
| Tensor Core 代 | 2nd | 4th | 5th | Da Vinci Cube | Da Vinci Cube |
| FP8 / FP4 | ✗ / ✗ | ✓ E4M3/E5M2 / ✗ | ✓ / ✓ MX-format | ✗ / ✗ | ✗ / ✗ |
| 2:4 稀疏 | ✗ | ✓ | ✓ | — | — |
| Transformer Engine | ✗ | ✓ 1st gen | ✓ 2nd gen + micro-scaling | 部分等价 (CANN) | 部分等价 (CANN) |
| NVENC/NVDEC | 1/2 (NVENC/NVDEC 7th) | 0/7 (无 NVENC) | 0/7 | 内置 H.264/H.265/AVS 编解码 | 部分 |
| MIG / 切分 | ✗ | ✓ 最多 7 | ✓ | vNPU (类似) | vNPU (类似) |
| 推理首选量化 | INT8 (TRT) | FP8 / INT4 (W4A16) | FP4 / FP8 | INT8 (ATC) | INT8 / W4A16 (MindIE-LLM) |

## 选卡决策树

按"模型类型 + 规模"快速判：

```
CNN / ViT 推理
├─ 边缘 / 视频流（每帧 < 10W）        → Atlas 300V Pro (264 路 1080p decode + AI)
├─ 老服务器 / 低成本                   → T4 (INT8 TRT)
├─ 主力数据中心                        → H100 (FP16/INT8/FP8)
├─ 国产化要求 + 训练同栈                → Atlas 910B (CANN 全栈)
└─ 下一代 / FP4 实验                   → B200

LLM 推理
├─ < 7B 模型，低成本                   → T4 (INT8/INT4) 或 Atlas 300V Pro (小模型 INT8)
├─ 7B-70B，性价比                      → H100 PCIe / NVL / A100
├─ 70B-405B，极致 throughput            → H100 SXM5 + TP=4/8 + FP8 / B200
├─ 405B+ / 长上下文 128K+ / MoE        → B200 / GB200 NVL72
├─ 国产化合规                          → Atlas 910B + MindIE-LLM
└─ 视频/多模态 LLM 解码端              → Atlas 300V Pro 解码 + Atlas 910B 推理

混合栈
├─ Atlas 300V Pro 做视频解码 + 前置 CNN  →  H100/910B 做主模型
├─ T4 节点接低 QPS，H100 接热点             →  按 SLO 分流
└─ NVIDIA + Huawei 共存                    →  服务层抽象（gRPC / OpenAI 兼容协议），后端各跑各的
```

## NVIDIA 与 Ascend 的"心智模型映射"

| NVIDIA 概念 | Ascend 等价 / 类似 |
|---|---|
| CUDA Core (vector) | Vector Unit (Da Vinci) |
| Tensor Core (matmul) | Cube Unit (16×16×16) |
| SM (Streaming Multiprocessor) | AI Core |
| Warp (32 threads) | 无完全等价（Da Vinci 是矩阵流水线，不是 SIMT） |
| Shared Memory / L1 | Unified Buffer (UB) + L1 Buffer |
| HBM | HBM2e (910B) / LPDDR4X (310P) |
| NVLink / NVSwitch | HCCS / HCCN |
| NCCL | HCCL |
| NVTX | msprofTx |
| nvidia-smi | npu-smi |
| Nsight Systems (nsys) | msprof / MindStudio Profiler |
| Nsight Compute (ncu) | msprof op-level / MindStudio |
| TensorRT | ATC + MindIE / OM (Offline Model) |
| TensorRT-LLM | MindIE-LLM |
| Triton Inference Server | MindIE Service / MindX SDK |
| CUDA Graphs | Graph Mode (TorchAir / GE) |
| cuBLAS / cuDNN | CANN OpAPI / aclnnOp |
| Triton (lang) / CUTLASS | Ascend C (类似 CUDA C++) |
| `__shfl_sync` 等 warp 原语 | 无（Da Vinci 没有 warp 概念） |
| FlashAttention | FlashAttentionScore (CANN 7.0+) / aclnnFlashAttention |
| MIG | vNPU partition |
| MPS | 多 Stream + 多 Context (aclrtSetCurrentContext) |
| FP8 (E4M3) | （910B 无原生 FP8，软件层模拟） |

更详细的命令级对照见 [`tooling-cheatsheet.md`](tooling-cheatsheet.md)。

## 各芯片相对优劣（推理视角，2025）

### T4 — 老兵，但仍能干活
- ✅ 70W 被动 + 16GB → 部署灵活，老机房改造首选
- ✅ INT8 + NVENC/NVDEC → 视频推理性价比仍好
- ✅ CUDA 全栈通用
- ❌ 无 BF16/FP8/FlashAttention 加速 → LLM 推理只能跑小模型 + INT8/INT4 (Marlin 在 Turing 兼容性需测)
- ❌ 无 NVLink → 多卡难扩
- ❌ Hopper 后的所有 LLM 优化论文几乎都假设 Ampere+

### H100 — 当下推理王者
- ✅ FP8 + FlashAttention v3 + TMA → LLM 推理生态最完善
- ✅ NVLink/NVSwitch → 单节点 8 卡 TP 满血
- ✅ Confidential Compute / MIG → 多租户支持好
- ✅ H100 NVL (188GB) / H200 (141GB) → 给 70B+ 留余地
- ❌ 700W → 散热/电力压力大
- ❌ 价格仍高（B200 出来后 H100 二级市场降）

### B200 — 未来 18 个月主力，但要新软件栈
- ✅ 192GB HBM3e → 单卡装 100B+ 模型不挤
- ✅ FP4 → LLM decode 吞吐再翻倍（精度可接受时）
- ✅ NVL72 (72 卡一柜) → 万亿参数推理一柜搞定
- ❌ 需要 CUDA 12.8+ / cuDNN 9.5+ / TRT-LLM 0.13+ → 老栈跑不起
- ❌ 1000W + 液冷必需 → 机房改造门槛
- ❌ 2-die 内部 NUMA → 有微妙的 perf cliff，调优手册仍在更新

### Atlas 300V Pro — 视频 AI 边缘王
- ✅ 72W 被动 + 264 路 1080p 解码 → 视频 AI 性价比无敌
- ✅ AIPP 链路 (decode → preproc → infer 全片上) → 极低 CPU 占用
- ✅ 国产化合规
- ❌ LPDDR 带宽 < 250 GB/s → 大模型 decode 吃力
- ❌ LLM 适配主要靠 MindIE-LLM，社区开源生态弱于 NV
- ❌ 自定义算子开发门槛（Ascend C）比 Triton lang 高

### Atlas 910B — 国产 AI 加速器顶配
- ✅ ~320 TFLOPS FP16 + 64GB HBM2e → 跑 7B-70B LLM 都可
- ✅ HCCS + HCCN → 多卡 TP / 多节点都支持
- ✅ MindIE-LLM 已具备 continuous batching / paged KV
- ✅ torch_npu → 大部分 HF PyTorch 模型 1-2 行 import 切换
- ❌ FP8 / FP4 缺失 → 极致 throughput 不如 H100/B200
- ❌ 算子覆盖：奇形怪状的 attention 变体 / 自定义层经常需要重写 Ascend C
- ❌ 性能高度依赖 shape，NZ format 数据布局是隐性陷阱
- ❌ 文档以中文为主，英文社区资源少
- ⚠️ SKU 变体多（910B / B1 / B2 / B3 / B4），实际单卡性能差异显著

## 推理硬件的常见误区

1. **以 TFLOPS / TOPS 选卡** —— 推理大部分时间 memory bound，HBM 带宽与显存容量更关键
2. **以"标称 INT8 TOPS"做横向对比** —— 实际利用率因 kernel / shape 差很多
3. **忽略互联** —— 单卡指标好不代表多卡 scaling 好（910B vs H100 在 8 卡时差距更明显或更小，因 HCCS vs NVLink 拓扑）
4. **忘了软件栈版本兼容** —— B200 需要 CUDA 12.8+，910B 需要 CANN 7+，老镜像跑不起新卡
5. **国产卡精度损失估算不足** —— 同样 INT8，不同硬件的量化 scheme/校准要求不同，CNN 不敏感 LLM 可能崩

## 使用本目录的方式

1. 写 prompt 给 Claude 时，提一句"我用 H100 PCIe" / "我有 Atlas 910B 8 卡"，相关 skill + 本文件会一起被加载
2. 跑 benchmark 前对照各文件的"实测命令"部分，建立基线
3. 调优遇到瓶颈时，对照各文件的"调优要点 / 常见坑"
4. 后续若引入 Atlas 800I A2 / 200 / 200I 等新卡，按本目录格式新增

> 备注：Huawei 部分规格官方披露有限，标"估"的为社区/合作伙伴公开测试数据，以实际机器跑出的为准。
