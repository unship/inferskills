# inferskills — Claude Code skills for NVIDIA GPU & Huawei Ascend NPU inference tuning

一组面向 **NVIDIA GPU (T4 / H100 / B200 / ...) 与华为 Ascend NPU (Atlas 300V Pro / 910B)** 上 **CNN / ViT / LLM / VLM 推理性能优化** 的 Claude Code skills。
覆盖从 OS/驱动到 kernel、再到 serving 全链路 —— 每个 skill 是可执行命令 + 决策树 + 反模式清单。

灵感来源：Chris Fregly《AI Systems Performance Engineering》(O'Reilly, 2025) Appendix 的 175+ item performance checklist。本仓库把其中与 **推理** 相关的部分重新组织、扩充为可被 Claude Code 自动触发的 skill，加入了大量 NVIDIA 工具链（nsys、ncu、TensorRT-LLM、vLLM、Triton、DCGM…）与华为 Ascend 工具链（npu-smi、msprof、ATC、MindIE、HCCL…）的实操命令与版本细节。

## Skill 列表

### NVIDIA 栈（`skills/nv-infer-*`）

| Skill | 主题 | 关键触发词 |
|---|---|---|
| [`nv-infer-workflow`](skills/nv-infer-workflow/SKILL.md) | 总入口、阶段化决策树、瓶颈对照表 | 提升推理性能 / 从哪下手 / checklist |
| [`nv-infer-profiling`](skills/nv-infer-profiling/SKILL.md) | Nsight Systems/Compute、NVTX、PyTorch profiler、tail latency | 找瓶颈 / GPU 利用率低 / p99 抖 |
| [`nv-infer-system-setup`](skills/nv-infer-system-setup/SKILL.md) | OS / 驱动 / NUMA / MIG / MPS / clocks / Docker / K8s | 机器配置 / NUMA / 持久化 / 降频 |
| [`nv-infer-memory`](skills/nv-infer-memory/SKILL.md) | HBM 层级、pinned mem、coalescing、channels_last、KV 预算、allocator | memory bound / OOM but free / 碎片化 |
| [`nv-infer-kernels`](skills/nv-infer-kernels/SKILL.md) | CUDA kernel fusion、streams、Graphs、occupancy、Hopper TMA/WGMMA | 自定义 kernel / 融合 / divergence |
| [`nv-infer-compile-stack`](skills/nv-infer-compile-stack/SKILL.md) | torch.compile、TensorRT、TensorRT-LLM、ONNX Runtime、Triton (lang) | 图优化 / 编译 / autotune / dynamic shape |
| [`nv-infer-precision`](skills/nv-infer-precision/SKILL.md) | FP16/BF16/FP8/INT8/INT4、AWQ/GPTQ/SmoothQuant、2:4 sparsity、KV 量化 | 量化 / Tensor Core / 精度损失 |
| [`nv-infer-llm-attention`](skills/nv-infer-llm-attention/SKILL.md) | FlashAttention、PagedAttention、KV cache、prefill/decode、speculative | LLM 推理 / KV / FA / 长上下文 |
| [`nv-infer-serving`](skills/nv-infer-serving/SKILL.md) | Triton / vLLM / TRT-LLM、continuous batching、autoscaling、prefix cache | 部署 / Triton / vLLM / p99 / SLO |
| [`nv-infer-multigpu`](skills/nv-infer-multigpu/SKILL.md) | TP/PP/EP/CP、NCCL、NVLink/NVSwitch、IB、GPUDirect RDMA、NIXL | 多卡 / 多节点 / scaling 不行 |
| [`nv-infer-cnn-vit`](skills/nv-infer-cnn-vit/SKILL.md) | CNN/ViT 专项：NHWC、cuDNN、conv 融合、Winograd、ViT/Swin/YOLO | CNN/ViT 推理 / 卷积 / patch embed |
| [`nv-infer-io-data`](skills/nv-infer-io-data/SKILL.md) | DataLoader / DALI / nvJPEG / GDS / tokenizer / 模型加载 / S3 | 数据 pipeline / DALI / 模型加载慢 |

### Ascend 栈（`skills/ascend-infer-*`）

| Skill | 主题 | 关键触发词 |
|---|---|---|
| [`ascend-infer-workflow`](skills/ascend-infer-workflow/SKILL.md) | NPU 总入口、流程、瓶颈对照表 | 昇腾 / CANN / 910B / 310P |
| [`ascend-infer-system-setup`](skills/ascend-infer-system-setup/SKILL.md) | CANN/driver/firmware 矩阵、npu-smi、vNPU、HCCN、Atlas 800 | npu-smi / Atlas 800 / vNPU / HCCN |
| [`ascend-infer-profiling`](skills/ascend-infer-profiling/SKILL.md) | msprof、MindStudio、Pipe Utilization、AICPU fallback 检测 | msprof / MindStudio / AICPU fallback |
| [`ascend-infer-memory`](skills/ascend-infer-memory/SKILL.md) | NZ format、UB/L1、torch_npu allocator、pinned host mem | NZ format / UB / NPU allocator |
| [`ascend-infer-kernels`](skills/ascend-infer-kernels/SKILL.md) | Ascend C、TBE、TIK、Cube/Vector pipeline、tiling | Ascend C / 自定义算子 / TBE |
| [`ascend-infer-compile-stack`](skills/ascend-infer-compile-stack/SKILL.md) | ATC、OM、TorchAir、GE、MindIE、AOE auto_tune | ATC / OM / TorchAir / 编译 |
| [`ascend-infer-precision`](skills/ascend-infer-precision/SKILL.md) | FP16/BF16/INT8 W8A8/W4A16/KV INT8（无 FP8/FP4） | Ascend 量化 / msmodelslim / AMCT |
| [`ascend-infer-llm-stack`](skills/ascend-infer-llm-stack/SKILL.md) | MindIE-LLM、atb_llm、paged KV、FlashAttentionScore、speculative | MindIE-LLM / NPU LLM / paged KV |
| [`ascend-infer-serving`](skills/ascend-infer-serving/SKILL.md) | MindIE Service、MindX SDK、K8s + Ascend device plugin、SLO | MindIE Service / 昇腾部署 |
| [`ascend-infer-multinpu`](skills/ascend-infer-multinpu/SKILL.md) | HCCL、HCCS、HCCN、TP/PP/EP、ranktable、Atlas 800 集群 | HCCL / HCCS / HCCN / NPU TP |
| [`ascend-infer-cnn-vit`](skills/ascend-infer-cnn-vit/SKILL.md) | DVPP + AIPP 片上 video pipeline、YOLO/CLIP/ViT 部署 | DVPP / AIPP / 昇腾 YOLO |
| [`ascend-infer-io-data`](skills/ascend-infer-io-data/SKILL.md) | DVPP/AIPP、torch_npu DataLoader、MindData、OBS 读取 | NPU DataLoader / MindData |

## 硬件参考 (`references/hardware/`)

每颗常用卡有独立 spec + 调优要点文件，给"具体卡能做/不能做什么"的硬约束：

| 文件 | 卡 | 角色 |
|---|---|---|
| [`nvidia-t4.md`](references/hardware/nvidia-t4.md) | NVIDIA T4 | 旧代低功耗推理 / 视频分析 |
| [`nvidia-h100.md`](references/hardware/nvidia-h100.md) | NVIDIA H100 / H200 / H800 NVL | 主力 LLM/CV |
| [`nvidia-b200.md`](references/hardware/nvidia-b200.md) | NVIDIA B100 / B200 / GB200 / NVL72 | 下一代 LLM/MoE/FP4 |
| [`huawei-atlas-300v-pro.md`](references/hardware/huawei-atlas-300v-pro.md) | Atlas 300V Pro (Ascend 310P3) | 视频 AI / 边缘 |
| [`huawei-atlas-910b.md`](references/hardware/huawei-atlas-910b.md) | Atlas 910B / B1-B4 (Ascend 910B) | 国产化训练 / LLM 推理 |
| [`tooling-cheatsheet.md`](references/hardware/tooling-cheatsheet.md) | NVIDIA ↔ Ascend 命令/概念对照 | 混合栈日常 |

入口与对比表见 [`references/hardware/README.md`](references/hardware/README.md)。

## 模型参考 (`references/models/`)

常用模型族的架构 / 显存预算 / 推荐量化 / 各卡部署模板 / 已知坑：

| 文件 | 模型 | 类型 |
|---|---|---|
| [`yolo-series.md`](references/models/yolo-series.md) | YOLOv8 / v9 / v10 / v11 全系 | CNN 检测 |
| [`clip-siglip.md`](references/models/clip-siglip.md) | CLIP / OpenCLIP / EVA-CLIP / SigLIP / SigLIP2 | ViT + 文本 encoder |
| [`minicpm-v.md`](references/models/minicpm-v.md) | MiniCPM-V 2.0 / 2.5 / 2.6 / 4.0 / MiniCPM-o-2.6 | 多模态 LLM (VLM) |
| [`qwen3-8b.md`](references/models/qwen3-8b.md) | Qwen3-8B (并兼容 Qwen2.5-7B / 后续 Qwen3.x) | Decoder-only LLM |

入口与选模型 × 选硬件矩阵见 [`references/models/README.md`](references/models/README.md)。

## 安装

### 方式 1：作为 user-global skills

```bash
git clone https://github.com/unship/inferskills /tmp/inferskills
mkdir -p ~/.claude/skills
cp -r /tmp/inferskills/skills/* ~/.claude/skills/
```

Claude Code 启动时会扫到这些 skill，按描述自动触发。

### 方式 2：作为 project-level skills

进入你要优化的项目根：

```bash
mkdir -p .claude/skills
cp -r /path/to/inferskills/skills/* .claude/skills/
git add .claude/skills && git commit -m "add inferskills"
```

### 方式 3：symlink（便于持续更新）

```bash
ln -s "$(pwd)/skills" ~/.claude/skills/inferskills
# 之后 git pull 即可拿到 skill 更新
```

> 提示：`references/` 不需要拷贝到 `~/.claude/skills/`；skill 内会用相对路径引用，建议保留原仓库目录并 symlink，或把整个仓库 clone 到工作树。

## 使用示例

让 Claude Code 干活时直接描述场景，相关 skill 会自动触发：

| 场景 | 自动触发 |
|---|---|
| "我的 LLaMA-3-70B 推理慢，p99 200ms" | `nv-infer-workflow` + `nv-infer-llm-attention` |
| "Qwen3-8B 在 H100 上怎么部署最快" | `nv-infer-llm-attention` + `references/models/qwen3-8b.md` |
| "ViT-L 上 INT8 没快多少为什么" | `nv-infer-precision` + `nv-infer-cnn-vit` |
| "8 卡 H100 跑 70B TP，吞吐才 4× 单卡" | `nv-infer-multigpu` |
| "GPU 利用率 40%，CPU 100%" | `nv-infer-io-data` + `nv-infer-profiling` |
| "MiniCPM-V 2.6 怎么在 24GB 卡上跑" | `references/models/minicpm-v.md` + `nv-infer-precision` |
| "YOLOv11 部署到 T4 / Atlas 300V Pro" | `references/models/yolo-series.md` + `nv-infer-cnn-vit` / `ascend-infer-cnn-vit` |
| "910B 上跑 Qwen3-8B" | `ascend-infer-llm-stack` + `references/models/qwen3-8b.md` |
| "MindIE Service 怎么配置" | `ascend-infer-serving` + `ascend-infer-llm-stack` |
| "8 卡 910B HCCL 性能不对" | `ascend-infer-multinpu` |

也可以显式让 Claude 用某条：

> "用 ascend-infer-llm-stack 帮我把 Qwen3-8B 部署到 910B"

## 推荐工作顺序

### NVIDIA

```
Phase 0  Baseline
Phase 1  nv-infer-system-setup   (OS/驱动/NUMA/clocks)
Phase 2  nv-infer-profiling      (定位瓶颈)
Phase 3  nv-infer-compile-stack  (torch.compile / TRT / TRT-LLM)
Phase 4  nv-infer-precision      (FP16/BF16 → FP8/INT8/INT4)
Phase 5  nv-infer-llm-attention  (LLM)  或  nv-infer-cnn-vit (CNN/ViT)
Phase 6  nv-infer-serving        (batching / SLO)
Phase 7  nv-infer-memory         (layout / allocator)
Phase 8  nv-infer-kernels        (自定义 kernel — 仅当必要)
Phase 9  nv-infer-multigpu       (多卡)
Phase 10 nv-infer-io-data        (pipeline)
```

### Ascend

```
Phase 0  Baseline
Phase 1  ascend-infer-system-setup  (CANN 版本对齐 / npu-smi / HCCN)
Phase 2  ascend-infer-profiling     (msprof / MindStudio)
Phase 3  ascend-infer-compile-stack (ATC / TorchAir / MindIE 编译)
Phase 4  ascend-infer-precision     (FP16/BF16 → W8A8/W4A16，无 FP8/FP4)
Phase 5  ascend-infer-llm-stack     (MindIE-LLM)  或  ascend-infer-cnn-vit
Phase 6  ascend-infer-serving       (MindIE Service / MindX)
Phase 7  ascend-infer-memory        (NZ format / allocator)
Phase 8  ascend-infer-kernels       (Ascend C — 仅必要时)
Phase 9  ascend-infer-multinpu      (HCCL / HCCS / HCCN)
Phase 10 ascend-infer-io-data       (DVPP / AIPP / DataLoader)
```

总入口 `*-infer-workflow` 都有完整决策树。

## 适用硬件

NVIDIA：
- ✅ Ampere（A100、A30、A10、L40S、RTX 30/40）
- ✅ Hopper（H100、H200、H800、GH200）
- ✅ Blackwell（B100、B200、GB200、NVL72）
- ⚠️ Turing/Volta（V100、T4）：能用但 FP8 / FlashAttention v3 不支持

Huawei Ascend：
- ✅ Atlas 300V Pro / 300I Pro / 300I Duo (Ascend 310P3)
- ✅ Atlas 910B / B1 / B2 / B3 / B4 (Ascend 910B)
- ❌ FP8 / FP4 路径 Ascend 硬件不支持，不要找

## 适用框架与版本

NVIDIA：
- PyTorch 2.3+（建议）/ TensorRT 10+ / TensorRT-LLM 0.10+
- vLLM 0.6+ / SGLang / Triton Inference Server 24.x+
- CUDA 12.4+（FP8 全栈）/ 12.8+（Blackwell）
- cuDNN 9+

Ascend：
- CANN 7.0+（推荐 8.0+）
- torch_npu 2.1+
- MindIE 1.0+（推荐 2.0+）
- 镜像：`ascendhub.huawei.com/public-ascendhub/ascend-pytorch:*-arm64`

## 贡献

欢迎 PR：
- 添加新优化模式、工具命令
- 补充模型族（DiT、Mamba、Hyena、扩散模型、视频生成）专项
- 修正版本相关变化
- 增加新硬件参考（A100、L40S、L4、Atlas 800I A2 等）

每个 skill 保持单文件 `SKILL.md`，frontmatter 中的 `description` 需含足够触发词（中英文都加）。

## License

代码与文档：MIT（见 `LICENSE`）。

致谢：技术内容受 Chris Fregly《AI Systems Performance Engineering》(O'Reilly, 2025) 的 175+ item checklist 启发，所有 skill 内容为本仓库原创撰写，结合 NVIDIA 官方文档、Huawei Ascend 文档、社区实践、生产部署经验。
