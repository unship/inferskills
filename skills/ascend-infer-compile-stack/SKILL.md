---
name: ascend-infer-compile-stack
description: Ascend NPU 推理的编译栈——ATC (Ascend Tensor Compiler)、OM (Offline Model)、TorchAir / torch_npu graph mode、GE (Graph Engine)、MindIE 编译、auto_tune、dynamic_dims、precision_mode 选择、AOE (Ascend Optimization Engine)。当用户问"ATC"、"OM"、"昇腾编译"、"TorchAir"、"GE"、"auto tune"、"动态 shape"、"op 不支持" 时使用。
---

# Ascend Compile Stack — ATC / TorchAir / MindIE

> 与 nv-infer-compile-stack 平行。Ascend 上**生产推理必须走编译路径**（eager 仅 debug）。
> 优先级：MindIE（LLM）> ATC OM（CNN/ViT）> TorchAir graph mode > eager torch_npu。

## 决策树

```
模型类型 + 用途
├─ LLM 服务化                              → MindIE-LLM (见 ascend-infer-llm-stack)
├─ CNN / ViT 静态 shape 生产               → ATC → OM → AscendCL/MindIE Service
├─ PyTorch 模型快速迁移、动态控制流多       → torch_npu + TorchAir graph mode
├─ 一次性 benchmark / 实验                  → torch_npu eager
└─ MindSpore 训练好的模型                  → 原生 MindSpore + GE 编译
```

## ATC（Ascend Tensor Compiler）

把 ONNX / Caffe / TensorFlow / MindSpore IR → `.om`（Offline Model），运行时由 AscendCL 加载执行。**类比 NV 的 TensorRT builder + `.plan`**。

### 基本编译

```bash
atc --model=model.onnx \
    --framework=5 \
    --soc_version=Ascend910B3 \
    --output=model_om \
    --input_format=NCHW \
    --input_shape="input:1,3,224,224" \
    --precision_mode=allow_mix_precision \
    --output_type=FP16
```

`--framework`：0=Caffe, 1=MindSpore, 3=TensorFlow, 5=ONNX
`--soc_version`：必须严格匹配你的卡（`npu-smi info -t product -i 0` 查），常见值：
- `Ascend310P3` — Atlas 300V Pro / 300I Pro
- `Ascend910B3` — Atlas 910B3
- `Ascend910B4`、`Ascend910A` 等

### precision_mode 选项

| 值 | 含义 | 何时用 |
|---|---|---|
| `force_fp16` | 全模型 FP16 | 默认快路径 |
| `allow_fp32_to_fp16` | 自动转 FP16 | 兼容 ONNX FP32 |
| `allow_mix_precision` | **混合精度（INT8/FP16）自动选** | **推荐**，性能精度均衡 |
| `must_keep_origin_dtype` | 严格按 ONNX 原 dtype | 调试 / 精度回归 |
| `allow_fp32_to_bf16` | BF16 | 910B 数值更稳 |

### Dynamic Shape

固定 shape 性能最好；动态 shape 用两种方式之一：

```bash
# 1) 离散枚举（dynamic_dims）
atc --model=m.onnx --framework=5 --soc_version=Ascend910B3 \
    --input_shape="input:-1,3,-1,-1" \
    --dynamic_dims="1,224,224;1,448,448;8,224,224;32,224,224" \
    --output=m_dyn_om
# 运行时 acl 接口选 shape index

# 2) 连续区间（dynamic_image_size / dynamic_batch_size）
atc --model=m.onnx --framework=5 --soc_version=Ascend910B3 \
    --input_shape="input:-1,3,224,224" \
    --dynamic_batch_size="1,2,4,8,16,32" \
    --output=m_dynbs_om
```

变长 LLM 输入：用 MindIE-LLM 内部 paged + remove_input_padding，不要自己用 ATC 撸。

### AIPP（preprocessing 嵌入）

`--insert_op_conf=aipp.cfg` 把图像 resize/CSC/normalize 合并进 OM，运行时 input 直接传 YUV420。**核心卖点**之一，CPU 几乎零占用。配置详见 `ascend-infer-cnn-vit`。

### AOE / auto_tune（强烈推荐）

AOE (Ascend Optimization Engine) 自动调优 op 实现，类比 TRT 的 builderOptimizationLevel：

```bash
# 在 ATC 阶段开
atc --model=m.onnx --framework=5 --soc_version=Ascend910B3 \
    --output=m_om \
    --auto_tune_mode="RL,GA"
# RL=Reinforcement Learning, GA=Genetic Algorithm
```

或独立用 AOE：
```bash
aoe --model=m.onnx --framework=5 --job_type=2 \
    --output=m_tuned_om --soc_version=Ascend910B3
```

构建慢（几分钟到几小时），运行时性能多数情况下 +20-50%。

### 调试 / 检查

```bash
# 看模型结构（在 MindStudio 里更清楚）
atc --mode=1 --om=m_om.om --json=m.json    # 转 JSON 可读
# 或
msame --model=m_om.om --dump=true            # 跑一次并 dump op 输出
```

## TorchAir（PyTorch graph mode on Ascend）

PyTorch eager 在 Ascend 上有 op launch / Python overhead，**TorchAir** 把 PyTorch graph 通过 GE 编译，类比 `torch.compile`：

```python
import torch
import torch_npu
import torchair                              # CANN 8.0+

config = torchair.CompilerConfig()
config.experimental_config.frozen_parameter = True
config.experimental_config.tiling_schedule_optimize = True

npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=False)

with torch.inference_mode():
    out = model(x.npu())
```

性能上：TorchAir 较 eager 通常 1.5-3×，与 ATC OM 接近（少数场景 OM 略快，因为更彻底的离线优化）。

### 关键 flag

| 字段 | 作用 |
|---|---|
| `dynamic=False` | 固定 shape，触发 fully static optimization |
| `frozen_parameter=True` | 推理时权重不动，更激进 fusion |
| `tiling_schedule_optimize` | 自动 tile 调优 |
| `experimental_config.cache_compile_result` | 编译结果落盘，下次启动免重编 |

### Graph break 排查

```python
import torch._dynamo
torch._dynamo.config.verbose = True
# 出现 graph break 的位置打印
```

与 NV torch.compile 一样：`.item()` / 数据依赖控制流 / 未注册 op 会断图。

## MindIE 编译（生产推理服务）

MindIE 是 Huawei 的"推理引擎全家桶"，子模块：
- **MindIE-Inference**（通用模型）
- **MindIE-LLM**（LLM 专用，含 paged KV / continuous batching）
- **MindIE-SD**（Stable Diffusion / 扩散模型）
- **MindIE-Service**（HTTP/gRPC 服务化层）

通用模型 → MindIE 编译流程（实质内部仍是 ATC + GE）：

```bash
mindie-cli convert --model_path some_model/ \
    --output_path some_model_om/ \
    --soc_version Ascend910B3 \
    --precision_mode allow_mix_precision \
    --auto_tune true \
    --max_batch_size 32 \
    --max_seq_len 2048
```

LLM 看 `ascend-infer-llm-stack`。

## op 不支持 / fallback 处理

ATC/GE 在编译时发现 op 不在 AI Core OpAPI 里：
1. **AICPU**：性能差但能跑（msprof 看 aicpu 占比）
2. **完全不支持**：编译失败

修法（按推荐顺序）：

1. **升 CANN** —— 每个 release 都补几十个 op。CANN 8.0+ 覆盖了 Qwen2/3、LLaMA-3、SDXL、SAM 等主流模型大部分算子。
2. **模型层替代** —— 等价改写。如：
   - `argsort` → `topk`
   - 复杂 ROI Align → 简单 crop+resize
   - `aten::scatter` 写法转 `index_put`
3. **写 Ascend C 自定义算子** —— 见 `ascend-infer-kernels`
4. **保留 ONNX 端但在框架层 hook** —— torch_npu 注册 fallback

## CNN / ViT 编译样板

```bash
# YOLOv8m INT8 + AIPP
atc --model=yolov8m.onnx --framework=5 --soc_version=Ascend310P3 \
    --output=yolov8m_int8_om \
    --input_format=NCHW \
    --input_shape="images:16,3,640,640" \
    --insert_op_conf=aipp_yuv2rgb_letterbox.cfg \
    --precision_mode=allow_mix_precision \
    --auto_tune_mode="RL,GA" \
    --buffer_optimize=l2_buffer_optimize
```

```bash
# CLIP ViT-L/14 BF16
atc --model=clip_vit_l14_visual.onnx --framework=5 --soc_version=Ascend910B3 \
    --output=clip_vit_l14_om \
    --input_shape="images:32,3,224,224" \
    --precision_mode=allow_fp32_to_bf16
```

## 缓存 / Build 复用

```bash
# 启用编译缓存，下次相同 graph 跳过重新编译
export ASCEND_CACHE_PATH=/var/cache/ascend
export ASCEND_CACHE_FILE_ENABLE=1
```

TorchAir 也支持 `cache_compile_result=True`（CANN 8.0+），重启服务跳过几分钟编译时间。

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| `soc_version` 写错 | 编译成功但运行时 crash | `npu-smi info -t product` 查后严格匹配 |
| ONNX opset 太新 | ATC 不识别某 op | opset 17 较稳；export 时显式指定 |
| dynamic shape 桶太多 | OM 文件膨胀、首次编译极慢 | 桶数 ≤ 16 |
| `--precision_mode=force_fp16` 模型精度跳水 | 关键层数值溢出 | 用 allow_mix_precision，敏感层 keep FP32 |
| AOE 跑超时 | 大模型 + 多 tile 组合爆炸 | 限制 `--auto_tune_mode=GA` 单一策略 |
| TorchAir 编译完跑出 NaN | `frozen_parameter=True` 与 weight 切换冲突 | 推理结束前别改权重 |
| OM 文件无法跨 SoC 复用 | 910B3 OM 在 910B4 上 load fail | 每个 SoC 单独 build |
| eager 比 OM 快 | 模型太小 launch overhead 反占大头 | 用 TorchAir graph 或 batch 拉大 |

## 验证编译生效

```python
import torch, torch_npu, torchair
# 看 backend 是否真的 NPU graph
print(torch.npu.is_available())
# msprof 后看 op 数（编译后应大幅减少）
```

## 与 NVIDIA 对应

| NV | Ascend |
|---|---|
| `torch.compile(mode="reduce-overhead")` | `torch.compile(backend=npu_backend)` (TorchAir) |
| `trtexec --onnx ...` | `atc --model ...` |
| TRT `--builderOptimizationLevel=5` | `--auto_tune_mode="RL,GA"` |
| TRT `--minShapes/--optShapes/--maxShapes` | `--dynamic_dims / --dynamic_batch_size` |
| TRT-LLM | MindIE-LLM |
| ONNX Runtime + TRT EP | ATC OM + AscendCL |
| TorchInductor Triton | Ascend C 自定义算子 |

## 反模式

- 上线还在用 eager torch_npu —— 性能差 3-10×
- AOE 跑一半 kill 掉 —— 缓存不完整，可能 corrupted
- OM 跨 SoC / CANN 版本复用 —— 必失败
- 大量 dynamic shape 不分桶 —— 每个 unique shape 触发 op 重编译，p99 崩
- 自定义 op 不注册到 torch_npu —— TorchAir 编译失败

## 参考

- CANN 文档 ATC 工具使用指南
- TorchAir GitHub (huawei-ascend/torch_npu)
- MindIE 用户指南
- `references/hardware/tooling-cheatsheet.md`
