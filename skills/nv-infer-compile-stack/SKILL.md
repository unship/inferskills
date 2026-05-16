---
name: nv-infer-compile-stack
description: NVIDIA GPU 推理的编译/图优化栈——torch.compile / TorchInductor、TensorRT、TensorRT-LLM、ONNX Runtime + TRT EP、CUDA Graphs、Triton、autotuning、dynamic shape 处理。当用户问"torch.compile"、"TensorRT"、"TRT-LLM"、"ONNX"、"图优化"、"算子融合"、"autotune"、"dynamic shape"、"FX graph"、"为什么编译完更慢"、"warmup"、"recompile 抖动" 时使用。
---

# Compile Stack — torch.compile / TensorRT / Graphs

> 这一层是单卡推理 ROI 最高的优化之一：常见 **1.5-3×** 提速且几乎免维护成本。
> 顺序：torch.compile → TensorRT(-LLM) → 自定义 Triton/CUTLASS。

## 选型决策

| 场景 | 首选 |
|---|---|
| PyTorch eager 模型，想最快上线 | `torch.compile` |
| LLM 服务化（vLLM/TRT-LLM 已支持的模型） | **TensorRT-LLM 或 vLLM 内置编译** |
| 固定 shape 的 CNN/ViT，离线部署 | **TensorRT** |
| 已有 ONNX 工件 | **ONNX Runtime + TensorRT EP** |
| LLM decode 极致低延迟 | torch.compile + CUDA Graphs / TRT-LLM in-flight batching |
| 框架没覆盖的奇异算子 | Triton 写一个，再用 torch.compile 接 |

## torch.compile 实操

```python
import torch

# CNN/ViT 推理建议 mode
model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)

# LLM decode（含动态 seq）
model = torch.compile(model, mode="reduce-overhead", dynamic=True)

# 训练或要最高峰值（编译时间长）
model = torch.compile(model, mode="max-autotune")
```

### 三个 mode 的区别

| mode | 行为 | 何时用 |
|---|---|---|
| `default` | 基础融合，无 graphs | 兼容性测试 |
| `reduce-overhead` | 启用 CUDA Graphs，消除 launch 开销 | **推理首选** |
| `max-autotune` | Inductor 试更多 tile/algo，编译慢 | 离线 / 跑很久的服务 |

### dynamic=True 的代价

每个新 shape 触发 **重编译**（数秒到分钟）。LLM 服务必须设 `dynamic=True`，否则每个 seq_len 重新编译会导致 p99 抖到几百 ms。
推荐：**做 shape padding** 把 seq_len round 到 32/64 的倍数，让 unique shapes 数量收敛到 < 20 个。

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64   # 默认 8，LLM 一定要调大

# 抑制 recompile 触发条件
torch._dynamo.config.recompile_limit = 8
```

### 排查没生效

```python
import torch._dynamo
torch._dynamo.config.verbose = True
# 看 fallback / graph break 在哪里
```

常见 graph break 原因：
- Python `print` / `pdb` / 任意 `.item()`
- 数据依赖控制流（`if x.sum() > 0: ...`）
- 调用未注册的 C 扩展
- 张量与 numpy/list 混算

### Warmup

编译第一次会非常慢（**10-60s**）。生产服务：
```python
# 启动时跑几个代表性 shape
for shape in representative_shapes:
    _ = model(torch.zeros(shape, device='cuda'))
torch.cuda.synchronize()
# 这之后才接流量
```

## TensorRT（CNN / ViT）

最佳路径：

```python
# PyTorch → ONNX → TRT engine
import torch
torch.onnx.export(model, (sample,), "m.onnx",
    input_names=["x"], output_names=["y"],
    dynamic_axes={"x": {0: "B"}},
    opset_version=17)

# trtexec 构建（命令行最方便）
# trtexec --onnx=m.onnx --saveEngine=m.plan \
#   --fp16 --useCudaGraph \
#   --minShapes=x:1x3x224x224 \
#   --optShapes=x:32x3x224x224 \
#   --maxShapes=x:128x3x224x224 \
#   --builderOptimizationLevel=5 \
#   --memPoolSize=workspace:8192
```

关键 flags：
- `--fp16` / `--bf16` / `--int8` / `--fp8`：精度，TRT 自动选最佳 kernel
- `--useCudaGraph`：执行时包 CUDA Graph
- `--builderOptimizationLevel=5`：试更多 tactics（构建慢，运行快）
- `--minShapes/--optShapes/--maxShapes`：dynamic shape 范围；优化在 opt 上做

### INT8 calibration

```bash
trtexec --onnx=m.onnx --int8 --calib=calib.cache --saveEngine=m_int8.plan
# 自行喂 calibration data：用 Python API + IInt8EntropyCalibrator2
```

参考 NVIDIA 官方的 `polygraphy` 工具做精度比对：
```bash
polygraphy run m.onnx --trt --fp16 --validate
polygraphy convert m.onnx -o m.plan --fp16
```

### 验证 TRT 用了什么 tactic

```bash
trtexec --loadEngine=m.plan --dumpLayerInfo --exportLayerInfo=layers.json
```
看每层用了 cublas/cudnn/myelin/foreignNode 哪个 path。

## TensorRT-LLM（LLM 专用）

LLM 单卡/多卡推理的"最快"路径，但模型适配工作量大。

```bash
# HF → TRT-LLM checkpoint → engine
python convert_checkpoint.py --model_dir hf_llama3_8b/ \
    --output_dir ckpt/ --dtype float16

trtllm-build --checkpoint_dir ckpt/ \
    --output_dir engine/ \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --use_fused_mlp \
    --use_paged_context_fmha enable \
    --kv_cache_type=fp8 \
    --max_batch_size 256 \
    --max_input_len 4096 \
    --max_output_len 2048 \
    --max_num_tokens 16384 \
    --tp_size 2 --pp_size 1
```

关键能力（必开）：
- `--remove_input_padding`：变长 batch 不浪费 token slot
- `--paged_kv_cache`：PagedAttention 风格 KV
- `--use_paged_context_fmha`：长 context prefill 用 paged FMHA
- `--use_fused_mlp`：MLP 融合
- `--kv_cache_type=fp8`（H100+）：KV 显存减半
- `--gemm_plugin`：选 `float16/bfloat16/fp8` 让 plugin 路径

```bash
# 跑 inference benchmark
python ../run.py --engine_dir engine/ --max_output_len 100 \
    --tokenizer_dir hf_llama3_8b/ --input_text "Once upon a time"

# 服务化
mpirun -n 2 --allow-run-as-root \
  python launch_triton_server.py --model_repo=triton_repo/
```

## ONNX Runtime + TensorRT EP

适合多框架（TF/Keras/PyTorch）统一 ONNX 后部署。

```python
import onnxruntime as ort
sess = ort.InferenceSession("m.onnx",
    providers=[
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_max_workspace_size": 8 * 1024**3,
            "trt_builder_optimization_level": 5,
        }),
        "CUDAExecutionProvider",
    ])
```

第一次跑慢（构 engine 缓存），之后正常。

## CUDA Graphs 独立用法

不要重复 `nv-infer-kernels` 里的写法，要点：
- decode-only LLM：每个 batch_size 桶一张 Graph
- ViT/CNN：每个 input shape 一张 Graph
- 配 PyTorch 用 `torch.cuda.graph()` context manager，必须 warmup 充分

## Triton（OpenAI Triton，非 Triton Inference Server）

何时写：
- 框架没的算子（特殊 attention mask、自定义量化）
- torch.compile 没 fuse 上
- 又想保持高级语言可读性

```python
import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

Triton 3.0+ 在 Hopper 上自动用 TMA / WGMMA。

## Autotuning

```python
import triton
@triton.autotune(
    configs=[
        triton.Config({'BM':128,'BN':128,'BK':32}, num_warps=4, num_stages=3),
        triton.Config({'BM':128,'BN':256,'BK':64}, num_warps=8, num_stages=4),
        ...
    ],
    key=['M','N','K'],   # 仅当这些 key 变化才重选
)
@triton.jit
def gemm_kernel(...): ...
```

torch.compile 的 `mode="max-autotune"` 内部对 Inductor 生成的 Triton kernel 也做 autotune。

## Dynamic shape 处理通用策略

1. **Bucketize**：把输入 shape padding 到 32/64/128 几个桶
2. **Fixed graphs**：每个桶一张 CUDA Graph / TRT profile
3. **Avoid 1-token paths**：LLM decode 始终 batch 1+padding，不要变 batch

## 验证编译生效

```python
# torch.compile
# 1. 看是否 graph break
import torch._logging
torch._logging.set_logs(graph_breaks=True)

# 2. 看实际跑的是 inductor 生成的 kernel（nsys 里 kernel 名字带 triton_）
```

```bash
# TRT
trtexec --loadEngine=m.plan --useCudaGraph --warmUp=2000 --iterations=1000 \
        --avgRuns=100 --noDataTransfers --separateProfileRun
```

## 常见坑

| 坑 | 现象 | 修 |
|---|---|---|
| `torch.compile` 第一次极慢 | p99 突刺 | warmup 跑遍代表性 shape |
| recompile 风暴 | 每个新 shape 都卡数秒 | shape padding + 调 `cache_size_limit` |
| TRT engine 与驱动/CUDA 强绑 | 换 GPU 报 deserialization fail | 每个 SM arch / cuda / trt 版本各 build 一次 |
| TRT FP16 精度掉 | 某层数值溢出 | `polygraphy debug precision` 找罪魁，单独保留 FP32 |
| ONNX export 包含 unsupported op | TRT EP fallback CPU | 替换为 supported op 或写 plugin |
| TRT-LLM `--remove_input_padding=disable` | KV 浪费 | 一定开 |
| CUDA Graph 抓时输入是 cpu tensor | replay 时数据没传 | 输入必须 pre-allocated GPU buffer，每次 copy_ 进去 |
