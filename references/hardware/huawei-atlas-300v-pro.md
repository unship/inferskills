# Huawei Atlas 300V Pro (Ascend 310P3)

> 华为面向**视频 AI**与**边缘推理**的主力卡。72W 被动 + 264 路 1080p 解码 + 140 TOPS INT8。
> 你买它的核心理由：**端到端片上 video pipeline**（解码 → AIPP 预处理 → 推理 → 后处理 全片上完成），CPU 几乎零占用。

## 一句话画像

```
Ascend 310P3 (Da Vinci) · 24 GB LPDDR4X · ~204 GB/s · 72W 被动 · PCIe Gen4 x16 · 无卡间互联
INT8: 140 TOPS · FP16: 70 TFLOPS · 视频解码 264×1080p@30 / 16×4K@30
```

## 同代产品矩阵（避免混淆）

| 卡 | 主要用途 | SoC | TDP | 显存 |
|---|---|---|---|---|
| **Atlas 300V Pro** | **视频推理（本文件）** | Ascend 310P3 | 72 W | 24 GB LPDDR4X |
| Atlas 300I Pro | 通用推理 | Ascend 310P3 | 72-150 W | 24 GB LPDDR4X |
| Atlas 300I Duo | 双芯通用推理 | 2× Ascend 310P3 | 150 W | 2 × 24 GB = 48 GB |
| Atlas 300V | 视频推理（前代/简化版） | Ascend 310 (非 P3) | 67 W | 8 GB LPDDR4 |

V Pro 与 I Pro 都基于 Ascend 310P3，**算力相同**，差异在视频编解码 IP 数与外围（V Pro 视频通道更多、I Pro 通用 I/O 更全）。

## 核心规格

| 项 | 值 |
|---|---|
| SoC | Huawei Ascend 310P3 |
| 架构 | Da Vinci (Lite core) |
| AI Core 数 | 8（含 Cube/Vector/Scalar 子单元） |
| **INT8** | **140 TOPS** |
| **FP16** | **70 TFLOPS** |
| INT4 | 部分支持，性能未官方披露 |
| 显存 | 24 GB LPDDR4X @ 4266 MT/s |
| 显存带宽 | ~204 GB/s |
| PCIe | Gen4 x16 (~32 GB/s 双向) |
| 卡间互联 | **无（不支持多卡 TP）** |
| 视频解码 | **264 路 1080p@30 / 16 路 4K@30**（H.264 / H.265 / AVS2 / AVS3 / MPEG-4） |
| 视频编码 | 16 路 1080p@30（H.264 / H.265） |
| JPEG 编解码 | 硬核 JPEG 编/解（用于 image pipeline） |
| TDP | 72 W |
| 冷却 | 被动 |
| 形态 | HHHL，单插槽 |
| vNPU 切分 | 支持（最多 4 切分，类比 MIG） |

## 软件栈

| 组件 | 角色 | 类比 NVIDIA |
|---|---|---|
| **CANN** (Compute Architecture for Neural Networks) | 全栈基础库（drivers / runtime / op lib / compiler） | CUDA + cuDNN + cuBLAS |
| **AscendCL** (acl) | Runtime API (C / C++ / Python binding) | CUDA Runtime API |
| **ATC** (Ascend Tensor Compiler) | 离线编译器：ONNX/Caffe/TF/MindSpore → `.om` | TensorRT builder |
| **OM** (Offline Model) | 编译后的可执行模型 | TRT `.plan` |
| **MindIE** | 推理引擎（含 LLM 子套件） | TensorRT + TRT-LLM |
| **AIPP** (AI Pre-Processing) | 片上预处理（resize/crop/normalize） | DALI / nvJPEG (片上) |
| **DVPP** (Digital Vision Pre-Processing) | 视频/图像编解码 + 预处理 IP | NVDEC/NVENC + nvJPEG |
| **MindSpore** | 华为深度学习框架 | PyTorch / TensorFlow |
| **torch_npu** | PyTorch on Ascend 适配层 | (PyTorch native CUDA) |
| **MindX SDK** | 视频分析、行业应用 SDK | DeepStream |
| **MindStudio** | IDE + Profiler + 调优 | Nsight Systems/Compute |
| **HCCL** | 集合通信库（300V 单卡用不上） | NCCL |
| **msprof** | 命令行 profiler | nsys/ncu |
| **npu-smi** | 卡管理 / 监控 | nvidia-smi |

### 推荐 CANN 版本
- 生产稳定：CANN 7.0.RC1 / 7.0.0
- 最新：CANN 8.0.RC* （新算子、PyTorch 2.1+ 支持完善）
- 检查兼容：每个 CANN 版本对应一套 driver/firmware，**不要混搭**

## 推理实用特性

- ✅ **AIPP**：JPEG 解码 + resize + normalize + dtype 转换全片上，CPU 不参与
- ✅ **DVPP**：视频流直接解码到 NPU 显存，**零拷贝**接入推理
- ✅ **多 Stream**：`aclrtStream`，并行 decode/preproc/infer
- ✅ **vNPU**：4 个切分，多模型隔离
- ❌ 不支持 BF16 训练（推理 FP16 即可）
- ❌ 不支持 FP8/FP4
- ❌ 不支持多卡 TP（无 HCCS）—— 多卡场景靠数据并行 (DP)
- ❌ 算子覆盖：奇异 attention 变体可能需要写 Ascend C kernel

## 适合 / 不适合的 workload

| 类型 | 适合度 |
|---|---|
| **视频 AI**（人脸/行为/车牌/检测/分类）大并发 | ✅✅✅ 这是它的舒适区 |
| YOLO 系列 + 视频流 INT8 | ✅✅✅ |
| ResNet / EfficientNet / MobileNet INT8 | ✅✅ |
| ViT-B/L 推理 (FP16) | ✅ |
| BERT-base / 小 NLP 模型 | ✅ |
| Stable Diffusion 1.5 | ⚠️ MindIE-SD 有适配，速度一般 |
| LLM ≤ 7B INT8 | ⚠️ 显存够，但 LPDDR 带宽 204 GB/s 限制 decode 吞吐 |
| LLM ≥ 13B | ❌ 显存装不下或太慢 |
| 训练任何模型 | ❌ 不是训练卡 |

## ATC 编译样板

```bash
# ResNet50 ONNX → OM (INT8)
atc --model=resnet50.onnx \
    --framework=5 \
    --soc_version=Ascend310P3 \
    --output=resnet50_int8 \
    --input_format=NCHW \
    --input_shape="input:1,3,224,224" \
    --insert_op_conf=aipp.cfg \
    --precision_mode=allow_mix_precision \
    --modify_mixlist=mixlist.json
# --framework: 0=Caffe, 3=TF, 5=ONNX, 1=MindSpore
```

### AIPP 配置示例（aipp.cfg）

```yaml
aipp_op {
    aipp_mode: static
    input_format: YUV420SP_U8           # DVPP 解码输出格式
    src_image_size_w: 1920
    src_image_size_h: 1080
    crop: true
    load_start_pos_h: 0
    load_start_pos_w: 0
    crop_size_w: 224
    crop_size_h: 224
    csc_switch: true                    # YUV → RGB
    matrix_r0c0: 256
    ...
    mean_chn_0: 124
    mean_chn_1: 117
    mean_chn_2: 104
    var_reci_chn_0: 0.017
    ...
}
```

AIPP 把"YUV decode → crop → CSC → normalize → uint8→fp16"一次完成在 NPU 上，**零 CPU 占用**。

## 推理代码骨架（C++/AscendCL）

```cpp
#include <acl/acl.h>
aclInit(nullptr);
aclrtSetDevice(0);
aclrtContext ctx;  aclrtCreateContext(&ctx, 0);
aclrtStream stream; aclrtCreateStream(&stream);

uint32_t model_id;
aclmdlLoadFromFile("resnet50_int8.om", &model_id);

aclmdlDataset *input  = aclmdlCreateDataset();
aclmdlDataset *output = aclmdlCreateDataset();
// allocate aclrtMalloc buffers + aclCreateDataBuffer + aclmdlAddDatasetBuffer ...

aclmdlExecuteAsync(model_id, input, output, stream);
aclrtSynchronizeStream(stream);
```

Python 端用 MindX SDK 或 ACL Python binding，写法类似。

## MindIE 推理 server（生产推荐）

```bash
# 启动 MindIE Service
export MIES_CONTAINER_IP=0.0.0.0
mindieservice_daemon \
    --model_path /models/resnet50.om \
    --port 1025 \
    --batch_size 16
```

MindIE 支持：
- 多模型/多实例
- 动态批处理（类似 Triton dynamic_batching）
- HTTP/gRPC
- Prometheus metrics

## 调优速记

1. **必做**：
   - DVPP decode → NPU 显存零拷贝（不要 D2H 后 CPU 再 H2D）
   - AIPP 把预处理全片上
   - ATC `--precision_mode=allow_mix_precision`，让 ATC 自动 INT8/FP16 混搭
   - 多 Stream 并行 (decode/preproc/infer 三流水线)
2. **量化**：
   - INT8 calibration：`atc --calibration_set=...` 用 200-500 样本
   - 精度敏感层留 FP16：`--keep_dtype=keep_fp16.cfg`
3. **batch / shape**：
   - 静态 shape 性能最好；动态 shape 用 `--dynamic_batch_size` 提前枚举
   - LPDDR 带宽限制下 batch 拉大常常**反而变慢**（cache 命中率掉）
4. **vNPU 切分**：
   - 多个小模型服务共卡时切 vNPU，避免互抢
   - `npu-smi set -t vnpu -i 0 -c 0 -v 1` 配置
5. **视频流**：
   - DVPP 解码后**保持 YUV 格式**，由 AIPP 一次转 RGB+normalize
   - 不要在用户代码里 CPU 跑 OpenCV resize / cvtColor

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| ATC 编译报某 op unsupported | 算子缺 | 升 CANN / 写 Ascend C 自定义 op / 替换模型 op |
| 模型转换后精度掉很多 | INT8 量化 scale 错 | 增 calibration 样本 / 关键层 keep_fp16 |
| 同模型在 V Pro 比 I Pro 慢 | V Pro 视频通道占了部分片上资源 | I Pro 跑纯计算更稳；视频流量大才 V Pro |
| DVPP 解码失败 | 视频码流不在支持列表 | 检查支持的 profile/level；某些 4K HDR / VP9 不支持 |
| 内存碎片，连续运行后 OOM | LPDDR 分配碎片 | aclrtMalloc 用 `ACL_MEM_MALLOC_HUGE_FIRST` |
| 自定义 OpenCV 数据流 | CPU 100% 占用，NPU idle | 改 DVPP/AIPP |
| torch_npu 装错版本 | import 报版本不匹配 | torch_npu / CANN / firmware / driver 必须按 release matrix 对齐 |
| INT4 量化 | 文档说支持但生态不齐 | LLM 上慎用，CNN/ViT 上 INT8 通常已经够 |

## 实测命令

```bash
# 卡况（npu-smi 是 Ascend 的 nvidia-smi）
npu-smi info
npu-smi info -t board -i 0
npu-smi info -t temp -i 0 -c 0      # 温度
npu-smi info -t power -i 0 -c 0     # 功耗
npu-smi info watch                   # 类似 nvidia-smi dmon

# 列芯片支持
npu-smi info -l                      # NPU 列表

# 测推理基线
# 用 ATC 编译完，然后 aclTools/benchmark 或 MindStudio benchmark
benchmark --model=resnet50_int8.om \
    --device=0 --loop=1000 --batch=16 \
    --output_text_path=./bench.log

# msprof 时间线
msprof --output=./prof_out --application="./infer_app --model resnet50_int8.om" \
       --aic-metrics=PipeUtilization,ArithmeticUtilization \
       --aicpu=on --runtime-api=on --task-time=on
```

## NPU 监控（生产）

CANN 提供 Prometheus 格式 metrics（通过 `npu-smi` JSON 输出或 MindX 监控组件）：
- `npu_utilization`（AI Core 利用率）
- `memory_utilization`（HBM/LPDDR 利用率）
- `temperature`、`power`、`voltage`
- `aicore_freq`、`memory_freq`
- 错误码 / ECC

## 部署建议

- 单服务器塞 4-8 张 300V Pro，**完全独立运行**（无 HCCS，无意义做 collective）
- 用 K8s + Huawei device plugin：`huawei.com/Ascend310P` resource type
- 镜像：`ascendhub.huawei.com/public-ascendhub/ascend-infer:7.0.0-310p-arm64` 之类
- ARM 节点（鲲鹏 920）更原生；x86 节点也支持但驱动包不同
- 与 NVIDIA 节点混部：service mesh 层抽象，HTTP/gRPC 端口对齐 OpenAI/Triton 协议

## 与本仓库 skill 的关系

当前 `nv-infer-*` 不直接覆盖 Ascend，但下列概念可类比迁移：

| nv-infer-* skill | 对应到 Ascend 上的做法 |
|---|---|
| `nv-infer-workflow` | 流程一致，工具换成 npu-smi/msprof/ATC |
| `nv-infer-profiling` | nsys → msprof，ncu → MindStudio AI Core analysis |
| `nv-infer-precision` | FP16/INT8 路径，**没有 FP8/BF16 Tensor Core** |
| `nv-infer-compile-stack` | torch.compile/TRT → torch_npu graph mode / ATC |
| `nv-infer-cnn-vit` | 全适用，channels_last 在 Ascend 上是 NCHW + NZ 内部格式 |
| `nv-infer-io-data` | DataLoader → DVPP + AIPP |
| `nv-infer-serving` | Triton → MindIE Service |
| `nv-infer-multigpu` | **不适用**（300V Pro 无 HCCS） |

详细命令对照见 [`tooling-cheatsheet.md`](tooling-cheatsheet.md)。

## 参考

- Huawei Ascend 文档中心 `support.huawei.com/enterprise/zh/doc/EDOC1100285915` 类（CANN 文档）
- Atlas 300V Pro 产品手册（华为官方）
- MindX SDK 视频分析示例（华为 ModelZoo）
- CANN 算子支持列表（每个 release notes 都有）
