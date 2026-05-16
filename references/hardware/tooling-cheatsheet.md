# NVIDIA ↔ Ascend 工具对照速查

> 同一件事，两套生态怎么做。给同时管 NV + 华为 NPU 的工程师用。
> 命令尽量给"能直接 copy 跑"的，不展开原理。

## 1. 卡状态 / 监控

| 你想做的事 | NVIDIA | Ascend (910B / 310P) |
|---|---|---|
| 列设备 | `nvidia-smi -L` | `npu-smi info -l` |
| 简表 | `nvidia-smi` | `npu-smi info` |
| 详情 | `nvidia-smi -q -i 0` | `npu-smi info -t board -i 0` |
| 单字段查询 | `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` | `npu-smi info -t common -i 0 -c 0` |
| dmon (实时) | `nvidia-smi dmon -s pucvmet -c 30` | `npu-smi info watch -i 0` |
| 拓扑 | `nvidia-smi topo -m` | `npu-smi info -t topo` |
| NVLink/HCCS 链路 | `nvidia-smi nvlink --status -i 0` | `npu-smi info -t link -i 0` |
| 进程占用 | `nvidia-smi pmon` | `npu-smi info -t proc -i 0 -c 0` |
| 时钟 | `nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv` | `npu-smi info -t freq -i 0 -c 0` |
| 温度 | `nvidia-smi --query-gpu=temperature.gpu --format=csv` | `npu-smi info -t temp -i 0 -c 0` |
| 功耗 | `nvidia-smi --query-gpu=power.draw --format=csv` | `npu-smi info -t power -i 0 -c 0` |
| ECC 状态 | `nvidia-smi -q -d ECC` | `npu-smi info -t ecc -i 0 -c 0` |

### 设置 / 控制

| 任务 | NVIDIA | Ascend |
|---|---|---|
| 持久化模式 | `sudo nvidia-smi -pm 1` | （Ascend 默认常驻，无对等开关） |
| 锁时钟 | `sudo nvidia-smi -ac MEM,GFX` | `npu-smi set -t freq -i 0 -c 0 -f <MHz>` |
| 功耗墙 | `sudo nvidia-smi -pl 600` | `npu-smi set -t power -i 0 -c 0 -p <W>` (部分支持) |
| 高性能模式 | `nvidia-smi --auto-boost-default=DISABLED` | `npu-smi set -t power-mode -i 0 -c 0 -p 1` |
| ECC 开关 | `sudo nvidia-smi -e 1/0` | （910B 默认开 ECC，关闭需固件层）|
| 切分 | `nvidia-smi mig -cgi <profile> -C` | `npu-smi set -t vnpu -i 0 -c 0 -v <profile>` |
| 切分查看 | `nvidia-smi mig -lgi` | `npu-smi info -t vnpu -i 0 -c 0` |
| 重置 | `sudo nvidia-smi --gpu-reset -i 0` | `npu-smi set -t reset -i 0 -c 0` |

## 2. Profiling

| 任务 | NVIDIA | Ascend |
|---|---|---|
| 时间线（粗粒度） | `nsys profile -t cuda,nvtx,osrt -o out python a.py` | `msprof --output=./prof --application="python a.py" --task-time=on --aicpu=on --runtime-api=on` |
| 单 kernel 微观 | `ncu --kernel-name regex:"name" --set full -o out python a.py` | `msprof --aic-metrics=PipeUtilization,ArithmeticUtilization,MemoryAccess --application="python a.py"` |
| 报表 | `nsys stats --report cuda_gpu_kern_sum out.nsys-rep` | MindStudio Profiler 打开 .prof 目录 |
| 手动打点 | `import torch.cuda.nvtx as nvtx; nvtx.range_push("x"); ...; nvtx.range_pop()` | `msprof.range_push("x"); ...; msprof.range_pop()` (或 MindSpore mindspore.profiler) |
| 框架 profiler | `torch.profiler` / TF Profiler | `torch_npu.profiler` / MindSpore Profiler |
| 火焰图 | `nsys-ui` / Chrome trace | MindStudio Trace View |

## 3. 编译 / 引擎构建

| 任务 | NVIDIA | Ascend |
|---|---|---|
| 离线编译 (ONNX → engine) | `trtexec --onnx=m.onnx --fp16 --saveEngine=m.plan` | `atc --model=m.onnx --framework=5 --soc_version=Ascend910B3 --output=m_om` |
| 引擎运行 | `trtexec --loadEngine=m.plan --shapes=...` | `benchmark --model=m_om.om --device=0 --loop=1000` 或 acl 自写 |
| INT8 calib | `trtexec --int8 --calib=calib.cache` | `atc --insert_op_conf=aipp.cfg --precision_mode=allow_mix_precision`（+ calibration tool） |
| dynamic shape | `--minShapes / --optShapes / --maxShapes` | `--dynamic_dims="1,3,224,224;1,3,448,448"` |
| PyTorch JIT 编译 | `torch.compile(model, mode="reduce-overhead")` | `torch_npu.contrib.module.npu_compile(model)` 或 TorchAir graph mode |
| 图优化级别 | `--builderOptimizationLevel=5` | `--auto_tune_mode="RL,GA"`（自动调优） |
| 检查 layer 信息 | `trtexec --dumpLayerInfo` | `atc --dump_data` / MindStudio Graph Viewer |

## 4. 推理服务

| 能力 | NVIDIA | Ascend |
|---|---|---|
| 推理服务器 | Triton Inference Server | MindIE Service / MindX SDK |
| LLM 推理引擎 | TensorRT-LLM / vLLM / SGLang | MindIE-LLM / atb_llm |
| 持续批处理 | 默认开（vLLM/TRT-LLM） | MindIE-LLM 7.0+ 默认开 |
| Paged KV | vLLM PagedAttention / TRT-LLM `--paged_kv_cache` | MindIE-LLM 内置 paged KV (CANN 7.0+) |
| Prefix cache | `--enable-prefix-caching` (vLLM) / `--enable_kv_cache_reuse` (TRT-LLM) | MindIE-LLM 配置项 `enablePrefixCache` |
| FlashAttention | `F.scaled_dot_product_attention` 自动 FA-2/3 | `aclnnFlashAttentionScore` (CANN 7.0+) |
| HTTP/gRPC | Triton 内置 | MindIE-Service 内置 |
| Prometheus | Triton 内置 `/metrics` | MindIE 内置 `/metrics` |

## 5. 通信 / 多卡

| 任务 | NVIDIA | Ascend |
|---|---|---|
| 集合通信库 | NCCL | HCCL |
| 单节点高速互联 | NVLink + NVSwitch | HCCS（节点内 8 卡全互联）|
| 跨节点 | InfiniBand / RoCE + GPUDirect RDMA | HCCN (RoCE) + Ascend Direct |
| 带宽测试 | `./all_reduce_perf -b 8 -e 1G -f 2 -g 8` (nccl-tests) | `./bin/all_reduce_test -b 8 -e 1G -f 2 -d fp16 -p 8` (hccl_test) |
| 关键 env | `NCCL_P2P_LEVEL=NVL NCCL_IB_HCA=... NCCL_DEBUG=INFO` | `HCCL_BUFFSIZE=512 HCCL_CONNECT_TIMEOUT=1200 HCCL_OVER_OFI=0` |
| TP/PP/EP | vLLM `--tensor-parallel-size` / TRT-LLM `--tp_size --pp_size --moe_ep_size` | MindIE-LLM TP/PP 配置 / DeepSpeed-NPU |
| Fabric Manager | `systemctl enable nvidia-fabricmanager`（NVSwitch 必需） | （Ascend HCCS 由 driver/firmware 内建）|

## 6. 量化

| 能力 | NVIDIA | Ascend |
|---|---|---|
| 框架 | TRT INT8 + nvidia-modelopt / llmcompressor | MindIE Quantizer / atb_llm 量化工具 |
| W8A8 PTQ | TRT `--int8 --calib` / SmoothQuant | MindIE-LLM `--quant w8a8` |
| W4A16 | AWQ / GPTQ + Marlin/Machete kernel | MindIE-LLM `--quant w4a16` (类 AWQ) |
| FP8 | TransformerEngine / TRT-LLM `--use_fp8` | **不支持**（无硬件） |
| FP4 / MXFP4 | Blackwell only，TRT-LLM 0.13+ | **不支持** |
| 2:4 sparsity | `torch.sparse.to_sparse_semi_structured` + cuSPARSELt | （硬件无对等结构稀疏）|
| KV cache 量化 | `--kv-cache-dtype fp8_e4m3 / int8` (vLLM)，`--kv_cache_type=fp8` (TRT-LLM) | MindIE-LLM `enableKvQuant=true`（INT8 KV） |

## 7. 容器 / K8s

| 任务 | NVIDIA | Ascend |
|---|---|---|
| Runtime | nvidia-container-toolkit | Ascend Docker Runtime / Ascend Container Toolkit |
| Docker flag | `--gpus all` | `--device=/dev/davinci0 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver` |
| K8s device plugin | `nvidia.com/gpu` | `huawei.com/Ascend910` 或 `huawei.com/Ascend310P` |
| 镜像源 | `nvcr.io/nvidia/*` | `ascendhub.huawei.com/public-ascendhub/*` |
| 拓扑感知 | Topology Manager + NUMA | Ascend Scheduler Plugin |
| 切分调度 | MIG-aware scheduling | vNPU-aware scheduling |

## 8. PyTorch 代码切换

```python
# NVIDIA
import torch
x = torch.randn(8, device='cuda')
model = model.cuda().half()

# Ascend (910B / 310P)
import torch
import torch_npu                              # 关键：加载 NPU backend
import torch_npu.contrib.transfer_to_npu      # 把 .cuda() 自动改写为 .npu()
x = torch.randn(8, device='npu')              # 或 device='npu:0'
model = model.npu().half()
```

环境变量 / 设备选择：
- NVIDIA: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Ascend: `ASCEND_VISIBLE_DEVICES=0,1,2,3` 或 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`

## 9. 内存 / 数据格式

| 概念 | NVIDIA | Ascend |
|---|---|---|
| device → host | `tensor.cpu()` | `tensor.cpu()` (torch_npu) / `aclrtMemcpy` |
| host → device | `.cuda(non_blocking=True)` | `.npu(non_blocking=True)` |
| pinned memory | `pin_memory=True` | `pin_memory=True`（torch_npu DataLoader 支持） |
| 推理偏好数据布局 | channels_last (NHWC) for conv | **NZ format** (内部分块 5D)，由 ATC 自动转换 |
| 卷积输入存储 | NCHW 或 NHWC | NCHW 进口 + NZ 内部 |
| 矩阵分块 | tile 16×8×16 (Hopper) | Cube 单元 16×16×16 |

## 10. 错误码 / 排障

| 类型 | NVIDIA | Ascend |
|---|---|---|
| 驱动版本 | `cat /proc/driver/nvidia/version` | `cat /usr/local/Ascend/driver/version.info` |
| Runtime 版本 | `nvcc --version` | `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` |
| Firmware 版本 | `nvidia-smi -q -d FIRMWARE` | `npu-smi info -t firmware -i 0 -c 0` |
| 内核日志 | `dmesg \| grep -i nvidia` | `dmesg \| grep -i ascend` 或 `/var/log/ascend_seclog/` |
| 重启服务 | `sudo systemctl restart nvidia-persistenced` | `sudo systemctl restart ascend_*`（driver/firmware） |
| 健康检查 | `dcgmi health -c` | `npu-smi info -t health -i 0 -c 0` |

## 11. 速查命令模板（直接套）

### 启动一个 NVIDIA H100 LLM 服务（生产）

```bash
sudo nvidia-smi -pm 1 && \
sudo systemctl start nvidia-fabricmanager && \
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled && \
ulimit -l unlimited && ulimit -n 65535 && \
numactl --cpunodebind=0 --membind=0 \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_DEVICE_MAX_CONNECTIONS=32 \
      NCCL_P2P_LEVEL=NVL \
  vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching --enable-chunked-prefill \
    --gpu-memory-utilization 0.92
```

### 启动一个 Ascend 910B LLM 服务（生产）

```bash
npu-smi set -t power-mode -i 0 -c 0 -p 1 && \
ulimit -n 65535 && \
export HCCL_CONNECT_TIMEOUT=1200 \
       HCCL_EXEC_TIMEOUT=1200 \
       HCCL_BUFFSIZE=512 \
       HCCL_OVER_OFI=0 \
       ASCEND_GLOBAL_LOG_LEVEL=3 \
       ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
mindieservice_daemon \
    --config /etc/mindie/service_llama3_70b.json
# service_*.json 内含 tp=8 / paged_kv / continuous batching / max_batch 等
```

### 视频推理：300V Pro + 主推理卡组合

```bash
# 节点 1: Atlas 300V Pro 做视频解码 + 前置 CNN
# DVPP → AIPP → 小检测网络 → 输出 ROI/feature
mindie_videoservice --config video_pipeline.json   # 264 路 1080p

# 节点 2: H100 / 910B 做主 LLM (多模态) 推理
vllm serve llava-onevision-72b-ov ...
# 或
mindieservice_daemon --config llava_72b.json
```

## 12. 同概念的"心智模型差异"

| 你直觉认为... | NVIDIA 现实 | Ascend 现实 |
|---|---|---|
| 写 kernel 是写线程束并发 | SIMT，32 线程一个 warp | **不是 SIMT** —— Cube 是矩阵流水线，写 Ascend C 像写 systolic |
| Shared memory 用就是了 | OK，48-228KB/SM | Unified Buffer (UB) 容量限制更严，需要 manual stage |
| 算子融合靠编译器 | torch.compile / TRT 自动 | ATC/GE 自动 + 部分需要写"fusion pass" 或 Ascend C |
| 显存大就能装大模型 | 是 | 是，但 NZ format 内部 padding 会膨胀 5-15% 占用 |
| INT8 calibration 跑一次稳了 | 通常是 | shape 一变可能要重 calib |
| 多卡 NCCL 一键调好 | 大致 | HCCL 要严格按拓扑 + 环境变量，错一个就降速 |
| ONNX 通用 | TRT 90%+ ops | ATC 覆盖好但**最新 op** 滞后；版本敏感 |

## 13. 文档定位

| 主题 | NVIDIA | Ascend |
|---|---|---|
| 用户文档 | docs.nvidia.com | support.huawei.com/enterprise (Ascend 文档中心) |
| 开发者社区 | developer.nvidia.com / forums.developer.nvidia.com | hiascend.com / developer.huaweicloud.com |
| GitHub | github.com/NVIDIA | github.com/Ascend (torch_npu, mindspore, modelzoo) |
| 模型仓库 | NGC catalog | Ascend ModelZoo |
| 容器仓库 | nvcr.io | ascendhub.huawei.com |

## 反模式（混合栈常见）

- 在 Ascend 上找 FP8 路径 — 没有，别找了
- 把 NVIDIA 调出来的 W4A16 awq.pt 直接 load 到 Ascend — 量化 scheme 不同，重做
- 在 Ascend 上 expecting `nvidia-smi` 一样的输出格式 — `npu-smi` 字段名/顺序都不同
- 设了 `CUDA_VISIBLE_DEVICES` 期望它影响 NPU — 用 `ASCEND_RT_VISIBLE_DEVICES`
- 一套 K8s manifest 跑两边 — resource type 不同，必须分配置
- 混部时 client SDK 不抽象 — 抽 OpenAI 兼容 HTTP 层就好

---

> 这份对照偏命令级；概念层的对照见 [`README.md`](README.md) 末尾"NVIDIA 与 Ascend 的心智模型映射"小节。
