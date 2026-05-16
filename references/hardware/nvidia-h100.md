# NVIDIA H100 (Hopper)

> 2022-2025 的推理王者。FP8 Tensor Core + FlashAttention v3 + NVLink/NVSwitch 让 70B 级 LLM 推理成为常态。
> 你要先认清自己手上是哪个 SKU —— H100 SXM5 / H100 PCIe / H100 NVL / H200 性能差异可达 2×。

## SKU 一览（先认对你的卡）

| SKU | 显存 | 显存 BW | TDP | NVLink | SM | 主要差异 |
|---|---|---|---|---|---|---|
| **H100 SXM5 80GB** | 80 GB HBM3 | **3.35 TB/s** | 700 W | 4th-gen, 900 GB/s + NVSwitch | 132 | 主力训练 + 推理，节点 8 卡 |
| **H100 PCIe 80GB** | 80 GB HBM3 | **2.0 TB/s** | 350 W | NVLink Bridge (双卡 600 GB/s) | 114 | 单/双卡，机房友好 |
| **H100 NVL 94GB×2** | 94 GB/卡 HBM3 (188 GB 套装) | **3.9 TB/s/卡** | ~400 W/卡 | 桥接 600 GB/s | 132 | 双卡套装，专为 LLM (装下 70B FP16) |
| **H200 SXM 141GB** | 141 GB **HBM3e** | **4.8 TB/s** | 700 W | 同 H100 SXM | 132 | 算力 ≈ H100，HBM 多 76%、带宽多 43% |
| **H200 NVL 141GB** | 141 GB HBM3e | 4.8 TB/s | 600 W | NVLink Bridge | 132 | PCIe 形态 H200 |
| **H800 SXM 80GB** | 80 GB HBM3 | 3.35 TB/s | 700 W | NVLink 400 GB/s (砍半) | 132 | 中国特供版，砍 NVLink 带宽 |
| **H20 96GB** | 96 GB HBM3 | 4.0 TB/s | 400 W | 900 GB/s | 78 | 砍算力保 BW & NVLink，LLM 推理友好 |

**记住**：
- LLM **prefill (compute-bound)** 看算力 → H100 全系 ≈
- LLM **decode (memory-bound)** 看 HBM BW → H200 > H100 NVL > H100 SXM5 > H100 PCIe
- 多卡 TP 看 NVLink → H100 SXM5 / H200 SXM > H800 (砍) > H100 NVL (桥接) > H100 PCIe

## 核心规格（H100 SXM5 80GB）

| 项 | 值 |
|---|---|
| 架构 | Hopper GH100 |
| Compute Capability | **9.0 (sm_90a)** |
| SM 数 | 132（GH100 有 144 SM，禁用 12 个良率 bin） |
| CUDA Cores | 16896 |
| Tensor Cores | 528（4th gen, FP64/TF32/BF16/FP16/FP8/INT8） |
| L2 Cache | 50 MB |
| 显存 | 80 GB HBM3 (5 stacks × 16 GB) |
| 显存带宽 | 3.35 TB/s |
| 时钟 (boost) | ~1830 MHz |
| FP64 (vector) | 33.5 TFLOPS |
| FP64 Tensor | 67 TFLOPS |
| FP32 (vector) | 67 TFLOPS |
| TF32 Tensor | 989 TFLOPS (sparse 1979) |
| BF16/FP16 Tensor | 989 TFLOPS (sparse 1979) |
| **FP8 Tensor (E4M3/E5M2)** | **1979 TFLOPS (sparse 3958)** |
| INT8 Tensor | 1979 TOPS (sparse 3958) |
| PCIe | Gen5 x16 (128 GB/s 双向) |
| NVLink | 4th gen, 18 link/卡, 900 GB/s 双向聚合 |
| NVSwitch | v3, 8 卡 all-to-all 全速 |
| NVENC | 0（H100 砍掉了 NVENC，只有 NVDEC×7） |
| NVDEC | 7 |
| MIG | 支持，最多 7 实例 (1g.10gb / 2g.20gb / 3g.40gb / 7g.80gb 等) |
| Confidential Compute | ✓ |
| TDP | 700 W |
| 冷却 | 液冷常见，风冷需大风量 |

## Hopper 独有硬件特性

| 特性 | 用途 |
|---|---|
| **TMA** (Tensor Memory Accelerator) | 异步 tile load/store，自动地址生成；FA-3、CUTLASS 3 大量用 |
| **WGMMA** (Warp-Group MMA) | 8 个 warp 协同发起大 GEMM tile，比 mma.sync 更省 register |
| **Thread Block Clusters (CGA)** | 多 block 共享 distributed shared memory，cross-block sync |
| **DPX 指令** | dynamic programming 指令（Smith-Waterman 等） |
| **Transformer Engine** | 软件库自动管理 FP8 amax scaling（PyTorch / JAX / TRT 都集成） |
| **FP8 (E4M3 + E5M2)** | E4M3 前向 / E5M2 反向梯度；INT8 速度，BF16 范围近似 |
| **MIG 1g.10gb** 起步 | A100 是 1g.5gb；H100 切的更粗，单实例更强 |
| **NVLink Network** (NVL72 雏形) | 跨 NVSwitch 域扩展（H100 时代未铺开，B200 NVL72 才完整） |

## 软件栈

| 组件 | 最低版本 | 推荐 (2025) |
|---|---|---|
| Driver | R525 | R570+ (FP8 完整) |
| CUDA | 12.0 | **12.4+** (FP8 + cuDNN 9) |
| cuDNN | 8.9 | **9.5+** |
| cuBLAS / cuBLASLt | 12.0 | 12.4+ |
| TensorRT | 8.6 | **10.4+** |
| TensorRT-LLM | 0.7 | **0.13+** |
| PyTorch | 2.0 | **2.3+** (FlashAttn 2 完整) / 2.4+ (FA-3 SDPA 路径) |
| Transformer Engine (TE) | 0.7 | 1.10+ |
| FlashAttention | v2 | **v3** (sm_90a 专属) |
| vLLM | 0.2 | **0.6+** |
| Triton (lang) | 2.0 | **3.0+** (TMA/WGMMA 自动) |
| NCCL | 2.18 | 2.22+ |

容器：`nvcr.io/nvidia/pytorch:24.10-py3` 是 2025 年起手好选择（CUDA 12.6 + PyTorch 2.5 + TE 1.10）。

## 推理实用配置模板

### 单卡 H100 SXM5 80GB，7B-13B LLM

```bash
# FP8 + vLLM 一行
vllm serve meta-llama/Llama-3-8B-Instruct \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 --max-num-seqs 256 \
    --kv-cache-dtype fp8_e4m3 --enable-prefix-caching \
    --enable-chunked-prefill

# 或 TRT-LLM
trtllm-build --checkpoint_dir ckpt/ --output_dir engine/ \
    --gemm_plugin float16 --gpt_attention_plugin float16 \
    --use_fp8_context_fmha enable --kv_cache_type=fp8 \
    --paged_kv_cache enable --remove_input_padding enable \
    --max_batch_size 256 --max_input_len 4096 --max_output_len 2048
```

### 单节点 8 卡 H100 SXM5，70B LLM

```bash
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 --max-num-seqs 128 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching --enable-chunked-prefill \
    --speculative-model meta-llama/Llama-3-8B-Instruct \
    --num-speculative-tokens 4
```

NVSwitch 必备 prereq：
```bash
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start  nvidia-fabricmanager
nvidia-smi nvlink --status -i 0 | head -5
```

### CNN/ViT 推理（TRT FP8）

```bash
trtexec --onnx=vit_l_16.onnx --fp8 --fp16 \
    --shapes=x:64x3x224x224 \
    --useCudaGraph --builderOptimizationLevel=5 \
    --saveEngine=vit_l_h100_fp8.plan
```

## 调优速记

1. **必开**：
   - `nvidia-smi -pm 1`
   - Fabric Manager
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - `CUDA_DEVICE_MAX_CONNECTIONS=32`
   - `NCCL_P2P_LEVEL=NVL`
2. **LLM 必开**：FlashAttention（默认 SDPA 即可）、PagedAttention、prefix caching、continuous batching
3. **量化选**：FP8 first，掉精度再 W4A16 (AWQ/GPTQ)；KV cache FP8 几乎免费
4. **NVENC=0 的坑**：H100 没有视频编码硬核 —— 做视频生成 (Sora-like) 推理时编码要落到 CPU 或独立 NVENC 卡（如 L4）
5. **MIG 推荐切法**：小服务 7×1g.10gb；中规模 3×2g.20gb；大服务全卡
6. **多卡 TP**：单节点 NVLink 必走 NVSwitch 才能 8 卡 all-reduce 满速；用 nccl-tests 验证 ≥ 350 GB/s
7. **H100 PCIe 不要做 TP=8** —— PCIe Gen5 跨卡 P2P 远不如 NVLink，会瓶颈

## 已知坑 / 陷阱

| 坑 | 现象 | 解 |
|---|---|---|
| FP8 amax 没 warmup | 第一 batch NaN/低精度 | TE `fp8_autocast` 配 amax_history_len ≥ 16，warmup 50 step |
| Confidential Compute 开了忘关 | 性能掉 5-10% | `nvidia-smi conf-compute --get-state` 看，按需关 |
| H800 当 H100 用 | 多卡 TP NCCL 慢 | 检查 `nvidia-smi nvlink -s`，H800 单 link 带宽减半 |
| 跨 NUMA 跑多卡 | scaling 差 | `numactl --cpunodebind` + `--gres=gpu:8` 拓扑感知 |
| H100 PCIe 跑 NVLink Bridge 没插稳 | P2P 失败 | `nvidia-smi topo -p2p w` 验证 |
| `flash_attn` 装错版本 | 用 FA-1 慢 path | 装 `flash-attn>=2.5`（FA-2）或 `>=3` |
| TRT-LLM build 与运行 SM arch 不匹配 | engine load failed | build 时 `--max_workers=N` 多 GPU 并行编译，每个 SM arch 单独 build |
| cuDNN 8.x + Hopper | conv FP8 路径走不到 | 升 cuDNN 9.x |
| `torch.compile` + Hopper TMA | 默认未启用 | PyTorch 2.4+ + Triton 3.0+ 自动用 |

## 实测命令

```bash
# 卡况
nvidia-smi -q -i 0 | head -60
nvidia-smi nvlink --status -i 0
nvidia-smi topo -m

# 微基线
# NCCL 8 卡 all-reduce
cd nccl-tests/build && ./all_reduce_perf -b 8 -e 4G -f 2 -g 8
# 期望: H100 SXM5 + NVSwitch3，1GB msg ≥ 380 GB/s

# HBM 带宽（Hopper roofline 验证）
# 用 NVIDIA bandwidthTest sample 或 cuda-samples bandwidthTest
./bandwidthTest --memory=pinned --mode=range --start=1073741824 --end=8589934592 --increment=1073741824

# FP8 / FlashAttention v3 sanity
python -c "
import torch
from torch.nn.functional import scaled_dot_product_attention
q = torch.randn(2,32,2048,128, device='cuda', dtype=torch.bfloat16)
k = q.clone(); v = q.clone()
o = scaled_dot_product_attention(q,k,v,is_causal=True)
print(o.shape, o.dtype)
"
```

## DCGM / 监控指标重点

```bash
dcgmi dmon -e 1001,1002,1003,1004,1005,1009,449,450 -c 60
# 1001 GR Active%, 1002 SM Active%, 1003 SM Occupancy
# 1004 Tensor Active%（≥ 60% 才说明真在用 Tensor Core）
# 1005 DRAM Active%
# 1009 FP/INT Active% （Hopper 拆分 FP16/FP8/INT8）
# 449/450 NVLink RX/TX bytes
```

LLM decode 阶段，DRAM Active 通常 80%+，Tensor Active 30-50% 才正常（memory bound）。

## 适合 / 不适合

| workload | 适合度 |
|---|---|
| LLM 7B-405B 推理（含 MoE） | ✅✅✅ 全 |
| LLM 训练 | ✅✅✅ |
| ViT/CNN 大 batch 推理 | ✅✅✅ |
| 视频生成模型 (Sora/CogVideoX) 推理 | ✅✅ (但视频编码要 L4/CPU) |
| 边缘 / 低功耗 | ❌ 700W |
| 老 CUDA 11 应用 | ⚠️ FP8 用不上 |
| 极致 LLM throughput (FP4) | ⚠️ Hopper 没有 FP4 → 找 B200 |

## 后继 / 互补

- **B200/B100** —— 下一代，FP4 加速、HBM3e 192GB
- **H200** —— H100 的 HBM3e refresh，**LLM 推理直接换** ROI 极高
- **L4 / L40S** —— 同代不同档，FP8 + 24GB / 48GB，适合更小服务
- **A100** —— Ampere，无 FP8 / TMA，但 80GB HBM2e、有 NVLink，仍然能干

## 参考

- NVIDIA H100 Architecture Whitepaper（官方）
- NVIDIA Hopper Tuning Guide
- 本仓库 [`../../skills/nv-infer-llm-attention/SKILL.md`](../../skills/nv-infer-llm-attention/SKILL.md)
- 本仓库 [`../../skills/nv-infer-multigpu/SKILL.md`](../../skills/nv-infer-multigpu/SKILL.md)
- 本仓库 [`../../skills/nv-infer-precision/SKILL.md`](../../skills/nv-infer-precision/SKILL.md)
