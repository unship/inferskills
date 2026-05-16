---
name: ascend-infer-precision
description: Ascend NPU 推理量化与精度——FP16/BF16/INT8 W8A8/W4A16、MindIE Quantizer、ATC calibration、SmoothQuant on Ascend、KV cache INT8、混合精度策略、精度回归。当用户问"昇腾量化"、"Ascend INT8"、"W4A16"、"AWQ on NPU"、"MindIE 量化"、"精度损失"、"calibration"时使用。
---

# Ascend Precision & Quantization

> 与 nv-infer-precision 平行。**没有 FP8 / FP4** —— 910B/310P 硬件不支持。
> 主线选项：**FP16 / BF16 / INT8 W8A8 / W4A16 / KV INT8**。

## 精度选项一览

| 格式 | 用途 | 支持卡 |
|---|---|---|
| FP32 | 调试基线 / 个别敏感层 | 全系（性能差，仅 Vector unit） |
| **BF16** | LLM / ViT 推理（910B+，数值稳） | 910B / 910B3 / 910B4（**910A 不支持**） |
| **FP16** | CNN/ViT 推理默认（Cube 全速） | 全系 |
| **INT8 W8A8** | CNN/ViT/LLM 推理（Cube INT8 全速） | 全系 |
| **W4A16** | LLM 显存压缩（类 AWQ/GPTQ） | 910B (MindIE-LLM) |
| INT4 | 实验性、个别 op；MindIE-LLM weight only | 910B 有限 |
| **KV INT8** | LLM 长上下文显存压缩 | 910B (MindIE-LLM) |
| FP8 / FP4 | **不支持** | — |

## 决策树

```
模型 + 目标：
├─ CNN/ViT 离线 batch / 视频推理     → INT8 W8A8 (ATC 编译路径)
├─ ViT-L/H 大模型，高精度要求         → BF16 (910B+)
├─ LLM 单卡，模型 < HBM/2              → BF16 / FP16
├─ LLM 单卡，模型 ≈ HBM 大小           → INT8 W8A8 (MindIE)
├─ LLM 显存紧 / 70B+ 单卡试           → W4A16 (MindIE)
├─ LLM 长上下文 32K+                  → KV INT8 + 任意权重量化
└─ 文生图 / SD                        → BF16 (MindIE-SD)
```

## CNN / ViT INT8 calibration（ATC 路径）

ATC 内置 INT8 量化工具：

```bash
# 1) 标准 calibration（生成 OM 时自动量化）
atc --model=resnet50.onnx \
    --framework=5 \
    --soc_version=Ascend910B3 \
    --output=resnet50_int8_om \
    --input_format=NCHW \
    --input_shape="input:16,3,224,224" \
    --insert_op_conf=aipp.cfg \
    --precision_mode=allow_mix_precision \
    --calibration_data_dir=./calib_imgs/        # 200-1000 张
```

或用独立的 **AMCT (Ascend Model Compression Toolkit)**，更灵活：

```bash
# 训练后量化（PTQ）
python -m amct_pytorch.calibration \
    --model_path resnet50.pt \
    --calib_data_dir calib_imgs/ \
    --output_dir resnet50_amct/ \
    --quant_type W8A8
```

精度校验：
```bash
# 跑 ImageNet val
python evaluate_om.py --model=resnet50_int8_om.om --data=imagenet_val/
# 期望: top-1 掉 < 0.5%; top-5 掉 < 0.3%
```

## LLM 量化（MindIE 路径）

### W8A8 SmoothQuant 风格

```bash
# MindIE-LLM 量化工具（CANN 8.0+）
msmodelslim --model_path Qwen3-8B/ \
    --quant_type w8a8 \
    --calib_data c4_calib.jsonl \
    --calib_size 512 \
    --output_dir Qwen3-8B-w8a8/
```

转 OM / MindIE 时指定量化目录：
```bash
mindie-cli convert --model_path Qwen3-8B-w8a8/ \
    --output_path Qwen3_8b_w8a8_om/ \
    --soc_version Ascend910B3 \
    --quant w8a8 \
    --enable_paged_kv true
```

### W4A16 (类 AWQ)

```bash
msmodelslim --model_path Qwen3-8B/ \
    --quant_type w4a16 \
    --group_size 128 \
    --calib_data c4_calib.jsonl \
    --output_dir Qwen3-8B-w4a16/
```

`group_size=128` 是标准选项（与 NV 侧 AWQ/GPTQ 一致）。

### KV Cache INT8

MindIE-LLM 配置中开启：
```json
{
  "kvCacheDtype": "int8",
  "enableKvScalesCalibration": true,
  "calibSampleCount": 200
}
```

或在转换时：
```bash
mindie-cli convert ... --quant w8a8 --kv_quant int8
```

## 精度回归（必做）

量化后跑相同 prompt 集做精度对比：

```bash
# LLM 上跑 MMLU / GSM8K / HumanEval
lm-eval --model hf --model_args pretrained=Qwen3-8B-w8a8/ \
        --tasks mmlu,gsm8k --device npu:0

# CNN/ViT 上跑 ImageNet / COCO
python eval_om.py --om resnet50_int8_om.om --dataset imagenet_val/
```

阈值（通用）：
- CNN top-1 掉 < 0.5%
- LLM MMLU 掉 < 1.0%
- COCO mAP 掉 < 1.0
- Qwen3 thinking mode 上 GSM8K 容忍稍宽松（CoT 累积误差）

掉太多：找罪魁层 → 关键层 keep FP16，混合精度（AMCT 工具支持 layer-level 控制）。

## 与 NV 量化的差异

| 维度 | NVIDIA | Ascend |
|---|---|---|
| W8A8 工具 | TensorRT INT8 / nvidia-modelopt / llmcompressor | ATC INT8 / AMCT / msmodelslim |
| W4A16 | AutoAWQ / AutoGPTQ / Marlin/Machete kernel | msmodelslim W4A16 / MindIE 自带 |
| FP8 | Transformer Engine / TRT-LLM | **不支持** |
| FP4 (Blackwell) | NVFP4 / MXFP4 | **不支持** |
| 2:4 稀疏 | cuSPARSELt | 硬件无对应 |
| Calibration 数据 | 通用 imagenet/c4 | 同；MindIE 推荐自家格式但兼容 jsonl |

**NV 量化文件不能直接 load 到 Ascend**：
- AWQ `.safetensors` 的 scales / zero points 编码方式不同
- TRT engine 完全不通用
- 必须用 Ascend 工具重做

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 量化数据集与线上分布不一致 | mAP / PPL 大跳 | 用线上抽样校准 |
| `force_fp16` 强转后某层溢出 | NaN / 数值乱 | `allow_mix_precision` 或 keep_fp32 配置 |
| W4A16 group_size 不是 128 | 部分 kernel fallback 慢 | 固定 group_size=128 |
| BF16 在 910A 上 | fallback FP32，慢 | 910A 用 FP16；910B 才有 BF16 |
| KV INT8 但没收 scale | decode 早期数值乱 | 跑 warmup 100+ step，启用 KvScalesCalibration |
| AMCT 跑出 op 不支持 | 算子未在量化白名单 | 升 CANN / 加 layer skip 列表 |
| 直接 import NV AWQ ckpt | 加载报错 | msmodelslim 重做 |
| Qwen3 thinking + INT4 | reasoning 任务掉 3-5% | thinking 关键场景考虑 W8A8 |

## 混合精度策略

```bash
# 关键层留 FP16，其余 INT8
atc ... --precision_mode=allow_mix_precision \
        --modify_mixlist=keep_fp16_list.cfg
```

`keep_fp16_list.cfg` 示例：
```
[mixlist]
fp16_op_list=Softmax,LayerNorm,RMSNorm
```

LLM 上 attention softmax + LayerNorm 通常需留高精度；模型主体 INT8/W4A16 即可。

## 反模式

- 量化完不做精度回归 → 上线被用户发现
- 用 1-2 张图 calibration → scale 全错
- 跨 SoC 复用量化模型 → 不一定兼容（重新 build OM）
- 期待 Ascend 上有 FP8 path → 没有，别浪费时间找
- 把 NV AWQ.pt 直接搬到 MindIE → 失败，要 msmodelslim 重做
- INT4 在 CNN 上 → 通常 INT8 已足够，不要为 INT4 牺牲精度

## 参考

- AMCT 文档 / msmodelslim 文档
- MindIE-LLM 量化指南
- `references/models/qwen3-8b.md` —— LLM 量化损失参考
- `references/models/yolo-series.md` —— CNN INT8 数字
