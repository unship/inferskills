---
name: ascend-infer-cnn-vit
description: Ascend NPU 上 CNN / ViT 推理专项——DVPP (视频/图像编解码硬核)、AIPP (片上 resize/CSC/normalize)、ATC INT8 编译、NZ format 自动转换、YOLO/ResNet/EfficientNet/ViT/CLIP/Swin 在 310P/910B 上的部署。当用户问"昇腾 YOLO"、"NPU CNN"、"DVPP"、"AIPP"、"Ascend ViT"、"视频推理"、"NPU 图像预处理"时使用。
---

# Ascend CNN / ViT Inference

> 与 nv-infer-cnn-vit 平行。Ascend 的招牌：**DVPP + AIPP 把视频解码 + 预处理全片上化**，CPU 几乎零占用。
> 模型部署路径：PyTorch → ONNX → ATC + AIPP → OM → MindIE Service / 自写 AscendCL。

## 通用 checklist

1. PyTorch / TF 模型导 ONNX (opset 17)
2. `atc --precision_mode=allow_mix_precision`（INT8/FP16 混搭）
3. 输入预处理用 **AIPP**（永远不要让 CPU 做 resize/normalize）
4. 视频用 **DVPP** 解码到 NPU 显存，零拷贝接 AIPP
5. 动态分辨率：`--dynamic_dims` 枚举几个固定值
6. 跑 `msprof` 验证 AI Core 利用率 > 70%

## DVPP（视频/图像硬核解码）

Ascend NPU 自带视频编解码 IP（300V Pro 264 路 1080p，910B 较少但够用）。

**核心 API**：`acldvppVdec*` (视频)、`acldvppVencode*` (编码)、`acldvppJpegD*` (JPEG decode)、`acldvppVpc*` (resize/crop/CSC)

支持 codec：
- H.264 / H.265 / VP9 / AVS2 / AVS3 / MPEG-4
- JPEG / PNG (decode)
- 输出格式：YUV420SP_U8 / YUV420SP_U16 / RGB888

### 直接调（C++）

```cpp
aclvdecChannelDesc *channel;
aclvdecCreateChannel(channel);
// 设置 channel: 输入 codec、输出分辨率、回调
aclvdecSendFrame(channel, in_stream_buf, ...);   // 异步
// 回调里取 output_buf (YUV420SP)
```

### Python (pyACL / MindX)

通常通过 MindX SDK / aclpy 包装；推荐 MindX VideoAnalysis 这类高层 SDK 串 pipeline。

## AIPP（片上图像预处理）

**最重要的 Ascend CNN/ViT 卖点**：把 DVPP decode 出的 YUV → resize / crop / CSC（YUV→RGB）/ mean/std normalize / dtype 转换 一次完成在 NPU 上，**直接喂模型**。

ATC 编译时通过 `--insert_op_conf=aipp.cfg` 嵌入。

### AIPP 配置样板

`aipp.cfg`：
```
aipp_op {
    aipp_mode: static
    related_input_rank: 0
    input_format: YUV420SP_U8
    src_image_size_w: 1920
    src_image_size_h: 1080
    csc_switch: true
    rbuv_swap_switch: false

    # YUV → RGB matrix (BT.601)
    matrix_r0c0: 256
    matrix_r0c1: 0
    matrix_r0c2: 359
    matrix_r1c0: 256
    matrix_r1c1: -88
    matrix_r1c2: -183
    matrix_r2c0: 256
    matrix_r2c1: 454
    matrix_r2c2: 0
    input_bias_0: 0
    input_bias_1: 128
    input_bias_2: 128

    # crop + resize
    crop: true
    load_start_pos_h: 0
    load_start_pos_w: 0
    crop_size_w: 1920
    crop_size_h: 1080

    # normalize (img - mean) * var_reci
    mean_chn_0: 124
    mean_chn_1: 117
    mean_chn_2: 104
    var_reci_chn_0: 0.0171247538316637       # 1/(58.395)
    var_reci_chn_1: 0.0175070028011204
    var_reci_chn_2: 0.0174291938997821
}
```

**动态 AIPP**（运行时设参，输入 size 不同）：`aipp_mode: dynamic` + 运行时 `aclmdlSetDynamicAIPP`。

## YOLO 系列部署（300V Pro 视频 AI）

完整 pipeline：

```bash
# 1. 导出 ONNX（Ultralytics，关 NMS plugin —— Ascend 自带或 client 做）
yolo export model=yolov11l.pt format=onnx opset=17 nms=False

# 2. ATC + AIPP（YUV 输入，letterbox+normalize 都在 AIPP）
atc --model=yolov11l.onnx \
    --framework=5 \
    --soc_version=Ascend310P3 \
    --output=yolov11l_int8_om \
    --input_format=NCHW \
    --input_shape="images:16,3,640,640" \
    --insert_op_conf=aipp_yolo.cfg \
    --precision_mode=allow_mix_precision \
    --calibration_data_dir=./calib_yuv/ \
    --auto_tune_mode="RL,GA"

# 3. 用 MindX SDK 串视频 pipeline，DVPP→AIPP→OM→NMS→tracker
```

**典型数字**（300V Pro，YOLOv8s INT8 + DVPP + AIPP）：
- 264 路 1080p@30 同时解码 + 检测，CPU < 20%

## CLIP / ViT 部署

```bash
# CLIP ViT-L/14 BF16（910B 推荐）
atc --model=clip_vit_l14_visual.onnx \
    --framework=5 \
    --soc_version=Ascend910B3 \
    --output=clip_vit_l14_om \
    --input_shape="images:64,3,224,224" \
    --precision_mode=allow_fp32_to_bf16 \
    --auto_tune_mode="RL,GA"
```

ViT 在 Ascend 上：
- attention 自动走 `aclnnFlashAttentionScore`（CANN 7.0+）
- patch embed = stride conv，ATC 自动选 Cube path
- 大 batch (64-256) 时 AI Core 利用充分

## ResNet / EfficientNet / MobileNet 套路

```bash
atc --model=resnet50.onnx \
    --framework=5 \
    --soc_version=Ascend910B3 \
    --output=resnet50_int8_om \
    --input_shape="input:16,3,224,224" \
    --insert_op_conf=aipp_imagenet.cfg \
    --precision_mode=allow_mix_precision \
    --calibration_data_dir=./calib_jpg/
```

要点：
- INT8 是默认选择（精度损失 < 1%）
- depthwise conv (MobileNet) 在 Cube 单元利用率低 → batch 拉大
- ATC 自动 fuse Conv-BN-ReLU

## Swin Transformer

windowed attention 的 custom mask 在 CANN 8.0+ 通过 `aclnnFlashAttentionScore` 的 `sparse_mode=2/3`（causal/window）支持有限。复杂 cyclic shift mask 可能 fallback math attention path（慢）。

应对：
- 用 MindIE-SD 的 vision branch（如有适配）
- 或将 windowed attention 改写成 chunked attention + 自定义 mask（需开发）

## 动态分辨率

```bash
atc --model=yolov8.onnx --framework=5 --soc_version=Ascend310P3 \
    --input_shape="images:-1,3,-1,-1" \
    --dynamic_dims="1,640,640;1,320,320;8,640,640;16,640,640" \
    --output=yolov8_dyn_om
```

运行时通过 `aclmdlSetDynamicHWSize` 选 shape。

## 视频 pipeline 最佳实践（MindX SDK 或 aclpy）

```
RTSP/RTMP/file
   ↓ DVPP 硬解 (NPU)
YUV420SP frame in NPU mem
   ↓ AIPP (resize/CSC/norm)
RGB FP16 in NPU mem
   ↓ OM 推理 (Cube + FA)
detection / classification logits
   ↓ post (NMS / sort)
result → CPU only for final emit
```

**关键**：YUV/RGB 始终在 NPU 显存，**永远不 D2H**。

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| AIPP CSC 矩阵设错 | 颜色反 / 偏 | BT.601 vs BT.709 区分 |
| YOLO 不做 letterbox | mAP 大跳 | AIPP 不直接支持 letterbox，外部 padding 或客户端预处理 |
| 输入 size 不是 stride 倍数 | 推理输出 shape 不对 | YOLO 必须 32 倍数 |
| 用 cv2.resize 在 CPU 跑 | NPU 闲 | DVPP+AIPP 全片上 |
| ONNX 含 EfficientNMS 自定义节点 | ATC 不支持 | export 时 nms=False，client/SDK 做 NMS |
| Swin window attention 走慢 path | AI Core 利用率低 | 改写 attention 或限制使用 |
| 视频码流不在支持列表（如 VP9 Profile2） | DVPP 解码失败 | CPU decode fallback，性能塌 |
| INT8 calibration 仅用 dataset.cifar | 实际 domain 不对，mAP 掉 | 用线上抽样校准 |
| BF16 在 910A | fallback FP32 | 910A 用 FP16，910B 才有 BF16 |
| 动态 shape 桶太多 | OM 巨大、首次编译慢 | 桶 ≤ 8 |

## 与 NV 对应

| NV | Ascend |
|---|---|
| channels_last (NHWC) | NZ format（自动） |
| cuDNN benchmark | ATC auto_tune (`RL,GA`) |
| Conv-BN-ReLU fusion | ATC 自动 fusion |
| Winograd conv | CANN 内部选 algo（不暴露） |
| TensorRT INT8 calibration | ATC `--calibration_data_dir` / AMCT |
| NVDEC | DVPP video decode |
| nvJPEG | DVPP JPEG decode |
| DALI | AIPP + DVPP + MindX SDK |
| FlashAttention | aclnnFlashAttentionScore |
| `torch.compile` channels_last | ATC `--input_format=NCHW` + 自动 NZ |

## 反模式

- AIPP 不开，让 CPU 做预处理 → NPU 干等，等于浪费 70% 算力
- 视频解码用 ffmpeg / OpenCV → DVPP 是免费的，不用白不用
- 把 YUV 数据 D2H 给 CPU 看一眼再上 NPU → PCIe 瓶颈
- Swin 期望 FlashAttention 自动加速 → 看 sparse_mode 是否支持
- 300V Pro 上跑 ViT-H/14 batch=64 → LPDDR 带宽不够，结果慢
- 同 OM 跨 SoC 复用 → 必须重 build

## 参考

- CANN AIPP 配置参考、DVPP API
- MindX SDK 视频示例
- `references/models/yolo-series.md`
- `references/models/clip-siglip.md`
- `references/hardware/huawei-atlas-300v-pro.md`
- `references/hardware/huawei-atlas-910b.md`
