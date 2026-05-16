---
name: ascend-infer-io-data
description: Ascend NPU 推理的数据 / IO pipeline——DVPP 视频/JPEG 解码、AIPP 片上预处理、torch_npu DataLoader (pin_memory)、MindData 数据 pipeline、模型加载加速、对象存储 (OBS/S3) 读取、tokenizer 加速、镜像预热。当用户问"昇腾 DataLoader"、"NPU 数据 pipeline"、"DVPP/AIPP"、"NPU JPEG"、"MindData"、"NPU 模型加载慢"、"NPU CPU 喂不动"、"OBS 慢" 时使用。
---

# Ascend I/O & Data Pipeline

> 与 nv-infer-io-data 平行。Ascend 上 IO/数据 pipeline 的核心策略：**DVPP + AIPP 让视频/图像预处理全片上化**，CPU 几乎只做调度。

## 触发场景速判

| 现象 | 大概率原因 |
|---|---|
| AI Core 利用率 < 70%，CPU 100% | 预处理 / decode 在 CPU 上 |
| AI Core 利用率锯齿状 | DataLoader prefetch 不够 |
| 首请求 200ms+ | tokenizer / model load 冷启动 |
| 服务启动几分钟 | 大 OM 从远端加载 |
| LLM TTFT 高但 model 在 NPU | tokenizer 单线程瓶颈 |
| PCIe RX 持续高 | host → NPU 拷贝瓶颈 |

## CNN / ViT pipeline

### 1. PyTorch DataLoader 黄金配置（torch_npu）

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,                # 必开
    persistent_workers=True,
    prefetch_factor=4,
)

for x, _ in loader:
    x = x.to('npu', non_blocking=True)
    with torch.inference_mode(): out = model(x)
```

`pin_memory=True` + `non_blocking=True` 配合让 H2D 异步。

### 2. DVPP + AIPP（强烈推荐生产）

详见 `ascend-infer-cnn-vit`。要点：
- **DVPP** 在 NPU 上做视频/JPEG 解码 → 输出 YUV420SP 到 NPU 显存
- **AIPP** 嵌入 OM，运行时直接吃 YUV → resize/CSC/normalize 一次完成
- CPU 几乎只做 RTSP/RTMP/file 读取

```bash
# 验证视频解码占用
npu-smi info -t dvpp -i 0 -c 0     # 部分 SKU 暴露 DVPP 利用率
```

### 3. JPEG decode（图片场景）

```python
import acl
# 直接调 acldvppJpegDecodeAsync 把 JPEG bytes 解码到 NPU 显存
# 或用 MindData 的 vision.Decode + .map(device_target="Ascend")
```

300V Pro 上单卡 JPEG decode 千张/秒级；910B 上 DVPP 较弱（910B 主算训练，视频 IP 不是主打）。

### 4. MindData (MindSpore 风格 pipeline)

类似 DALI 的高层 API：

```python
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

dataset = ds.ImageFolderDataset("./imgs")
dataset = dataset.map(operations=[
    vision.Decode(),                          # JPEG decode (NPU DVPP)
    vision.Resize((256, 256)),
    vision.CenterCrop((224, 224)),
    vision.Normalize(mean=[0.485*255, ...], std=[0.229*255, ...]),
    vision.HWC2CHW(),
], num_parallel_workers=8, python_multiprocessing=False)

dataset = dataset.batch(64).repeat()
```

`python_multiprocessing=False` 让 op 在 device-side worker 上跑，不走 Python multiprocess。

## LLM 数据 pipeline

### 1. Tokenizer 加速

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
```

MindIE-LLM 内部默认用 fast tokenizer。生产高 QPS 配 tokenizer worker pool：

```json
"ScheduleConfig": {
  "tokenizerWorkers": 8
}
```

### 2. Prompt 长度桶化

变长 prompt → 编译 cache 失效抖动。
- MindIE-LLM 内部 paged + remove_input_padding，已自动处理
- 自写 server 才需要考虑（不推荐）

### 3. 二进制 / preprocessed input

客户端预 tokenize（传 `input_ids` 而非字符串）：
- 省 server tokenize 开销
- 长 context 时 JSON payload 大 → 用 gRPC + proto

## 模型加载加速

70B FP16 = 140 GB，从普通磁盘读 5-10 分钟；910B 上同理。

加速：
- **NVMe local cache**：节点本地 NVMe 挂 cache 目录
- **OBS / S3 流式**：MindIE-LLM 支持远端模型路径（隐式 mmap），但首次加载慢
- **SafeTensors**：mmap + 多线程 shard load，加载快 2-5×
- **预热 daemon**：节点空闲时 prefetch 常用 OM 到本地

```bash
# 提前 cache OM
cp -r /obs_mount/models/Qwen3-8B-om/ /var/cache/mindie/models/
# 启动 service 指向本地
mindieservice_daemon --config qwen3.json   # modelWeightPath: /var/cache/...
```

## OBS / 对象存储读取

Atlas 集群常配 Huawei OBS（S3 兼容）：

```python
# obs-python-sdk 多线程读
from obs import ObsClient
client = ObsClient(access_key_id=..., secret_access_key=..., server=...)
# 同样适用 boto3 + endpoint=obs.cn-east-3.myhuaweicloud.com
```

并发上限通常受网卡 + OBS 限速；单 obs-client 跑得 ~100-500 MB/s。

## H2D / D2H 高效写法

```python
# ✅ pinned + non_blocking
x = torch.empty(shape, pin_memory=True)
x.copy_(some_data)
x_npu = x.to('npu', non_blocking=True)

# ❌ 不 pinned，sync 拷贝
x_npu = torch.tensor(some_data).npu()

# ❌ 循环里 100 次小 H2D
for x in items: ...to('npu')...
```

## Checkpoint / 周期性 IO

推理服务一般无 ckpt。但有：
- Prometheus metrics → 异步、独立线程
- Request log → 采样写、不要 sync flush
- msprof trace → 离线开关，生产关

## 镜像 / 启动预热

ascendhub.huawei.com 镜像通常 5-15 GB（CANN + PyTorch + 工具链）。
- K8s `imagePullPolicy=IfNotPresent`
- Lazy pulling（nydus / stargz）
- 模型权重不打进镜像，挂 PVC

## 验证

```bash
# 1. CPU 瓶颈检测
top -H -p $(pgrep -f mindieservice_daemon)
# 某线程 100% → 那里是瓶颈

# 2. PCIe 流量
# 通过 npu-smi 间接看 host ↔ NPU 流量（部分 SKU 暴露）

# 3. NPU 是否真在等
msprof --output=./prof --application="..." --task-time=on
# 时间线 NPU 大段 idle 配 CPU 忙 → IO 瓶颈
```

## 常见数据 pipeline 模板

### CNN 服务（300V Pro，视频）

```
RTSP/RTMP
   ↓ DVPP video decode (NPU)
YUV420SP frames in NPU mem
   ↓ AIPP (resize/CSC/normalize)
RGB FP16
   ↓ OM 推理
results → emit via gRPC
```

CPU 占用：< 20%（只做 RTSP 协议解析 + emit）

### LLM 服务（910B）

```
HTTP/gRPC request
   ↓ tokenizer (CPU pool, fast tokenizer)
input_ids
   ↓ MindIE-LLM (NPU)
output_ids
   ↓ detokenize (CPU)
HTTP response
```

CPU 占用：tokenizer 占大头（高 QPS 时调多 worker）

### VLM 服务（910B，MiniCPM-V）

```
HTTP request (含 image base64)
   ↓ base64 decode (CPU)
JPEG bytes
   ↓ DVPP JPEG decode (NPU)
RGB image
   ↓ vision encoder (SigLIP) (NPU)
image embeddings
   ↓ LLM prefill + decode (NPU)
output tokens
```

## 与 NV 对应

| NV | Ascend |
|---|---|
| DataLoader pin_memory=True | torch_npu pin_memory=True |
| DALI | AIPP + DVPP + MindData |
| nvJPEG / nvImageCodec | DVPP JPEG decode (acldvppJpegD*) |
| NVDEC | DVPP video decode (aclvdec*) |
| GPUDirect Storage (kvikio) | （Ascend 上无完全对等；通过 NVMe local cache + mmap） |
| tensorizer (vLLM) | MindIE-LLM 远端 model load |
| boto3 / s3transfer | obs-python-sdk |

## 反模式

- `pin_memory=False` + `non_blocking=True` → non_blocking 无效
- 视频流走 CPU FFmpeg/OpenCV → DVPP 闲置
- DataLoader `num_workers=0` → 主线程做预处理
- 把 JPEG 在 CPU decode 后再 H2D → 浪费 DVPP IP
- 模型权重每次启动从 OBS 拉 → 节点本地 cache
- 容器 `--shm-size=64m` → DataLoader worker IPC 阻塞
- LLM tokenizer slow 版本 → CPU 单核 100% 限速

## 参考

- DVPP API（CANN 文档）
- AIPP 配置参考
- MindData 教程
- `references/hardware/huawei-atlas-300v-pro.md`
- `references/hardware/huawei-atlas-910b.md`
- `ascend-infer-cnn-vit` —— DVPP + AIPP 完整 pipeline
