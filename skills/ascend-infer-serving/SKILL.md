---
name: ascend-infer-serving
description: Ascend NPU 推理服务化——MindIE Service、MindX SDK、Triton 通过 atb backend、HTTP/gRPC、dynamic batching、autoscaling、K8s + Ascend device plugin、Prometheus 监控、warmup、SLO、tail latency 控制。当用户问"昇腾部署"、"MindIE Service"、"MindX SDK"、"昇腾 K8s"、"NPU autoscaling"、"NPU 服务化" 时使用。
---

# Ascend Serving Stack

> 与 nv-infer-serving 平行。Ascend 上服务化的核心是 **MindIE Service**（LLM 与通用模型）+ **MindX SDK**（视频/视觉应用）。
> 多数情况下不需要写代码，做好配置即可。

## 选型

| 场景 | 推荐 server |
|---|---|
| LLM / VLM 推理 | **MindIE Service**（LLM engine 子模块） |
| CNN / ViT 通用模型 | **MindIE Service**（Inference engine） |
| 视频分析 / 多模型 pipeline | **MindX SDK** (含 VideoAnalysis) |
| 多模型 ensemble (像 Triton) | MindIE Service + 自定义 client orchestration / 或 Triton + atb backend (CANN 8.0+) |
| 嵌入式 / 边缘小集群 | MindX 系列（DLI / Boost） |

## MindIE Service —— LLM 部署

见 `ascend-infer-llm-stack` 的完整配置 + 启动模板。

关键监控端点：
```bash
curl http://localhost:1025/metrics      # Prometheus 格式
# vllm 等价的：
#   mindie:time_to_first_token_seconds
#   mindie:time_per_output_token_seconds
#   mindie:num_requests_running
#   mindie:num_requests_waiting
#   mindie:kv_cache_usage_perc
```

## MindIE Service —— 通用模型（CNN/ViT）

```json
{
  "ServerConfig": {
    "ipAddress": "0.0.0.0",
    "httpPort": 1025,
    "managementPort": 1026
  },
  "BackendConfig": {
    "backendName": "mindieservice_inference_engine",
    "ModelDeployConfig": {
      "ModelConfig": [{
        "modelName": "yolov11l",
        "modelWeightPath": "/models/yolov11l_om/",
        "worldSize": 1,
        "instanceNum": 4,                  // 同卡多实例（vNPU 切分或共享）
        "backendType": "atb"
      }]
    },
    "ScheduleConfig": {
      "maxBatchSize": 32,
      "maxQueueDelayMicroseconds": 5000,
      "preferredBatchSizes": [8, 16, 32]   // 类似 Triton dynamic batching
    }
  }
}
```

启动：
```bash
mindieservice_daemon --config /etc/mindie/yolov11_service.json
```

## Dynamic Batching 参数

类比 Triton，关键 trade-off：

```json
{
  "maxBatchSize": 64,
  "preferredBatchSizes": [16, 32, 64],
  "maxQueueDelayMicroseconds": 5000        // ≤ SLO 的 10-20%
}
```

`maxQueueDelay` 越大 batch 越满 → 吞吐高、延迟差。设为 SLO 的 10-20%（p99 = 50ms → queue delay ≤ 5-10ms）。

## MindX SDK（视频分析 pipeline）

适合：多模型串成视频 AI 流水线（解码 → 检测 → 跟踪 → ReID → 行为分析）。

```python
# pipeline.json 描述 GStreamer-like 流
pipeline = {
    "videoanalysis": {
        "stream0": [
            { "props": {"file": "rtsp://cam1"}, "factory": "videodecode" },
            { "props": {"model": "yolov11l_om.om"}, "factory": "mxpi_modelinfer" },
            { "props": {"track_class": "boy,girl"}, "factory": "mxpi_motsimplesortV2" },
            { "factory": "mxpi_dataserialize" },
            { "factory": "appsink" }
        ]
    }
}
```

MindX 在底层用 DVPP + AIPP + OM 推理 + tracker，**CPU 几乎只调度**。

## Triton with atb backend（实验性）

CANN 8.0+ 提供 Triton Inference Server 的 atb backend 适配，让 NV/Ascend 混部统一接入 Triton：
```bash
tritonserver --backend-directory=/usr/local/lib/triton/backends \
             --model-repository=/triton_models/
# 模型 config.pbtxt 中 backend: "atb"
```

成熟度：迭代中，**主线生产仍推荐 MindIE Service**。

## Autoscaling

### 关键指标（不要用 AI Core util）

LLM continuous batching 下 AI Core util 永远 80%+，不适合做扩缩信号。

推荐：
- `mindie:num_requests_waiting > 0` 持续 N 秒 → 扩
- `mindie:time_to_first_token_seconds` p95 > SLO 持续 → 扩
- `mindie:kv_cache_usage_perc > 0.9` → 接近 KV 撑爆，扩

### K8s HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metrics:
- type: Pods
  pods:
    metric: { name: mindie_num_requests_waiting }
    target: { type: AverageValue, averageValue: "2" }
- type: Pods
  pods:
    metric: { name: mindie_time_to_first_token_seconds_p95 }
    target: { type: AverageValue, averageValue: "1.0" }
behavior:
  scaleDown:
    stabilizationWindowSeconds: 600
```

## Tail Latency / p99

NPU 上 p99 抖的常见因素：
| 来源 | 检测 | 修 |
|---|---|---|
| 首次 OM load + 编译缓存 miss | 重启后头几十请求慢 | warmup 跑遍代表 shape |
| AICPU fallback 串行化 | msprof aicpu 占比 | 升 CANN 消除 fallback |
| Dynamic shape 触发 op 再编译 | p99 周期性飙 | shape 桶化 + 缓存 |
| 邻居噪音（vNPU 共享） | 同卡多服务 | 切独立 vNPU 实例 |
| 温度降频 | `npu-smi info -t temp` > 80℃ | 散热 / 降功耗 |
| HCCL 抖动（多卡） | hccl_*.csv 时长抖 | RoCE PFC/ECN |
| Tokenizer 单线程（LLM） | Python CPU 满 | tokenizer pool 多 worker |

## Warmup 流程

```python
# Service 启动后、ready 之前
shapes_to_prime = [(1, 128), (1, 512), (1, 2048), (8, 128), (8, 512), (32, 128)]
for bs, seq in shapes_to_prime:
    _ = mindie_client.generate(dummy_input(bs, seq), max_new_tokens=8)
print("warmup done")
# K8s readiness probe 只在此后变 ready
```

## Prefix Cache（LLM 大杀器）

见 `ascend-infer-llm-stack` —— 服务化时几乎一定要开。

## 多实例 / 多模型一卡部署

策略：
1. **vNPU 切分** —— 硬件隔离（推荐多租户）
2. **同进程多 instance** —— `instanceNum: 4`（共享 HBM 与 AI Core 时间分片）
3. **进程级共卡** —— ASCEND_RT_VISIBLE_DEVICES + 多 server 实例

注意：910B 同卡跑 N 个 LLM 实例会互抢 HBM，**不推荐**；通用模型 (CNN/ViT) 共享 OK。

## SLO 拆解模板

每个端点写明：
- TTFT p50/p95/p99（LLM）
- TPOT p50/p95/p99（LLM）
- 端到端 latency p95
- 失败率 (5xx / OOM / timeout)
- 吞吐 (req/s 或 output tokens/s)
- AI Core / HBM 利用率

## 多机部署（跨节点）

```json
{
  "BackendConfig": {
    "ModelDeployConfig": {
      "ModelConfig": [{
        "modelName": "Llama-3-405B",
        "worldSize": 16,
        "npuDeviceIds": [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]],   // 两节点
        "pipelineParallelSize": 2,
        "tensorParallelSize": 8
      }]
    }
  }
}
```

HCCN 必须先 init OK（见 `ascend-infer-system-setup`）。

## 容器 / K8s 部署

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      runtimeClassName: ascend            # 用 Ascend Docker Runtime
      containers:
      - name: mindie
        image: ascendhub.huawei.com/.../mindie:1.0.0
        resources:
          limits:
            huawei.com/Ascend910: 8
        env:
        - name: HCCL_BUFFSIZE
          value: "512"
        - name: PYTORCH_NPU_ALLOC_CONF
          value: "expandable_segments:True"
        volumeMounts:
        - { name: models, mountPath: /models }
        readinessProbe:
          httpGet: { path: /healthz, port: 1025 }
          initialDelaySeconds: 300        # warmup 时间
```

## 反模式

- 用 AI Core utilization 触发扩缩 → 永不触发
- 同进程塞多个大 LLM → HBM 互抢，p99 全坏
- 没 warmup 就 ready → 头 100 请求慢
- 客户端 timeout < 服务端 → 服务端跑完了但客户端断
- Dynamic batching `maxQueueDelay=50ms` → 吞吐高 SLO 崩
- 长 prompt 不 chunked prefill → 单 5K prompt 占 NPU 数秒

## 与 NV 对应

| NV | Ascend |
|---|---|
| Triton Inference Server | MindIE Service |
| DeepStream | MindX SDK (VideoAnalysis) |
| vLLM | MindIE-LLM |
| TensorRT-LLM | MindIE-LLM (mindie-cli convert) |
| dynamic_batching | preferredBatchSizes + maxQueueDelayMicroseconds |
| ensemble model (Triton) | MindX pipeline / 多步 client orchestration |
| `/metrics` Prometheus | `/metrics`（端点路径相同） |
| K8s nvidia.com/gpu | huawei.com/Ascend910 / 310P |

## 参考

- MindIE Service 用户指南
- MindX SDK 视频分析示例
- `references/hardware/tooling-cheatsheet.md`
- `ascend-infer-llm-stack` —— LLM 专项
