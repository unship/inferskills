---
name: nv-infer-serving
description: NVIDIA GPU 推理服务化——Triton Inference Server、vLLM、TensorRT-LLM + Triton、SGLang、TGI、动态/连续批处理、调度策略、autoscaling、warmup、tail latency 控制、prefix caching、SLO 拆解 (TTFT/TPOT/p99)、ensemble pipeline、HTTP/gRPC、Kubernetes 部署。当用户问"怎么部署"、"Triton"、"vLLM"、"动态批处理"、"continuous batching"、"autoscaling"、"warmup"、"冷启动"、"p99"、"tail latency"、"prefix cache"、"SLO" 时使用。
---

# Serving Stack — Triton / vLLM / TRT-LLM 部署与调度

> 选对 server + 关键 flag 调对，单卡吞吐就能再涨 2-4×。
> 这一层不需要写代码，主要是配置艺术。

## 选型

| 模型 / 场景 | 推荐 server | 备注 |
|---|---|---|
| LLM 高并发、HF 模型 | **vLLM** | 部署最快、社区最活 |
| LLM 极致单请求延迟 + 多模型 | **TensorRT-LLM + Triton** | 性能上限最高，构建复杂 |
| 结构化输出 / agent / 工具调用 | **SGLang** | RadixAttention + 结构化生成最快 |
| CNN / ViT 服务化 | **Triton** (TRT 后端) | 多框架后端、动态 batch 完善 |
| Embedding / 多模态多模型 | **Triton** (Python/ONNX 后端) | ensemble 串多模型 |
| 边缘 / 低端 | **Triton** with ONNX EP | 跨架构兼容 |

## A. vLLM 实战

```bash
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --quantization awq \
    --swap-space 16 \
    --disable-log-requests \
    --uvicorn-log-level warning
```

**关键 flag 解读**：
- `--gpu-memory-utilization 0.92`：留 8% 给 cudnn / TRT-LLM workspace；卡显存紧时调到 0.95，但易 OOM
- `--max-num-seqs`：并发上限，越大 throughput 越好但 TTFT 越长
- `--max-num-batched-tokens`：单 step 最大 token 数，chunked prefill 阈值
- `--enable-prefix-caching`：共享 system prompt KV，省 prefill 几乎免费
- `--enable-chunked-prefill`：长 prefill 不一次性塞，平滑 TTFT/TPOT
- `--swap-space 16`：抢占式调度时 swap 出去的 KV 上限（CPU mem GB）
- `--speculative-model + --num-speculative-tokens`：开 spec decoding

监控：
```bash
curl http://localhost:8000/metrics    # Prometheus 格式
# 看 vllm:time_to_first_token_seconds, vllm:time_per_output_token_seconds
#    vllm:num_requests_running, vllm:num_requests_waiting
#    vllm:gpu_cache_usage_perc
```

## B. TensorRT-LLM + Triton

引擎构建参考 `nv-infer-compile-stack`。Triton 配置：

```protobuf
# config.pbtxt（核心字段）
backend: "tensorrtllm"
max_batch_size: 256

model_transaction_policy { decoupled: True }

instance_group [{ count: 1, kind: KIND_GPU }]

parameters: {
  key: "gpt_model_path" value: { string_value: "/engine" }
  key: "kv_cache_free_gpu_mem_fraction" value: { string_value: "0.9" }
  key: "enable_kv_cache_reuse" value: { string_value: "true" }
  key: "batching_strategy" value: { string_value: "inflight_fused_batching" }
  key: "max_attention_window_size" value: { string_value: "8192" }
  key: "max_num_tokens" value: { string_value: "16384" }
  key: "enable_chunked_context" value: { string_value: "true" }
}
```

启动：
```bash
tritonserver --model-repository=/repo \
    --grpc-port=8001 --http-port=8000 \
    --pinned-memory-pool-byte-size=4294967296 \
    --cuda-memory-pool-byte-size=0:8589934592 \
    --backend-config=tensorrtllm,enable-trace=false
```

## C. Triton 给 CNN/ViT 的关键 flag

```protobuf
backend: "tensorrt"   # 或 onnxruntime / pytorch
max_batch_size: 64

dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000     # ≤ p99 SLO 的 10-20%
}

instance_group [{ count: 2, kind: KIND_GPU, gpus: [0] }]

optimization {
  cuda { graphs: true graph_spec: [{ batch_size: 1 }, { batch_size: 8 }, { batch_size: 32 }] }
  execution_accelerators {
    gpu_execution_accelerator: [{
      name: "tensorrt"
      parameters: { key: "precision_mode" value: "FP16" }
      parameters: { key: "max_workspace_size_bytes" value: "8589934592" }
    }]
  }
}
```

**dynamic batching 的核心 trade-off**：
- `max_queue_delay_microseconds` 越大 batch 越满 → 吞吐高但延迟差
- 设为 SLO 的 10-20%（p99 = 50ms → queue delay ≤ 5-10ms）

## D. Continuous / In-flight Batching

LLM 服务必开。vLLM / TRT-LLM 默认就是 continuous batching；Triton 的 `inflight_fused_batching` 同义。

对比传统 dynamic batching：
- 传统 dynamic：组好一个 batch 直到 decode 完
- continuous：每个 step 后剔除 done、加入新 request

**这是 LLM 服务化最重要的一个特性**，没开就别谈吞吐。

## E. Tail Latency / p99 优化

p50 好、p99 差是生产环境的常见痛点。

### 触发源

| 来源 | 检测 | 修 |
|---|---|---|
| Cold start | 重启后头 100 请求慢 | 启动 warmup（见下） |
| `cudaMalloc` 触发 | nsys 看 cudaMalloc 在 hot path | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Recompile (torch.compile) | recompile log | shape padding + cache_size_limit |
| GPU clock throttle | `nvidia-smi -q -d CLOCK,PERFORMANCE` 显示 SW_THERMAL | 散热 / 锁时钟 |
| 邻居噪音 (共享 GPU) | 多租户共享 | MIG / MPS percent limit |
| 网络 / 客户端慢 | server 内 metrics OK 但 e2e 差 | 加 client-side timing |
| Python GIL / GC | py-spy / cProfile | gc.disable() / 多 worker |
| KV preemption (vLLM) | metric: num_preempted | 调小 max-num-seqs |
| 长 prompt 抢占 | 单请求拖死 batch | chunked prefill |

### Warmup 流程

```python
# 启动时
shapes_to_prime = [(1, 128), (1, 512), (1, 2048),
                   (8, 128), (8, 512), (32, 128)]
for bs, seq in shapes_to_prime:
    _ = model.generate(dummy_input(bs, seq), max_new_tokens=8)
torch.cuda.synchronize()
print("warmup done")
# 这之后才暴露 /healthz=ready
```

K8s readiness probe 应等到 warmup 完成才放流量。

## F. Autoscaling

### 指标选择
**不要用 GPU util 触发扩缩** —— LLM continuous batching 永远 80%+。

正确指标：
- `vllm:num_requests_waiting > 0` 持续 N 秒 → 扩
- p95 TTFT > SLO 一段时间 → 扩
- `vllm:gpu_cache_usage_perc > 0.9` → 接近 KV 撑爆，扩

### K8s 配置

```yaml
# Prometheus Adapter custom metric
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metrics:
- type: Pods
  pods:
    metric: { name: vllm_num_requests_waiting }
    target: { type: AverageValue, averageValue: "2" }
- type: Pods
  pods:
    metric: { name: vllm_time_to_first_token_seconds_p95 }
    target: { type: AverageValue, averageValue: "1.0" }
behavior:
  scaleDown:
    stabilizationWindowSeconds: 600   # LLM 缩太快 cold-start 很痛
```

### Spot / 抢占式
- LLM 模型加载慢（70B 要 5-10 分钟），不适合 spot
- 接 spot 必须开 graceful drain：收到 termination 信号后 stop accepting，等 in-flight 跑完

## G. Prefix Caching（system prompt 复用）

同 system prompt 重复请求时，prefill 阶段 KV 完全可复用：

```bash
vllm serve ... --enable-prefix-caching
# TRT-LLM
trtllm-build ... --enable_kv_cache_reuse
```

效果：chatbot/agent 场景 TTFT 降 **50-90%**。

注意：
- 默认按 token id 完全匹配；prompt 中插入用户 ID 会立刻 invalidate
- 用 `cache_salt` / `request_id` 分租户避免 cross-tenant 缓存命中（**安全**）

## H. Ensemble / Pipeline (Triton)

```protobuf
# ensemble model
platform: "ensemble"
ensemble_scheduling {
  step [{
    model_name: "preprocess"
    input_map: { key: "raw" value: "RAW" }
    output_map: { key: "tensor" value: "PREP" }
  },
  {
    model_name: "vit_b16"
    input_map: { key: "x" value: "PREP" }
    output_map: { key: "y" value: "LOGITS" }
  },
  {
    model_name: "postprocess"
    input_map: { key: "y" value: "LOGITS" }
    output_map: { key: "result" value: "RESULT" }
  }]
}
```

ensemble 比独立 client 调用快：减少 RTT、共享 pinned mem、跨步可 overlap。

## I. SLO 拆解模板

每个端点写明：
- TTFT p50/p95/p99
- TPOT p50/p95/p99
- 端到端 latency p95
- 失败率 (5xx / OOM / timeout)
- 吞吐 (req/s 或 output tokens/s)

部署完上线前必须：
1. 跑 50/100/500 并发负载，确认 p99 不破 SLO
2. 跑 24h 长测，看是否有显存泄漏 / KV 碎片化
3. 故障注入：杀掉一个 worker 后多久恢复

## 反模式

- 静态 batching 上 LLM → 长请求把短请求拖死
- continuous batching 但不开 prefix cache → 几十倍 prefill 浪费
- GPU util 触发扩容 → 永不触发或永远扩
- 单实例同时塞多模型 → 互相挤 SM，p99 全坏，应该 MIG/MPS
- 没 warmup 就 ready → 头 100 个请求全慢
- 客户端 timeout < 服务端 → 服务端跑完了但客户端已断
- 长 prompt 不 chunked prefill → 单 5K prompt 占 GPU 3s，期间 p99 崩
- `dynamic_batching.max_queue_delay` 拉到 50ms → 吞吐高 SLO 崩
