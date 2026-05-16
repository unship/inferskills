---
name: ascend-infer-kernels
description: Ascend NPU 自定义算子开发——Ascend C（类 CUDA C++）、TBE (Tensor Boost Engine)、TIK、Cube/Vector/Scalar/MTE pipeline 编排、UB/L1 staging、NZ layout、kernel 注册到 torch_npu / GE。当用户问"Ascend C"、"昇腾自定义算子"、"TBE"、"TIK"、"NPU kernel"、"AICPU 写算子"时使用。仅在 op 缺失或性能瓶颈无法绕过时才动手。
---

# Ascend Kernel Development (Ascend C / TBE)

> 与 nv-infer-kernels 平行。**最后手段**：先确认 CANN 是否已支持该 op、ATC/MindIE 是否已优化，再考虑写自定义算子。
> 心智模型与 CUDA 显著不同：Ascend 是 **Cube + Vector + Scalar + MTE 流水线** 模型，不是 SIMT。

## 何时写自定义算子

仅当：
1. CANN OpAPI 未覆盖（升级 CANN 后仍缺）
2. msprof 显示某 AICPU fallback 占总时间 > 15%
3. 模型主体 op 已 ATC 优化好，剩下的小 op 串成瓶颈
4. 自家专有算法（如 router、特殊 mask）

不要：
- 期望比 cuBLAS 等价 (Cube) 写出更优的 GEMM —— 写不过
- 给 elementwise 写自定义 —— ATC fusion 已经做了

## 三套开发栈（选一）

| 工具 | 抽象层级 | 上手 | 性能 | 何时用 |
|---|---|---|---|---|
| **Ascend C** | C++ DSL (类似 CUDA C++ + Eigen) | 中 | 高 | 新算子开发首选（CANN 7.0+） |
| **TBE-DSL** | Python DSL | 低 | 中 | 简单算子、快速原型 |
| **TIK 2.0** | Python，更接近底层 | 中-高 | 高 | TBE 不够灵活时 |
| **AscendCL plugin** | C++ runtime | 高 | — | 自定义 host-side runtime op |

主推 **Ascend C**（CANN 7+ 的主线）。

## Ascend C 基础概念

```
AI Core 内：
  Scalar Unit       — 控制流、地址计算
  Vector Unit       — elementwise / reduce
  Cube Unit         — 矩阵乘 (16×16×16 systolic)
  MTE (Memory Transfer Engine) — HBM ↔ UB/L1 异步搬运

Buffer 层级：
  HBM (全局)
  L1 Buffer (1 MB 片上)
  L0A/L0B (Cube 输入 64 KB)
  L0C (Cube 累加 256 KB)
  UB (Vector 工作区 192-256 KB)
```

写 kernel = 编排 **MTE 搬数据 → Vector/Cube 算 → MTE 搬回**，关键在 **流水线重叠**（让 MTE 与计算同时进行）。

## 最小 Ascend C 算子样板

```cpp
// add_custom.cpp — 一个 element-wise add (Ascend C 风格)
#include "kernel_operator.h"
using namespace AscendC;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t total_len) {
        xGm.SetGlobalBuffer((__gm__ half*)x, total_len);
        yGm.SetGlobalBuffer((__gm__ half*)y, total_len);
        zGm.SetGlobalBuffer((__gm__ half*)z, total_len);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LEN * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LEN * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LEN * sizeof(half));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileNum; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t pid) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[pid * TILE_LEN], TILE_LEN);
        DataCopy(yLocal, yGm[pid * TILE_LEN], TILE_LEN);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t pid) {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, TILE_LEN);   // vector add
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t pid) {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[pid * TILE_LEN], zLocal, TILE_LEN);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;
    constexpr static int TILE_LEN = 1024;
    constexpr static int BUFFER_NUM = 2;
    int32_t tileNum = ...;
};
```

关键点：
- **TQue**：自动 double/triple buffering，与 MTE 异步配合实现 pipeline 重叠
- **LocalTensor / GlobalTensor**：UB / HBM 张量抽象
- **DataCopy**：MTE 触发的异步搬运
- **Add / Mul / Cast / Brcb / ReduceSum / ...**：Vector API
- **MatMul**：Cube API（带 L0A/L0B/L0C staging）

## Cube GEMM 样板（Ascend C 高层 API）

```cpp
matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>,
               matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>,
               matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>
    mm;
mm.SetTensorA(a_gm);
mm.SetTensorB(b_gm);
mm.SetBias(bias_gm);
mm.IterateAll(c_gm);
```

CANN 的 `matmul` 模板自动处理 L0/L1 staging 与 K-loop tiling，写起来接近 CUTLASS 的高层 API。

## 注册到 PyTorch / GE

写完 Ascend C kernel 编译为 .o，再注册：

1. **GE 自定义算子**（ATC 编译时识别）
   - 写 op_proto (shape inference)
   - 写 op_info (tiling 策略)
   - 编译为 .so 放 CANN custom op 路径
2. **torch_npu** —— 通过 PyTorch C++ API 注册：
   ```cpp
   TORCH_LIBRARY_IMPL(my_ops, NPU, m) {
       m.impl("my_add", my_add_npu);
   }
   ```
3. **MindIE** —— `MindIE-Atb` 算子库可二次开发

CANN 8.0+ 提供 **OpsFactory** 模板项目，从 cookiecutter 起步：
```bash
msopgen gen -i op_def.json -f tf -c ai_core-Ascend910B3 -lan cpp -out my_op/
```

## Tiling 策略

Ascend C 的 tiling 不是写在 kernel 里，而是**单独的 host 侧 tiling function**：

```cpp
// my_op_tiling.cpp
ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // 根据 shape 算出每个 AI Core 的 tile
    auto totalLen = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();    // 取 AI Core 数（910B = 25 / 30 等）
    auto tilingData = context->GetTilingData<MyTilingData>();
    tilingData->tileLen = totalLen / coreNum;
    tilingData->coreNum = coreNum;
    return ge::GRAPH_SUCCESS;
}
```

错的 tiling 让算子在 1 个 Core 上跑，**性能塌一档**。

## 性能调优速记

| 现象（msprof） | 诊断 | 处方 |
|---|---|---|
| MTE 利用率高，Cube/Vector 闲 | 数据搬运瓶颈 | 增 tile、减 HBM ↔ UB 往返、加 double buffering |
| Cube 高，Vector 闲（GEMM 主导） | 算力充分 | OK |
| Vector 高，Cube 闲（elementwise） | 没用 Cube | 改写为 matmul 等价 / 增 batch |
| Pipe stall 多 | TQue 深度不够 | BUFFER_NUM=2/3 增到 4 |
| UB 不够 | tile 太大 | 缩 tile、分段 |
| NZ ↔ ND 反复转 | 输入/输出 layout 不匹配 | host 侧 entry/exit 各一次，中间统一 |

## 与 CUDA 心智差异

| CUDA | Ascend |
|---|---|
| Thread (SIMT, 32/warp) | 无 warp 概念，AI Core 是矩阵 + 向量流水线 |
| Block / Grid | "Core" 维度（数据并行到多 AI Core） |
| Shared Memory + L1 | UB / L0 / L1 buffer |
| `__syncthreads()` | TQue 自动管理 pipeline 同步 |
| Tensor Core (mma.sync) | Cube API (matmul 模板) |
| TMA (Hopper async copy) | MTE 异步 DataCopy |
| `cp.async` | TQue + DataCopy 自动 |
| CUTLASS | CANN matmul 模板 / OpsFactory |
| Triton lang | TBE-DSL / TIK 2.0 |
| `__shfl_sync` | 无（架构不同） |

## 已知坑

| 坑 | 现象 | 解 |
|---|---|---|
| 忘了写 tiling function | 算子单 core 跑慢 | 必须实现 TilingFunc |
| UB 算超了 | 编译报 buffer overflow | 缩 tile 或拆 stage |
| TQue BUFFER_NUM=1 | 无 pipeline 重叠 | 至少 2，常用 3 |
| NZ format 假设错 | 数值 garbage | 显式调 TransData 或在 op_info 声明 input format |
| float 累加 half 输入 | 精度丢失 | Cube 累加用 float / 单独 cast |
| 没注册 op shape inference | 后续图编译失败 | op_proto 必须实现 InferShape |
| AI Core 数硬编码 | 跨 SoC 失败 | `GetCoreNumAic()` 动态查 |
| `__aicore__` 函数调 host 函数 | 编译失败 | device 侧函数加 `__aicore__ inline` |

## 调试

```bash
# Ascend C 模拟器（CPU 上验证逻辑）
ascend_op_test --op=my_add --device_id=0 --use_simulator=1

# 真机跑 + msprof 看 pipe util
msprof --output=./prof --application="./test_my_op" \
       --aic-metrics=PipeUtilization,ArithmeticUtilization
```

## 反模式

- 一上来就写 Ascend C —— 99% 情况下 CANN 已支持，先升 CANN
- 期望像 CUDA 那样 SIMT 思考 —— Ascend 是 systolic + vector pipeline，需换思路
- tile 设很大期望 cache 命中率 —— UB 总共 256 KB，太大 spill 到 L1/HBM
- 跨 SoC 不重 build —— OM 跨架构必然不通

## 参考

- CANN Ascend C 开发指南
- huawei-ascend GitHub: `samples/operator/ascendc` 模板项目
- MindStudio Op Development 教程
- `references/hardware/huawei-atlas-910b.md`
