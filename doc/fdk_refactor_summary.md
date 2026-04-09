# FDK Pipeline 重构设计摘要

## 1. 命名空间分层

| 层级 | 命名空间 | 内容 |
|------|----------|------|
| 公开 API | `YK::Filter` / `YK::Fdk` | 对外暴露的类和函数 |
| 内部实现 | `YK::Filter::detail` / `YK::Fdk::detail` | `__global__` kernels + device helpers |

- Filter 模块用 `YK::Filter`，不嵌入 CT 层（避免过早分层）
- FDK 专属逻辑（BP 等）用 `YK::Fdk`
- `detail` 子命名空间与 `detail/` 子目录对应，物理和逻辑双重隔离

---

## 2. 文件扩展名规则

| 扩展名 | 适用场景 |
|--------|----------|
| `.cuh` | 含 `__global__` 或 `__device__` 代码的文件 |
| `.hpp` | 纯 host 代码（Processor 类、上下文结构体等） |

---

## 3. 编译单元决策

- 目标：未来分离 `.cu` 和 `.cpp` 编译单元
- Processor 类放入 `.cu` 编译，而非 `.cpp`
  - 原因：Processor 本质是对 launch 函数的封装，强行拆到 `.cpp` 需要额外的声明/实现分离，维护成本高收益低
- Processor `.hpp` 允许 `#include` `.cuh`

---

## 4. 每个 Processor 的文件分层规则

| 层 | 文件 | 命名空间 |
|----|------|----------|
| Processor 类 | `YkXxxProcessor.hpp` | `YK::Fdk` |
| Launch 函数 | `detail/YkXxxLaunch.cuh` | `YK::Fdk::detail` |
| `__global__` Kernels | `detail/YkXxxKernels.cuh` | `YK::Fdk::detail` |
| Device Helpers | `detail/YkXxxHelpers.cuh` | `YK::Fdk::detail` |
| Umbrella Header | `YkXxx.cuh` | 包含以上全部 |

---

## 5. 完整目录结构

```
FDK/
├── YkVecFDK.hpp                        ← 对外接口，pipeline 总入口（纯 host）
├── Context/                            ← 上下文与几何（全部 .hpp）
│   ├── YkVecGeo.hpp
│   ├── YkFDKVecGeoDerived.hpp
│   ├── YkFDKGpuContext.hpp
│   └── YkFdkPipelineContext.hpp
└── Processors/
    ├── PreWeight/
    │   ├── YkFDKPreWeightProcessor.hpp
    │   ├── detail/
    │   │   ├── YkFDKPreWeightKernels.cuh
    │   │   └── YkFDKPreWeightLaunch.cuh
    │   └── YkFDKPreWeight.cuh              ← umbrella
    ├── Filter/
    │   ├── YkFDKFilterProcessor.hpp
    │   ├── detail/
    │   │   ├── YkFDKFilterKernels.cuh
    │   │   └── YkFDKFilterLaunch.cuh
    │   └── YkFDKFilter.cuh                 ← umbrella
    └── BackProject/
        ├── YkBackProjectProcessor.hpp
        ├── detail/
        │   ├── YkFDKBpHelpers.cuh
        │   ├── YkFDKBpKernels.cuh
        │   ├── YkFDKBpLaunch.cuh
        │   └── YkFDKPrecompute.cuh
        └── YkFDKBackProject.cuh            ← umbrella
```

---

## 6. 已完成的拆分

| 原文件 | 拆分结果 |
|--------|----------|
| `YkFDKFilterProcessor.hpp`（滤波核部分） | `YkFilterKernelHelpers.cuh` / `YkFilterKernelKernels.cuh` / `YkFilterKernelFFT.cuh` / `YkFilterKernel.cuh` |
| `YkBackProject.hpp`（BP 部分） | 设计确认完毕，待生成 |

---

## 7. 后续行动

- [ ] 按目录结构对剩余文件拆分：PreWeight、Filter、BackProject 各 Processor 目录
- [ ] BackProject 待生成文件：
  - `detail/YkFDKBpHelpers.cuh`
  - `detail/YkFDKBpKernels.cuh`
  - `detail/YkFDKBpLaunch.cuh`
  - `detail/YkFDKPrecompute.cuh`
  - `YkBackProjectProcessor.hpp`
  - `YkFDKBackProject.cuh`（umbrella）
