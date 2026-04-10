# CUDA 中 inline 使用经验总结（工程实践版）

## 1. 核心结论（先记住这一句）

> `inline` 在 CUDA 中本质是"链接与编译单元可见性控制"，不是单纯的性能优化手段。
> 错误使用 `inline` 是 CUDA 工程中 LNK2019 / LNK2005 的高频来源之一。

---

## 2. `inline` 的本质（必须理解）

在 CUDA（NVCC + MSVC）中：

- `inline` **不一定**生成独立的外部链接符号
- `inline` / `__forceinline__` 可能导致：
  - 不生成外部链接符号（编译器认为调用方自己展开即可）
  - 被多个 TU 各自展开，产生重复定义（LNK2005）
  - 函数体被完全优化消除，外部引用找不到实体（LNK2019）

> **结论**：声明了 `inline` 的函数，定义必须对所有调用方可见（即放在头文件中）。
> 但放在头文件中又引发多 TU 重复定义。这个矛盾是大量链接错误的根源。

---

## 3. CUDA 工程函数分层模型

建议按职责划分三层，不同层对应不同的声明/定义规则：

### 3.1 kernel 层（设备执行，不跨 TU）

```cpp
// .cu 文件内定义，不在头文件暴露实现
__global__ void g_backproject(float* vol, ...) { ... }

// 细粒度 device helper，仅在本 TU 内使用
static __device__ float compute_weight(float u, float v) { ... }
// 或者
__device__ __forceinline__ float compute_weight(float u, float v) { ... }
```

**规则**：

- `__global__` 不跨 TU 调用，定义留在 `.cu` 内
- `__device__` helper 如需跨 TU，必须开启 `-rdc=true`（Relocatable Device Code）
- `__device__ __forceinline__` / `static __device__` 只在本 TU 内有效，不暴露外部链接

---

### 3.2 launch 层（Host 调用，跨 TU 可见）

```cpp
// YkFDKBpKernels.cuh —— 只放声明
namespace YK::Fdk::detail {
    void bp_launchBpPrecomputed(
        const uint64_t* texHandles,
        float*          vol,
        const SVolGeom& volGeom,
        int             nViews,
        CUstream        stream);
}

// YkFDKBpKernels.cu —— 只放定义（绝对不加 inline）
namespace YK::Fdk::detail {
    void bp_launchBpPrecomputed(...) {
        g_backproject<<<grid, block, 0, stream>>>(...);
    }
}
```

**规则**：

- launch 函数是普通 host 函数，**不加 `inline`**
- `.cuh` 只放函数原型（声明）
- 定义唯一存在于对应 `.cu` 中
- 这是跨 TU 链接最安全的模式

---

### 3.3 接口层（对外 API，工厂 / Processor 类）

```cpp
// IProcessor.h —— 纯虚接口，只有声明
class IProcessor {
public:
    virtual void process(const void* in, void* out, CUstream stream) = 0;
    virtual ~IProcessor() = default;
};

// BpProcessor.cu —— 实现，不加 inline
void BpProcessor::process(const void* in, void* out, CUstream stream) {
    detail::bp_launchBpPrecomputed(...);
}
```

---

## 4. 各关键字速查表

| 关键字 | 适用位置 | 外部符号 | 跨 TU | 典型用途 |
|---|---|---|---|---|
| `inline` | host 函数 | 不保证生成 | ❌ 定义须在头文件 | 短小工具函数放头文件 |
| `__forceinline__` | host / device | 不生成 | ❌ | 强制展开的小 helper |
| `__device__ __forceinline__` | device only | 不生成 | ❌（无 rdc） | kernel 内部小函数 |
| `static`（函数） | host / device | TU 内部链接 | ❌ | 文件私有实现函数 |
| 无修饰（普通函数） | host | **生成** | ✅ | launch 层、API 层首选 |
| `__device__` + `-rdc=true` | device | **生成** | ✅ | 跨 TU 共享 device 函数 |

---

## 5. 头文件中可以安全放什么

| 内容 | 是否安全 | 说明 |
|---|---|---|
| 函数声明（原型） | ✅ | 标准做法 |
| `inline` 函数定义 | ⚠️ | 可以，但每个 TU 各自展开，不产生独立符号 |
| `__device__ __forceinline__` 定义 | ✅ | 仅 device 可见，各 TU 独立展开 |
| 普通函数定义（无 `inline`） | ❌ | 多 TU include → LNK2005 |
| `__global__` kernel 定义 | ❌（通常） | 无 rdc 时不可跨 TU，放头文件无意义且危险 |
| `static` 函数定义 | ✅ | 各 TU 独立副本，不暴露外部符号 |

---

## 6. `-rdc=true` 开启条件与代价

### 何时必须开启

- `__device__` 函数需要跨 `.cu` 文件调用
- `__global__` kernel 需要从另一个 `.cu` 发射（动态并行除外）

### 开启方式（VS2022）

```
Project Properties
  → CUDA C/C++
    → Common
      → Generate Relocatable Device Code
        → Yes (-rdc=true)
```

### 代价

- 编译时间增加（需要额外的设备链接步骤 `-dlink`）
- 运行时少量额外寄存器压力（函数调用不再强制展开）
- 不是所有 GPU 架构版本都默认支持，需确认 `-arch` 匹配

---

## 7. 常见错误模式与修法

### 模式 A：launch 函数定义写在 `.cuh` 头文件中（无 `inline`）

```cpp
// ❌ 错误：YkFDKBpKernels.cuh
namespace YK::Fdk::detail {
    void bp_launchBpPrecomputed(...) {   // 定义在头文件，无 inline
        g_backproject<<<...>>>(...);
    }
}
// 结果：多个 TU include 此头文件 → LNK2005 重复定义
```

```cpp
// ✅ 修法：头文件只留声明
namespace YK::Fdk::detail {
    void bp_launchBpPrecomputed(...);   // 仅声明
}
// 定义移入 YkFDKBpKernels.cu
```

---

### 模式 B：launch 函数加了 `inline`，定义在头文件

```cpp
// ❌ 错误：头文件中
inline void bp_launchBpPrecomputed(...) {
    g_backproject<<<...>>>(...);
}
// 结果：编译器可能不生成符号实体 → LNK2019 找不到符号
// 或各 TU 各自展开但实体不一致
```

```cpp
// ✅ 修法：去掉 inline，定义移入 .cu
```

---

### 模式 C：`__device__` helper 跨 TU 调用但未开 rdc

```cpp
// fileA.cu
__device__ float compute_weight(float u) { return u * 0.5f; }

// fileB.cu
__device__ float compute_weight(float u);  // 声明
__global__ void kernel() {
    float w = compute_weight(1.0f);  // ❌ 链接失败，无 rdc
}
```

```
// ✅ 修法：Project Properties → rdc=true
// 或将 helper 移入共享头文件并加 __device__ __forceinline__（各 TU 独立展开）
```

---

## 8. 本项目推荐规范

```
YkFDKBpKernels.cuh     → 只放 launch 函数声明，不含任何定义
YkFDKBpKernels.cu      → kernel 定义 + launch 函数定义（唯一实现）
YkFDKFilterKernels.cuh → 同上
YkFDKFilterKernels.cu  → 同上
IProcessor.h           → 纯虚接口声明
BpProcessor.h          → 类声明（不含方法实现）
BpProcessor.cu         → 类方法实现
```

> 一句话原则：**头文件只管"是什么"，`.cu` 文件才管"怎么做"。**