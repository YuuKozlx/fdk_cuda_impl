# YKCBCT 迭代重建使用说明

本文说明当前统一迭代重建接口，包括 SIRT、SART、OS-SART、CGLS、
PWLS 以及代数重建正则化。旧类仍可使用，但新代码建议优先使用统一门面。
TIGRE 风格 ASD-POCS、PCSD、AwTV 和 Bregman 算法的完整状态机与参数映射
见 [tigre_gradient_algorithms.md](tigre_gradient_algorithms.md)。

## 1. 统一代数重建的配置维度

代数重建被拆成四个互不绑定的维度：

| 维度 | 选项 | 含义 |
|---|---|---|
| 数据更新方法 | `Sirt`、`Sart`、`Ossart` | 决定每轮使用的子集数量 |
| 权重模型 | `DetailedSubset`、`TigreApprox` | 决定行、列归一化权重的计算方法 |
| 子集顺序 | `Sequential`、`GoldenRatio` | 决定 Detailed 模式的子集遍历顺序 |
| 正则化 | `None`、`SmoothedTv` | 决定每轮数据更新后的先验约束 |

几何入口同样独立：

- `AlgebraicReconstructor` 根据 `SCBCTParams` 构造标准圆轨迹；
- `AlgebraicReconstructorEx` 接收逐视角 `SConeProjGeomVec`，可用于螺旋、
  偏置或其他显式 geometry；
- `Ex` 只表示几何输入方式，不隐含算法、权重或正则化类型。

### 1.1 OS-SART 基本用法

```cpp
#include "Iter/YkAlgebraicReconstructor.hpp"

YK::Iter::AlgebraicReconstructor::Config config{};
config.method = YK::Iter::EAlgebraicMethod::Ossart;
config.weight_model = YK::Iter::EAlgebraicWeightModel::DetailedSubset;
config.subset_order = YK::Iter::EAlgebraicSubsetOrder::GoldenRatio;
config.iterations = 30;       // 完整扫描全部子集的轮数
config.subset_count = 20;
config.relaxation = 0.8f;
config.relaxation_reduction = 0.98f;
config.use_min = true;
config.min_constraint = 0.f;
config.fp_task = YK::ETask::FP_Joseph;
config.bp_task = YK::ETask::BP_Joseph_v3;

YK::Iter::AlgebraicReconstructor reconstructor;
if (!reconstructor.prepare(params, config, stream, device_id)) {
    // 处理参数或显存错误
}
if (!reconstructor.reconstruct(d_projection, d_volume)) {
    // 处理重建错误
}
```

### 1.2 显式 geometry 用法

```cpp
#include "Iter/YkAlgebraicReconstructorEx.hpp"

YK::Iter::AlgebraicReconstructorEx reconstructor;
reconstructor.prepare(params, geometry, config, stream, device_id);
reconstructor.reconstruct(d_projection, d_volume);
```

`geometry.size()` 必须等于 `params.iPAng`。接口不会把显式 geometry 静默
退化成由 `SCBCTParams` 生成的圆轨迹。

### 1.3 SIRT、SART 与 OS-SART 的统一表达

```cpp
config.method = EAlgebraicMethod::Sirt;   // 实际子集数为 1
config.method = EAlgebraicMethod::Sart;   // 实际子集数为 iPAng
config.method = EAlgebraicMethod::Ossart; // 使用 subset_count
```

`iterations` 始终表示完整外循环数，而不是子集级更新次数。
`totalSubsetUpdates()` 可查询已经执行的数据更新次数。

`iterateSubsetUpdates()` 是兼容旧接口的低层增量入口，其参数表示子集级
更新次数。该接口只执行数据一致性更新，不会在不完整子集周期后隐式执行
正则化；启用正则化时应使用 `reconstruct()`。

### 1.4 权重模型

`DetailedSubset` 对每个子集实时计算：

```text
R = A_s * 1
C = A_s^T * 1
```

其显存占用较低，支持普通和显式 geometry，是默认推荐方案。

`TigreApprox` 使用 TIGRE 风格近似权重：

- 通过 2×2×2 粗体积估计行权重；
- 保留体积 XY 尺寸的 `1.1` 缩放和列权重的 `0.9` 缩放语义；
- 子集使用连续分块；
- 当视角数不能整除请求子集数时，实际块数可通过
  `actualSubsetCount()` 查询。

## 2. OS-SART-TV

第一版采用“完成一轮 OS-SART 数据更新，再执行平滑 TV 梯度下降”的交替
流程：

```text
全部子集数据更新
→ TV 内迭代
→ 非负性/上下限约束
→ 下一外循环
```

```cpp
config.regularization.type =
    YK::Iter::EAlgebraicRegularizer::SmoothedTv;
config.regularization.tv_dimensionality =
    YK::Iter::ETvDimensionality::Volume3D;
config.regularization.strength = 1e-4f;
config.regularization.inner_iterations = 5;
config.regularization.epsilon = 1e-6f;
config.regularization.strength_reduction = 0.98f;
```

可选维度：

- `Volume3D`：在 X/Y/Z 三个物理方向计算 TV；
- `Slice2D`：每个 XY 层独立计算 TV，适合层厚明显大于平面像素的数据。

差分会除以 `vox_x_mm/vox_y_mm/vox_z_mm`，因此参数强度基于物理空间，
不会把非等体素误当成等体素。

建议从较弱正则化开始：

```text
strength            1e-5 ～ 1e-4
inner_iterations    2 ～ 5
epsilon             1e-6 ～ 1e-4
strength_reduction  0.95 ～ 1.0
```

如果边缘过度平滑，优先减小 `strength` 或 `inner_iterations`。不要通过增加
子集数补偿锐度，因为这会同时改变数据更新轨迹。

## 3. CGLS

CGLS 同样把策略和几何入口分开：

```cpp
#include "Iter/YkCglsReconstructor.hpp"

YK::Iter::CglsReconstructor::Config config{};
config.strategy = YK::Iter::ECglsStrategy::RobustRestart;
config.iterations = 20;
config.epsilon = 1e-8f;
config.restart_on_divergence = true;
config.use_min = true;

YK::Iter::CglsReconstructor reconstructor;
reconstructor.prepare(params, config, stream, device_id);
reconstructor.reconstruct(d_projection, d_volume);
```

策略：

- `RobustRestart`：额外计算真实 `||b-Ax||`，发散时回退并可重启搜索方向；
- `AstraClassic`：直接递推残差，不执行额外 Ax 检查和回退。

显式几何使用 `CglsReconstructorEx`。`Ex` 与策略不绑定。

## 4. PWLS

`ParallelPwlsReconstructor` 提供并行 surrogate 更新，并支持无正则、二次先验
和 Huber 先验。它与 OS-SART-TV 的交替式 TV 流程属于不同算法，不应只按
字段名比较正则化强度。

```cpp
YK::Iter::ParallelPwlsConfig config{};
config.iterations = 20;
config.relaxation = 0.7f;
config.regularizer = YK::Iter::EParallelPwlsRegularizer::Huber;
config.regularization = 2e-3f;
config.huber_delta = 1e-3f;

YK::Iter::ParallelPwlsReconstructor reconstructor;
reconstructor.prepare(params, geometry, config, stream, device_id);
reconstructor.reconstruct(d_projection, d_volume);
```

## 5. 旧接口兼容关系

旧 include 和类名仍作为适配器保留：

| 旧接口 | 统一配置映射 |
|---|---|
| `SIRT` | `Sirt + DetailedSubset` |
| `SART` | `Sart + DetailedSubset` |
| `OSSART` | `Ossart + DetailedSubset + GoldenRatio` |
| `OSSARTEx` | `Ossart + DetailedSubset + Sequential + Ex geometry` |
| `OSSART_TIGRE` | `Ossart + TigreApprox` |
| `CGLS` | `RobustRestart` |
| `CGLSEx` | `RobustRestart + Ex geometry` |
| `CGLSAstra` | `AstraClassic` |

旧接口适合保持已有调用方源码兼容；新功能和正则化应通过统一门面配置。

## 6. 测试命令

```powershell
cmake --build out/build/x64-refactor-check --config Release `
  --target YKCBCT ykcbct_manual_tests --parallel 8

out/build/x64-refactor-check/Yktest/Release/ykcbct_manual_tests.exe recon

cmake --build out/build/yktest --target YKCBCT_unit_test --parallel 8
ctest --test-dir out/build/yktest --output-on-failure
```

当前 `recon` 分类覆盖统一代数、旧代数适配器、OS-SART-TV、统一 CGLS 和旧
CGLS 适配器。
