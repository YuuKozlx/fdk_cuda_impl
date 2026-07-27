# 代数重建正则化规范与变更说明

## 1. 变更目标

原代数重建流程只包含数据一致性更新和上下限约束。若直接在 OSSART 后端中
加入 TV，会产生以下问题：

- TV 与 `DetailedSubset` 或 `TigreApprox` 权重实现绑定；
- 普通圆轨迹和 Ex geometry 容易形成重复实现；
- SIRT-TV、SART-TV 和 OS-SART-TV 需要分别复制代码；
- 后续 ASD-POCS、Chambolle-Pock、FISTA 和 ADMM 无法共享调度协议；
- 正则化执行频率可能随子集数变化，参数缺乏稳定语义。

本次变更将正则化提升为统一代数重建中的独立策略层。

## 2. 规范变更前后

### 2.1 变更前

```text
AlgebraicReconstructor
  └─ 根据 method/weight_model 选择数据更新后端
       └─ 后端一次执行 iterations × subset_count 次更新
```

配置只描述：

- 算法类型；
- 权重模型；
- 子集顺序和数量；
- 松弛因子；
- 上下限约束。

不存在统一正则化类型、生命周期、执行上下文或状态保存协议。

### 2.2 变更后

```text
AlgebraicReconstructor / AlgebraicReconstructorEx
  ├─ Data backend
  │    ├─ AlgebraicDetailedWeightBackend
  │    └─ AlgebraicTigreBackend
  ├─ Regularizer strategy
  │    ├─ None
  │    └─ SmoothedTv
  └─ Outer-loop scheduler
       ├─ 执行一轮全部真实子集
       ├─ 调用 regularizer.apply(context)
       ├─ 应用上下限约束
       └─ 进入下一轮
```

算法、权重、几何和正则化成为四个正交维度。

## 3. 公共配置规范

```cpp
enum class EAlgebraicRegularizer : int {
    None = 0,
    SmoothedTv = 1
};

enum class ETvDimensionality : int {
    Slice2D = 2,
    Volume3D = 3
};

struct AlgebraicRegularizationConfig {
    EAlgebraicRegularizer type;
    float strength;
    int inner_iterations;
    ETvDimensionality tv_dimensionality;
    float epsilon;
    float strength_reduction;
};
```

参数语义：

| 字段 | 规范 |
|---|---|
| `type` | `None` 时不得分配正则化工作区或改变原重建结果 |
| `strength` | 单次 TV 梯度下降的步长，不宣称等于严格目标函数中的 β |
| `inner_iterations` | 每个完整数据外循环之后执行的正则化内迭代次数 |
| `tv_dimensionality` | 使用逐层二维或完整三维物理差分 |
| `epsilon` | 平滑 TV 分母中的正数稳定项 |
| `strength_reduction` | 每完成一次正则化外循环后对步长进行的乘法衰减 |

启用正则化时以上数值必须为正，体素尺寸也必须为正。无效配置应在
`prepare()` 阶段失败，而不是进入 CUDA kernel 后产生 NaN。

## 4. 执行时机规范

默认且当前唯一支持的调度是 `AfterOuterIteration`：

```text
数据子集 0
数据子集 1
...
数据子集 N-1
正则化 inner_iterations 次
上下限约束
```

不在每个子集后执行 TV，原因是：

- 正则化总次数不应随 `subset_count` 隐式放大；
- SIRT、SART、OS-SART 应共享可比较的参数语义；
- TIGRE 连续分块在视角数不能整除请求子集数时，实际块数可能不同；
- 每子集正则化会显著增加显存带宽和 kernel 调度开销。

调度器必须使用后端的 `actualSubsetCount()`，不能假定请求子集数等于真实
块数。

## 5. 增量接口规范

`iterateSubsetUpdates()` 继续表示纯数据一致性子集更新：

- 不触发 TV；
- 不把任意调用次数解释成一个完整外循环；
- 保持旧类增量调用兼容性。

启用正则化时应调用 `reconstruct()`。若未来需要交互式逐轮执行，应新增
明确的 `iterateOuterLoops()`，而不是改变现有增量接口语义。

## 6. 第一版平滑 TV 定义

三维形式：

```text
TVε(x) = Σ sqrt((Dx x)^2 + (Dy x)^2 + (Dz x)^2 + ε^2)
```

梯度：

```text
∇TVε(x) = -div(grad(x) / sqrt(|grad(x)|² + ε²))
```

内迭代更新：

```text
x <- x - strength * ∇TVε(x)
```

`Slice2D` 忽略 Z 差分，其他公式相同。X/Y/Z 差分分别除以物理体素尺寸，
因此非等体素下正则化作用于物理梯度而不是数组索引梯度。

TV 梯度写入独立工作区，再更新体积；禁止一个 kernel 在读取邻域的同时原地
覆盖输入，否则同一轮内会出现线程执行顺序相关结果。

## 7. 约束顺序变化

变更前，各数据后端可在每个子集更新后应用上下限约束。

变更后，为保持旧数值行为，后端仍可在子集后执行约束；启用正则化时，统一
调度器还会在 TV 后再次执行约束，确保 TV 步骤不会重新产生负值或越界值。

因此：

```text
子集更新 → 后端约束 → ... → TV → 调度器最终约束
```

`regularization=None` 时外循环调度不改变数据更新次数、松弛衰减次数和旧接口
结果语义。

## 8. 扩展协议

所有正则器实现以下生命周期：

```cpp
class IAlgebraicRegularizer {
public:
    virtual bool prepare(...) = 0;
    virtual bool apply(float* volume,
        const AlgebraicRegularizationContext& context) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual bool requiresDataUpdateNorm() const;
};
```

上下文包含：

- 当前外循环编号；
- 已完成的数据子集更新数；
- 当前数据松弛因子；
- 数据更新 L2 幅度预留字段。

`requiresDataUpdateNorm()` 为后续状态型算法预留。返回 `true` 时，调度器应：

1. 在数据更新前保存体积快照；
2. 完成整轮数据更新后计算 `||x_after-x_before||₂`；
3. 将结果写入 `context.data_update_l2`；
4. 再调用正则器。

后续算法建议映射：

| 算法 | 扩展方式 |
|---|---|
| ASD-POCS | 正则器要求数据更新范数，并限制 TV 步长与其比例 |
| Chambolle-Pock | 正则器内部持有 primal/dual 工作区和步长状态 |
| FISTA + TV prox | 正则器内部持有动量体积并调用 TV prox |
| Split Bregman / ADMM | 正则器内部持有辅助变量和 Bregman/对偶变量 |

这些扩展不得直接依赖 Detailed/TIGRE 后端类型，也不得根据是否 Ex geometry
选择不同实现。

## 9. 兼容性与测试要求

每次新增正则器至少验证：

1. `None` 路径仍通过全部代数重建测试；
2. 标准圆轨迹和 Ex geometry 都可配置该正则器；
3. 输出有限且非空；
4. 实际子集更新数等于 `iterations × actualSubsetCount()`；
5. `reset()` 恢复正则化步长和内部状态；
6. `release()` 可重复调用且不泄漏显存；
7. 非等体素、2D/3D 模式分别有数值测试；
8. 对需要数据更新范数的算法验证调度器快照协议。

当前回归入口：

```text
ykcbct_manual_tests recon/ossart-tv
ykcbct_manual_tests recon
```

第一版测试覆盖 OS-SART + Detailed 权重 + Ex geometry + 3D 平滑 TV、步长
衰减、非负约束和子集更新计数。
