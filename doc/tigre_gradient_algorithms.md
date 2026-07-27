# TIGRE 风格梯度重建算法说明

本文说明 YKCBCT 对 TIGRE 梯度类算法的行为复现、公共接口和参数语义。
实现行为参考 CERN TIGRE 的 MATLAB/Python 版本；工程复用自身投影算子、
显存管理和 CUDA 调度，不直接依赖 TIGRE 运行时。

## 1. 支持范围

统一入口提供以下算法：

| 配置枚举 | TIGRE 名称 | 数据更新 | 正则化状态机 |
|---|---|---|---|
| `Sart` | SART | 每个子集 1 个视角 | 无 |
| `OsSart` | OS-SART | 每个子集 `block_size` 个视角 | 无 |
| `Sirt` | SIRT | 一个子集包含全部视角 | 无 |
| `AsdPocs` | ASD-POCS | SART | TV + POCS 调度 |
| `OsAsdPocs` | OS-ASD-POCS | OS-SART | TV + POCS 调度 |
| `BAsdPocsBeta` | B-ASD-POCS-β | SART + Bregman 投影 | TV + POCS 调度 |
| `Pcsd` / `OsPcsd` | PCSD / OS-PCSD | SART / OS-SART | PCSD 步长调度 |
| `AwPcsd` / `OsAwPcsd` | AwPCSD / OS-AwPCSD | SART / OS-SART | 自适应加权 TV |
| `AwAsdPocs` / `OsAwAsdPocs` | Aw-ASD-POCS / OS-Aw-ASD-POCS | SART / OS-SART | 自适应加权 TV + POCS |

`TigreGradientReconstructor` 根据 `SCBCTParams` 构造标准圆轨迹；
`TigreGradientReconstructorEx` 接受逐视角 `SConeProjGeomVec`。`Ex` 只代表
geometry 输入方式，算法和调参能力完全相同。

## 2. 基本用法

```cpp
#include "Iter/YkTigreGradientReconstructor.hpp"

YK::Iter::TigreGradientReconstructor::Config config{};
config.algorithm = YK::Iter::ETigreGradientAlgorithm::OsAsdPocs;
config.iterations = 30;
config.block_size = 20;       // 每个子集的视角数，不是子集数量
config.lambda = 1.0f;         // TIGRE 中的 lambda / beta
config.lambda_reduction = 0.99f;
config.tv_iterations = 20;
config.alpha = 0.002f;
config.alpha_reduction = 0.95f;
config.maximum_update_ratio = 0.95f;
config.max_l2_error = -1.f;   // 自动估计
config.non_negative = true;

YK::Iter::TigreGradientReconstructor reconstructor;
reconstructor.prepare(params, config, stream, device_id);
reconstructor.reconstruct(d_projection, d_volume);
const auto& statistics = reconstructor.statistics();
```

## 3. 参数与 TIGRE 的对应关系

| C++ 字段 | TIGRE 参数/变量 | 说明 |
|---|---|---|
| `iterations` | `niter` | 最大外循环数 |
| `block_size` | `blocksize` | 一个子集包含的视角数 |
| `lambda` | `lambda` / `beta` | 数据一致性更新初始步长 |
| `lambda_reduction` | `lambda_red` / `beta_red` | 每个外循环后的乘法衰减 |
| `tv_iterations` | `tviter` / `ng` | 每轮 TV 内迭代次数 |
| `alpha` | `alpha` | 首轮 `dtvg = alpha * dp` |
| `alpha_reduction` | `alpha_red` | TV 更新过大时衰减 `dtvg` |
| `maximum_update_ratio` | `ratio` / `rmax` | 判断 `dg > rmax * dp` |
| `max_l2_error` | `maxl2err` / `epsilon` | 投影一致性停止阈值 |
| `minimum_beta` | 固定 `0.005` | beta 下限；本实现允许显式配置 |
| `adaptive_delta` | `delta` | AwTV 的边缘权重参数 |

必须特别注意 `max_l2_error` 的历史语义：ASD-POCS 系列比较
`dd > epsilon`，而 TIGRE 的 PCSD/AwPCSD 实现判断
`d_p * d_p > epsilon`。本实现有意保留这一差异，以便同一组 TIGRE 参数得到
可对照的调度行为。

## 4. 初始化与 Nesterov

`initialization` 支持：

- `Zero`：零体积；
- `Fdk`：先以相同 geometry 执行 FDK；设备投影会下载到主机以适配当前
  `FdkPipeline`；
- `DeviceVolume`：从 `d_initial_volume` 复制调用方已有设备体积。

`relaxation_mode=Nesterov` 复现 TIGRE MATLAB SART/SIRT/OS-SART 的
`lambda='nesterov'` 更新：每个子集先产生候选解，再与上一候选解按动量系数
组合。每个完整外循环检查投影残差；残差上升时与 TIGRE 一样直接停止，
不回退到上一体积。

Nesterov 是 ART 数据更新选项，不是 ASD/PCSD 家族的公共正则化开关。

## 5. ASD-POCS 与 PCSD 状态

每个 POCS 外循环显式维护：

```text
dd    当前投影残差
dp    数据一致性更新幅度
dg    TV 更新幅度
c     数据更新与 TV 更新的方向余弦
dtvg  TV 绝对步长
beta  数据更新松弛因子
```

普通 ASD-POCS 首轮使用 `dtvg = alpha * dp`；若
`dg > maximum_update_ratio * dp` 且投影残差仍高于阈值，则按
`alpha_reduction` 缩小后续 TV 步长。

PCSD 首轮 TV 步长为 1，后续使用当前数据更新前投影距离与首轮基准距离的
比值。TV/AwTV 每次内迭代都将梯度归一化为 L2 范数 1，再应用绝对步长。

B-ASD-POCS-β 使用私有工作投影执行 Bregman 递推：

```text
b_work <- b_work + mu * (b_work - A*x)
mu     <- mu * bregman_beta_reduction
```

调用者传入的测量投影保持只读。

## 6. 统计与停止原因

`statistics()` 返回完成外循环数、子集更新数、`dd/dp/dg/c`、当前 beta、
TV 步长以及停止原因。可能的提前停止包括：

- 方向余弦接近 -1 且投影误差达到阈值；
- beta 低于 `minimum_beta`；
- Nesterov 模式投影残差开始上升。

这些统计用于调参和回归检查，不改变求解状态。

## 7. 与通用代数重建接口的边界

`AlgebraicReconstructor` 的 `SmoothedTv` 是无状态的“数据外循环后正则化”
策略，适合普通 OS-SART-TV。ASD-POCS、PCSD 和 Bregman 算法需要跨轮维护
`dp/dg/dtvg/beta` 等状态，因此由 `TigreGradientReconstructor` 作为完整
求解器实现，不能等价地降为一次 `regularizer.apply()`。

## 8. 回归测试

```powershell
cmake --build out/build/x64-refactor-check --config Release `
  --target YKCBCT ykcbct_manual_tests --parallel 8

out/build/x64-refactor-check/Yktest/Release/ykcbct_manual_tests.exe `
  recon/tigre-gradient-family
```

该入口覆盖全部算法枚举、Nesterov、标准/Ex geometry、FDK/设备体积初始化，
并验证 Bregman 流程不会修改调用者的测量投影。

## 9. 大箭头模体切换算法

`large/arrow-tigre` 固定使用 `160×160×96` 带六面方向字符的箭头模体、
`384×128` 平板探测器和 360°/360 张投影。重建方法和常用参数可从命令行
切换，模体与采集条件保持不变，适合横向比较：

```powershell
# 默认 OS-ASD-POCS
ykcbct_manual_tests.exe large/arrow-tigre --log-level info

# OS-SART，不执行 TV
ykcbct_manual_tests.exe large/arrow-tigre `
  --arrow-method os-sart `
  --arrow-iterations 20 `
  --arrow-block-size 30 `
  --arrow-lambda 0.2

# AwPCSD
ykcbct_manual_tests.exe large/arrow-tigre `
  --arrow-method aw-pcsd `
  --arrow-iterations 10 `
  --arrow-lambda 0.25 `
  --arrow-tv-iterations 5
```

`--arrow-method` 支持：

```text
sart, os-sart, sirt,
asd-pocs, os-asd-pocs, b-asd-pocs-beta,
pcsd, os-pcsd, aw-pcsd, os-aw-pcsd,
aw-asd-pocs, os-aw-asd-pocs
```

`--arrow-block-size` 只对 `os-*` 方法生效，语义仍是“每个子集包含的视角
数”。SART/ASD/PCSD 非 OS 版本固定每个子集一张，SIRT 固定一个子集包含
全部 360 张。

结果写入 `out/test-artifacts/large-arrow-<method>/`，因此不同算法不会互相
覆盖。每个目录包含模体、投影、重建、误差、JSON 参数和三层对比 BMP。
