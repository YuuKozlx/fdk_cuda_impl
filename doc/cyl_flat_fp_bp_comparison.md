# Cyl / Flat 正反投交叉对比

本文记录当前 CUDA 测试中平板探测器（Flat）与圆柱探测器（Cyl）的 FP/BP
交叉组合。所有耗时均为 GPU kernel 重复执行的平均值，不包含首次资源创建、
纹理建立、主机传输和结果下载。

## 测试尺寸

| 项目 | Flat kernel benchmark | Cyl kernel benchmark |
|---|---:|---:|
| 体积尺寸 `Nx x Ny x Nz` | `64 x 64 x 32` | `64 x 64 x 32` |
| 探测器尺寸 `Nu x Nv` | `96 x 32` | `96 x 32` |
| 投影视图数 | `640` | `640` |
| 射线数 | `1,966,080` | `1,966,080` |
| 体素尺寸 | `0.8 x 0.8 x 0.8 mm` | `0.8 x 0.8 x 0.8 mm` |
| 探测器像素 | `1.0 x 1.0 mm` | `1.0 x 1.0 mm` |
| SID / SDD | `160 / 300 mm` | `160 / 300 mm` |
| 圆柱半径 | 不适用 | `240 mm`（性能测试） |

圆柱 FDK 专项测试另使用 `64 x 48 x 32` 探测器、`17 x 15 x 9` 体积和
`32` 个视图；该尺寸用于解析权重验证，不与下面的性能数字混用。

## Kernel 耗时

| 探测器 | FP | Joseph BP | V3 BP | 备注 |
|---|---:|---:|---:|---|
| Flat | `0.800 ms` | `4.488 ms` | `1.152 ms` | Joseph BP 为原子散射版本 |
| Cyl | `0.329 ms` | `3.760 ms` | `1.829 ms` | Cyl Joseph 为纹理投影 + 浮点 Joseph |

圆柱解析 BP 专项测试：

| 算子 | 平均耗时 |
|---|---:|
| Cyl FDK BP | `0.077 ms` |
| Cyl FDK-matched BP | `0.088 ms` |

完整纹理 kernel 基准（同一 `1,966,080` 射线规模）：

| 探测器 | 算子 | 平均耗时 |
|---|---|---:|
| Flat | Siddon FP | `0.413 ms` |
| Flat | Joseph FP | `0.800 ms` |
| Flat | Joseph BP | `4.488 ms` |
| Flat | Joseph-v2 BP | `1.185 ms` |
| Flat | Joseph-v3 BP | `1.071 ms` |
| Flat | FDK BP | `0.992 ms` |
| Flat | FDK-matched BP | `1.076 ms` |
| Flat | Siddon BP（texture voxel-v2） | `1.180 ms` |
| Flat | Siddon RayDriven BP（texture） | `1.533 ms` |
| Cyl | Siddon FP | `1.223 ms` |
| Cyl | Joseph FP | `0.329 ms` |
| Cyl | Joseph BP | `3.760 ms` |
| Cyl | V3 BP | `1.829 ms` |
| Cyl | FDK BP | `1.393 ms` |
| Cyl | FDK-matched BP | `1.368 ms` |

## 伴随性

指标定义为：

```text
relative = | <FP(x), y> - <x, BP(y)> |
           / max(|<FP(x), y>|, |<x, BP(y)|)
```

| 探测器 | FP | BP | FP 耗时 | BP 耗时 | `<FP(x),y>` | `<x,BP(y)>` | relative |
|---|---|---|---:|---:|---:|---:|---:|
| Flat | Siddon | Siddon RayDriven | `0.509 ms` | `1.541 ms`（纹理） | `-1.089545e+01` | `-1.089545e+01` | `0.000000` |
| Flat | Joseph | Joseph | `0.804 ms` | `4.461 ms` | `-1.084180e+01` | `-1.084057e+01` | `0.000113` |
| Flat | Siddon | FDK-matched | `0.504 ms` | `1.077 ms` | `-1.089545e+01` | `-1.106810e+01` | `0.015599` |
| Flat | Joseph | FDK-matched | `0.804 ms` | `1.077 ms` | `-1.084180e+01` | `-1.106810e+01` | `0.020446` |
| Flat | Siddon | FDK | `0.504 ms` | `1.018 ms` | `-1.089545e+01` | `-1.446952e+00` | `0.867197` |
| Cyl | Joseph | Joseph | `0.378 ms` | `4.153 ms` | `-6.215422e+01` | `-6.199106e+01` | `0.002625` |
| Cyl | Joseph | FDK-matched | `0.378 ms` | `1.456 ms` | `-7.909501e+01` | `-8.296442e+01` | `0.046639` |
| Cyl | Joseph | FDK | `0.378 ms` | `1.390 ms` | `-7.909501e+01` | `-2.365472e+01` | `0.700933` |

表中耗时来自 `fpcyl/kernel-benchmark` 的独立 CUDA event 统计，测试尺寸见
上文。伴随性测试和性能测试的几何尺寸不同，所以耗时用于算子量级比较，
不能解释为伴随测试那一组尺寸的精确运行时间。Flat `Siddon RayDriven BP`
本表使用新增的纹理读取 overload；旧的线性内存 overload 仍保留但不纳入统计。
`Siddon BP（texture voxel-v2）` 是另一种体素驱动实现，不能与 RayDriven 混称。

## 纹理数值契约

- 离散采样算子使用 `Point + Border + unnormalized`：Flat/Cyl Siddon FP、
  Flat/Cyl ray-driven Joseph/Siddon BP。
- 连续坐标插值算子使用 `Linear + Border + unnormalized`：Joseph FP、
  voxel-driven Joseph-v2/v3、FDK 和 FDK-matched BP。
- `Tex3DHandle` 保存尺寸、过滤模式和边界模式。Cyl `Operator` 在 launch 前
  验证这些元数据，错误模式直接返回失败；测试已覆盖 Linear/Point 误传。
- `cudaArray` 是源线性显存的快照，不会自动跟随源数据更新。迭代流程和公共
  FP/BP 算子均在调用 stream 上显式调用 `updateTex3DFromDeviceAsync()`，保证
  数据写入、纹理更新和 kernel 读取具有确定顺序。
- 3D 纹理的视图深度受设备 `MaxTexture3DDepth` 限制，通常早于 `grid.z`
  达到上限。纹理控制器会在分配前明确拒绝超限尺寸；此类超长序列必须由
  pipeline 按连续视图分包，不能假设一个超深纹理可由 kernel 内部分块处理。

Point 纹理对 Siddon 投影的逐像素读回最大绝对误差为 `0`。RayDriven BP 的
raw/texture 两次独立 launch 因 `atomicAdd` 顺序不确定，最大相对输出差异在
三次串行测试中为 `8.040e-7`、`1.206e-6` 和 `1.005e-6`；测试单独采用
`5e-6` 原子累加容差，不再用该容差判断纹理采样是否正确。

说明：Flat 的 Siddon/Joseph 结果来自历史算子矩阵测试基线（该测试已从注册
列表移除）；Cyl 的 Joseph 结果来自 `fpcyl/adjoint`，Cyl FDK 结果来自
`fpcyl/fdk-adjoint`。新的统一性能测试入口是 `fpcyl/kernel-benchmark`。
两组测试使用的几何尺寸不同，数值只能用于各自组合的回归比较，不能直接
比较内积绝对值。

## 单点数值

单点测试使用数组布局 `[view][v][u]` 和 `[z][y][x]`，索引均从 0 开始。
性能测试当前只输出总耗时，没有把单点值写入日志；下面的数值来自解析 BP
专项测试，因此可复现且不受随机输入影响。

| 探测器 / 算子 | 输入投影点 `(view,v,u)` | 输入 FP 投影值 | 输出体素 `(z,y,x)` | BP 输出值 |
|---|---|---:|---|---:|
| Cyl FDK BP | 任意点（恒定投影） | `1.0` | `(4, 7, 8)` 中心体素 | `32.000000` |
| Cyl FDK-matched BP | 任意点（恒定投影） | `1.0` | `(4, 7, 8)` 中心体素 | `112.506294` |

对应解析期望值分别为 `32.0` 和 `112.5`；matched 中心值的 `5.6e-5`
相对误差来自单精度圆轨迹 `sin/cos` 构造。Flat/Cyl 的随机伴随测试不使用
单点恒定投影，因此不能从内积值反推出某个像素或体素的数值。

## 运行测试

```powershell
build-helical\\Yktest\\Release\\ykcbct_manual_tests.exe fpcyl/adjoint
build-helical\\Yktest\\Release\\ykcbct_manual_tests.exe fpcyl/fdk-adjoint
build-helical\\Yktest\\Release\\ykcbct_manual_tests.exe fpcyl/kernel-benchmark
build-helical\\Yktest\\Release\\ykcbct_manual_tests.exe fpcyl/fdk-backprojectors
```

## 结论

- 严格离散伴随：Flat `Siddon FP + Siddon RayDriven BP`。
- 高伴随度快速组合：Flat `Joseph FP + Joseph BP`；圆柱当前也是
  `Joseph FP + Joseph BP`，但误差略高。
- `FDK-matched BP` 比普通 FDK BP 更接近 FP，但仍不是离散转置。
- Joseph BP 约为 V3 BP 的 `4.5` 倍（Flat）或 `2.3` 倍（Cyl），是迭代流程
  中的主要耗时项。
