# DLTFpBp：由逐帧 DLT 矩阵直接驱动的 FP/BP

这个目录是一套独立的 DLT 射线正投影、反投影和代数重建实现。DLT 矩阵、
`SDltRayGeometry` 和现有的 `SConeProjGeomVec` 都可以作为迭代入口的输入；
进入迭代器后统一转换成 `SDltRayGeometry`，FP、BP、归一化和子集调度都由本目录
的 DLT 算子完成。

`SConeProjGeomVec` 与 DLT 的等效平板转换关系仍然保留，用于需要和现有平板接口
交换几何的场景。这个等效平面不是机械意义上的真实探测器平面。

## 1. 目录职责

```text
DLTFpBp/
├── YkDltProjectionGeometry.hpp       DLT 矩阵到射线参数的转换
├── YkDltSiddonProjector.hpp          GPU 内存和 FP/BP 的宿主端入口
├── YkDltAlgebraicReconstructor.hpp   独立的 SIRT/SART/OS-SART 代数后端
└── kernels/
    ├── YkDltAlgebraicKernels.cu      残差、归一化和体积更新 kernel
    ├── YkDltAlgebraicLaunch.cuh      代数更新 kernel 启动接口
    ├── YkDltSiddonTraversal.cuh      FP/BP 共用的 Siddon 体素遍历
    ├── YkDltSiddonKernels.cu         匹配 Siddon 与体素驱动 BP kernel
    └── YkDltSiddonLaunch.cuh         kernel 启动接口
```

数值回归测试位于 `Yktest/test_dlt_fpbp.cpp`，生产源码目录不包含测试实现。

## 2. 输入矩阵表示什么

每一帧输入一个按行存储的 `3 x 4` 投影矩阵：

```text
SDltProjectionMatrix::value =
    [p00 p01 p02 p03
     p10 p11 p12 p13
     p20 p21 p22 p23]
```

三维点必须是齐次坐标 `Xh=(x,y,z,1)^T`，其中 `x/y/z` 的单位是毫米：

```text
q = P * Xh = (q0,q1,q2)^T
u = q0/q2
v = q1/q2
```

`u/v` 是投影数组的连续像素索引，不是毫米坐标。重建体的世界坐标必须与求 DLT
时使用的模体坐标处于同一个坐标系。模体姿态已经包含在 `P` 中，重建时不能再把
同一个模体姿态重复施加一次。

## 3. 从 P 得到一条射线

将投影矩阵写成：

```text
P = [M | p4]
```

只要 `M` 可逆，投影中心，也就是射线源点，为：

```text
C = -inverse(M) * p4
```

探测器像素 `(u,v)` 对应的世界坐标射线方向为：

```text
d(u,v) = inverse(M) * [u,v,1]^T
```

因此射线可以写成适合 GPU 计算的仿射形式：

```text
d(u,v) = ray00 + u*rayU + v*rayV
X(t)   = C + t*d(u,v), t >= 0
```

这里没有“射线必须终止于某个虚拟探测器点”的条件。kernel 只将半射线与重建体的
轴对齐包围盒相交，并在进入点和离开点之间积分。

投影矩阵允许乘任意非零全局尺度。`buildRayGeometry()` 会统一三个方向基向量的尺度，
并根据重建体中心消除 `P` 的正负号歧义。测试同时覆盖正尺度和负尺度。

## 4. 图像 v 轴约定

DLT 拟合点和原始投影数组必须采用相同的行坐标方向。如果 DLT 使用向上的 `v`，
而数组行号向下增加，应在构建几何前调用 `flipImageV(P, rows)`。它执行：

```text
P_down = [ 1  0       0  ] P_up
         [ 0 -1  rows-1 ]
         [ 0  0       1  ]
```

不能只在 kernel 中把 `v` 取负，否则主点平移 `rows-1` 会丢失。

## 5. FP 与 BP 的离散定义

`YkDltSiddonTraversal.cuh` 对每条射线产生完全相同的 `(voxelIndex,length)` 序列。
FP 和 BP 只是对这份序列采用不同的累加器：

```text
FP: projection[r] += length(r,k) * volume[k]
BP: volume[k]     += length(r,k) * projection[r]
```

其中 `length(r,k)` 是射线穿过体素 `k` 的毫米长度。匹配 BP 使用 `atomicAdd`，以处理
不同射线同时写入同一体素。这样构造的 BP 是 FP 离散矩阵的转置，而不是 FDK 加权反投影。

另外提供一个独立的体素驱动近似 BP：每个体素通过 DLT 的逆基求出连续投影坐标，读取
该位置的最近邻或双线性投影值，再乘以该连续射线穿过体素 AABB 的弦长。它不遍历所有
射线，也不做原子加法，通常更快，但它不是 `A^T`，不能使用伴随误差作为正确性判据。

严格伴随关系为：

```text
<A*x, y> = <x, A^T*y>
```

GPU float 计算中仅剩累加次序造成的舍入差异。当前非理想 12 帧回归用例的代表结果为：

```text
relative error ≈ 1e-9
agreement      > 99.9999998%
```

回归门限是 `1e-5`，实际结果远小于该门限。

## 6. 在 C++ 中使用

代码调用只有三个阶段，适合直接从现有重建类中组合，不需要建立命令行流程：

```cpp
std::vector<YK::DltFpBp::SDltRayGeometry> rays;
bool geometry_ok = YK::DltFpBp::buildRayGeometry(
    projection_matrices,
    detector_channels,
    detector_rows,
    volume_geometry.center,
    rays);

YK::DltFpBp::DltSiddonProjector projector;
bool prepared = geometry_ok && projector.prepare(
    rays, volume_geometry, detector_channels, detector_rows, device_id);

projector.forward(device_volume, device_projection, false, stream);
projector.backproject(device_projection, device_volume, true, stream);
// 速度优先的非伴随版本；最后一个参数 true 表示双线性采样。
projector.backprojectVoxelDriven(device_projection, device_volume,
    true, true, stream);
```

`forward(..., accumulate=false)` 覆盖投影输出；设为 `true` 时累加到已有投影。
`backproject(..., clear_volume=true)` 在 BP 前清零体积；迭代算法需要自行控制何时清零。

### 6.1 代数重建后端

`DltAlgebraicReconstructor` 通过明确枚举选择方法：

```cpp
SDltAlgebraicConfig config;
config.method = EDltAlgebraicMethod::Ossart;
config.ossart_subset_size = 9;
config.relaxation = 0.8f;

DltAlgebraicReconstructor reconstructor;
reconstructor.prepare(rays, volume_geometry, Nu, Nv, config, stream);
reconstructor.iterate(measured_projection, volume, 8, stream);
```

也可以直接传入 DLT 矩阵，或者传入已有平板几何：

```cpp
reconstructor.prepare(projection_matrices, volume_geometry, Nu, Nv, config);
reconstructor.prepare(existing_vec_geometry, volume_geometry, Nu, Nv, config);
```

- `Sirt`：一个子集包含全部投影帧；
- `Sart`：一个子集只包含一帧；
- `Ossart`：按照 `ossart_subset_size` 组织交错子集；
- `iterate(..., sweeps)` 中一个 sweep 始终表示所有子集都被访问一次。

`ExactSubset` 在每个子集上用 DLT Siddon 算子计算 `A_s*1` 和 `A_s^T*1`。
`TigreApprox` 保留精确行归一化，将列权重沿 z 平均成 2D 图像；两种归一化都
不依赖物理 `SID/SDD/offset` 参数。

`SDltAlgebraicConfig::backprojection_model` 默认是
`EDltBackprojectionModel::MatchedSiddon`。设置为 `VoxelDriven` 后，代数重建的
列权重和残差回投都使用体素驱动近似 BP；`voxel_interpolation` 可选择
`Nearest` 或 `Bilinear`。该模式是速度优先的非伴随算法，建议与较小松弛因子和实际
重建结果一起评估，不能再声称严格满足 `<Ax,y>=<x,A^Ty>`。

## 7. 能做什么，不能做什么

这对匹配算子可直接用于 SIRT、SART、CGLS、PWLS 等迭代重建。每帧完整 `P_i` 已经
定义了从世界点到像素、以及从像素回到空间射线的关系，因此不要求唯一分离机械意义的
源偏移和探测器偏移。

它不能直接替代现有 FDK 几何。FDK 的预加权、卷积尺度和反投影权重依赖真实平板的
SID、SDD、像素尺寸和探测器姿态；一般 DLT 矩阵没有唯一的机械参数分解。若需要 FDK，
应另做带机械约束的几何拟合，而不是给 `detS` 填入虚拟值。

## 8. 回归测试覆盖范围

`Yktest/test_dlt_fpbp.cpp` 使用：

- `18 x 16 x 14` 的非零中心重建体；
- 三个方向不同的体素尺寸；
- 12 帧非理想源轨迹；
- 逐帧变化的焦距、主点、朝向和面内旋转；
- 随机体数据和随机投影数据；
- `P` 的交替正负全局尺度。

测试先确认缩放后的 `P` 生成同一射线，再以 host double 计算两侧内积。它已注册到
Visual Studio/CMake 生成的 `ykcbct_manual_tests` 项目中，测试名称为 `dlt/adjoint`。
