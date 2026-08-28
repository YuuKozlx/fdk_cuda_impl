# FP/BP CUDA kernel 与 launch 约定

本文只约束底层 Flat/Cyl FP、BP 的 CUDA 实现，不改变重建器、Session 或
pipeline 的业务接口。目标是让新增算子按同一种方式接入，同时保留不同数学
模型各自独立的 kernel。

## 1. 文件职责

- `src/common/cuda/operators/YkOperatorKernelTypes.cuh`：写入模式、投影形状、
  view 范围、主轴分组、几何体素化和 Flat Joseph 方向策略。
- `src/common/cuda/operators/YkSamplingReaders.cuh`：投影数据读取策略。
- `YkFlat*Launch.cuh`：按 Joseph、Siddon、FDK 数学语义声明 Flat launch。
- `YkFPLaunch.cuh` 等旧头：只作为兼容聚合头，不再加入新声明。
- `YkCylFpBpLaunch.cuh`：当前 Cyl 兼容聚合入口；具体实现仍按 Joseph、
  Siddon、FDK 和 voxel-driven 分文件维护。
- `Yktest/cuda`：只服务测试的 CUDA kernel，不得编入正式 DLL。

## 2. launch 的固定职责

一个公开 launch 只做以下工作：

1. 把兼容接口的 `bool accumulate` 转为 `CudaOp::EWriteMode`。
2. 在覆盖模式下清零需要原子累加的输出，统一调用
   `CudaOp::clearIfOverwrite()`。
3. 根据 view 数、纹理深度和 grid 限制进行分包。
4. 选择算法模板、采样倍率和 launch policy，发射 kernel。
5. 在最后调用一次 `YK_CUDA_KERNEL_CHECK()`。

参数合法性、几何构造和纹理生命周期属于上层 prepare/context，不应复制进
每个 kernel launch。launch 只防止零长度发射和 CUDA 维度越界。

## 3. 写入模式

kernel 内不得用含义不明的 `bool accumulate`。需要在 kernel 内区分覆盖与
累加时，传递 `CudaOp::EWriteMode`，并使用 `CudaOp::accumulates()`。

- FP 通常由每个线程唯一写一个探测器像素，可在 kernel 内覆盖或累加。
- Ray-driven BP 使用 `atomicAdd`，覆盖模式必须在 launch 前异步清零。
- Voxel-driven BP 每个线程唯一写体素，可以在 kernel 内实现覆盖或累加。
- 在线分包除第一包外必须使用 `Accumulate`，清零操作必须和 kernel 使用
  同一个 stream，以维持流内顺序。

## 4. 采样 Reader 与纹理规则

投影布局固定为 `[view][row][channel]`。同一 kernel 若需支持不同存储方式，
使用编译期 Reader 模板，禁止同时传线性指针、纹理对象和运行时 `use_texture`
标志。

- `RawProjectionPointReader`：精确读取线性投影值，仅用于保留或数值对照。
- `TextureProjectionPointReader`：使用 Point 纹理读取离散投影值。
- 体积 Joseph FP 使用 Linear 纹理完成硬件插值。
- 离散 Siddon 体积读取必须使用 Point 纹理。
- 所有纹理采用 Border 地址模式。不能使用 Clamp 伪造体积或探测器边界外值；
  边界外的数学值应为 0。

硬件 Linear 纹理的小数权重存在量化，因此与软件浮点 Joseph BP 只能视为
高度伴随。严格伴随对照 FP 保留 Point 纹理和软件权重版本，但不作为默认路径。

## 5. 几何与主轴

- 外部 `SConeProjGeomVec` 是逐视图几何唯一真源。
- 公共 geometry 只保存 `src/detS/detU/detV/angle`。`centerRay`、
  `detectorNormal`、主点和平面距离只能在 prepare/precompute 阶段派生。
- `offsetU/offsetV` 沿倾斜后的最终探测器 U/V 轴，在 builder 中直接烘焙进
  `detS`；source offset 直接烘焙进 `src`。kernel 不接收或重复解释 offset。
- `volume.offsetX/Y/Z` 描述体素阵列在 Object 世界坐标系中的中心，可不在
  原点；刚体 geometry 变换不会把体素中心强制归零。
- 主轴由 `src -> detectorCenter` 的派生中心射线判断，不接受独立方向输入。
- Flat Joseph 使用 `CudaOp::forEachAxisRun()` 合并连续相同主轴的 view。
- 世界坐标转体素坐标统一使用 `normalizeToVoxel()` 或
  `normalizeToVoxelBatch()`，不得在 FP/BP 各维护一套换算。
- `AxisX/Y/Z` 只负责坐标排列；积分、散射和几何权重仍属于具体算法。

### FDK 三层几何

FDK 的几何参数严格分为三层：

1. `SID/SDD/offset/tilt/sourceOffset` 是默认轨迹 builder 的输入。生成逐视图
   geometry 后，它们不再传入 FDK stage；有外部 geometry 时不会参与计算。
2. `SConeProjGeomVec` 是唯一实际几何，校准偏移和刚体变换必须已体现在
   `src/detS/detU/detV` 中。
3. prepare 从最终 geometry 逐视图派生 `source_to_axis_mm`、
   `source_to_radial_detector_mm`、`source_to_detector_plane_mm`、主点、径向射线
   和探测器坐标架。预加权、Parker、坐标映射和 BP 只消费这些实际量。

校准 geometry 偏离理想圆轨迹时，pipeline 执行近似 FDK，并通过
`FdkGeometryDiagnostics` 报告 SOD/SDD 范围、源轴向漂移、主点、探测器法向
偏离和 U/V 非正交程度；只有退化 geometry 或重复角度会被拒绝。

Parker 的每像素扇角由最终 `src/detS/detU/detV` 和逐视图径向射线直接计算，
不再使用标称 SID/SDD。当前 FFT 滤波器仍为一个 chunk 共享一套频域权重，使用
chunk 第一帧的实际 `du`；因此逐视图 `du` 变化会被报告为近似项，但不会被
错误地替换成标称 `du`。

## 6. 超大数据与启动策略

- 一维数据优先使用 `SKernelLaunchPolicy::make1D()`，kernel 必须使用
  grid-stride loop，避免数组超过单次 grid 覆盖范围后漏数据。
- 把 view 放在 `grid.z` 的 kernel 必须按 `maxAngleChunk()` 分包，不能假定
  `views <= gridDim.z`。
- 纹理分包时 Reader 的 `view_offset` 表示纹理内偏移；设备几何指针是否偏移
  必须在 launch 中明确，不能把两者混为同一个 offset。
- 所有分包使用同一 stream，覆盖/累加模式只在包边界转换一次。

## 7. 命名与兼容

新文件和内部实现使用算法语义命名，例如 `JosephRayDrivenBp`、
`JosephVoxelDrivenBp`、`SiddonRayDrivenBp`、`FdkBp`、`FdkMatchedBp`。
`v2/v3` 只允许作为已有 public launch 的兼容名字，并应在声明处注明真实语义。

Legacy 代码只有在仍用于数值或性能对照时保留，并放在独立文件中。整段注释掉
的旧 kernel、重复 include 和开发过程记录应删除；版本历史由 Git 保存。

## 8. 最低验证

底层改动至少执行：

```powershell
cmake --build build-helical --config Release --target YKCBCT ykcbct_manual_tests
build-helical/Yktest/Release/ykcbct_manual_tests.exe fpcyl/kernel-benchmark
build-helical/Yktest/Release/ykcbct_manual_tests.exe fpcyl/adjoint
build-helical/Yktest/Release/ykcbct_manual_tests.exe fpcyl/fdk-adjoint
git diff --check
```

重构不得只验证相关系数。伴随测试应同时记录两个内积、relative error 和 FP/BP
耗时；重建测试还需保留材料真值附近的定量指标。
