# Cyl Siddon FP/BP 算法对照

本文件覆盖 `src/CylFpBp/kernels/siddon` 中全部 Siddon 实现。V2/V3 是体素
驱动 BP 的不同数值语义，不是单纯的编译优化版本。

## 算法总览

| 模型 | 文件 | 线程方向 | 投影采样 | 射线定义 | 用途 |
|---|---|---|---|---|---|
| Siddon FP | `YkCylSiddonForwardKernels.cu` | 每条探测器射线 | 体积 Point/Border 纹理 | 源点到圆柱像素中心，逐体素边界积分 | 精确射线正投 |
| SiddonRayDriven BP | `YkCylSiddonBackprojectKernels.cu` | 每条探测器射线 | 投影 Point/Border 纹理 | 与 FP 相同射线和段长，`atomicAdd` | 严格离散转置 |
| SiddonVoxelV2 BP | `YkCylSiddonVoxelBackprojectKernels.cu` | 每个体素，Z 分组 | 投影 Linear/Border 纹理 | 连续圆柱交点射线 | 连续采样近似 |
| SiddonVoxelV3 BP | `YkCylSiddonVoxelBackprojectKernels.cu` | 每个体素，Z 分组 | 投影 Point/Border 纹理 | 最近像素中心射线 | 离散快速近似 |

`legacy/` 是兼容回归实现，`joseph/` 下的 JosephV3 使用 Joseph 权重，均不
属于 Siddon 语义。

## 历史 VoxelDriven 实现说明

旧版本中曾有 `YkCylFpBpVoxelDrivenKernels.cu` 和
`YkCylVoxelDrivenLaunch.cuh`。它实现的是 **Joseph 风格的 voxel-driven
BP**：使用主轴/物理长度权重，并不是 Siddon 体素段长算法。该历史实现已
从当前 Cyl Siddon 生产路径移除；当前对应的 Joseph 体素驱动实现位于
`joseph/YkCylJosephBackprojectV3Kernels.cu`，公开名称为 `JosephV3`。

当前 Cyl 中没有另一个未列出的“原始 Siddon VoxelDriven”模型：Siddon 的
体素驱动分支就是下文的 `SiddonVoxelV2` 和 `SiddonVoxelV3`。其中 V2 保留
连续坐标/Linear 采样语义，V3 保留离散像素/Point 采样语义；二者都使用
体素盒真实交长，而不是旧 Joseph VoxelDriven 的主轴权重。

## Siddon FP 与 RayDriven BP

FP 为每个 `(view,row,channel)` 构造源点到圆柱像素中心的射线，先与体积盒
求交，再沿体素边界递推并累加 `voxel * segment_length`。逐通道基射线在
prepare 阶段缓存为 `SSiddonChannelRay`。

RayDriven BP 使用同一射线集合和同一体素段长，将
`projection_value * segment_length` 通过 `atomicAdd` 写回体积，因此是当前
Siddon FP 的严格离散转置。代价是原子写冲突和不确定的浮点累加顺序。

## SiddonVoxelV2 BP

V2 是体素驱动近似：计算源点到体素方向的圆柱交点，得到连续
`channel/row`，使用 Linear 投影纹理插值，并用连续交点射线计算体素盒交长。

```text
continuous channel/row -> Linear texture -> continuous-ray segment length
```

它不等于 ray-driven FP 的严格转置，适合需要平滑采样的迭代算法。

## SiddonVoxelV3 BP

V3 计算连续圆柱坐标后四舍五入为 `iu/iv`，使用 Point 投影纹理读取最近
像素，并用该离散像素中心射线计算体素盒交长。

```text
round(channel/row) -> Point texture -> pixel-center-ray segment length
```

V3 在 prepare 阶段缓存 `[view][channel]` 的离散通道射线，并对固定
`(x,y,view)` 复用圆柱求交和 XY slab 区间。它通常较快，但不是 V2 的纯
编译优化版，采样结果和射线定义均不同。

当所有 view 都满足“源点位于圆柱轴上、圆柱轴与体积 Z 轴对齐”时，V3
会选择独立的 `siddon_voxel_v3_standard_kernel`，去掉通用 kernel 的几何
条件分支。若任一 view 不满足条件，则自动回退到通用 V3，保持原有几何
支持范围。

## V2/V3 对照

| 项目 | V2 | V3 |
|---|---|---|
| 坐标 | 连续 channel/row | round 后的 iu/iv |
| 纹理 | Linear | Point |
| 射线 | 连续圆柱交点 | 离散像素中心 |
| 投影值 | 双线性插值 | 最近像素值 |
| 伴随性 | 非严格转置 | 非严格转置 |
| 典型特征 | 平滑、较慢 | 离散、通常较快 |

二者共同使用源点垂直分量预计算、标准圆柱简化求交、Z 分组和 XY 区间复用；
这些是等价计算优化，不会消除采样语义差异。

标准几何专用 kernel 当前只用于 V3。V2 必须保留连续坐标和 Linear 纹理
语义，不能直接复用 V3 的离散像素射线路径。

## 选择建议

- 严格伴随测试或需要 `A^T`：`Siddon FP + SiddonRayDriven BP`；
- 连续采样近似：`SiddonV2`；
- 快速体素驱动近似：`SiddonV3`；
- 解析 FDK/wFBP：使用对应解析 BP，不使用这些 Siddon BP。

公开配置名为 `siddon`、`siddon-ray-driven`、`siddon-v2`、`siddon-v3`。
其中 `siddon` 与 `siddon-ray-driven` 保持 ray-driven 兼容语义。

## 边界和 CUDA 约定

- 投影纹理使用 `cudaAddressModeBorder`，越界返回 0，禁止 Clamp；
- RayDriven BP 使用 `atomicAdd`，V2/V3 每个体素线程独占写入；
- kernel 不同时携带线性指针和纹理指针，也不使用 `use_*` 标志切换路径；
- 性能必须以 `fpcyl/kernel-benchmark` 在目标 GPU 和目标尺寸上的实测为准。
