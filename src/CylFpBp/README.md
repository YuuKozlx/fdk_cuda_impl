# CylFpBp 圆柱探测器算子

本目录提供独立于平面 `SConeProjGeomVec` 的圆柱 U / 线性 V 探测器模型，
用于生成 FreeCT wFBP 所需的原生等角弧面投影，并提供多种 FP/BP。

## 几何

每个 view 使用：

- `source`：射线源点；
- `detectorCenter`：探测器数组中心在圆柱表面上的点；
- `detectorU`：数组中心处一个 channel 的圆周切向步长；
- `detectorV`：一个 row 的轴向步长；
- `viewParameters.x`：唯一采集角度来源；
- `viewParameters.y`：圆柱曲率半径；`z/w` 预留。调用端应使用
  `cylViewAngle()` / `cylDetectorRadius()`，不要直接依赖分量布局。

`SCylConeProjGeomVec` 定义在公共 `YkProjectionGeometry.hpp`，采用 ASTRA
`cyl_cone_vec` 范式并固定为 5 个 `float4`。探测器尺寸属于算子上下文，
`principal_u/principal_v`、圆柱轴线和单位基向量全部由 prepare 阶段派生。

像素中心为：

```text
principal_u = (Nu - 1) / 2
principal_v = (Nv - 1) / 2
delta = (u - principal_u) * |detectorU| / radius

D(u,v) = C
       + A * radius * cos(delta)
       + B * radius * sin(delta)
       + detectorV * (v - principal_v)
```

其中 `C` 是圆柱轴线上与 detectorCenter 同一 V 位置的点，`A` 是径向
单位向量，`B` 是 U 切向单位向量。`radius_mm` 与焦点到主射线落点的距离
SDD 是独立量：主通道点为 `source + A*SDD`，圆柱轴线为
`source + A*(SDD-radius_mm)`。所以 `radius_mm < SDD` 和
`radius_mm > SDD` 都可表达。

`buildCylindricalArcGeometry(p, radius_mm)` 用 `p.du_mm` 作为圆弧物理像素
弧长，并由 `du_mm/radius_mm` 推导通道角度。`buildFreeCtArcGeometry(p)` 是
兼容便捷函数，固定 `radius_mm=SDD`，其圆柱轴线经过焦点，适合 FreeCT 的
源中心等角弧面数据。

`Helical::Wfbp::EInputDetector::EquiangularArc` 仍表示源中心等角弧面，即
`radius_mm=SDD`。非源中心圆柱（`radius_mm!=SDD`）应使用
`EInputDetector::CylindricalArc` 并设置 `arc_curvature_radius_mm`；wFBP 会先
按真实射线方向把通道和行二维插值到源中心等角弧面，再进入 FreeCT 重排。
该适配目前不与 phi/z FFS 组合使用。

## 算子接口

正投和反投不再由一个组合 `Operator` 管理，公开层分为两组纯虚接口：

1. `IForwardProjection` / `IBackProjection` 接收线性设备内存，内部管理纹理；
2. `IForwardOperator` / `IBackOperator` 接收已经准备好的纹理和共享几何，
   供迭代器等需要复用 GPU 资源的内部流程使用；
3. `makeForwardProjection(model)`、`makeBackProjection(model)` 以及底层工厂
   返回接口指针，业务代码不持有具体实现类；
4. Joseph FP/BP 通过 `JosephGeometryHandle` 共享同一份只读设备几何，不会
   因接口拆分而重复打包和上传逐视图参数。

`IForwardProjection` 支持 Joseph、完整浮点 matched-reference 和 Siddon；
`IBackProjection` 支持 Joseph、VoxelDrivenV3、FDK 和 FDK-matched。

所有纹理必须使用 `cudaAddressModeBorder`，越界返回 0，禁止使用会复制边缘
值的 Clamp。`IterativeReconstructor` 在全部 OS 子集间复用一份体积纹理和
一份投影纹理，纹理显存不会随子集数增长。

BP 保留完整浮点 Joseph 散射权重，是软件浮点 Joseph FP 的离散转置。默认
FP 使用硬件 Linear 纹理，其内部会量化插值小数；因此是 FP 导致默认 FP/BP
只能成为高伴随度组合，不能标记为严格 matched。伴随测试会报告内积一致度，
CGLS 也使用该组合。原始等距离三线性、预计算三线性和主轴 Joseph
线性 FP/BP kernel 及 launch 仍保留在 `detail/kernels` 层，仅用于兼容、数值
对照和性能基线，不能从公开接口选择。

`EForwardProjection::Siddon` 提供圆柱探测器的 Siddon 正投版本。它沿源点到
圆柱像素中心的射线，按体素边界逐段积分，体素值从 Point/Border 纹理读取；
因此比主轴 Joseph FP 更接近精确射线积分，但通常更慢。该版本只实现 FP，
不会自动改变所选 BP，也没有对应的圆柱 Siddon BP。配置化
任务中使用 `projector.model = "cylindrical-siddon"` 即可选择它。

底层还保留 `launch_main_axis_forward_texture_matched()`：它要求 Point 体积
纹理，从纹理读取四个体素后使用完整浮点 Joseph 权重插值，因此可与 Joseph
BP 严格匹配。该版本此前伴随误差约为 `1e-7`，但 FP 明显慢于单次硬件
Linear 读取，因此作为 `JosephMatchedReference` 对照模型保留，不作为默认 FP。

## 体素驱动 V3 BP

`VoxelDrivenBackprojectorV3` 是独立的高性能近似 BP，与 Joseph 接口实现的
浮点 Joseph BP 不同。它为每个体素遍历全部 view，从源点到体素的射线与
一般圆柱面求交，再反解连续通道角和轴向行坐标，通过持久化 3D 投影纹理
完成双线性采样。一个线程只写自己的体素，因此不使用 `atomicAdd`。

调用分为三个阶段：

1. `prepare(...)`：预计算圆柱轴、径向/切向基和角度/行距倒数，并一次性
   创建投影 cudaArray 与纹理对象；
2. `uploadProjection(d_projection)`：在绑定 stream 上异步更新持久化纹理；
3. `backproject(d_volume, accumulate)`：只提交体素驱动 BP kernel。

该算子模仿平板 `BP_Joseph_v3` 的主轴物理长度权重，但不是
纹理主轴 Joseph FP 的离散转置。它可作为 SART/SIRT/OS-SART 等带归一化
更新的近似 BP，不可用于 CGLS 或伴随测试，也不能假设它和 Joseph 的
浮点 Joseph BP 具有固定的全局幅值比例。

## 圆柱 FDK 与 FDK-matched BP

`FdkBackprojector` 和 `FdkMatchedBackprojector` 是面向解析流程的独立体素驱动
BP。二者都使用 Linear + Border 的持久化投影纹理，调用顺序与 V3 一致：
`prepare(...)`、`uploadProjection(...)`、`backproject(...)`。

当前实现只接受 `R=SDD` 的源中心等角弧面。`prepare()` 会逐 view 校验圆柱
半径、源到主通道的距离以及径向/切向/圆柱轴正交性；一般 `R!=SDD` 圆柱
投影必须由上游先按真实射线方向重映射到 `R=SDD`，不能直接套用这里的
解析权重。

设体素为 `X`、源点为 `S`、中央射线单位方向为 `a`、`SID` 为源到旋转
中心距离，则普通圆柱 FDK BP 使用：

```text
w_fdk = SID^2 / dot(X-S, a)^2
```

FDK-matched BP 使用圆柱探测器的曲面 Jacobian。令 `v_det` 为射线落点的
轴向物理坐标，`L^2=R^2+v_det^2`，则：

```text
w_matched = L^3 / (R * |X-S|^2) * voxel_volume / (du * dv)
```

这里 `R` 来自源中心圆柱上的 `n dot ray`，对应平板 matched 公式中的平面
法向距离项。两种 BP 都只实现反投影采样与空间权重，不包含频域滤波、
短扫描/Parker 或其他冗余权重，也不包含角度步长归一化；完整 FDK 流程必须
在外部完成这些阶段。
