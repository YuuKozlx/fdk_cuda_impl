# Cyl 解析重建几何适配

`ProjectionMapper` 将物理圆柱探测器投影重采样到以源点为轴心、半径为
`SDD` 的虚拟等角圆柱。它只处理解析重建所需的输入适配，不属于 Cyl
迭代 FP/BP 算子。

## 数据流

```text
采集/配置层
    原始 geometry
    -> GeometryCanonicalizer 刚体变换与 offset 解释
    -> corrected_geometry（目标世界坐标中的物理柱面）
    -> ProjectionMapper 曲率重排
    -> canonical_geometry（虚拟等角柱面，R=SDD）

解析重建层
    canonical projection + canonical_geometry
    -> CylFdkPipeline 预加权
    -> 弧长域滤波
    -> analytic Cyl-FDK 深度加权反投影
```

简化的数据流是：

```text
物理圆柱 (R 可不等于 SDD)
    -> ProjectionMapper
虚拟等角圆柱 (R = SDD)
    -> Cyl-FDK 的滤波与解析反投影
```

映射按目标通道生成源到虚拟圆柱的射线，再求该射线与物理圆柱的交点，
最后以连续通道和行坐标做双线性采样。越过物理探测器可见范围的目标射线
输出零，不使用边界 clamp。

`R = SDD` 时，映射退化为等角圆柱上的同坐标采样；`R != SDD` 时会发生
曲率重排和轴向坐标缩放。重排只改变投影采样位置，不附加 FDK 权重或
滤波 Jacobian，后续解析算法应在虚拟等角坐标中计算这些量。

当前实现针对标准同轴圆柱和等间距视图。任意逐视图姿态、倾斜探测器和
复杂轨迹不属于此 mapper 的契约；这些情况继续使用一般 Cyl 迭代算子。

## 通用解析重建入口

`YkCylAnalyticReconstruction.hpp` 中的
`YK::CylFpBp::Analytic::Reconstruction` 将上述映射和柱面 FDK 串成一个
可复用入口：

```cpp
YK::CylFpBp::Analytic::Reconstruction reconstruction;
YK::CylFpBp::Analytic::ReconstructionConfig config{};
config.source_to_detector_mm = calibrated_sdd_mm;
config.launch = launch_policy;
if (!reconstruction.prepare(volume, Nu, Nv, physical_geometry, config,
        SFilterKernelDesc::RamLak(), stream, device_id)) {
    // 几何不满足标准同轴圆柱契约，或资源分配失败
}
reconstruction.reconstruct(d_projection, d_volume, true);
```

`source_to_detector_mm` 必须来自扫描标定，不能用 `R` 或“源到圆柱轴的
距离”替代。`SCylConeProjGeomVec` 的 `detectorCenter/U/V` 描述物理采样
网格，适配器会自动处理数组中心偏移；输出是半径为 SDD 的虚拟等角网格。
因此 CFDK、XFDK、柱面 wFBP 可以共用该输入层，而迭代重建不要调用它。

若采集坐标需要转换到解析规范坐标，可设置：

```cpp
config.canonical_from_acquisition = canonical_from_acquisition;
```

该刚体变换只规范化源点、探测器中心和 U/V 方向。调用方必须保证传给
解析后端的 `SVolGeom` 已使用相同的目标坐标系；后端不会隐式移动体积，
也不会在 kernel 内再次校正采集几何。
