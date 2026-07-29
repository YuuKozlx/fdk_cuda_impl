# CylFpBp 圆柱探测器匹配算子

本目录提供独立于平面 `SConeProjGeomVec` 的圆柱 U / 线性 V 探测器模型，
用于生成 FreeCT wFBP 所需的原生等角弧面投影，并提供离散匹配的 FP/BP。

## 几何

每个 view 使用：

- `source`：射线源点；
- `detector_principal`：主射线与圆柱面的交点；
- `detector_u_tangent`：主射线处一个 channel 的圆周弧长向量；
- `detector_v`：一个 row 的轴向向量；
- `radius_mm`：圆柱半径；
- `principal_u/principal_v`：主射线对应的连续像素坐标；
- `angle.x`：唯一角度来源。

像素中心为：

```text
delta = (u - principal_u) * |detector_u_tangent| / radius

D(u,v) = C
       + A * radius * cos(delta)
       + B * radius * sin(delta)
       + detector_v * (v - principal_v)
```

其中 `C` 是圆柱轴线上与 principal point 同一 V 位置的点，`A` 是径向
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

## 算子

FP 对每条射线执行显式三线性 Joseph 积分；BP 对完全相同的采样点和
权重执行原子散射，因此 BP 是 FP 的离散转置。该 BP 用于共轭测试和
迭代算法，不包含 FDK/wFBP 的解析权重。

首版以正确性和几何验证为优先，尚未实现 ASTRA 圆柱 kernel 的主轴分组、
纹理缓存和探测器超采样优化。
