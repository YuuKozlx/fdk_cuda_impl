# 螺旋重建模块

`Heli` 表示螺旋扫描几何域，不等同于某一个重建算法。

- `WFBP`：FreeCT 风格的螺旋 wFBP 解析管线。它负责重排、预加权、滤波、Parker/冗余处理和解析反投影。
- `../../Iterative/Helical`：螺旋迭代门面。`YkHelicalFlatIterativeReconstructor.hpp` 使用通用 Flat 的 SIRT、SART、OS-SART、CGLS、PWLS；`YkHelicalCylIterativeReconstructor.hpp` 使用 Cyl 专用的相同算法族。

两类迭代门面都接收完整的逐视图 geometry。螺旋的源点 z 位移由 geometry builder 在前端生成，迭代器不会根据标称 SID/角度重新构造圆轨迹。Flat 与 Cyl 保持独立的数据类型和算子，以免曲率语义在适配时丢失。

解析和迭代是两条不同路径：迭代不使用 C-FDK 的曲率 map、余弦预加权或解析深度权重；Cyl 迭代也不要求 `R=SDD`。解析 Cyl-FDK 的几何限制和映射逻辑位于 `src/CylFpBp/Analytic`。
