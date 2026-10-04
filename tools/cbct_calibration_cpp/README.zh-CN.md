# 独立 C++ 标定层

这个目录是一个不加入主工程 `CMakeLists.txt` 的独立 CMake 项目。它把几何标定的核心计算和模体业务分开：

| 模体 | 点集/业务入口 | 当前状态 |
|---|---|---|
| Cho 双环 | `makeChoDoubleRing()` + Cho 业务 tracker | 点集工厂已提供 |
| Yang 双环 | `makeYangDoubleRing()` + Yang PIC 业务层 | 点集工厂已提供 |
| 可编号双环 | `makeIdentifiableDoubleRing()` | 完整 C++ DLT/源圆/联合拟合 |
| 单排钢珠 | `makeSingleRow()` + `trackSingleRow()` | C++ 检测、亚像素质心、永久失效跟踪 |

## marker 约定

`PhantomSpec::marker_points_mm` 是唯一的标志点定义：

- 空 vector：无标志；
- 一个点：单标志；
- 两个点：双标志。

标志点始终排在环珠点之后。标志点不是按照投影面积直接编号；Python 业务层会把候选点集和已知三维点一起做 DLT 残差判定。C++ 核心接受已经有序的二维/三维点，因此不会把 Cho/Yang 的论文坐标约定写死到 DLT 和联合拟合内部。

## C++ 核心调用

```cpp
PipelineConfig config;
config.phantom.marker_points_mm = {};  // no marker
auto points3d = makeChoDoubleRing(12, 50.0, 50.0);
CalibrationResult result = calibrateIndexedPoints(points2d, points3d,
                                                  views, config);
```

`calibrateIndexedPoints()` 依次执行逐帧归一化 DLT、源点圆轨迹拟合、固定源偏移约束下的联合拟合。它只需要同一坐标系下的有序点，不需要调用方先恢复 SID、SDD、offset 或 tilt。

## 单排追踪

`trackSingleRow()` 是单排业务的预处理入口。第一帧按检测器行坐标建立暂时编号；之后每帧用门限内一对一匹配，检测失败、出界或超过 `max_jump_px` 后立即永久禁用该 ID，不会在后续帧重新启用。输出的 `points` 和 `observed` 可交给现有椭圆拟合层。

## 构建和回归

项目使用自己的 `CMakeLists.txt` 和 vcpkg 配置，构建目录位于 `out/cbct-calibration-cpp-build`。合成回归覆盖：

- 24 个双环点的 DLT；
- 模体旋转和平移；
- SID/SDD、探测器 offset 和三轴 tilt；
- 源圆拟合；
- 固定源偏移的联合拟合；
- Cho/Yang/单排点集工厂。

真实 RAW 入口仍是显式测试程序，不能把某一批 RAW 的高残差隐藏为“通过”；真实数据必须另外检查追踪、粘连分割和 map 是否正确。
