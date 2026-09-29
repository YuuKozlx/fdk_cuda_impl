# Yang 2017 PIC 独立实现

`yang_pic.py` 是一套独立的逐视角实现，输入每帧 12 个钢珠的亚像素中心，顺序必须是：

```text
[E1, E2, E3, E4, E5, E6, F1, F2, F3, F4, F5, F6]
```

其中 E、F 是两个平行圆环，每环 6 颗珠，圆环半径 `r` 和两环中心距 `2l` 已知。E1/F1 需要有可识别标记，才能得到最后的 gantry angle `t`。

## 算法阶段

1. 将 `(E1,F4)` 与 `(E4,F1)` 等三组投影直线交叉，求 `O`。`O` 是 phantom 中心 `W` 在探测器上的投影，不是源点垂足，也不是 Wu 单排钢珠椭圆法中的主点。
2. 用 `E1F1`、`E4F4` 求 `D` 和 yaw `eta`；平行时自动改用公共线方向。
3. 将坐标旋转到 yaw=0，分别拟合 E/F 两个一般椭圆。
4. 用两个椭圆与 `x=0` 的交点代入论文式 (6)-(12)，枚举交点顺序，求 roll `theta`、源点 `S` 和 phantom 中心 `W` 在虚拟探测器坐标系 `i` 中的位置。
5. 对 pitch `phi` 做一维搜索，最小化论文式 (17) 的两圆环椭圆一致性。
6. 若 E1/F1 已标记，使用 Yang 式(26)的线性最小二乘求 `sin(t), cos(t)`，得到 gantry/stage angle `t`；线性系统退化时回退到周期拟合。未标记时可以关闭此步。

## 单帧调用

```python
import numpy as np
from yang_pic import YangConfig, calibrate_view

config = YangConfig(
    ring_radius_mm=30.0,
    ring_half_spacing_mm=25.0,
    pixel_size_mm=(0.254, 0.254),
)
pose = calibrate_view(points_px, config)
print(np.rad2deg([pose.yaw_rad, pose.roll_rad, pose.pitch_rad]))
print(pose.o_px)                         # O 的图像坐标
print(pose.source_i_mm)                  # S 在虚拟探测器坐标系中的坐标
print(pose.phantom_center_i_mm)          # W 在虚拟探测器坐标系中的坐标
```

## 批量调用

```python
poses = calibrate_stack(points_stack_px, config)
```

`points_stack_px` 的形状是 `[views, 12, 2]`。每个视角独立解算，符合 PIC 的 pose-independent 设计。固定源/固定探测器系统可以进一步调用 `extended_calibrate_stack` 做论文第 7 节联合约束。

```python
from yang_pic import extended_calibrate_stack

extended = extended_calibrate_stack(
    points_stack_px, config,
    source_constraint_weight=12.0,
    initial_poses=poses,
)
```

extended 目标函数由每帧钢珠重投影误差和源点在真实探测器网格坐标中的跨帧一致性误差组成。它只适用于源、探测器固定而模体/转台改变姿态的系统；源随 gantry 移动时不能强加这个约束。

## 转换为重建几何

`geometry_from_pose(pose)` 将 PIC 结果转换为一帧可用于重建的几何：

```python
from yang_pic import YangConfig, calibrate_stack_geometries

config = YangConfig(30.0, 25.0, (0.254, 0.317))
poses, geometries = calibrate_stack_geometries(points_stack_px, config)
P0 = geometries[0].projection_matrix
```

每个 `ViewGeometry` 包含：

- `source_mm`：以 `O` 为原点的源点坐标；
- `detector_origin_mm`、`u_axis`、`v_axis`、`normal`：探测器平面和方向；
- `principal_point_px`：主点像素坐标；
- `projection_matrix`：将模体固定坐标 `(xp, yp, zp, 1)` 映射到该帧像素坐标的 `3 x 4` 矩阵；
- `pixel_size_mm`：u/v 方向像素尺寸。

命令行输出的 JSON 也会包含 `reconstruction_geometries`。`gantry_angle_rad` 是构造模体固定坐标到当前虚拟探测器坐标变换所必需的，因此使用 `--no-gantry` 时仍可做 PIC 参数计算，但不能生成这个投影矩阵。

## 命令行

```powershell
python tools/cbct_calibration/yang_pic.py points.npy `
  --radius-mm 30 --half-spacing-mm 25 `
  --pixel-u-mm 0.254 --pixel-v-mm 0.254 `
  --output out/yang_pic.json
```

当前文件不包含图像检测和钢珠跨帧跟踪，也不把单排钢珠 RAW 自动转换为双圆环 12 球输入。必须先得到可靠的 12 球索引坐标；如果使用的是当前两份单排钢珠数据，它们不满足 Yang phantom 的先验结构，不能直接套用这套 PIC 解算器。

## 参数含义和实现边界

每帧的两个椭圆来自该帧的两个圆环，每个椭圆仅用 6 个球心。不是同一颗球跨帧形成的轨迹。文中的 opposing beads 指标定体上固定对置的球，不是相隔 180 度的两帧。

`principal_point_px` 是源点沿探测器法向投影的垂足；`o_px` 是 W 的透视投影，二者分别输出。
`source_detector_distance_mm` 是源到探测器平面的垂直距离，`source_phantom_distance_mm` 是 |SW|，
`source_axis_distance_mm` 是源到 phantom 轴线的距离。在非圆轨道系统中它不自动等于机械旋转轴 SOD。
`reprojection_rmse_px` 是两坐标分量合并的 RMS，单位为实际探测器像素。

输入须遵守右手坐标和论文球编号，E 位于 phantom 轴负侧、F 位于正侧。
图像行向下的原始坐标应先转换到约定的 v 方向，不能仅靠改角度符号修补镜像。
源应位于两环轴向高度之间，源在 phantom 的 +z 一侧，roll 的有效分支限制在 ±50 度内。
缺球、NaN 或退化的交叉线会明确失败，不会虚构缺失的投影。

实现说明：第 6 步默认使用式 (26) 的线性 `sin/cos` 解，退化时回退到周期一维重投影搜索；自动比较两个编号方向并在 diagnostics 中记录所选方向。第 7 节的固定源/固定探测器联合优化已由 `extended_calibrate_stack` 实现。零 roll 使用式 (14)-(16)，无穷远 D 输出为 null。

测试命令：

```powershell
python -m unittest discover -s tools/cbct_calibration -p test_yang_pic.py -v
```

独立射线/平面交点生成的点数据覆盖零 roll、近零 roll、非零三个角度、非方形像素，
同时检查源位置、phantom 中心、SDD、主点和像素重投影误差。这是几何求解器验证，尚非真实 FP 图像全链路验证。
