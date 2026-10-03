# 当前 Cho 数据与完整真值对比

## 完整机械真值

```text
coordinate_convention = right_handed
detector_normal       = detector_to_source
detector_axes         = U_cross_V_equals_N

SID                   = 440.000 mm
SDD                   = 770.000 mm
source_offset         = (0, 10, 0) mm
detector_offset       = (U=2.085, N=0, V=4.170) mm
detector_tilt         = (U=1, V=2, N=3) deg
phantom_offset        = (10, 15, 20) mm
phantom_rotation      = (2, 0, 0) deg
```

## 结论

当前 Cho 流程的二维点编号和投影矩阵估计是正确的，但还没有把所有机械参数逐项唯一分解出来。

| 比较量 | 同一 Cho 定义下的真值 | RAW估计值 | 误差 |
|---|---:|---:|---:|
| SDD | 769.763 mm | 770.605 mm | +0.842 mm |
| DLT主点 u0 | 467.775 px | 467.733 px | -0.043 px |
| DLT主点 v0 | 535.800 px | 535.710 px | -0.090 px |
| 等效源轨迹半径 | 440.114 mm | 440.627 mm | +0.513 mm |

第一行采用“同一 Cho 投影矩阵分解定义”进行比较。机械配置中的 `SDD=770 mm` 与当前 RAW 估计值相差约 `0.605 mm`。

## 为什么源轨迹半径不是机械 SID

源具有切向偏移：

\[
source\_offset_y=10\text{ mm}.
\]

机械 SID 是 440 mm，但来源点轨迹半径是：

\[
R_{source}=\sqrt{440^2+10^2}=440.114\text{ mm}.
\]

源偏移还引入等效相位变化：

\[
\Delta\beta=\tan^{-1}(10/440)=1.302^\circ.
\]

因此，只拟合来源点圆时，无法仅由圆半径把 `SID=440 mm` 和 `source_offset_y=10 mm` 唯一拆开。

## 为什么 DLT 主点不是机械 detector offset

Cho 分解得到的主点表示中心射线与探测器平面的穿刺位置。机械 `detector_offset_u/v` 表示探测器中心相对于机械基准的位置。探测器存在三个倾角时，两者不是同一参数。

因此：

```text
DLT主点偏移 (-43.767, 24.210) px
```

不能直接与：

```text
机械 offsetU = 5 px
机械 offsetV = 10 px
```

逐项比较。

## 当前已经算对的内容

- 360帧的24个二维钢珠编号；
- 每帧完整投影矩阵；
- 每帧来源点和探测器相机姿态；
- 同一Cho定义下的主点，误差小于0.1像素；
- 等效来源点圆半径，误差约0.51 mm；
- SDD，和机械真值相差约0.61 mm；
- 平均逐帧重投影误差约0.051像素。

## 当前还没有独立算出的机械量

- `source_offset_y=10 mm`；
- 机械 `SID=440 mm` 与源切向偏移的唯一分解；
- 机械 `detector_offset_u=2.085 mm`；
- 机械 `detector_offset_v=4.170 mm`；
- 机械 `tilt_u=1°`、`tilt_v=2°`、`tilt_n=3°`；
- 模体机械平移和探测器偏移之间的唯一规范分解。

要得到这些机械量，需要在 Cho 每帧投影矩阵之外，再加入扫描架坐标系、源偏移规范和模体刚体姿态的联合模型。仅凭每帧 DLT 分解不能自动完成这一步。
