# 可编号双环钢珠几何标定

当前入口是 `workflow.py`：

```text
RAW -> tracker.py -> 编号二维点 -> 每帧 DLT
    -> source_circle.py -> joint_fit.py -> calibration.json
```

这条链路面向“二维点可以和已知三维点建立编号对应”的模体。默认模体是不同半径、不同相位的
双环，并可在环外增加一个或两个标记点。标记用于确定零相位和减少手性歧义，不是把面积直接当作编号。

## 1. 三维模型和编号

默认 `phantom.py::marked_points()` 生成 24 个环珠，再追加配置中的标记点：

```text
上环：半径 50 mm，z=+50 mm，12 球
下环：半径 40 mm，z=-50 mm，相对上环相位 15 度，12 球
标记：默认 (50, 0, 80) mm；也支持 0、1 或 2 个标记
```

二维点顺序与模型点顺序是独立的。编号候选表示 `observed[k] -> model[indices[k]]`；坐标变换表示
`X_target = R X_source + t`。前者不能用 detector 轴翻转代替，后者也不能通过交换环来代替。

## 2. 图像追踪

`FormalTrackerConfig` 和 `track_stack()` 位于 `tracker.py`。追踪器只看 RAW 图像、像素尺寸和模体点
定义，不读取 SID、SDD、offset、tilt、源偏移或模体姿态。

处理步骤：

1. 阈值分割、连通域筛选、圆度和边界检查，计算强度加权亚像素质心；
2. 选择目标分离度较好的锚点帧；
3. 对环交换、环内方向、循环相位、标记排列和可能的镜像分支建立候选；
4. 用已知三维点拟合物理相机，以完整重投影 RMSE 选择候选；
5. 从锚点向前、向后跟踪。预测来自已测量的相邻帧，不来自系统几何；
6. 粘连时在同一连通区域内按预测点分区后分别求质心；出界、检测失败或超过跳变门限的 ID 永久停用。

因此，不能依靠“最大面积标记球”单独决定编号，也不能在后续帧重新启用失败 ID。跟踪结果中的
`points.npy`、`observed.npy` 和 `tracking_report` 必须先通过检查，才进入联合拟合。

## 3. 每帧 DLT

对第 `i` 帧的三维点 `X_j=(x_j,y_j,z_j,1)^T` 和图像点 `q_ij=(u_ij,v_ij,1)^T`，求：

\[
q_{ij} \sim P_i X_j, \qquad P_i\in\mathbb{R}^{3\times4}.
\]

`dlt.py::calibrate_view()` 使用归一化 DLT。每帧至少需要 6 个非退化点；多余点用于降低噪声并识别错误编号。
必须保存重投影 RMSE、最大误差、条件数和失败状态。DLT 的整体尺度没有物理意义，但它完整定义了该帧射线。

由 `P=[M|p_4]` 可得源点：

\[
S_i=-M^{-1}p_4.
\]

`S_i` 位于模体坐标系中，不能直接当成机械源偏移或机械等中心位置。

## 4. 源轨迹

`source_circle.py::fit_source_circle()` 将所有 `S_i` 拟合为三维圆，得到圆心、法向、半径、径向残差和轴向残差。
它的用途是：

- 检查逐帧 DLT 是否形成合理的圆轨迹；
- 提供 SID 和相位的初始化；
- 为后面的等效全局模型提供稳定初值。

源偏移、模体平移和模体旋转存在补偿自由度，所以圆半径不自动等于唯一机械 SID。

## 5. 固定源规范下的联合拟合

`joint_fit.py::fit()` 固定规范量：

```text
source_offset_x = source_offset_y = source_offset_z = 0
offset_n = 0
```

在该规范下优化共享机器量：

```text
SID, SDD, offset_u, offset_v, tilt_u, tilt_v, tilt_n
```

同时优化模体的 6 个刚体 nuisance 量（旋转和平移），目标是所有帧、所有有效点的鲁棒重投影误差：

\[
\min_\Theta\sum_{i,j}\rho\left(\left\|q_{ij}-\Pi(X_j;\Theta,\theta_i)\right\|^2\right).
\]

如果真实源存在偏移，它会被等效地吸收到 offset、tilt 和模体位姿中。因此联合结果适合生成一组一致的
FDK 几何，但不能单独解释为唯一机械安装误差。若只做迭代重建，可以跳过联合拟合，直接使用每帧 DLT
转换出的 `source/detector_center/U/V/N`。

## 6. 镜像和手性

单个平面对称标记通常仍留下一个镜像编号分支。当前流程对镜像候选和扫描方向候选计算整组多帧 RMSE，
选择全局残差较小者；这一步发生在 correspondence 层，不修改公共右手 detector 坐标系。双标记或不共面的
标记可以从模体设计上消除该歧义，但代码仍应保留候选评分和重投影验收。

## 7. 重建和验收

逐帧 DLT 结果可直接用于 DLT FP/BP 或代数重建：

```text
P_i -> dlt_to_conevec.py -> source, detS, r_u, r_v
```

重建体坐标必须和 DLT 的三维点坐标在同一坐标系，模体姿态不能再重复施加。使用联合参数进行 FDK 时，
必须整组使用 SID、SDD、两个面内 offset 和三轴 tilt。

最低验收项目：

1. 每帧 DLT 重投影 RMSE 和最大误差；
2. 源圆径向、轴向残差；
3. 联合拟合 RMSE、参数边界和收敛状态；
4. 标记点和两环在重投影中的编号正确性；
5. FDK 或迭代重建中环是否闭合、是否出现双边缘和拖影。

## 8. 输出文件

`calibrate_identifiable_phantom()` 保存：

```text
points.npy               编号后的亚像素点
observed.npy             有效观测掩码
calibration.json         DLT、源圆、对应关系和联合拟合结果
tracking_diagnostics.png 追踪和逐帧误差图（由 tracker 配置生成）
```

仿真真值只能用于外部验证，不传入 tracker、DLT map 或优化器。真实 RAW 若存在长期缺失点，当前密集联合拟合
应先改成带观测掩码的目标函数，不能用插值点伪造完整观测。
