# Cho 方法独立目录

本目录保存 Cho 等人 2005 年完整几何标定思路的实现和运行结果。它和单排椭圆法、编号可辨识模体的等效几何法分开维护。

## 直接运行

打开 `api_example.py`，只修改顶部配置中的 RAW 路径和输出目录，然后在 IDE 中直接调用 workflow。

`ChoWorkflowConfig` 由调用方提供 `raw_path` 和 `output_directory`；文档不绑定某一批
本地 RAW 文件名，避免把历史实验产物误当成算法输入。

流程是：

```text
RAW
→ 24 个钢珠亚像素中心
→ 首帧 24 点编号
→ 后续帧按运动预测分配像素
→ 每帧 DLT
→ 每帧投影矩阵分解
→ 多帧共享 SDD/主点联合优化
→ 源点轨迹圆和每帧探测器姿态统计
```

## 这套 Cho 实现直接估计什么

### Cho 原文的面内旋转 eta

`pic.py::calibrate_frame_cho()` 已按 Cho 2005 式 (14)-(16) 实现 eta，不再使用“两个椭圆长轴角平均”的近似。

对两个拟合椭圆

\[
a_k(u-u_k)^2+b_k(v-v_k)^2+2c_k(u-u_k)(v-v_k)=1,
\]

先由式 (14) 计算虚拟平面的点：

\[
P_a=\frac{P_1\sqrt{a_1/b_1}+P_2\sqrt{a_2/b_2}}
{\sqrt{a_1/b_1}+\sqrt{a_2/b_2}}.
\]

每个椭圆在探测器 U 方向的两个解析极值点由附录式 (A4)-(A5) 给出：

\[
U_k=\pm\sqrt{\frac{b_k}{a_kb_k-c_k^2}},
\qquad
V_k=-\frac{c_k}{b_k}U_k.
\]

连接同一椭圆的两个解析极值点得到 (L_1,L_2)。最后按式 (16) 计算：

\[
\eta=
\frac{|P_1P_a|A(X_I,L_1)+|P_aP_2|A(X_I,L_2)}{|P_1P_2|}.
\]

实现使用有符号线角，并按 Cho 式 (5) 的坐标变换处理符号。输出同时保留：

- `converging_point_pa_px`；
- 两个椭圆的解析极值连线 `ellipse_lines`；
- `sqrt_a_over_b`；
- 两条线角和距离权重；
- 解析 eta；
- 完整投影矩阵分解得到的 eta 交叉检查值。

完整投影矩阵还输出源点、SDD、主点、探测器三轴、\(\phi,\theta,\eta\) 和当前模体坐标规范下的相位。机械角度仍需先把模体坐标变换到机器坐标；否则这些角度是相对于模体参考系的姿态。

### 当前联合优化的固定约束

当前联合模型采用一个明确的等效规范，避免源端偏移和探测器法向偏移互相补偿：

```text
source_offset_x = source_offset_y = source_offset_z = 0 mm
offset_n = 0 mm
```

因此优化变量只有 7 个机器参数：

```text
SID, SDD, offset_u, offset_v, tilt_u, tilt_v, tilt_n
```

模体旋转和平移仍作为 6 个 nuisance 变量参与重投影拟合，但不计入机器参数报告。这个约束不是声称机械源绝对没有偏移，而是选择了一个可重建、可复现的等效坐标规范；在该规范下，完整射线和重投影是确定的。

`joint_fit.py` 使用 13 维状态：前 7 项为上面的机器变量，后 6 项为模体刚体姿态。被固定的量不在优化向量中，结果 JSON 的 `fixed_parameters` 字段会再次明确记录它们。

### 每一帧

对第 `i` 帧，求出完整投影矩阵：

\[
q_{ij}\sim P_iX_j.
\]

随后分解得到：

- 每帧投影矩阵 `P_i`；
- 每帧来源点 `S_i`；
- 每帧源到探测器距离 `SDD_i`；
- 每帧相机坐标旋转矩阵；
- 每帧探测器 U/V/N 三轴；
- 每帧主点；
- 每帧重投影误差。

这些量直接定义了该帧射线，可用于逐帧投影/反投影。`calibration.json` 中的 `pic`、
`dlt_projection_matrices` 和 `source_circle_geometry` 保存了这些逐帧结果。

### 多帧共享量

利用固定平板的约束，对所有帧联合估计：

- 共享 `SDD`；
- 共享主点 `(u0,v0)`；
- 相对于图像中心的主点偏移；
- 联合重投影 RMSE。

这些是 Cho 投影模型中最稳定、最直接的系统级量。

### 源轨迹统计

把每帧来源点拟合为三维圆，可得到：

- 源轨迹半径，通常对应 `SID`；
- 源轨迹圆心；
- 旋转轴方向；
- 源轨迹径向、轴向、切向残差；
- 源点轨迹是否接近理想圆。

## 如何验收一次 RAW 运行

运行 `workflow.py::calibrate_raw()` 后，结果目录至少应包含：

```text
points_indexed.npy
calibration.json
```

从 `calibration.json` 检查有效帧数、每帧 DLT RMSE、共享 SDD/主点、源轨迹残差、
`fixed_source_joint_fit.optimizer_success` 和联合 RMSE。不同 RAW、阈值和粘连情况会改变
数值，因此本指南不固化某一批数据的旧结果。

`u0/v0` 是 DLT 相机分解中的投影主点，不应直接等同于仿真器机械配置中的 `offset_u/offset_v`。
探测器倾斜时，中心射线穿刺点和探测器机械中心并不重合。

## 为什么不能直接把结果写成机械 tilt

Cho 的每帧 DLT 是在模体三维坐标系中表达的。当前 RAW 中模体还有：

```text
phantom_offset = (10, 15, 20) mm
phantom_rotation_x = 2 deg
```

因此，若没有再建立“模体坐标系 → 扫描架坐标系”的刚体变换，模体姿态会和探测器姿态混合。此时可以可靠地报告：

- 每帧 `P_i`；
- 每帧来源点；
- 每帧探测器三轴；
- 共享 SDD/主点；
- 源轨迹和重投影误差。

但不能仅凭 Cho 的 DLT 分解把仿真器的机械 `tilt_u=1°`、`tilt_v=2°`、`tilt_n=3°` 原样唯一拆出来。要得到机械意义上的 tilt，还需要额外的扫描架坐标规范或联合拟合模体刚体姿态。

## 输出文件

- `points_indexed.npy`：`(views, 2*beads_per_ring, 2)` 的编号二维质心；
- `calibration.json`：`workflow.py` 保存的完整结果；
- `pic`、`dlt_projection_matrices`、`source_circle_geometry` 和 `fixed_source_joint_fit`：报告中的主要字段。

当前建议重建时优先使用每帧完整 `P_i`，或者使用共享 SDD/主点和每帧外参；不要把 `u0/v0` 直接当成机械 offset 填入重建配置。

## 实际 RAW 数据验证

Cho 的逐帧验证由 `pic.py::calibrate_stack_pic()` 完成；完整 RAW 入口是
`workflow.py::calibrate_raw()`。它读取 tracker 输出的亚像素编号点，逐帧计算：

- Cho 椭圆解析 `eta`；
- 完整 DLT 分解得到的 `eta`；
- 两者按直线方向的 180 度周期计算的差值；
- DLT 重投影 RMSE 和异环对径线交点 RMS。

验证报告应由当前输入和当前输出目录生成，不在本指南中固化某一批 RAW 的数值。
逐帧 `eta_i` 是虚拟探测器坐标系下的姿态量；它随机架相位变化并不表示探测器机械
姿态在扫描中真的改变。机械 `tilt_n` 必须由统一坐标系下的多帧模型估计，不能用
任意单帧 `eta_i` 直接替代。
