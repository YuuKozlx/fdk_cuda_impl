# CBCT 几何校正

本目录保留四条可独立阅读、可与论文逐步对照的业务链路：

```text
single_row/             单排等间距钢珠
identifiable_phantom/   大小环、异相位、外部标记的可编号模体
cho/                    Cho 2005 PIC/DLT
yang/                   Yang 2017 PIC/DLT
```

每条链路均按以下顺序阅读：

```text
tracker -> 编号二维点 -> 理论三维点 -> 几何求解 -> 验证 -> 导出
```

`workflow.py` 是业务入口，`api_example.py` 是 IDE 调用样例。追踪算法与模体结构强耦合，始终留在业务目录，不提供表面统一的通用 tracker。

## 工具层

```text
common/io.py          RAW、NumPy、JSON 读写
math/rigid.py         右手刚体变换 x_target = R x_source + t
math/dlt.py           点归一化、DLT、投影、RQ 相机分解
coordinates/          图像坐标、探测器 U/V/N 和坐标适配
geometry/             DLT、cone-vector、像素射线和 Siddon
```

工具层只处理数学，不知道钢珠编号、论文步骤或机械参数含义。Yang 的论文坐标常量位于 `yang/coordinate_adapter.py`，不会反向污染公共坐标工具。

## 坐标约定

公共图像坐标与 `fdk-test` 一致：

```text
RAW shape = [view, row, column]
u = column，向右增加
v = row，向下增加
image center = ((Nu - 1) / 2, (Nv - 1) / 2)
U x V = N
```

探测器是三维右手平面，而不是孤立的二维数组：

```text
D(u,v) = detS + u detU + v detV
```

翻转图像行列时必须同步改变完整的 `detS/detU/detV/N`。模体坐标变换为

```text
X_target = R X_paper + t,   R^T R = I, det(R) = +1
```

点编号是独立关系 `observed[k] -> model[permutation[k]]`，不能混入坐标变换。若论文投影矩阵满足 `q_paper ~ P_paper X_paper`，则

```text
P_target = H_image P_paper H_world
H_world  = [R^T  -R^T t; 0 0 0 1]
```

每次转换都必须满足投影闭合：

```text
project(P_target, X_target) == q_target
```

## DLT 与重建

逐帧 DLT 解算

```text
q_ij ~ P_i X_j
```

令 `P=[M|p4]`，则相机中心和像素射线为

```text
C = -M^-1 p4
d(u,v) = M^-1 [u,v,1]^T
```

因此逐帧 `P_i` 已完整定义迭代重建射线，不要求先分解 SID、SDD 或 detector offset。`geometry/dlt_to_conevec.py` 将同一射线写成

```text
d(u,v) = b + u r_u + v r_v
```

供现有 cone-vector 接口使用。DLT 的整体尺度不能唯一确定真实探测器平面距离；给定 `sdd_mm` 只是选取等效平面，不改变射线。

FDK 需要共享圆轨道模型。此时以逐帧 DLT 和源圆作为初值，在明确固定 `source_offset=(0,0,0)`、`offset_n=0` 的规范下联合估计等效 `SID/SDD/offset_u/offset_v/tilt_u/tilt_v/tilt_n`。这些参数可用于重建，但不自动等于唯一机械安装误差。

## 业务层边界

必须留在业务目录的内容包括：

- 模体理论点和标记定义；
- 检测、粘连分割、出界、永久停用和跨帧追踪；
- 首帧编号、环交换、相位、手性和镜像候选；
- Cho 的 `S/W/eta` 和 Yang 的 `O/yaw/ellipse/roll/pitch/gantry angle`；
- 各方法的联合拟合残差、固定量和可观测性约束。

公共工具的判断标准是：删除模体定义和论文参数含义后，该函数仍具有完整、唯一的数学意义。

## 保留文档

- `docs/DLT_GEOMETRY_AND_ANALYTIC_CALIBRATION.zh-CN.md`：DLT、坐标系、cone-vector、联合拟合与重建的完整推导；
- `docs/CALIBRATION_FORMULAS.zh-CN.md`：单排钢珠公式；
- `single_row/SINGLE_ROW_GUIDE.zh-CN.md`：单排全链路；
- `identifiable_phantom/IDENTIFIABLE_PHANTOM_GUIDE.zh-CN.md`：可编号模体全链路；
- `cho/CHO_GUIDE.zh-CN.md`：Cho 论文步骤；
- `yang/README.zh-CN.md` 与 `yang/PIC_DERIVATION.zh-CN.md`：Yang 流程与逐式推导。

实现调整后统一运行 `python -m unittest discover -s tools -p "test_*.py"`，并以重投影误差和真实 RAW 全链路结果作为行为闭合检查。
