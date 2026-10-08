# CBCT 几何校正代码与文档

本目录是当前实现的唯一 Python 标定入口。文档只描述现行代码和稳定的数学约定；
旧实验的固定数值、缓存结果和已经删除的脚本不再作为说明的一部分。

## 业务链路

```text
single_row/             单排等间距钢珠
identifiable_phantom/   大小环、异相位、外部标记的可编号模体
cho/                    Cho 2005 PIC/DLT
yang/                   Yang 2017 PIC/DLT
```

每条链路都按下面的边界组织：

```text
tracker -> 编号二维点 -> 理论三维点 -> 几何求解 -> 验证 -> 导出
```

每个业务目录中的 `workflow.py` 是可调用入口，`api_example.py` 是最小调用示例。
追踪器与模体结构强耦合，保留在对应业务目录，不把不同模体强行抽成同一个 tracker。

## 方法选择和输出边界

| 方法 | 必要先验 | 直接输出 | 重建用途 | 不能单独确定 |
|---|---|---|---|---|
| 单排钢珠 | 等间距、圆轨道、简化平板姿态 | SOD、SDD、主点、面内角、钢珠位置 | 简化圆轨道 FDK | 完整三轴 tilt、源/探测器偏移的唯一分解 |
| 可编号双环 | 已知三维点和可靠编号 | 每帧 DLT、源圆、固定源规范下的 7 参数 | DLT 迭代重建或等效 FDK | 无机器基准时的唯一机械 offset |
| Cho PIC/DLT | 同步双环、对置点关系、标记辅助编号 | 每帧 eta、穿刺点、DLT、源点、主点和 detector 三轴 | 逐帧射线；联合后可生成等效 FDK | 模体姿态与机械姿态的无约束分离 |
| Yang PIC/DLT | 双环、已知环参数和编号，默认 6/12 球每环 | O、eta、roll、pitch、S、W、SDD、主点、相位和 DLT | 逐帧射线；固定规范联合后可生成等效 FDK | 无机器基准时的真实逐帧机械 offset |

这里的“固定源规范”指 `source_offset=(0,0,0)`、`offset_n=0`。它解决参数规范自由度，
不表示真实机器的源绝对没有安装偏移。

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

## 文档阅读顺序

- `docs/DLT_GEOMETRY_AND_ANALYTIC_CALIBRATION.zh-CN.md`：总流程、坐标系、DLT、射线转换、联合拟合和重建边界；
- `docs/CALIBRATION_FORMULAS.zh-CN.md`：单排钢珠的逐式公式；
- `single_row/SINGLE_ROW_GUIDE.zh-CN.md`：单排钢珠从 RAW 到结果；
- `identifiable_phantom/IDENTIFIABLE_PHANTOM_GUIDE.zh-CN.md`：可编号双环和标记点的完整链路；
- `cho/CHO_GUIDE.zh-CN.md`：Cho 的 PIC、DLT、源轨迹和固定源规范联合拟合；
- `yang/README.zh-CN.md`、`yang/PIC_DERIVATION.zh-CN.md`：Yang 的 PIC、坐标适配和联合拟合；
- `../cbct_calibration_cpp/README.zh-CN.md`：独立 C++ 核心的接口和测试边界。

测试入口：

```powershell
python -m unittest discover -s tools -p "test_*.py"
```

行为闭合检查必须同时看 DLT 重投影 RMSE、源轨迹残差、联合拟合 RMSE 和输出的逐帧
几何；不能只看某个中位数，也不能用仿真真值初始化生产流程。
