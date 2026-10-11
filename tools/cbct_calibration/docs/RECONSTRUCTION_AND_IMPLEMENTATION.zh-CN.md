# 从标定结果到重建与工程接口

本文说明三件事：标定输出如何变成重建射线，逐帧 DLT 如何驱动 FP/BP 与代数重建，以及全局七参数如何进入 FDK。最后给出 Python/C++ 边界、输出结构和测试要求。

## 1. 三种重建几何

### 1.1 每帧投影矩阵

空间模体标定直接得到：

\[
P_i\in\mathbb R^{3\times4},
\qquad
q\sim P_iX.
\]

`P_i` 定义从三维坐标到原始探测器像素的投影。它适合：

- 逐帧机械几何不完全一致；
- 不希望强行拆分源偏移和探测器偏移；
- 使用 SIRT、SART、OS-SART 等迭代重建；
- 保留标定时实际观测到的每帧射线。

### 1.2 每帧 cone-vector/射线几何

同一射线可以写成：

```text
source
ray00
rayU
rayV
```

或有限探测器表示：

```text
src
detS
detU
detV
```

前者不需要选择有限探测器平面；后者需要选定一个等效平面，但只要每个像素对应的直线不变，Siddon 路径长度也不变。

### 1.3 全局七参数

传统圆轨道 FDK 通常接收：

```text
SID、SDD
offset_u、offset_v、offset_n
tilt_u、tilt_v、tilt_n
起始角、旋转方向和逐帧角度
```

当前标定固定源偏移和 `offset_n` 后联合估计其余七项。它是选定规范下的等效圆轨道模型，不一定等于唯一机械安装值。

## 2. DLT 矩阵如何定义一条射线

### 2.1 相机中心

将投影矩阵写成：

\[
P=[M\mid p_4].
\]

相机中心 `C` 满足 `PC_h=0`，因此：

\[
C=-M^{-1}p_4.
\]

这就是该帧在 DLT 三维坐标系中的源点。

### 2.2 像素反投影方向

像素 `(u,v)` 的齐次图像坐标为：

\[
\tilde q=(u,v,1)^T.
\]

空间射线方向可取：

\[
d(u,v)=M^{-1}\tilde q.
\]

于是射线为：

\[
X(t)=C+t\,d(u,v),\qquad t\ge0.
\]

把矩阵乘法展开：

\[
d(u,v)=r_{00}+u,r_u+v,r_v,
\]

其中：

\[
r_{00}=M^{-1}\begin{bmatrix}0\\0\\1\end{bmatrix},
\quad
r_u=M^{-1}\begin{bmatrix}1\\0\\0\end{bmatrix},
\quad
r_v=M^{-1}\begin{bmatrix}0\\1\\0\end{bmatrix}.
\]

这四个三维量正是 GPU DLT 投影器使用的规范相机。

### 2.3 RQ 分解能解释什么

若需要解释内参与相机姿态，可对 `M` 做 RQ 分解：

\[
M=KR,
\]

经过对角符号和尺度规范化后：

\[
K=
\begin{bmatrix}
f_u&s&u_0\\
0&f_v&v_0\\
0&0&1
\end{bmatrix},
\qquad RR^T=I.
\]

这里 `f_u/f_v` 是像素单位的等效焦距，`s` 是像素轴 skew，`u_0/v_0` 是代数相机主点，`R` 是世界到相机的旋转。对理想方形像素平板通常期望 `s` 接近零，但 DLT 本身不要求它严格为零。

必须注意：RQ 分解解释的是当前 DLT 坐标规范中的相机。只有在已经建立“模体坐标到机器坐标”的刚体关系后，`R` 和主点才能继续与机械 detector tilt/offset 比较。RQ 也不能消除源偏移、探测器偏移和世界原点之间的规范自由度。

### 2.4 为什么 DLT 的全局尺度不影响重建

若：

\[
P'=\lambda P,\qquad \lambda\ne0,
\]

则相机中心不变，而 `M^{-1}` 只产生一个公共反比例尺度。射线集合仍是同一组直线。实现会统一归一化每帧 `ray00/rayU/rayV`，并根据重建体中心选择射线正方向。

### 2.5 图像 v 轴变换

如果输入 DLT 使用 `v` 向上，而原始投影数组使用行号向下，需要对整个投影矩阵左乘：

\[
H_v=
\begin{bmatrix}
1&0&0\\
0&-1&N_v-1\\
0&0&1
\end{bmatrix},
\qquad
P_{down}=H_vP_{up}.
\]

不能只在 kernel 内令 `v=-v`，否则会漏掉 `N_v-1` 的平移。

## 3. 模体坐标和重建坐标

### 3.1 DLT 并不依赖模体继续存在

模体只是获得 `P_i` 的三维控制点。一旦 `P_i` 已求出，投影矩阵描述该扫描系统相对于标定三维坐标的射线，不会在被扫描物换成其他对象后失效。

但重建体的坐标必须与 `P_i` 使用同一世界参考。如果 `P_i` 仍位于模体坐标系，那么体素中心也必须用该模体坐标表达，或者先把矩阵转换到目标重建坐标系。

### 3.2 刚体转换

若：

\[
X^w=R_{wm}X^m+t_{wm},
\]

则：

\[
P_i^w=P_i^mH_{mw},
\]

\[
H_{mw}=
\begin{bmatrix}
R_{wm}^T&-R_{wm}^Tt_{wm}\\
0&1
\end{bmatrix}.
\]

矩阵转换后，模体旋转和平移已经包含在 `P_i^w` 中，重建时不能再次给体数据施加相同姿态。

### 3.3 重建原点和体积范围

DLT 不会自动指定重建体大小。重建原点、体素尺寸和维度由用户根据待重建对象定义。标定模体重建时，可以用已知模体外包围盒或其中心作为体积范围；实际扫描对象应使用扫描床、机械等中心或目标 ROI 的物理定义。

如果只知道模体坐标而不知道机器等中心，可先在模体坐标中重建，再用源圆拟合得到的刚体变换转换结果。对于代数重建，几何正确性只要求体素坐标与 `P_i` 一致。

## 4. DLT 到有限 cone-vector

### 4.1 等效探测器平面

定义：

\[
r_{00}=M^{-1}(0,0,1)^T,
\qquad
r_u=M^{-1}(1,0,0)^T,
\qquad
r_v=M^{-1}(0,1,0)^T.
\]

选择任意非零公共尺度 `s`，定义：

\[
detS=C+s\,r_{00},
\qquad
detU=s\,r_u,
\qquad
detV=s\,r_v.
\]

于是像素中心 `(u,v)` 对应的等效探测器点为：

\[
D(u,v)=detS+u\,detU+v\,detV
=C+sM^{-1}(u,v,1)^T.
\]

因此：

\[
D(u,v)-C=s\,d(u,v),
\]

它和 DLT 反投影方向严格共线。这不是近似，也不要求 `detU、detV` 与中心射线正交；一般 DLT 对应的代数平面可能带有 skew。

由右手探测器法向：

\[
n=\frac{detU\times detV}{\|detU\times detV\|},
\]

该等效平面的垂直源距为：

\[
L_0=|(detS-C)\cdot n|.
\]

如果调用方希望等效平面具有指定正距离 `SDD_e`，只需选择：

\[
s=\frac{SDD_e}{|r_{00}\cdot n_0|},
\qquad
n_0=\frac{r_u\times r_v}{\|r_u\times r_v\|}.
\]

改变 `s` 会同比改变 `detS-C、detU、detV`，但每个像素的射线不变。这个等效平面可以满足 cone-vector 接口，却不是从 DLT 唯一恢复的机械平板。代码中的 `dlt_to_conevec(..., sdd_mm=None)` 使用代数尺度；传入 `sdd_mm` 时只调整这个公共尺度，不改变任何射线。

### 4.2 SConeProjGeomVec 输入

已有库几何：

```text
ray00 = detS - src
rayU  = detU
rayV  = detV
source = src
```

也会被统一转换为相同 `SDltRayGeometry`。因此 DLT 矩阵和现有 vector 几何共用一套 FP/BP 实现，不需要在 kernel 内分别处理正交或非正交探测器。

## 5. Siddon 正投与匹配反投

### 5.1 正投定义

对射线 `r`，Siddon 遍历体素网格，得到相交体素 `k` 和毫米路径长度 `l_{rk}`：

\[
(Ax)_r=\sum_k l_{rk}x_k.
\]

这是真正的射线积分，不是假设每个体素中心都被穿过的体素驱动正投。

### 5.2 严格匹配反投

匹配 BP 使用同一组路径长度：

\[
(A^Ty)_k=\sum_r l_{rk}y_r.
\]

不同射线可能同时更新同一体素，所以 CUDA 版本使用原子加法。验证条件是：

\[
\langle Ax,y\rangle=\langle x,A^Ty\rangle.
\]

当前回归在浮点误差范围内满足伴随性，测试阈值为 `1e-5`。

### 5.3 快速体素驱动 BP

还提供体素驱动近似反投：

1. 对每个体素中心计算其 DLT 投影位置；
2. 用最近邻或双线性插值采样探测器；
3. 乘局部体素弦长/AABB 权重；
4. 累加各帧贡献。

该模式不需要原子写冲突，速度更高，但不是正投的严格转置。它适合工程重建，不能用于依赖严格伴随算子的数值证明。

## 6. 代数重建后端

### 6.1 方法枚举

`EDltAlgebraicMethod` 明确提供：

```text
Sirt
Sart
Ossart
```

统一以 subset 解释：

- SIRT：一个 subset 包含全部视角；
- SART：每个 subset 只包含一帧；
- OS-SART：按 `ossart_subset_size` 组织多帧 subset。

一次 `sweep` 始终表示访问所有 subset 一次，因此 SIRT 每 sweep 更新一次，SART 每 sweep 更新 `views` 次，OS-SART 每 sweep 更新 `subset_count` 次。

### 6.2 更新公式

对子集 `s`：

\[
r_s=b_s-A_sx,
\]

\[
x\leftarrow
\max\left(0,
x+\lambda\,
\frac{A_s^T(r_s/R_s)}{C_s+\epsilon}
\right),
\]

其中：

\[
R_s=A_s\mathbf1,
\qquad
C_s=A_s^T\mathbf1.
\]

`relaxation` 对应 `lambda`，`nonnegative` 控制非负约束。

### 6.3 归一化模型

`EDltAlgebraicWeightModel` 提供：

```text
ExactSubset
TigreApprox
```

`ExactSubset` 对每个 subset 用完整 DLT Siddon 算子计算 `A_s*1` 和 `A_s^T*1`，精确但需要三维列权重。

`TigreApprox` 保留精确行权重，将列权重沿 z 平均为二维权重图，显著降低内存。它是近似归一化，不改变正投射线，也不需要用户配置缓存字节数。

### 6.4 BP 模型

`EDltBackprojectionModel` 提供：

```text
MatchedSiddon
VoxelDriven
```

体素驱动模式还能选择：

```text
Nearest
Bilinear
```

如果算法依赖严格 `A^T`，必须选择 `MatchedSiddon`；如果主要关心速度和最终重建质量，可以评估 `VoxelDriven + Bilinear`。

## 7. C++ DLT FP/BP 接口

### 7.1 几何结构

```cpp
struct SDltProjectionMatrix {
    std::array<double, 12> value;
};

struct SDltRayGeometry {
    float4 source;
    float4 ray00;
    float4 rayU;
    float4 rayV;
};
```

矩阵按行存储。三维坐标单位必须为毫米，`u/v` 是原始像素坐标。

### 7.2 生成射线

```cpp
std::vector<YK::DltFpBp::SDltRayGeometry> rays;
bool ok = YK::DltFpBp::buildRayGeometry(
    projection_matrices,
    detector_channels,
    detector_rows,
    volume_geometry.center,
    rays);
```

`volume_geometry.center` 只用于确定射线正方向，不会修改源点或像素投影关系。

### 7.3 独立 FP/BP

```cpp
YK::DltFpBp::DltSiddonProjector projector;
projector.prepare(rays, volume_geometry,
                  detector_channels, detector_rows, device_id);

projector.forward(device_volume, device_projection,
                  false, stream);

projector.backproject(device_projection, device_volume,
                      true, stream);

projector.backprojectVoxelDriven(device_projection, device_volume,
                                 true, true, stream);
```

`forward(..., accumulate)` 控制是否累加到已有投影；`backproject(..., clear_volume)` 控制是否先清零体数据。

### 7.4 迭代重建

```cpp
using namespace YK::DltFpBp;

SDltAlgebraicConfig config;
config.method = EDltAlgebraicMethod::Ossart;
config.weight_model = EDltAlgebraicWeightModel::ExactSubset;
config.ossart_subset_size = 9;
config.relaxation = 0.8f;
config.backprojection_model =
    EDltBackprojectionModel::MatchedSiddon;

DltAlgebraicReconstructor reconstructor;
reconstructor.prepare(projection_matrices, volume_geometry,
                      detector_channels, detector_rows,
                      config, stream, device_id);
reconstructor.iterate(device_projection, device_volume,
                      sweeps, stream);
```

`prepare()` 也可以直接接受 `SDltRayGeometry` 或 `SConeProjGeomVec`。

## 8. DLT 参数如何进入 FDK

### 8.1 为什么不能直接把 P 当作 FDK 参数

FDK 的预权重、滤波坐标和反投权重通常假设共享圆轨道和明确的有限探测器平面。一般 `P_i` 只唯一决定射线，不唯一决定机械 SID、SDD、offset 和探测器平面尺度。

因此不能从每帧 RQ 分解得到一组波动参数后直接塞给标准 FDK。正确路线是把全部帧约束到一个显式圆轨道机器模型。

### 8.2 当前 FDK 路线

```text
每帧 DLT
→ 源点和源圆
→ 模体到扫描仪规范的刚体初值
→ 固定 source_offset=0、offset_n=0
→ 联合估计 SID/SDD/offset_u/offset_v/tilt_u/tilt_v/tilt_n
→ 用同一组参数生成全部帧 SConeProjGeomVec
→ FDK
```

联合拟合还优化模体六自由度，但模体姿态只用于把已知控制点映射到扫描仪坐标；最终扫描未知对象时不把标定模体姿态填入被扫描对象配置。

### 8.3 参数符号

联合模型的公共零度约定为：

```text
source = (0, -SID, 0)
radial = 从源指向等中心
U = +x
V = +z
N = U × V
detector_center = -(SDD-SID)*radial
                  + offset_u*U + offset_v*V
```

然后按 `tilt_u → tilt_v → tilt_n` 的代码顺序旋转探测器基。参数导出必须与重建器使用相同旋转顺序和法向定义。

## 9. Python 标定接口

主要接口都是普通函数，不要求通过命令行运行。

### 9.1 单排钢珠

```python
from tools.cbct_calibration.single_row.single_row_workflow import (
    SingleRowConfig, calibrate_single_row,
)

result = calibrate_single_row(SingleRowConfig(
    raw_path=raw_path,
    output_directory=output_dir,
    views=views,
    bead_count=bead_count,
    spacing_mm=spacing_mm,
    pixel_size_mm=(pixel_u_mm, pixel_v_mm),
))
```

### 9.2 可编号双环

```python
from tools.cbct_calibration.identifiable_phantom.workflow import (
    IdentifiablePhantomConfig,
    calibrate_identifiable_phantom,
)

report = calibrate_identifiable_phantom(
    IdentifiablePhantomConfig(...)
)
```

### 9.3 螺旋模体

```python
from tools.cbct_calibration.spiral_phantom.workflow import (
    SpiralCalibrationConfig,
    calibrate_spiral_phantom,
)

report = calibrate_spiral_phantom(SpiralCalibrationConfig(...))
```

### 9.4 Cho 和 Yang

```python
from tools.cbct_calibration.cho.workflow import calibrate_raw as calibrate_cho
from tools.cbct_calibration.yang.workflow import calibrate_raw as calibrate_yang

cho_report = calibrate_cho(cho_config)
yang_report = calibrate_yang(yang_config)
```

业务 workflow 负责 RAW、追踪、编号、掩码、几何求解和报告；底层 `pic.py`/`dlt.py` 只接受已经编号的点，不负责图像检测。

## 10. 独立 C++ 标定接口

独立工程位于：

```text
tools/cbct_calibration_cpp/
```

它有自己的 `CMakeLists.txt`，不加入主库构建。

### 10.1 已编号点入口

```cpp
cbct::calibration::PipelineConfig config;

auto result = cbct::calibration::calibrateIndexedPoints(
    indexed_points,
    world_points,
    usable,
    views,
    config);
```

数组布局：

```text
indexed_points[view * targets + target]
usable[view * targets + target]
world_points[target]
```

每帧至少 6 个 `usable` 点，所有可用二维点必须有限。

### 10.2 RAW 入口

```cpp
PipelineConfig config;
config.tracker.raw_path = raw_path;
config.tracker.views = 360;
config.tracker.image = {1024, 1024};
config.phantom = phantom_spec;
config.pixel = {0.417, 0.417};

CalibrationResult result = calibrateRaw(config);
```

当前 C++ RAW 入口主要覆盖可编号双环；单排钢珠有独立 tracker。Cho/Yang 的完整 PIC 和螺旋 RAW 业务仍由 Python 提供。

### 10.3 C++ 输出

`CalibrationResult` 包含：

```text
tracking.points/observed/merged/usable
每帧 DltCamera
SourceCircle
MachineGeometry
模体旋转向量和平移
rotation_direction
joint_rmse_px/max_point_error_px
迭代次数和收敛状态
```

`writeCalibrationJson()` 可把结果保存为 JSON。

## 11. 输出文件和字段责任

### 11.1 追踪输出

典型文件：

```text
points.npy
observed.npy
merged.npy
usable.npy
tracking_report.json
tracking_diagnostics.png
```

`points.npy` 可以在粘连帧保留预测位置；只有 `usable.npy=true` 的点才是几何测量。任何下游程序只读取 `points` 而忽略掩码都会产生伪精确结果。

### 11.2 标定输出

`calibration.json` 应至少包含：

```text
坐标约定和像素尺寸
三维模型点定义
二维—三维对应关系
frame_id
每帧 P_i 和 DLT 误差
源圆
PIC逐帧结果（适用时）
固定源规范和七参数结果
模体姿态
优化与有效点统计
```

输出中的距离必须标明毫米或像素，角度必须标明弧度或度。

### 11.3 重建配置

迭代重建优先直接读取 `P_i`。FDK 配置只读取联合拟合后的全局参数和扫描角；不读取标定模体的位姿作为被扫描对象位姿。

## 12. 验证层级

### 12.1 几何闭合

对所有有效控制点检查：

\[
\hat q_{ij}=\pi(P_iX_j),
\qquad
e_{ij}=\|q_{ij}-\hat q_{ij}\|.
\]

必须同时保存均方根、最大值、逐帧和逐目标误差。单一总 RMSE 可能掩盖局部编号错误。

### 12.2 射线闭合

对多个像素比较 DLT 反投方向与 cone-vector 像素方向，要求两者共线；同时验证经过坐标转换后的三维控制点仍投影到相同 `u/v`。

### 12.3 伴随性

匹配 Siddon FP/BP 使用随机体数据和随机投影验证：

\[
\epsilon=
\frac{|\langle Ax,y\rangle-\langle x,A^Ty\rangle|}
{\max(|\langle Ax,y\rangle|,|\langle x,A^Ty\rangle|,\epsilon_0)}.
\]

体素驱动 BP 不参加这项严格测试。

### 12.4 重建闭合

至少比较：

- 已知真值几何的重建；
- 逐帧 DLT 的迭代重建；
- 七参数等效几何的 FDK；
- 目标边缘位置、双边缘、拉伸、中心偏移和轴向倾斜。

## 13. 构建和回归

统一回归环境位于 `tools/cbct_calibration/container/`。镜像包含 Python、NumPy、SciPy、pytest、CMake、Ninja、GCC 和 Eigen。构建产物写入 `out/`，不写进源码目录。

当前回归覆盖：

- Python 五条业务链的数学与追踪单元测试；
- 坐标刚体变换和 DLT 射线闭合；
- Cho/Yang PIC 合成几何；
- 粘连屏蔽和后续恢复；
- C++ DLT、源圆和联合拟合；
- DLT FP/BP 射线、路径长度和伴随性；
- SIRT/SART/OS-SART 枚举与归一化模型。

真实 RAW 测试必须从原始投影重新生成点和结果，不能读取旧缓存冒充本次运行。缺少 RAW 时只能报告合成回归和程序可编译，不能声称真实数据重新验证通过。

## 14. 实现边界

- DLT 直接重建不需要唯一机械参数；
- FDK 需要共享圆轨道，因此使用等效七参数；
- `P_i` 的坐标系必须与重建体一致；
- 真实源偏移和探测器偏移在没有外部基准时不唯一；
- PIC 逐帧量必须经过统一三维坐标转换后才能比较；
- 快速体素 BP 不是严格伴随；
- C++ 标定核心没有完全复制 Python 的全部模体业务逻辑；
- 实验性 TOML 不代表对应导出链已经成为稳定公开接口。
