# 几何与数据

本篇定义几何和数据契约。算法允许哪些几何见 [算法矩阵](algorithms.md#公共支持矩阵)。主要依据：[参数类型](../include/YKCBCT/geometry/YkGeometryBuilderTypes.hpp)、[轨迹 builder](../include/YKCBCT/geometry/YkTrajectoryGeometryBuilder.hpp)、[探测器 builder](../include/YKCBCT/geometry/YkDetectorGeometryBuilders.hpp)、[系统 builder](../include/YKCBCT/geometry/YkSystemGeometry.hpp)。

## 坐标与单位

使用右手世界坐标，规则机架绕世界 Z 轴旋转。长度以 mm、角度以 rad 表示。零角度、零偏移、零倾斜时：

| 几何量 | 世界坐标 |
| --- | --- |
| 源点 | `(0, -SID, z)` |
| 探测器标称主点 | `(0, SDD-SID, z)` |
| U 方向 | `(+1, 0, 0)` |
| V 方向 | `(0, 0, +1)` |
| N 方向 | `(0, -1, 0)`，由探测器指向源点 |

`U × V = N`。角度正向为绕 Z 的右手旋转，rotation_direction=+1/-1 表示角度递增/递减；不要把 RAW 图像显示中的上下翻转当成世界坐标约定。

SID 是规则轨迹的标称源到旋转轴距离，SDD 是标称源到探测器主点距离，要求 `SDD > SID > 0`。偏移或倾斜后，最终射线几何由 builder 输出决定，不能把标称距离当作所有逐视图派生距离。

## 规则圆轨迹与螺旋

圆轨迹使用 `SRegularCircularScanSpec`，螺旋使用 `SRegularHelicalScanSpec`。两者均要求 total_views>0、views_per_turn>=2，方向为 ±1。

```text
theta[i] = start_angle_rad + rotation_direction * 2*pi*i/views_per_turn
圆轨迹：z[i] = circular.z_mm
规则螺旋：z[i] = start_z_mm + pitch_mm_per_turn*i/views_per_turn
```

圆轨迹 total_views<views_per_turn 表示短扫，等于表示一整圈，大于表示多圈；一圈通常不重复提交 2π 终点。它们只是 builder 可表达的采集，不代表所有解析算法都允许该范围。

规则螺旋 pitch 是带符号的物理进床量，正负控制 Z 方向，与机架旋转方向无关；pitch=0 会被螺旋 builder 拒绝，应选择 Circular。显式 `SHelicalTrajectorySpec` 则按绝对角度计算 `z=start_z+pitch*angle/(2*pi)`；系统 builder 会转换截距与 pitch 符号，不应再重复转换。

`source_offset_mm` 在随机架旋转的局部 X/Y/Z 中移动焦点，不同时移动探测器。源偏移必须由 builder 烘焙进最终源坐标。

## 探测器姿态与采样

`SDetectorPoseSpec.offset_unv_mm` 分量顺序是 **U、N、V**，即 x=U、y=N、z=V，不能按变量名误读成 U/V/N。tilt 按 U、V、N 顺序绕当前轴施加；偏移沿最终倾斜后的轴解释。

Flat：channels/rows 是列/行数量，channel_size_mm/row_size_mm 是物理采样步长。数组中心 `(Nu-1)/2,(Nv-1)/2` 在偶数尺寸时落在四个像素中心之间。

Cyl：channel_arc_mm 是圆柱面相邻通道的弧长，曲率半径 R=curvature_radius_mm，角步长为 arc/R。U offset 按圆弧旋转通道中心并更新切向，不能当成平板平移。V offset 是轴向平移，N offset 改变表面位置。迭代算子不要求 R=SDD；解析路径另有约束。

## 逐视图几何

Flat `SConeProjGeomVec` 保存 `src/detS/detU/detV/angle`：

```text
像素中心 P(u,v) = detS + u*detU + v*detV
detS = 像素 (0,0) 的中心，不是数组中心
detU/detV = 一像素的物理增量，不是单位向量
angle.x = 该视图角度
```

Cyl `SCylConeProjGeomVec` 保存源点、表面数组中心、通道切向步长、行向步长和 viewParameters；半径通过 `cylDetectorRadius()` / `setCylDetectorRadius()` 访问，避免直接依赖其存储位置。Cyl detectorCenter 与 Flat detS 的含义不同，不能直接复制字段。

由 Cyl frame 推导单位切向 T、轴向 V、径向 Rn（从圆柱轴指向表面）、圆柱轴中心 C，令 `phi=(u-(Nu-1)/2)*arc/R`，像素中心为：

```text
P(u,v) = C + R*(Rn*cos(phi)+T*sin(phi))
           + (v-(Nv-1)/2)*row_size*V
```

逐视图几何是底层算子的实际输入。主点、探测器法线、中心射线、主轴与 FDK 距离在 prepare 阶段派生；kernel 不再接收并重复解释采集 offset。对实际几何执行刚体变换时，点用完整变换、方向用旋转，体积必须位于相同世界坐标。

公共规则 Session 根据 `SSystemConfig` 生成逐视图几何，执行请求不带第二份角度。安装 SDK 虽提供显式轨迹及逐视图标定 builder，当前 `SSystemSpec` 没有任意外部 geometry 数组入口；任意校准 geometry 的内部测试能力不能直接等同于 Session 导出能力。

## 体积网格与正投姿态

`SVolumeGridSpec.center_mm` 是体素阵列中心，不是强制的旋转轴或探测器主点。体素尺寸可各向异性。索引为整数的体素中心：

```text
X(x) = center.x + (x-(nx-1)/2)*voxel_x_mm
Y(y) = center.y + (y-(ny-1)/2)*voxel_y_mm
Z(z) = center.z + (z-(nz-1)/2)*voxel_z_mm
```

`SVolGeom.origin()` 是第一个体素中心。体积外边界比最外层中心再向外半个体素；射线盒求交与中心采样不能混用。

`forward_projection_pose` 仅供独立 ForwardProjection pipeline，Euler 旋转绕体积中心作用，按固定世界 X、Y、Z 轴依次旋转，列向量合成矩阵为 Rz*Ry*Rx；后端通过逆变换射线几何读取原数组。重建 Session 会清除此正投姿态，不能用它替代真实重建几何。不要与探测器 tilt 的当前局部轴顺序混用。刚体定义见 [变换实现](../include/YKCBCT/geometry/YkRigidTransform.hpp)。

## 非标准平板轨迹

两类轨迹均选择 Flat，参数放在 `SSystemSpec.nonstandard` 对应分支；公共边界只允许 FDK 和 ForwardProjection。这里的 FDK 会分派到专门管线。

### PlanarSourceOrbit

探测器和对象固定，源沿 XZ 平面绕 Y 轴方向运动，角度是源位置参数。填写 total_views、start_angle_rad、scan_range_rad、rotation_direction、sid_mm、detector_y_mm、source_orbit_radius_mm、source_lateral_offset_mm。重建还需固定窗口 reconstruction_channels/rows 和 resample_mode。

`InPlaneRotated` 保持物理平面并旋转局部 U/V；`PerspectiveOrthogonal` 一次投影映射到以源到体积中心射线为法线的虚拟平面。采样与几何使用同一映射，边界填零。窗口取整残差保留为主点偏移。当前 Session 将该轨迹几何按 volume.center 平移，与普通规则轨迹对体积中心的处理不同。

FDK 窗口还需满足设备线性纹理对齐要求。不要把采集全幅尺寸与重建窗口尺寸互换；缓冲区始终按采集探测器尺寸提供。

### NikonTilted

倾斜改变零角度源/探测器姿态，后续视图仍绕世界 Z 轴旋转；对象与扫描轴保留在世界坐标。填写 total_views、start_angle_rad、scan_range_rad、rotation_direction、sid_mm、sdd_mm、tilt_rad、tilt_axis_azimuth_rad、source_offset_mm、rotation_axis_pivot_mm；探测器姿态仍来自 flat_detector.pose。

“倾斜初始机架”不能解释为整个世界坐标系跟着倾斜。FDK 需满足探测器线性纹理对齐要求。两类非标准轨迹的几何与重建验证入口见 [测试](testing-and-validation.md)。

## 数组与物理单位

| 数据 | 连续布局 | 元素索引 |
| --- | --- | --- |
| 投影 | float32 `[view][v][u]` | `(view*Nv+v)*Nu+u` |
| 体积 | float32 `[z][y][x]` | `(z*Ny+y)*Nx+x` |

公共 Buffer 没有 stride、pitch 或多通道字段。RAW 文件不携带尺寸与几何，必须同时记录配置、float32 类型和字节序；不要把带行填充的图像直接当作连续数组。

以 mm 为长度时，线衰减体积为 mm⁻¹，投影为无量纲线积分 `p=∫mu dl`。重建输入应为线积分，探测器计数、强度、对数与散射校正属于上游转换；本体 Session 不负责多能谱/BHC 等应用层物理预处理。负值处理和显示窗宽不是数组坐标变换。

## 几何验证要点

检查零角度射线、旋转方向、pitch 正负、偏移轴、偶数尺寸中心、非零体积中心、刚体变换一致性，再检查图像。几何 builder 成功仅表示其输入满足构造条件；解析算法仍需满足圆轨迹、覆盖、等间距和姿态等自己的要求。尤其不能靠把不支持的 offset 清零来获得“成功”。
