# Flat-panel CBCT 几何标定 Python 原型

完整的路线说明、DLT 到 cone-vector 推导、7 参数联合求解和 Cho/Yang
PIC 偏移解释见：

[`DLT_GEOMETRY_AND_ANALYTIC_CALIBRATION.zh-CN.md`](DLT_GEOMETRY_AND_ANALYTIC_CALIBRATION.zh-CN.md)

使用通用椭圆中心连线初始化，后续几何关系参考 Wu 等，
*Geometric Calibration of Cone-beam CT with a Flat-panel Detector*，
2011 IEEE NSS/MIC，pp. 2952-2955。输入已经对应好的钢珠中心坐标；
输出静态圆轨道几何参数、钢珠空间位置和重投影误差。解析部分只依赖 NumPy，
可选用 SciPy 联合精修，用 Matplotlib 输出诊断图。

## 标定哪些参数

| 输出 | 意义 | 单位 |
| --- | --- | --- |
| `sdd_mm` / D | 源到探测器平面的距离，本模型为中心射线距离 | mm |
| `sod_mm` / R | 源到旋转轴距离，工程中也常记 SID | mm |
| `u0_px`, `v0_px` | 中心射线与探测器的交点，即主点，并非必然等于图像中心 | pixel |
| `eta_deg` | 探测器面内旋转角 | degree |

标定按阶段输出，便于逐步检查和决定是否继续：

1. `stage1_ellipse_and_inplane_angle`：拟合每颗珠的通用椭圆，检查椭圆拟合条件数和中心直线残差，得到 `eta`。
2. `stage2_sdd_and_principal_point`：使用旋正后的椭圆系数，得到 SDD、`u0`、`v0`。
3. `stage3_sod_and_bead_positions`：先得到无量纲轨道，再用已知珠间距恢复 SOD 和珠子三维位置。
4. `stage4_joint_refinement`：只有显式使用 `--refine` 时才执行，用原始投影和距离约束做联合精修。

因此可以只使用前两阶段检查平板几何，或者在第三阶段检查钢珠间距约束；第四阶段不是解析解的必要组成部分。

论文同时讨论角步长 Δβ；本原型**使用输入角度，不独立估计真实角步长**。
等角度采样时输出的 `delta_beta_deg` 是输入步长；非等角度时为 `null`，
`angular_coverage_deg` 记录首末采样点的实际角度跨度。论文通过跨周期投影相似度寻找一周长度，
实际实现需要多于一周的数据和可靠的重复周期搜索，不能从任意单周数据直接假定已标定。

两个面外倾角固定为零，未建模每帧摆动、轴晃动、焦点漂移和探测器畸变。
论文认为忽略面外倾角在其条件下影响较小；高分辨率、大锥角设备不能直接沿用这个结论。
这里不是完整八参数标定，也没有把结果直接接入 C++ 重建器。

## 为什么钢珠间距可以解出几何

设珠子在旋转坐标系中的轨迹是 `(r*cos(beta), r*sin(beta), z)`。
消除面内旋转后，投影坐标满足论文式 (6)：

```text
u = u0 - D*r*sin(beta)/(R + r*cos(beta))
v = v0 + D*z/(R + r*cos(beta))
```

消去 beta 后是椭圆 `a*(u-uc)^2 + b*(v-vc)^2 = 1`。注意：
**椭圆中心不等于三维圆心的投影**，不能直接平均轨迹坐标当作主点。

1. 对每颗珠的轨迹直接拟合通用椭圆 `A*u^2+B*u*v+C*v^2+D*u+E*v+F=0`，
   包含交叉项，使用中心化/缩放后的 Halir-Flusser 直接最小二乘算法，约束 `4*A*C-B^2>0`。
2. 令 `Q=[[A,B/2],[B/2,C]]`，由 `center=-0.5*solve(Q,[D,E])` 求椭圆中心。
   当前零面外倾角模型中，这些中心都在旋转轴投影上；以总最小二乘拟合中心直线，
   方向向量取 `d_v>0`，用 `eta=atan2(d_u,d_v)` 确定面内旋转角。
3. 把椭圆写成 `(p-center)^T*Q*(p-center)=1`，按同一个面内角旋正中心和二次型：
   `center'=rotation*center`，`Q'=rotation*Q*rotation.T`。
   直接取 `Q'` 对角元素为 `(a,b)`，与旋正中心组成 `(uc,vc,a,b)`。
4. 由 `(vc-v0)^2 = 1/b + (a/b)*D^2`，对珠子两两作差，线性求 `v0,D^2`；
   `u0` 取各椭圆横向中心的平均。
5. 椭圆给出 `rho=r/R=1/sqrt(1+a*D^2)` 和 `zeta=z/R=(vc-v0)*(1-rho^2)/D`。
   使用所有视角的实际角度对每颗珠求连续相位 `alpha_k`，得到无量纲三维位置，
   已知任一珠对距离即可恢复 R。这里不使用径向对，也不要求存在 180° 对径视角。
6. 可选精修：同时优化五个几何参数与所有珠子位置，最小化像素重投影残差和已知间距残差。

### 式 (12)-(13) 的计算细节

椭圆只给出轨道半径比例 `rho` 和轴向比例 `zeta`，还需要从整条投影轨迹恢复钢珠的相位 `alpha`。
这里完全不使用径向对，也不要求有恰好 180 度的视角。令旋正后的物理坐标为
`u_i=u_i*-u0*`、`v_i=v_i*-v0*`，第 `i` 帧角度为 `beta_i`，前一阶段求得 SDD 为 `D`，并定义：

```text
x = rho*cos(alpha),  y = rho*sin(alpha),  z = zeta
```

将三维轨道旋转到第 `i` 帧：

```text
x_i = x*cos(beta_i) - y*sin(beta_i)
y_i = x*sin(beta_i) + y*cos(beta_i)
```

由透视投影式 `u_i=-D*y_i/(1+x_i)`、`v_i=D*z/(1+x_i)`，消去分母后得到每帧两行线性方程：

```text
(u_i*cos(beta_i)+D*sin(beta_i))*x
 +(-u_i*sin(beta_i)+D*cos(beta_i))*y = -u_i

(v_i*cos(beta_i))*x +(-v_i*sin(beta_i))*y - D*z = -v_i
```

把全部视角堆叠为 `A*[x,y,z]=b`，用缩放最小二乘求解。随后：

```text
rho = sqrt(x*x+y*y)
alpha = atan2(y,x)
zeta = z
```

这就是代码中 `bead_phases_rad` 的来源。`alpha` 是钢珠相对于旋转轴的物体相位，
`beta_i` 是扫描角，两者不是同一个量。

得到每颗珠的 `q_k=[x_k,y_k,zeta_k]` 后，图中式 (12) 的无量纲距离可直接写成：

```text
delta_hat_kk'^2 = (x_k-x_k')^2 + (y_k-y_k')^2 + (zeta_k-zeta_k')^2
```

展开 `x=rho*cos(alpha)`、`y=rho*sin(alpha)` 就是论文式 (12)。实际距离满足
`d_kk'=R*delta_hat_kk'`，所以对所有已知正珠间距用过原点最小二乘恢复 SOD：

```text
R = sum(delta_hat_kk' * d_kk') / sum(delta_hat_kk'^2)
```

距离矩阵可以只提供部分珠对；每个已知珠对提供一个尺度约束。没有已知距离、珠子重合或所有无量纲距离接近零时，
绝对 SOD 不可辨识，程序会明确报错。

不再使用对径点、180° 配对、配对插值或交点求解。中心拟合及 SDD/主点求解不依赖角度配对；
当前一般珠布局下，相位恢复和联合精修仍使用逐帧实际角度。
带噪声时 `Q'` 可能有非零交叉项；解析初始化按零面外倾角模型取对角项，
输出归一化交叉项 `Q'01/sqrt(Q'00*Q'11)`，绝对值超过 0.05 时提示检查。
`ellipse_centers_mm` 与 `ellipse_quadratic_forms_per_mm2` 保留原始平板坐标中的拟合中心和完整二次型，
`ellipse_parameters_mm` 是旋正后的系数。这些字段记录解析初始化，精修不会覆盖它们。

相对论文的实现调整：用椭圆中心直线替代对径交点直线；拟合前中心化和缩放；式 (10) 乘掉分母，
避免除以相近的椭圆纵向中心差；相位从所有视角的投影方程线性恢复，
替代离散短轴索引，避免索引量化及相差 180° 的相位歧义。联合精修是额外步骤。

## 钢珠怎么摆

单排、珠间距已知、近似沿旋转轴方向的钢珠杆可以使用，不要求知道杆的精确空间位置。
也不要求珠子一定共线：只要距离约束与输入珠子编号对应即可。

- 至少 3 颗非退化钢珠，建议 5 颗以上、覆盖较大的轴向范围。
- 钢珠杆要偏离旋转轴；放在轴上时轨迹缩成点，标定失效。
- 避免珠子刚好位于源所在中平面 `z=0`；此时轨迹退化为线。
  接近中平面时椭圆过扁，解析初值对定位噪声敏感。
- 不要让所有珠子处于同一高度；选择有高度分布且轨迹能完整落在平板上的布局。
- 本版要求每帧每颗珠都可见、珠 ID 固定，实际角度解包裹后严格单调。
  支持 359° 等接近一周的扫描、奇数帧、非等角采样及正反转。
  当前接口接收不超过一周的数据，取消大于 180° 和对径点数量要求。
  椭圆拟合必须非退化且数值条件合格；较短弧段会使拟合对噪声敏感，
  能在无噪声短弧上恢复参数不代表实际短扫描具有可靠精度。
  不支持单帧缺失珠、轨迹截断或自动关联；整帧缺失可移除对应轨迹及角度后输入。
- 需已知两个方向的像素尺寸；先转成 mm 再估计旋转角，支持非方形像素。

实际图像的前处理应包括暗场/平场校正、分割、亚像素中心定位及跨帧关联。
金属饱和、散射和有限球径可能导致中心偏差；本模拟直接给点中心加高斯噪声，
没有模拟这些成像效应，因此验证的是求解器而非完整成像链。

## 图像检测与跨帧索引

`detect.py` 对 `.npy` 投影堆栈 `[view,row,col]` 逐帧做自适应阈值分割，
用 8 邻域连通域、面积/半径和圆度筛选圆形钢珠；质心使用阈值以上的强度加权矩，
保留亚像素位置。默认把图像行坐标转换为向上的 `v` 坐标，和标定器一致。

跨帧索引不是按候选列表下标硬连，而是：

1. 第一帧按位置排序建立固定 BB ID；
2. 后续帧在仍然 active 的 ID 和当前候选之间做总距离最小的唯一匹配，且受 `max_jump_px` 门限限制；
3. 某个 ID 在任意一帧没有候选、越界候选被边界规则拒绝，或位移超过门限，该 ID 立即永久停用；
4. 最后只保留所有视角都有效的 ID。少于 3 个时直接报错，避免生成不可用椭圆；
5. 因此不会让出界钢珠在后续帧重新出现时错误复活，也不会将新钢珠接到旧 ID 上。

检测器会拒绝接触边界（可用 `--border-margin-px` 扩大安全边界）。如果第一帧本身没有至少 3 个可用目标，
或者所有轨迹中途失效，程序明确失败；这类数据不能交给后面的椭圆拟合。

示例：

```powershell
python tools/cbct_calibration/detect.py projections.npy detected_tracks.npz --expected-count 5 --border-margin-px 3
```

`detected_tracks.npz` 中的 `tracks_px` 可与 `angles_rad`、`pixel_size_mm`、`distances_mm` 一起组成标定输入。
当前检测器不负责暗场/平场校正、投影图像读入格式转换或钢珠在第一帧完全不可见时的全局初始化；这些需要在设备数据接口层处理。

## 运行

从仓库根目录执行，使用已安装 Python 3.10+ 的环境：

```powershell
python -m pip install -r tools/cbct_calibration/requirements.txt
python tools/cbct_calibration/calibrate.py --demo --refine --plot
python -m unittest discover -s tools/cbct_calibration -p "test_*.py" -v
```

默认输出 `out/cbct_calibration/`：

- `calibration.json`：最终参数、解析初值、珠子位置、条件数及误差。
- `observations.npz`：输入坐标和标定条件，可供复现。
- `reprojection.npz`：预测坐标与像素残差，shape 与输入一致。
- `calibration.png`：使用 `--plot` 时生成轨迹叠加及残差直方图。

不加 `--refine --plot` 可以只用 NumPy 运行解析解。
精修的残差权重由 `--center-sigma-px`（默认 0.1 pixel）和
`--distance-sigma-mm`（默认 0.001 mm）控制，应按定位误差和标定件加工/测量精度设置。
间距是带权软约束，默认值不代表实际标定件一定达到 1 微米精度。
当前为普通最小二乘，不含离群点自动剔除。

精修始终使用原始观测及其实际角度；整个流程不生成插值点。
旧的 `--max-pair-gap-deg` 选项及 `antipodal_pairing` 输出已删除，
改为输出各椭圆拟合条件数、中心直线拟合残差和旋正后残余交叉项。
角度必须来自真实采集定义或编码器，不能为凑足一周把 359° 重标为 360°。

## 导入真实数据

NPZ 是 NumPy 数组压缩包，四个键如下：

```python
import numpy as np

# tracks_px[n, k] = [u, v]，每帧相同 k 指向同一颗珠。
# angle_rad 需与轨迹行顺序匹配，支持正/反转和非零起始角。
# 此处例子为 360 张，角度 0,1,...,359 度。
angles_rad = np.arange(360) * 2 * np.pi / 360
# 若实际为 359 张且从 0 扫到 359 度（包含两端），则应使用：
# angles_rad = np.deg2rad(np.linspace(0, 359, 359))
pixel_size_mm = np.array([0.25, 0.25])

# 5 颗共线珠，编号沿杆顺序，中心间距 6 mm。
rod_positions = np.arange(5) * 6.0
distances_mm = abs(rod_positions[:, None] - rod_positions[None, :])
# 非共线珠必须填实际三维中心距离；未知距离用 NaN，矩阵对称、对角为 0。

np.savez_compressed("tracks.npz", tracks_px=tracks_px,
                    angles_rad=angles_rad, pixel_size_mm=pixel_size_mm,
                    distances_mm=distances_mm)
```

```powershell
python tools/cbct_calibration/calibrate.py --input tracks.npz --refine --plot --output out/my_calibration
```

坐标约定：`u` 向右、`v` 向上，以左下角像素**中心**为 `(0,0)`。
如果检测程序用左上角原点、行号向下，应先转换 `v = image_height - 1 - row`。
像素不是物理长度，不能将 mm 单位的珠间距和像素坐标直接混算。
沿 z 轴正转时，源固定为 `(-R,0,0)`；旋正后的平板 u 轴为 `(0,-1,0)`，v 轴为 `(0,0,1)`。
对于行向量，`aligned_mm = measured_mm @ rotation(eta).T`。
图像中心通常为 `((Nu-1)/2, (Nv-1)/2)`，但当前工程 offset 的符号、v 方向、
探测器基向量必须与重建器逐项核对，不能直接把 u0/v0 当作 offset 写回配置。

## 已执行验证与范围

单元测试覆盖无噪声单排珠、最少三珠、独立三维射线/平面投影、非方形像素、
反向扫描、不同珠子的不同相位、只知道一对距离、带噪声精修，以及退化输入拒绝。
另覆盖 359° 无精确对径点的纯解析恢复、359 帧缺末帧、非等角带噪扫描、
跨零度角度解包裹、150° 无噪声短弧解析恢复以及通用旋转椭圆的中心/二次型验证，共 12 项测试。

默认模拟：360 视角、5 珠、6 mm 间距、0.25 mm 像素、0.1 pixel 独立高斯定位噪声，seed=2026。

| 参数 | 真值 | 解析解 | 联合精修 |
| --- | ---: | ---: | ---: |
| SDD / mm | 300 | 299.68651 | 299.97336 |
| SOD / mm | 180 | 179.81261 | 179.98711 |
| u0 / pixel | 255.5 | 255.51477 | 255.50213 |
| v0 / pixel | 127.5 | 127.53881 | 127.54754 |
| eta / degree | 2 | 2.01537 | 1.99976 |
| 每坐标重投影 RMSE / pixel | 噪声 sigma=0.1 | 0.21256 | 0.10021 |

这是单次模拟结果，不是实际设备精度指标或置信区间。小重投影误差也不保证所有系统误差已消除，
应使用不同摆位的独立扫描和重建结果进一步验证。未进行真实扫描标定或重建质量比较。
