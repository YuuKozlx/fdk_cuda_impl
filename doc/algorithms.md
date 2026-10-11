# 算法

本篇定义算法行为与适用边界。参数的公共消费情况见 [公共 API](public-api.md)，数值证据和待验证事项见 [测试与验证](testing-and-validation.md)。支持矩阵依据 [DLL 边界](../src/interface/YkReconstructionApi.cpp) 与 [执行后端](../src/interface/YkSession.cpp)，不能只根据枚举或测试配置推断。

## 公共支持矩阵

“放行”表示公共边界接受该算法/几何组合，后续仍需通过算法 prepare 校验；不代表任意偏移、扫描范围、输入尺寸或图像质量均已验证。“测试中”会返回 AlgorithmUnderTest。矩阵描述当前源码，实际 DLL 必须来自对应构建。

| Pipeline | Flat Circular | Cyl Circular | Flat Helical | Cyl Helical |
| --- | --- | --- | --- | --- |
| ForwardProjection | 放行 | 放行 | 放行 | 放行 |
| FDK | 放行 | 放行，重采样到平板 | 测试中 | 测试中 |
| XFDK | 放行，理想完整圆扫 | 测试中 | 测试中 | 测试中 |
| CFDK | 测试中 | 测试中 | 测试中 | 测试中 |
| SIRT / OSSART / CGLS / PWLS | 放行 | 放行 | 放行 | 放行 |
| SART | 放行 | 边界放行，后端接线不完整，勿依赖 | 放行 | 边界放行，后端接线不完整，勿依赖 |
| TigreGradient | 放行 | 测试中 | 放行，具体选项另受算法限制 | 测试中 |
| WFBP | 测试中 | 测试中 | 测试中 | 放行，要求螺旋解析模块开启 |

Cyl SART 的区别来自当前 Session：initialize 的 SART 分支没有 Cyl 专用准备，而 execute 的 Cyl 分支统一调用 `cyl_iterative_`。这是现有后端问题，不应把边界放行写成可用承诺。

| 非标准轨迹 | 探测器 | 公共算法 | 执行方式 |
| --- | --- | --- | --- |
| PlanarSourceOrbit | Flat | ForwardProjection、FDK 专用管线 | 完整视图序列 |
| NikonTilted | Flat | ForwardProjection、FDK 专用管线 | 完整视图序列 |

非标准轨迹的其他 pipeline 或 Cyl 组合在边界判为非法配置，不是常规迭代算法自动获得支持。

| 公共执行路径 | projection / volume 位置 | 连续分包 | 主要限制 |
| --- | --- | --- | --- |
| ForwardProjection | Host 或 Device | 不支持 | 完整序列，Joseph/Siddon |
| Flat Circular FDK | projection 为 Host；volume 为 Host 或 Device | 支持 | 连续 offset，仅首包清零；Parker 与几何覆盖另检验 |
| Cyl Circular FDK | Host 或 Device | 不支持 | 标准圆柱采集；源偏移、U/N 偏移及 tilt 被拒绝，V 偏移保留 |
| 非标准 FDK | Host 或 Device | 公共 Session 不支持分包提交 | 公共路径要求完整投影驻留 Device；内部独立流式接口见下文；窗口/探测器需纹理对齐 |
| XFDK | 双方均 Device | 不支持 | 理想等角完整 2π、无偏移/倾斜等 |
| 已接通的迭代算法 | 双方均 Device | 不支持 | 完整序列；并非所有内部参数都导出 |
| Cyl Helical WFBP | Host 或 Device | 不支持 | 无 FFS、规范几何及后端 prepare 限制 |

公共 Joseph/Siddon 模型字段供 FP/迭代配对使用，不是在 FDK 中选择另一套解析 BP。普通 FDK BP、FDK-matched BP、Siddon BP 的数学语义不同。

## 流式范围与当前完成状态

本次新增流式以“收到一帧或一小包即可独立处理并累加，不等待其他投影帧”为范围。初始化仍准备完整采集几何和角度权重；这是投影数据流式，不是未知轨迹的几何在线推断。

| 内部路径 | 当前状态 | 帧间投影依赖 |
| --- | --- | --- |
| 普通 Flat FDK / 校准几何近似 FDK | 已有分包入口；本次增加显式 Device 输入用于柱面衔接 | 每帧预加权、行滤波和 BP 独立 |
| SourceOrbit 近似 FDK | 已完成 Host/Device 流式，两种映射均覆盖 | 每帧重采样、滤波和 BP 独立 |
| NikonTilted 近似 FDK | 已完成 Host/Device 流式 | 每帧预加权、滤波和 BP 独立 |
| Cyl Circular FDK | 已完成 Host/Device 流式，旧门面复用同一实现 | 每帧柱面→平板映射、滤波和 BP 独立 |
| xFDK | 保留已有内部流式实现 | 重排需要跨视图邻域和首尾周期缓存，属于已有特殊路径 |
| FP / 普通 BP | 底层具备分批计算/累加条件；本次未新增统一流式入口 | 视图计算可独立；不能把底层条件写成已完成的入口 |
| C-FDK / 螺旋 wFBP / 迭代重建 | 保持现有完整输入路径；不列入本次流式扩展计划 | 重排涉及跨帧数据，或迭代需要多轮访问及全局运算 |

C-FDK、wFBP 和迭代重建不适合本次逐帧即时累加的要求，不通过额外缓存、重排调度或多轮投影读写强行统一为此类流式。当前自然流式重建范围已完成；内部状态不等于公共 Session 分包能力，公共边界仍按上表和 API 文档执行。

## 离散投影模型与 FP/BP

以 `b=A x+e` 表示离散成像。`x` 为体素衰减值，`b` 为测量线积分，`A_ij` 为第 i 条射线对第 j 个体素的权重；在本库长度单位下为 mm。BP 的实现可为精确离散转置或近似反投，必须说明是哪一种。

| 算子 | 核心计算 | 伴随性质 |
| --- | --- | --- |
| Siddon FP | 射线与体积盒求交，沿体素边界递推，累加体素值×真实段长 | 与同射线/同段长 RayDriven BP 配对 |
| Siddon RayDriven BP | 相同离散像素中心射线，将投影值×段长原子散射到体素 | 对应离散 FP 的转置，仍有浮点累加顺序误差 |
| Joseph FP | 按射线主轴采样，以插值读取体积并按物理长度积分 | 硬件 Linear 权重量化，不能无条件称严格伴随 |
| Joseph voxel-driven BP | 体素投影到探测器，插值和相应长度权重 | 与 FP 为近似配对，需具体数值测试 |
| FDK BP | 解析距离/角度权重的反投 | 不是上述 FP 的离散转置 |
| FDK-matched BP | 使用体素、像素与距离相关的匹配权重 | 不能等同于精确 Aᵀ，不能保证比普通 FDK 更接近离散转置 |

内部 Cyl Siddon V2 用连续 channel/row、Linear 纹理、连续射线盒交长；V3 将 channel/row 四舍五入，用 Point 纹理和离散像素中心射线盒交长。二者都是体素驱动近似，不是 RayDriven FP 的严格转置，也不是同算法的纯性能版本。

V3 在全部视图满足源在圆柱轴上、圆柱轴与体积 Z 对齐时选标准几何专用 kernel，否则回退通用 V3。旧 Joseph 风格 voxel-driven 不能因为使用体素线程就改称 Siddon。

DLT 投影矩阵算子位于 `src/DLTFpBp/`，有独立伴随与代数测试；当前公共 `SSystemSpec` 没有任意 DLT 矩阵输入。CVP 属于实验内部实现，也不是公共第三种 projection_model。

## 普通 Flat FDK

实际管线位于 [FdkPipeline](../src/Reconstruction/Analytic/Circular/Flat/FDK/YkFdkPipeline.hpp)：最终逐视图几何 → 派生/诊断 → Parker（按配置）→ 预加权 → U 向 FFT 滤波 → 解析 BP → 分包累加。

FDK 是圆轨迹锥束近似重建，完整圆扫不保证大锥角轴向区域精确恢复。校准几何偏离理想圆轨迹时内部管线仍可近似执行并报告诊断；公共规则 Session 的几何输入范围见 [几何](geometry-and-data.md)。

### 几何与尺度分层

1. 标称 SID/SDD/offset/tilt 是 builder 的输入。
2. `SConeProjGeomVec` 是实际射线几何。
3. prepare 从实际几何派生源轴距离、源到径向探测器距离、源到平面距离、主点、U/V 轴与角步长。

各 stage 只消费已派生量，不能重复把标称 offset 加回 kernel。对于零姿态标准圆扫，令 R=SID、D=SDD、世界点 X 的径向深度为 L，探测器位置由源到 X 的直线与探测器平面求交。物理探测器 u/v 的放大尺度使用 D；等中心虚拟探测器使用 R，两套坐标不能混写。

解析表达可拆为预加权、ramp 卷积和角度加权 BP。若用物理探测器坐标，标准余弦项为 `D/sqrt(D²+u²+v²)`，离散卷积吸收步长 du，BP 含距离及角积分权重。代码可以把常数分配到不同 stage，不能在多个 stage 重复补乘 `1/du`、`1/N` 或 `DeltaTheta/2`。最终定量一致性通过均匀线积分、滤波和深度分母测试验证；具体系数以 [派生几何](../src/Reconstruction/Analytic/Circular/Flat/FDK/YkFDKVecGeoDerived.hpp) 和 stage/kernel 实现为准。

`FdkGeometryDiagnostics` 报告 SOD/SDD 范围、源轴向漂移、主点、法向偏离、U/V 非正交及采样变化。当前 FFT 权重由 chunk 首帧实际 du 构建；逐视图 du 变化仍是近似项。接受几何不代表变动采样已严格校正。

### Parker

公共 `fdk.parker.mode` 为 Auto/Disabled/Enabled，当前只对 Flat Circular 生效。Auto 按 `2*pi*total_views/views_per_turn < 2*pi-容差` 判定短扫；它不独立证明数据满足短扫所需覆盖范围。

Parker 扇角由最终逐视图几何及径向射线计算。扫描范围、旋转方向、扇角符号、冗余权重与角积分常数必须一致。不要把论文中某一归一化形式直接复制到所有 stage；短扫与全扫都需检查材料绝对值。

## 滤波与 Ram-Lak 离散化

主要实现：[滤波类型](../src/global/YkFilterTypes.hpp)、[频域构建器](../src/Filter/YkCreateFilterKernelFFT.hpp)、[kernel helpers](../src/Filter/YkCreateFilterKernelHelpers.cuh)。

连续 ramp 的响应为 `H(f)=|f|`，Nyquist 为 `1/(2*du)`。连续核在采样点的值乘 du 后，正确离散卷积核为：

```text
h[0] = 1/(4*du)
h[n] = -1/(pi²*n²*du)  （n 为奇数）
h[n] = 0               （其余非零偶数）
```

代码的 du=1 数字核 `h0[0]=1/4`、`h0[n]=-1/(pi²*n²)` 没有物理单位，**正确离散核为 h0/du**。不能把 h0 直接等同于任意 du 下的连续核样值；连续核还有另一项 `1/du`。例如 du=0.25 mm 时，连续中心值为 4 mm⁻²，正确离散中心值为 1 mm⁻¹，数字核中心值为 0.25。

离散 FFT 卷积是纯求和；cuFFT C2R 不归一化，需单独吸收 `1/N`。有限核的 DC 处理、加窗、gain 与物理 `1/du` 均需由核构建路径核对，不能再按旧文档随意在末尾补尺度。

内部有三种 weights source：AnalyticFreq、DiscreteRLFFT、SpatialRampFFT。**公共 Session 的 makeFilterDesc 默认使用 AnalyticFreq**；内部 `SFilterKernelDesc::RamLak()` 默认使用 DiscreteRLFFT。同为 RamLak 名称不保证这两种离散频谱完全相同。有限离散 ramp 的 DC 强制抑制是固定行为；滤波回归必须区分构建源。

公共 filter 包含 None、RamLak、SheppLogan、Cosine、Hann、Hamming、Blackman、Butterworth、Kaiser、Tukey；内部还支持 Custom 和空域 ramp。公共 cutoff 采用数字频率，0.5 表示完整 Nyquist 带宽，不能写成“0.5 表示 Nyquist 的一半”。None 也不能作为有定量重建保证的默认配置。

## Cyl Circular FDK

当前公共路径：[Cyl Analytic Reconstruction](../src/CylFpBp/Analytic/YkCylAnalyticReconstruction.hpp)。

```text
物理圆柱几何 → 刚体规范化与主点派生
物理圆柱投影 → 以射线求交映射到虚拟平板
虚拟平板投影 + flat_geometry → 普通 Flat FDK
```

目标平板法向由中心射线定义，映射按目标射线与物理圆柱交点反算通道/行，双线性采样，超出可见范围填零。名义目标 U 间距采用物理弧长采样间距。V 平移通过目标主点保留。R 可不同于 SDD，但 SDD 必须来自采集配置，不能用曲率半径或源到圆柱轴距离替代。

公共边界拒绝源偏移、探测器 U/N offset 与三个 tilt；支持的几何还需通过 canonicalizer 的逐视图一致性检查。公共 Session 仍只接受完整序列；内部 Reconstruction 已支持下面的逐包输入。公共 Parker 选择只针对 Flat Circular，不能假设 Cyl 短扫自动加权。

### 柱面 FDK 内部流式契约

[Analytic::Reconstruction](../src/CylFpBp/Analytic/YkCylAnalyticReconstruction.hpp) 使用 prepare → beginStreaming → enqueueBatch → completeStreaming。每包的柱面映射独立，随后进入已准备完整平板几何的 Flat FDK，直接累加到 Device 体积，不需要相邻帧或首尾投影缓存。

`ReconstructionConfig.max_chunk_views` 默认 32，限制原始 Host 上传暂存、映射后投影及 FDK 中间工作区；实际取该值、32 和总视图数的最小值。完整 geometry 仍在 prepare 时准备，投影缓存大小不再随总视图数增长。

```cpp
// reconstruction 已 prepare，d_volume 为调用方持有的 Device 输出。
if (!reconstruction.beginStreaming(d_volume, true)) return false;
for (int offset = 0; offset < total_views; offset += batch_views) {
    const int count = std::min(batch_views, total_views-offset);
    YK::CylFpBp::Analytic::CylFdkProjectionBatch batch{};
    batch.data = packet.data(); // 当前包的采集柱面投影，不是映射后的平板投影
    batch.first_view = offset;
    batch.view_count = count;
    batch.location = YK::EMemoryLocation::Host;
    batch.element_count = size_t(count)*channels*rows;
    // 调用前把对应视图读取/采集到 packet 中。
    if (!reconstruction.enqueueBatch(batch)) {
        reconstruction.cancelStreaming();
        return false;
    }
}
return reconstruction.completeStreaming();
```

输入可为 Host（普通或 pinned）或 Device，按 `[view][row][channel]` 连续排列。first_view 必须从 0 连续推进，容量以 float 元素数填写；包可大于内部 chunk。enqueueBatch 同步消费，返回后可复用输入。输出始终为 Device 体积，清零只在 beginStreaming 时执行；clear_output=false 保留已有体积并累加新一轮。

complete 仅在收齐全部视图时成功；未收齐可继续 enqueue。空包、容量不足、跳帧、重复、越界和非法位置在计算前拒绝；计算失败后必须 cancel 或重新 prepare，避免重试造成部分投影重复累加。cancel 保留准备资源和已有体积，不自动清零。每次开始新流都会 reset Flat FDK 的在线位置。

完整 Device 输入 reconstruct 保留，复用同一执行路径；旧 [CylFdkPipeline 门面](../src/CylFpBp/Analytic/YkCylFdkPipeline.hpp) 委托给 Reconstruction，并转发同样的流式方法，其 prepare 最后参数可指定 max_chunk_views。新内部调用优先使用显式 SDD 的 Reconstruction，避免依赖旧门面的距离推断。

映射后的 Device 数据以 FdkProjectionBatch.location=Device 提交给内部 Flat FDK，采用 DeviceToDevice 复制；普通 Flat 内部批次默认 Host，兼容原有调用。公共 Flat Session 的投影仍要求 Host，公共 Cyl Session 仍要求完整序列，这些内部扩展未增加新的 SDK 分包能力。

内部仍有 Cyl FDK/FDK-matched BP kernel 和专项测试；它们不是当前公共 FDK pipeline 的同义描述。旧“映射到 R=SDD 虚拟圆柱再进行柱面 FDK”的开发说明不能代表当前主路径。

## xFDK

实现：[XfdkPipeline](../src/Reconstruction/Analytic/Circular/Flat/XFDK/YkXfdkPipeline.hpp) 与 `XFDK/kernels/`。通过平行束变量重排和体素相关部分扫描权重扩展轴向有效范围。

理想圆扫下，R 为源半径，beta 为论文扇角，gamma 为锥角：

```text
vartheta = alpha + beta
xi = -R*sin(beta)
beta = asin(-xi/R), alpha = vartheta-beta
项目物理平板：u = -SDD*tan(beta), v = SDD*tan(gamma)/cos(beta)
```

重排同时乘 `cos(beta)*cos(gamma)`，沿 xi 做 ramp 滤波；BP 对每个体素计算可见角半范围、部分扫描平滑权重和径向过渡权重，乘 `0.5*DeltaTheta*wC`。它不能简化成普通 FDK 分块。

令 c=cot(gamma_max)、关注圆盘半径 Rm<R，理想几何单侧范围为：

```text
z_FDK = (R-Rm)/c
z_xFDK = (R²-Rm²)/(c*R)
范围比 = 1+Rm/R
```

这是理想覆盖条件下的理论范围，不是对任意模体的质量保证。当前实现只接受理想、等角、不重复终点的完整 2π，无源偏移、探测器 offset/tilt/skew、FFS。超出可见区域的权重为零。

内部流式入口为 prepareBatched → beginStreaming → 连续 enqueueBatch → completeStreaming，源 batch 使用 pinned Host 内存并按顺序提供，内部负责跨视图 halo 和周期首部缓存。公共 Session 当前只提供完整 Device 投影重建，不能通过 view_offset 调用此内部流式路径。

## C-FDK

实现：[CurveFilteredFdkPipeline](../src/Reconstruction/Analytic/Circular/Flat/CFDK/YkCurveFilteredFdkPipeline.hpp)。这里 C 指 curve-filtered，**不是 Cyl 探测器**。当前仅供内部验证，公共 CFDK pipeline 未放行。

虚拟等中心平面坐标 (a,b) 重排为曲线坐标 (t,c)，R=SID、D=SDD：

```text
q = sqrt(R²-t²)
a = t*R/q
依据 |c| 与 c0、s0、bm 的关系分段求 b 和 Jacobian
原始平板采样位置由 (D/R)*(a,b) 换算
沿 t 做 ramp 滤波
BP 用三段逆映射得到 c，纹理采样并乘 0.5*DeltaTheta
```

`c0/s0/bm` 是重排分段边界，分别对应中心、过渡和外侧区域；不是 xFDK 可见角边界。闭区间与浮点容差、正负 z 对称、前后映射往返和分界连续性必须一起检验。`R²-t²<=0` 或超出 bm 的样本填零；空间边界为 zero border，视图维度单独周期处理。

限制为理想等角完整 2π，无源偏移、offset/tilt/skew、非水平轨迹和短扫；输入是完整 Device 投影。它改善锥角伪影但不扩展原始探测器的轴向覆盖，也不是 z 向反卷积。解析椭球、3×3 子射线、论文原始模体和插值损失的严格复现仍待完成。

## 螺旋 wFBP

实现：[WFBP Pipeline](../src/Reconstruction/Analytic/Helical/WFBP/YkWfbpPipeline.hpp)。FreeCT 风格解析管线执行重排、预加权、滤波、冗余处理和 BP，与螺旋迭代路径独立。

公共接口仅开放 Cyl Helical、无 FFS，需要 `YKCBCT_BUILD_HELICAL=ON`；Flat 输入存在内部测试入口但仍未公共放行。源偏移和探测器 N offset 在边界被拒绝，其他几何条件继续由 wFBP prepare 判断，不能因为某项边界未拒绝就承诺任意 tilt。

`redundancy_flat` 与滤波参数使用 wFBP 自己的定义，不能照搬普通 FDK 的 Parker 或 cutoff。保留 FreeCT 来源及 [授权声明](../LICENSES/FreeCT_wFBP_NOTICE.md)。

## 通用代数更新与 ART 推导

单射线约束 `a_i*x=b_i` 中，使当前 x 满足该约束的最小二范数修正沿 `a_iᵀ`：

```text
delta = (b_i-a_i*x)/||a_i||² * a_iᵀ
ART: x <- x + lambda*delta
```

噪声数据通常不满足所有超平面交集；欠松弛、提前停止、非负和正则化共同决定实际解。ART 是逐射线更新，当前公共 SART 是逐视图更新，二者不能当作同一 pipeline。

子集代数形式：`x <- P[x + lambda*C_s^-1*A_sᵀ*R_s^-1*(b_s-A_s*x)]`，R/C 为行列归一化，零支撑由 epsilon/有效掩码处理，P 为约束。SIRT 每轮一个全数据子集，SART 每轮逐视图，OS-SART 每轮使用配置子集。

内部 Flat 统一门面把 method、weight_model、subset_order、regularization 分开：

| 维度 | 内部选项 |
| --- | --- |
| 更新 | Sirt、Sart、Ossart |
| 权重 | DetailedSubset、TigreApprox |
| 顺序 | Sequential、GoldenRatio |
| 正则 | None、SmoothedTv |

DetailedSubset 为每子集计算 `R=A_s*1`、`C=A_sᵀ*1`。TigreApprox 用 2×2×2 粗体积估计行权重、XY 范围 1.1 缩放、列权重范围 0.9 语义和连续视角分块；实际子集数以 backend 查询值为准。

普通门面按 scan 构造圆轨迹，Ex 门面接收完整逐视图 geometry；Ex 只表示几何入口，不代表另一种权重或算法。Cyl、Helical 使用对应类型和算子，不能降格成平板圆轨迹。

### 平滑 TV 与收敛

内部 TV 定义：`TV_epsilon(x)=sum sqrt(Dx²+Dy²+Dz²+epsilon²)`，梯度为 `-div(grad(x)/sqrt(|grad(x)|²+epsilon²))`，更新为 `x <- x-strength*gradient`。差分分别除以物理体素间距；Slice2D 忽略 Z。

调度为完整一轮数据子集 → TV inner_iterations 次 → 最终约束。子集后可以保留数据后端约束，TV 后再约束以防产生越界值。梯度读取旧数组写独立工作区，禁止同一 kernel 邻域读取并原地覆盖。

`iterateSubsetUpdates()` 只更新数据一致性，不自动执行 TV；正则化使用 reconstruct 完整外循环。公共 SDK 未导出该通用 TV Config。

内部收敛支持相对残差 `||b-Ax||/||b||`、相对体积更新和残差停滞阈值，以及 minimum_iterations/check_interval/patience。阈值为零禁用；检查残差可能额外执行完整 FP。统计区分外循环和子集更新。参数能否由公共 Session 消费见 [公共 API](public-api.md#系统参数)，不能把内部能力全部写成公共能力。

## CGLS 与 PWLS

CGLS 求解最小二乘问题。以残差 r=b-Ax、s=Aᵀr、方向 p=s 开始，反复计算 q=Ap、alpha=(sᵀs)/(qᵀq)、x+=alpha*p、r-=alpha*q，再以新梯度能量比更新方向。若 BP 不是 A 的精确转置，标准共轭性质需以实际算子组合评估。

内部 RobustRestart 计算真实残差并在发散时回退/重启；AstraClassic 递推残差，不执行额外检查与回退，且不支持统一收敛阈值。公共 Flat Session 当前固定使用旧 CGLS 适配器，未消费公开 cgls 策略/停止组；Cyl 使用自己的通用迭代实现，不应根据同名字段推断两者一致。

PWLS 目标为 `0.5*(Ax-b)ᵀW(Ax-b)+beta*R(x)`，支持 None、Quadratic、Huber 与上下限。当前实现使用 surrogate 更新；正则权重、Huber 阈值和 OS-SART-TV 的 strength 没有可直接互换的尺度。

内部 PWLS 可接收权重数组；公共 Session 当前未提供该数组，使用单位权重。Cyl 的 JosephMatched 与 Siddon 数据模型由公共 projection_model 映射；轨迹由完整 geometry 决定。初始化、约束和清零语义以具体重建器为准。

## TIGRE 风格梯度族

实现：[TigreGradientReconstructorEx](../src/Reconstruction/Iterative/Flat/YkTigreGradientReconstructorEx.hpp)。支持 Sart、OsSart、Sirt、AsdPocs、OsAsdPocs、BAsdPocsBeta、Pcsd、OsPcsd、AwPcsd、OsAwPcsd、AwAsdPocs、OsAwAsdPocs。

统一状态为数据更新、约束、残差、TV 更新、步长/停止状态：

- 普通 SART/OS-SART/SIRT 共享 TigreApprox 更新，每个完整外循环衰减 lambda。Nesterov 用候选解与上一候选解组合，内部动量递推；残差增长时停止，不自动承诺回退。
- ASD-POCS 从数据步的范数 `dp=||xd-x0||` 初始化 TV 步长 `dtvg=alpha*dp`。TV 更新量 `dg` 相对 dp 过大且残差尚未达标时，按 alpha_reduction 缩小步长。
- PCSD 使用数据一致性/残差比控制 TV 与投影更新；Aw 变体用自适应加权 TV，权重与梯度必须按该实现的状态更新。
- B-ASD-POCS-beta 保留 Bregman 数据状态，不能按普通 ASD-POCS 每轮清空。

初始化可选 Zero/FDK，非负、Nesterov、FDK 初值和 max_l2_error 的支持条件由内部 prepare 校验；公共 Flat Helical 放行不代表圆轨迹 FDK 初值适用于螺旋。公共 block_size 的实际换算见 API 文档。

内部梯度族提供统计与停止原因，但公共 Session 只有 execute 成功/失败和错误消息。需要更详细统计时应明确扩展公共契约，而不是把内部类型暴露给调用方。

## 非标准解析管线

SourceOrbit 的数据流为全幅固定探测器 → 安全固定窗口/射线保持重采样 → 预加权 → U 向滤波 → 当前源位置 BP → 累加。权重参考射线指向配置的体积中心，不能替换成整数窗口中心。

SourceOrbit 和 NikonTilted 的内部管线支持真正的逐包输入：prepare → beginStreaming → 连续 enqueueBatch → completeStreaming。原 process 完整 Device 输入接口保留，复用同一执行路径。公共 Session 仍排除非标准轨迹的 streaming_fdk，并要求完整序列；通过公共 Host 输入时仍整份上传。直接使用以下内部接口可消除完整原始投影的显存需求。

### 非标准 FDK 内部流式契约

共享批次类型位于 [YkNonStandardFdkStreaming.hpp](../src/NonStandardFpBp/Geometry/YkNonStandardFdkStreaming.hpp)。SourceOrbit prepare 的最后参数和 NikonTilted prepare 的最后参数均可指定 max_chunk_views，默认为 kMaxChunkAng（32）；实际工作区视图数为该值、32 和总视图数的最小值，必须大于零。

| 调用/字段 | 语义 |
| --- | --- |
| beginStreaming(d_volume, clear_output) | 开始一次重建；输出为完整 Device 体积，默认仅在此时清零 |
| batch.data | 本包第一帧指针，不是整次采集起点；投影按采集探测器 [view][v][u] 排列 |
| batch.first_view / view_count | 全局连续视图起点与正视图数；可以提交大于内部 chunk 的包 |
| batch.location | EProjectionBatchLocation::Host 或 Device；Host 可为普通或 pinned 内存 |
| batch.element_count | 从 data 起可访问的 float 元素容量，至少 view_count×采集 rows×channels |
| enqueueBatch(batch) | 同步消费该包；成功返回后可复用输入，体积已累加该包；无需保留历史投影 |
| completeStreaming() | 仅全部视图提交后成功；结束当前重建，下一次可重新 begin |
| cancelStreaming() | 等待内部工作并清除流式状态，保留 prepare 资源；不清零已有体积 |
| streamingActive / receivedStreamingViews | 查询活动状态和已成功消费的视图数 |

首次 first_view=0，随后必须等于已接收视图数；重复、跳帧、越界、空包、非法位置和容量不足会在发射前拒绝，不推进状态。未收齐时 complete 返回 false，仍可继续提交。计算失败后需 cancel 或重新 prepare，不能直接重试可能已部分累加的包。已有活动重建时不能 begin 或调用完整 process。

```cpp
// pipeline 已 prepare；d_volume 为调用方持有的 Device 输出。
if (!pipeline.beginStreaming(d_volume, true)) return false;
for (int offset = 0; offset < total_views; offset += batch_views) {
    const int count = std::min(batch_views, total_views - offset);
    NonStandardFdkProjectionBatch batch{};
    batch.data = host_projection.data() + size_t(offset) * acquired_view_elements;
    batch.first_view = offset;
    batch.view_count = count;
    batch.location = EProjectionBatchLocation::Host;
    batch.element_count = size_t(count) * acquired_view_elements;
    if (!pipeline.enqueueBatch(batch)) {
        pipeline.cancelStreaming();
        return false;
    }
}
return pipeline.completeStreaming();
```

实际应用可用同一 Host 包缓冲逐包采集/读取，示例中的完整 host_projection 仅用于说明指针语义。enqueue 是同步消费接口，本版不提供传输与计算双缓冲重叠。调用方须保证 Device 输入的外部生产者已完成，输出体积保持有效到 complete/cancel，外部 stream 必须晚于 pipeline 释放。

SourceOrbit 仅在 Host 输入时分配最多一个原始探测器 chunk 的设备暂存，重采样指针使用包内索引、map/geometry 使用全局视图索引；Device 包直接读取调用方数据。NikonTilted 直接向现有 chunk_in 暂存复制当前 chunk。几何、map 和派生量仍在 prepare 时准备完整序列，因此几何显存仍随视图数增长，但原始投影显存不再需要随总视图数增长。

内部拆包保留 prepare 定义的计算块边界和原有共享滤波尺度，任意输入包边界不会重新选择 du；BP 仍按完整几何的角权重累加。包大小改变可能带来浮点累加顺序差异，需用定量误差验证。prepare 的工作区块大小属于配置，不能把改变该配置与改变输入包大小混为同一比较。

NikonTilted 用专门 geometry 和 pipeline 处理倾斜初始源/探测器及世界 Z 轴转动。这两类路径复用部分 FDK stage，但不能绕过普通圆轨迹验证后直接标成标准 FDK。入口字段见 [非标准几何](geometry-and-data.md#非标准平板轨迹)。

## 实现依据与理论资料

本篇保留解释实现所需的公式、离散尺度和边界；旧长篇论文翻译、开发过程及无版本性能表通过 Git 历史查阅。xFDK 理论对应 Grimmer 等《Cone-beam CT image reconstruction with extended z range》，C-FDK 对应 Li 等曲线滤波 FDK 工作；论文坐标、项目坐标和代码归一化需逐项转换，不能仅因公式编号一致就宣布完整复现。

修改算法时同时更新支持矩阵与验证证据。当前未完成的论文级复现、Cyl SART 接线、公共参数映射及量纲验证见 [待验证事项](testing-and-validation.md#待验证与实现限制)。
