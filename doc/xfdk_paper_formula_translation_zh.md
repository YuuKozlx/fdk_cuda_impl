# 《Cone-beam CT image reconstruction with extended z range》公式推导中文翻译

> 原文：Rainer Grimmer, Markus Oelhafen, Ulrik Elstrøm, Marc Kachelrieß,
> *Medical Physics*, 36(7), 3363-3370, 2009。DOI: 10.1118/1.3148560。
>
> 本文档忠实翻译原文第 II 节和第 III 节的几何与 xFDK 推导。为了便于核对，
> 保留原文公式编号和符号。标记为“工程说明”的内容是结合本项目坐标系得到的
> 实现解释，不属于论文原句。
>
> 原论文只显式标出了式 (1)-(5)。本文从 (6) 起为后续未编号公式补充了本地
> 编号，便于文档和代码互相引用；这些编号不是论文原编号。

## 1. 论文目的

圆轨迹锥束 CT 通常只重建每个体素都获得完整 $360^\circ$ 数据的轴向范围。
有限高度平板之外还存在一段区域：其中的体素没有完整 $360^\circ$ 覆盖，但仍
至少被照射 $180^\circ$。xFDK 的目标是使用这部分数据扩展可重建的 $z$ 范围。

该方法的关键不是修改普通 FDK 的深度权重，而是：

1. 把扇束投影重排为平行束数据；
2. 只执行一次滤波；
3. 在反投影时按体素计算其实际可见角范围；
4. 在全扫描权重与体素相关的部分扫描权重之间平滑过渡。

## 2. 几何模型（原文第 II 节）

论文考虑 Feldkamp 型锥束投影 $p(\alpha,\beta,\gamma)$，它是物体
$f(\mathbf r)$ 上的射线积分。焦点位置 $\mathbf s(\alpha)$ 与射线方向
$\boldsymbol\Theta(\alpha,\beta,\gamma)$ 定义为

$$
\mathbf s(\alpha)=
\begin{pmatrix}
R_F\sin\alpha\\
-R_F\cos\alpha\\
0
\end{pmatrix},
$$

$$
\boldsymbol\Theta(\alpha,\beta,\gamma)=
\begin{pmatrix}
-\sin(\alpha+\beta)\cos\gamma\\
\cos(\alpha+\beta)\cos\gamma\\
\sin\gamma
\end{pmatrix}.
$$

其中：

| 符号 | 原文含义 |
| --- | --- |
| $R_F$ | 焦点到等中心的距离，即 SID |
| $R_M$ | 成像范围（field of measurement）的半径 |
| $\alpha$ | 投影角 |
| $\beta$ | 扇角 |
| $\gamma$ | 锥角 |

原始数据来自等距采样的线性平板，但论文使用扇角 $\beta$ 参数化水平方向
射线。这样做会显著简化推导，因为 $\alpha$ 与 $\beta$ 都同平行束角呈线性关系。

平行束射线满足

$$
x\cos\vartheta+y\sin\vartheta=\xi,
$$

其中平行束角为

$$
\boxed{\vartheta=\alpha+\beta} \tag{1}
$$

而射线到等中心的有符号距离为

$$
\boxed{\xi=-R_F\sin\beta}. \tag{2}
$$

$\vartheta$ 很重要，因为完整二维重建要求同一体素获得跨度为 $180^\circ$ 的
$\vartheta$ 数据。

## 3. 可重建范围（原文第 III.A 节）

由于探测器纵向范围有限，离中平面较远的层并不完全受照射。论文利用上下
对称性先讨论 $z>0$。对于与坐标轴正确对齐的平板探测器，受照射区域是半空间

$$
\boxed{R_F-(x\sin\alpha-y\cos\alpha)\geq cz}. \tag{3}
$$

分界线垂直于中心射线，它到焦点的距离是 $cz$，并且

$$
\boxed{c=\cot\gamma_{\max}},
$$

其中 $\gamma_{\max}$ 是中心矢状面中探测器纵向边缘对应的最大锥角。

### 3.1 普通 FDK 范围

要让半径 $R_M$ 内的每个体素都被完整 $360^\circ$ 照射，需要

$$
cz\leq R_F-R_M.
$$

因此普通 FDK 的单侧纵向范围为

$$
z_{\mathrm{FDK,max}}=\frac{R_F-R_M}{c}.
$$

### 3.2 xFDK 扩展范围

要让半径 $R_M$ 内的每个体素至少获得 $180^\circ$ 数据，需要

$$
cz\leq\frac{R_F^2-R_M^2}{R_F}.
$$

所以 xFDK 的单侧纵向范围为

$$
z_{\mathrm{xFDK,max}}=
\frac{R_F^2-R_M^2}{cR_F}.
$$

两者的范围比为

$$
\boxed{
\frac{R_F^2-R_M^2}{R_F(R_F-R_M)}
=1+\frac{R_M}{R_F}}.
$$

例如 $R_F=2R_M$ 时，理论纵向范围增加 $50\%$。论文使用的 Varian OBI
参数为 $R_F=1000\ \mathrm{mm}$、$R_M=130\ \mathrm{mm}$，范围增加约 $13\%$。

## 4. 扩展 FDK 推导（原文第 III.B 节）

### 4.1 为什么不能直接逐体素做 Parker 预加权

传统扇束 FBP 在少于 $360^\circ$ 的数据上，会在卷积之前施加 Parker 权重。
但是扩展区中每个体素的可见角范围都不同。如果仍在卷积前加权，原则上每个
体素都需要单独的加权、卷积和反投影，计算量不可接受。已有方法通过把相似
权重的邻近区域分段合并来降低成本，但分段边界会产生不连续。

xFDK 的处理方式是：先重排并完成一次公共卷积，再在反投影时施加连续的体素
相关权重。因此它只需要一次卷积和一次反投影。

### 4.2 扇束到平行束重排

按式 (1)、式 (2) 将

$$
p(\alpha,\beta,\gamma)\cos\epsilon
$$

重排为

$$
p(\vartheta,\xi,\gamma),
$$

其中 $\cos\epsilon$ 是射线长度修正因子。对于本文的理想平板几何，射线与
中心射线的夹角满足

$$
\cos\epsilon=\cos\beta\cos\gamma.
$$

由式 (1)、式 (2)，给定输出 $(\vartheta,\xi,\gamma)$ 后，反解为

$$
\beta=\arcsin\left(-\frac{\xi}{R_F}\right),\qquad
\alpha=\vartheta-\beta.
$$

工程说明：本项目真实平板使用距离坐标 $(u,v)$，且 `detU` 与论文正 $\beta$
方向相反，因此采样坐标为

$$
u=-D_{SD}\tan\beta,
\qquad
v=\frac{D_{SD}\tan\gamma}{\cos\beta}.
$$

### 4.3 单个体素的照射角范围

把体素横断面坐标写成极坐标

$$
x=-r\sin\varphi,\qquad y=r\cos\varphi.
$$

代入式 (3)，该体素被所有满足下式的投影角 $\alpha$ 照射：

$$
\boxed{
|\alpha-\varphi|\leq
\frac{\pi}{2}+\arcsin\frac{R_F-cz}{r}}. \tag{4}
$$

接着求该体素相对于焦点的扇角。令

$$
\mathbf s(\alpha)+\lambda\boldsymbol\Theta(\alpha,\beta,\gamma)
=
\begin{pmatrix}
-r\sin\varphi\\
r\cos\varphi\\
z
\end{pmatrix},
$$

对 $\beta$ 求解得到

$$
\boxed{
\beta=-\arctan
\frac{r\sin(\alpha-\varphi)}
{R_F+r\cos(\alpha-\varphi)}}.
$$

在式 (4) 给出的 $|\alpha-\varphi|$ 两个极值处，扇角为

$$
\boxed{
\beta=\mp\arctan
\frac{\sqrt{r^2-(R_F-cz)^2}}{cz}}. \tag{5}
$$

利用 $\vartheta=\alpha+\beta$，可得到该体素在平行束域中的可用角区间

$$
\boxed{[\vartheta_1-\Delta\vartheta,\ 
\vartheta_2+\Delta\vartheta]},
$$

其中

$$
\vartheta_1=\varphi-\frac{\pi}{2},\qquad
\vartheta_2=\varphi+\frac{\pi}{2},
$$

并且

$$
\boxed{
\Delta\vartheta=
\arcsin\frac{R_F-cz}{r}
-\arctan\frac{\sqrt{r^2-(R_F-cz)^2}}{cz}}. \tag{6a}
$$

论文还给出等价形式

$$
\boxed{
\Delta\vartheta=
\arcsin
\frac{R_F(R_F-cz)-r^2}
{r\sqrt{R_F(2cz-R_F)+r^2}}}. \tag{6b}
$$

两个重要边界是：

$$
cz=\frac{R_F^2-r^2}{R_F}
\quad\Longrightarrow\quad
\Delta\vartheta=0,
$$

以及

$$
r\leq R_F-cz
\quad\Longrightarrow\quad
\Delta\vartheta=\frac{\pi}{2}.
$$

前者表示该体素刚好只剩 $180^\circ$ 数据；后者表示该体素仍有完整
$360^\circ$ 数据。

### 4.4 体素相关的部分扫描权重

定义平滑函数

$$
s(t)=\sin\left(\frac{\pi t}{2}\right).
$$

原文的部分扫描权重为

$$
\boxed{
w_{PS}(\mathbf r,\vartheta)=
\begin{cases}
0,
& \vartheta<\vartheta_1-\Delta\vartheta,\\[4pt]
1+s\!\left(\dfrac{\vartheta-\vartheta_1}{\Delta\vartheta}\right),
& \vartheta<\vartheta_1+\Delta\vartheta,\\[8pt]
2,
& \vartheta<\vartheta_2-\Delta\vartheta,\\[4pt]
1-s\!\left(\dfrac{\vartheta-\vartheta_2}{\Delta\vartheta}\right),
& \vartheta<\vartheta_2+\Delta\vartheta,\\[8pt]
0,
& \text{其他情况}.
\end{cases}} \tag{7}
$$

$\vartheta_1$、$\vartheta_2$ 和 $\Delta\vartheta$ 都依赖体素位置，而
$\vartheta$ 是当前反投影视角。其他奇对称平滑函数也可以使用，只要满足
$s(-t)=-s(t)$ 并保持 $180^\circ$ 归一化。

这里有一个容易写错的量级细节：$w_{PS}$ 本身取值范围是 $[0,2]$，对一圈
的积分为 $2\pi$；最终反投影公式外部还有 $1/2$，所以有效积分才是 $\pi$。
不能同时在 $w_{PS}$ 内和反投影外各乘一次 $1/2$。

### 4.5 全扫描权重与平滑过渡

在仍有完整 $360^\circ$ 数据的区域，论文希望使用剂量效率更好的全扫描权重

$$
\boxed{w_{FS}(\mathbf r,\vartheta)=1}. \tag{8}
$$

为了避免从全扫描权重突然切换到部分扫描权重，引入径向过渡权重

$$
\boxed{
w_T(\mathbf r,\vartheta)=\frac{1}{2}
\begin{cases}
0, & r<r_0-\Delta r,\\[4pt]
1+s\!\left(\dfrac{r-r_0}{\Delta r}\right),
& r<r_0+\Delta r,\\[8pt]
2, & \text{其他情况},
\end{cases}} \tag{9}
$$

其中

$$
\boxed{
\Delta r=\frac{R_M^2}{2R_F},\qquad
r_0=R_F-cz-\Delta r}. \tag{10}
$$

当 $w_T=0$ 时使用全扫描权重；当 $w_T=1$ 时使用部分扫描权重；中间区域
连续混合。该构造满足：

- $r_0+\Delta r=R_F-cz$，分界线及其外侧完全使用部分扫描权重；
- 最后可重建层满足 $r_0-\Delta r=0$；
- 在分界线内侧距离不超过 $2\Delta r$ 的区域完成平滑过渡。

最终复合权重为

$$
\boxed{
w_C(\mathbf r,\vartheta)=
[1-w_T(\mathbf r,\vartheta)]w_{FS}(\mathbf r,\vartheta)
+w_T(\mathbf r,\vartheta)w_{PS}(\mathbf r,\vartheta)}. \tag{11}
$$

### 4.6 xFDK 反投影

最终反投影为

$$
\boxed{
f(\mathbf r)=\frac{1}{2}\int d\vartheta\,
w_C(\mathbf r,\vartheta)\,
k(\xi)*p(\vartheta,\xi,\gamma)}. \tag{12}
$$

其中 $k(\xi)$ 是沿平行束距离 $\xi$ 的一维卷积核，例如 Shepp-Logan 核。
权重位于卷积之后，因此所有体素共享同一份滤波数据。

## 5. 工程反投影坐标推导

这一节不是原文逐句翻译，而是将式 (1)、式 (2) 映射到本项目 CUDA kernel。

对体素 $(x,y,z)$ 和平行束角 $\vartheta$：

$$
\xi=x\cos\vartheta+y\sin\vartheta,
$$

$$
\eta=-x\sin\vartheta+y\cos\vartheta.
$$

由 $\xi=-R_F\sin\beta$ 得

$$
R_F\cos\beta=\sqrt{R_F^2-\xi^2}.
$$

焦点到体素的水平射线长度为

$$
L_h=\sqrt{R_F^2-\xi^2}+\eta,
$$

所以应在滤波数据中读取

$$
\boxed{
\gamma=\arctan\frac{z}{L_h}}.
$$

本项目的离散反投影对应

$$
f[\mathbf r]\approx
\frac{\Delta\vartheta}{2}
\sum_i w_C(\mathbf r,\vartheta_i)
\tilde p(\vartheta_i,\xi_i,\gamma_i).
$$

## 6. 与当前代码的对应关系

| 论文步骤 | 当前实现 |
| --- | --- |
| 理想圆轨迹和范围派生 | `YkXfdkPipeline.hpp::deriveGeometry_()` |
| 式 (1)、式 (2) 的重排 | `xfdkRebinKernel()` |
| $\cos\epsilon$ 长度修正 | `correction = cos_beta * cos_gamma` |
| 沿 $\xi$ 的公共 ramp 滤波 | 复用 `FilterProcessor` |
| 式 (6) 覆盖余量 | `xfdkCoverageHalfRange()` |
| 式 (7) 部分扫描权重 | `xfdkPartialWeight()` |
| 式 (9) 过渡权重 | `xfdkTransitionWeight()` |
| 式 (11) 复合权重 | `xfdkCompositeWeight()` |
| 式 (12) 反投影 | `xfdkBackprojectionKernel()` |
| 独立流程 | `Fdk::XfdkPipeline` |

## 7. 当前实现边界

第一版严格限制为论文模型：

- 平板探测器；
- 源位于 $z=0$ 的理想圆轨迹；
- 等角、不重复终点的完整 $2\pi$ 扫描；
- 固定 SID、SDD、像素间距；
- 无源偏移、探测器 offset、tilt 或 skew；
- 全量投影输入，不支持在线 chunk。

这些限制不是普通 FDK 框架的限制，而是当前 xFDK 推导本身的限制。任意逐视图
geometry、非等角采样和在线分包不能静默近似为该模型。

## 8. 当前验证状态

已完成的回归包括：

- $\Delta\vartheta$ 的完整覆盖、部分覆盖和不可重建三类边界；
- $w_{PS}$ 在一圈上的数值积分为 $2\pi$；
- 最后一层 $\Delta\vartheta=0$ 按极限处理为连续 $180^\circ$ 区间，而不是
  误判为无数据；
- 小尺寸 ASTRA Shepp-Logan 模体经 Joseph FP 后，xFDK 的重排、滤波和反投影
  能完成执行，输出全部有限且非零；
- 倾斜或偏移 geometry 在 `prepare()` 阶段被明确拒绝。

尚未完成论文级结论验证：需要构造“普通探测器投影 + 虚拟加高探测器真值”，
比较普通 FDK、xFDK 与大探测器 FDK 在最后扩展层的材料绝对值、冠状面、矢状面、
噪声和空间分辨率。当前 pipeline 可以用于这一步验证，但不能仅凭冒烟测试宣称
已经复现论文图 4-13 的图像质量。
