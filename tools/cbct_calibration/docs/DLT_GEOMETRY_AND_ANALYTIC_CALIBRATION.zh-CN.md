# CBCT 几何校正的三条使用路线

本文把当前代码中涉及的三种用途放在同一个坐标和公式框架下：

1. **仅做迭代重建**：逐帧 DLT 直接转换成源点和探测器平面向量，使用 Siddon 等射线驱动 FP/BP。
2. **做 FDK 等解析重建**：以逐帧 DLT 作为观测，联合拟合一个具有全局意义的圆轨道和固定平板模型。
3. **分析每帧姿态和偏移**：使用 Cho/Yang 的 PIC 思路求每帧的模体投影、源点、椭圆姿态和探测器姿态；要把这些量解释成机械 offset，还必须建立模体坐标系和机器/重建坐标系之间的关系。

这三条路线的输入都可以来自以下模体：

- 大小双环模体；
- 带特殊标记的双环模体；
- 螺旋钢珠模体（当前还需要专门测试编号和 DLT 条件）。

它们的共同要求不是“必须知道 SID、SDD、offset”，而是：在每个视角中能得到足够数量的已编号三维点与二维质心对应关系。

---

## 0. 先区分三类坐标

### 0.1 模体坐标系 (m)

三维钢珠坐标是在模体设计中给出的，例如双环可写为

\[
X_j^{(m)}=(x_j,y_j,z_j,1)^T.
\]

这个坐标系的原点通常放在模体中心，轴线沿两个圆环的公共轴。它不必一开始就等于扫描机架的等中心。

### 0.2 机器/重建坐标系 (w)

这是最终重建体的坐标系。理想圆轨道通常在该坐标系中表达为

\[
S_i^{(w)}=
SID\begin{bmatrix}
\sin\alpha_i\\-\cos\alpha_i\\0
\end{bmatrix},
\]

其中 (S_i) 是第 (i) 帧源点，\(\alpha_i\) 是机架角。

### 0.3 探测器局部坐标系

探测器有三个单位方向：

- (U_i)：图像列增加方向；
- (V_i)：图像行增加方向；
- (N_i)：探测器法向。

本文采用右手系

\[
U_i\times V_i=N_i.
\]

在当前平板实现中，(N_i) 从探测器指向源点。因此理想探测器中心位于源点的反方向；若使用相反的法向定义，下面的法向符号要整体翻转，但射线本身不变。

### 0.4 三个坐标系的刚体关系

如果已知模体坐标到机器坐标的刚体变换：

\[
X^{(w)}=T_{w\leftarrow m}X^{(m)},\qquad
T_{w\leftarrow m}=\begin{bmatrix}R&t\\0&1\end{bmatrix},
\]

那么在模体坐标中得到的 DLT 矩阵可以变换为机器坐标中的 DLT 矩阵：

\[
q_i\sim P_i^{(m)}X^{(m)},
\]

\[
X^{(m)}=T_{m\leftarrow w}X^{(w)},
\qquad
P_i^{(w)}=P_i^{(m)}T_{m\leftarrow w}.
\]

这一步是后面讨论机械 offset 的关键。没有它时，DLT 只知道“相对于模体坐标系的相机”，不知道“相对于扫描机架等中心的相机”。

---

## 1. DLT 如何得到逐帧投影矩阵

### 1.1 观测数据

第 (i) 帧中，已经完成检测和编号的钢珠观测为

\[
q_{ij}=(u_{ij},v_{ij}),
\]

对应的模体三维点为

\[
X_j=(X_j,Y_j,Z_j,1)^T.
\]

编号关系比模体的绝对摆放更重要。只要 (j) 在所有帧中表示同一颗球，DLT 就可以建立对应关系。

### 1.2 DLT 线性方程

写投影矩阵为

\[
P_i=
\begin{bmatrix}
p_{11}&p_{12}&p_{13}&p_{14}\\
p_{21}&p_{22}&p_{23}&p_{24}\\
p_{31}&p_{32}&p_{33}&p_{34}
\end{bmatrix}.
\]

齐次投影满足

\[
\begin{bmatrix}u_{ij}\\v_{ij}\\1\end{bmatrix}
\sim P_iX_j.
\]

消去比例因子得到两行线性约束：

\[
\left(X_j^T\otimes[1,0,-u_{ij}]\right)\operatorname{vec}(P_i)=0,
\]

\[
\left(X_j^T\otimes[0,1,-v_{ij}]\right)\operatorname{vec}(P_i)=0.
\]

把所有钢珠的约束堆成矩阵 (A_i)，对

\[
A_i p_i=0
\]

做 SVD。最小奇异值对应的右奇异向量重排成 (3\times4) 矩阵，就是 (P_i) 的估计。

DLT 矩阵只有一个非零尺度自由度：

\[
P_i\sim\lambda_iP_i.
\]

因此保存或比较矩阵前必须统一归一化，例如令 (P_{i,34}=1)，或令某个稳定的矩阵元素为 1。归一化只影响数值表示，不影响射线。

### 1.3 DLT 质量检查

对每颗球重新投影：

\[
\hat q_{ij}=\pi(P_iX_j),
\]

其中 \(\pi([a,b,c]^T)=(a/c,b/c)\)。计算

\[
RMSE_i=\sqrt{\frac1{N_i}\sum_j\|\hat q_{ij}-q_{ij}\|^2}.
\]

DLT 只应使用检测可靠、编号可靠的帧。若某帧缺球、粘连、越界或编号错误，最小二乘仍可能给出一个矩阵，但矩阵的几何意义会变差；因此应同时保存有效点数、重投影 RMSE、奇异值间隙和异常帧状态。

---

## 2. DLT 到逐帧 cone-vector 几何

这是“只做迭代重建”时最重要的转换。它不需要先求 SID、SDD、探测器 tilt 或机械 offset。

### 2.1 DLT 的相机中心

把 DLT 分成

\[
P=[M\mid p_4],\qquad M=P_{[:,1:3]}.
\]

相机中心（也就是 X 射线源点）满足

\[
P\begin{bmatrix}S\\1\end{bmatrix}=0.
\]

因此

\[
\boxed{S=-M^{-1}p_4.}
\]

这一步只需要 DLT，不需要模体中心在机器中的位置。得到的 (S) 仍然是在 DLT 使用的坐标系中；如果 DLT 是模体坐标，(S) 也是模体坐标。

### 2.2 任意像素对应的射线

令像素齐次坐标为

\[
q(u,v)=(u,v,1)^T.
\]

则从源点出发的射线方向可以直接写成

\[
d(u,v)=M^{-1}q(u,v).
\]

射线为

\[
X(\tau)=S+\tau d(u,v).
\]

如果只关心射线方向，可以将 (d) 单位化。注意：DLT 的矩阵尺度会同时改变 (M^{-1})，但方向不变。

这已经足以做逐射线的 Siddon FP/BP。它不把体素中心当成射线，也不需要假定体素穿过中心；Siddon 会计算每条射线穿过每个体素的实际长度。

### 2.3 RQ 分解后的物理解释

也可以对 (M) 做 RQ 分解：

\[
M=KR,
\]

其中 (K) 是上三角内参矩阵，(R) 是正交矩阵。经过符号调整后，可以写成

\[
K=\begin{bmatrix}
f_u&0&u_0\\
0&f_v&v_0\\
0&0&1
\end{bmatrix}.
\]

这里：

- (f_u,f_v) 是以像素为单位的等效焦距；
- ((u_0,v_0)) 是源点沿相机法向投影到代数探测器平面的主点；
- (R) 把世界/模体坐标变到相机坐标。

令相机矩阵已经经过 RQ 分解，并把 (K) 归一化为 (K_{33}=1)。这一步很重要：原始 DLT 只确定 (P) 的任意尺度，不能直接把未归一化的 (M^{-1}q) 当作“到固定物理平面距离 (L)”的点。令

\[
e_u=R^T(1,0,0)^T,\quad
e_v=R^T(0,1,0)^T,\quad
e_z=R^T(0,0,1)^T.
\]

它们是世界坐标中的三个相机轴方向。选一个正的虚拟探测器平面距离 (L)，则平面上像素 ((u,v)) 对应的物理点可定义为

\[
D(u,v)=S+L\,M^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix}.
\]

在 (M=KR)、(K_{33}=1) 的规范下，

\[
e_z^T M^{-1}[u,v,1]^T=1,
\]

所以该点确实位于 (S+Le_z) 的平面上。若使用的是尚未做这种规范化的原始 DLT，则应先用 RQ 分解归一化，或者对每个像素方向使用

\[
D(u,v)=S+\frac{L}{e_z^Td(u,v)}d(u,v),
\qquad d(u,v)=M^{-1}[u,v,1]^T.
\]

这样才能保证所有像素点位于同一个虚拟探测器平面。

因为 (K^{-1}[u,v,1]^T=((u-u_0)/f_u,(v-v_0)/f_v,1)^T)，所以也可以展开为

\[
D(u,v)=D_0+(u-u_0)\frac{L}{f_u}e_u
                 +(v-v_0)\frac{L}{f_v}e_v,
\]

其中

\[
D_0=S+Le_z.
\]

### 2.4 `src/detS/detU/detV` 的具体公式

cone-vector 接口使用的约定是：

\[
D(u,v)=detS+u\,detU+v\,detV,
\]

其中 `detS` 是像素 ((0,0)) 的探测器物理点，`detU`、`detV` 是每增加一个像素时的物理位移向量。

因此：

\[
\boxed{detU=\frac{L}{f_u}e_u,\qquad
detV=\frac{L}{f_v}e_v,}
\]

\[
\boxed{detS=D_0-u_0detU-v_0detV.}
\]

等价地，直接用 DLT 写成：

\[
\boxed{detS=S+L M^{-1}(0,0,1)^T,}
\]

\[
\boxed{detU=L M^{-1}(1,0,0)^T,\qquad
detV=L M^{-1}(0,1,0)^T.}
\]

这两个写法完全相同。

`L` 可以取 DLT 分解得到的 SDD，也可以取一个固定的虚拟距离，例如 1000 mm。对重建射线而言，`L` 不是一个必须等于真实 SDD 的物理测量值，因为把同一帧的 `detS/detU/detV` 同时乘以一个正比例，只改变平面参数化，不改变源到每个像素的射线。

实际代码中会检查代数探测器平面是否位于源点朝向重建体的一侧。如果 RQ 分解产生的 (e_z) 方向相反，则同时翻转平面方向和面内步长，保持所有射线方向一致。

### 2.5 像素坐标方向

如果 DLT 使用图像向上坐标，而 RAW 的行号是向下增加，则必须先做：

\[
v_{down}=N_v-1-v_{up}.
\]

对应的矩阵变换为

\[
P_{down}=
\begin{bmatrix}
1&0&0\\
0&-1&N_v-1\\
0&0&1
\end{bmatrix}P_{up}.
\]

如果不处理这一点，矩阵重投影可能看似合理，但 `detV` 方向会与重建器的行方向相反。

### 2.6 为什么这就可以直接做迭代重建

对每个视角、每个探测器像素，cone-vector 给出一条完整射线：

\[
r_{iuv}(\tau)=S_i+\tau\left(D_i(u,v)-S_i\right).
\]

Siddon 将这条射线与体素盒相交，得到穿过每个体素的长度 (ell_{iuv,k})。于是系统矩阵为

\[
A_{r,k}=\ell_{r,k},
\qquad
g_r=\sum_kA_{r,k}f_k.
\]

FP 是 (g=Af)，精确的离散 BP 是 (A^Tg)。SART、OS-SART、SIRT 都只是在这个 (A/A^T) 上选择不同的批次和归一化方式。

因此，迭代重建所需的是：

```text
每帧 DLT 或等价 cone-vector
探测器像素尺寸和像素索引定义
重建体的原点、尺寸、体素间距
```

不需要先把 DLT 唯一分解成真实机械 `SID/SDD/offset/tilt`。这也是 DLT 对迭代重建最直接的价值。

### 2.7 重建体原点如何确定

DLT 只在“输入三维点所在的坐标系”中定义射线。如果所有 DLT 都是在模体坐标系中计算的，有两种合法选择：

1. 直接在模体坐标系中定义重建体；
2. 求一个刚体变换 (T_{w\leftarrow m})，将所有 DLT 或所有 cone-vector 变到重建坐标系。

不能只把 `src` 转到机器坐标而不转 `detS/detU/detV`。完整刚体变换必须同时作用于源点和探测器向量：

\[
src^w=Rsrc^m+t,
\qquad
detS^w=RdetS^m+t,
\qquad
detU^w=RdetU^m,
\qquad
detV^w=RdetV^m.
\]

---

## 3. 为解析重建联合求解全局几何参数

### 3.1 为什么不能只把逐帧 DLT 当作 FDK 参数

逐帧 DLT 描述的是每一帧真实射线，它允许：

- 源点轨迹不是严格圆；
- 探测器每帧有摆动；
- 模体坐标系任意平移和旋转；
- 每帧的 SDD、主点和探测器轴不同。

这对通用射线驱动迭代重建没有问题，但标准 FDK 通常需要一个共享的圆轨道模型：

\[
SID,\ SDD,\ offset_u,\ offset_v,\ tilt_u,\ tilt_v,\ tilt_n,
\]

以及每帧角度 (alpha_i)。因此需要用 DLT 观测来拟合这个全局模型。

### 3.2 当前采用的约束规范

当前联合求解固定：

\[
source\_offset_x=source\_offset_y=source\_offset_z=0,
\qquad offset_n=0.
\]

优化的机器参数是 7 个：

\[
\beta=(SID,SDD,o_u,o_v,t_u,t_v,t_n).
\]

模体刚体姿态仍作为 nuisance state：

\[
\gamma=(\rho_x,\rho_y,\rho_z,t_x,t_y,t_z).
\]

总状态为

\[
x=(\beta,\gamma),
\]

但最终报告的机器参数只有 (eta)。

固定源偏移不是断言真实机器源点绝对没有偏移，而是选定一个可重建的等效规范。源偏移、探测器偏移和模体整体平移存在补偿自由度；不固定其中一部分时，优化可以得到多组重投影同样准确、但机械解释不同的结果。

### 3.3 共享圆轨道模型

令

\[
e_{r,i}=(\cos\varphi_i,\sin\varphi_i,0),
\]

\[
U_{0,i}=(-\sin\varphi_i,\cos\varphi_i,0),
\qquad
V_{0,i}=(0,0,1),
\qquad
N_{0,i}=e_{r,i}.
\]

当前右手系中，源点为

\[
S_i=SID\,e_{r,i}.
\]

先绕局部 (U) 轴施加 (t_u)，再绕局部 (V) 轴施加 (t_v)，最后绕局部 (N) 轴施加 (t_n)，得到 (U_i,V_i,N_i)。这里的旋转顺序必须与投影器保持一致；换一个顺序，三个角度的数值就不再对应同一物理定义。

理想探测器中心为

\[
D_{0,i}=-(SDD-SID)e_{r,i}.
\]

在当前约束下，实际探测器中心为

\[
D_i=D_{0,i}+o_uU_i+o_vV_i.
\]

如果开放 `offset_n`，还要加 (o_nN_i)；本方案将其固定为 0。

### 3.4 模体点变换

模体设计点 (X_j^{(m)}) 经刚体姿态变为机器坐标：

\[
X_j^{(w)}=R(\rho)X_j^{(m)}+t.
\]

第 (i) 帧的射线与探测器平面相交。令

\[
r_{ij}=X_j^{(w)}-S_i.
\]

交点参数为

\[
\lambda_{ij}=\frac{(D_i-S_i)\cdot N_i}
                         {r_{ij}\cdot N_i}.
\]

于是交点为

\[
H_{ij}=S_i+\lambda_{ij}r_{ij}.
\]

预测像素坐标：

\[
\hat u_{ij}=u_c+\frac{(H_{ij}-D_i)\cdot U_i}{\Delta u},
\]

\[
\hat v_{ij}=v_c+\frac{(H_{ij}-D_i)\cdot V_i}{\Delta v}.
\]

其中 ((u_c,v_c)=((N_u-1)/2,(N_v-1)/2))，(Delta u,Delta v) 是像素尺寸。

### 3.5 联合优化目标

对所有视角和所有已编号钢珠建立重投影残差：

\[
r_{ij}(x)=
\begin{bmatrix}
\hat u_{ij}(x)-u_{ij}\\
\hat v_{ij}(x)-v_{ij}
\end{bmatrix}.
\]

最小化

\[
\min_x\sum_{i,j}\rho\left(
\frac{\|r_{ij}(x)\|^2}{\sigma^2}
\right),
\]

其中 (ho) 可使用 soft-(L_1) 或 Huber 损失，降低少数质心异常点对全局参数的影响。

### 3.6 DLT 如何提供联合求解初值

推荐的初始化顺序是：

1. 对每帧 DLT 求源点 (S_i=-M_i^{-1}p_{4,i})；
2. 将所有 (S_i) 拟合三维圆，圆半径作为 (SID) 初值；
3. 从 DLT RQ 分解得到每帧 SDD 和主点，使用中位数作为 (SDD)、(o_u)、(o_v) 初值；
4. 由源轨迹平面和标记钢珠确定模体坐标到机器坐标的刚体初值；
5. 将模体旋转和平移放入 nuisance state；
6. 使用所有帧、所有钢珠一起进行非线性重投影优化。

这比用仿真真值直接初始化更符合实际标定流程。仿真真值可以用于验证，但不能作为生产标定输入。

### 3.7 联合优化输出及其用途

应至少输出：

```text
SID, SDD
offset_u, offset_v, offset_n=0
tilt_u, tilt_v, tilt_n
模体旋转和平移 nuisance state
每帧/总体重投影 RMSE
每帧残差和异常帧
固定参数清单
```

得到 7 个全局参数后，可生成每帧的标准圆轨道 cone-vector，供 FDK 使用。也可以保留逐帧 DLT cone-vector，与 FDK 结果进行对照。

需要特别区分：

- `DLT -> cone-vector` 给出的是实际射线；
- `DLT -> 联合参数` 给出的是在约束规范下的全局等效机械模型；
- 后者若固定了源偏移，不能再声称恢复了真实机械源偏移。

---

## 4. Cho/Yang PIC 如何求每帧姿态和相关量

### 4.1 PIC 的核心思想

PIC（pose-independent calibration）的目标是：不要求模体在世界坐标中精确摆放，仅凭单帧中专门设计的钢珠结构，求出该帧相对于模体/虚拟探测器的几何量。

Yang 双环模体的单帧输入是两组六点：

```text
E1...E6：第一圆环
F1...F6：第二圆环
```

Cho 的一般 DLT 方法则直接使用三维点和二维点对应关系，适用于更一般的空间模体。两者都可以得到逐帧投影关系，但 PIC 的中间量更具有论文中的几何意义。

### 4.2 (O)：模体中心 (W) 的投影

在模体中，以下空间连线经过模体中心 (W)：

\[
E_1F_4,\quad E_4F_1,
\]

以及循环交换得到的两组线。透视投影保持共线关系，所以两条投影直线的交点是 (W) 在探测器上的投影 (O)：

\[
O_{14}=\operatorname{line}(E_1,F_4)\cap
        \operatorname{line}(E_4,F_1).
\]

使用三组交点平均：

\[
\boxed{O=\frac{O_{14}+O_{25}+O_{36}}3.}
\]

这里的 (O) 不是椭圆中心，也不是主点。它是模体中心 (W) 的透视投影。

若二维直线写为 (a u+b v+c=0)，两条线的交点可由叉乘求得。若两条线近似平行，交点趋于无穷远，说明当前姿态对该 PIC 步骤退化，应拒绝该帧或换用稳定的线方向约束。

### 4.3 (eta)：探测器面内旋转（yaw）

取两条沿模体轴线方向的投影线，例如 (E_1F_1) 与 (E_4F_4)，求它们的交点 (D)。在未校正 yaw 的图像坐标中，交点方向给出面内旋转：

\[
\eta=\operatorname{atan2}(x_D^I,y_D^I).
\]

然后对所有像素点作二维旋转：

\[
\begin{bmatrix}x'\\y'\end{bmatrix}
=
\begin{bmatrix}
\cos\eta&-\sin\eta\\
\sin\eta&\cos\eta
\end{bmatrix}
\begin{bmatrix}x^I\\y^I\end{bmatrix}.
\]

如果交点在无穷远，则使用两条线的公共方向估计 (eta)。

### 4.4 两个椭圆及其中心

旋正 yaw 后，每个圆环的投影拟合为椭圆：

\[
(p-c_k)^TQ_k(p-c_k)=1,
\qquad k\in\{E,F\}.
\]

其中 (c_k) 是椭圆中心，(Q_k) 是完整二次型。

椭圆中心只表示该圆环投影的中心位置：

\[
c_k=-\frac12Q_k^{-1}d_k
\]

（若使用展开形式 (p^TQp+d^Tp+f=0)）。它一般不等于 (O)，也不等于主点。

### 4.5 (	heta)、(S^i)、(W^i)

在包含模体轴线的虚拟探测器截面中，令

\[
S^i=(0,y_S,z_S),
\qquad W^i=(0,y_W,z_W).
\]

由于 (O) 是 (W) 的投影，(S,O,W) 共线：

\[
\frac{y_S}{z_S}=\frac{y_W}{z_W}.
\]

令两个椭圆与 (x=0) 轴的交点为 (A_1,B_1,A_2,B_2)。把这些点绕 x 轴旋转 (	heta)：

\[
\begin{bmatrix}y^i\\z^i\end{bmatrix}
=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\begin{bmatrix}y^I\\0\end{bmatrix}.
\]

圆环半径为 (r)，两环半间距为 (l)。论文利用四个交点的比例关系求 roll：

\[
g=\frac{y_{B_1}^I}{y_{A_2}^I},
\qquad
g'=\frac{y_{B_2}^I}{y_{A_1}^I},
\]

\[
\theta=\arctan\left[
\frac{r[-(1+g')y_{B_1}^I+(1+g)y_{B_2}^I]}
{l[(1+g)(1+g')y_D^I-(1+g')y_{B_1}^I-(1+g)y_{B_2}^I]}
\right].
\]

再由论文式 (8)-(9) 求 (z_W,y_W,y_S)：

\[
z_S=y_D^I\sin\theta,
\]

\[
z_W=z_S+\frac{(g-1)z_Sr}
{(1+g)z_S-2z_{B_1}^i},
\]

\[
y_W=z_W\left[
\frac{2(ry_{B_1}^i-lz_{B_1}^i)}{(1+g)z_Sr}+\frac lr
\right],
\qquad
y_S=y_W\frac{z_S}{z_W}.
\]

当 (	heta\approx0) 时，某些辅助交点位于无穷远，不能直接代入上式，需要使用论文的零 roll 分支公式。

### 4.6 (phi)：探测器 pitch

此时已知 (O,eta,	heta,S^i,W^i)。对任意候选 pitch (phi)，将理论圆环点投影到实际探测器：

\[
P_{E,\alpha}^i=
\begin{bmatrix}
r\cos\alpha\\
y_W-l+r\sin\alpha\\
z_W
\end{bmatrix}.
\]

把候选点代入两个实测椭圆隐式函数 (T_E,T_F)，构造一致性目标：

\[
J(\phi)=\sum_\alpha T_E(E_\alpha^I(\phi))^2
       +\sum_\alpha T_F(F_\alpha^I(\phi))^2.
\]

\[
\boxed{\hat\phi=\arg\min_\phi J(\phi).}
\]

因此 pitch 不是从椭圆中心直接读出的，而是通过“已知圆环几何投影后仍落在实测椭圆上”的一致性搜索得到。

### 4.7 (t)：机架/模体相位角

两个六球圆环自身具有六重对称性，仅靠椭圆不能区分圆周起点。因此需要标记球确定 E1/F1。设候选相位为 (t)，理论圆周角可写为

\[
\alpha_j=-t-(j-1)\frac\pi3.
\]

将 12 个理论点投影后，最小化

\[
J_t(t)=\sum_{j=1}^{12}
\left\|p_j^{pred}(t)-p_j^{meas}\right\|^2.
\]

原论文可把这个问题整理成关于 (sin t,cos t) 的线性最小二乘；当前实现使用周期一维搜索，并比较正、反编号方向。

### 4.8 PIC 输出的距离和主点

将 (S^i) 变换到实际探测器坐标，可得到：

- `SDD`：源点到探测器平面的法向距离

  \[
  SDD=|z_S^I|;
  \]

- 源到模体中心距离

  \[
  |SW|=\|S^i-W^i\|;
  \]

- 源到模体轴线的距离。在 (x_S=0) 的 PIC 坐标中为

  \[
  \sqrt{x_S^2+z_S^2}=|z_S|;
  \]

- 主点：源点沿探测器法向投影到探测器平面上的点。

如果实际探测器坐标中源点为 (S^I=(x_S^I,y_S^I,z_S^I))，主点为

\[
P_0^I=(x_S^I,y_S^I,0).
\]

将其换成像素坐标后得到 `principal_point_px`。

`O` 的像素坐标则是 `o_px`。二者必须分别保存：

```text
o_px                  = 模体中心 W 的投影
principal_point_px    = 源点法向投影
```

不能把 `o_px` 直接当作 detector offset。

### 4.9 Cho 的面内旋转 eta

Cho 2005 的 eta 同样表示探测器绕自身法向的面内旋转，但计算方法不同于 Yang。Cho 不使用椭圆长轴方向的平均值，而是严格使用论文式 (14)-(16)。

对第 (k) 个椭圆：

\[
a_kU^2+b_kV^2+2c_kUV=1,
\]

其 U 方向解析极值点满足附录式 (A4)-(A5)：

\[
U_k=\pm\sqrt{\frac{b_k}{a_kb_k-c_k^2}},
\qquad
V_k=-\frac{c_k}{b_k}U_k.
\]

连接同一椭圆的两个极值点得到 (L_k)。两个椭圆中心为 (P_1,P_2)，先按式 (14) 计算：

\[
P_a=\frac{P_1\sqrt{a_1/b_1}+P_2\sqrt{a_2/b_2}}
{\sqrt{a_1/b_1}+\sqrt{a_2/b_2}}.
\]

然后按式 (16) 计算：

\[
\boxed{
\eta=\frac{|P_1P_a|A(X_I,L_1)+|P_aP_2|A(X_I,L_2)}{|P_1P_2|}.}
\]

当前 `cho_analytic.py` 保存 (P_a,L_1,L_2)、线角、权重和解析 eta，并用完整投影矩阵分解得到的 eta 作独立交叉检查。以前的“两个椭圆长轴角平均”近似已经移除。

---

## 5. 为什么 PIC 能求姿态，却不自动给出真实 detector offset

### 5.1 从逐帧 PIC/DLT 能可靠得到什么

在模体坐标系或虚拟探测器坐标系中，逐帧可以得到：

```text
O_i：模体中心投影
W_i：模体中心在虚拟坐标中的位置
S_i：源点位置
theta：roll
phi：pitch
eta：yaw
t：模体/机架相位
SDD_i
principal_point_i
每帧完整投影矩阵或 cone-vector
```

这些量足以定义该帧的射线，也足以检查姿态是否稳定。

### 5.2 detector offset 的定义需要机器基准

令 DLT/PIC 分解后第 (i) 帧探测器中心为

\[
D_i=detS_i+\frac{N_u-1}{2}detU_i
                  +\frac{N_v-1}{2}detV_i.
\]

在机器坐标中，理想探测器中心应为

\[
D_i^0=S_i-SDD_iN_i^0,
\]

其中 (N_i^0) 是理想探测器法向。实际中心相对理想中心的差为

\[
\Delta_i=D_i-D_i^0.
\]

投影到实际探测器局部轴：

\[
o_{u,i}=\Delta_i\cdot U_i,
\qquad
o_{v,i}=\Delta_i\cdot V_i,
\qquad
o_{n,i}=\Delta_i\cdot N_i.
\]

这才是机械意义下的 detector offset 分解。

但是 (D_i,S_i,U_i,V_i) 必须已经在机器/重建坐标系中。若它们仍在模体坐标系中，模体的平移和旋转会一起出现在 (Delta_i) 中。

### 5.3 模体严格摆放时为什么容易

若假设：

\[
W\equiv O_{recon},qquad
R_{m\to w}\approx I,qquad
\text{模体轴线}\parallel\text{机架旋转轴},
\]

则模体坐标原点可以近似直接作为重建坐标原点，PIC 得到的探测器中心和姿态可以直接与理想圆轨道比较。因此论文中常见“由 (O_i,W_i) 推导 offset”的公式隐含了这个坐标规范。

### 5.4 实际模体摆放不准确时的问题

若模体有平移 (t_m) 或旋转 (R_m)，则同一组投影可以由不同的组合解释：

```text
模体平移 + detector offset
模体旋转 + detector tilt
源偏移 + SID/SDD/offset 的补偿
```

仅凭一组模体投影，通常不能把这些量全部唯一拆开。因此：

- 逐帧 DLT/cone-vector 仍然可靠；
- PIC 的逐帧姿态量仍然有意义；
- 但把它们直接命名为机械 detector offset，需要额外机器坐标约束；
- 联合求解固定 source offset 和 offsetn 后得到的是等效参数。

### 5.5 Cho 的参考坐标思想

Cho 的核心做法不是要求模体精确放置，而是用逐帧 DLT 得到的源点拟合一个圆轨迹，并建立一个使该轨迹最接近理想机架圆周的替代参考系。结果相当于：

1. 先把模体坐标中的源轨迹变换到一个“机架化”的参考坐标；
2. 在该参考坐标中报告源轨迹、探测器姿态和主点；
3. 重建时使用这个参考坐标，而不是原来模体的任意摆放坐标。

因此 Cho 的结果可以稳定地用于射线重建，但它不自动恢复“模体相对于真实机械等中心的绝对平移”。

---

## 6. 三种模体在当前流程中的用途

### 6.1 大小双环模体

适合做：

- 逐帧 DLT；
- 源轨迹拟合；
- 7 参数全局联合求解；
- 迭代重建的 cone-vector 输入。

大小不同的环可以帮助首帧编号和环的身份判定，但本质仍是“空间信息丰富、点对应可确定”的 DLT 模体。

### 6.2 带标记双环模体

在大小双环基础上增加唯一标记，可解决：

- 两个环交换；
- 顺时针/逆时针编号；
- 圆周相位 (t)；
- DLT 到机器参考系的初始相位。

它特别适合从逐帧 DLT 源轨迹构造刚体变换，再进行联合拟合。

### 6.3 螺旋钢珠模体

螺旋分布破坏了平面圆环的对称性，理论上能提供更强的三维编号辨识能力。要用于当前流程，需要验证：

1. 三维点集的 DLT 条件数；
2. 是否能稳定得到首帧编号；
3. 后续帧是否能用运动预测完成永久索引；
4. 局部粘连和出界时是否仍有足够点支撑 DLT；
5. DLT 得到的 cone-vector 与已知仿真射线的重投影误差。

如果这些条件满足，螺旋模体不需要额外的 FDK 参数分解，也可以直接服务于迭代重建。

---

## 7. 推荐的实际工作流

### 路线 A：迭代重建优先

```text
RAW
→ 圆形目标检测
→ 亚像素质心
→ 跨帧永久编号
→ 每帧 DLT
→ DLT 投影矩阵质量检查
→ DLT → src/detS/detU/detV
→ Siddon FP/BP
→ SART / OS-SART / SIRT
```

这条路线不要求先求真实机械参数。

### 路线 B：需要 FDK 或统一解析几何

```text
逐帧 DLT
→ 源点圆轨迹和刚体参考系
→ DLT 初始 SID/SDD/主点/姿态
→ 固定 source_offset=0、offset_n=0
→ 7 参数 + 模体姿态联合重投影优化
→ 生成全局解析几何
→ FDK
```

必须同时报告重投影 RMSE 和固定规范，不能只报告一组看似精确的机械数字。

### 路线 C：分析逐帧姿态/偏移

```text
专用双环 PIC 或逐帧 DLT
→ O_i、椭圆、S_i、W_i、theta/phi/eta、SDD_i、主点_i
→ 建立模体→机器刚体变换
→ 求实际探测器中心 D_i
→ 与理想中心 D_i^0 比较
→ 得到 offset_u/v/n 的机器坐标解释
```

如果没有模体到机器坐标的可靠约束，最后一步只能报告“相对于模体参考系的等效偏移”，不能称为真实机械 offset。

---

## 8. 当前代码对应关系

| 目标 | 主要实现 |
|---|---|
| DLT 到射线和 cone-vector | `tools/cbct_calibration/geometry/dlt_to_conevec.py`、`src/DLTFpBp/YkDltProjectionGeometry.hpp` |
| 逐帧 DLT 射线 Siddon | `tools/cbct_calibration/geometry/dlt_siddon.py` |
| 7 参数联合求解 | 对应业务目录中的 `joint_fit.py` |
| Yang PIC | `tools/cbct_calibration/yang_pic.py` |
| Yang 参数推导 | `tools/cbct_calibration/yang/PIC_DERIVATION.zh-CN.md` |
| Cho 逐帧 DLT/共享参数 | `tools/cbct_calibration/cho/dlt.py` |

最重要的边界是：

```text
DLT / cone-vector：定义射线，可直接用于迭代重建
联合 7 参数：定义全局等效机器模型，适合 FDK
PIC：提供每帧姿态和论文中的中间几何量
机械 offset：还需要机器坐标基准或额外约束
```
