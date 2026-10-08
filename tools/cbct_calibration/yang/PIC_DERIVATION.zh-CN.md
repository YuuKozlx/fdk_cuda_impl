# Yang 2017 PIC 几何校正参数计算说明

本文记录 Yang et al. 2017 中 PIC（pose-independent calibration）方法的参数计算思路，并对应到
`yang/pic.py` 的实现。方法针对两个平行圆环；代码支持每环 6 球或每环 12 球，环半径、环间距和编号
由 `YangConfig` 提供。

```text
E1 E2 E3 E4 E5 E6    第一圆环
F1 F2 F3 F4 F5 F6    第二圆环
```

每个圆环有 6 个等角分布的钢珠。设圆环半径为 (r)，两个圆环的中心距离为 (2l)。输入是单个投影视角中 12 个钢珠的亚像素中心坐标。

这套方法是逐视角求解。它不把所有帧强行拟合成理想圆轨道，因此适合存在机械摆动、源点漂移或模体运动不规则的系统。

## 1. 坐标和参数定义

使用三个坐标系：

- 模体坐标系 (p)：原点为模体中心 (W)，轴向为 (y_p)；
- 虚拟探测器坐标系 (i)：原点 (O) 位于模体中心 (W) 的投影，(y_i) 与模体轴线平行；
- 实际探测器坐标系 (I)：坐标轴沿平板像素网格。

论文中，一个视角的主要未知量是：

```text
O          模体中心 W 在探测器上的透视投影
S^i        X 射线源在虚拟探测器坐标系中的位置
W^i        模体中心在虚拟探测器坐标系中的位置
(uoffset, voffset)  O 对应的探测器坐标
theta      roll
phi        pitch
eta        yaw
t          模体/gantry 转角
```

注意：(O) 不是主点。主点是源点沿探测器法向投影到平板上的点；在 `pic.py` 中分别输出为 `o_px` 和 `principal_point_px`。

输入像素坐标先转换为毫米：

\[
x_A^I=S_f(u_A-u_O),\qquad
y_A^I=S_f(v_A-v_O),
\]

其中 (S_f) 是探测器像素尺寸。非方形像素时，u、v 方向分别使用各自的像素尺寸。

## 2. 参数计算总流程

```text
12 个球心
   ↓
对置线交点
   ↓
O：模体中心投影
   ↓
D 或平行线方向
   ↓
eta：yaw
   ↓
两个六点椭圆
   ↓
椭圆与 x=0 的交点 A1、B1、A2、B2
   ↓
theta、S^i、W^i
   ↓
椭圆一致性一维搜索
   ↓
phi：pitch
   ↓
已标记 E1/F1 的投影
   ↓
t：gantry angle
```

## 3. (O)：模体中心投影

在模体中，以下两条空间直线经过中心 (W)：

\[
E_1F_4,\quad E_4F_1.
\]

透视投影保持共线关系，因此投影线的交点就是 (W) 的投影 (O)。另外使用：

\[
E_2F_5\cap E_5F_2,
\qquad
E_3F_6\cap E_6F_3.
\]

三个交点取平均：

\[
O=\frac{1}{3}(O_{14}+O_{25}+O_{36}).
\]

代码位置：`estimate_o()`。

输出：

```text
o_px = O 的像素坐标
```

如果两条线平行或几乎平行，交点在无穷远，说明当前姿态退化，不能用这组线稳定求 (O)。

## 4. (eta)：探测器 yaw / 面内旋转

取两个同样沿模体轴线方向的投影线：

\[
E_1F_1,\qquad E_4F_4.
\]

它们的交点记为 (D)。在 yaw 尚未校正的实际探测器坐标系中：

\[
\eta=\arctan\frac{x_D^I}{y_D^I}.
\]

如果两条投影线平行，(D) 在无穷远，此时使用公共线方向：

\[
\eta=\arctan\frac{d_x}{d_y}.
\]

随后对所有球心作二维旋转，把坐标转换到 yaw 为零的坐标系：

\[
\begin{bmatrix}x'\\y'\end{bmatrix}
=
\begin{bmatrix}
\cos\eta&-\sin\eta\\
\sin\eta&\cos\eta
\end{bmatrix}
\begin{bmatrix}x^I\\y^I\end{bmatrix}.
\]

代码位置：`estimate_yaw()`。

## 5. 两个投影椭圆

在 yaw=0 坐标系中，分别使用 E 环和 F 环的 6 个球心拟合一般椭圆：

\[
a_j(x-x_{\Omega_j})^2
b_j(y-y_{\Omega_j})^2
2c_j(x-x_{\Omega_j})(y-y_{\Omega_j})=1,
\qquad j=1,2.
\]

代码中保存为：

\[
(p-c_j)^TQ_j(p-c_j)=1.
\]

其中：

- (c_j) 是椭圆中心；
- (Q_j) 是完整二次型；
- 非零的交叉项表示投影平面还没有完全旋正。

代码位置：`fit_ellipse()` 和 `fit_ring_ellipses()`。

这里的椭圆中心 (Omega_j) 只用于后续拟合，不等于 (O)，也不等于主点。

## 6. 椭圆轴线交点 (A_1,B_1,A_2,B_2)

令 yaw=0 后的纵轴为 (x=0)。将每个椭圆与该轴求交，得到两个 y 坐标：

\[
A_1,B_1\in T_1\cap\{x=0\},
\qquad
A_2,B_2\in T_2\cap\{x=0\}.
\]

如果椭圆写成：

\[
q_{11}(x-x_c)^2+2q_{12}(x-x_c)(y-y_c)+q_{22}(y-y_c)^2=1,
\]

令 (x=0)，得到关于 y 的一元二次方程：

\[
q_{22}(y-y_c)^2
2q_{12}(-x_c)(y-y_c)
q_{11}x_c^2=1.
\]

其两个根就是交点位置。

代码位置：`Ellipse.x_axis_intersections()`。

## 7. (	heta)、(S^i)、(W^i)：二维截面解算

在包含模体轴线的 (y_i-z_i) 截面中，两个圆环可以看作一个已知矩形：

- 横向尺寸由半径 (r) 给出；
- 轴向中心距为 (2l)。

源和模体中心均位于 (y_i-z_i) 平面：

\[
S^i=(0,y_S,z_S),
\qquad
W^i=(0,y_W,z_W).
\]

由于 O 是 W 的投影，S、O、W 共线：

\[
\frac{y_S}{z_S}=\frac{y_W}{z_W}.
\]

将椭圆交点分配为 (A_1,B_1,A_2,B_2)，并将它们绕 x 轴旋转 (	heta)：

\[
\begin{bmatrix}y^i\\z^i\end{bmatrix}
=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\begin{bmatrix}y^I\\0\end{bmatrix}.
\]

论文定义：

\[
z_S=y_D^I\sin\theta.
\]

记：

\[
g=\frac{y_{B_1}^I}{y_{A_2}^I},
\qquad
g'=\frac{y_{B_2}^I}{y_{A_1}^I}.
\]

则 roll 可以由论文式 (12) 写成：

\[
\theta=\arctan\left(
\frac{r[-(1+g')y_{B_1}^I+(1+g)y_{B_2}^I]}
{l[(1+g)(1+g')y_D^I-(1+g')y_{B_1}^I-(1+g)y_{B_2}^I]}
\right).
\]

然后使用论文式 (8)-(9) 计算：

\[
z_W=z_S+
\frac{(g-1)z_Sr}{(1+g)z_S-2z_{B_1}^i},
\]

\[
y_W=z_Wleft[
\frac{2(ry_{B_1}^i-lz_{B_1}^i)}{(1+g)z_Sr}
 +\frac{l}{r}
\right],
\]

\[
y_S=y_W\frac{z_S}{z_W}.
\]

代码位置：`roll_candidates()` 和 `_roll_candidate()`。

交点有顺序歧义，所以实现会枚举四种根分配，再通过后面的 pitch 椭圆一致性选择物理分支。

### 零 roll 情况

当 (	heta\approx0) 时，(D) 在无穷远，不能把它当作有限点代入上面的公式。论文使用辅助线推导式 (14)-(16)，直接利用两个椭圆交点之间的距离计算 (z_S,z_W,y_W,y_S)。代码在 `roll_candidates()` 中单独处理这一分支。

## 8. (phi)：detector pitch

到此为止，已知：

\[
O,\eta,\theta,S^i,W^i.
\]

剩余的 pitch (phi) 通过已知圆环的理论投影与拟合椭圆的一致性求解。

圆环上任一点的虚拟探测器坐标为：

\[
P_{E,\alpha}^i=
\begin{bmatrix}
r\cos\alpha\\
y_W-l+r\sin\alpha\\
z_W
\end{bmatrix},
\]

F 环只需把 (y_W-l) 换成 (y_W+l)。

给定 (phi)，将点从虚拟坐标系变换到实际探测器坐标系，并做射线投影：

\[
E_\alpha^I(\phi)=\operatorname{project}(S^i,P_{E,\alpha}^i;\theta,\phi).
\]

代入两个椭圆隐式函数：

\[
J(\phi)=
\sum_\alpha T_1(E_\alpha^I(\phi))^2
       +\sum_\alpha T_2(F_\alpha^I(\phi))^2.
\]

求：

\[
\hat\phi=\arg\min_\phi J(\phi).
\]

代码使用粗网格扫描加一维 bounded refinement，不需要多参数黑盒优化。

代码位置：`estimate_pitch()` 和 `_ellipse_consistency()`。

## 9. (t)：gantry / 模体转角

圆环本身六重对称，只使用椭圆时无法区分具体的圆周相位。因此需要预先标记 E1、F1。

对于 E 环：

\[
\alpha_j=-t-(j-1)\frac{\pi}{3},
\qquad j=1,\ldots,6.
\]

F 环使用相同圆周相位，但轴向位置不同。

将 12 个已标记钢珠按照候选 (t) 投影到探测器，最小化：

\[
J_t(t)=\sum_{k=1}^{12}
\left\|p_k^{\text{pred}}(t)-p_k^{\text{meas}}\right\|^2.
\]

原论文将其整理成关于 (sin t,cos t) 的线性最小二乘问题。当前代码采用周期一维搜索，同时比较正、反两个编号方向，并记录选择的方向。

代码位置：`estimate_gantry_angle()`。

## 10. SDD、SOD 和主点

Yang PIC 的主要目标是逐视角完整几何，而不是只输出单一圆轨道下的 SDD/SOD。代码同时给出几个距离：

### SDD

源点变换到实际探测器坐标系后，到探测器平面的垂直距离：

\[
\mathrm{SDD}=|z_S^I|.
\]

这与 (|SO|) 不同。

### 源到模体中心距离

\[
|SW|=\|S^i-W^i\|.
\]

代码字段：`source_phantom_distance_mm`。

### 源到模体轴线距离

如果模体轴线是 (y_i)，则源到该轴线的距离为：

\[
\sqrt{x_S^2+z_S^2}=|z_S|
\]

因为 (x_S=0)。代码字段：`source_axis_distance_mm`。

它只有在模体处于理想圆轨道并且轴线就是旋转轴时，才可以解释为机械 SOD。

### 主点

源点在实际探测器平面上的法向投影是主点。若实际探测器坐标中源点为 (S^I=(x_S^I,y_S^I,z_S^I))，则主点为：

\[
P_0^I=(x_S^I,y_S^I,0).
\]

再转换回像素坐标得到 `principal_point_px`。

## 11. 偏移量和像素单位

`o_px` 是 (O) 的图像坐标，对应：

\[
u_O=\frac{x_O^I}{S_{f,u}}+u_{ref},
\qquad
v_O=\frac{y_O^I}{S_{f,v}}+v_{ref}.
\]

如果重建器把主点相对图像中心定义为 offset，则应使用：

\[
\mathrm{offset}_u=u_0-\frac{N_u-1}{2},
\qquad
\mathrm{offset}_v=v_0-\frac{N_v-1}{2}.
\]

但不同系统可能把 v 轴定义为向下，或者把 offset 定义成反号，因此不能直接把 `o_px` 写成 `offsetu/offsetv`。

## 12. 可见性和放置条件

算法对模体的绝对摆放姿态不要求很严格，但需要：

- 两个圆环的内部半径、间距和编号准确；
- 12 个球在当前视角都能被可靠检测；
- 源位于两圆环之间的轴向范围附近；
- 关键交叉线不退化为平行线；
- detector 角度尽量控制在约 (pm50^\circ) 内；
- 投影球心不要接触边界或严重重叠；
- E1/F1 标记可靠，否则无法得到绝对 (t)。

一旦有球缺失、交叉线退化、椭圆不是正定椭圆或物理分支无法选择，程序应当失败，而不是用插值或默认值伪造参数。

## 13. 与单排钢珠轨迹法的关系

两种方法不能混为一谈：

- Yang：每一帧用两组六点拟合两个椭圆，利用已知模体结构逐视角解算；
- 单排轨迹法：同一颗钢珠跨多个角度形成椭圆，再由椭圆中心、二次型和已知球间距恢复固定圆轨道参数。

Yang 方法可以处理逐帧姿态变化，但需要专用双圆环模体；单排钢珠方法可以使用普通钢珠杆，但对扫描轨迹、角度和跨帧索引的要求更高。

## 14. 代码和测试

实现文件：

```text
tools/cbct_calibration/yang/pic.py
```

测试文件：

```text
tools/cbct_calibration/yang/test_pic.py
```

运行：

```powershell
python -m unittest discover -s tools -p "test_*.py" -v
```

当前测试包含：

- 一般椭圆拟合；
- 三组交叉线求 (O)；
- 零 roll 和近零 roll；
- 非零 roll/pitch/yaw；
- 非方形像素；
- 噪声亚像素中心；
- 源位置、模体中心、SDD、主点和重投影误差；
- 缺失球和退化输入拒绝。

这些测试验证的是几何求解器。真实 FP 数据还需要单独完成平场/暗场处理、圆形目标检测、亚像素质心和 12 球固定编号。

参考论文：Yang et al., *Geometry calibration method for a cone-beam CT system*, Medical Physics, 2017, DOI: 10.1002/mp.12163。
