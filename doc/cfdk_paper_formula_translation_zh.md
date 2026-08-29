# 圆轨迹锥束 CT 的曲线滤波 FDK（C-FDK）重建算法

> 原文：Liang Li, Yuxiang Xing, Zhiqiang Chen, Li Zhang, Kejun Kang, “A curve-filtered FDK (C-FDK) reconstruction algorithm for circular cone-beam CT”, *Journal of X-Ray Science and Technology*, 19 (2011), 355-371。
>
> DOI：[10.3233/XST-2011-0299](https://doi.org/10.3233/XST-2011-0299)
>
> 以下内容按照原文第 2 节 “Materials and methods” 的段落顺序直译。公式编号、图号、符号和引用编号均与原文保持一致。原文中少数图形符号无法从 PDF 文本层可靠提取，已按原图保留其上下文，不补写推测性的曲线名称。

## 2. 材料与方法

### 2.1 用于三维重建的锥束几何

图 1 显示了锥束几何的圆形源轨迹和投影数据。本文考虑由这些锥束数据重建密度函数 $f(x,y,z)$，其中 $Oxyz$ 表示笛卡尔坐标系。由于 X 射线源与平板探测器之间的实际距离对于我们的讨论并不重要，我们选择令其等于源轨迹半径 $R$。也就是说，平板探测器被放置在旋转轴 $z$ 轴上，使该探测器的 $b$ 轴与 $z$ 轴重合。探测器像素由笛卡尔坐标 $a$ 和 $b$ 定位。一条 X 射线及其投影可以由 $p(\beta,a,b)$ 唯一确定，其中 $\beta$ 是角度参数。圆形轨迹可以表示为

$$
S(\beta)=R\cdot(\cos\beta,\sin\beta,0),
\qquad \beta\in[\beta_{\mathrm{start}},\beta_{\mathrm{end}}].
\tag{1}
$$

其中，$\beta_{\mathrm{start}}$ 和 $\beta_{\mathrm{end}}$ 分别对应扫描轨迹的起点和终点。对于整个物体的重建，角度扫描范围应满足

$$
\beta_{\mathrm{end}}-\beta_{\mathrm{start}}\geq\pi+2\gamma_m,
$$

其中 $\gamma_m$ 是最大扇角的一半。本文不考虑所谓的超短扫描 CT 重建 [27,28]。在医学和工业应用中，数据通常由平板探测器或圆柱探测器采集。在本文中，算法是针对平板探测器推导的。

> **图 1.** 采用平板探测器的圆形锥束几何。

### 2.2 FDK 和 T-FDK 算法

FDK 算法由 Feldkamp、Davis 和 Kress 于 1984 年提出，用于重建沿圆形源轨迹测得的锥束投影 [1]。FDK 是二维扇束滤波反投影（FBP）算法的一种经验性三维扩展 [29,30]。基于图 1 所示的上述锥束几何，FDK 算法可以重写为

$$
f^{\mathrm{FDK}}(x,y,z)
=\frac{1}{2}\int_0^{2\pi}
\frac{R^2}{(R+x\cos\beta+y\sin\beta)^2}
\widetilde p\!\left(\beta,a(x,y,\beta),b(x,y,z,\beta)\right)
\,\mathrm d\beta,
\tag{2}
$$

$$
\widetilde p(\beta,a,b)
=\left(
\frac{R}{\sqrt{R^2+a^2+b^2}}\,p(\beta,a,b)
\right)*h(a),
\tag{3}
$$

$$
a(x,y,\beta)
=R\frac{-x\sin\beta+y\cos\beta}
{R+x\cos\beta+y\sin\beta},
\tag{4}
$$

$$
b(x,y,z,\beta)
=\frac{zR}{R+x\cos\beta+y\sin\beta},
\tag{5}
$$

其中，$h(\bullet)$ 是众所周知的斜坡滤波器，$*$ 表示逐行的一维卷积。

概括而言，FDK 的实现包含三个步骤：数据预加权、逐行一维斜坡滤波和加权三维反投影。FDK 的简单性和易实现性使其成为锥束重建中使用最广泛的算法。然而，FDK 是近似的，其含义是：无论测量分辨率如何，重建结果都会在某种程度上偏离被测物体，尤其是在远离中平面的平面中。对于较小的锥角（约 $\pm2^\circ$），这些差异通常是可以接受的。随着锥角增大，FDK 重建的锥束伪影迅速恶化。由于 FDK 应用广泛，如何提高 FDK 重建的准确性已成为一项紧迫且具有挑战性的任务。

受螺旋重排方法 [31] 的启发，Turbell 于 1999 年通过将原始锥束数据重排为锥形平行束数据，提出了 P-FDK 算法 [10]。如图 2(a) 所示，所得射束几何在扇角方向上是平行的，而在锥角方向上是发散的。重排数据由下式得到：

$$
p^{\mathrm{P-FDK}}(\theta,t,b)
=p\!\left(
\theta-\arcsin\frac{t}{R},
\frac{tR}{\sqrt{R^2-t^2}},b
\right).
\tag{6}
$$

数据重排之后，图像按照下列公式通过预加权、斜坡滤波和反投影进行重建：

$$
\widetilde p^{\mathrm{P-FDK}}(\theta,t,b)
=\left(
\frac{R^2}{\sqrt{R^4+R^2b^2-b^2t^2}}
p^{\mathrm{P-FDK}}(\theta,t,b)
\right)*h(t),
\tag{7}
$$

$$
f^{\mathrm{P-FDK}}(x,y,z)
=\frac{1}{2}\int_0^{2\pi}
\widetilde p^{\mathrm{P-FDK}}
\!\left(\theta,t(x,y,\theta),b(x,y,z,\theta)\right)
\,\mathrm d\theta,
\tag{8}
$$

$$
t(x,y,\theta)=y\cos\theta-x\sin\theta,
\tag{9}
$$

$$
b(x,y,z,\theta)
=\frac{zR^2}
{(x\cos\theta+y\sin\theta)\sqrt{R^2-t(x,y,\theta)^2}
+R^2-t(x,y,\theta)^2}.
\tag{10}
$$

需要注意的是，P-FDK 算法只在一个方向上重排锥束数据。行坐标 $b$ 保持不变。如图 2(a) 所示，由于锥形平行束源位置与探测器之间的距离可变，P-FDK 中的滤波是在中心虚拟探测器内沿凸曲线进行的。P-FDK 通过消除反投影距离权重，提高了噪声均匀性和计算效率。然而，它并不能改善中平面之外的重建图像质量。

> **图 2.** 由 (a) P-FDK 和 (b) T-FDK 重排到中心虚拟探测器中的数据。

在 P-FDK 的基础上，Grass 等人于 2000 年提出了 T-FDK 算法，该算法还沿虚拟探测器的垂直线进行了额外重排。重排数据由下式得到：

$$
p^{\mathrm{T-FDK}}(\theta,t,s)
=p\!\left(
\theta-\arcsin\frac{t}{R},
\frac{tR}{\sqrt{R^2-t^2}},
\frac{sR^2}{R^2-t^2}
\right).
\tag{11}
$$

数据重排之后，T-FDK 按照下列公式通过预加权、斜坡滤波和反投影重建图像：

$$
\widetilde p^{\mathrm{T-FDK}}(\theta,t,s)
=\left(
\frac{\sqrt{R^2-t^2}}{\sqrt{R^2-t^2+s^2}}
p^{\mathrm{T-FDK}}(\theta,t,s)
\right)*h(t),
\tag{12}
$$

$$
f^{\mathrm{T-FDK}}(x,y,z)
=\frac{1}{2}\int_0^{2\pi}
\widetilde p^{\mathrm{T-FDK}}
\!\left(\theta,t(x,y,\theta),s(x,y,z,\theta)\right)
\,\mathrm d\theta,
\tag{13}
$$

$$
s(x,y,z,\theta)
=\frac{z\sqrt{R^2-t(x,y,\theta)^2}}
{\sqrt{R^2-t(x,y,\theta)^2}+x\cos\theta+y\sin\theta}.
\tag{14}
$$

与 P-FDK 不同，T-FDK 在两个正交方向上重排原始锥束数据 [9,10]。它沿虚拟矩形探测器 $(t,s)$ 上的水平线对投影数据进行滤波。与 FDK 相比，T-FDK 获得了改善的图像质量，这体现为强度下降减小，尤其是在远离中平面的切片中 [9,10]。

### 2.3 C-FDK 算法

本节介绍一种新的曲线滤波 FDK（C-FDK）算法。与 T-FDK 类似，C-FDK 算法也包含四个步骤：重排、预加权、逐行一维斜坡滤波以及三维反投影。主要区别在于数据重排步骤。尽管重排也在两个正交方向上进行，C-FDK 在垂直方向上以不同的方式重排锥束数据。图 3(a) 从俯视图显示了 P-FDK、T-FDK 和 C-FDK 的三种不同重排方法。

P-FDK 的重排数据位于点划线所定义圆柱面的各行上，可由式 (6) 计算。然而，T-FDK 的重排数据位于实线所定义平面的各行上。

不同于 P-FDK 和 T-FDK，C-FDK 的重排数据位于图 3(a) 中虚线所定义圆柱面的各行上。这条虚线曲线是一段与圆形轨迹具有相同曲率的圆弧。假设重排数据表示为 $p^{\mathrm{C-FDK}}(\theta,t,c)$。考虑图 3 所示的一条穿过 $(x,y,z)$ 的 X 射线，它在原始锥束几何中的投影表示为 $p(\beta,a,b)$。$C$、$O'$ 和 $P$ 分别是该 X 射线与 P-FDK、T-FDK 和 C-FDK 虚拟探测器相交后在轨迹平面上的投影点。根据上述定义，$p(\beta,a,b)$ 与 $p^{\mathrm{C-FDK}}(\theta,t,c)$ 的关系为

$$
a=OP=\frac{tR}{\sqrt{R^2-t^2}},
\tag{15}
$$

$$
b=\frac{SP}{SC}c
=\frac{cR^2}{2(R^2-t^2)-R\sqrt{R^2-t^2}}.
\tag{16}
$$

因此，C-FDK 的重排数据可以由下式得到：

$$
p^{\mathrm{C-FDK}}(\theta,t,c)
=p\!\left(
\theta-\arcsin\frac{t}{R},
\frac{tR}{\sqrt{R^2-t^2}},
\frac{cR^2}{2(R^2-t^2)-R\sqrt{R^2-t^2}}
\right).
\tag{17}
$$

数据重排之后，C-FDK 同样通过预加权、斜坡滤波和反投影来重建图像。预加权因子在几何上被解释为射线与中平面之间夹角 $\alpha$ 的余弦：

$$
\cos\alpha
=\frac{SC}{\sqrt{SC^2+c^2}}
=\frac{2\sqrt{R^2-t^2}-R}
{\sqrt{5R^2-4t^2-4R\sqrt{R^2-t^2}+c^2}}.
\tag{18}
$$

随后，C-FDK 按照下式对重排数据进行预加权和滤波：

$$
\widetilde p^{\mathrm{C-FDK}}(\theta,t,c)
=\left(
\frac{2\sqrt{R^2-t^2}-R}
{\sqrt{5R^2-4t^2-4R\sqrt{R^2-t^2}+c^2}}
p^{\mathrm{C-FDK}}(\theta,t,c)
\right)*h(t),
\tag{19}
$$

其中，$h(\bullet)$ 是斜坡滤波器，$*$ 表示逐行的一维卷积。

三维反投影公式写为

$$
f^{\mathrm{C-FDK}}(x,y,z)
=\frac{1}{2}\int_0^{2\pi}
\widetilde p^{\mathrm{C-FDK}}
\!\left(\theta,t(x,y,\theta),c(x,y,z,\theta)\right)
\,\mathrm d\theta,
\tag{20}
$$

其中，穿过体素 $(x,y,z)$ 的 X 射线的投影坐标为

$$
t(x,y,\theta)=y\cos\theta-x\sin\theta,
\tag{21}
$$

$$
c(x,y,z,\theta)
=\frac{SC}{SO'+v}z
=z\frac{2\sqrt{R^2-t^2}-R}
{\sqrt{R^2-t^2}+x\cos\theta+y\sin\theta}.
\tag{22}
$$

其中，$SC$、$SO'$ 和 $v$ 如图 3(b) 所示；图 3(b) 是一个平行于 $z$ 轴并包含穿过点 $(x,y,z)$ 的 X 射线的垂直截面。

> **图 3.** P-FDK、T-FDK 和 C-FDK 的不同重排方法。(a) 俯视图：点划线表示 P-FDK 的重排虚拟探测器；实线表示 T-FDK 的重排虚拟探测器；虚线表示 C-FDK 的重排虚拟探测器。(b) 侧视图：穿过 $(x,y,z)$ 的 X 射线的反投影路径，该图沿平行于 $z$ 轴的方向绘制。

现在，让我们分析由 C-FDK 重排得到的锥形平行束数据。如前所述，由式 (17) 重排的数据位于由与轨迹具有相同曲率的圆弧所定义圆柱面的各行上。该圆柱面如图 4(a) 所示。为了更清楚地比较不同方法，P-FDK、T-FDK 和 C-FDK 的数据被投影到图 4(b) 所示的中心虚拟平面上。在该平面内，由 C-FDK 重排且具有相同垂直位置 $c$ 的数据，位于图 4(b) 中虚线所示的一系列凹曲线上。这些曲线形成了以浅色部分表示的凹形区域。然而，在同一虚拟平面内，T-FDK 和 P-FDK 的重排数据分别位于水平线（虚线）和凸曲线（点划线）上。因此，C-FDK 沿这些凹曲线对重排数据进行滤波，这是 C-FDK 与其他现有 FDK 类算法之间最主要的区别。

为了使斜坡滤波有效，数据在横向上必须不被截断。根据式 (17)，只有图 4(b) 中浅色区域内的锥形平行束数据能够在不截断的情况下完成重排。深色区域内的投影数据在横向上被截断。为了保证斜坡滤波的有效性，应当丢弃深色区域内的数据，这会导致 C-FDK 重建区域缩小。因此，式 (17)-(20) 的物体重建区域小于 T-FDK 和 FDK 的重建区域。为了充分利用原始锥束数据并避免数据截断，深色区域内的数据应沿不同曲线重排，这些曲线既包括凹曲线，也包括凸曲线。

如图 4(b) 所示，每一行重排数据由穿过图 4(b) 中同一高度曲面的 X 射线投影计算得到。对于浅色区域内的数据，该曲面是由图 3(a) 中半径为 $R$ 的虚线圆弧定义的圆柱面。然而，为了避免截断，对于深色区域内的数据，该曲面由图 3(a) 中从虚线曲线逐渐变化到点划线曲线的曲线所定义。

> **图 4.** C-FDK 的重排数据。(a) 重排数据位于该曲面的各行上。(b) C-FDK 在中心虚拟平面中的重排数据。(c) 不同算法重建区域的侧视图：蓝色区域对应 FDK、P-FDK 和 C-FDK；橙色区域对应 T-FDK；橙色加红色区域对应 HT-FDK；蓝色加绿色区域对应 xFDK、CW-FDK 和 Tang 方法。

假设实际平板探测器的宽度和高度分别为 $-a_m\sim a_m$ 和 $-b_m\sim b_m$。根据式 (17)，无截断重排数据的最大 $c$ 可以计算为

$$
c_0=b_m\frac{2(R^2-a_m^2)-R\sqrt{R^2-a_m^2}}{R^2}.
\tag{23}
$$

类似地，通过 T-FDK 算法式 (11)，无截断重排数据的最大值 $s$ 可以计算为

$$
s_0=b_m\frac{R^2-a_m^2}{R^2}.
\tag{24}
$$

由于深色区域中的数据是以截断方式重排的，这里我们引入一种渐进重排方法，将原始锥束数据重排到由图 4 中从虚线曲线逐渐变化到点划线曲线的一组渐变曲线所定义曲面的各行上。因此，中心虚拟探测器中的数据按以下三种情况进行重排：

1. $|c|\leq c_0$。数据由式 (17) 重排。它们包含在图 4(b) 的浅色区域内。
2. $|c|>b_m$。这些数据无法由该圆形锥束数据重排得到。
3. $c_0<|c|\leq b_m$。数据包含在图 4(b) 的深色区域内。为了避免数据截断，采用一种新的重排方法，该方法可以看作从 C-FDK 到 P-FDK 的渐进过渡。其关键思想是保证沿每条曲线重排的数据均不被截断。该深色区域中的数据被分为两个部分。

#### 3.1 $c_0<|c|\leq s_0$

数据被重排到图 5(a) 中由逐渐变化的虚线曲线所定义曲面的各行上。

为了确定该曲线的路径，我们需要知道 $SC'$ 的长度，它可以通过下列比例关系计算：

$$
\begin{aligned}
SC'
&=SC+\frac{|c|-c_0}{s_0-c_0}CO'\\
&=2\sqrt{R^2-t^2}-R
+\frac{|c|-c_0}{s_0-c_0}
\left(R-\sqrt{R^2-t^2}\right).
\end{aligned}
\tag{25}
$$

因此，数据可以沿 $c$ 轴按下式重排：

$$
\begin{aligned}
b
&=\frac{SP}{SC'}c\\
&=\frac{cR^2}
{\sqrt{R^2-t^2}\left[
2\sqrt{R^2-t^2}-R
+\dfrac{|c|-c_0}{s_0-c_0}
\left(R-\sqrt{R^2-t^2}\right)
\right]}.
\end{aligned}
\tag{26}
$$

#### 3.2 $s_0<|c|\leq b_m$

与 3.1 类似，数据被重排到图 5(b) 中由逐渐变化的虚线曲线所定义曲面的各行上。

在这种情况下，$SC'$ 的长度可以通过下列比例关系计算：

$$
\begin{aligned}
SC'
&=SO'+\frac{|c|-s_0}{b_m-s_0}O'P\\
&=\sqrt{R^2-t^2}
+\frac{|c|-s_0}{b_m-s_0}
\frac{t^2}{\sqrt{R^2-t^2}}.
\end{aligned}
\tag{27}
$$

因此，数据可以沿 $c$ 轴按下式重排：

$$
\begin{aligned}
b
&=\frac{SP}{SC'}c\\
&=\frac{cR^2}
{R^2-t^2+\dfrac{|c|-s_0}{b_m-s_0}t^2}.
\end{aligned}
\tag{28}
$$

> **图 5.** 一种渐进重排方法，该方法将锥束数据重排到由从虚线曲线逐渐变化到点划线曲线的渐变曲线所定义曲面的各行上。(a) 当 $c_0<|c|\leq s_0$ 时，该曲面由逐渐变化的虚线曲线定义。(b) 当 $s_0<|c|\leq b_m$ 时，该曲面由逐渐变化的虚线曲线定义。

这里我们总结最终的 C-FDK 算法，该算法仍然包括三个步骤：数据重排，预加权和斜坡滤波，三维反投影。由原始锥束数据得到的重排过程表示为

$$
p^{\mathrm{C-FDK}}(\theta,t,c)=
\begin{cases}
p\!\left(
\theta-\arcsin\dfrac{t}{R},
\dfrac{tR}{\sqrt{R^2-t^2}},
\dfrac{cR^2}{2(R^2-t^2)-R\sqrt{R^2-t^2}}
\right),
& |c|\leq c_0,\\[2.4ex]
p\!\left(
\theta-\arcsin\dfrac{t}{R},
\dfrac{tR}{\sqrt{R^2-t^2}},
\dfrac{cR^2}
{\sqrt{R^2-t^2}\left[
2\sqrt{R^2-t^2}-R
+\dfrac{|c|-c_0}{s_0-c_0}
\left(R-\sqrt{R^2-t^2}\right)
\right]}
\right),
& c_0<|c|\leq s_0,\\[3.4ex]
p\!\left(
\theta-\arcsin\dfrac{t}{R},
\dfrac{tR}{\sqrt{R^2-t^2}},
\dfrac{cR^2}
{R^2-t^2+\dfrac{|c|-s_0}{b_m-s_0}t^2}
\right),
& s_0<|c|\leq b_m.
\end{cases}
\tag{29}
$$

随后，重排数据由 $\cos\alpha$ 进行预加权，并由斜坡滤波器进行滤波。由于 $\alpha$ 是图 1 所示的锥角，我们有

$$
\cos\alpha=\frac{SP}{\sqrt{SP^2+b^2}}.
\tag{30}
$$

最终的预加权和滤波结果可以写为

$$
\widetilde p^{\mathrm{C-FDK}}(\theta,t,c)=
\begin{cases}
\left(
\dfrac{2\sqrt{R^2-t^2}-R}
{\sqrt{5R^2-4t^2-4R\sqrt{R^2-t^2}+c^2}}
p^{\mathrm{C-FDK}}(\theta,t,c)
\right)*h(t),
& |c|\leq c_0,\\[3ex]
\left(
\dfrac{R^2}{\sqrt{R^4+b^2(R^2-t^2)}}
p^{\mathrm{C-FDK}}(\theta,t,c)
\right)*h(t),
& c_0<|c|\leq b_m.
\end{cases}
\tag{31}
$$

其中，$h(\bullet)$ 是斜坡滤波器，$*$ 表示逐行的一维卷积。根据不同的 $c$，$b$ 分别由式 (16)、式 (26) 和式 (28) 确定。三维反投影公式写为

$$
f^{\mathrm{C-FDK}}(x,y,z)
=\frac{1}{2}\int_0^{2\pi}
\widetilde p^{\mathrm{C-FDK}}
\!\left(\theta,t(x,y,\theta),c(x,y,z,\theta)\right)
\,\mathrm d\theta,
\tag{32}
$$

其中，穿过体素 $(x,y,z)$ 的 X 射线的探测器坐标为

$$
t(x,y,\theta)=y\cos\theta-x\sin\theta,
\tag{33}
$$

$$
c(x,y,z,\theta)=
\begin{cases}
\dfrac{z\,SC}{SO'+v}, & |c|\leq c_0,\\[1.5ex]
\dfrac{z\,SC'}{SO'+v}, & c_0<|c|\leq b_m,
\end{cases}
\tag{34}
$$

其中 $v=x\cos\theta+y\sin\theta$。由于 $SC$ 或 $SC'$ 在式 (29) 的三个重排公式中以不同方式计算，$c(x,y,z,\theta)$ 也按三种不同情况计算：

$$
c(x,y,z,\theta)=
\begin{cases}
\dfrac{z(2q-R)}{q+v},
& |z|\leq\dfrac{q+v}{2q-R}c_0,\\[2.4ex]
\dfrac{z\left[(2q-R)s_0-c_0q\right]}
{(q+v)(s_0-c_0)+z(q-R)},
& \dfrac{q+v}{2q-R}c_0<|z|\leq\dfrac{q+v}{q}s_0,\\[3ex]
\dfrac{z\left[(b_m-s_0)q^2-s_0t^2(x,y,\theta)\right]}
{(q^2+vq)(b_m-s_0)-zt^2(x,y,\theta)},
& \dfrac{q+v}{q}s_0<|z|\leq\dfrac{q^2+vq}{R^2}b_m,
\end{cases}
\tag{35}
$$

其中，$q(x,y,\theta)$ 由下式确定：

$$
q(x,y,\theta)=\sqrt{R^2-t^2(x,y,\theta)}.
\tag{36}
$$

> **原文公式勘误说明（不属于译文正文）**：式 (25)、式 (27) 和式 (29)
> 均以 $|c|$ 定义关于中平面对称的渐变曲线。将式 (25)、式 (27) 与式
> (34) 联立反解时，式 (35) 第二、第三段分母中的 $z$ 应按 $|z|$ 参与
> 曲线幅值计算，最后再恢复 $c$ 的符号。若直接使用印刷式中的有符号 $z$，
> 会导致 $c(x,y,-z)\ne-c(x,y,z)$，并使负 $z$ 侧在两个分界处不连续。
> 本项目通过前向曲线/反解往返、正负镜像和分界连续性数值测试采用这一解释。

> **坐标约定说明（不属于译文正文）**：原文的 $a/t/v$ 方向由图 1、图 3
> 定义。移植到其他 vector geometry 时，式 (29) 的视角符号和式 (35) 的
> $v$ 符号必须整体转换，不能只按变量名照抄。本项目中探测器 U 方向与论文
> $a$ 轴相反，因此重排视角写作 $\beta=\theta+\arcsin(t/R)$；项目源位于
> 正径向一侧，因此式 (35) 使用 $v=-(x\cos\theta+y\sin\theta)$。
