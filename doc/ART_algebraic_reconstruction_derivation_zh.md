# CT 迭代代数重建：ART 推导与算法说明

## 1. 概述

代数重建技术（Algebraic Reconstruction Technique，ART）将 CT 重建表述为一个大型线性方程组的求解问题。它逐条或分组利用投影射线的测量误差修正图像，使当前估计逐步满足投影数据。

ART 的核心思想是：对每一条射线对应的线性约束，将当前图像估计投影到该约束超平面上。该方法的数学基础是 Kaczmarz 迭代。

本文采用二维离散图像说明，三维锥束 CT 的形式完全类似，只是像素替换为体素、射线长度权重由三维投影算子计算。

## 2. CT 的离散前向模型

将待重建图像离散为 (N) 个像素，像素衰减系数组成向量

\[
\mathbf{x}=[x_1,x_2,\ldots,x_N]^T.
\]

将所有射线测量值排列为投影向量

\[
\mathbf{p}=[p_1,p_2,\ldots,p_M]^T.
\]

第 (i) 条射线的线积分可以近似写成

\[
p_i=\sum_{j=1}^{N}a_{ij}x_j,
\]

其中 (a_{ij}) 是第 (i) 条射线在第 (j) 个像素中的有效路径长度。因此所有射线组成线性系统

\[
\boxed{\mathbf{A}\mathbf{x}=\mathbf{p}},
\]

其中 (mathbf{A}\in\mathbb{R}^{M\times N}) 为系统矩阵。

在实际实现中通常不显式存储完整的 (mathbf{A})，而是通过 Joseph、Siddon、distance-driven 等前向投影和反投影算子计算矩阵-向量乘积。

## 3. 单条射线对应的约束

系统矩阵的第 (i) 行记为

\[
\mathbf{a}_i^T=[a_{i1},a_{i2},\ldots,a_{iN}].
\]

第 (i) 条射线给出一个线性约束

\[
\mathbf{a}_i^T\mathbf{x}=p_i.
\]

从几何上看，这个方程定义了图像空间中的一个超平面。理想无噪声情况下，所有射线超平面的交集就是重建解。

给定当前估计 (mathbf{x}^{(k)})，它在第 (i) 条射线上的预测投影为

\[
\hat p_i^{(k)}=\mathbf{a}_i^T\mathbf{x}^{(k)}.
\]

相应的投影残差为

\[
r_i^{(k)}=p_i-\hat p_i^{(k)}
=p_i-\mathbf{a}_i^T\mathbf{x}^{(k)}.
\]

## 4. ART 更新公式的推导

希望从当前点 (mathbf{x}^{(k)}) 出发，找到距离它最近、同时满足第 (i) 条射线方程的点：

\[
\min_{\mathbf{x}}\frac{1}{2}\|\mathbf{x}-\mathbf{x}^{(k)}\|_2^2
\quad\text{s.t.}\quad
\mathbf{a}_i^T\mathbf{x}=p_i.
\]

构造拉格朗日函数

\[
L(\mathbf{x},\mu)=
\frac{1}{2}\|\mathbf{x}-\mathbf{x}^{(k)}\|_2^2
+\mu(\mathbf{a}_i^T\mathbf{x}-p_i).
\]

令对 (mathbf{x}) 的梯度为零：

\[
\nabla_{\mathbf{x}}L
=\mathbf{x}-\mathbf{x}^{(k)}+\mu\mathbf{a}_i=0.
\]

于是

\[
\mathbf{x}=\mathbf{x}^{(k)}-\mu\mathbf{a}_i.
\]

代入约束 (mathbf{a}_i^T\mathbf{x}=p_i)，得到

\[
\mathbf{a}_i^T\mathbf{x}^{(k)}-mu\|\mathbf{a}_i\|_2^2=p_i,
\]

从而

\[
\mu=\frac{\mathbf{a}_i^T\mathbf{x}^{(k)}-p_i}{\|\mathbf{a}_i\|_2^2}.
\]

代回可得标准正交投影：

\[
\mathbf{x}^{(k+1)}
=\mathbf{x}^{(k)}
+\frac{p_i-\mathbf{a}_i^T\mathbf{x}^{(k)}}{\|\mathbf{a}_i\|_2^2}\mathbf{a}_i.
\]

引入松弛因子 (lambda) 后，ART 的基本更新公式为

\[
\boxed{
\mathbf{x}^{(k+1)}
=\mathbf{x}^{(k)}
+\lambda\frac{p_i-\mathbf{a}_i^T\mathbf{x}^{(k)}}{\|\mathbf{a}_i\|_2^2}\mathbf{a}_i
}
\]

其中：

- (lambda=1)：完整投影到当前超平面；
- (0<lambda<1)：欠松弛，通常更稳定、抗噪声更好；
- (1<lambda<2)：过松弛，可能加快收敛，但更容易放大噪声。

## 5. 像素级更新形式

第 (j) 个像素的更新为

\[
x_j^{(k+1)}
=x_j^{(k)}
+lambda
\frac{p_i-\sum_{l=1}^{N}a_{il}x_l^{(k)}}{\sum_{l=1}^{N}a_{il}^2}
a_{ij}.
\]

该式说明：

1. 先用当前图像计算第 (i) 条射线的预测值；
2. 计算实测投影与预测投影之间的残差；
3. 按照射线在各像素中的路径权重将残差分配回图像；
4. 用 (sum_l a_{il}^2) 归一化更新幅度。

## 6. 一轮 ART 迭代

对所有射线依次更新：

\[
i=1,2,\ldots,M.
\]

若第 (i) 条射线更新后的估计记为 (mathbf{x}^{(k,i)})，则

\[
\mathbf{x}^{(k,i)}
=\mathbf{x}^{(k,i-1)}
+\lambda
\frac{p_i-\mathbf{a}_i^T\mathbf{x}^{(k,i-1)}}{\|\mathbf{a}_i\|_2^2}\mathbf{a}_i.
\]

处理完所有射线后：

\[
\mathbf{x}^{(k+1)}=\mathbf{x}^{(k,M)}.
\]

这样的一次完整扫描称为一个 sweep。射线可以按角度顺序、随机顺序或交替顺序处理；顺序会影响收敛速度和噪声表现。

## 7. 噪声、约束与正则化

实际投影通常满足

\[
\mathbf{p}=\mathbf{A}\mathbf{x}+\boldsymbol{\varepsilon},
\]

其中 (oldsymbol{\varepsilon}) 表示测量噪声。因此不同射线方程可能没有严格的公共交点。常见处理方式如下。

### 7.1 欠松弛

使用 (0<\lambda<1)，每次只修正部分残差，降低噪声放大。

### 7.2 提前停止

迭代早期通常主要恢复物体结构，迭代过多则可能开始拟合噪声。因此迭代次数本身可以作为一种隐式正则化参数。

### 7.3 非负约束

X 射线衰减系数通常满足 (x_j\geq0)。可在每次更新后执行

\[
x_j\leftarrow\max(x_j,0).
\]

### 7.4 显式正则化

更一般地，可求解

\[
\min_{\mathbf{x}\geq0}
\frac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{p}\|_2^2
+\beta R(\mathbf{x}),
\]

其中 (R(\mathbf{x})) 可以是二范数平滑项、梯度平滑项或全变分项。实际迭代中通常交替执行数据一致性更新和正则化更新。

## 8. 与整体最小二乘的关系

若直接最小化数据残差：

\[
f(\mathbf{x})=\frac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{p}\|_2^2,
\]

其梯度为

\[
\nabla f(\mathbf{x})
=\mathbf{A}^T(\mathbf{A}\mathbf{x}-\mathbf{p}).
\]

整体梯度下降一次使用全部射线，而 ART 一次只使用一条射线，并按该射线的行范数归一化：

\[
\mathbf{x}^{(k+1)}
=\mathbf{x}^{(k)}
+\lambda
\frac{p_i-\mathbf{a}_i^T\mathbf{x}^{(k)}}{\|\mathbf{a}_i\|_2^2}\mathbf{a}_i.
\]

因此，ART 可以理解为一种逐行、归一化的梯度或投影迭代。

## 9. SART 与 ART 的关系

ART 每次使用一条射线，更新及时但容易受到单条射线噪声影响。SART（Simultaneous ART）则对一组射线进行归一化的同时更新。其典型形式为

\[
x_j^{(k+1)}=x_j^{(k)}+
\lambda
\frac{
\displaystyle\sum_{i\in\mathcal{G}}
\frac{a_{ij}}{\sum_l a_{il}}
\left(p_i-\sum_l a_{il}x_l^{(k)}\right)
}{
\displaystyle\sum_{i\in\mathcal{G}}a_{ij}}
.
\]

其中 (mathcal{G}) 是当前投影子集。相较 ART，SART 通常更新更平滑、抗噪性更好，也更适合实际 CT。

## 10. 算法伪代码

```text
初始化 x = 0，或使用 FBP 结果作为初值

重复若干轮：
    对每一条射线 i：
        p_hat = a_i^T x
        r = p_i - p_hat
        x = x + lambda * r * a_i / ||a_i||^2
        可选：x = max(x, 0)
```

实现中的主要计算是前向投影和残差反投影。大型 CT 系统一般采用专门的投影算子和 GPU 加速，而不是显式构造完整系统矩阵。

## 11. ART、SART 与 FBP 对比

| 特性 | FBP | ART | SART |
|---|---|---|---|
| 方法类型 | 解析重建 | 逐射线迭代 | 分组迭代 |
| 计算速度 | 快 | 较慢 | 较慢 |
| 稀疏角度适应性 | 较弱 | 较强 | 较强 |
| 噪声控制 | 依赖滤波器 | 依赖松弛和停止准则 | 通常更平滑 |
| 加入先验约束 | 不方便 | 方便 | 方便 |
| 系统几何建模 | 相对固定 | 灵活 | 灵活 |

## 12. 带正则化的 ART 推导

### 12.1 MAP 与变分模型

考虑带协方差 Σ 的高斯投影噪声：

\[
\mathbf{p}=\mathbf{A}\mathbf{x}+\boldsymbol{\varepsilon},
\qquad \mathbf{W}=\boldsymbol{\Sigma}^{-1}.
\]

若图像先验写成

\[
p(\mathbf{x})\propto\exp[-\beta R(\mathbf{x})],
\]

则最大后验估计等价于求解

\[
\boxed{
\min_{\mathbf{x}\in C}
F(\mathbf{x})
=\frac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{p}\|_{\mathbf{W}}^2
+\beta R(\mathbf{x})
},
\]

其中 \(C\) 可表示非负约束，\(\|\mathbf{v}\|_{\mathbf{W}}^2=\mathbf{v}^T\mathbf{W}\mathbf{v}\)。第一项要求数据一致，第二项引入平滑、稀疏或边缘保持先验。

### 12.2 可微正则项

数据项梯度为

\[
\nabla D(\mathbf{x})
=\mathbf{A}^T\mathbf{W}(\mathbf{A}\mathbf{x}-\mathbf{p}).
\]

若 \(R\) 可微，一阶最优条件为

\[
\mathbf{A}^T\mathbf{W}(\mathbf{A}\mathbf{x}^*-\mathbf{p})
+\beta\nabla R(\mathbf{x}^*)=0.
\]

相应的投影梯度迭代为

\[
\boxed{
\mathbf{x}^{(k+1)}
=P_C\left[
\mathbf{x}^{(k)}-\tau_k\left(
\mathbf{A}^T\mathbf{W}(\mathbf{A}\mathbf{x}^{(k)}-\mathbf{p})
+\beta\nabla R(\mathbf{x}^{(k)})\right)
\right].
}
\]

在 ART 中，可用一轮逐射线更新近似数据项步骤：

\[
\mathbf{y}^{(k)}=\operatorname{ARTSweep}(\mathbf{x}^{(k)}),
\qquad
\mathbf{x}^{(k+1)}=P_C[\mathbf{y}^{(k)}-\tau_k\beta\nabla R(\mathbf{y}^{(k)})].
\]

这就是“数据一致性更新 + 正则化更新”的交替形式。

### 12.3 Tikhonov 正则化

取

\[
R(\mathbf{x})=\frac{1}{2}\|\mathbf{L}\mathbf{x}\|_2^2,
\]

其中 \(\mathbf{L}\) 可以是单位矩阵、梯度算子或 Laplace 算子，则

\[
\nabla R(\mathbf{x})=\mathbf{L}^T\mathbf{L}\mathbf{x}.
\]

无约束时的一阶条件为

\[
\boxed{
(\mathbf{A}^T\mathbf{W}\mathbf{A}+\beta\mathbf{L}^T\mathbf{L})\mathbf{x}
=\mathbf{A}^T\mathbf{W}\mathbf{p}
}.
\]

它也等价于增广最小二乘问题：

\[
\min_{\mathbf{x}}
\frac{1}{2}
\left\|
\begin{bmatrix}\mathbf{W}^{1/2}\mathbf{A}\\\sqrt{\beta}\mathbf{L}\end{bmatrix}\mathbf{x}
-\begin{bmatrix}\mathbf{W}^{1/2}\mathbf{p}\\\mathbf{0}\end{bmatrix}
\right\|_2^2.
\]

因此，可以对增广系统使用 Kaczmarz/ART：真实投影方程推动图像匹配测量，正则化方程推动 \(\mathbf{L}\mathbf{x}\) 变小。这是“正则化 ART”最直接的代数解释。

### 12.4 TV 正则化

#### 12.4.1 目标函数

对二维图像，将水平和垂直一阶差分分别记为

\[
\mathbf{g}_x=D_x\mathbf{x},\qquad
\mathbf{g}_y=D_y\mathbf{x}.
\]

各向同性全变分（Total Variation，TV）为

\[
R_{\mathrm{TV}}(\mathbf{x})
=\sum_j\sqrt{(\mathbf{g}_x)_j^2+(\mathbf{g}_y)_j^2}.
\]

带 TV 正则的 CT 重建目标为

\[
\boxed{
\min_{\mathbf{x}\geq0}
\frac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{p}\|_{\mathbf{W}}^2
+\beta R_{\mathrm{TV}}(\mathbf{x})
}.
\]

数据项要求重建图像与投影一致，TV 项惩罚图像中的总梯度，使噪声和小幅振荡受到抑制，同时尽量保留较大的结构边缘。

#### 12.4.2 平滑 TV 的梯度推导

原始 TV 在 \(\mathbf{g}_x=\mathbf{g}_y=0\) 处不可微。引入 \(\epsilon>0\) 后，得到平滑 TV：

\[
R_{\mathrm{TV},\epsilon}(\mathbf{x})
=\sum_j q_j,
\qquad
q_j=\sqrt{(\mathbf{g}_x)_j^2+(\mathbf{g}_y)_j^2+\epsilon^2}.
\]

对任意扰动 \(\delta\mathbf{x}\)，有

\[
\delta R_{\mathrm{TV},\epsilon}
=\sum_j\frac{(\mathbf{g}_x)_j(D_x\delta\mathbf{x})_j
+(\mathbf{g}_y)_j(D_y\delta\mathbf{x})_j}{q_j}.
\]

写成内积形式：

\[
\delta R_{\mathrm{TV},\epsilon}
=\left\langle\frac{\mathbf{g}_x}{\mathbf{q}},D_x\delta\mathbf{x}\right\rangle
+\left\langle\frac{\mathbf{g}_y}{\mathbf{q}},D_y\delta\mathbf{x}\right\rangle.
\]

利用差分算子的伴随关系

\[
\langle\mathbf{u},D_x\delta\mathbf{x}\rangle
=\langle D_x^T\mathbf{u},\delta\mathbf{x}\rangle,
\]

得到

\[
\boxed{
\nabla R_{\mathrm{TV},\epsilon}(\mathbf{x})
=D_x^T\left(\frac{\mathbf{g}_x}{\mathbf{q}}\right)
+D_y^T\left(\frac{\mathbf{g}_y}{\mathbf{q}}\right)
}.
\]

连续域中，该式对应

\[
\nabla R_{\mathrm{TV},\epsilon}(x)
=-\operatorname{div}\left(
\frac{\nabla x}{\sqrt{|\nabla x|^2+\epsilon^2}}
\right).
\]

这里的负号来自 \(D_x^T,D_y^T\) 对应离散负散度算子。不同边界条件（零边界、周期边界、Neumann 边界）会导致差分伴随矩阵的具体形式不同。

#### 12.4.3 ART-TV 交替迭代

先对当前图像执行一轮 ART 数据一致性更新：

\[
\mathbf{y}^{(k)}=\operatorname{ARTSweep}(\mathbf{x}^{(k)}).
\]

再沿 TV 梯度下降，并投影到非负集合：

\[
\boxed{
\mathbf{x}^{(k+1)}
=P_{+}\left[
\mathbf{y}^{(k)}
-\tau_k\beta
\nabla R_{\mathrm{TV},\epsilon}(\mathbf{y}^{(k)})
\right]
},
\]

即

\[
\mathbf{x}^{(k+1)}
=P_{+}\left[
\mathbf{y}^{(k)}
-\tau_k\beta\left(
D_x^T\frac{D_x\mathbf{y}^{(k)}}{\mathbf{q}^{(k)}}
+D_y^T\frac{D_y\mathbf{y}^{(k)}}{\mathbf{q}^{(k)}}
\right)
\right],
\]

其中

\[
q_j^{(k)}
=\sqrt{(D_x\mathbf{y}^{(k)})_j^2+(D_y\mathbf{y}^{(k)})_j^2+\epsilon^2},
\qquad
P_{+}(z)=\max(z,0).
\]

该算法的含义是：ART 先消除投影残差，TV 步骤再抑制图像中的噪声梯度，非负投影保证衰减系数具有物理意义。

#### 12.4.4 原始 TV 的近端形式

当 \(\epsilon\to0\) 时，平滑近似趋近原始 TV，但梯度不再处处存在。此时使用近端算子：

\[
\operatorname{prox}_{\gamma R}(\mathbf{v})
=\arg\min_{\mathbf{x}}
\left\{\frac{1}{2}\|\mathbf{x}-\mathbf{v}\|_2^2+\gamma R(\mathbf{x})\right\},
\]

ART-TV 的近端更新为

\[
\boxed{
\mathbf{x}^{(k+1)}
=P_{+}\left[
\operatorname{prox}_{\tau_k\beta R_{\mathrm{TV}}}
\left(\mathbf{y}^{(k)}\right)
\right],
\qquad
\mathbf{y}^{(k)}=\operatorname{ARTSweep}(\mathbf{x}^{(k)}).
}
\]

TV 近端子问题可通过 Chambolle 投影、Chambolle--Pock 原始-对偶法或 ADMM 求解。以原始-对偶形式为例，利用

\[
R_{\mathrm{TV}}(\mathbf{x})=\|\mathbf{D}\mathbf{x}\|_{2,1},
\qquad
\mathbf{D}=\begin{bmatrix}D_x\\D_y\end{bmatrix},
\]

其对偶表示为

\[
R_{\mathrm{TV}}(\mathbf{x})
=\max_{\|\mathbf{q}_j\|_2\leq1}
\langle\mathbf{D}\mathbf{x},\mathbf{q}\rangle.
\]

因此 TV 近端计算可以转化为交替更新图像变量和满足
\(\|\mathbf{q}_j\|_2\leq1\) 的对偶变量。

#### 12.4.5 ART-TV 伪代码

```text
初始化 x = 0 或 FBP 结果
重复 K 轮：
    y = x

    # ART 数据一致性步骤
    对每条射线 i：
        r = p_i - a_i^T y
        y = y + lambda * r * a_i / ||a_i||^2

    # TV 正则化步骤（二选一）
    g_x = D_x y
    g_y = D_y y
    q = sqrt(g_x^2 + g_y^2 + epsilon^2)
    grad_tv = D_x^T(g_x / q) + D_y^T(g_y / q)
    x = y - tau * beta * grad_tv

    # 物理约束
    x = max(x, 0)
```

如果使用原始不可微 TV，则将 TV 梯度步骤替换为 TV 近端求解器：

```text
    x = TV_prox(y, tau * beta)
    x = max(x, 0)
```

其中 \(\lambda\) 控制 ART 数据更新幅度，\(\tau\) 控制 TV 梯度步长，\(\beta\) 控制 TV 强度，\(\epsilon\) 控制平滑程度。通常 \(\epsilon\) 取较小正数；\(\beta\) 过大将造成过度平滑，过小则抑噪不足。

### 12.5 正则化 ART 伪代码

```text
初始化 x = 0 或 FBP 结果
重复 K 轮：
    y = x
    对每条射线 i：
        r = p_i - a_i^T y
        y = y + lambda * r * a_i / ||a_i||^2

    若 R 可微：
        x = y - tau * beta * grad_R(y)
    否则：
        x = prox_(tau * beta * R)(y)

    x = max(x, 0)
```

\(\beta\) 越大，先验约束越强、图像通常越平滑；\(\beta\) 越小，数据拟合更强但噪声风险更高。实际选择应结合投影噪声、角度数量、空间分辨率和验证数据确定。

## 13. 小结

ART 的推导链条可以概括为：

\[
\text{CT 线积分}
\Rightarrow \mathbf{A}\mathbf{x}=\mathbf{p}
\Rightarrow \mathbf{a}_i^T\mathbf{x}=p_i
\Rightarrow \text{投影到单条射线对应的超平面}
\Rightarrow
\boxed{
\mathbf{x}^{(k+1)}
=\mathbf{x}^{(k)}
+\lambda
\frac{p_i-\mathbf{a}_i^T\mathbf{x}^{(k)}}{\|\mathbf{a}_i\|_2^2}\mathbf{a}_i
}.
\]

其本质是：利用当前射线的投影误差，沿该射线对应的系统矩阵行向量，将误差反投影回图像空间。通过松弛、非负约束、提前停止和显式正则化，可以在不完整或含噪数据下获得更稳定的重建结果。带正则化时，完整目标函数是“加权数据一致性项 + 先验惩罚项”，而 ART 负责逐射线数据更新，梯度或近端步骤负责施加图像先验。
