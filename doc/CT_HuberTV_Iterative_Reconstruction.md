# CT 代数迭代重建中的 Huber-TV：SART 同步正则化、近端法与预条件优化

> 修订版（2026-10-12）。**核心选择**：对已有的 CUDA Joseph/代数重建系统，优先实现 **SART 风格归一化的 Huber-TV 同步更新**，并保留 **SART + Huber-TV Prox** 作为对照。普通梯度下降只用于推导和校验，不建议作为大规模 CBCT 的首选实际求解器。

## 1. 问题、符号与要区分的算法

离散 CT 模型为 $b=Ax+\varepsilon$：$x\in\mathbb R^{N_v}$ 是体素线性衰减系数，$b\in\mathbb R^{N_p}$ 是对数投影，$A$ 是射线路径积分算子，$A^T$ 必须是其**严格离散伴随**。$D$ 是三维空间梯度，$D^T$ 是它的伴随。$W$ 是统计残差权重，$R,C$ 则表示 **SART 几何归一化矩阵**，二者不能混为一谈。$\alpha$ 为松弛/步长，$\lambda$ 为正则化系数，$\delta>0$ 为 Huber 转折阈值。

本文对照四种实现：

1. **同步更新（重点）**：SART 几何归一化的投影校正与 Huber-TV 梯度，在**同一状态 $x^k$** 上形成**一个更新方向**。
2. **SART + Huber-TV Prox**：先更新代数重建，再求解独立的 Huber-TV 近端子问题。
3. **统计加权预条件梯度**：将实际噪声权重 $W$ 与体素预条件 $M$ 分离，进一步推广同步法。
4. **Newton-CG / Hessian 对角预条件**：可选高级优化，不等于将 SART 与牛顿法机械串联。

普通 Landweber/梯度下降仅作为统一数学基线，因为 $A^TWA$ 在 CT 中往往病态、尺度不均衡，固定步长的一阶法可能收敛很慢。

## 2. Huber-TV 正则项及其梯度

选用各向同性三维 Huber-TV：

$$
R_{\mathrm{HTV}}(x)=\sum_{j=1}^{N_v}\rho_\delta\big(\|(Dx)_j\|_2\big),\qquad
\rho_\delta(t)=\begin{cases}
\dfrac{t^2}{2\delta},&0\le t\le\delta,\\
t-\dfrac\delta2,&t>\delta.
\end{cases}
$$

三维前向差分（实际必须处理边界）为

$$
(Dx)_j=\left(\frac{x_{j+e_x}-x_j}{h_x},\frac{x_{j+e_y}-x_j}{h_y},\frac{x_{j+e_z}-x_j}{h_z}\right).
$$

设 $q_j(x)=(Dx)_j/\max\{\delta,\|(Dx)_j\|_2\}$，那么

$$\boxed{g_R(x)=\nabla R_{\mathrm{HTV}}(x)=D^Tq(x).}$$

**实现注意**：$D^T$ 必须是实际离散差分的精确转置；若使用负散度的记号，$D^T=-\operatorname{div}$。固定 $\delta>0$ 时该项可微，但在 Huber 转折点不一定有经典二阶导数。对体素间距 $h_x,h_y,h_z$ 的缩放必须一致。若 $x$ 是 $\mu$（mm$^{-1}$），梯度和 $\delta$ 单位为 mm$^{-2}$；若是 HU，$\delta$ 单位为 HU/mm。以 HU 为变量时不能简单施加非负约束。

## 3. SART 基本更新及其梯度解释

对于系统矩阵 $A\ge0$，定义（仅对正对角元取逆）

$$
R=\operatorname{diag}(A\mathbf1),\qquad C=\operatorname{diag}(A^T\mathbf1).
$$

全投影的 SART 风格归一化更新为

$$
\boxed{x^{k+1}=x^k+\alpha C^{-1}A^TR^{-1}(b-Ax^k).}
$$

它等价于对几何加权目标

$$F_R(x)=\frac12(Ax-b)^TR^{-1}(Ax-b)$$

作对角预条件梯度步：$x^{k+1}=x^k-\alpha C^{-1}\nabla F_R(x^k)$。**这是全投影同步形式**；标准 SART 常按单个角度/子集使用 $A_m,R_m,C_m$，每子集更新体积，并不能简单把它与全批量梯度步画等号。

$R^{-1}$ 是**几何行和归一化**，并不天然等于泊松噪声的逆方差。若需要 PWLS，应另行定义统计 $W$。

## 4. 核心方法 A：SART + Huber-TV 同步更新

### 4.1 对应的完整目标

使用几何加权数据项：

$$
\boxed{F(x)=\frac12(Ax-b)^TR^{-1}(Ax-b)+\lambda R_{\mathrm{HTV}}(x).}
$$

梯度为 $\nabla F(x)=A^TR^{-1}(Ax-b)+\lambda g_R(x)$。将 $C^{-1}$ 同时作用于数据项及正则项，得到

$$
\boxed{x^{k+1}=\Pi_{\mathcal C}\!\left[x^k+\alpha_k C^{-1}\!\left(A^TR^{-1}(b-Ax^k)-\lambda g_R(x^k)\right)\right].}
\tag{A1}
$$

$\Pi_{\mathcal C}$ 是可选物理约束：以 $\mu$ 为变量可取 $\mathcal C=\{x\ge0\}$；以 HU 为变量则通常不应简单投影到非负数。**关键是两个方向都在同一 $x^k$ 上计算，合成后只更新一次**，并不是先 SART 再 TV 去噪。

逐体素形式为：

$$
\boxed{x_j^{k+1}=\Pi_{\mathcal C}\left[x_j^k+\frac{\alpha_k}{c_j}\left(\sum_i a_{ij}\frac{b_i-(Ax^k)_i}{r_i}-\lambda\,[D^Tq(x^k)]_j\right)\right],}
\tag{A2}
$$

其中 $r_i=\sum_l a_{il}$，$c_j=\sum_i a_{ij}$。对无射线覆盖的体素采用安全逆或冻结处理；必要时给预条件矩阵加下限以限制异常更新。

### 4.2 完整角度同步与分块（推荐作为基准）

$$
\boxed{x^{k+1}=\Pi_{\mathcal C}\left[x^k+\alpha_k C^{-1}\left(\sum_{m=1}^M A_m^TR_m^{-1}(b_m-A_mx^k)-\lambda g_R(x^k)\right)\right].}
\tag{A3}
$$

每个 chunk 可以将修正累加到 `d_dataCorr`，但在所有 chunk 完成之前，**不得改变 $x^k$**。只需保留当前 chunk 的 AVU 投影，不必存储全部角度中间量。Huber 梯度每外轮只计算一次。

其特点：有明确的全局几何加权目标；便于监控目标函数；内存可通过 chunk 控制。它更精确的名称是**全批量 SART 归一化的预条件梯度法**，而不是标准逐角度 SART。

### 4.3 真正逐角度 / 子集 SART 的同步正则化

对第 $m$ 个子集，给定当前迭代状态 $x^{k,m}$，可定义：

$$
\boxed{x^{k,m+1}=\Pi_{\mathcal C}\left[x^{k,m}+\alpha_{k,m}C_m^{-1}\left(A_m^TR_m^{-1}(b_m-A_mx^{k,m})-\lambda_m g_R(x^{k,m})\right)\right].}
\tag{A4}
$$

遍历全部 $M$ 个子集之后令 $x^{k+1,0}=x^{k,M}$。若期望子集正则**梯度总和**对应一个全局 $\lambda g_R$，在无预条件差异的形式中可取 $\lambda_m=\lambda/M$；但在 (A4) 中因 $C_m^{-1}$ 不同、更新状态变化，**这一分配不保证与 (A3) 等价，也不是一般收敛证明**。

尤其注意：对很少射线覆盖的子集，$C_m$ 可能含大量零或很小对角元，正则梯度除以 $C_m$ 会产生不稳定的局部放大。可采用有下限的 $C_m$、较大的子集、独立正则步长，或优先使用全批量 (A3)。固定松弛参数的 Ordered-Subset 法接近最优点时可能振荡，宜研究递减松弛、子集随机化或周期性全梯度校正。

### 4.4 CUDA 伪代码：全批量同步更新

```cpp
for (int epoch = 0; epoch < maxEpochs; ++epoch) {
    clear(d_dataCorrection);

    // 所有 chunk 必须使用同一个 d_x（x^k）
    for (auto chunk : projectionChunks) {
        forwardJoseph(d_x, chunk, d_Ax);
        normalizedResidual(d_b, d_Ax, d_rowSum, chunk, d_residual);
        adjointBackProjectAccumulate(d_residual, chunk, d_dataCorrection);
    }

    gradient3D(d_x, d_grad, spacing);
    huberFlux(d_grad, d_flux, delta);
    adjointGradient3D(d_flux, d_regGrad, spacing);

    // 一次更新：x += alpha * C^-1 * (dataCorrection - lambda*regGrad)
    fusedUpdate(d_x, d_dataCorrection, d_regGrad,
                d_colSum, alpha, lambda, constraint);
}
```

`normalizedResidual` 在行和为零的位置应产生 0；`fusedUpdate` 在列和为零的位置必须使用安全策略。若要严格匹配目标函数，在一次迭代内不可将某些 chunk 使用更新后体积、另一些使用更新前体积。

### 4.5 步长与收敛监控

令 $H_0=A^TR^{-1}A$，在忽略约束时有梯度 Lipschitz 上界

$$L_F\le\|R^{-1/2}A\|_2^2+\frac{\lambda}{\delta}\|D\|_2^2.$$

由于 (A1) 使用非平凡的正对角预条件器 $M=C^{-1}$，不能直接沿用普通梯度下降的 $\alpha\le1/L_F$ 而不考虑 $M$。一个可用于无约束或适当兼容约束的保守尺度来自 $\|M^{1/2}H_0M^{1/2}\|+\lambda\|M^{1/2}D^T\|^2/\delta$；实际实现建议使用回溯线搜索或单调性监测。对约束情况，欧氏逐点投影配合对角正预条件器仍可实现，但要明确其对应的度量与固定点条件。

跟踪 $F(x^k)$、$\|Ax^k-b\|$、$\|x^{k+1}-x^k\|$、图像 HU 指标和运行时间，而不能只看迭代次数。

## 5. 核心方法 B：SART + Huber-TV Prox 交替迭代

先执行一次子集更新或一整个 SART sweep：

$$z^k=\operatorname{SARTSweep}(x^k,b).$$

然后独立求解 Huber-TV 近端问题：

$$
\boxed{x^{k+1}=\operatorname*{arg\,min}_{u\in\mathcal C}\left[\frac12\|u-z^k\|_2^2+\gamma R_{\mathrm{HTV}}(u)\right].}
\tag{B1}
$$

近端子问题可以使用若干次内层梯度迭代（也可用 PDHG/FISTA）：

$$
\begin{aligned}
u^0&=z^k,\\
q_i^j&=\frac{(Du^j)_i}{\max(\delta,\|(Du^j)_i\|_2)},\\
u^{j+1}&=\Pi_{\mathcal C}\left[u^j-\eta\left(u^j-z^k+\gamma D^Tq^j\right)\right],\\
x^{k+1}&=u^J.
\end{aligned}
\tag{B2}
$$

内层 $\eta\le 1/(1+\gamma\|D\|^2/\delta)$ 是方便的保守选择。**增加 $J$ 只提高近端子问题的求解精度，不等于增大 $\gamma$。**

```cpp
for (int epoch = 0; epoch < maxEpochs; ++epoch) {
    sartSweepInPlace(d_x, projections, alpha);
    copy(d_z, d_x);
    for (int j = 0; j < proxIterations; ++j) {
        gradient3D(d_x, d_grad, spacing);
        huberFlux(d_grad, d_flux, delta);
        adjointGradient3D(d_flux, d_regGrad, spacing);
        proxUpdate(d_x, d_z, d_regGrad, gamma, eta, constraint);
    }
}
```

优点：适合复用独立的 `HuberTVProx` 模块（也可用于 FDK 图像后处理）。缺点：近端内循环增加计算，**SART 与 Prox 的简单交替并不自动保证收敛到第 4 节 $F(x)$ 的最优解**，更不能因为用了近端就宣称整个 SART 算法具有标准 proximal gradient 收敛保证。

## 6. 将噪声统计与 SART 归一化分开：预条件 PWLS-Huber-TV

用于医疗 CT 的统计目标可写为：

$$
\boxed{F_W(x)=\frac12(Ax-b)^TW(Ax-b)+\lambda R_{\mathrm{HTV}}(x).}
$$

其中 $W=\operatorname{diag}(1/\sigma_i^2)$ 来自对数投影噪声模型；理想高计数 Poisson 情况下 $W_{ii}\approx \bar N_i$。CBCT 能量积分探测器、散射校正及电子噪声会使这一简化失效，需要实测方差模型。

一般**预条件同步更新**为：

$$
\boxed{x^{k+1}=\Pi_{\mathcal C}\left[x^k-\alpha_k M\left(A^TW(Ax^k-b)+\lambda g_R(x^k)\right)\right],}
\tag{C1}
$$

$M\succ0$ 可以取安全缩放的 SART 列和逆 $C^{-1}$、其他对角矩阵，或更高阶预条件器。**使用统计 $W$ 后，若仍使用 SART 的 $C^{-1}$，它应被明确视为预条件器，而不是说公式仍是标准 SART。**

## 7. Newton、SART 与 Hessian 对角预条件的关系

对数据项 $\tfrac12\|W^{1/2}(Ax-b)\|^2$：

$$g_{\mathrm{data}}=A^TW(Ax-b),\qquad H_{\mathrm{data}}=A^TWA.$$

- 普通梯度下降：$M=I$；
- SART 风格预条件梯度：$M=C^{-1}$，并在几何版本中令 $W=R^{-1}$；
- Hessian 对角（Jacobi）预条件：$M=[\operatorname{diag}(A^TWA)+\epsilon I]^{-1}$；
- Newton-CG：近似解 $H_k d_k=-g_k$，再线搜索更新 $x^{k+1}=x^k+\alpha_k d_k$。

$\operatorname{diag}(A^TWA)_j=\sum_i w_i a_{ij}^2$，**不同于** $A^TW\mathbf1$。完整 Hessian 不需要显式存储，其数据部分的矩阵-向量积是 $H_{\mathrm{data}}v=A^TW(Av)$；加入 Huber-TV 后还需要相应的正则 Hessian-向量积、平滑/广义 Hessian 或 IRLS 近似。牛顿法与 SART **不冲突**，但也不是“是否做行列归一化”这样简单的区别。

Newton-CG 可以用 $C^{-1}$ 作为候选预条件器，但不能保证它最优；通常应与 Hessian 对角、阻尼预条件器比较。完整牛顿内迭代可能需要多次 FP/BP，务必比较到达相同精度的**总耗时**而不是外层轮数。

## 8. 三种主要方案对照与选择

| 项目 | A：同步 SART-Huber | B：SART + Prox | C：统计 PWLS 预条件法 |
|---|---|---|---|
| 数据方向 | $A^TR^{-1}(b-Ax)$ | SART 子集校正 | $A^TW(b-Ax)$ |
| 正则方式 | $-\lambda g_R(x^k)$ 同步合成 | 独立近端子问题 | $-\lambda g_R(x^k)$ 同步合成 |
| 行列归一化 | $R^{-1},C^{-1}$ | SART 内部归一化 | 可选 $M=C^{-1}$ 或其他 |
| 统一目标 | 全批量时明确 | 一般不等价于统一 PWLS 最优解 | 明确的统计加权目标 |
| Huber 梯度调用 | 全批量每外轮一次 | 每个近端内迭代一次 | 每外轮一次 |
| 开发优先级 | **第一优先** | **第二优先** | 噪声模型完善后扩展 |

普通无预条件梯度下降 $x^{k+1}=x^k-\tau\nabla F(x^k)$ 仍是正确的数学基线，可用于验证目标函数梯度，但对大规模医学 CBCT，不建议将其未经预条件和加速直接作为主力算法。

## 9. CUDA 工程建议与验证

建议模块接口：

```cpp
struct HuberTVParams {
    float lambda;
    float delta;
    float3 voxelSpacing;
};

enum class ReconOptimizer {
    SART_Huber_Simultaneous,
    SART_Huber_Proximal,
    PWLS_Preconditioned,
    NewtonCG
};

// 核心模块：FP Joseph、严格伴随 BP、行和/列和归一化、
// 3D Gradient、HuberFlux、GradientAdjoint、单次融合更新、
// HuberTVProx、目标函数与误差监控。
```

### 必需的测试

1. **FP/BP 伴随**：随机数据验证 $\langle Ax,y\rangle\approx\langle x,A^Ty\rangle$；普通体素驱动 BP 不一定是 Joseph FP 的离散转置。
2. **差分伴随**：验证 $\langle Dx,q\rangle\approx\langle x,D^Tq\rangle$，覆盖三维边界。
3. **有限差分梯度**：验证 $\big(F(x+hv)-F(x-hv)\big)/(2h)\approx\langle\nabla F(x),v\rangle$。
4. **归一化**：用小规模显式矩阵与 CUDA 的 $R,C$、更新方向逐元素对照；检查零覆盖体素。
5. **同步/子集区分**：全批量 chunk 内不改变 $x^k$；逐子集模式每次更新后立刻使用新 $x$，分别测试。
6. **收敛与质量**：跟踪目标函数、投影残差、HU 偏差、均匀区噪声、骨边缘及软组织低对比度；SART-Prox 不应仅凭“图像更平滑”判定收敛。
7. **单位与约束**：SART 的 $A$ 与重建物理坐标统一为 mm；HU 值可以为负。正则化参数随体素间距和变量单位变化。

### 推荐实验顺序

- 第一阶段：小系统矩阵验证 SART 行列归一化、FP/BP 伴随和 Huber 梯度，完成 **A3 全批量同步**。
- 第二阶段：加入 A4 子集模式，重点测试固定/递减松弛、子集数量与预条件下限。
- 第三阶段：完成 B 的 HuberTVProx，与 A 在相同模体、相同测量数据下比较。
- 第四阶段：建立真实探测器 `NoiseVarianceMap`，引入 C1 统计 $W$；最后再比较 Newton-CG/Jacobi 预条件。

**最终建议**：以 (A3) 作为第一个可验证的 CUDA 工程实现，以 (A4) 作为真正 SART 的加速试验，以 (B1) 作为交替近端方法对照。普通梯度下降只保留在推导和测试中，不再作为本文主推荐的实施路线。
