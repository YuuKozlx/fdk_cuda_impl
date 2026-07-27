# TIGRE 风格梯度重建算法说明

本文说明 YKCBCT 对 TIGRE 梯度类算法的行为复现、公共接口和参数语义。
实现行为参考 CERN TIGRE 的 MATLAB/Python 版本；工程复用自身投影算子、
显存管理和 CUDA 调度，不直接依赖 TIGRE 运行时。

## 1. 支持范围

统一入口提供以下算法：

| 配置枚举 | TIGRE 名称 | 数据更新 | 正则化状态机 |
|---|---|---|---|
| `Sart` | SART | 每个子集 1 个视角 | 无 |
| `OsSart` | OS-SART | 每个子集 `block_size` 个视角 | 无 |
| `Sirt` | SIRT | 一个子集包含全部视角 | 无 |
| `AsdPocs` | ASD-POCS | SART | TV + POCS 调度 |
| `OsAsdPocs` | OS-ASD-POCS | OS-SART | TV + POCS 调度 |
| `BAsdPocsBeta` | B-ASD-POCS-β | SART + Bregman 投影 | TV + POCS 调度 |
| `Pcsd` / `OsPcsd` | PCSD / OS-PCSD | SART / OS-SART | PCSD 步长调度 |
| `AwPcsd` / `OsAwPcsd` | AwPCSD / OS-AwPCSD | SART / OS-SART | 自适应加权 TV |
| `AwAsdPocs` / `OsAwAsdPocs` | Aw-ASD-POCS / OS-Aw-ASD-POCS | SART / OS-SART | 自适应加权 TV + POCS |

`TigreGradientReconstructor` 根据 `SCBCTParams` 构造标准圆轨迹；
`TigreGradientReconstructorEx` 接受逐视角 `SConeProjGeomVec`。`Ex` 只代表
geometry 输入方式，算法和调参能力完全相同。

## 2. 基本用法

```cpp
#include "Iter/YkTigreGradientReconstructor.hpp"

YK::Iter::TigreGradientReconstructor::Config config{};
config.algorithm = YK::Iter::ETigreGradientAlgorithm::OsAsdPocs;
config.iterations = 30;
config.block_size = 20;       // 每个子集的视角数，不是子集数量
config.lambda = 1.0f;         // TIGRE 中的 lambda / beta
config.lambda_reduction = 0.99f;
config.tv_iterations = 20;
config.alpha = 0.002f;
config.alpha_reduction = 0.95f;
config.maximum_update_ratio = 0.95f;
config.max_l2_error = -1.f;   // 自动估计
config.non_negative = true;

YK::Iter::TigreGradientReconstructor reconstructor;
reconstructor.prepare(params, config, stream, device_id);
reconstructor.reconstruct(d_projection, d_volume);
const auto& statistics = reconstructor.statistics();
```

## 3. 参数与 TIGRE 的对应关系

| C++ 字段 | TIGRE 参数/变量 | 说明 |
|---|---|---|
| `iterations` | `niter` | 最大外循环数 |
| `block_size` | `blocksize` | 一个子集包含的视角数 |
| `lambda` | `lambda` / `beta` | 数据一致性更新初始步长 |
| `lambda_reduction` | `lambda_red` / `beta_red` | 每个外循环后的乘法衰减 |
| `tv_iterations` | `tviter` / `ng` | 每轮 TV 内迭代次数 |
| `alpha` | `alpha` | 首轮 `dtvg = alpha * dp` |
| `alpha_reduction` | `alpha_red` | TV 更新过大时衰减 `dtvg` |
| `maximum_update_ratio` | `ratio` / `rmax` | 判断 `dg > rmax * dp` |
| `max_l2_error` | `maxl2err` / `epsilon` | 投影一致性停止阈值 |
| `minimum_beta` | 固定 `0.005` | beta 下限；本实现允许显式配置 |
| `adaptive_delta` | `delta` | AwTV 的边缘权重参数 |

必须特别注意 `max_l2_error` 的历史语义：ASD-POCS 系列比较
`dd > epsilon`，而 TIGRE 的 PCSD/AwPCSD 实现判断
`d_p * d_p > epsilon`。本实现有意保留这一差异，以便同一组 TIGRE 参数得到
可对照的调度行为。

## 4. 迭代更新公式

### 4.1 符号

设完整投影方程为：

```math
A x = b
```

其中：

- $x\in\mathbb{R}^{N_xN_yN_z}$ 为待重建体积；
- $b\in\mathbb{R}^{N_uN_vN_\theta}$ 为测量投影；
- $A$、$A^T$ 分别表示当前配置的 FP 和 BP；
- $S_s$ 是第 $s$ 个连续视角子集；
- $A_s$、$b_s$ 是限制到 $S_s$ 后的投影算子和投影数据；
- $k$ 表示完整外循环编号，$s$ 表示该外循环内的子集编号；
- $\beta_k$ 是数据更新松弛因子，即配置中的 `lambda`；
- $P_+(x)=\max(x,0)$ 表示非负投影。

若总视角数为 $N_\theta$，`block_size=m`，则 OS 方法的实际子集数为：

```math
N_s=\left\lceil\frac{N_\theta}{m}\right\rceil.
```

最后一个子集可以少于 $m$ 张。SART 固定 $m=1$；SIRT 固定
$m=N_\theta$；OS 方法使用用户配置的 $m$。

### 4.2 TIGRE 近似行权重和列权重

每个子集的数据更新写成：

```math
x^{k,s+1}
=P_+\!\left[
x^{k,s}+\beta_k V_s^{-1}A_s^T W_s
\left(b_s-A_sx^{k,s}\right)
\right].
```

$W_s$ 是投影域行权重，$V_s$ 是图像域列权重。当前
`AlgebraicTigreBackend` 按 TIGRE 近似方式构造它们。

行权重首先用 $2\times2\times2$ 的全 1 粗体积计算射线路径长度：

```math
\widetilde w=A_{\mathrm{coarse}}\mathbf 1,
\qquad
W_{ii}=\begin{cases}
1/\widetilde w_i,&\widetilde w_i>\frac12\min(\Delta x,\Delta y,\Delta z),\\
0,&\text{其他情况}.
\end{cases}
```

粗体积的 XY 物理范围扩大为原来的 $1.1$ 倍。全视角的 $W$ 在初始化阶段
一次性计算，每次更新取当前连续子集对应的切片。

列权重对当前子集的全 1 投影执行反投影，并沿 Z 求均值：

```math
\widetilde V_s(x,y,z)=A_s^T\mathbf 1,
\qquad
V_s(x,y)=\frac1{N_z}\sum_{z=0}^{N_z-1}\widetilde V_s(x,y,z).
```

更新时把二维 $V_s(x,y)$ 沿 Z 广播：

```math
x(x,y,z)\leftarrow x(x,y,z)
+\beta_k\frac{[A_s^TW_s(b_s-A_sx)](x,y,z)}{V_s(x,y)+\varepsilon}.
```

为复现 TIGRE 的列权重范围，权重计算时体素尺寸还会乘以：

```math
q=0.9\frac{\max(L_x,L_y)}{\sqrt{L_x^2+L_y^2}},
\qquad L_x=N_x\Delta x,\quad L_y=N_y\Delta y.
```

### 4.3 SART、OS-SART 和 SIRT

三者共享上一节的数据更新公式，只改变子集大小：

```math
\begin{aligned}
\text{SART:}&\quad |S_s|=1,\quad N_s=N_\theta,\\
\text{OS-SART:}&\quad |S_s|=\texttt{block\_size},\\
\text{SIRT:}&\quad |S_0|=N_\theta,\quad N_s=1.
\end{aligned}
```

一个完整外循环依次执行全部 $N_s$ 个子集：

```math
x^{k+1,0}=x^{k,N_s}.
```

普通松弛模式在完整外循环结束后衰减：

```math
\beta_{k+1}=r_\beta\beta_k,
\qquad r_\beta=\texttt{lambda\_reduction}.
```

因此一次重建的子集更新总数为：

```math
N_{\mathrm{update}}=
N_{\mathrm{completed\ outer}}N_s.
```

### 4.4 Nesterov 更新

Nesterov 只用于 SART、OS-SART 和 SIRT。对每个子集，先用固定数据步长 1
产生候选解：

```math
y^{k,s}=P_+\!\left[
x^{k,s}+V_s^{-1}A_s^TW_s(b_s-A_sx^{k,s})
\right].
```

再与上一候选解组合：

```math
x^{k,s+1}=(1-\gamma_k)y^{k,s}+\gamma_k y^{\mathrm{prev}},
\qquad y^{\mathrm{prev}}\leftarrow y^{k,s}.
```

动量参数按 TIGRE MATLAB 形式更新：

```math
t_0=\frac{1+\sqrt5}{2},\qquad \gamma_0=0,
```

```math
t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
\qquad
\gamma_{k+1}=\frac{1-t_k}{t_{k+1}}.
```

每个完整外循环后计算：

```math
d_k=\lVert Ax^k-b\rVert_2.
```

若 $k>0$ 且 $d_k>d_{k-1}$，算法立即停止并保留当前结果，不回退到上一轮。

### 4.5 ASD-POCS 和 OS-ASD-POCS

每个外循环开始时保存：

```math
x_0=x^k.
```

执行一轮 SART 或 OS-SART 数据更新得到 $x_d$，并定义：

```math
p=x_d-x_0,
\qquad
d_p=\lVert p\rVert_2,
\qquad
d_d=\lVert Ax_d-b\rVert_2.
```

首轮 TV 绝对步长为：

```math
d_{\mathrm{tvg}}=\alpha d_p.
```

从 $x_d$ 出发执行 `tv_iterations` 次归一化 TV 梯度下降，得到 $x_r$。
正则化更新量为：

```math
g=x_r-x_d,
\qquad
d_g=\lVert g\rVert_2.
```

数据更新与正则化更新的方向余弦为：

```math
c=\frac{\langle g,p\rangle}
{\max(d_g\,d_p,10^{-6})}.
```

若 TV 更新相对数据更新过大，而且投影误差仍未达到目标：

```math
d_g>r_{\max}d_p
\quad\land\quad
d_d>\epsilon,
```

则缩小下一轮使用的 TV 步长：

```math
d_{\mathrm{tvg}}\leftarrow
r_\alpha d_{\mathrm{tvg}},
```

其中 $r_{\max}$ 对应 `maximum_update_ratio`，$r_\alpha$ 对应
`alpha_reduction`。数据步长同时按：

```math
\beta_{k+1}=r_\beta\beta_k
```

衰减。ASD-POCS 每个数据子集包含一张投影，OS-ASD-POCS 使用
`block_size` 张投影。

### 4.6 PCSD、OS-PCSD、AwPCSD 和 OS-AwPCSD

PCSD 在数据更新前先计算投影距离：

```math
d_p^{\mathrm{proj}}=\lVert Ax^k-b\rVert_2.
```

为兼容 TIGRE 的公开实现，仅当下式成立时才执行本轮数据更新：

```math
\left(d_p^{\mathrm{proj}}\right)^2>\epsilon.
```

这里是平方距离与 `max_l2_error` 比较，不应改写成
$d_p^{\mathrm{proj}}>\epsilon$。

PCSD 的 TV 步长不是 ASD-POCS 的 $\alpha d_p$。首轮使用：

```math
s_0=1.
```

首轮数据更新后记录基准投影距离：

```math
d_{\mathrm{first}}=\lVert Ax_d^0-b\rVert_2.
```

后续外循环使用数据更新前的投影距离比例：

```math
s_k=\frac{d_{p,k}^{\mathrm{proj}}}
{\max(d_{\mathrm{first}},10^{-6})},\qquad k>0.
```

然后用 $s_k$ 作为 TV/AwTV 每次内迭代的绝对下降步长。PCSD 与 AwPCSD
的差异只在正则化梯度：PCSD 使用普通 TV，AwPCSD 使用自适应加权 TV；
OS 版本只改变数据子集大小。

### 4.7 普通 TV 的离散更新

当前 TIGRE TV kernel 使用三维后向差分。对体素 $i=(x,y,z)$：

```math
D_xx_i=x_i-x_{i-\hat x},\qquad
D_yx_i=x_i-x_{i-\hat y},\qquad
D_zx_i=x_i-x_{i-\hat z},
```

边界外的差分按 0 处理。定义：

```math
n_i=\sqrt{(D_xx_i)^2+(D_yx_i)^2+(D_zx_i)^2}+\varepsilon.
```

kernel 计算的 TV 梯度分量等价于离散负散度：

```math
q_i=
\frac{D_xx_i+D_yx_i+D_zx_i}{n_i}
-\frac{D_xx_{i+\hat x}}{n_{i+\hat x}}
-\frac{D_yx_{i+\hat y}}{n_{i+\hat y}}
-\frac{D_zx_{i+\hat z}}{n_{i+\hat z}}.
```

每次 TV 内迭代先把整个梯度归一化：

```math
\widehat q=\frac{q}{\lVert q\rVert_2},
```

再按绝对步长 $s$ 更新：

```math
x\leftarrow x-s\widehat q.
```

若 $\lVert q\rVert_2\le\varepsilon$，提前结束本轮 TV 内迭代。ASD-POCS
中的 $s=d_{\mathrm{tvg}}$，PCSD 中的 $s=s_k$。

### 4.8 自适应加权 TV

AwTV 对每个方向的差分设置边缘权重：

```math
w_d(i)=\exp\!\left[-\left(\frac{D_dx_i}{\delta}\right)^2\right],
\qquad d\in\{x,y,z\}.
```

其中 $\delta$ 对应 `adaptive_delta`；当 $\delta=0$ 时权重退化为 1。
加权局部范数为：

```math
n_i^{(w)}=
\sqrt{
w_x(i)(D_xx_i)^2+
w_y(i)(D_yx_i)^2+
w_z(i)(D_zx_i)^2
}+\varepsilon.
```

对应离散梯度为：

```math
q_i^{(w)}=
\frac{
w_x(i)D_xx_i+w_y(i)D_yx_i+w_z(i)D_zx_i
}{n_i^{(w)}}
-\frac{w_x(i+\hat x)D_xx_{i+\hat x}}{n_{i+\hat x}^{(w)}}
-\frac{w_y(i+\hat y)D_yx_{i+\hat y}}{n_{i+\hat y}^{(w)}}
-\frac{w_z(i+\hat z)D_zx_{i+\hat z}}{n_{i+\hat z}^{(w)}}.
```

随后仍按：

```math
x\leftarrow x-s\frac{q^{(w)}}{\lVert q^{(w)}\rVert_2}
```

更新。大梯度位置的 $w_d$ 较小，因此相较普通 TV 会减弱跨边缘平滑。

### 4.9 B-ASD-POCS-β

B-ASD-POCS-β 的图像更新与 ASD-POCS 相同，但数据更新使用私有工作投影
$b^{(j)}$。初始值为：

```math
b^{(0)}=b.
```

每隔 `bregman_interval` 个外循环执行：

```math
b^{(j+1)}=b^{(j)}+\mu_j\left(b^{(j)}-Ax\right),
```

```math
\mu_{j+1}=r_\mu\mu_j,
\qquad
r_\mu=\texttt{bregman\_beta\_reduction}.
```

后续数据一致性更新使用 $b^{(j+1)}$，而调用者传入的原始 $b$ 始终只读。

### 4.10 公共停止条件

ASD/PCSD 家族在每个外循环结束后检查：

```math
\left(c<-0.99\ \land\ d_d\le\epsilon\right)
\quad\lor\quad
\left(\beta<\beta_{\min}\right).
```

其中：

```math
\epsilon=\texttt{max\_l2\_error},
\qquad
\beta_{\min}=\texttt{minimum\_beta}.
```

未显式设置 `max_l2_error` 时，先执行 FDK 初始化体积 $x_{\mathrm{FDK}}$，
自动估计：

```math
\epsilon=0.2\lVert Ax_{\mathrm{FDK}}-b\rVert_2.
```

若当前 geometry 不支持 FDK，则退回：

```math
\epsilon=0.2\lVert b\rVert_2.
```

## 5. 初始化与 Nesterov

`initialization` 支持：

- `Zero`：零体积；
- `Fdk`：先以相同 geometry 执行 FDK；设备投影会下载到主机以适配当前
  `FdkPipeline`；
- `DeviceVolume`：从 `d_initial_volume` 复制调用方已有设备体积。

`relaxation_mode=Nesterov` 复现 TIGRE MATLAB SART/SIRT/OS-SART 的
`lambda='nesterov'` 更新：每个子集先产生候选解，再与上一候选解按动量系数
组合。每个完整外循环检查投影残差；残差上升时与 TIGRE 一样直接停止，
不回退到上一体积。

Nesterov 是 ART 数据更新选项，不是 ASD/PCSD 家族的公共正则化开关。

## 6. ASD-POCS 与 PCSD 状态

每个 POCS 外循环显式维护：

```text
dd    当前投影残差
dp    数据一致性更新幅度
dg    TV 更新幅度
c     数据更新与 TV 更新的方向余弦
dtvg  TV 绝对步长
beta  数据更新松弛因子
```

普通 ASD-POCS 首轮使用 `dtvg = alpha * dp`；若
`dg > maximum_update_ratio * dp` 且投影残差仍高于阈值，则按
`alpha_reduction` 缩小后续 TV 步长。

PCSD 首轮 TV 步长为 1，后续使用当前数据更新前投影距离与首轮基准距离的
比值。TV/AwTV 每次内迭代都将梯度归一化为 L2 范数 1，再应用绝对步长。

B-ASD-POCS-β 使用私有工作投影执行 Bregman 递推：

```text
b_work <- b_work + mu * (b_work - A*x)
mu     <- mu * bregman_beta_reduction
```

调用者传入的测量投影保持只读。

## 7. 统计与停止原因

`statistics()` 返回完成外循环数、子集更新数、`dd/dp/dg/c`、当前 beta、
TV 步长以及停止原因。可能的提前停止包括：

- 方向余弦接近 -1 且投影误差达到阈值；
- beta 低于 `minimum_beta`；
- Nesterov 模式投影残差开始上升。

这些统计用于调参和回归检查，不改变求解状态。

## 8. 与通用代数重建接口的边界

`AlgebraicReconstructor` 的 `SmoothedTv` 是无状态的“数据外循环后正则化”
策略，适合普通 OS-SART-TV。ASD-POCS、PCSD 和 Bregman 算法需要跨轮维护
`dp/dg/dtvg/beta` 等状态，因此由 `TigreGradientReconstructor` 作为完整
求解器实现，不能等价地降为一次 `regularizer.apply()`。

## 9. 回归测试

```powershell
cmake --build out/build/x64-refactor-check --config Release `
  --target YKCBCT ykcbct_manual_tests --parallel 8

out/build/x64-refactor-check/Yktest/Release/ykcbct_manual_tests.exe `
  recon/tigre-gradient-family
```

该入口覆盖全部算法枚举、Nesterov、标准/Ex geometry、FDK/设备体积初始化，
并验证 Bregman 流程不会修改调用者的测量投影。

## 10. 大箭头模体切换算法

`large/arrow-tigre` 固定使用 `160×160×96` 带六面方向字符的箭头模体、
`384×128` 平板探测器和 360°/360 张投影。重建方法和常用参数可从命令行
切换，模体与采集条件保持不变，适合横向比较：

```powershell
# 默认 OS-ASD-POCS
ykcbct_manual_tests.exe large/arrow-tigre --log-level info

# OS-SART，不执行 TV
ykcbct_manual_tests.exe large/arrow-tigre `
  --arrow-method os-sart `
  --arrow-iterations 20 `
  --arrow-block-size 30 `
  --arrow-lambda 0.2

# AwPCSD
ykcbct_manual_tests.exe large/arrow-tigre `
  --arrow-method aw-pcsd `
  --arrow-iterations 10 `
  --arrow-lambda 0.25 `
  --arrow-tv-iterations 5
```

`--arrow-method` 支持：

```text
sart, os-sart, sirt,
asd-pocs, os-asd-pocs, b-asd-pocs-beta,
pcsd, os-pcsd, aw-pcsd, os-aw-pcsd,
aw-asd-pocs, os-aw-asd-pocs
```

`--arrow-block-size` 只对 `os-*` 方法生效，语义仍是“每个子集包含的视角
数”。SART/ASD/PCSD 非 OS 版本固定每个子集一张，SIRT 固定一个子集包含
全部 360 张。

结果写入 `out/test-artifacts/large-arrow-<method>/`，因此不同算法不会互相
覆盖。每个目录包含模体、投影、重建、误差、JSON 参数和三层对比 BMP。

当前 OS-SART、`block_size=20`、`lambda=0.25` 的实测收敛数据如下：

| 外循环 | 子集更新数 | 重建耗时 | Correlation | NRMSE |
|---:|---:|---:|---:|---:|
| 20 | 360 | 934.7 ms | 0.939268 | 0.343185 |
| 40 | 720 | 1952.7 ms | 0.960102 | 0.279651 |
| 80 | 1440 | 3886.5 ms | 0.971467 | 0.237173 |

80 轮结果保存在 `out/test-artifacts/large-arrow-os-sart/`。上述耗时只包含
重建阶段；同次 80 轮测试的正投影约 15.5 ms、准备阶段约 30.9 ms。
