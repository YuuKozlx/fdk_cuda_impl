# FDK 锥束重建：从连续公式到离散实现

---

## 一、符号定义

| 符号 | 含义 | 单位 |
|------|------|------|
| $D$ | 源到旋转中心距离（SOD） | mm |
| $D'$ | 源到探测器距离（SDD） | mm |
| $u, v$ | 探测器物理坐标（U水平，V垂直） | mm |
| $d_u, d_v$ | 探测器像素间距 | mm |
| $f_s = 1/d_u$ | U方向采样率 | samples/mm |
| $f_c = 1/(2d_u)$ | 奈奎斯特截止频率 | cycle/mm |
| $\theta$ | 机架旋转角 | rad |
| $p(u, v, \theta)$ | 原始投影数据 | 无量纲（线积分） |
| $f(x, y, z)$ | 重建体素值 | mm⁻¹（线衰减系数） |
| $N_u, N_v$ | 探测器列数、行数 | — |
| $N_a$ | 投影角度数 | — |
| $\Delta\theta$ | 角度步长 $= 2\pi / N_a$ | rad |
| $N$ | FFT填零后长度（paddedN） | — |

---

## 二、连续公式推导

### 2.1 扇束到锥束的扩展

FDK 算法是 Feldkamp、Davis、Kress 在 1984 年将 2D 扇束重建推广到 3D 锥束的近似算法。其核心思路是：**对每一排探测器行独立做扇束重建，近似忽略锥角带来的误差**。

### 2.2 连续 FDK 公式

$$f(x, y, z) = \frac{1}{2} \int_0^{2\pi} \frac{D^2}{(D + x\sin\theta - y\cos\theta)^2} \cdot \hat{p}\!\left(\frac{D(x\cos\theta + y\sin\theta)}{D + x\sin\theta - y\cos\theta},\ \frac{Dz}{D + x\sin\theta - y\cos\theta},\ \theta\right) d\theta$$

其中 $\hat{p}(u, v, \theta)$ 是**滤波后的加权投影**，定义为：

$$\hat{p}(u, v, \theta) = \int_{-\infty}^{\infty} \tilde{p}(u', v, \theta) \cdot h(u - u') \, du'$$

$$\tilde{p}(u, v, \theta) = \frac{D}{\sqrt{D'^2 + u^2 + v^2}} \cdot p(u, v, \theta)$$

**各项物理意义：**

- $\dfrac{D^2}{(D + x\sin\theta - y\cos\theta)^2}$：反投影距离权重，补偿锥束几何放大效应
- $\dfrac{D}{\sqrt{D'^2 + u^2 + v^2}}$：预加权（余弦权重），将锥束投影等效为平行束
- $h(u)$：Ram-Lak 滤波核，频域响应 $H(f) = |f|$
- $\int \cdot\, d\theta$：沿所有角度反投影

### 2.3 连续 Ram-Lak 核

频域定义：

$$H(f) = |f|, \quad |f| \leq f_c = \frac{1}{2d_u}$$

空域解析表达（连续采样点 $t = n \cdot d_u$）：

$$h(n \cdot d_u) = \begin{cases} \dfrac{1}{4d_u^2} & n = 0 \\[8pt] 0 & n \text{ 偶},\ n \neq 0 \\[8pt] \dfrac{-1}{\pi^2 n^2 d_u^2} & n \text{ 奇} \end{cases}$$

单位：$\text{mm}^{-2}$（因为 $H(f)$ 的单位是 $\text{mm}^{-1}$，IFFT 后再除以一个 mm）

---

## 三、离散公式推导

### 3.1 离散化策略

将连续公式中的积分替换为有限求和，步长即为采样间隔：

$$\int f(x) dx \approx \sum_n f(n \cdot \Delta) \cdot \Delta$$

对 FDK 中的两个积分分别离散化：

- **卷积积分**（沿 $u$ 方向）：步长 $d_u$
- **角度积分**（沿 $\theta$）：步长 $\Delta\theta = 2\pi / N_a$

### 3.2 离散 Ram-Lak 核

连续核采样后乘以黎曼步长 $d_u$：

$$h[n] = h(n \cdot d_u) \times d_u = \begin{cases} \dfrac{1}{4d_u} & n = 0 \\[8pt] 0 & n \text{ 偶},\ n \neq 0 \\[8pt] \dfrac{-1}{\pi^2 n^2 d_u} & n \text{ 奇} \end{cases}$$

单位：$\text{mm}^{-1}$

**关键点：** $d_u$ 不能省略。FFT 实现的离散卷积是纯数字求和，不含任何步长因子，必须在核或结果中显式补偿。

### 3.3 离散卷积

$$\hat{p}[m, k, i] = \sum_{m'} \tilde{p}[m', k, i] \cdot h[m - m']$$

其中 $m$ 为探测器 U 方向像素索引，$k$ 为 V 方向索引，$i$ 为角度索引。

用 FFT 实现：

$$\hat{P}[f, k, i] = \tilde{P}[f, k, i] \times H[f]$$

cuFFT C2R 不归一化，输出放大 $N$ 倍，需在 $H[f]$ 中预先烘焙 $1/N$。

### 3.4 离散角度积分

$$f(x, y, z) \approx \frac{1}{2} \sum_{i=0}^{N_a - 1} w_i \cdot \frac{D^2}{U_i(x,y)^2} \cdot \hat{p}\!\left[u_i(x,y,z),\ v_i(x,y,z),\ i\right] \cdot \Delta\theta$$

其中：

$$U_i(x, y) = D + x\sin\theta_i - y\cos\theta_i$$

$$u_i(x,y,z) = \frac{D(x\cos\theta_i + y\sin\theta_i)}{U_i}, \quad v_i(x,y,z) = \frac{Dz}{U_i}$$

$w_i$ 为 Parker 短扫描权重（全扫描时 $w_i = 1$）。

$\Delta\theta$ 通常折叠进归一化系数或 Parker 权重中处理。

---

## 四、离散实现拆分

FDK 离散实现分为五个独立步骤，每步有明确的输入输出和单位：

```
原始投影 p[u,v,θ]  [无量纲]
    ↓ Step 1: Parker 加权
加权投影 p_w[u,v,θ]  [无量纲]
    ↓ Step 2: 余弦预加权
预加权投影 p̃[u,v,θ]  [无量纲]
    ↓ Step 3: Ram-Lak 滤波
滤波投影 p̂[u,v,θ]  [mm⁻¹]
    ↓ Step 4: FDK 反投影
重建体积 f[x,y,z]  [mm⁻¹]
```

---

### Step 1：Parker 短扫描加权

**目的：** 对短扫描（扫描角 $< 2\pi$）进行冗余数据加权，使重建结果等价于全扫描。全扫描时权重为 1。

**连续形式：**

$$p_w(u, v, \theta) = w_{\text{parker}}(u, \theta) \cdot p(u, v, \theta)$$

**Parker 权重定义（以 $\beta = \arctan(u/D')$ 为扇角）：**

$$w_{\text{parker}}(\beta, \theta) = \begin{cases}
\sin^2\!\left(\dfrac{\pi}{2} \cdot \dfrac{\theta}{2\beta_{\max} - 2\beta}\right) & 0 \leq \theta < 2\beta_{\max} - 2\beta \\[8pt]
1 & 2\beta_{\max} - 2\beta \leq \theta \leq \pi + 2\beta \\[8pt]
\sin^2\!\left(\dfrac{\pi}{2} \cdot \dfrac{\pi + 2\beta_{\max} - \theta}{2\beta_{\max} + 2\beta}\right) & \pi + 2\beta < \theta \leq \pi + 2\beta_{\max}
\end{cases}$$

**归一化约束：**

$$\Delta\theta_{\text{eff}} = \frac{2\pi}{N_a} \quad \Rightarrow \quad \text{fScale} = \frac{\Omega}{\pi}$$

其中 $\Omega$ 为实际扫描范围。短扫描时 $\text{fScale} \neq 1$，需补偿。

**单位分析：** 输入无量纲，权重无量纲，输出无量纲。

**伪代码：**

```
for each angle i:
    for each detector pixel (u, v):
        beta = arctan(u / SDD)
        p_w[u, v, i] = parker_weight(beta, theta[i]) * p[u, v, i]
```

---

### Step 2：余弦预加权

**目的：** 将锥束投影等效为平行束，补偿探测器像素到射线的距离差异。

**连续形式：**

$$\tilde{p}(u, v, \theta) = \frac{D}{\sqrt{D'^2 + u^2 + v^2}} \cdot p_w(u, v, \theta) = \cos\alpha(u, v) \cdot p_w(u, v, \theta)$$

其中 $\alpha(u, v)$ 是像素 $(u, v)$ 对应射线与探测器法线的夹角（锥角）。

**离散实现：**

$$\cos\alpha[m, k] = \frac{D'}{\sqrt{D'^2 + (m \cdot d_u)^2 + (k \cdot d_v)^2}}$$

注意权重只依赖探测器坐标，与角度无关，可预计算。

**单位分析：** 权重无量纲，输出无量纲。

**伪代码：**

```
预计算（与角度无关）:
for each detector pixel (m, k):
    u = (m - (Nu-1)/2) * du
    v = (k - (Nv-1)/2) * dv
    cos_weight[m, k] = SDD / sqrt(SDD² + u² + v²)

for each angle i:
    for each detector pixel (m, k):
        p̃[m, k, i] = cos_weight[m, k] * p_w[m, k, i]
```

---

### Step 3：Ram-Lak 滤波

#### 3.1 滤波核生成

**离散 Ram-Lak 核（du=1 纯数字版本）：**

$$\tilde{h}[n] = \begin{cases}
\dfrac{1}{4} & n = 0 \\[8pt]
0 & n \text{ 偶},\ n \neq 0 \\[8pt]
\dfrac{-1}{\pi^2 n^2} & n \text{ 奇}
\end{cases}$$

与物理正确核的关系：$\tilde{h}[n] = h[n] \times d_u$，即少乘了采样间隔 $d_u$。

**FFT 变换并烘焙 $1/N$（抵消 cuFFT IFFT 的 $N$ 倍放大）：**

$$\tilde{H}[k] = \frac{1}{N} \sum_{n=0}^{N-1} \tilde{h}[n] \cdot e^{-j2\pi kn/N}$$

**物理化（补偿缺失的 $d_u$，即补偿采样率 $f_s = 1/d_u$）：**

$$H[k] = \tilde{H}[k] \times \frac{1}{d_u}$$

**物理意义：** FFT 卷积是纯数字求和，不含黎曼步长。连续卷积近似离散化时：

$$\int w(u') h(u-u') du' \approx \sum_{m} w[m] h[n-m] \times d_u$$

步长 $d_u$ 等价于采样间隔 $1/f_s$，必须显式补偿，不可省略。

**加窗（可选，Hann / Hamming 等）：**

$$H_w[k] = H[k] \times W[k]$$

其中 $W[k]$ 在归一化频率 $[0, 0.5]$（奈奎斯特为 0.5）上定义，截止频率 $f_{\text{cutoff}} = 0.5$。

#### 3.2 填零与滤波

**目的：** 避免 FFT 循环卷积的边界效应。

$$N_{\text{pad}} = \text{next\_power\_of\_2}(2 N_u)$$

**批量 FFT 滤波（对所有 $N_a \times N_v$ 行同时处理）：**

$$\hat{P}[k, v, \theta] = \tilde{P}[k, v, \theta] \times H_w[k]$$

$$\hat{p}[u, v, \theta] = \text{IFFT}(\hat{P}[k, v, \theta])$$

**单位分析：** 输入 $\tilde{p}$ 无量纲，$H_w$ 单位 $\text{mm}^{-1}$，输出 $\hat{p}$ 单位 $\text{mm}^{-1}$。

**伪代码：**

```
# 滤波核生成（只做一次）
for n = 0..N-1:
    h̃[n] = rl_kernel(n)          # du=1 纯数字核
h̃ *= 1/N                         # bake_invN，抵消 IFFT 放大
H̃ = FFT(h̃)
H = H̃ * (1/du)                   # 物理化，补偿采样间隔
H *= window                       # 加窗（可选）

# 滤波（每次重建）
for each row (v, θ):
    p̃_padded = zero_pad(p̃[:, v, θ], N)
    P̃ = FFT(p̃_padded)
    P̂ = P̃ * H                    # 逐点乘
    p̂[:, v, θ] = IFFT(P̂)[0:Nu]  # 取有效段
```

---

### Step 4：FDK 反投影

**目的：** 将所有角度的滤波投影累加回体素空间。

**离散形式：**

$$f[x, y, z] = \frac{\Delta\theta}{2} \sum_{i=0}^{N_a-1} \frac{D^2}{U_i(x,y)^2} \cdot \hat{p}\!\left[u_i(x,y,z),\ v_i(x,y,z),\ i\right]$$

其中：

$$U_i(x, y) = D + x\sin\theta_i - y\cos\theta_i$$

$$u_i = \frac{D(x\cos\theta_i + y\sin\theta_i)}{U_i \cdot d_u} + \frac{N_u - 1}{2} \quad \text{（像素坐标）}$$

$$v_i = \frac{D \cdot z}{U_i \cdot d_v} + \frac{N_v - 1}{2} \quad \text{（像素坐标）}$$

**角度步长归一化：**

$\Delta\theta = 2\pi / N_a$，通常折叠进 Parker 权重的 $\text{fScale}$ 系数中：

$$\text{fScale} = \frac{\Omega}{\pi} \quad \text{（}\Omega\text{ 为扫描范围）}$$

全扫描 $\Omega = 2\pi$ 时 $\text{fScale} = 2$，与公式前的 $1/2$ 相消，净系数为 1。

**探测器插值：** $u_i, v_i$ 通常为非整数，需双线性插值（或 GPU 纹理采样）。

**单位分析：** 输入 $\hat{p}$ 单位 $\text{mm}^{-1}$，距离权重无量纲，$\Delta\theta$ 单位 rad，积分后输出 $f$ 单位 $\text{mm}^{-1}$。

**伪代码：**

```
for each voxel (x, y, z):
    f[x, y, z] = 0
    for each angle i:
        Ui  = D + x*sin(θ[i]) - y*cos(θ[i])
        ui  = D*(x*cos(θ[i]) + y*sin(θ[i])) / (Ui * du) + (Nu-1)/2
        vi  = D*z / (Ui * dv) + (Nv-1)/2
        w   = D² / Ui²                        # 距离权重
        val = bilinear_interp(p̂[:,:,i], ui, vi)
        f[x, y, z] += w * val * dθ / 2
```

---

## 五、各环节系数汇总

| 步骤 | 引入系数 | 来源 | 单位变化 |
|------|---------|------|---------|
| Parker 加权 | $w_{\text{parker}}$，$\text{fScale} = \Omega/\pi$ | 冗余补偿 + 角度归一化 | 无量纲 → 无量纲 |
| 余弦预加权 | $D'/\sqrt{D'^2 + u^2 + v^2}$ | 锥束几何 | 无量纲 → 无量纲 |
| 滤波核 bake_invN | $1/N$ | 抵消 cuFFT IFFT 放大 | — |
| 滤波核物理化 | $1/d_u$（采样率 $f_s$） | 补偿离散卷积缺失的黎曼步长 | 无量纲 → mm⁻¹ |
| 反投影距离权重 | $D^2 / U_i^2$ | 锥束放大补偿 | mm⁻¹ → mm⁻¹ |
| 角度积分步长 | $\Delta\theta / 2$ | 黎曼近似 + FDK 前置系数 | mm⁻¹·rad → mm⁻¹ |

---

## 六、完整单位链

$$\underbrace{p}_{\text{无量纲}}
\xrightarrow{\times w_{\text{parker}}} \underbrace{p_w}_{\text{无量纲}}
\xrightarrow{\times \cos\alpha} \underbrace{\tilde{p}}_{\text{无量纲}}
\xrightarrow{* h[n] \cdot (1/d_u)} \underbrace{\hat{p}}_{\text{mm}^{-1}}
\xrightarrow{\times D^2/U^2 \cdot \Delta\theta/2} \underbrace{f}_{\text{mm}^{-1}}$$
