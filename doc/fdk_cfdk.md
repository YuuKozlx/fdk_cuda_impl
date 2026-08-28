# FDK 重建算法完整公式手册
## （平面探测器 vs 柱面探测器）

---

## 1. 符号说明

| 符号 | 含义 |
| :--- | :--- |
| $f(x,y,z)$ | 待重建的三维体素值 |
| $\beta$ | 投影角度（射线源绕旋转轴旋转的角度），范围 $[0, 2\pi]$ |
| $D_{so}$ 或 $R$ | 射线源到旋转中心（物体中心）的距离 |
| $\gamma$ | 扇角（仅柱面探测器），当前射线与中心射线在水平面上的夹角 |
| $u, v$ | 平面探测器的探测器坐标（距离坐标，单位：mm） |
| $q$ | 柱面探测器的垂直坐标（距离坐标，单位：mm） |
| $s$ | 重建点在射线方向上的投影距离，$s = x\cos\beta + y\sin\beta$ |
| $L$ | 重建点到射线源的真实三维欧氏距离的水平投影 |
| $p(\cdot)$ | 原始投影数据 |
| $p'(\cdot)$ | 加权后的投影数据 |
| $\tilde{p}(\cdot)$ | 滤波后的投影数据 |
| $h(\cdot)$ | 一维斜坡滤波器的卷积核 |
| $N$ | 总投影角度数 |
| $\Delta u$ | 平面探测器单元的采样间隔 |
| $\Delta\gamma$ | 柱面探测器单元的角采样间隔 |

---

## 2. 平面探测器 FDK 算法

### 2.1 坐标系统

- 探测器为**平面板**，坐标用距离 $(u, v)$ 表示
- $u$：水平方向到中心射线的距离（单位：mm）
- $v$：垂直方向到中心射线的距离（单位：mm）
- 原始投影数据：$p_\beta(u, v)$

---

### 2.2 步骤一：锥角-扇角加权（Cosine Weighting）

对原始投影数据进行加权，补偿射线倾斜导致的路径长度变化：

$$
\boxed{p'_\beta(u, v) = \frac{D_{so}}{\sqrt{D_{so}^2 + u^2 + v^2}} \cdot p_\beta(u, v)}
$$

该权重可拆解为两个因子：

$$
\frac{D_{so}}{\sqrt{D_{so}^2 + u^2 + v^2}}
= \frac{D_{so}}{\sqrt{D_{so}^2 + u^2}} \cdot \frac{\sqrt{D_{so}^2 + u^2}}{\sqrt{D_{so}^2 + u^2 + v^2}}
$$

- $\frac{D_{so}}{\sqrt{D_{so}^2 + u^2}}$：**扇角加权**（补偿水平倾斜）
- $\frac{\sqrt{D_{so}^2 + u^2}}{\sqrt{D_{so}^2 + u^2 + v^2}}$：**锥角加权**（补偿垂直倾斜）

---

### 2.3 步骤二：滤波（Filtering）

沿水平方向（$u$ 方向）进行一维斜坡滤波：

$$
\boxed{\tilde{p}_\beta(u, v) = p'_\beta(u, v) * h(u)}
$$

其中 $*$ 表示卷积运算，$h(u)$ 是一维斜坡滤波器的空间域核。

其频率响应为：

$$
H(\omega) = |\omega|
$$

实际实现时需加窗（如 Shepp-Logan、Hamming、Hanning 窗）抑制高频噪声。

离散形式的滤波公式：

$$
\tilde{p}_\beta(u_i, v) = \Delta u \sum_{j} p'_\beta(u_j, v) \cdot h(u_i - u_j)
$$

---

### 2.4 步骤三：加权反投影（Weighted Back-Projection）

将滤波后的投影数据沿锥束射线路径加权累加回三维空间：

$$
\boxed{f(x,y,z) = \int_{0}^{2\pi} \frac{D_{so}^2}{(D_{so} - s)^2} \cdot \tilde{p}_\beta\big(u_{(x,y,\beta)},\; v_{(x,y,z,\beta)}\big) \, d\beta}
$$

其中：

**深度加权因子**：

$$
W_{\text{depth}} = \frac{D_{so}^2}{(D_{so} - s)^2}, \qquad s = x\cos\beta + y\sin\beta
$$

**水平坐标映射**：

$$
u_{(x,y,\beta)} = \frac{D_{so} \cdot (-x\sin\beta + y\cos\beta)}{D_{so} - s}
$$

**垂直坐标映射**：

$$
v_{(x,y,z,\beta)} = \frac{D_{so} \cdot z}{D_{so} - s}
$$

---

### 2.5 离散化实用公式

$$
\boxed{f(x,y,z) \approx \frac{2\pi}{N} \sum_{i=1}^{N} \frac{D_{so}^2}{(D_{so} - s_i)^2} \cdot \tilde{p}_{\beta_i}\big(u_{(x,y,\beta_i)},\; v_{(x,y,z,\beta_i)}\big)}
$$

---

## 3. 柱面探测器 FDK 算法（C-FDK）

### 3.1 坐标系统

- 探测器为**圆柱面**，坐标用角度 $(\gamma, q)$ 表示
- $\gamma$：扇角，即当前射线与中心射线在水平面上的夹角（单位：弧度）
- $q$：垂直方向到中心平面的距离（单位：mm）
- 原始投影数据：$p(\beta, \gamma, q)$

---

### 3.2 步骤一：锥角-扇角加权（Cosine Weighting）

对原始投影数据进行加权，补偿射线倾斜导致的路径长度变化：

$$
\boxed{p'(\beta, \gamma, q) = \cos\gamma \cdot \frac{R}{\sqrt{R^2 + q^2}} \cdot p(\beta, \gamma, q)}
$$

其中：
- $\cos\gamma$：**扇角加权**，补偿水平方向的倾斜
- $\frac{R}{\sqrt{R^2 + q^2}}$：**锥角加权**，补偿垂直方向（z方向）的倾斜

> 当 $q=0$ 时，锥角加权为 1，退化为扇束情况。

---

### 3.3 步骤二：滤波（Filtering）

沿水平方向（即扇角 $\gamma$ 方向）对加权后的投影数据进行一维斜坡滤波：

$$
\boxed{\tilde{p}(\beta, \gamma, q) = p'(\beta, \gamma, q) * h(\gamma)}
$$

其中 $*$ 表示卷积运算，$h(\gamma)$ 是一维斜坡滤波器在**角度域**的卷积核。

其频率响应为：

$$
H(\omega) = |\omega|
$$

实际实现时需加窗抑制高频噪声。

离散形式的滤波公式：

$$
\tilde{p}(\beta, \gamma_i, q) = \Delta\gamma \sum_{j} p'(\beta, \gamma_j, q) \cdot h(\gamma_i - \gamma_j)
$$

其中 $\Delta\gamma$ 是探测器单元的角采样间隔。

---

### 3.4 步骤三：加权反投影（Weighted Back-Projection）

将滤波后的投影数据沿锥束射线路径加权累加回三维空间：

$$
\boxed{f(x,y,z) = \int_{0}^{2\pi} \frac{R^2}{L^2(x,y,\beta)} \cdot \tilde{p}\left(\beta,\; \gamma_{(x,y,\beta)},\; q_{(x,y,z,\beta)}\right) \, d\beta}
$$

其中：

**深度加权因子**：

$$
W_{\text{depth}} = \frac{R^2}{L^2(x,y,\beta)}
$$

**深度加权分母**：

$$
L(x,y,\beta) = \sqrt{(R + x\cos\beta + y\sin\beta)^2 + (-x\sin\beta + y\cos\beta)^2}
$$

**扇角映射**：

$$
\gamma_{(x,y,\beta)} = \arctan \frac{-x\sin\beta + y\cos\beta}{R + x\cos\beta + y\sin\beta}
$$

**垂直坐标映射**：

$$
q_{(x,y,z,\beta)} = z \cdot \frac{R}{\sqrt{(R + x\cos\beta + y\sin\beta)^2 + (-x\sin\beta + y\cos\beta)^2}}
$$

---

### 3.5 离散化实用公式

$$
\boxed{f(x,y,z) \approx \frac{2\pi}{N} \sum_{i=1}^{N} \frac{R^2}{L^2(x,y,\beta_i)} \cdot \tilde{p}\left(\beta_i,\; \gamma_{(x,y,\beta_i)},\; q_{(x,y,z,\beta_i)}\right)}
$$

---

## 4. 平面探测器 vs 柱面探测器：完整对比表

| 对比项 | 平面探测器 FDK | 柱面探测器 FDK（C-FDK） |
| :--- | :--- | :--- |
| **坐标定义** | 距离坐标 $(u, v)$ | 角度坐标 $(\gamma, q)$ |
| **锥角-扇角加权** | $\displaystyle \frac{D_{so}}{\sqrt{D_{so}^2 + u^2 + v^2}}$ | $\displaystyle \cos\gamma \cdot \frac{R}{\sqrt{R^2 + q^2}}$ |
| **滤波方向** | 沿 $u$ 方向（距离域） | 沿 $\gamma$ 方向（角度域） |
| **滤波核频率响应** | $H(\omega) = \|\omega\|$ | $H(\omega) = \|\omega\|$ |
| **水平坐标映射** | $\displaystyle u = \frac{D_{so}(-x\sin\beta + y\cos\beta)}{D_{so} - x\cos\beta - y\sin\beta}$ | $\displaystyle \gamma = \arctan \frac{-x\sin\beta + y\cos\beta}{R + x\cos\beta + y\sin\beta}$ |
| **垂直坐标映射** | $\displaystyle v = \frac{D_{so} \cdot z}{D_{so} - x\cos\beta - y\sin\beta}$ | $\displaystyle q = z \cdot \frac{R}{\sqrt{(R + x\cos\beta + y\sin\beta)^2 + (-x\sin\beta + y\cos\beta)^2}}$ |
| **深度加权** | $\displaystyle \frac{D_{so}^2}{(D_{so} - s)^2}$ | $\displaystyle \frac{R^2}{L^2}$ |
| **深度加权分母** | $s = x\cos\beta + y\sin\beta$ | $L = \sqrt{(R + x\cos\beta + y\sin\beta)^2 + (-x\sin\beta + y\cos\beta)^2}$ |
| **算法精度** | 小角度近似下精确 | 几何关系更精确，无近似 |
| **伪影表现** | 锥角大时伪影明显 | 锥角大时伪影相对更小 |

---

## 5. 两种探测器的几何转换关系

### 5.1 柱面 → 平面（重排）

柱面探测器单元 $(\gamma, q)$ 对应到平面探测器坐标 $(u, v)$：

$$
u = R \cdot \tan\gamma
$$

$$
v = q \cdot \frac{R \cdot \cos\gamma}{\sqrt{R^2 + q^2}}
$$

### 5.2 平面 → 柱面（重排）

$$
\gamma = \arctan\left(\frac{u}{R}\right)
$$

$$
q = v \cdot \frac{\sqrt{R^2 + u^2}}{R}
$$

> **注**：重排过程涉及插值，会引入轻微的图像质量损失，但能显著改善大锥角下的图像均匀性。

---

## 6. 关键公式编号汇总

| 编号 | 公式 | 适用探测器 |
| :--- | :--- | :--- |
| (1) | $p'_\beta(u, v) = \dfrac{D_{so}}{\sqrt{D_{so}^2+u^2+v^2}} \cdot p_\beta(u, v)$ | 平面 |
| (2) | $\tilde{p}_\beta(u, v) = p'_\beta(u, v) * h(u)$ | 平面 |
| (3) | $f(x,y,z) = \displaystyle\int_{0}^{2\pi} \dfrac{D_{so}^2}{(D_{so}-s)^2} \cdot \tilde{p}_\beta(u,v) \, d\beta$ | 平面 |
| (4) | $p'(\beta, \gamma, q) = \cos\gamma \cdot \dfrac{R}{\sqrt{R^2+q^2}} \cdot p(\beta, \gamma, q)$ | 柱面 |
| (5) | $\tilde{p}(\beta, \gamma, q) = p'(\beta, \gamma, q) * h(\gamma)$ | 柱面 |
| (6) | $f(x,y,z) = \displaystyle\int_{0}^{2\pi} \dfrac{R^2}{L^2} \cdot \tilde{p}(\beta, \gamma, q) \, d\beta$ | 柱面 |

---

## 7. 参考文献与说明

- Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984). Practical cone-beam algorithm. *JOSA A*, 1(6), 612-619.
- 柱面探测器 FDK（C-FDK）公式源自扇形束 CT 几何的推广，常见于第三代 CT 系统和螺旋 CT 中。
- 本手册中所有公式均假设圆形扫描轨迹、等中心扫描几何。

---

**文档版本**：1.0  
**最后更新**：2026-08-28