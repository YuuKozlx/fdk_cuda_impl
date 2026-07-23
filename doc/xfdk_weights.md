# xFDK 权重公式整理

没确认对不对，暂时不要相信

> 参考文献：Grimmer et al., "Cone-beam CT image reconstruction with extended z range," *Medical Physics*, 2009.

---

## 1. 符号定义

| 符号 | 含义 | 对应代码量 |
|------|------|-----------|
| $R_F$ | 源–等中心距（FOV 半径） | `SID` |
| $c$ | 锥角斜率 | `iPV * dv_mm / (2 * SDD)` |
| $r$ | 体素到旋转轴距离 | `sqrt(x² + y²)` |
| $\varphi$ | 体素方位角 | `atan2(x, y)` |
| $z$ | 体素 z 坐标 | `worldZ` |
| $\alpha$ | 源旋转角（扫描角） | `angle_list[i]` |
| $\beta$ | 体素在当前视角下的扇角 | 见第 2 节 |
| $\vartheta$ | 平行束投影角，$\vartheta = \alpha + \beta$ | `vartheta` |
| $\Delta\vartheta$ | 体素的覆盖角度余量 | `delta_vt` |

---

## 2. 扇角 $\beta$ 的计算

源旋转角为 $\alpha$ 时，体素 $(r, \varphi)$ 对应的扇角：

$$
\beta = -\arctan \frac{r \sin(\alpha - \varphi)}{R_F + r \cos(\alpha - \varphi)}
$$

平行束角度：

$$
\vartheta = \alpha + \beta
$$

---

## 3. 体素覆盖范围 $\Delta\vartheta$

定义分界线（demarcation line）：

$$
\text{dem}(z) = R_F - c\,|z|
$$

**情形一：完整覆盖区** $r \leq \text{dem}(z)$

$$
\Delta\vartheta = \frac{\pi}{2}
$$

体素被完整 360° 投影覆盖。

**情形二：扩展区** $r > \text{dem}(z)$ 且 $\text{dem}(z) > 0$

$$
\Delta\vartheta = \arcsin\frac{R_F\,(R_F - c|z|) - r^2}{r\,\sqrt{R_F\,(2c|z| - R_F) + r^2}}
$$

等价形式（论文 Eq. 推导）：

$$
\Delta\vartheta = \arcsin\frac{R_F - c|z|}{r} - \arctan\frac{\sqrt{r^2 - (R_F - c|z|)^2}}{c|z|}
$$

**情形三：超出最后可重建层** $\text{dem}(z) \leq 0$

$$
\Delta\vartheta = 0
$$

---

## 4. 短扫描权重 $w_{\mathrm{PS}}$（体素依赖 Parker 权重）

定义辅助角度：

$$
\vartheta_1 = \varphi - \frac{\pi}{2}, \qquad \vartheta_2 = \varphi + \frac{\pi}{2}
$$

过渡函数 $s(t) = \sin\!\left(\dfrac{\pi t}{2}\right)$，权重定义为：

$$
w_{\mathrm{PS}}(\boldsymbol{r},\,\vartheta) = \frac{1}{2}
\begin{cases}
0 & \vartheta < \vartheta_1 - \Delta\vartheta \\[4pt]
1 + s\!\left(\dfrac{\vartheta - \vartheta_1}{\Delta\vartheta}\right) & \vartheta_1 - \Delta\vartheta \leq \vartheta < \vartheta_1 + \Delta\vartheta \\[8pt]
2 & \vartheta_1 + \Delta\vartheta \leq \vartheta < \vartheta_2 - \Delta\vartheta \\[4pt]
1 - s\!\left(\dfrac{\vartheta - \vartheta_2}{\Delta\vartheta}\right) & \vartheta_2 - \Delta\vartheta \leq \vartheta < \vartheta_2 + \Delta\vartheta \\[8pt]
0 & \vartheta \geq \vartheta_2 + \Delta\vartheta
\end{cases}
$$

**归一化性质**：$w_{\mathrm{PS}}$ 满足 180° 归一化条件，即对完整覆盖体素：

$$
\int w_{\mathrm{PS}}\,\mathrm{d}\vartheta = \pi
$$

---

## 5. 全扫描权重与过渡权重

对完整 360° 覆盖区，改用全扫描权重以提升剂量效率：

$$
w_{\mathrm{FS}}(\boldsymbol{r}, \vartheta) = 1
$$

定义过渡参数：

$$
\Delta r = \frac{R_M^2}{2 R_F}, \qquad r_0 = R_F - c\,z - \Delta r
$$

其中 $R_M$ 为最后可重建层处的最大重建半径。

过渡权重：

$$
w_T(\boldsymbol{r}, \vartheta) = \frac{1}{2}
\begin{cases}
0 & r < r_0 - \Delta r \\[4pt]
1 + s\!\left(\dfrac{r - r_0}{\Delta r}\right) & r_0 - \Delta r \leq r < r_0 + \Delta r \\[8pt]
2 & r \geq r_0 + \Delta r
\end{cases}
$$

含义：
- $w_T = 0$：体素离轴近，用全扫描权重 $w_{\mathrm{FS}}$
- $w_T = 1$：体素离轴远（扩展区），用短扫描权重 $w_{\mathrm{PS}}$
- $0 < w_T < 1$：平滑过渡区

---

## 6. 最终复合权重 $w_C$

$$
w_C(\boldsymbol{r}, \vartheta) = \bigl(1 - w_T(\boldsymbol{r}, \vartheta)\bigr)\,w_{\mathrm{FS}}(\boldsymbol{r}, \vartheta) + w_T(\boldsymbol{r}, \vartheta)\,w_{\mathrm{PS}}(\boldsymbol{r}, \vartheta)
$$

展开后：

$$
w_C = (1 - w_T) \cdot 1 + w_T \cdot w_{\mathrm{PS}} = 1 - w_T\,(1 - w_{\mathrm{PS}})
$$

---

## 7. 反投影公式

$$
f(\boldsymbol{r}) = \frac{1}{2} \int \mathrm{d}\vartheta\; w_C(\boldsymbol{r}, \vartheta)\; k(\xi) * p(\vartheta, \xi, \gamma)
$$

其中 $k(\xi) * p$ 表示滤波后的投影数据，$\gamma$ 为扇角坐标。

---

## 8. 实现归一化

反投影累积量：

$$
\text{angle\_accum}(\boldsymbol{r}) = \sum_i \Delta\vartheta_i \cdot w_{\mathrm{PS},i}(\boldsymbol{r})
$$

归一化：

$$
f(\boldsymbol{r}) \leftarrow f(\boldsymbol{r}) \times \frac{\pi}{\text{angle\_accum}(\boldsymbol{r})}
$$

对完整覆盖体素，$\text{angle\_accum} \to \pi$，归一化因子为 1。

---

## 9. CUDA kernel 对应关系

| 公式量 | kernel 变量 | 来源 |
|--------|------------|------|
| $R_F$ | `sqrtf(gC_coeffs[0].SID2)` | constant memory |
| $c$ | `iPV * dv_mm / (2 * sqrt(SDD2))` | constant memory |
| $r$ | `sqrtf(fX*fX + fY*fY)` | 体素坐标 |
| $\varphi$ | `atan2f(fX, fY)` | 体素坐标 |
| $\alpha$ | `c.alpha`（原 `nReserved1`） | `FdkAffineCoeff` |
| $\beta$ | `-atan2f(r*sin(α-φ), R_F+r*cos(α-φ))` | kernel 内计算 |
| $\vartheta$ | `alpha + beta` | kernel 内计算 |
| $\Delta\vartheta$ | `delta_vt` | kernel 内逐 iz 计算 |
| $w_{\mathrm{PS}}$ | `w_PS` | kernel 内计算，乘 0.5 |
| angle\_accum | `A[iz] += dtheta * w_PS` | 写回独立 buffer |
