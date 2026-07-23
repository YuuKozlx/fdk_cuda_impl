# 螺旋 CT 参数设计公式

> 适用场景：等中心平板探测器螺旋 CT，Feldkamp（FDK）或 Katsevich 重建。  
> 所有线性量单位为 **mm**，角度单位为 **rad**。

---

## 符号表

| 符号 | 字段名 | 含义 |
|---|---|---|
| $N_V$, $d_z$ | `iVZ`, `vox_z_mm` | 体积 Z 层数、体素 Z 尺寸 |
| $N_V^{row}$, $d_v$ | `iPV`, `dv_mm` | 探测器排数、排间距 |
| $N_V^{col}$, $d_u$ | `iPU`, `du_mm` | 探测器列数、列间距 |
| $\text{SID}$, $\text{SDD}$ | `SID`, `SDD` | 源–等中心距、源–探测器距 |
| $h$ | `pitch_mm` | 螺旋床进量（mm/圈） |
| $p_f$ | — | Pitch 比（推荐 0.75 ~ 1.0） |
| $N_r$ | `n_rotations` | 旋转圈数 |
| $N_\phi$ | `views_per_rot` | 每圈投影数 |
| $z_{\text{start}}$ | `start_z_mm` | 扫描起始 Z 位置 |

---

## 公式 1：几何放大比

$$M = \frac{\text{SDD}}{\text{SID}}$$

所有探测器物理尺寸换算到等中心均除以 $M$。

---

## 公式 2：探测器 Z 方向等中心覆盖

单排在等中心处的有效宽度：

$$\delta_z = \frac{d_v}{M}$$

全部排的 Z 向总覆盖：

$$\Delta Z_{\text{det}} = N_V^{row} \cdot \delta_z = \frac{N_V^{row} \cdot d_v}{M}$$

---

## 公式 3：Pitch 约束（双向推导）

**给定探测器排数，求最大允许 pitch：**

$$\boxed{h_{\max} = p_f \cdot \Delta Z_{\text{det}} = p_f \cdot \frac{N_V^{row} \cdot d_v}{M}}$$

**给定 pitch，求最少探测器排数：**

$$\boxed{N_V^{row} \geq \left\lceil \frac{h \cdot M}{p_f \cdot d_v} \right\rceil}$$

> **推荐取值：** $p_f = 0.75$（标准），$p_f \leq 1.0$（极限无重叠），$p_f > 1.0$ 需 z 插值补偿。

---

## 公式 4：体积 Z 范围

$$Z_{\text{vol}} = N_V \cdot d_z$$

---

## 公式 5：扫描 ramp 余量

前后各需一个探测器覆盖 + 一个 pitch 作为加减速缓冲：

$$Z_{\text{margin}} = \Delta Z_{\text{det}} + h$$

---

## 公式 6：扫描起始位置

体积中心位于 $z = 0$：

$$\boxed{z_{\text{start}} = -\left(\frac{Z_{\text{vol}}}{2} + Z_{\text{margin}}\right) = -\left(\frac{N_V \cdot d_z}{2} + \frac{N_V^{row} \cdot d_v}{M} + h\right)}$$

---

## 公式 7：所需旋转圈数

$$\boxed{N_r = \left\lceil \frac{Z_{\text{vol}} + 2\,\Delta Z_{\text{det}}}{h} \right\rceil + 1}$$

$+1$ 保证末圈完整落在体积边界之外。

---

## 公式 8：每圈投影数（角度 Nyquist 采样）

探测器 U 方向等中心半宽：

$$W_u = \frac{N_V^{col} \cdot d_u}{2M}$$

扇角半角：

$$\gamma_{\max} = \arctan\!\left(\frac{W_u}{\text{SID}}\right)$$

单列角步长（精确）：

$$\Delta\gamma = \arcsin\!\left(\frac{d_u / M}{\text{SID}}\right)$$

Nyquist 采样下界（短扫描取等号）：

$$\boxed{N_\phi \geq \left\lceil \frac{\pi + 2\gamma_{\max}}{\Delta\gamma} \right\rceil}$$

近似式（$d_u / M \ll \text{SID}$ 时）：

$$N_\phi \approx \frac{(\pi + 2\gamma_{\max}) \cdot M \cdot \text{SID}}{d_u}$$

---

## 公式 9：总投影数

$$N_{\text{total}} = N_r \cdot N_\phi$$

---

## 公式 10：最大锥角（检验是否需要锥束修正）

$$\boxed{\alpha_{\max} = \arctan\!\left(\frac{\Delta Z_{\text{det}} / 2}{\text{SID}}\right)}$$

| $\alpha_{\max}$ | 建议 |
|---|---|
| $< 2°$ | Feldkamp (FDK) 直接适用 |
| $2° \sim 5°$ | 加权 FDK（Parker + 锥角补偿）|
| $> 5°$ | Katsevich / WEDGE / 迭代重建 |

---

## 参数验证示例（本项目当前配置）

| 计算项 | 结果 | 状态 |
|---|---|---|
| $M = 1000/500$ | $2.0$ | ✓ |
| $\Delta Z_{\text{det}} = 32 \times 0.25 / 2$ | $4.0\ \text{mm}$ | ✓ |
| $p_f = 3.0 / 4.0$ | $0.75$ | ✓ 推荐范围 |
| $Z_{\text{vol}} = 400 \times 0.45$ | $180\ \text{mm}$ | ✓ |
| $z_{\text{start}} = -(90 + 4 + 3)$ | $-97\ \text{mm}$ | ⚠ 原值 $-25$ mm 不足 |
| $N_r = \lceil(180+8)/3\rceil + 1$ | $64\ \text{圈}$ | ⚠ 原值 15 圈不足 |
| $\alpha_{\max} = \arctan(2/500)$ | $0.23°$ | ✓ FDK 直接适用 |

---

## 参考文献

- Kak & Slaney, *Principles of Computerized Tomographic Imaging*, SIAM 2001  
- Feldkamp, Davis & Kress, *J. Opt. Soc. Am. A*, 1(6), 1984  
- Buzug, *Computed Tomography*, Springer 2008
