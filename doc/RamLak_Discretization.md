# Ram-Lak 滤波器离散化推导

## 符号定义

| 符号 | 含义 | 单位 |
|------|------|------|
| `du` | 探测器像素间距 | mm |
| `N`  | FFT 长度（paddedN） | — |
| `f`  | 物理频率 | cycle/mm |
| `fc` | 奈奎斯特截止频率 = 1/(2·du) | cycle/mm |
| `h(t)` | 连续空域核 | mm⁻² |
| `h[n]` | 正确离散核 | mm⁻¹ |
| `h̃[n]` | 代码实现的离散核（少乘 du） | mm⁻² |

---

## 1. 连续域频域定义

$$H(f) = |f|, \quad f \in \left[-f_c,\ f_c\right], \quad f_c = \frac{1}{2 d_u}$$

单位：$\text{cycle/mm} = \text{mm}^{-1}$

---

## 2. 连续空域核

$$h(t) = 2\int_0^{f_c} f \cos(2\pi f t)\, df$$

采样点 $t = n \cdot d_u$ 处的解析值：

$$h(n \cdot d_u) = \begin{cases}
\dfrac{1}{4 d_u^2} & n = 0 \\[8pt]
0 & n \text{ 偶},\ n \neq 0 \\[8pt]
\dfrac{-1}{\pi^2 n^2 d_u^2} & n \text{ 奇}
\end{cases}$$

单位：$\text{mm}^{-2}$

---

## 3. 离散化（连续 → 离散求和）

将连续卷积积分离散化为黎曼求和：

$$q(u) = \int w(u')\, h(u - u')\, du' \approx \sum_m w[m]\, h[n-m] \cdot d_u$$

正确的离散核需要吸收黎曼步长 $d_u$：

$$h[n] = h(n \cdot d_u) \times d_u = \begin{cases}
\dfrac{1}{4 d_u} & n = 0 \\[8pt]
0 & n \text{ 偶},\ n \neq 0 \\[8pt]
\dfrac{-1}{\pi^2 n^2 d_u} & n \text{ 奇}
\end{cases}$$

单位：$\text{mm}^{-1}$

---

## 4. 代码实现的离散核

```cpp
if (n == 0)      val = 0.25f;
else if (an & 1) val = -1.0f / (pi * pi * n * n);
```

对应：

$$\tilde{h}[n] = \begin{cases}
\dfrac{1}{4} & n = 0 \\[8pt]
\dfrac{-1}{\pi^2 n^2} & n \text{ 奇}
\end{cases}$$

与正确离散核的关系：

$$\tilde{h}[n] = h[n] \times d_u \quad \Rightarrow \quad \tilde{h}[n] = h(n \cdot d_u)$$

**即：代码实现省略了离散化步骤中的黎曼步长 $d_u$，核是纯数字离散的，不含物理单位。**

---

## 5. FFT 变换到频域

$$\tilde{H}[k] = \sum_{n=0}^{N-1} \tilde{h}[n]\, e^{-j2\pi kn/N} \approx d_u \cdot H\!\left(\frac{k}{N d_u}\right) = \frac{k}{N}$$

---

## 6. 卷积结果的单位分析

FFT 卷积（cuFFT C2R 不归一化，输出放大 $N$ 倍）：

$$q[n] = \mathrm{IFFT}\bigl(W[k] \cdot \tilde{H}[k]\bigr) \times N$$

`bake_invN` 将 $1/N$ 烘焙进核，抵消放大因子，最终：

$$q[n] \approx q_{\text{correct}}(n \cdot d_u) \times d_u$$

与物理正确值的关系：

$$q_{\text{correct}} = q[n] \times \frac{1}{d_u}$$

---

## 7. postScale 的来源

| 步骤 | $d_u$ 因子 | 说明 |
|------|-----------|------|
| 连续核采样 $h(nd_u)$ | $1/d_u^2$ | 连续核本身含 $d_u^{-2}$ |
| 正确离散化 $\times d_u$ | $1/d_u$ | 黎曼步长 |
| 代码实现（省略 $d_u$） | $1/d_u^2$ | 漏掉了黎曼步长 |
| FFT 卷积（缺 $\times d_u$） | 净缺 $1/d_u$ | 离散求和 vs 连续积分 |
| **postScale = $1/d_u$** | **补回 $1/d_u$** | **结果单位正确 mm⁻¹** ✓ |

**结论：`postScale = 1/du` 补偿的是离散化时被省略的黎曼积分步长 $d_u$。**

---

## 8. 数值验证（du = 0.25 mm）

| 量 | 值 |
|----|-----|
| $f_c = 1/(2 \cdot 0.25)$ | $2.0\ \text{cycle/mm}$ |
| $h(0) = 1/(4 \cdot 0.25^2)$ | $4.0\ \text{mm}^{-2}$ |
| $\tilde{h}[0]$（代码） | $0.25$（无量纲） |
| 比值 $h(0)/\tilde{h}[0]$ | $16 = 1/d_u^2$，差了 $d_u^2$ |
| 离散化补 $d_u$，净差 | $1/d_u = 4.0\ \text{mm}^{-1}$ |
| postScale $= 1/0.25$ | $4.0$ ✓ |
