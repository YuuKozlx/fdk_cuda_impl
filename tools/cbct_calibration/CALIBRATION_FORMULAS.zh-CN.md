# 平板 CBCT 钢珠几何校正公式

本文把完整计算拆成独立阶段。模型是圆轨道、固定几何、面外倾角为零；面内旋转角保留。

## 1. 图像和坐标

对投影图像做背景估计和阈值分割。候选区域用面积、等效半径和圆度筛选，并拒绝接触图像边界的区域。亚像素质心使用阈值以上强度加权：

```text
w_p = max(I_p-T, 0)
u = sum(w_p*column_p)/sum(w_p)
v = image_height-1-sum(w_p*row_p)/sum(w_p)
```

因此 `v` 向上，和后面的几何公式一致。物理长度为 `u_mm=u_px*du`、`v_mm=v_px*dv`。

## 2. 跨帧建立钢珠 ID

第一帧候选建立 `BB0,BB1,...`。后续帧只在仍有效的 ID 中做唯一距离匹配：

```text
c(j,l) = ||previous_position_j-current_candidate_l||_2
```

选择总距离最小的分配，并要求 `c(j,l)<=max_jump_px`。检测失败、接触边界或超过门限后，ID 永久停用，不能在后续帧重新激活。只保留所有帧有效的 ID；少于 3 颗直接失败。输出为 `tracks_px[view, bead, (u,v)]`。

## 3. 每颗钢珠拟合椭圆

对第 `k` 颗钢珠的所有轨迹点拟合：

```text
A*u^2+B*u*v+C*v^2+D*u+E*v+F=0
Q = [[A,B/2],[B/2,C]]
center_k = -0.5*solve(Q,[D,E])
```

约束 `4*A*C-B^2>0` 保证是椭圆。保存中心 `center_k=[ubar_k,vbar_k]` 和归一化二次型 `Q_k`。椭圆中心不等于三维圆心的投影，但在当前模型下所有椭圆中心位于旋转轴投影直线上。

## 4. 由椭圆中心求面内角

对椭圆中心做总最小二乘直线拟合：

```text
center_k = center_mean + t_k*d
```

SVD 的第一方向为 `d=[d_u,d_v]`，统一方向使 `d_v>0`，然后：

```text
eta = atan2(d_u,d_v)
```

这一阶段不需要径向对，也不需要相差 180 度的视角。

## 5. 旋正椭圆，求 SDD 和主点

```text
T = [[cos(eta),-sin(eta)],[sin(eta),cos(eta)]]
center'_k = T*center_k
Q'_k = T*Q_k*T^T
a_k = Q'_k[0,0]
b_k = Q'_k[1,1]
```

理想模型下 `Q'_k[0,1]` 应接近零。椭圆关系为：

```text
(vbar_k-v0*)^2 = 1/b_k + (a_k/b_k)*D^2
```

对不同珠子两两相减，得到关于 `v0*` 和 `D^2` 的线性方程。所有珠对一起最小二乘求解：

```text
D = SDD = sqrt(D^2)
u0* = mean(ubar_k)
```

## 6. 椭圆得到轨道比例

```text
rho_k  = 1/sqrt(1+a_k*D^2)
zeta_k = (vbar_k-v0*)*(1-rho_k^2)/D
```

这里 `rho_k=r_k/R`，`zeta_k=z_k/R`。此时仍缺少钢珠在旋转平面内的相位 `alpha_k`。

## 7. 用全部视角求 alpha，不使用径向对

定义：

```text
x_k = rho_k*cos(alpha_k)
y_k = rho_k*sin(alpha_k)
z_k = zeta_k
```

第 `i` 帧角度为 `beta_i`，旋转后：

```text
x_i = x_k*cos(beta_i)-y_k*sin(beta_i)
y_i = x_k*sin(beta_i)+y_k*cos(beta_i)
```

令旋正投影减主点为 `u_i=u_i*-u0*`、`v_i=v_i*-v0*`。投影模型为：

```text
u_i = -D*y_i/(1+x_i)
v_i =  D*z_k/(1+x_i)
```

消去分母，每帧给出两行线性方程：

```text
(u_i*cos(beta_i)+D*sin(beta_i))*x_k
 +(-u_i*sin(beta_i)+D*cos(beta_i))*y_k = -u_i

(v_i*cos(beta_i))*x_k
 +(-v_i*sin(beta_i))*y_k - D*z_k = -v_i
```

将全部视角堆叠为 `A_k*[x_k,y_k,z_k]=b_k`，用缩放最小二乘求解。最后：

```text
alpha_k = atan2(y_k,x_k)
```

`alpha_k` 是钢珠相对于旋转轴的物体相位，`beta_i` 是扫描角。当前实现用椭圆计算的 `rho_k,zeta_k` 作为最终轨道比例，线性方程主要确定 `alpha_k` 并检查轨迹一致性。

## 8. 式 (12) 和 SOD

钢珠的无量纲位置为：

```text
q_k = [rho_k*cos(alpha_k), rho_k*sin(alpha_k), zeta_k]
delta_hat_kl = ||q_k-q_l||
```

展开即论文式 (12)：

```text
delta_hat_kl^2 = (zeta_k-zeta_l)^2 + rho_k^2 + rho_l^2
                 - 2*rho_k*rho_l*cos(alpha_k-alpha_l)
```

实际珠间距满足 `d_kl=R*delta_hat_kl`。对距离矩阵中所有已知正距离，用过原点最小二乘恢复 SOD：

```text
SOD = R = sum(delta_hat_kl*d_kl)/sum(delta_hat_kl^2)
```

因此径向对不是 SOD 计算的必要条件；它只是论文中获得相位差的一种方法，当前实现不采用。

## 9. 主点、offset 和精修

旋正主点转回原始探测器坐标：

```text
[u0_mm,v0_mm] = T*[u0*,v0*]
u0_px = u0_mm/du
v0_px = v0_mm/dv
```

若重建器的 offset 定义为相对图像中心：`offset_u=u0_px-(Nu-1)/2`、`offset_v=v0_px-(Nv-1)/2`。若重建器的 `v` 向下，`offset_v` 可能需要变号，必须和重建器基向量约定核对。

解析结果可作为联合精修初值，同时优化 `[D,R,u0,v0,eta]` 和钢珠三维位置，残差包括像素重投影误差与已知珠间距误差。

## 10. 代码结果对应

`calibration.json` 中，`stage1` 是椭圆和面内角，`stage2` 是 SDD/主点，`stage3` 是 SOD/钢珠位置，`stage4` 是可选联合精修；`bead_phases_rad` 对应 `alpha_k`，`geometry.sdd_mm` 对应 `D`，`geometry.sod_mm` 对应 `R`。
