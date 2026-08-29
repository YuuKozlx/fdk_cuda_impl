# C-FDK 实现与验证状态

本文记录当前 C-FDK 实现已经确认的性质、仍存在的限制，以及后续复现实验
必须满足的条件。这里的 C-FDK 指 Li 等人在 2011 年提出的平板圆轨迹
curve-filtered FDK，不是圆柱探测器 FDK。

## 当前实现

当前数据流为：

1. 将平板锥束投影重排为 `p(theta,t,c)`；
2. 按论文式 (29)-(31) 完成三段重排与预加权；
3. 沿 `t` 方向执行一维 ramp 滤波；
4. 按论文式 (32)-(36) 反投影。

实现目前只接受理想、等角、完整 `2pi` 圆扫描。源偏移、探测器 offset、
tilt、skew、短扫和任意逐视图轨迹会被明确拒绝。

## 已确认事项

- 三段 `c` 映射通过前向/反向往返、正负 z 镜像和分界连续性测试；
- 项目坐标约定下的重排角度和径向符号已经过反向 A/B 测试；
- 普通 FDK 与 ASTRA Toolbox 2.5.0 的官方 `FDK_CUDA` 使用同一投影比较，
  全体素相关系数约为 `0.999993`；
- 大锥角水模中，ASTRA FDK 与项目 FDK 的端部/中心比均约为 `0.822`；
- 同一水模下当前 C-FDK 的端部/中心比约为 `0.846`，说明它确实缓解了
  z 向密度下降，但没有完全恢复到真值；
- 论文几何下的 Shepp-Logan 测试中，C-FDK 对远离中平面的材料值也有
  可测量改善。

## 尚未完成的严格验证

当前结果不能视为对论文实验的完整复现，C-FDK 仍需进一步检验：

- 当前投影来自体素化模体和 Joseph 单射线前投；论文使用数学椭球解析
  线积分，并对每个探测器像素采用 `3x3` 子射线；
- 当前 ASTRA Shepp-Logan 表与论文引用的 Turbell 标准 3-D 模体不保证
  完全相同；
- 重排过程包含角度、探测器和 `(t,c)` 纹理插值，尚未单独量化每一级
  插值造成的强度与空间分辨率损失；
- 平顶有限高水柱的硬端面主要检验 z 向边界恢复。C-FDK 不是 z 向反卷积，
  该模体不能单独替代论文的平滑椭球均匀性实验；
- 大尺寸重排的性能和个别中央条纹仍需单独排查。

在完成解析椭球、`3x3` 子射线和论文原始模体的同条件对比前，应将当前
C-FDK 标记为“公式与基础数值性质已验证，论文级图像质量仍待复现”。

## 复现入口

```text
ykcbct_manual_tests fdk/curve-filtered-piecewise-mapping
ykcbct_manual_tests fdk/curve-filtered-reconstruction
ykcbct_manual_tests fdk/curve-filtered-large-cone
ykcbct_manual_tests fdk/curve-filtered-large-water
python scripts/compare_astra_fdk.py
```

测试输出保存到 `output/cfdk`，其中 raw 体数据和投影属于本地测试产物，
不应提交到 Git。
