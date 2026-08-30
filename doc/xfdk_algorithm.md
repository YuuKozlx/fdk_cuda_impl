# xFDK 实现说明

## 定位

xFDK（Grimmer 等，2009）通过平行束变量重排和部分扫描权重扩展 FDK 的轴向有效范围。当前实现面向平板探测器、理想等角完整圆轨迹，入口为 `Fdk::XfdkPipeline`。

## 坐标和重排

输入为 `[alpha][v][u]`。内部坐标为 `(vartheta,gamma,xi)`：

- `xi` 是平行束横向距离，`xi=-R sin(beta)`；
- `gamma` 是锥角；
- `vartheta` 是重排后的视图角。

每个输出样本反算原始 `(alpha,u,v)`，乘 `cos(beta)cos(gamma)` 后从输入纹理线性插值。`u/v` 越界使用 border-zero；视图窗口另外保留相邻视图和扫描首尾周期缓存。

## 滤波与反投影

重排后的投影沿 `xi` 方向用公共 FDK ramp 滤波。反投影对每个体素计算 `r=sqrt(x^2+y^2)`、方位角 `phi`、射线深度和 `gamma`，从滤波纹理采样，并乘 `0.5*DeltaTheta*wC`。

`wC` 由三部分组成：

1. `xfdkCoverageHalfRange` 根据 `beta_max`、`gamma_max` 和体素 `(r,z)` 判断可见角范围；
2. `xfdkPartialWeight` 在约 180 度边界使用 `sin(pi*t/2)` 平滑过渡；
3. `xfdkTransitionWeight` 在径向过渡区混合普通 FDK 与部分扫描权重。

因此 xFDK 不是简单地把投影分块后做普通 FDK。它需要完整角度顺序，并对每个 theta chunk 自动计算跨视图 halo；流式接口只缓存当前源窗口及周期首部，不要求整圈投影常驻显存。

## 适用条件和注意事项

- 只接受理想、等角、不重复终点的完整 `2pi` 圆轨迹；
- 不接受源偏移、探测器 offset/tilt/skew、FFS 或非等间隔角度；
- `gamma_max` 决定锥角覆盖，体素超出论文可见区域时权重为零；
- 流式调用顺序为 `prepareBatched`、`beginStreaming`、连续 `enqueueBatch`、`completeStreaming`；输入 batch 必须按采集顺序且来自 pinned host memory；
- 分批只降低投影缓存，不改变算法数学结果。离线与流式结果应在浮点误差范围内一致。

## 测试

`test-configs/reconstruction-static-flat.toml` 提供 Catphan 的 xFDK case；大锥角、流式一致性和 z 向 profile 测试分别位于专项测试入口。
