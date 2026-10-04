# 多能投影标签模体生成器

这是一个独立的纯 C++ 工具，用于生成可直接交给 `multispectrum_sim` 的
`uint8` 三维标签模体。工具只输出 RAW 标签体和 TOML 描述，不依赖 Python，
也不生成 JSON。

## 构建

```powershell
cmake -S example/phantom_generator -B out/build/phantom-generator -G Ninja
cmake --build out/build/phantom-generator
```

## 使用

```powershell
out/build/phantom-generator/phantom_generator.exe `
  low_contrast outputs/low-contrast.raw 256 256 64 1.0
```

参数依次为模体类型、输出 RAW、列数、行数、层数和各向同性体素尺寸（mm）。
支持以下类型：

- `shepp_logan`：三维多材料 Shepp-Logan 模体
- `tungsten_wire`：直径 0.05 mm 的轴向钨丝
- `tungsten_wire_slanted`：直径 0.05 mm、满足 `z = 0.42 * x` 的斜钨丝，用于层厚评测
- `gold_foil`：厚度 0.05 mm、垂直于 z 轴的金片，用于 z-MTF 评测
- `water_cylinder`：带 PMMA 壳体的圆柱水模
- `water_ellipse`：带 PMMA 壳体的椭圆柱水模
- `catphan`：含空气、特氟龙、Delrin、亚克力、聚苯乙烯、LDPE 和水插入物
- `low_contrast`：直径 15、9、8、7、6、5、3 mm、厚 5 mm，密度低于水 0.3%、0.5%、1% 的 21 个插入物
- `tungsten_bead_line_2mm_5mm`：单排直线 2 mm 钨球，球心间距 5 mm，外部 30 mm × 120 mm PMMA 圆柱支撑
- `tungsten_bead_line_3mm_10mm`：单排直线 3 mm 钨球，球心间距 10 mm，外部 30 mm × 120 mm PMMA 圆柱支撑
- `tungsten_bead_double_ring`：PMMA 圆柱内两圈相互对齐的钨珠；额外参数依次为圆直径、圆间距、钨珠直径、支撑直径、支撑长度和可选的每圈珠子数（默认 6）
- `tungsten_bead_double_ring_cylinder_marker`：对称双环的起点标记变体；上下环均保留相位 0° 的普通钨球，并在上环球正上方、下环球正下方 10 mm 处分别增加一个沿 Z 轴的 5 mm 直径、5 mm 高钨圆柱，其余参数与 `tungsten_bead_double_ring` 相同。
- `tungsten_bead_double_ring_cylinder_replace`：保留旧版变体；上下环相位 0° 的普通钨球替换为圆柱，因此每环为 1 个圆柱加 5 个小球，即总计 2 个圆柱和 10 个小球。
- `tungsten_bead_double_ring_marker`：上下环可使用不同直径，下环可设置相位差，并可在上环第 0 颗小球的轴向上方放置一个独立大球；可选的第二标记球使用独立的方位角和轴向高度。额外参数依次为上环直径、下环直径、圆间距、小球直径、下环相位差、大球直径、大球高于上环的距离、支撑直径、支撑长度、每圈珠子数，以及可选的第二标记球方位角和高于上环的距离。
- `tungsten_bead_spiral_marker`：钨珠按固定半径、轴向层距和相位步长形成螺旋；在最下方正常球的正下方放置一个大标记球。额外参数依次为螺旋半径、层距、相位步长、小球直径、大球直径、大球低于最下球的距离和小球数量。

生成的 `<名称>.raw.toml` 记录 RAW 布局、尺寸、体素尺寸、插入物位置及
`[[projection.materials]]`。将其中材料表复制到投影配置，并把
`projection.label_volume` 指向生成的 RAW 文件即可。

钨丝和金片属于亚毫米质量评测靶，应使用独立的小视野微体素网格。生成时
`voxel_mm` 必须不大于 0.025 mm。细丝和薄片质量评测推荐使用 0.0005 mm，
使 0.05 mm 的目标特征跨约 101 个体素，减小圆截面和倾斜方向的体素锯齿；
否则工具会拒绝输出，
避免圆丝、斜丝和薄片出现严重欠采样及阶梯失真。

低对比模体的 `voxel_mm` 必须不大于 0.25 mm，推荐使用 0.1 mm。推荐值下
最小 3 mm 插入物的直径覆盖 30 个体素，可以显著减轻圆柱边缘的锯齿。

直线钨球模体建议使用 `voxel_mm=0.125` mm，球径分别覆盖 16 和 24 个体素。
输出 TOML 的 `[bead_line]` 记录球径、球心间距、PMMA 支撑直径和长度；钨球标签为 100、PMMA 标签为 1；两种模体均沿
圆柱纵向（Z 轴）生成单排球面钨珠，球心位于圆柱轴线上。

## 验证

```powershell
ctest --test-dir out/build/phantom-generator --output-on-failure
```
