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

## 验证

```powershell
ctest --test-dir out/build/phantom-generator --output-on-failure
```
