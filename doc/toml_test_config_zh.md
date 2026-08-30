# TOML 测试配置说明

Yktest 使用 toml++ 解析配置。程序入口为：

```text
ykcbct_manual_tests --config <file.toml>
ykcbct_manual_tests --config <file.toml> --case <case-name>
```

相对路径相对于 TOML 文件所在目录解析，不依赖 IDE 的工作目录。

## 任务类型

| `task` | 入口 | 几何 | 支持内容 |
|---|---|---|---|
| `reconstruction` | `YkConfiguredReconstruction` | Flat 圆轨迹 | FDK、C-FDK、xFDK、SIRT、SART、OS-SART、CGLS、TIGRE、PWLS |
| `cylindrical-reconstruction` | `YkConfiguredCylReconstruction` | Cyl 静态/螺旋 | Cyl-FDK（静态约束）、SIRT、SART、OS-SART、CGLS、PWLS |
| `helical-reconstruction` | `YkConfiguredWfbp` | Flat 螺旋 | wFBP、SIRT、SART、OS-SART、CGLS、PWLS |
| `forward-projection` | `YkConfiguredForwardProjection` | Flat/Cyl 静态和螺旋 | 正投仿真 |

Heli 解析和迭代是两条独立路径。wFBP 使用 `Wfbp::InputGeometry`；Flat
迭代使用 `Helical::Iterative::FlatReconstructor`；Cyl 迭代使用
`Helical::Iterative::CylReconstructor`。两种迭代入口都接收 builder 生成
的完整逐视图 geometry，不根据标称角度重新生成圆轨迹。

## 几何字段

```toml
[case.scan]
nu = 48
nv = 32
views = 64
du_mm = 1.0
dv_mm = 1.0
sid_mm = 80.0
sdd_mm = 160.0
scan_range_deg = 720.0

[case.volume]
nx = 32
ny = 32
nz = 24
voxel_mm = [1.0, 1.0, 1.0]

[case.geometry]
pitch_mm = 8.0          # 0 表示静态圆轨迹
start_z_mm = -8.0
views_per_rotation = 32
curvature_radius_mm = 160.0  # Cyl；迭代不要求等于 SDD
```

Flat/Cyl 的几何结构不同，不能互换 `geometry` 的探测器字段。生产 builder
位于 `include/YKCBCT/geometry/`；测试辅助转换位于 `Yktest/YkTestGeometry.hpp`。

## 输入和输出

内置模体：

```toml
[case.input]
mode = "phantom"
phantom = "catphan"
```

Flat 重建和 wFBP 可读取外部投影：

```toml
[case.input]
mode = "projection-raw"
projection = "data/projection.raw"
```

投影必须是连续的 float32 `[view][row][channel]` 数组，元素数量为
`views * nv * nu`。Cyl 配置化重建当前只接受内置 Catphan，因此不会发生
外部投影几何与 Cyl builder 不一致的问题。

输出字段：

```toml
[case.output]
volume = "output/reconstruction.raw"
preview = "output/reconstruction.bmp"
```

## 迭代参数

Flat Heli 使用对应的算法表：`[case.sirt]`、`[case.sart]`、
`[case.ossart]`、`[case.cgls]`、`[case.pwls]`。Cyl Heli 使用
`[case.algorithm]` 选择 FP/BP 与 OS 子集，PWLS 参数放在 `[case.pwls]`。

Cyl PWLS 的 `projection_weight_scale` 默认是 1；不提供逐射线权重时，
目标函数中的 W 就是单位阵。`data_model = "joseph-matched"` 是高伴随性
参考路径，`data_model = "siddon"` 是严格路径长度但较慢的路径。

## 构建

Heli 测试必须在配置阶段打开：

```powershell
cmake -S . -B build-helical -DYKCBCT_BUILD_HELICAL=ON -DBUILD_TESTS=ON
cmake --build build-helical --config Release --target ykcbct_manual_tests --parallel 8
```
