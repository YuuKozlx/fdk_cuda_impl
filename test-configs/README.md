# TOML 重建与投影测试

配置目录只保留四种几何的重建矩阵和一份正投仿真入口。所有配置默认使用
内置 Catphan，输出写入 `test-configs/output/`。

| 文件 | 几何 | 当前覆盖 |
|---|---|---|
| `reconstruction-static-flat.toml` | 静态平板 CT | FDK、C-FDK、xFDK、SIRT、SART、OS-SART、CGLS、TIGRE local、PWLS、FDK+OS-SART、FDK+CGLS |
| `reconstruction-static-cyl.toml` | 静态柱面 CT | Cyl-FDK 近似、SIRT、SART、OS-SART、CGLS、FDK+OS-SART |
| `reconstruction-heli-flat.toml` | 螺旋平板 CT | wFBP、SIRT、SART、OS-SART、CGLS、PWLS |
| `reconstruction-heli-cyl.toml` | 螺旋柱面 CT | SIRT、SART、OS-SART、CGLS、PWLS |
| `forward-projection.toml` | 四种几何 | 静态 Flat、静态 Cyl、HeliFlat、HeliCyl 的 Catphan 正投 |

## 构建要求

静态重建和正投可以使用普通构建：

```powershell
cmake --build build-windows --config Release --target ykcbct_manual_tests --parallel 8
```

HeliFlat、HeliCyl 的 wFBP 和迭代入口需要启用螺旋模块：

```powershell
cmake -S . -B build-helical -DYKCBCT_BUILD_HELICAL=ON
cmake --build build-helical --config Release --target ykcbct_manual_tests --parallel 8
```

## 运行

```powershell
build-windows/Yktest/Release/ykcbct_manual_tests.exe --config test-configs/reconstruction-static-flat.toml
build-windows/Yktest/Release/ykcbct_manual_tests.exe --config test-configs/reconstruction-static-cyl.toml
build-helical/Yktest/Release/ykcbct_manual_tests.exe --config test-configs/reconstruction-heli-flat.toml
build-helical/Yktest/Release/ykcbct_manual_tests.exe --config test-configs/reconstruction-heli-cyl.toml
build-helical/Yktest/Release/ykcbct_manual_tests.exe --config test-configs/forward-projection.toml
```

使用 `--case` 可只运行一个 case：

```powershell
build-windows/Yktest/Release/ykcbct_manual_tests.exe `
  --config test-configs/forward-projection.toml `
  --case catphan-heli-cyl
```

## 正投输出

`forward-projection.toml` 的四个 case 分别输出：

```text
output/forward-projection/static-flat.raw
output/forward-projection/static-cyl.raw
output/forward-projection/heli-flat.raw
output/forward-projection/heli-cyl.raw
```

数据布局统一为连续的 `[view][row][channel]` 单精度浮点数组。

## 配置任务和输入

`reconstruction-static-flat.toml` 使用 `task = "reconstruction"`，支持
Flat 的 FDK、C-FDK、xFDK、SIRT、SART、OS-SART、CGLS、TIGRE 和 PWLS。
`reconstruction-static-cyl.toml` 和 `reconstruction-heli-cyl.toml` 使用
`task = "cylindrical-reconstruction"`，Cyl 迭代路径直接消费
`SCylConeProjGeomVec`；Cyl 解析路径只在自身支持的静态几何约束下运行。

`reconstruction-heli-flat.toml` 使用 `task = "helical-reconstruction"`，
解析分支为 wFBP，迭代分支为 SIRT/SART/OS-SART/CGLS/PWLS。Heli 的迭代
重建不会把逐视图 z 位移退回成圆轨迹。

四种几何的正投统一使用 `forward-projection.toml`。`input.mode =
"phantom"` 由测试程序生成 Catphan；Flat 重建和 wFBP 还可以使用
`input.mode = "projection-raw"` 读取外部 `[view][row][channel]` 投影，
并通过 `--case` 选择单个 case。Cyl 重建配置目前只接受内置 Catphan，
以保证几何和投影由同一个 builder 生成。

算法选择示例：

```toml
reconstructor = "cgls"       # HeliFlat
[case.cgls]
iterations = 10
back = "joseph-v3"

pipeline = "pwls"             # HeliCyl
[case.pwls]
iterations = 5
data_model = "joseph-matched"
projection_weight_scale = 1.0
```

相对路径均以 TOML 文件所在目录为基准解析，因此从 IDE 或命令行启动时
不会因当前工作目录不同而读取不同数据。输出 raw/BMP 属于测试产物，
建议放在 `test-configs/output/` 或单独的外部输出目录，不要提交到 Git。
