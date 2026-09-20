# 多能谱 CBCT 仿真示例

本例通过 `YKCBCT` 公共 DLL 完成材料路径正投影。TOML 指定标签模体、材料（标签、化学式、密度）、能谱、XCOM 数据目录和四类几何名称：`flat_cbct`、`flat_helical`、`cyl_cbct`、`cyl_helical`。几何 offset 等系统参数由公共 geometry builder 生成。

当前已实现标签/材料到 XCOM 质量衰减系数，再到多能量透射积分和逐视图 CBCT 投影输出的闭环。投影按照 `view、v、u` 顺序写入 float32 二进制文件；`runToFile()` 按视图流式写盘，避免完整投影长期驻留内存。路径长度单位是 cm，质量衰减系数单位是 cm²/g；每个材料的线性系数为 `density * mu_over_rho`。

配置中的 `path_cache_file` 指定材料路径积分缓存文件。程序采用两阶段流式流程：第一阶段逐视图读取标签体并写入缓存，第二阶段逐视图读取缓存、执行多能谱积分并写出最终投影，因此不会同时在内存中保存完整投影或全部视图的材料路径。

缓存布局为：固定头 `PathCacheHeader`，随后按视图排列；每个视图依次存储每种材料的一张 `detector_v × detector_u` `float32` 路径积分图。头部包含 magic、版本、视图数、探测器尺寸和材料数，读取时会严格校验这些字段。缓存是中间产物，可以在后续实现 GPU 路径积分时直接复用。

当前已实现的物理过程是材料路径积分、能谱指数衰减和常数探测器效率。散射、电子/光学串扰、余晖和电子噪声保留配置字段与扩展位置，但尚未宣称已经生效。

在顶层工程开启 `BUILD_MULTISPECTRUM_SIM=ON` 并同时设置 CMake 选项 `MULTISPECTRUM_USE_LIBRARY_FP=ON` 时，配置中的 `use_library_fp = true` 会链接 `YKCBCT` 的公开导入库，并在运行时调用 `YKCBCT.dll` 的 `ReconstructionSessionFactory` 和 `IReconstructionSession`。适配器为每种材料构造二值标签体，通过公共 `ForwardProjection` Session 生成路径投影，再写入同样的路径缓存；它不访问 `src` 内部 kernel、geometry 或内存类。Flat/Cyl、Circular/Helical 四种组合均使用同一套 `SSystemSpec` 构造。适配器使用 Host buffer，CUDA 工作区由 DLL 管理。独立构建或未开启该 CMake 选项时，启用配置会明确报错，不会静默退回 CPU。默认关闭该选项，保证常规示例构建稳定。

探测器效率、散射、光学串扰、余晖、电子噪声均已保留配置字段。当前仅效率常数参与积分，其余字段明确为关闭占位，后续通过可组合效果接口实现。

构建：`cmake -S example/multispectrum_sim -B out/multispectrum_sim`，然后 `cmake --build out/multispectrum_sim --config Release`。相对路径以 TOML 所在目录为基准。能谱 CSV 每个有效数据行由 `energy_keV,relative_photons` 构成，读取器接受任意正数个采样点，不固定为 150。

## BrainWeb 测试数据

BrainWeb 标签体不随仓库提交。新机器上先运行仓库内的下载脚本，脚本会在示例目录下创建与平台无关的输入目录，并校验官方压缩包和解压后标签体的 SHA256：

```powershell
python example/multispectrum_sim/scripts/download_brainweb.py
```

脚本默认准备完整的 `181 x 217 x 181` 标签体，同时从中裁剪中心 33 层回归测试输入。需要重新下载时加 `--force`；只准备完整体积时加 `--no-center33`。BrainWeb 服务器若要求登记信息，可直接编辑脚本中的请求参数或通过已有网络环境下载后放入 `example/multispectrum_sim/inputs/brainweb/source/`。

BrainWeb TOML 使用相对于配置文件的路径，输出统一写入 `example/multispectrum_sim/outputs/`。因此复制仓库并完成上述数据准备后，无需修改 `D:/...`、`C:/...` 等本机路径即可运行，例如：

```powershell
cmake -S . -B out/brainweb -G "Visual Studio 17 2022" -A x64 -DBUILD_MULTISPECTRUM_SIM=ON -DMULTISPECTRUM_USE_LIBRARY_FP=ON -DYKCBCT_BUILD_HELICAL=ON
cmake --build out/brainweb --config Release --target multispectrum_sim
out/brainweb/example/multispectrum_sim/Release/multispectrum_sim.exe example/multispectrum_sim/configs/brainweb-center33-small-cone-fdk.toml
out/brainweb/example/multispectrum_sim/Release/multispectrum_sim.exe example/multispectrum_sim/configs/brainweb-center33-small-cone-heli-cyl-wfbp.toml
```

示例文件按职责分开保存：

- `configs/`：仅 TOML 运行配置；
- `spectra/`：原始能谱和转换后的 CSV；
- `inputs/`：外部标签体；
- `outputs/path-cache/`：材料路径积分缓存；
- `outputs/projections/`：最终 `-log(I/I0)` 投影；
- `outputs/reconstructions/`：float32 重建体；
- `outputs/images/`：用于观察的中心切片图。

普通调用会从 TOML 的 `simulation.label_volume` 读取外部 `uint8` 标签体，布局为
`[z][y][x]`，文件元素数必须严格等于 `volume_x * volume_y * volume_z`。内部水模只是一种
测试输入，也可先导出后再按外部体积流程运行：

```powershell
multispectrum_sim --export-water-cylinder configs/water-cylinder-120kv-fullfov.toml `
  inputs/water-cylinder-512x512x256-u8.raw
multispectrum_sim configs/water-cylinder-120kv-fullfov.toml
```

## CUDA 多能谱积分

顶层 DLL 构建支持 `simulation.use_cuda_spectral = true`。CUDA 积分器让任意长度的
能谱权重和各材料 `density * mu_over_rho` 表常驻显存，每次只上传当前视图的
`[material][v][u]` 路径并回传一张投影，不缓存完整扫描。配置关闭该开关时仍使用 CPU
实现，便于回归比较；独立无 CUDA 构建若开启该选项会明确报错。

`water-cylinder-120kv-fullfov.toml` 使用从原始 `.spc` 转换得到的 150 点 120 kV 谱。
150 不是程序限制，CSV 中任意正数个有效 `(energy_keV, relative_photons)` 数据行均可读取。

XCOM 核心及 `data/MDATX3.*` 来自 `nist-xcom-portable`，保留 GPL-3.0-or-later 和第三方声明。

## Matplot++ 分析工具（可选）

仓库提供一个可选的 Matplot++ 分析工具，用于查看 float32 重建体的 z 向均值、中心轴曲线和数值直方图。当前 Matplot++ 使用 Gnuplot 后端，因此除了 vcpkg 的 `matplotplusplus` 库，还需要 `gnuplot` 可执行程序，并将其加入 `PATH`。它不属于 DLL 或默认测试依赖，只有启用 vcpkg 的 `plotting` feature 和 CMake 选项时才构建：

```powershell
winget install --id gnuplot.gnuplot --scope user
cmake --preset x64-Release-plotting
cmake --build out/build/x64-Release-plotting --config Release --target multispectrum_plot_reconstruction
out/build/x64-Release-plotting/example/multispectrum_sim/plot/multispectrum_plot_reconstruction.exe `
  --input example/multispectrum_sim/outputs/reconstructions/volume.raw `
  --size 512 512 256 --output volume-profile.png
```

启用该选项时，CMake 通过 `find_package(Matplot++ CONFIG REQUIRED)` 查找 vcpkg 提供的 `Matplot++::matplot` 目标，并通过 `find_program(gnuplot)` 检查运行时后端；任一依赖缺失都会在配置阶段明确失败，不会影响默认构建。Matplot++ 只负责绘图 API，真正生成 PNG 的是 Gnuplot 子进程。

## 典型水模测试

仓库只保留 `configs/water-cylinder-120kv-fullfov.toml` 这一份完整示例。它使用
`512 x 512 x 256` 标签体、`1024 x 1024` 平板探测器、360 个视图和 120 kV 能谱，
随后通过 DLL 的 Flat Circular FDK 重建 `512 x 512 x 256` 体积。投影、路径缓存、
重建体和 BMP 均写入 `outputs/`，该目录不纳入版本控制。

### 几何偏移参数

`[geometry_config]` 中的偏移均以 mm 表示：`phantom_offset_x/y/z_mm` 定义标签模体
中心，`reconstruction_offset_x/y/z_mm` 定义重建网格中心，二者分别只作用于正投和重建。
`source_offset_x/y/z_mm` 定义焦点相对标称轨迹的扫描架局部偏移，该偏移随旋转架一起旋转。探测器支持局部
`U/N/V` 三方向偏移（`offset_u_mm`、`offset_n_mm`、`offset_v_mm`）；探测器姿态偏转
暂不在仿真配置中开放，因为不同投影/重建模型对偏转的支持范围不同。`views_per_turn`
和 `rotation_direction` 控制规则角度采样；这些值最终交给公共 geometry builder，正投和
重建不会各自解释一份几何。

多能谱 TOML 保留 `reconstruction.pipeline` 字段。材料路径积分固定调用公共
`ForwardProjection`；示例接入 `flat_cbct + fdk` 和 `cyl_helical + wfbp`。
其他组合在配置读取时拒绝。BrainWeb 示例要求顶层 DLL 构建，独立 CPU 构建不能运行这些配置。
FDK 的 filter 支持 ramlak/ram-lak、shepp-logan、cosine、hann、hamming；拼写错误直接拒绝。
wFBP 的 filter 仅接受 ramlak/ram-lak，它表示 FreeCT ramp 核；使用
`wfbp_cutoff`（默认 1，范围 (0,1]）和 `wfbp_apodization`（默认 1，范围 [0,1]）调节核。
这些参数在 FDK 配置中会被拒绝，运行日志打印实际选择。

wFBP 仍为全量执行，`chunk_views` 对它无效。示例提交 Host 投影和 Host 体积，
Session 用库内 MemoryController 管理设备暂存，并在返回前完成下载。
日志中的 input/volume 字节数不包含重排、滤波等算法工作区；暂存分配前检查剩余显存。
此检查不是完整峰值显存保证，底层算法工作区仍在初始化时分配。
WDDM 空闲物理显存不足时仅告警，因为驱动可能通过分页满足分配。
底层分配器仍使用项目既有 CUDA 错误宏，实际分配失败的处理尚非完整可恢复异常契约。
已有投影可通过 `multispectrum_sim --reconstruct-only <config.toml>` 单独重建，
不重新计算材料路径与多能积分；使用者须保证配置与已有投影的几何相符（入口校验文件尺寸）。

### 材料 ROI 回归

分析脚本需要 Python 3.11+ 和 numpy；路径均以输入 TOML 所在目录解析。

```powershell
python -m pip install numpy
python example/multispectrum_sim/scripts/report_material_roi.py example/multispectrum_sim/configs/brainweb-center33-small-cone-fdk.toml --contrast-labels 2 3
python example/multispectrum_sim/scripts/report_material_roi.py example/multispectrum_sim/configs/brainweb-center33-small-cone-heli-cyl-wfbp.toml --contrast-labels 2 3
```

默认标签 ROI 腐蚀一个体素（26 邻域），排除首尾四层；输出 RAW 同目录的 `.roi.json`，
包括样本数、均值、标准差和可选标签均值差，单位 mm^-1。不同模体/重建 offset 会拒绝分析，
防止错位标签统计。使用 `--reference reference.json` 可检查显式材料基准与绝对容差：

```json
{"labels":{"2":{"mean_mm_inv":0.021,"tolerance_mm_inv":0.001}}}
```

以上数字仅示范格式，不代表灰质真值。超差退出码为 1；无参考值时 reference_passed 为 null。
`tests/reference-brainweb-center33-fdk.json` 与 `tests/reference-brainweb-center33-wfbp.json`
保存本次对应 TOML 的 CSF/灰质/白质/骨组织回归基准，绝对容差为 0.00001 mm^-1；
可分别传给 `--reference`。它们用于发现实现变化，不是物理正确性定标；修改能谱、几何或材料后不能沿用。
多能谱结果不能直接当作某个单能的衰减真值。BrainWeb 整层均值受组织面积影响，不能作为
z 均匀性指标；水模可加 `--fixed-radius-mm 20`，生成固定中心圆形 ROI 的 z 均值曲线、
标准差及极差。脚本要求这个 ROI 在保留的所有 z 层中只有一种材料，否则拒绝统计。
内部水模须先用 `--export-water-cylinder` 导出标签体，再通过报告脚本的
`--labels <labels.raw>` 指定该标签文件（此命令行路径相对于当前目录）。

offset 的处理按算法能力执行：Flat FDK、FP 和迭代算子保留配置中的 offset；XFDK、
C-FDK、Cyl Analytic FDK、wFBP 等要求规范采集几何的管线，在 DLL Session 初始化时将
后端几何副本中不支持的源端/探测器 offset 分量强制置为 0，并通过日志输出中文提示。
wFBP 保留其支持的 U/V 主点 offset，仅清除源端和 N 分量。调用方配置、正投影几何和
体积中心 offset 均不改变。探测器 tilt 仍未在 TOML 中开放。

标签体不随仓库提交，可先由同一个可执行文件生成，再按外部体积输入流程运行：

```powershell
out/top-multispectrum-dll-vs/example/multispectrum_sim/Release/multispectrum_sim.exe `
  --export-water-cylinder example/multispectrum_sim/configs/water-cylinder-120kv-fullfov.toml `
  example/multispectrum_sim/inputs/water-cylinder-512x512x256-u8.raw

out/top-multispectrum-dll-vs/example/multispectrum_sim/Release/multispectrum_sim.exe `
  example/multispectrum_sim/configs/water-cylinder-120kv-fullfov.toml
```

也可用 `--water-cylinder` 直接在内存中生成纯水圆柱。该模式以横向短边的 40% 作为
圆柱半径，并使用 `name = "water"` 或 `formula = "H2O"` 对应的材料标签。

## 内置人体材料

材料可以继续使用 `formula`，也可以通过 `preset` 引用仓库内置材料。内置表保存
密度和元素质量分数，查询 XCOM 时使用质量分数混合，适合不能用单一化学式准确
描述的人体组织。

当前内置表包含 MC-GPU 材料目录中的全部 202 个材料，以及 GateMaterials.db 的
全部 62 个材料，共 264 条。内容覆盖元素、常用化合物、探测器材料、ICRU/CIRS/NCAT 人体组织、含碘
血液和水溶液等。完整列表以 `multispectrum_sim --list-materials` 的输出为准。

```toml
[[materials]]
label = 2
name = "brain"
preset = "brain"
```

预设密度来自 Gate/MC-GPU 中的组织材料数据。省略 `density_g_cm3` 时自动使用预设
密度；显式填写时允许覆盖预设密度，但仍必须为正数。这样可以在保持相同元素配比的
情况下模拟灰质/白质等低对比度材料。自定义化合物继续使用 `formula`：

```toml
[[materials]]
label = 3
name = "iodine"
formula = "I"
density_g_cm3 = 4.93
```

同一材料必须且只能填写 `preset` 或 `formula` 之一。

可在不加载配置文件的情况下查询内置表：

```powershell
multispectrum_sim --list-materials
multispectrum_sim --material-info tissue.brain
multispectrum_sim --material-info Al
```

规范标识按材料语义分组：纯元素为 `element.*`，明确化合物为 `compound.*`，混合物为
`mixture.*`，人体复杂组织为 `tissue.*`，Gate 来源为 `gate.*`。TOML 推荐使用规范
标识；短名称和元素符号只作为输入别名保留。

内置表由以下命令从已授权材料源生成，不在运行时读取外部材料目录：

```powershell
powershell -ExecutionPolicy Bypass -File tools/generate_builtin_materials.ps1 `
  -InputDirectory D:\Work\Tool\mc-gpu\MCGPU_materials\material `
  -OutputFile example/multispectrum_sim/src/BuiltinMaterials.inc

powershell -ExecutionPolicy Bypass -File tools/generate_gate_materials.ps1 `
  -InputFile thirdparty/nist_xcom/data/GateMaterials.db `
  -OutputFile example/multispectrum_sim/src/BuiltinGateMaterials.inc
```

例如，灰质和白质可以共用脑组织配比，只通过密度产生低对比度：

```toml
[[materials]]
label = 10
name = "gray_matter"
preset = "ICRU_brain_adult"
density_g_cm3 = 1.040

[[materials]]
label = 11
name = "white_matter"
preset = "ICRU_brain_adult"
density_g_cm3 = 1.050
```

接口组合回归、BrainWeb 数值基线与水模固定 ROI 纵向对比，见
[重建验证记录](tests/VALIDATION.zh-CN.md)。记录包含复现命令、结果及尚未消除的数值偏差。

示例输出包括 `[view][v][u]` 排列的 float32 投影、材料路径缓存、`[z][y][x]`
排列的 float32 FDK 重建体，以及轴位、冠状位和矢状位中心切片 BMP。BMP 仅用于观察，
数值分析应读取重建体 raw。
