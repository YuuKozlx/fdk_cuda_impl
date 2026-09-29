# 多能谱 CBCT 仿真示例

本例使用独立 CUDA kernel 在线完成多能谱前投，不生成材料路径缓存，也不通过 DLL 做 FP。DLL 只作为 FDK、wFBP 和迭代算法的重建后端。

重建管线支持 `fdk`、`wfbp`，以及 DLL 已放行的迭代管线 `tigre_sart`、`tigre_sirt`、`tigre_os_sart`、`tigre_sart_tv`、`tigre_os_sart_tv`，以及原有的 `sirt`、`ossart`、`cgls`。解析重建和迭代重建分别放在 `reconstruction.analytic` 与 `reconstruction.iterative` 下，避免参数混用。迭代管线必须一次提交完整投影，不使用 `chunk_views` 分包；它们使用显存中的投影/体缓冲，程序负责 Host 与 Device 之间的拷贝。迭代参数示例：

```toml
[reconstruction]
enabled = true
output_volume_file = "../outputs/reconstructions/iterative.raw"
slice_prefix = "../outputs/images/iterative"
save_slices = false        # 默认不导出轴向、冠状和矢状 BMP

[reconstruction.iterative]
algorithm = "tigre_os_sart_tv"
iterations = 20
relaxation = 0.8
subsets = 8               # sirt/cgls 可保持 1
forward_projector = "joseph" # joseph / siddon
back_projector = "joseph_v3" # joseph / joseph_v3 / siddon / siddon_v2 / siddon_v3
tv_iterations = 20
tv_alpha = 0.002
tv_alpha_reduction = 0.95
maximum_update_ratio = 0.95
non_negative = true
```

TIGRE 梯度族使用 `tigre_` 前缀：普通迭代为 `tigre_sart`、`tigre_sirt`、`tigre_os_sart`；TV 版本为 `tigre_sart_tv`、`tigre_os_sart_tv`，且 `tv_alpha` 必须大于 0。

迭代重建目前支持四类宏观几何；实际组合仍由 DLL 的公开能力校验，若某个构建未放行会在初始化时报错，不会静默退回 FDK。

## 能量积分探测器输出

纯正投不生成投影预览图片。重建的轴向、冠状和矢状 BMP 默认关闭，只有显式
设置 `reconstruction.save_slices = true` 才会生成。

CSV 第二列按光子数谱解释。CUDA 多能谱积分对每个能量 bin 计算“光子数 * energy_keV * exp(-路径衰减)”，并同时写出两种 float32 投影：

- simulation.output_file：以同一像素空场能量归一化后的 -log(E/E0) 衰减域投影，供 FDK 和迭代重建使用；几何通量因子在空场归一化中抵消。
- projection.energy_output_file：未做空场归一化的相对能量积分信号。平板探测器按实际源点和像素位置应用 cos(theta)/r^2，并考虑 source_offset_x/y/z_mm 与 offset_u/n/v_mm。系数相对探测器主点归一化，因此结果是相对能量信号，不是绝对剂量。

`projection.apply_geometry_flux = true` 控制上述平板几何通量修正。能量积分输出由独立 CUDA 前投直接产生，不需要额外开关。
当前已实现标签/材料到 XCOM 质量衰减系数，再到多能量透射积分和逐视图 CBCT 投影输出的闭环。投影按照 `view、v、u` 顺序写入 float32 二进制文件；`runToFile()` 按视图流式写盘，避免完整投影长期驻留内存。路径长度单位是 cm，质量衰减系数单位是 cm²/g；每个材料的线性系数为 `density * mu_over_rho`。

在线前投直接完成焦点与探测器位置采样、射线与体素盒求交、材料路径累计和能谱积分，并逐视图写出能量域与对数投影，不保存中间路径表。

当前已实现的物理过程是材料路径积分、能谱指数衰减和常数探测器效率。散射、电子/光学串扰、余晖和电子噪声保留配置字段与扩展位置，但尚未宣称已经生效。

顶层工程开启 `BUILD_MULTISPECTRUM_SIM=ON` 后会编译在线 CUDA 前投；链接 YKCBCT 时同时提供重建能力。配置中不再区分 DLL FP 或 CPU/CUDA 能谱积分。

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
cmake -S . -B out/brainweb -G "Visual Studio 17 2022" -A x64 -DBUILD_MULTISPECTRUM_SIM=ON -DYKCBCT_BUILD_HELICAL=ON
cmake --build out/brainweb --config Release --target multispectrum_sim
out/brainweb/example/multispectrum_sim/Release/multispectrum_sim.exe example/multispectrum_sim/configs/pipelines/analytic/brainweb-center33-fdk.toml
out/brainweb/example/multispectrum_sim/Release/multispectrum_sim.exe example/multispectrum_sim/configs/pipelines/analytic/brainweb-center33-heli-cyl-wfbp.toml
```

示例文件按职责分开保存：

- `configs/projection/`：纯投影配置；`configs/reconstruction/analytic/` 和 `configs/reconstruction/iterative/`：读取已有投影的重建配置；`configs/pipelines/analytic/` 和 `configs/pipelines/iterative/`：投影后立即重建的组合配置；`configs/quality/`：图像质量模体配置；`configs/calibration/geometry/`：双环、螺旋等几何标定模体；`configs/calibration/reconstruction/`：水模重建数值校验；
- `spectra/`：原始能谱和转换后的 CSV；
- `inputs/`：外部标签体；
- `outputs/projections/`：最终 `-log(I/I0)` 投影；
- `outputs/reconstructions/`：float32 重建体；
- `outputs/images/`：用于观察的中心切片图。

普通调用会从 TOML 的 `projection.label_volume` 读取外部 `uint8` 标签体，布局为
`[z][y][x]`，文件元素数必须严格等于 `volume_x * volume_y * volume_z`。模体生成不属于
本程序职责；水模或其他标签体应由独立模体项目生成后作为输入提供。

## CUDA 多能谱积分

`pixel_local_random` 在每个探测器像素内随机采样，`detector_global_random` 在整个探测器面上随机采样。两种路线均直接在 CUDA 前投 kernel 中计算并生成能量积分信号和对数投影；`photon_count_mode` 可选择 `fixed` 或 `poisson`。

`configs/pipelines/analytic/water-cylinder-120kv-fullfov.toml` 使用从原始 `.spc` 转换得到的 150 点 120 kV 谱。
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

仓库只保留 `configs/pipelines/analytic/water-cylinder-120kv-fullfov.toml` 这一份完整示例。它使用
`512 x 512 x 256` 标签体、`1024 x 1024` 平板探测器、360 个视图和 120 kV 能谱，
随后可使用 `configs/reconstruction/analytic/brainweb-center33-fdk.toml` 这类独立配置调用 DLL 解析重建。投影、
重建体和 BMP 均写入 `outputs/`，该目录不纳入版本控制。

### 几何偏移参数

`[geometry]` 中的偏移均以 mm 表示：`phantom_offset_x/y/z_mm` 定义标签模体
中心，`reconstruction_offset_x/y/z_mm` 定义重建网格中心，二者分别只作用于正投和重建。
`source_offset_x/y/z_mm` 定义焦点相对标称轨迹的扫描架局部偏移，该偏移随旋转架一起旋转。探测器支持局部
`U/N/V` 三方向偏移（`offset_u_mm`、`offset_n_mm`、`offset_v_mm`）；探测器姿态偏转
暂不在仿真配置中开放，因为不同投影/重建模型对偏转的支持范围不同。`views_per_turn`
和 `rotation_direction` 控制规则角度采样；这些值最终交给公共 geometry builder，正投和
重建不会各自解释一份几何。

多能谱 TOML 将解析重建和迭代重建分开配置：在 `[reconstruction]` 中用 `type = "analytic"`
或 `type = "iterative"` 选择类别，再分别填写 `[reconstruction.analytic]` 或
`[reconstruction.iterative]`。材料路径积分固定调用公共 `ForwardProjection`；示例接入
`flat_cbct + fdk` 和 `cyl_helical + wfbp`，迭代重建支持 `sart_tv` 和 `os_sart_tv`。
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
已有投影可使用 `workflow.mode = "reconstruct"` 的配置直接单独重建，
不重新计算材料路径与多能积分；使用者须保证配置与已有投影的几何相符（入口校验文件尺寸）。

### 材料 ROI 回归

分析脚本需要 Python 3.11+ 和 numpy；路径均以输入 TOML 所在目录解析。

```powershell
python -m pip install numpy
python example/multispectrum_sim/scripts/report_material_roi.py example/multispectrum_sim/configs/pipelines/analytic/brainweb-center33-fdk.toml --contrast-labels 2 3
python example/multispectrum_sim/scripts/report_material_roi.py example/multispectrum_sim/configs/pipelines/analytic/brainweb-center33-heli-cyl-wfbp.toml --contrast-labels 2 3
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
水模标签可通过报告脚本的 `--labels <labels.raw>` 显式指定（此命令行路径相对于当前目录）。

offset 的处理按算法能力执行：Flat FDK、FP 和迭代算子保留配置中的 offset；XFDK、
C-FDK、Cyl Analytic FDK、wFBP 等要求规范采集几何的管线，在 DLL Session 初始化时将
后端几何副本中不支持的源端/探测器 offset 分量强制置为 0，并通过日志输出中文提示。
wFBP 保留其支持的 U/V 主点 offset，仅清除源端和 N 分量。调用方配置、正投影几何和
体积中心 offset 均不改变。探测器 tilt 仍未在 TOML 中开放。

标签体不随仓库提交，需由独立模体项目生成，再按外部体积输入流程运行：

```powershell
out/top-multispectrum-dll-vs/example/multispectrum_sim/Release/multispectrum_sim.exe `
  example/multispectrum_sim/configs/pipelines/analytic/water-cylinder-120kv-fullfov.toml
```

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

示例输出包括 `[view][v][u]` 排列的 float32 投影、`[z][y][x]`
排列的 float32 FDK 重建体，以及轴位、冠状位和矢状位中心切片 BMP。BMP 仅用于观察，
数值分析应读取重建体 raw。
