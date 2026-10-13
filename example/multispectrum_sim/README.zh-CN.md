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

投影配置将几何通量、量子噪声、探测器响应和焦点散焦分开控制：

- `[projection.geometry_flux].enabled`：启用按每条射线源到探测器距离平方反比的通量修正；像素面积已用于谱通量换算。
- `[projection.quantum_noise].enabled`：对能量 bin 的光子计数施加泊松采样；`seed` 只控制可复现性。
- `[projection.detector_response].enabled`：启用保护层、杂质层和闪烁体层的探测器吸收效率响应，不包含光学/电子串扰。
- `[projection.focal_spot]`：`normal_size_u/v_mm` 和 `defocus_size_u/v_mm` 分别定义正常焦点与散焦焦点尺寸；`epsilon` 是分配给散焦分量的曝光比例，正常焦点占 `1 - epsilon`。例如 `epsilon = 0.01` 表示 1% 曝光使用散焦尺寸，不会缩小焦点尺寸。有限正常焦点和散焦同时启用时，大矩形的 U/V 尺寸均须不小于正常矩形。实际采用分层采样：正常矩形内 N 点复用于散焦内部，另采 N 个外圈点；外圈不重复采内部。令 r=正常面积/散焦面积，内部总曝光权重为 `(1-epsilon)+epsilon*r`，外圈为 `epsilon*(1-r)`，每组除以 N。散焦等效使用内部 N 加外圈 N 点，实际最多仍为 2N 条射线。两矩形等大时外圈权重为零，只计算内部 N 条；epsilon=0 时只计算正常焦点。

能量积分输出由独立 CUDA 前投直接产生。
当前已实现标签/材料到 XCOM 质量衰减系数，再到多能量透射积分和逐视图 CBCT 投影输出的闭环。投影按照 `view、v、u` 顺序写入 float32 二进制文件；`runToFile()` 按视图流式写盘，避免完整投影长期驻留内存。路径长度单位是 cm，质量衰减系数单位是 cm²/g；每个材料的线性系数为 `density * mu_over_rho`。

在线前投直接完成焦点与探测器位置采样、射线与体素盒求交、材料路径累计和能谱积分，并逐视图写出能量域与对数投影，不保存中间路径表。

当前已实现材料路径积分、多能谱指数衰减、分层探测器响应，以及可独立开关的散射、量子噪声、光学/电子串扰、余晖和 DAS 信号处理。具体组合由配置校验和后端能力检查决定。

顶层工程开启 `BUILD_MULTISPECTRUM_SIM=ON` 后会编译在线 CUDA 前投；链接 YKCBCT 时同时提供重建能力。配置中不再区分 DLL FP 或 CPU/CUDA 能谱积分。

探测器处理按流水线阶段组合，开关关闭时跳过对应阶段。能谱可从文件导入，也可由内置原生生成器产生。

构建：`cmake -S example/multispectrum_sim -B out/multispectrum_sim`，然后 `cmake --build out/multispectrum_sim --config Release`。相对路径以 TOML 所在目录为基准。能谱 谱文件为 SpekCalc 格式：表头后的每个有效数据行由 `energy_keV fluence` 构成，单位固定为 `photons/(keV cm² mAs) @ 1m`，读取器自动计算 bin 宽。

## 导入与生成能谱

不填写 `[projection.spectrum]` 时保持原行为，使用 `projection.spectrum_file`。
选择本地生成时移除 `spectrum_file`，填写：

```toml
[projection.spectrum]
source = "generated"        # file / generated，默认 file
model = "spekpy-v1"          # spekpy-v1 / spekcalc，钨反射靶
kvp = 120.0
anode_angle_deg = 12.0
bin_width_keV = 1.0
export_file = "../../outputs/spectra/generated-120kv.spec" # 可选
filters = [
    { material = "Al", thickness_mm = 2.0 },
    { material = "Cu", thickness_mm = 0.1 },
]
```

生成器来自仓库内 `thirdparty/spek_spectrum`，以静态库链接到多能前投工具。
正常构建、运行无需 Python、SciPy 或外部数值表；不增加 YKCBCT 核心 DLL 的依赖。
只生成或导出能谱可运行 `multispectrum_sim --export-spectrum <config.toml> <output.spec>`，
此命令不加载标签体、不运行投影或重建，也可使用独立 C++ 构建。实际前投仍需要 CUDA 父工程构建。
支持的过滤材料快捷名为 Be/Ti/Al/Cu/Mo/Rh/Ag/Sn/W/Au/Pb；其他元素可写为
`{ atomic_number = 13, density_g_cm3 = 2.699, thickness_mm = 2.0 }`，此时不要同时填写 `material`。
厚度单位为 mm，密度为 g/cm³；未设置过滤层表示无附加过滤。生成控制应在模型和数据适用范围内使用。

生成结果直接使用能量中心及显式分箱宽度，通量固定为
`photons/(keV cm² mAs) @ 1m`。实际曝光量由 `projection.mAs_per_view` 控制，
几何距离由前投几何控制，不再在生成参数里重复设置曝光和距离。
`export_file` 相对 TOML 所在目录解析，保存为带中心能量和分箱宽度的 `YK-SPECTRUM 1.0`，
可通过原文件导入入口重新读取，读回后积分与内存生成结果一致。
带 `Format: MCGPU-SPEC 1.0` 标记的文件按区间边界读取，最后一行必须是零通量上边界；
无该标记的旧两列文件继续按中心能量读取并推导分箱宽度。

水板硬化校正未单独指定能谱文件时，使用同一前投能谱来源，支持本地生成。
`loadProjectionSpectrum()` 供界面和命令行共用；开启 `BUILD_IMGUI_GUI` 后可在 Spectrum 页签导入、生成、预览和导出能谱。

验证：`ctest --test-dir <build-dir> -R multispectrum_spectrum_source --output-on-failure`。
覆盖两个模型的平均能量、过滤层效果、曝光归一化、中心格式读写、MCGPU 边界转换及配置校验。

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

CUDA 前投在每个探测器像素内随机采样，并直接计算初级射线能量积分信号；量子噪声在逐能量探测器吸收阶段生成。

现有配置统一使用 SpekCalc 绝对光子数谱。能量 bin 数量不固定，读取器根据严格递增的能量采样自动计算 bin 宽。

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

### 正式固定迭代去散射

默认 `scatter.solver.type = "fixed"` 使用 CUDA 常驻迭代：每帧上传测量信号和
空气场一次，散射预测、固定步长更新和逐像素正值截断均在 GPU 上完成，结束后
仅下载最终初级信号。FFT 计划、核频谱和设备缓冲区跨帧复用；每次预测仍有小型
有效性标志回读，不再逐轮传输整张投影。无需增加配置开关，也无需启用实验插件。

参数仍由 `scatter.correction.iterations`、`damping` 和 `scatter_scale` 控制。
正值下限仍为对应空气信号的 `1e-8`，没有增加自适应步长或统计降噪，低曝光截断
导致的条纹风险仍然存在。

可选的平滑引导由正式与自适应求解器共用：

```toml
[reconstruction.preprocessing.scatter.guide]
enabled = true # 默认false
sigma_u_pixels = 2.0
sigma_v_pixels = 2.0
```

该选项在 CUDA 上平滑测量能量图和空气场，使用平滑副本估计掩膜及散射；最终
仍从原始能量图扣除预测散射，不把平滑图作为重建输入。关闭时保留原流程。
正式 `fixed` 不需要开启实验插件，次数和 damping 仍在 `scatter.correction` 中设置。
原实验性的 `solver.adaptive.guide` 配置已删除，统一使用上述共享配置。

### 实验性动态散射求解插件

固定迭代仍是默认流程。实验插件需在 CUDA 顶层构建时显式打开
`MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER=ON`，然后在
`[reconstruction.preprocessing.scatter.solver]` 中选择
`type = "adaptive_experimental"`。独立参数放在 `solver.adaptive` 下；此时
`scatter.correction` 不再填写 `iterations` 和 `damping`，仍保留散射幅度和物体掩膜阈值。

示例见 `configs/reconstruction/experimental/water-cylinder-adaptive-scatter.toml`，
算法与限制见 `plugins/adaptive_scatter/README.md`。插件在 GPU 上执行像素更新和
残差统计，在 CPU 上控制回退与固定次数；日志逐轮和逐帧报告实际迭代次数、预测次数、最终
damping、残差、信号变化量和耗时。插件不设置容差提前停止，正常运行 iterations 次
后仅报告 completed；最低安全步长仍无法下降会单独报告异常状态。插件不自动调整
gamma 或散射幅度，也不使用图像平坦度作为求解目标。

## 多能前投 Pipeline 与 PCB 模体

多能前投采用“后端 → 按物理阶段排序的信号处理 → 输出端”的流程。新增处理可实现
`IProjectionStage` 并通过 `SpectralProjectionPipeline::addStage` 注册；后端负责 CUDA
资源，处理模块负责时序与随机状态，输出端负责空气归一化和文件保存。帧数据区分能量积分
与光子计数，后者预留能量网格、计数阈值、多通道计数及空气参考，尚未实现实际计数探测器响应。
当前模块边界使用主机帧，设备端处理需另行扩展设备帧接口。

flat 焦点尺寸沿探测器局部 U/V 定义，普通几何与 ConeVec 使用同一采样函数。
空气参考按每一帧的几何和通量计算，余辉状态与物体信号独立；空气输出为
`[view][row][column]` 的 float32，元数据帧数与投影一致。`AirReferenceReader` 同时支持
旧的单帧空气文件（广播）与逐帧空气序列。

`planar_source_orbit` 与 `nikon_tilted` 轨迹支持 flat 探测器的 U/N/V 偏移和绕 U/V/N
的转角，正投和 DLL 重建消费相同 ConeVec。非标准解析重建一次提交完整视图序列；
Planar CT 会检查裁窗范围。非标准轨迹暂不支持探测器分层响应。

PCB 模体由独立工具生成。以下命令在仓库根目录执行（Linux/CUDA 容器）：

```sh
cmake -S example/phantom_generator -B out/docker-phantom-generator -G Ninja
cmake --build out/docker-phantom-generator
mkdir -p example/multispectrum_sim/inputs/generated
./out/docker-phantom-generator/phantom_generator pcb_test \
  example/multispectrum_sim/inputs/generated/pcb-test-100mm-0p1mm.raw 1000 1000 32 0.1
```

模体尺寸为 100 × 100 mm，基板厚 1.6 mm，包含 5 × 5 焊盘、走线、焊料空隙、通孔和
断线缺陷。铜层采用测试网格可表达的 0.1 mm 厚度。材料为铜、SAC305 配比焊料和玻璃主导
的 FR-4 等效材料；FR-4 的 SiO2 模型不代表完整玻纤/环氧配方。

两份组合案例位于 `configs/pipelines/analytic/pcb-nikon-tilted-fdk.toml` 和
`pcb-planar-pose-fdk.toml`。生成标签体后，将配置路径传给 `multispectrum_sim` 即可运行。

小模块测试直接调用 CPU 函数与 CUDA kernel，覆盖材料路径、多能透射、焦点、探测器响应、
噪声、串扰、余辉、DAS、空气读写和姿态映射。容器内运行：

```sh
ctest --test-dir out/docker-test/example/multispectrum_sim -L unit --output-on-failure
ctest --test-dir out/docker-test/example/multispectrum_sim -R multispectrum_projection_pipeline --output-on-failure
```
## 可选实时显示

主工程启用 `BUILD_VTK_VISUALIZATION=ON` 并提供 VTK C++ 包后，在 TOML 增加 `[visualization] enabled=true`，即可边前投边显示实际材料标签体、源/探测器姿态与偏移和当前帧投影贴图。体数据保留原分辨率，仅显示贴图可降采样；关闭显示时计算和输出不变。显示模块独立位于 `visualization/`，不依赖 geometry_viewer 工具源码。构建、配置及自生成验证见 [实时显示层说明](../../visualization/README.zh-CN.md)。

## 可选 ImGui 前投界面

顶层开启 `BUILD_VTK_VISUALIZATION=ON`、`BUILD_IMGUI_GUI=ON`，配置中设置
`[visualization] enabled=true, interactive=true`（project 工作流）。界面与 VTK 共用窗口，
第一版支持 Flat 圆扫描、Flat 螺旋扫描、Planar CT、Tilted CT。
可在界面导入 Raw uint8 标签体、编辑尺寸/间距/材料映射、导入或生成能谱、修改几何和效果开关，
保存/加载 TOML。按计算按钮后先计算全批，完成后再通过底部滑条浏览。
计算使用界面/TOML 设置的每像素射线数；体积不降采样。更多说明见 `visualization/README.zh-CN.md`。

启动界面：`multispectrum_sim --gui [config.toml]`。不传配置路径时先弹出 TOML 文件选择窗口。
加载后参数自动填充，同时读取标签体、材料和能谱并显示三维场景；点击 Calculate all views 后才开始计算投影。

### BrainWeb 界面与全链路示例

`configs/ui/brainweb-fullchain.toml` 使用完整 `181 x 217 x 181`、1 mm、uint8 标签体；
`configs/ui/brainweb-center33-fullchain.toml` 使用中心 33 层，适合快速试用。
两者都配置 60 帧平板圆扫描、SID=709 mm、SDD=1143.699951 mm，
本地生成 120 kVp / 12° / 2 mm Al 能谱、10 个标签材料映射、量子噪声、可选探测器处理参数。
散射关闭，无需外部散射标定文件。输入数据不随源码提交，可复用已有 BrainWeb 数据或按前述脚本准备。

在界面 Load TOML 后，体积、材料、能谱和几何会一起加载；Calculate 执行前投，勾选 Recon 页的 Reconstruct after projection 时继续校正和重建。
完整投影、生成谱、空场和重建结果写入 `outputs/brainweb-full-ui/`（中心版为 `outputs/brainweb-ui/`）。
直接运行 `multispectrum_sim configs/ui/brainweb-fullchain.toml` 可执行
前投、水板硬化校正、FDK 重建，并导出三张正交切片。
60 帧是流程试用设置；水板校正为示例，并非骨材料专项标定。Recon 页也可单独运行重建；界面锁定期间仍刷新状态。

界面加载允许先填充缺少可选运行文件的配置，并显示缺失散射模型等提示；
点击计算仍执行完整文件校验，不会静默关闭效果。

顶部 Volume opacity 可实时控制标签体显示透明度，默认比例 0.03；不会改变计算输入。
Recon 页支持 FDK 与现有迭代入口、输出网格和文件路径、切片导出、硬化/散射校正开关。
底部 Run reconstruction 可直接运行当前输入投影的重建。

Volume 页内置材料无需手填 ID：选择 Builtin list 后，可用名称/ID/别名搜索及分类筛选。
选中自动填入名称、规范 preset 与默认密度，密度可自行覆盖或恢复默认。已有 TOML 的密度覆盖值保留。

重建完成后右侧直接显示三幅重建切片，Results 页可切换层号和前投图，并调整 HU 窗宽、窗位。
CT 值水基准与 BHC 拟合参考能量共用：`HU=1000*(mu/mu_water-1)`，水衰减先换算为 mm^-1。
显示变换不修改原始重建数据。材料下拉列表固定宽度，长名称省略并通过悬停完整显示。

Effects 现可编辑余晖时间/权重及每像素采样射线数；Recon 可编辑 BHC 多项式/截断、散射引导和自适应求解参数。
自适应后端是可选构建项 `MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER=ON`。


### 重建前处理流水线与显存池

前处理现在集中在 `src/recon/preprocess/`，由 `IStage` 声明输入/输出的数据域和 Host/Device 位置。连续 CUDA 阶段保持数据驻留显存，内置顺序为能量恢复 → 散射校正 → 转对数/BHC；最后两步融合执行。固定、自适应、引导散射均接受设备缓冲。新增插件通过 `Pipeline::append()` 注册，CPU 插件需要明确声明 Host。

显存池按设备管理，初始化帧工作缓冲，散射及 FFT 工作区在准备阶段借用，结束后归还供下次任务复用。不会按总显存预分配：新分配默认保留 `max(512 MiB, 20% 总显存)`（小显卡上限为总显存的一半），池总量最多 60%，归还后的空闲缓存最多 256 MiB。空间不足时先清理空闲缓存；活动缓冲和 VTK 对象不会被释放。详细接口、插件约定及传输边界见 [前处理开发说明](src/recon/preprocess/README.md)。


### 批量 CUDA 正投

正投按批次启动 CUDA kernel，以 `[角度][像素]` 并行计算，共用体积、能谱及材料表。整批的期望能量、采样能量、air、处理信号、处理参考、对数投影及物体掩码放在连续显存中，先在 CUDA 执行后处理，再每批一次下载到 pinned host memory 供写盘和显示缓存。随机种子仍使用全局角度，因此调整批大小不会改变采样和噪声结果。末批只下载实际帧数。

```toml
[projection.batching]
views = 64             # 每批最多 1..64 帧；1 可用于逐帧对照
max_buffer_mib = 256   # GPU 打包缓冲的预算，1..256 MiB
```

旧配置不填即采用上述默认值。UI 的 Effects 页提供 `Batch views` 和 `Batch buffer MiB`，Load/Save TOML 保留参数。实际批大小还受可用显存和 VTK 预留约束，分配不足时缩小批次；单帧超过预算时会报错，不会突破缓冲预算。线性正投缓冲接入显存池；标签纹理单独持有整轮使用。内置后处理和前向散射保持设备驻留；CPU 自定义插件会形成明确的整批传输边界。写盘和显示缓存仍在 CPU，批量传输减少的是调用/同步次数，并不减少结果本身的字节数。日志 `projection_batch` 给出实际批容量、下载次数和 GPU kernel/下载计时。


### 仿真阶段的设备状态和插件管理

`SpectralProjectionPipeline` 通过 `IProjectionBackend / IProjectionStage / IProjectionOutputSink` 管理后端、阶段和输出。CUDA 后端返回借用设备帧，内置阶段声明 Device；CPU 后端保留原来的 Host 阶段实现。自定义插件用 `addStage()` 注册、声明 `domain()/location()/phase()`，流水线按物理阶段及注册顺序执行。阶段逐批按角度顺序处理，状态跨批保留，每轮初始化及异常退出时统一清理。

CUDA 内置链路：多能正投/一次射线噪声 → fASKS 散射（含 GPU 边缘距离变换）→ 散射量子噪声 → 光学串扰 → 余晖 → 电子串扰 → DAS → air 归一化及负对数。参数/离散核等在 CPU 准备一次后上传，整幅图像不会在内置阶段间往返。fASKS 仍会回读少量有效性标志；内置链路最后每批一次完整下载，CPU 插件额外产生整批下载/上传。物体与 air 的余晖状态独立，air 不加入随机噪声或 DAS，保持原有定义。

CUDA 电子噪声使用全局角度/像素/配置 seed 的确定性 Gaussian 序列，调整批大小不改变结果；与旧 CPU `mt19937 + normal_distribution` 的样本序列不同。无电子噪声的探测器处理与 CPU 基准进行数值对照；CUDA 对数函数可能产生少量浮点末位差异。设备帧指针仅在当前批次有效，插件不得更换或跨批保存它们，其他 CUDA 流需要在返回前与默认流建立依赖。

256 MiB 限制包含批缓冲的全部设备图像（6 个 float32 平面及 uint8 掩码，共 25 字节/像素/帧）。例如 1536×1024 默认最多容纳 6 帧，另有散射、余晖和 FFT 工作区从受预算约束的显存池借用。阶段耗时先同步计算，避免把正投等待时间误归入下一阶段；日志单独记录正投、传输边界、各阶段及归一化。

进入重建 DLL 前，应用会调用 `yk::spectral::cuda_resources::trimDevicePool()` 释放当前设备的空闲缓存，并打印释放字节数。活动借用、VTK 和 DLL 自己的缓冲不在清理范围内。纯正投任务之间继续保留池缓存；跨到重建时优先让出空间，之后再次正投需要按需分配。


前向散射核选择在 UI 的 `Effects → Scatter kernel evaluation` 中设置：`All bins` 使用全部模型核节点，`Segmented` 显示 `Scatter kernel segments`（至少 2 段）。该项控制 `projection.scatter.kernel`，与 Recon 页的散射校正核参数独立。分段模式在模型的对数衰减区间等分取代表节点，并从原模型插值得到核形状，不是直接截取原文件中的前几个核。


### 时间看门狗与分段诊断

`YK_TIMING_WATCHDOG=1` 启用可选诊断，`YK_TIMING_LOG` 指定日志文件；`YK_TIMING_DETAIL=1` 记录每次阶段完成，默认只记录首次及超过 `YK_TIMING_SLOW_MS`（默认 200 ms）的调用。后台线程每 `YK_TIMING_HEARTBEAT_MS`（默认 5000 ms）报告长任务及该线程当前最内层阶段，每 10 秒和退出时输出次数、总耗时、平均与最大耗时。看门狗不终止计算；诊断模式的散射 CUDA 子步骤会同步以获得实际执行时间，正常运行不增加这些同步。

覆盖正投批计算/分配/传输、各处理阶段、散射核生成与 FFT 子步骤、输出读写、VTK 下采样与渲染，以及重建初始化/执行/读写和前处理校验/引导/边缘窗口。启动本地诊断 UI 可运行 `out/run-imgui-profile.cmd <config.toml>`，日志写入 `out/performance/manual-timing.log`。只测校正可运行 `multispectrum_sim --profile-preprocess <config.toml>`，生成输入投影同目录的 `.profile-preprocessed.raw`，不调用重建。

前处理校验已避免在每个像素上构造错误消息字符串。正式 fixed 校正的掩码和边缘距离变换在 GPU 执行；实验 adaptive 仍保留 CPU 掩码/边缘路径。修复了原 CPU 距离变换在弹出包络节点后未重新计算交点的问题，并使用暴力最近背景点距离验证；GPU 结果与修正后的 CPU 基准对照。旧错误路径的少量像素可能改变。重建入口将校正投影持久保存在输入同目录，规则见下节，不再使用系统盘临时目录。


原 scalar 正投的谱积分保留“每条射线分别衰减、累计各能量期望光子数、最后一次量子采样”的顺序。没有经过有效材料的射线直接使用透射率 1，跳过材料衰减求和和指数运算；仍计算原有几何通量、探测器响应和空气参考。能量点数不超过 128 时使用 128 项计数缓冲，129..256 点使用 256 项缓冲，减少常见能谱的每线程 local-memory 占用；数组仍是动态索引的线程私有存储，并非全部放进寄存器。空气参考与最终能量求和统一读取能量常量表（不可用时读取设备表）。采样数、随机种子和计数累计顺序保持不变，未启用快速数学近似。多采样测试还覆盖 119、128、129、256 个能量点，以及带探测器响应和量子噪声时两种计数容量的逐值一致性。

默认 warp 正投位于独立的 `src/primary/WarpSpectralTransport.cuh`，原 `SpectralTransport.cuh` kernel 保留用于回退及数值基准。不设置 `YK_PRIMARY_KERNEL`、设置为空或设置 `warp` 均使用 warp；设置 `scalar` 切回原版；其他值会报错。此选择不写入 TOML，普通 UI 启动器也默认使用 warp。每 warp 负责一个像素，32 条射线分块并行遍历，共享材料路径；各线程负责至多 4/8 个能量点，在寄存器累计计数，使用相邻能量的设备表读取避免常量表不同地址串行访问。量子采样仍在全部射线累计后进行，随机种子使用全局角度、像素、能量索引。最后能量归约和空气参考的累加顺序不同，结果允许浮点差异，不保证与原版字节一致。原版的 `YK_PRIMARY_PROFILE` 分项采样计时不适用于 warp；warp 在此开关下仅报告编译器资源，正常阶段 CUDA-event 耗时仍可用。

`multispectrum_unit_warp` 独立验证两种 kernel 和双精度 CPU 参考，覆盖 1/2/8/30/31/32/33/64 条采样、32 条分块边界、3/119/128/129/256 能量点、两焦点分量及端点、空体积、异质材料/体积姿态、响应/量子噪声、10 材料和通用槽位、整批/尾部像素及确定性。

`YK_PRIMARY_PROFILE=1` 进一步启用正投 kernel 内的稀疏采样计时（独立于阶段看门狗）。按全局角度/像素哈希选择约 1/1024 像素，报告射线几何、体积遍历、逐射线谱衰减/计数累计、最终量子采样/能量求和的周期及射线/体素访问数，并输出普通与采样 kernel 的寄存器和 local-memory 分配。周期比例不是墙钟时间的精确分账，采样版本的寄存器占用可能不同；正常运行不启用该版本。`multispectrum_unit_multisample` 使用逐体素盒相交的独立 CPU 基准验证异质材料、多采样、体积姿态、剂量平均和通用材料槽位的边界。

默认 warp 正投进一步为 10 材料、能量点数不超过 128 的情况启用材料衰减系数缓存：各线程提前读取自己负责的能量/材料系数，射线循环内从静态槽读取。其他材料数和 129..256 能量点保留原 warp 路径。`YK_WARP_COEFFICIENT_CACHE=0` 可关闭缓存用于对照；不设置或设置 `1` 使用上述自动选择。缓存不改变路径遍历、权重运算顺序、能量点数、采样数和噪声随机种子。`multispectrum_unit_warp_coefficients` 对照原 warp/scalar 和独立双精度 CPU 基准，覆盖能量及射线分块边界、材料与体积姿态、响应/噪声、确定性和批大小。

在 RTX 3070 Laptop、BrainWeb 12 帧 1024×1024、119 能量点/10 材料/60 射线每像素（保留原噪声和散射）的三次交替试验中，正投 kernel 中位数从 1352.83 ms 降至 1238.84 ms，减少约 8.4%，三类输出与原 warp 字节一致。缓存版本每线程寄存器从 81 增至 119，local-memory 仍为 72 字节；收益需在具体显卡/配置上评估。同轮固定权重合并收益约 2%，几何预计算慢约 21%，均未保留到正式路径。几何预计算将寄存器增至 135 且改变浮点顺序，组合版本也更慢。实验版本和详细日志保留在本地 `out/performance/brainweb-fullchain2/warp-three-options/`。

整理后的同一可执行程序再做三次开关缓存对照，中位数为 1306.04 -> 1238.82 ms，减少 5.1%；三类输出仍与原 warp 字节一致。跨轮次有运行波动，实际收益约 5%–8%，默认启用条件采用上述已验证范围。

正常焦点与散焦现采用面积分层和内部射线复用（scalar/warp 共用采样器），避免在完整散焦矩形再随机抽 N 点时内部与外圈点数不均。外圈按四个互不重叠边带的面积选择采点，无拒绝采样循环；正常分量仍均匀采正常矩形。例：正常 0.4×0.4 mm、散焦 1×1 mm、epsilon=0.01，内部 N 点权重 0.9916，外圈 N 点权重 0.0084；两者保持同一个像素的总曝光。相对旧随机抽样实现，有限 N 的结果和噪声样本允许改变，能量积分仍逐射线执行，最终一次量子采样。测试验证外圈范围、边带面积比例和中心对称性、面积权重、等大/单边扩展/零面积内部，以及独立 CPU 材料路径和剂量基准。

UI 将正常焦点和散焦拆为 `Finite normal focus` 与 `Defocus` 两个独立开关。TOML 的 `[projection.focal_spot] enabled` 控制正常分量是否为有限矩形，新增 `defocus_enabled` 控制散焦；正常关闭时其分量为点源，散焦关闭时 epsilon 暂不参与计算、正常分量获得全部曝光，两者都关闭为点源。关闭任一开关保留尺寸和 epsilon 参数。旧 TOML 缺少 `defocus_enabled` 时继承 `enabled`，保持之前总开关的行为；UI 保存时明确写出两个开关。

### 校正投影持久缓存与 TIGRE-SART 进度

启用重建预处理时，DLL 重建入口自动在原投影旁生成 `<input>.corrected.raw` 和 `<input>.corrected.raw.json`。RAW 是经过散射/BHC 校正的 float32 对数投影，布局为 `[view][v][u]`；JSON 记录尺寸、数据域、依赖和校正设置。完整校正成功后才发布缓存，失败时清理未完成文件；后续重建失败不删除已经完成的校正结果。

重建前检查输入投影、空气场、散射模型和文件能谱的规范路径、大小及修改时间，以及探测器尺寸/间距、全部校正参数和用于水板拟合的生成能谱设置。匹配且缓存完整时跳过校正，日志显示 `corrected_projection_cache hit`。只修改重建算法、迭代次数或重建网格不使缓存失效；重新前投或修改校正依赖后重新生成。文件检查使用元数据，外部编辑时应保留正常修改时间更新；若强制保留大小和时间，可删除 JSON 强制重算。内部校正实现修改时需要递增缓存版本。每个输入保留最新的一份校正结果，不无限累积不同参数的大文件。

如需显式重建导出的校正 RAW，将 `reconstruction.input_projection_file` 指向该文件，并关闭 `reconstruction.preprocessing.enabled`，避免二次校正。`--profile-preprocess` 仍是独立测量入口，不使用此缓存。文件读取和重建上传仍然存在，不是全链路零拷贝。

TIGRE-SART 一轮包含所有视角的逐视角更新。720 视角、10 轮共有 7200 次更新，不能按 10 次正反投理解。界面现在报告轮内子集更新的全程完成数；TIGRE-SART 每个视角完成后更新进度，OS-SART 每个子集更新后报告，TV 算法另包含正则化工作。

### OSSART 精确归一化权重缓存

内部 `sart` 默认改用 TIGRE 粗估归一化：2×2×2 粗体积估计行权重，二维列权重沿 Z 广播，不增加配置字段。该路径保留内部 SART 的松弛、epsilon 和约束，不进入 TigreGradient 的 TV/POCS 工作区；SIRT/OSSART 默认保持原精确权重。

### TIGRE 工作区与分层权重缓存

普通 `tigre_sart`、`tigre_os_sart`、`tigre_sirt` 直接读取调用者的设备投影，不再复制完整工作投影；模型投影仅在 TV/POCS 或 Nesterov 残差计算需要时分配，五份 TV 工作体仅在 TV/POCS 路径分配，动量体仅在 Nesterov 模式分配。投影和重建体在整个迭代过程中常驻显存。

内部 `sart` 和 TIGRE 共享 `IterationWeightCache`。每个视角/子集的行权重、二维列权重在首次使用时按原算子和公式计算为 float32，不改成新的粗估方法。固定子集工作区循环复用，选定缓存槽跨完整轮次保留；GPU 缓存命中直接使用，CPU 缓存命中上传当前权重，未缓存项重新计算。默认 GPU 附加预算 256 MiB，CPU 预算 4096 MiB；均按需分配，优先为较小的二维列权重保留槽位。GPU 预算在首个完整正反投更新、必需纹理和外部输入体积已分配后规划，预留 `max(512 MiB,20% 总显存)`。可选缓存分配失败时回退；Session reset 保留相同几何缓存，重新初始化或释放时清理。

诊断变量 `YK_TIGRE_WEIGHT_GPU_MIB` 支持 0..512，`YK_TIGRE_WEIGHT_CPU_MIB` 支持 0..8192，0 禁用对应层，非法值采用默认预算。CPU 预算是每个后端的上限；多个并行会话各自计算预算，并非进程统一内存池。重建入口及粗估后端检查必需线性缓冲和纹理预算，空间不足时明确失败，避免无意依靠共享显存运行；投影数据源支持情况见下节；尚未把投影缓冲跨重建任务持久复用。

`multispectrum_tigre_weight_cache` 覆盖内部 SART、TIGRE SART/OS-SART/SIRT/TV，验证无缓存、CPU、GPU、混合缓存及 reset/几何重新初始化。调整缓存不改变权重公式、视角顺序或迭代更新边界。

DetailedWeight OSSART 将固定几何的 `A_s*1` 行权重和 `A_s^T*1` 列权重按子集缓存为原始 float32，保持原正反投算子、归一化公式和子集顺序。默认附加预算 256 MiB；首次正反投完成必要纹理分配后，查询可用显存，并保留 `max(512 MiB,20% 总显存)` 的余量给 VTK/其他缓冲。优先为较小的行权重安排缓存，再为放得下的列权重安排缓存，实际只分配计划使用的字节。首次访问计算权重，其后同一子集直接复用；放不下或可选分配失败的权重继续现算。缓存槽跨完整数据轮次保留，避免循环子集的 LRU 每次都失效。Session reset 保留固定几何权重；重新初始化或释放时清理。

诊断环境变量 `YK_OSSART_WEIGHT_CACHE_MIB=0` 关闭缓存用于 A/B，允许 0..512 MiB；非法值沿用 256 MiB 默认。日志打印计划字节、缓存行/列子集数和释放时的命中次数。8 GiB GPU 上不能完整缓存 40 个 512×512×400 列权重（约 15.6 GiB），大配置仅缓存预算内的一部分，不做半精度或二维近似，也不改变进度透传。

`multispectrum_ossart_weight_cache` 对照关闭、仅行、部分列和完整缓存，并验证 Session reset 与已填充缓存后的几何重初始化。128³、48 帧、256² 探测器、8 子集的预热后六轮 OSSART 对照，关闭/开启缓存的多次耗时中位数约 252/135 ms（约 46% 减少）；该配置的全部权重约 76 MiB，可全部缓存。首次建权重和大配置部分缓存的收益需单独评估。

### SART / OS-SART / SIRT / TV 投影分批加载

内部 `sart`、`sirt`、`ossart` 与 `tigre_sart`、`tigre_os_sart`、`tigre_sirt`、`tigre_sart_tv`、`tigre_os_sart_tv` 支持从已校正 RAW 文件分批读取，不需要先构造完整 CPU 投影数组或上传完整设备投影。默认优先使用完整设备投影；预留显示空间后预算不足时自动选择分批路径，默认预取 64 帧、不超过 256 MiB，并保持重建体常驻。日志 `reconstruction_projection mode=resident/batched` 标明实际选择。权重缓存仍按原来的 GPU/CPU 预算复用。

加载遵循算法的视角索引：SART 保持逐视角更新，TIGRE OS-SART 保持连续子集，内部 OSSART 保持交错子集和既有遍历顺序。尾批正常处理；加载批次不改变更新次数、松弛衰减或权重。预算至少需要容纳一个算法子集，放不下时增加 OS-SART 子集数量或降低投影尺寸，不会静默改算法。读取中监测输入文件大小和修改时间，文件变化或短读会失败。

诊断变量 `YK_RECON_PROJECTION_INPUT=auto/resident/batched` 可用于强制对照，默认 auto。SDK 可传完整 Host 数组或 `SExecutionRequest.projection_read` 回调；可选尾部保留旧请求 ABI。CGLS/PWLS、柱面迭代和 TIGRE Nesterov/FDK 初值、其他高级 TV 变体仍使用原路径，尚不支持此回调。分批意味着每轮可能重新读取/上传投影，显存足够时仍优先常驻；跨重建任务复用尚未实现。

`multispectrum_batched_reconstruction` 验证八种算法的设备常驻、Host 和回调输入，覆盖不同批次、交错索引、尾批、reset、读取失败/异常/重试及旧 ABI。

SIRT 的计算批次不作为算法子集：分批计算行权重和残差，累加完整数据的反投与列归一化，最后统一更新体积。TIGRE SIRT 仍采用粗估二维列权重，内部 SIRT 仍采用精确三维列权重。批次变化可能改变浮点累加分组，允许小量舍入误差；视角顺序和算法轮次不变。TV 数据更新仍按原 SART/OS-SART 顺序进行，残差、自动阈值的 FDK 估计使用有界批缓冲，五份必需 TV 工作体保持常驻。不可变 TV 输入不再复制完整投影，Bregman 等可变输入变体仍保留原完整工作数组。

目录职责：`src/Reconstruction/Iterative/Flat/YkSirtBatchBackend.hpp` 是算法实现；迭代共用的投影数据源、权重缓存和显存预算在 `src/Reconstruction/Iterative/Common/`。示例仿真/校正共用显存池在 `include/cuda/DeviceMemoryPool.hpp` 与 `src/cuda/DeviceMemoryPool.cpp`，命名空间为 `yk::spectral::cuda_resources`，不属于重建预处理阶段。核心 SDK 的 `src/common/YkDeviceWorkspace.hpp` 是独立通用设备缓冲封装，不依赖示例的显存池。


### 内部代数重建的 TV 插件

UI 的 Reconstruction 页在选择 `sart`、`sirt` 或 `ossart` 后可选择 Regularization：
None / Smoothed TV / Huber TV Prox。载入和保存 TOML 会同步对应字段。
Huber TV Prox 在每轮完整角度更新后执行，可调强度、Huber 阈值、内步数、2D/3D 和强度衰减；
Prox step 为 0 时自动选安全步长。非负约束沿用重建页的 Non negative 开关。
默认 None，现有配置不会自动开启 TV。示例：

```toml
[reconstruction.iterative]
algorithm = "ossart"
iterations = 10
subsets = 12
relaxation = 1.0
projection_model = "joseph"
non_negative = true

[reconstruction.iterative.regularization]
plugin = "huber_tv_prox"
strength = 0.002
huber_delta = 0.001
inner_iterations = 10
dimensions = 3
step_size = 0.0
strength_reduction = 1.0
```

数值是起始示例，强度和阈值应按数据与物理体素间距调整；有限内步求得近似 Prox。
Huber 插件增加两份显存体积工作区，程序预算会计入，并根据余量选择投影常驻或分批输入。
保留现有 FP/BP，当前不要求严格伴随；算法边界与插件开发方式见 `doc/algorithms.md`。


### 统计与带正则代数重建

UI Reconstruction 的 Method 增加 Statistical：分别设置数据项、正则化和优化器。
当前提供带正则 SIRT（默认）和普通 GD，支持 Flat 圆扫/螺旋，复用同一套投影、校正缓存、进度、输出和 CT 显示。
可选 Smoothed TV / Huber TV / Tikhonov / Standard TV；两项正则超参数为目标权重 strength 和形状 shape，Tikhonov 不使用 shape。
统计 lambda 与旧代数 TV 的下降步长或 Prox gamma 不能直接互换数值。
正则化的组合策略和内层求解参数由库内部决定，UI/TOML 不提供这些调试项。
默认 Huber-TV 在带正则代数重建中采用近端求解，内部自动安全步长与收敛停止，安全上限 1000 步；strength=0 保持无正则行为。普通 GD 仍采用联合梯度，不改算法定义。
旧实验 TOML 中的 composition 表会被忽略，重新保存时移除。Standard TV 需选择带正则代数重建。
权重文件为空使用 W=I；非空权重 RAW 必须与投影同形状，内容为非负、有限的 float32 精度权重。
文件读取采用独立批缓冲，SDK 调用者也可使投影及 W 常驻设备。

```toml
[reconstruction]
type = "statistical"
input_projection_file = "corrected.raw"
output_volume_file = "statistical.raw"

[reconstruction.statistical]
projection_model = "joseph"
non_negative = true

[reconstruction.statistical.data_fidelity]
type = "weighted_least_squares"
scale = 1.0
normalize_by_ray_count = false
# weights_file = "precision-weights.raw"

[reconstruction.statistical.regularization]
plugin = "huber_tv"
strength = 1.0
shape = 0.00005
dimensions = 3

[reconstruction.statistical.optimizer]
type = "regularized_algebraic"
iterations = 10
step_size = 0.0
backtracking = true
max_backtracking = 20
step_shrink = 0.5
relative_tolerance = 0.0
minimum_iterations = 1
```

以上是配置起点，不代表头部数据的最优正则强度。输入 corrected.raw 已校正时关闭 reconstruction.preprocessing。
步长 0 使用初估并配合回溯；实际日志给出初末目标、最后步长、回溯次数和完成轮数。
当前不要求严格伴随，但也不宣称严格统计优化收敛。FISTA 尚未实现，选择它会报不支持。


`configs/ui/brainweb-statistical-gd.toml` 是完整示例，读取已有本地 BrainWeb 标签资产，
生成 512×512×360 投影，重建 256×256×200、1.2 mm 网格，输出到独立的 out/statistical-brainweb。
示例启用量子噪声和 BHC，关闭散射与散射校正，因此不需要额外散射核文件。
可以从 UI 载入；命令行也可使用 `multispectrum_sim <该 TOML 路径>` 运行完整链路。


统计重建现在默认采用“归一化 SIRT 数据更新 + 正则梯度”，优化器类型为 `regularized_algebraic`。
`gradient_descent` 表示纯 GD，`raw_gradient_descent` 为其兼容名称。
UI 的 Data update 可切换两者。新模式的数据项默认松弛 1，step_size 只控制正则子步，
关闭正则时与内部 SIRT 一致；其回溯只作用于正则子步，完整诊断目标不承诺逐轮下降。
正式 BrainWeb 示例已切换新模式。

UI/TOML 的 TIGRE OS 和普通 ossart 都使用 subsets 表示子集数量。TIGRE 应用层按 ceil(views/subsets) 换算一次后传入 SDK 的 block_size，后端直接采用，末块可以较短。
TIGRE POCS 默认每外轮将数据松弛乘 0.99，普通 TIGRE SART/SIRT/OS-SART 默认不衰减；该族使用 0.1 mm⁻¹ 的逆行权重上限保护短路径。

ImGui 的 Reconstruction / Method 提供 WFBP (helical)，用于规范平板螺旋的解析重建。
选择后可控制 WFBP cutoff 与 apodization，使用 FreeCT ramp 滤波；加载和保存 TOML 保留这些参数。
对应配置为 geometry.kind=flat_helical、reconstruction.type=analytic、reconstruction.analytic.pipeline=wfbp，filter=ramlak。
WFBP 当前消费完整投影序列，重排和 FFT 工作区较大；不使用 FDK 的 chunk_views 分包逻辑。UI 的整幅完成进度与 FDK/迭代的分阶段进度区分。

### 几何相对显示

ImGui 界面顶部的 `Gantry fixed (relative)` 可切换为机架固定的观察坐标系。以扫描首帧为参考，源、探测器、光束及其局部坐标系保持固定，模体及模体坐标系按扫描轨迹反向运动。圆扫描反向旋转，螺旋扫描同时反向平移；原有模体姿态与位置偏置仍保留。橙色轨迹切换为模体中心的相对运动轨迹。

此开关只改变显示，不改变计算几何、投影贴图或重建结果，不需要重新计算。源与探测器必须构成刚性机架；不满足该条件的轨迹会禁用开关，避免显示失真。重新加载配置后仍以新扫描首帧为参考。

ImGui 的内部正则插件配置仅在内部 `sirt`、`sart`、`ossart` 下显示。CGLS 与所有 TIGRE 算法隐藏这组正则插件并清除插件选择。TIGRE 自带的 `tigre_sart_tv`、`tigre_os_sart_tv` 仍可选择，并显示它们自己的 TV 迭代步数与 alpha 参数；不与内部正则插件绑定。
