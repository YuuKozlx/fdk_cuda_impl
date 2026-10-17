# 多能谱 CBCT 仿真示例

首次使用请按 [客户端构建、SDK 与运行](../../doc/build-ui.md) 准备 VTK/matio SDK，再编译并启动界面。

Example 菜单现可直接加载 BrainWeb 与 ICRP145 压缩 MAT 数据、材料及扫描预设，
无需先生成 TOML。数据布局、LFS 获取与运行方法见
[内置体素数据说明](../../data/phantoms/README.zh-CN.md)。

ICRP145 男女头部、胸部/腹部带手臂及去手臂的十个全材料示例见
[ICRP145 示例说明](configs/icrp145/README.zh-CN.md)，配置位于 `configs/ui/icrp/`。
材料保留原介质组成和男女密度，默认使用螺旋 WFBP 解析重建。

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
FDK 与 WFBP 共用 `SAnalyticFilterParams`、既有 `EFdkFilter` 枚举和公共滤波实现。
`filter` 支持 ramlak/ram-lak、shepp-logan、cosine、hann、hamming、blackman、butterworth、kaiser、tukey。
`cutoff` 统一使用 cycles/sample，范围 (0,0.5]，0.5 为完整 Nyquist 带宽；`gain` 默认 1。
Butterworth/Kaiser/Tukey 分别使用 `butterworth_order`、`kaiser_beta`、`tukey_alpha`。
旧 `wfbp_cutoff` 和 `wfbp_apodization` 已移除，读取时会明确报错，不静默映射。

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

主工程启用 `BUILD_VTK_VISUALIZATION=ON` 并提供 VTK C++ 包后，在 TOML 增加 `[visualization] enabled=true`，即可边前投边显示实际材料标签体、源/探测器姿态与偏移和当前帧投影贴图。体数据保留原分辨率，仅显示贴图可降采样；关闭显示时计算和输出不变。显示模块位于本示例的 `visualization/`，不依赖 geometry_viewer 工具源码。构建、配置及自生成验证见 [实时显示层说明](visualization/README.zh-CN.md)。

## 可选 ImGui 前投界面

顶层开启 `BUILD_VTK_VISUALIZATION=ON`、`BUILD_IMGUI_GUI=ON`，配置中设置
`[visualization] enabled=true, interactive=true`（project 工作流）。界面与 VTK 共用窗口，
第一版支持 Flat 圆扫描、Flat 螺旋扫描、Planar CT、Tilted CT。
可在界面导入 Raw uint8 标签体、编辑尺寸/间距/材料映射、导入或生成能谱、修改几何和效果开关，
保存/加载 TOML。按计算按钮后先计算全批，完成后再通过底部滑条浏览。
计算使用界面/TOML 设置的每像素射线数；体积不降采样。更多说明见 [实时显示层说明](visualization/README.zh-CN.md)。

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

默认前投使用 `src/primary/WarpSpectralTransport.cuh` 的统一 kernel，由 `include/primary/PrimaryLaunchPolicy.hpp` 按实际材料数、能量点数和每像素射线数分派布局。UI/TOML 不增加线程参数，不做运行时自动测速，也不按投影尺寸细分。所有布局保留相同 Siddon 遍历、射线随机种子、逐射线指数衰减及累计后一次量子采样。线程归约顺序可能产生浮点末位差异。

能量点数不超过 128 时，默认规则为：

| 射线/像素 | 1..32 材料：线程/像素 | 33..64 材料：线程/像素 |
|---|---|---|
|1..2|4|4|
|3..4|4|8|
|5..8|8|8|
|9..16|16|16|
|17 及以上|32|16|

每块通常 128 线程，33..64 材料的 16 线程组用 64 线程/块。129..256 能量点统一采用 32/128，不外推低能量容量的小组性能结论。材料路径按 10/32/64 容量编译，系数读取遵守实际材料数，系数缓存的空槽归零；小材料集合不需要扩充输入系数表。缓存分支会包含容量内的零项，因此计算成本也有容量档位特征。大多数布局在共享内存直接累计路径；不超过 10 材料、32 线程/像素时采用线程私有路径并缓存材料衰减系数，避免受控高采样测试中的性能回退。这些都属于同一个计算 kernel 的布局与存储选择，没有独立“旧版固定32线程”入口，也不再通过 `YK_WARP_SHARED_PATHS`、`YK_WARP_SPARSE_MATERIALS`、`YK_WARP_COEFFICIENT_CACHE` 选择实验策略。

启动日志输出 `primary_kernel`、`group_threads` 和 `block_threads`。`YK_PRIMARY_PROFILE=1` 在 warp 模式下报告寄存器、local、静态与动态共享内存资源；`YK_WARP_PROFILE=1` 对实际选定的布局采样阶段周期，支持所有 1..64 材料及 128/256 能量容量。每批 `warp_sample` 输出 setup、geometry/path/init、shared exchange、spectrum、noise/energy sum、reduction/output，以及采样 pixel groups、射线、命中和体素数。按全局角度/像素哈希采样约 1/1024 组。周期是采样线程组的驻留周期，不能作为可直接相加的 GPU 墙钟毫秒；路径桶还包括几何构造和数组初始化，采样代码也会改变寄存器占用。正常运行默认关闭采样。

`YK_PRIMARY_KERNEL=scalar` 仍可运行独立 scalar 实现用于数值研究；不设置、空值或 `warp` 均使用上述自动分派。该设置不写入 TOML。`YK_PRIMARY_PROFILE` 在 scalar 模式下仍提供该实现原有的分项周期。

`multispectrum_unit_warp_policy` 覆盖 1/2/10/11/32/33/53/64 材料，1/2/3/4/5/8/9/16/17/32/50 射线的分派边界；数值对照覆盖 3/119/129/238/256 能量点、体积姿态、双焦点分量、探测器响应、量子噪声、尾部线程组、整批/逐帧一致性及实际布局采样开关。小材料用真实长度的系数表测试，防止容量填充导致越界读取。其他 `warp`、`warp_coefficients`、`warp_profile`、`multisample` 和正式 projection pipeline 测试继续校验 CPU 独立基准与设备输出。

原生性能工具 `multispectrum_benchmark_warp` 为 `EXCLUDE_FROM_ALL` 目标，不参与日常默认构建、不依赖 Python。显式构建后运行：

```text
multispectrum_benchmark_warp <RAW标签TOML> <探测器行数> <每像素射线数> <结果CSV> [all|policy|controlled:N]
```

支持 1..64 材料、最多 128 能量点、flat circular/helical 几何，探测器响应须关闭。普通模式读取与网格匹配的 uint8 RAW 标签；固定 1024 列、12 帧及每周 12 帧，通过调整像素间距保留 TOML 的探测器物理尺寸。每个布局先完整预热和验证，再按预设次序运行三次，取 CUDA event 中位数；不包含上传、下载、其他物理插件和文件输出。结果包含普通 kernel 编译资源。`policy` 仅测少量代表布局；`all` 用于完整布局对照，256 线程/块的大材料布局需要支持 64 KiB opt-in 动态共享内存的 GPU。

`controlled:N` 在 CPU 直接生成固定外形的同心材料分层圆柱，不读取外部体积：网格必须是 512×512×400、0.5 mm，圆柱半径 96 mm、半高 80 mm，背景标签 255 不参与衰减。从输入配置取前 N 种实际材料并重编号到 0..N-1，所有材料壳层均有体素。关闭焦点、散焦和量子噪声，固定几何、能谱、网格及外形，只改变材料种类数和射线数。N 不得超过配置的实际材料数。此模式对照 32 线程数值参考和自动选定布局，最后单独采样选定布局的周期，不把采样版本混入性能中位数。原 GPU 输出缓冲限制为 256 MiB，不导出大型 RAW。它用于计算负载研究，不作为人体图像质量模型。

2026-10-17 的 RTX 3070 Laptop 测试：受控 2/10/32/53 材料 × 1/4/16/50 射线、119 能量点、1024×256、12 帧。相同射线数下，四种材料数的采样体素访问数完全一致；1/4/16/50 射线对应约 133/531/2124/6637 万次访问，遍历量近似随射线数线性增加。材料衰减求和的操作量随实际材料数增长，但几何、遍历、指数和求和等固定工作仍存在，耗时不按材料数直接倍增。每像素1条射线时，2→53 材料的自动布局耗时约 150→185 ms；采样周期中谱积分占比约 5.2%→18.9%。这些比例是诊断线索，不是精确墙钟分账；跨采样数还会改变线程布局，不能据此拟合单一线性耗时公式。

完整受控矩阵、边界 2/8/32/50 射线试验、阶段计数、资源与正式入口结果保存在 `out/warp-layout-benchmark`，最终汇总见 `controlled-results.txt` / `controlled-summary.csv`。少材料50射线的早期共享存储结果已由 `controlled-final-*.log` 覆盖；不要把初始回退值当成当前默认。两布局测试按参考后自动布局相邻运行，几个百分点内的波动不视作稳定收益。早期 BrainWeb/ICRP 布局对照是两组不同网格和组织结构，不能把它们的绝对耗时差归于单独的材料数。此前固定32线程的分项记录在 `out/warp-segment-profile`，早期8线程试验在 `out/shared-path-benchmark`；它们属于历史对照，不代表当前默认策略。设备频率、热状态和具体几何会影响实测时间，规则按代表场景收敛，未按每种显卡建立调参表。

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
选择后与 FDK 共用 Filter 下拉框、cutoff、gain；参数化窗只显示自己的形状参数。加载和保存 TOML 保留同一套声明。
对应配置为 geometry.kind=flat_helical、reconstruction.type=analytic、reconstruction.analytic.pipeline=wfbp，filter=ramlak。
WFBP 当前消费完整投影序列，重排和 FFT 工作区较大；不使用 FDK 的 chunk_views 分包逻辑。UI 的整幅完成进度与 FDK/迭代的分阶段进度区分。

### 几何相对显示

ImGui 界面顶部的 `Gantry fixed (relative)` 可切换为机架固定的观察坐标系。以扫描首帧为参考，源、探测器、光束及其局部坐标系保持固定，模体及模体坐标系按扫描轨迹反向运动。圆扫描反向旋转，螺旋扫描同时反向平移；原有模体姿态与位置偏置仍保留。橙色轨迹切换为模体中心的相对运动轨迹。

此开关只改变显示，不改变计算几何、投影贴图或重建结果，不需要重新计算。源与探测器必须构成刚性机架；不满足该条件的轨迹会禁用开关，避免显示失真。重新加载配置后仍以新扫描首帧为参考。

ImGui 的内部正则插件配置仅在内部 `sirt`、`sart`、`ossart` 下显示。CGLS 与所有 TIGRE 算法隐藏这组正则插件并清除插件选择。TIGRE 自带的 `tigre_sart_tv`、`tigre_os_sart_tv` 仍可选择，并显示它们自己的 TV 迭代步数与 alpha 参数；不与内部正则插件绑定。

### 重建后处理

重建后的处理和导出通过独立的 `recon/postprocess` 流水线管理，包含处理 Stage、只读 Exporter 和工厂 Registry。当前已接入原有 RAW/三视图 BMP 导出，并提供可选的 HU 转换 Stage；DICOM、降噪、骨硬化和 MAR 算法尚未实现。数据位置与数值域检查及扩展方式见 [后处理插件说明](src/recon/postprocess/README.md)。

### 统一流水线基础层

多能前投、重建前处理及后处理已直接采用同一个有类型 Stage/Output 契约与执行器，统一数据位置、数值域转换、生命周期、进度、计时和工厂。物理阶段排序、空气参考和三维体积等业务差异仍由各自接口表达，详见 [统一流水线说明](include/pipeline/README.md)。

三条流水线的 Stage/Exporter 强制通过 IPlugin::info() 提供静态自描述；FactoryRegistry 支持不实例化插件的 list()/find(id) 查询。内置描述目录与参数说明详见 [插件自描述与查询](include/pipeline/README.md)。

多能前投已作为注册的 `MultiSpecFpStage` 源阶段接入：配置/体积/材料/能谱准备、批次生成、输出校验及 Backend 生命周期由源统一管理；现有融合 kernel、批次显存预算和设备驻留保持不变。源 ID 为 `projection.source.multi_spec_fp`，可查询注册描述和 Pipeline::sourceInfo()。


## 通用图像处理 Stage

投影与体积的升降采样、均值、高斯、双边、自引导滤波现提供 CPU/CUDA 实现；体积可选逐层二维或空间三维处理。API、显存生命周期、投影重采样的几何要求和实测结果见 [图像处理 Stage](include/image/README.md)。当前为 C++ 接口，尚未增加 TOML 或 UI 滤波面板。


## HU 转换、标定、校正与分割

支持可变水／空气基准的 μ↔HU 转换、空气／水双材料 ROI 标定、线性 HU 校正和带命名标签的多阈值分割，均有 CPU/CUDA Stage。标定与校正分别位于独立目录，结果可跨任务复用。见 [接口、公式及示例](include/recon/postprocess/stages/README.md)。

## BrainWeb 骨束硬化校正实验

`scripts/brainweb_bone_bhc_experiment.py` 保留历史分割—重投影实验编排。正常客户端现已通过 `BoneBhcStage` 接入物理前置／后置和双材料 LUT 方案，入口为 Reconstruction → Bone hardening correction，默认关闭、仍标为实验性；阶段顺序、TOML 参数及验证范围见 [统一骨硬化策略](include/recon/bhc/README.md)。下面的脚本和旧结果属于独立历史实验，不代表所有新 UI 参数组合的质量验收。

先构建 `multispectrum_bone_bhc_prepare` 目标（默认不构建），它通过正式 CUDA μ→HU / 阈值分割 Stage 处理**第一次重建的结果**。Python/NumPy 仅负责实验编排、模型差值及指标统计；绘图改为仓库的 C++ / Matplot++ 工具 `multispectrum_plot_bone_bhc`，不再使用 Matplotlib。绘图工具可直接读取已有结果目录，单独运行时完全不需要 Python。Matplot++ 仍需要 Gnuplot 后端，见前面的可选绘图构建说明；也可用 `plot/CMakeLists.txt` 独立构建，不必重新编译 CUDA 库。应用本身没有增加 Python 运行依赖。

在仓库根目录运行，输出目录必须不存在：

```powershell
cmake --build out/build/viewer-integrated --target multispectrum_bone_bhc_prepare -j4
out/vtk-env/Scripts/python.exe example/multispectrum_sim/scripts/brainweb_bone_bhc_experiment.py `
  --exe out/build/viewer-integrated/example/multispectrum_sim/multispectrum_sim.exe `
  --prepare out/build/viewer-integrated/example/multispectrum_sim/multispectrum_bone_bhc_prepare.exe `
  --plotter <multispectrum_plot_bone_bhc.exe的路径> `
  --labels <已有的brainweb-normal-1mm-center33-181x217x33-u8.raw> `
  --output <新的G盘结果目录>
```

实验固定采用本地 BrainWeb 中心 33 层，体积 181×217×33、1 mm；圆扫描 360 帧，探测器 512×96、0.945×1.26 mm；SID 709 mm、SDD 1143.699951 mm。120 kVp / 2 mm Al 能谱，单条射线采样，关闭散射、量子噪声及探测器附加效应。水校正沿用三阶水板拟合，参考能量 70 keV。HU 使用库中 XCOM 查询的水衰减 0.019285194762051105 mm⁻¹。

分割阈值预设为 −500 / 300 HU，输出空气、水等效软组织、骨三类；模型密度分别为 0.001205、1.03、1.61 g/cm³，骨使用 `tissue.ncat_skull`。不利用原始标签替代重建分割，不对结果施加平滑。能谱和骨材料组成仍假设已知，且与模拟测量匹配；这比真实未知材料条件理想。

设 `Q` 为相同的水校正，`M` 为上述重建分割模型，校正公式为：

```text
p_corrected = Q(p_measured) + P_mono70(M) - Q(P_poly(M))
```

最终重建关闭水校正，避免对已校正的投影重复处理。流程属于分割重投影的模型差值校正，可参考 [Modelling the physics in iterative reconstruction for transmission computed tomography](https://pmc.ncbi.nlm.nih.gov/articles/PMC3725149/)。本实验没有另外拟合骨多项式或读取参考单能投影参与校正。

2026-10-13 RTX 3070 Laptop 实测，以下 RMSE 是**相对同几何、同 FDK 的 70 keV 单能重建**，用于隔离束硬化影响，不等于相对真实连续物体的总误差。评估 z 索引 [8,25)，避开有限体积上下边缘；“内部”使用 26 邻域腐蚀一体素，“含边界”不腐蚀。

| 区域 RMSE / HU | 未校正 | 仅水校正 | 水＋骨校正 |
| --- | ---: | ---: | ---: |
| 软组织内部 | 60.76 | 5.38 | 1.14 |
| 骨内部 | 174.17 | 57.30 | 2.57 |
| 全组织，含边界 | 97.93 | 25.75 | 3.72 |
| 骨，含边界 | 204.52 | 83.83 | 4.31 |

分割骨 Dice 0.876（完整 33 层）。初始水校正前投＋重建约 3.40 s；额外的分割、模型双前投、水校正、再次重建共约 7.39 s，含独立进程启动和 RAW 文件读写，不代表集成后纯 GPU 耗时。

结果包含各步 `run.toml`、RAW、日志、`comparison-matplot.png`、`segmentation-matplot.png`、`metrics.json` 和 `timings.json`。`bone-bhc/run.toml` 是已校正投影的重建配置，**不执行整套骨校正**；完整复现使用上面的脚本。校正结果是 μ（mm⁻¹），图中转换成 HU 显示。重绘命令为 `multispectrum_plot_bone_bhc.exe <结果目录>`，不会重新仿真或改动 RAW。

该基础实验验证了中心层、已知能谱与骨材料条件下的可行性；全头螺旋实验见后文，材料或能谱失配、金属、不同阈值的稳健性尚未验证。仍存在与单能参考共有的采样/FDK 误差，不能将其误认为束硬化或降噪效果。

### 包含散射及探测器效应的流程

给脚本增加 `--physics-config <头部TOML>`，会读取其中的能谱、散射模型、焦点、随机射线采样、量子噪声、探测器响应/串扰/余晖/DAS 和散射校正参数。默认处理 `--labels` 的**完整体积**，输入完整 BrainWeb 时为 181×217×181；保持 360 帧圆扫描和 FDK，探测器高度按体积范围增大（完整头部为 512×320），**不直接执行该 TOML 原来的螺旋几何**。只有显式给出 `--center-z` 才截取该位置附近 33 层；`--display-z` 只选择主对比图层面，不改变计算体积。输出仍写入新的实验目录，不修改传入 TOML 或它的原始输出。当前该选项要求散射施加和散射校正都开启。

链路是：多能测量（含物理效应）→ 恢复能量信号 → 迭代散射校正 → 水 BHC → 初次重建 → HU 分割 → 模型多能/单能重投影差值 → 再次重建。模型重投影关闭散射和随机噪声，但保留确定性的焦点采样、串扰和余晖；没有通过重复加噪声生成校正量。单能参考仅用于评估，不是测量输入，也不参与校正。现有前处理尚未提供余晖和串扰反校正。

另存 `water-only` 分支展示不校正散射、只做水 BHC 的结果，使用独立输入文件以免覆盖散射＋水校正缓存。目录 `water-bhc` 在此模式下实际为“散射＋水 BHC”，图标题明确区分。

2026-10-13 首轮中心 33 层试验使用用户的 `brainweb-fullchain-new.toml` 物理配置：120 kVp、6 mm Al、119 能量点、FASKS v4 散射、固定 5 次迭代/阻尼 0.7 校正、量子噪声、焦点与散焦、两种串扰、余晖和 DAS；探测器响应仍按该配置关闭。评估方式与上面的基础实验相同（下表不是完整体积结果）：

| RMSE / HU | 未校正 | 仅水 BHC | 散射＋水 BHC | 再加骨校正 |
| --- | ---: | ---: | ---: | ---: |
| 软组织内部 | 80.53 | 103.74 | 19.85 | 19.32 |
| 骨内部 | 92.48 | 140.78 | 36.70 | 26.86 |
| 全组织，含边界 | 79.13 | 97.64 | 31.03 | 29.33 |
| 骨，含边界 | 102.64 | 148.31 | 45.57 | 54.06 |

结论：散射校正必须在骨校正之前；这个案例里只做水 BHC 反而变差。骨校正在内部和整体区域改善，但骨边界仍有过校正，不能直接设为默认方案。将确定性成像效应加入校正模型后，含边界骨 RMSE 从初版 55.63 降为 54.06 HU，仍高于不加骨校正的 45.57 HU。此结果只验证现有散射模型的施加/校正闭环；施加和校正共用同一 FASKS 模型，并不代表已通过独立蒙特卡洛或实测散射标定验证。

### 完整体积与骨间条纹检查

后续使用**完整 181×217×181 标签体积**，360 帧、512×320 探测器，按上述完整物理链重新计算。`--display-z 26` 只将主对比图选到颅底骨集中层；前投、分割、模型重投影及最终重建均处理全部 181 层。没有给重建结果应用平滑。此数据的下边界在颅底附近，没有完整下颌与牙列，因此不能声称验证了完整下巴的去伪影效果。

除了整体误差，逐层用“非空气且非骨”的标签掩模腐蚀两体素，避免骨边缘主导统计。对相对同几何单能 FDK 的**误差图**做 sigma=2 像素的 XY 高斯分解（只用于分析，不改变重建）：低频误差反映缓变暗带/条纹及偏差，高频误差包含噪声、细节和采样误差，不能直接等同纯量子噪声。区域 RMS 按有效体素数加权。

| 指标 / HU | 散射＋水 BHC | 再加骨校正 |
| --- | ---: | ---: |
| 全体积软组织低频误差 RMS | 7.43 | 4.66 |
| 颅底 z=[8,48) 低频误差 RMS | 9.86 | 5.21 |
| 同区低频误差空间变化量 | 9.52 | 4.56 |
| 同区高频误差变化量 | 30.72 | 30.12 |
| 全组织（含 z 边缘）整体 RMSE | 38.46 | 35.56 |

完整三视图显示骨间成片暗带减弱，斑点大多保留；并非所有位置都改善，顶部 z=162…178 的低频误差变大。结论是本配置下骨校正有去缓变伪影价值，但没有明显降噪效果，也不应直接启用为默认。上面的中心 33 层结果和本次完整体积结果不能混用：散射场、投影覆盖和重建边界均不同。

输出新增 `artifact-by-slice-matplot.png`（全部 181 层曲线）、`whole-volume-orthogonal-matplot.png`（轴/冠/矢三视图与误差图）。`metrics.json` 的 `artifact_by_slice` 和 `regional_artifacts` 保存统计及掩模定义。仿真测量＋散射/水校正＋初次重建约 14.69 s，骨校正额外流程约 18.89 s；Matplot++/Gnuplot 四张图导出约 122.37 s，计时单独记录，不计入校正计算性能。

完整体积复现时，`--labels` 指定 `brainweb_normal_1mm_181x217x181_uint8.raw`，加上 `--physics-config <头部TOML> --display-z 26`，省略 `--center-z`。Windows 绘图构建目录还生成 `plot-bone-bhc.cmd`，自动设置已配置的 Gnuplot 路径，可以直接传入结果目录重绘，不需要 Python。原生绘图支持 `--label-overview <完整标签RAW> <PNG>` 查看骨分布定位图。

### 螺旋 OS-SART 全体积实验

`scripts/brainweb_helical_bone_bhc.py` 接续已经完成的螺旋基线，直接保留基线 TOML 的扫描几何、OS-SART 和 TV 参数，不使用上面圆扫描脚本的固定几何。实验目录首先需要 `water-bhc/run.toml`、`water-bhc/reconstruction.raw` 和 `water-bhc/projection.raw.corrected.raw`。先用正式程序运行基线配置，再执行：

```text
python scripts/brainweb_helical_bone_bhc.py --exe <multispectrum_sim.exe> --prepare <multispectrum_bone_bhc_prepare.exe> --plotter <multispectrum_plot_bone_bhc.exe> --output <实验目录>
```

本轮参数为 SID/SDD=709/1143 mm、探测器 1024×256 / 0.6 mm、5 周×500 帧、40 mm/周、起始 z=-100 mm；重建 512×512×400 / 0.5 mm，OS-SART 10 次外迭代、50 子集。保留 Huber-TV 近端强度 `1e-4`、delta `5e-5`；原 TOML 的 `inner_iterations=20` 等字段也不改动，但 Huber-TV 内部实际使用自动步长与收敛判定，并非固定执行 20 步。脚本没有修改算法默认值。

模型重投影的标签、尺寸、间距和位置来自重建网格；原始 BrainWeb 181×217×181 / 1 mm 标签仅用于测量、独立单能参考及评估。评估标签按物理坐标最近邻采样到重建网格，越界为空气。全部 400 层参与统计；颅底采用物理 z=[-82,-42) mm，误差分析高斯 sigma=2 mm。投影差值以 16 帧分块处理，避免同时载入多份 2.44 GiB 投影。

只计算散射＋水校正、增加骨校正、相同重建参数的单能参考三组；`comparison_cases` 指定原生绘图的对应列。单能参考不参与校正。该脚本是实验编排工具，尚不是正式的骨校正 Stage 或 UI 功能；再次执行会重新计算后续步骤，因此请保留独立实验目录。

2026-10-13 此螺旋配置实测，全部 400 层软组织低频误差 RMS 从 6.49 降至 3.45 HU，颅底从 8.55 降至 2.70 HU；全组织含界面 RMSE 从 24.83 降至 19.00 HU，骨内部从 44.88 降至 9.57 HU。这些误差的参考依然是**相同 TV 的单能重建**，不能用来证明细节没有被平滑。

针对过度平滑，额外复用 `bone-corrected.raw`，只将 TV 强度从 `1e-4` 降到 `1e-5`，其余重建参数不变，另存于 `bone-bhc-tv-1e-5`，不覆盖基线配置。中央 z=[-19.75,39.75] mm 的原始灰白质标签腐蚀 1 mm 后，材料本身对比度为 3.99 HU，原强度结果仅剩 0.88 HU，弱 TV 为 3.84 HU；单能参照使用原强度时也只有 0.96 HU。白质区域空间标准差从 0.66 升到 3.95 HU，反映细节恢复伴随更大波动，不能将此统计直接当作纯噪声标准差。

`multispectrum_plot_bone_bhc.exe --tv-comparison <实验目录>` 用原生 Matplot++ 生成 `tv-strength-matplot.png`，显示中央层面的中心 256×256 像素细节，WL=35、WW=80 HU，不降低体积采样；包含强 TV、弱 TV、强 TV 的单能重建以及原始材料衰减参照。原始材料参照未经重建，只用于显示细节上限，和带成像模糊的重建不完全等价。

Windows 下 `multispectrum_plot_bone_bhc.exe --bone-window <实验目录>` 快速导出同窗比较的原生 BMP：窗宽/窗位为 1500/500 HU 和 1000/700 HU，列为散射＋水 BHC、再加骨 BHC、同 TV 的单能参照；层面为 `display_z`、后移 24 和 48 层。`bone-detail-ww1000-wl700.bmp` 放大颅底局部，使用原始像素重复显示，不插值或修改体积。`bone-window.json` 保存窗口及裁剪参数。此诊断图直接采用 Windows 原生位图导出，避免高分辨率多图经 Gnuplot 导出的长时间等待，不引入新的第三方库。

骨窗对比中，骨内部灰度更接近单能参照且更均匀；小骨轮廓和边缘没有明显锐化。原始标签体素为 1 mm、骨只有单一材料/密度，没有连续的骨小梁密度纹理，重建网格的 0.5 mm 间距不会增加输入本身的信息量。

### 理想单遍骨校正与 WFBP / 5 mAs 验证

新实验将 `mAs_per_view` 设为 5，保留 2500 帧、1024×256 / 0.6 mm 平板、5 周 / 40 mm 每周步进、512×512×400 / 0.5 mm 重建网格；重建改为 WFBP，默认 Ram-Lak（共享 cutoff=0.5、gain=1），没有 TV 或额外降噪。测量端仍包含散射、焦点/散焦、量子噪声、串扰、余晖和 DAS，随后先做散射＋水 BHC。

`multispectrum_bone_bhc_singlepass` 是显式构建的实验 CUDA 工具，不依赖仓库外源码。它复用内部扫描几何、姿态变换和 Siddon 材料路径遍历。每个像素只发一条中心射线并遍历一次，得到空气、水等效软组织和骨路径；共享这些路径计算原能谱下的能量积分和 70 keV 目标单能值。再用**正式前处理的同一水板拟合函数和钳位范围**计算 `delta=mono-Q(poly)`，直接输出 `bone-corrected.raw` 和 `model-residual.raw`。模型端不施加焦点、随机像素采样、串扰、余晖、DAS、噪声或散射；这属于简化模型，实测中未逆校正的非理想效应仍可能造成边界失配。

```text
cmake --build <集成构建目录> --target multispectrum_bone_bhc_singlepass
multispectrum_bone_bhc_singlepass --self-test
python scripts/brainweb_helical_bone_bhc.py --exe <multispectrum_sim.exe> --prepare <multispectrum_bone_bhc_prepare.exe> --plotter <multispectrum_plot_bone_bhc.exe> --singlepass <multispectrum_bone_bhc_singlepass.exe> --native-images --output <已完成基线的实验目录>
```

`--self-test` 用解析路径检查空气、纯水、纯骨和混合路径，并使用非恒等水多项式核对 GPU 校正值。单能参照仍是独立评估支路，使用原始标签和相同 WFBP 参数，不参与分割或校正。`singlepass.json` 记录拟合系数、参考能量、输入路径、批次容量、修正范围及耗时。

2026-10-13 RTX 3070 Laptop：2500 帧单遍校正 GPU 核心 12.61 s，文件读写及处理 27.57 s，含进程启动 29.97 s；再次 WFBP 重建约 21.40 s。相对同几何单能 WFBP，颅底低频误差 RMS 从 8.91 降至 3.24 HU，全体积骨区域（含界面）RMSE 从 49.89 降至 31.42 HU，骨内部从 49.51 降至 21.14 HU。高频误差变化量在颅底为 18.05→17.96 HU，说明该校正主要减少缓变暗带，没有明显降噪效果。单能参照仍存在细条纹，不能把所有残余条纹都视为束硬化；此结果也不能与原 OS-SART＋强 TV / 1 mAs 的指标直接作算法优劣比较。

Windows 下 `--artifact-window` 原生导出轴、冠、矢三视图（WW=250、WL=25 HU），覆盖原始体积范围，不限制统计层数。`--native-images` 同时导出骨窗和暗带窗口，避免 Gnuplot 多图导出的长时间等待。

本次同时修复 WFBP 的大工作区问题：根据显存预算自动选择批量工作区，在 Host 暂存滤波结果，GPU 分批完成带完整插值邻域的重排/FFT，然后按半圈相位组打包所有圈的观测反投。每个相位的 W(q) 分子/分母仍使用完整观测，未裁掉有效扫描或降低尺寸。公共 Session 仍一次提交完整扫描，不表示可以外部逐帧调用 WFBP。保留至少 512 MiB 或 20% 总显存的预算余量；大尺寸 FFS 飞焦点重建暂不走此批量路径。平板/圆柱、Host/Device 输入输出、重复执行和 reset 与常驻路径的回归最大差约 7.5e-8。

该结果可用 `multispectrum_bone_bhc_prepare --bilateral <重建TOML> <重建RAW> <独立输出目录>` 做一次后处理双边滤波。调用正式 Stage 链：μ→HU→CUDA Bilateral→HU→μ→RAW 导出；按 BHC 参考能量的水衰减转换 HU。当前实验参数为逐层 XY、5×5 邻域、空间 sigma=0.75 mm、灰度 sigma=30 HU，Z 层间不混合；输出仍是 mm^-1，尺寸、间距、原点与方向保持不变，并写出 `.grid` 和 `bilateral.json`。输出与输入同路径时明确拒绝，原始 RAW 保留。

512×512×400 实测双边 Stage 约 918 ms，转换及导出链共约 1.70 s，仅一次上传和下载。观察到颗粒感减轻、骨轮廓大体保留，宽条纹仍有残余；显示平滑不等于进一步完成了骨硬化校正。`multispectrum_plot_bone_bhc.exe --bilateral-window <实验目录>` 输出同窗的原图／双边结果／未滤波单能参照，以及骨细节和冠矢状面图。此选项是实验后处理工具，尚未添加 UI/TOML 滤波开关。

### 共享解析滤波设置

FDK/WFBP 的 SDK 设置分别位于 `reconstruction.fdk.filter` 与 `reconstruction.wfbp.filter`，类型完全相同；通过 `.type=EFdkFilter::Hann` 选择窗，不再用浮点 apodization 宣称窗类型。FDK 的 Parker 设置仍归 FDK，WFBP 的 redundancy_flat、重排和螺旋加权仍归 WFBP。

公共 `Filter::analyticFilterDesc` 统一映射参数，公共核生成器和 `FrequencyFilter` 处理窗函数、有限离散 ramp、物理间距及 IFFT 归一化。WFBP 保留二倍通道重排与零填充，并使用公共有限离散 Ram-Lak 基核，避免复制窗函数或频域卷积代码。SDK 结构声明有所调整，应用应与新头文件一同重编译。

```toml
[reconstruction.analytic]
pipeline = "wfbp"
filter = "hann"
cutoff = 0.5
gain = 1.0
```

把 pipeline 改为 fdk 即复用相同窗配置。WFBP 的公共执行仍要求完整投影序列；此声明不改变扫描几何或启用逐帧流式调用。

### 螺旋配置与 SDK 床进方向

示例 TOML 的 `pitch_mm_per_turn` 描述随源角度增加的源探 Z 位移，前投、模型重投影和显示共用该约定。SDK 的 `SRegularHelicalScanSpec::pitch_mm_per_turn` 描述物理床进，因此示例在 `LibraryGeometry` 中按 `-pitch_mm_per_turn * rotation_direction` 转换；SDK 调用方仍遵循物理床进约定。

## 内置示例与 IExampleStage

内置示例直接生成 SimulationConfig、标签体积和几何真值，UI 与前投使用内存数据，不写入再读取 TOML/RAW。无需外部模体或 Python。

| ID | 模体 |
|---|---|
| water-circle | 20 cm 水圆柱，圆扫，SID 709 / SDD 1143，360 帧 |
| water-helix | 同一水模，螺旋两周，每周 360 帧、步进 40 mm，预置 WFBP |
| bead-line | 直线排列，12 颗 3 mm 钨珠、间隔 10 mm |
| bead-double-ring | 直径 100/80 mm 非对称双环，每环 12 颗珠及额外标记 |
| bead-cylinder-marker | 同步双环，每环 6 颗珠及两个圆柱标记 |
| bead-spiral | 24 颗螺旋排列钨珠及下方标记 |
| bead-sync-6 | 同步双环，每环 6 颗珠及两个球形标记 |
| bead-sync-12 | 同步双环，每环 12 颗珠及两个球形标记 |

六种几何模体保留原模体网格（直线 0.125 mm，其余 0.2 mm），不降低体积显示分辨率。预览采用 1024×256、U/V 间距 0.417/1.668 mm 探测器，保留原 1024×1024、0.417 mm 的视野，默认 360 帧。水模体为 2 mm 体素，圆扫探测器 1024×256、U/V 间距 0.4/1.6 mm；螺旋为 1024×256、U/V 间距 0.4/0.8 mm，两周共 720 帧，起点 Z=-40 mm。重建网格各方向为原模体尺寸的一半（向上取整），体素间距相应调整以保持物理范围。全部使用原生 120 kVp 能谱、2 mm Al 滤过、每像素一条射线，关闭噪声和散射。这些默认值用于快速验证流程，不是重建质量验收参数；已保存的 TOML 不受默认值调整影响。

命令行（主工作树根目录）：

    .\out\build\main-ui\example\multispectrum_sim\multispectrum_sim.exe --list-examples
    .\out\build\main-ui\example\multispectrum_sim\multispectrum_sim.exe --gui-example bead-line .\out\examples\bead-line
    .\out\build\main-ui\example\multispectrum_sim\multispectrum_sim.exe --example water-circle .\out\examples\water-circle

GUI 通过 **Example → Water phantoms / Geometry calibration** 选择模体，在独立参数窗口中调整尺寸、珠列/环/螺旋、标记和体素网格，点击 **Generate and load**，即可生成体积并填入材料、能谱和轨迹参数。点击 Calculate 才开始前投并写结果；示例生成本身不创建文件。可直接调整扫描参数。生成后的内存体积不能直接重新解释维数；改变网格需在模体窗口重新生成。生成会清除旧投影和重建缓存，并创建新的示例扫描设置。超出网格、非整数珠数、非法数值以及超过 512 MiB 的标签体积会被拒绝，不会静默裁剪。主动 Save TOML 时才同时导出配套标签 RAW，原内存示例仍可继续使用。

IExampleStage : pipeline::IPlugin 是准备资源的接口，提供 generate(ExampleRequest) -> ExampleResult，复用 PluginInfo 与注册表。结果包含强类型配置、标签数组、模体局部坐标下的球/圆柱标记真值。它不插入像素处理链，不执行珠心检测或几何求解。新增示例实现接口并在 builtinExamples() 注册即可。ExampleRequest.output_directory 仅指定计算结果路径，views=0 使用示例默认帧数。

现有 configs/calibration/geometry 等 TOML 保留为具体实验配置，与内置示例独立。

### 客户端菜单

- File：加载/保存配置、体积与材料工作区、加载已配置的重建结果、导出能谱。
- Example：按类别选择内置模体；每种模体独立参数窗口，可恢复默认值、取消或生成。修改参数不会立即触发计算。
- Settings：跳转到扫描、体积/材料、能谱、物理效应、重建及结果参数页。
- Compute：前投、重建、能谱预览，以及前投后自动重建开关。
- View：打开投影/重建抽屉、关闭抽屉、机架固定相对模式。
- Help：基本操作流程。

左侧作为参数工作区，保留常用计算快捷按钮；图像通过右侧抽屉查看。计算期间菜单中的数据修改和任务操作禁用，避免同时修改正在计算的数据。

IExampleStage.parameters() 返回该模体的参数描述（标识、名称、分组、默认值、范围、整数约束）；ExampleRequest.parameters 仅传入用户覆盖值。UI 依据描述生成表单，新增模体无需在 UI 写一套字段分发。参数在生成端再次校验，命令行及其他调用者也遵守相同约束。

菜单栏顺序：File → Settings → Compute → View → Example → Help；Example 固定在倒数第二项。

各示例通过 IExampleStage.scanPreset() 绑定常规扫描配置。示例参数窗口展示 SID/SDD、帧数、探测器及重建算法；点击 Generate and load 后整体应用并跳到 Scan 页。只打开窗口不修改当前设置，Cancel 不产生副作用。

### 几何标定与几何校正 Stage（实验性集成）

Example → Phantom calibration (experimental) 提供几何模体的实验性标定窗口，仅支持完整圆扫描、平板探测器。六个内置模体自动绑定各自的已知尺寸和编号方法：

| 内置模体 | 标定方法与限制 |
| --- | --- |
| 双环（圆柱标记）`bead-cylinder-marker` | 圆柱辅助编号，圆柱投影质心不参与最终几何拟合；环上球珠参与空间七参数拟合 |
| 同步双环带标记 6 球 `bead-sync-6` | 每环 6 球；标记编号、双向跟踪、空间七参数拟合 |
| 同步双环带标记 12 球 `bead-sync-12` | 每环 12 球；重叠珠点排除，双向整圈跟踪按帧选择一致性较好的结果 |
| 大小双环 `bead-double-ring` | 已知两环半径、间距、相位及标记位置，空间七参数拟合 |
| 带标记螺旋 `bead-spiral` | 利用螺旋尺寸和标记建立编号，再跟踪和空间七参数拟合；螺旋指模体形状，不是扫描轨迹 |
| 单排钢珠 `bead-line` | 五参数圆扫描联合拟合：SID、SDD、探测器 U/V 平移、面内旋转；要求离轴放置，探测器面外倾斜可忽略 |

空间七参数为 SID、SDD、探测器 U/V 平移及三个倾角，同时估计模体姿态。单排钢珠不能使用空间 DLT，不复用双环的七参数求解器；采用受限圆扫描模型，以明确填写的标称距离初始化，检查数值可辨识性，拒绝轴心退化情形。它不是 Wu 解析公式的实现。单排示例默认离轴放置且探测器面外倾角为零，保留面内旋转和平移用于验证。

步骤：

1. Example 中选择上述任一模体，生成后自动带入已知尺寸。水模体不能用于这套珠点标定。
2. 在 Scan 设置探测器偏移/倾斜、源偏移，用前投生成非理想几何的负对数投影。模体位置/姿态可以变化。
3. 实验性标定窗口中设置检测阈值、跟踪门限、拟合迭代数和可接受 RMSE，再执行标定。输入仅为已知模体尺寸、像素尺寸和实际投影，不传入源/探测器误差真值。
4. 查看报告中的收敛状态、重投影误差和拟合几何。通过质量门限后，分别运行标称几何重建和标定几何重建。结果使用 .nominal.raw 和 .calibrated.raw 后缀，保留原投影及仿真配置。

GeometryCalibrationStage 复用原生 cbct_calibration 库，输入域 Measurements、输出域 Geometry；支持已编号非共面二维点/三维点及可用性掩码，也支持上述六类模体 float32 负对数 RAW 的检测跟踪与拟合。PipelineConfig.phantom_kind 在内部选择 DoubleRing、Spiral 或 SingleRow，双环变体通过 PhantomSpec 的尺寸与标记配置区分，不需要六套独立 Stage。
GeometryCorrectionStage 从标定结果生成 GeometrySpec，域 Geometry → CorrectedGeometry，拒绝未收敛、非有限值及超过误差门限的拟合。它更新重建使用的几何，不对投影做像素扭曲，也不把标定值覆盖到仿真真值。两者复用顶层 IStage 和 PluginInfo，CPU 上运行；UI 异步执行并显示耗时，核心未提供的逐迭代进度不伪造。

限制：螺旋扫描及非完整圆扫描不可套用；空间模体至少需要一帧标记与球珠充分分离以建立编号，严重重叠、截断或低信噪比可能导致拒绝拟合。阈值需匹配实际负对数投影。单排钢珠的限制见上表，不能因为残差较小就假定其恢复了面外倾斜。源偏移和探测器法向偏移采用零值规范约束，拟合的是等效几何，不承诺独立恢复所有机械偏移。标定后的重建坐标系可能与原仿真坐标系相差刚体变换，比较体积时需配准；报告同时保留拟合的模体姿态、模型名称和重投影误差。单排模型没有独立 DLT 源轨迹，报告的 source_circle 为 null。标称几何重建的 SID/SDD 可在窗口中明确指定。

原生求解器独立位于 tools/cbct_calibration_cpp；上层通过其 CMake target 链接，不引入 Python 或 geometry_viewer 源码。优先使用已安装 Eigen3，否则首次配置下载校验过的 Eigen 3.4.0 头文件依赖，后续构建复用本地缓存。

测试 cbct_calibration_phantom_tests 覆盖六类合成 RAW 的检测、编号、跟踪、几何恢复及报告写出，并覆盖单排的噪声、缺失点与轴心退化拒绝。实际多能投影集成验证使用 512×512、360 帧，重建网格为 256×256×320、0.6 mm；标称和标定几何分别输出结果。这些验证检查流程与几何残差，不等价于所有模体摆位、噪声水平下的图像质量验收。

2026-10-16 内置默认配置验证：六类全部收敛，且标称/标定两次 FDK 均完成。圆柱标记双环、同步 6 球、同步 12 球、大小双环、标记螺旋、单排钢珠的重投影 RMSE 分别为 0.1282、0.1227、0.1254、0.1251、0.1229、0.1954 px。结果位于 `out/examples/calibration-family-validation/<example-id>/`，报告为 `projection.raw.geometry-calibration.json`。这里的残差是几何拟合指标，不是重建图像伪影或分辨率指标。

### 普通重建的独立几何

Recon 页提供 Reconstruction geometry 分组。默认沿用输入投影关联的扫描几何；
启用 Use independent reconstruction geometry 后，可独立设置 SID、SDD、源 XYZ 偏移、
探测器 U/V/法向偏移和三轴倾斜。界面角度单位为度，TOML 使用弧度。
Copy scan geometry 复制扫描参数，Zero offsets / tilts 将偏移和倾斜清零。
这些设置只影响重建，不修改正投几何；自动重建也使用该设置。
投影尺寸、帧顺序、扫描轨迹和重建网格继续沿用原配置。

配置保存在 [reconstruction.geometry]，用 enabled 控制启用；字段名与扫描几何一致。
关闭后仍保留自定义值，便于切换比较。几何标定窗口的标称/标定比较按钮使用各自的几何，
不受普通重建的独立几何开关覆盖。

### 参数面板组织

界面面板拆分在 include/ui：ScanPanel、VolumePanel、SpectrumPanel、EffectsPanel 和
ReconstructionPanel；Widgets 统一基础输入控件。各面板编辑配置并返回操作意图，
不持有 VTK 视图、GPU 资源或计算任务；客户端统一负责执行、结果加载及任务状态。
VolumePanel 单独持有材料列表的搜索状态。主题定义集中在 visualization/ImGuiTheme.hpp。

### 重建查看与 MPR

重建完成或 File → Load reconstruction 加载 float32 RAW 后，View → MPR workspace
可将右侧图像抽屉展开为四视图：轴向 XY、冠状 XZ、矢状 YZ 与任意斜切面。
关闭 Expand viewer 可回到几何视图旁的抽屉；双击任一图像放大，再双击或点 Four views 恢复。
查看器状态/交互位于 include/ui/ReconstructionViewerPanel.hpp，切面几何和 VTK 重采样位于
visualization/Mpr.hpp、Mpr.cpp，与重建算法和 TOML 参数编辑分开。

- 预设提供 Brain（WW 80 / WL 40）、Soft tissue（400 / 40）、Bone（2000 / 500）、
  Lung（1500 / -600）、Abdomen（350 / 50）、Subdural（200 / 80），单位 HU；也可手填 WW/WL。
  这些是可调整的常用起点，不是所有数据固定使用的标准值。
- 点击或左键拖动图像联动三轴定位；滚轮在正交视图翻层，在斜切视图沿法线移动。
  右键拖动分别调窗宽/窗位；预设与调窗作用于全部四图。
- 前三幅图的彩色十字线是另外两幅切面的实际交线。拖动参考线或线上圆点可绕当前
  切面法线旋转另外两个切面，三者保持正交；当前图像的显示方向保持稳定。
  拖动中心圆环或点击图像空白处移动共同交点。参考线旋转后，前三图也会重新采样，
  滚轮沿当前旋转切面的法线翻层。下面的 MPR rotation 显示同一旋转的 X/Y/Z 角度：
  拖参考线实时更新角度，手动改角度同步更新切面和参考线。第四幅斜切图同步该方向，保留独立法线偏移。
- 斜切面以三视图交点为旋转中心，角度按网格轴依次施加 X、Y、Z（Rz·Ry·Rx），
  Plane offset 沿旋转后的法线平移。默认三线性插值，也可选最近邻或三次插值。
  Reset MPR 同时清除参考线旋转、独立斜切角度和法线偏移，Center 将交点移回体积中心。
- MPR 对已经重建的体积取样，不重新执行 CT 重建。原始体积、网格和衰减值不改变；
  VTK 借用同一份体积内存，仅缓存二维切片。调窗复用切片，Auto WW 使用加载时缓存的百分位窗。
- HU 基准沿用当前配置 BHC 拟合能量对应的水衰减。加载其他任务的 RAW 时需使用匹配的配置。
  优先读取同名 `.raw.grid` 的尺寸、间距、原点及方向；没有 sidecar 时沿用当前配置网格。
  当前 XY/XZ/YZ 与角度指重建网格轴，坐标显示采用 sidecar 的物理坐标；没有患者方向信息时
  不推断 DICOM 的左/右、前/后。斜切视野外显示黑色，无效样本不计作 HU 数值。

数值测试 multispectrum_mpr 覆盖正交切片、非等距体素、90° 与复合旋转、线性场插值、
物理原点/方向、三视图定位、体积外背景及清空/重载；multispectrum_reconstruction_display
验证输出网格元数据的读取。界面还支持 File → Load reconstruction 直接选择已有 RAW。
参考线测试还覆盖连续交替旋转、当前显示方向保持、共同交点保持、三切面正交性及旋转后的插值数值。
multispectrum_mpr_interaction 通过 ImGui 输入事件验证拖线旋转、松开结束旋转、中心拖动定位，以及参考线与切面几何的一致性。

重建页分为方法、输出体积与文件、投影校正、几何四组。独立几何关闭时只保留开关，
隐藏所有独立参数；关闭保存切片时隐藏切片前缀。投影算子、批处理性能、探测器姿态、
物体姿态采用可展开的高级分组，噪声关闭时不显示随机种子。显示设置收进公共折叠组。
