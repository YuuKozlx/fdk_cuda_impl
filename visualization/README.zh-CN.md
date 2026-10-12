# 多能前投实时显示层

`ykcbct_visualization` 是独立 C++ 模块，接口和实现均位于本目录，不依赖 `tools/geometry_viewer` 的源码。只有启用时才查找、链接 VTK；YKCBCT 核心 DLL 不增加 VTK 依赖。运行不需要 Python。

## 构建与使用

在现有主工程配置中增加 `-DBUILD_VTK_VISUALIZATION=ON -DVTK_DIR=<VTK CMake 包目录>`，保持 `BUILD_MULTISPECTRUM_SIM=ON`，然后构建 `multispectrum_sim`。`BUILD_VTK_VIEWER` 是独立工具开关，不影响实时显示。

在多能前投 TOML 中增加：

```toml
[visualization]
enabled = true
texture_max_size = 512
max_fps = 15
projection = "signal"
hold_open = true
```

交互参数界面使用纯 VTK。在 `project` 工作流中增加 `interactive=true`，并保持 `hold_open=true`、`offscreen=false`。左侧是分组参数栏（Object / Source / Detector / Geometry），中间三维场景与角度滑条，右侧上下显示能量信号和衰减投影。

调整滑条仅修改参数草稿；点击绿色 Apply / Recompute 按钮或按 Enter 后，实际重新前投全部视图。计算期间禁用参数与角度输入，显示进度；完成后开放缓存浏览。右侧能量图取自实际 signal_energy，衰减图与输出文件采用同一负对数公式和同帧空气参考。两图与三维场景随角度共同切换。Geometry 组可设置 SID、SDD、探测器 Columns/Rows 和 Pixel U/V 间距；修改后重新生成几何与投影缓存。交互预览固定每像素 1 次射线采样，体数据原分辨率不变。

交互修改只在运行中生效，不改写原始 TOML；完整投影输出会由当前计算替换，旁边的 `.vtk-parameters.txt` 记录本次采用的参数。联合重建工作流保持只读显示，以免后续重建使用未同步的配置。

默认 `enabled=false`，不开窗口、不生成显示体或贴图、不查询显示几何。未编译显示层却设置 `enabled=true` 会明确报错。显示只用于 `project` / `project_and_reconstruct` 工作流。

`projection` 可选 `signal`（探测器处理后的信号）、`primary`（期望一次射线能量）、`log`（使用同帧处理后空气参考的负对数比值）。贴图按当前帧有限值范围归一化为灰度，常量帧显示中灰，不改变投影输出。图像行 0 对应探测器 V 最小边，列 0 对应 U 最小边。长边最多 `texture_max_size` 像素，小图不放大，降采样使用区域平均。

体数据保持原始尺寸、体素间距和 uint8 材料标签，最近邻分类渲染；未映射标签透明，密度小于 0.01 g/cm³ 的材料作为空气透明显示。其他材料按固定颜色表显示。体绘制不按显示贴图大小重采样。

窗口在计算完成帧后同步更新。刷新上限由 `max_fps` 控制，计算和文件输出仍包含全部帧。每个已完成帧的几何和降采样 RGB 贴图都缓存下来，刷新时跳过的帧也能回看；不额外缓存完整分辨率投影或复制体数据。贴图缓存内存约为 `11 × 显示宽 × 显示高 × 帧数`（贴图 RGB 与两张 float32 投影） 字节。最后一帧自动更新，几何、帧号和贴图始终一起提交。鼠标只操作观察相机，关闭窗口只关闭显示，前投继续计算。当前使用计算线程在帧间处理窗口事件，因此单帧计算很长时窗口响应也要等待该帧完成。

底部滑条选择已计算扫描帧；左右键逐帧切换，End 返回最新帧并恢复跟随计算。手动选择旧帧后保持该帧，继续计算不会抢走选择。源和探测器局部坐标系、物体姿态及贴图一起切换，保持当前观察相机。滑条仅浏览既有轨迹，不编辑物体或扫描约束。

橙色实线为源轨迹，青色虚线为探测器中心轨迹，均取自后端最终几何。按 T 可隐藏/显示轨迹。等 SID 与 SDD-SID 的圆轨迹中，两条路径在空间上重合；虚实线区分显示，不人为移动轨迹。

源使用小圆柱实体标记：前端面中心位于实际焦点，圆柱朝向源到探测器中心的主射线，尺寸是显示符号，不代表实际射线管外壳。淡黄色半透明光束从焦点延伸至实际探测器边界，支持平板和柱面，不覆盖投影贴图端面；按 B 可隐藏/显示光束。切帧时圆柱、光束、局部轴和贴图共同更新。

`hold_open=true` 在前投结束后保留窗口，关闭窗口才进入后续重建/返回；连续批处理可设 `false`。`offscreen=true` 可用于自动验证；`snapshot_directory` 可选，保存最终截图、原分辨率本地标签体、降采样贴图、探测器网格以及 scene.txt 中的物体变换矩阵和源坐标。路径相对于 TOML 文件。

## 几何与姿态

后端 `geometryForDisplay(view)` 提供其实际使用的几何。规则平板/柱面轨迹的 `scannerRay` 同时供 CUDA 前投和 Host 显示使用；非标准轨迹使用同一份已解析 ConeVec 的 Host 副本，不读取 Device 指针。支持六种现有轨迹、探测器 U/N/V 偏移和 U/V/N 倾斜、源 XYZ 偏移及螺距。

物体变换严格匹配前投：先加 phantom_offset，再依次绕世界 X/Y/Z 旋转；这是已有前投约定，不能理解成先旋转再平移。显示矩阵与 CUDA 的 phantomInverse 互逆。显示曲面覆盖全部像素边界；材料体、体积盒及物体轴共用姿态，源和探测器轴随实际几何更新。焦点标记表示标称实际焦点位置，随机焦斑采样不绘成逐条射线。

当前只显示能量积分帧，未接入计数探测器的阈值通道，也未实现 CUDA/VTK 零拷贝。显示层只复制 Host 输入，不修改配置或扫描轨迹。

## 可复用接口

`LiveView.hpp` 不暴露 VTK 类型。调用方在窗口所属线程使用 `setVolume`、`setTrajectory`、`update` 和 `finish`。`LabelVolume` 是原分辨率材料体；`FrameGeometry` 包含源、探测器网格/轴与行主序物体矩阵。每个完成帧都调用 `update`，由内部限制实时刷新；`selectFrame` 可用于程序控制切帧，经过同一滑条回调更新画面。所有输入在调用中复制，不保留计算侧指针。

## 自生成验证

以本目录独立配置：

```powershell
cmake -S visualization -B out/build/vtk-live -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DBUILD_VTK_VISUALIZATION_TESTS=ON -DVTK_DIR=<VTK CMake 包目录>
cmake --build out/build/vtk-live
out/build/vtk-live/spectral_live_test.exe out/live-checks
```

测试自身生成椭球材料模体、夹杂、能谱和实际 CUDA 多能投影，不依赖下载数据，也不引用查看器源码。验证六种轨迹、非零姿态和偏移、原分辨率标签读回、贴图缩小和方向、同帧导出以及显示开关前后投影文件逐字节一致。额外生成 `live.toml`、`headless.toml`、phantom.raw 和 spectrum.spec，供正式 multispectrum_sim 入口运行。能谱是验证用的简化二能模型。

测试增加 `--interactive` 会展示生成模体的实时计算窗口，在柱面螺旋轨迹的最终帧停留。普通可复用显示库独立构建只需要 C++ 和 VTK；生成 CUDA 验证程序时才需要 CUDA 工具链。

2026-10-11，本机 Windows / MSVC 19.44 / CUDA 12.6 / VTK 9.6.2 验证：独立显示库和正式应用均可构建；六种轨迹的生成数据检查通过；正式应用使用生成的水/铝模体和二能谱，显示开关前后完整投影一致。CUDA 几何、路径、噪声、探测器响应、多能透射、焦点、流水线及配置验证通过。关闭两个 VTK 构建开关后，无有效 VTK 包也可配置主工程；核心 DLL 和未启用显示的流水线测试程序没有 VTK DLL 依赖。

显示优化：交互重算期间保留全部显示缓存，但三维场景只绘制首帧和末帧，进度画面最多每 250 ms 刷新。贴图与两个投影面板合并为一次区域采样，并复用输出模块已计算的衰减图。浏览时保持完整体数据，不做体素降采样；日志单独记录 display_and_cache 耗时。

## ImGui 界面（Windows 第一版）

顶层构建同时设置 `BUILD_VTK_VISUALIZATION=ON`、`BUILD_MULTISPECTRUM_SIM=ON`
和 `BUILD_IMGUI_GUI=ON`。关闭 ImGui 选项时保持原纯 VTK 界面。
ImGui 1.91.9b 源码及官方 Win32/OpenGL3 后端位于 `thirdparty/imgui`，随工具编译，
无需 Qt、Python、GLFW 或 SDL，不增加核心 YKCBCT DLL 的依赖。

当前使用 VTK 原生窗口和 OpenGL 上下文，ImGui 在最后一层绘制。
窗口输入先交给 ImGui，控件捕获的鼠标与键盘事件不再传给 VTK 相机。
三维区域仍使用 VTK 的体绘制、实际几何和局部坐标系；两幅投影也由 VTK 显示。
体积保持完整原始分辨率，投影显示可以降采样。

第一版扫描范围：`flat_cbct`、`flat_helical`、`planar_source_orbit`、`nikon_tilted`。
当前六个页签：

- Scan：类型、帧数、每圈帧数、起始角、螺距，以及 Planar/Tilted 专有参数；探测器尺寸、像素间距、姿态、源偏移。
- Volume：Raw uint8 标签体文件、XYZ 尺寸与间距、模体姿态、标签到材料的映射及加载体积的标签计数。
- Spectrum：文件导入或本地生成、管电压、靶角、过滤层、能量步长、曝光量、曲线预览和导出。
- Effects：几何通量、量子噪声、有限焦点、散射、串扰、余晖和 DAS 开关。串扰离散核保留加载配置中的值；余晖时间常数与权重可编辑。
- Recon：FDK/迭代方法、滤波器、分包/迭代参数、输出网格、投影输入、结果路径、切片导出、硬化/散射校正开关。

使用 `multispectrum_sim --gui` 启动并选择 TOML，或
`multispectrum_sim --gui <config.toml>` 直接加载。加载时自动读取体积、核对材料映射、读取或生成能谱、显示模体姿态与源/探测器/轨迹，不会自动计算投影。
之后可在窗口 Load TOML / Save TOML；加载新 TOML 会自动更新能谱曲线与标签计数，同时清除旧投影缓存，避免将新场景与旧投影混用。
已启用 interactive 的 TOML 也可继续通过原命令启动。
修改仅更新草稿，按 Calculate all views 后校验、重新读取标签体、构建能谱和计算全部帧。
计算期间禁止修改参数和浏览角度，窗口继续刷新状态与进度；完成后开放缓存浏览。
体积字节数必须等于 XYZ 乘积，所有非零标签必须有材料，材料数不超过 32。
ImGui 每像素采样射线数由 TOML/界面控制（1..4096），CUDA 能量分箱不超过 256；输入错误会显示原因。
加载 TOML 时先完整读取、校验资源，验证通过后才替换场景并清除旧缓存；加载失败时显示原因并保留原配置。修改草稿后点击计算时，无效输入在进入流水线之前拒绝，保持上一次缓存；计算执行失败时显示失败状态。

每次成功计算写出 `<projection.raw>.imgui.toml`，记录实际计算配置。
保存界面配置时，勾选自动重建会保存为 project_and_reconstruct，否则保存为 project；保留模板中尚未开放的设置，并将数据路径转为绝对路径。
此版本尚未提供浮点 CT 体积的分割或 DICOM 导入。
`multispectrum_ui_config` 回归测试覆盖四种扫描、几何与能谱配置读写，以及缺失材料和体积尺寸错误。

### 重建与显示透明度

顶部 Volume opacity 为显示比例，默认 0.03，可实时调节，不改变标签体、材料密度或投影计算。
通过 `[visualization].volume_opacity_scale` 保存，0 完全透明、1 使用原始材料不透明度；
修改该值无需重新计算。多层组织叠加会增加整体不透明程度。

底部 Run reconstruction 或 Recon 页同名按钮可单独运行重建；
勾选 Reconstruct after projection 会在 Calculate all views 成功后继续执行校正和重建。
TOML 的 project_and_reconstruct 工作流会自动勾选此选项。
FDK 适用于平板圆扫描及已有支持的 Planar/Tilted 近似；平板螺旋不能用圆扫描 FDK。
迭代入口使用现有 DLL 算法，并按其能力检查几何组合。
对刚计算的投影，单独重建固定使用该投影的扫描几何，输出网格与算法设置来自 Recon 页；
选择其他输入文件时使用当前配置几何，文件长度仍由后端严格校验。

重建在后台工作线程执行，主线程继续刷新窗口和运行耗时，期间锁定参数和帧浏览；
完成后恢复操作，显示结果文件路径。输出为 float32 体积，启用 Save orthogonal slices 时导出三张 BMP。
当前三维场景显示标签模体；右侧可在前投图与三幅重建切片之间切换。
本地验证覆盖 BrainWeb 的 UI 单独 FDK、水板校正、SIRT（2 次迭代）及前投后自动 FDK 全链路。

### 内置材料选择

Volume 页的每个标签映射提供 Material source：Builtin list / Chemical formula。
内置模式从 `builtinMaterialsBegin/End()` 的正式材料表选择，支持按名称、ID、别名搜索，
以及 Tissue/Element/Compound/Mixture 分类筛选。选择后自动写入规范 preset ID、显示名称和默认密度，
清空 formula；标签编号保持不变。悬停显示默认密度与元素质量分数。
密度可覆盖，Use default density 恢复当前预设密度；Add material 默认使用内置水。
已有 TOML 预设及其密度覆盖值在加载时保留，不会被默认密度自动覆盖。
自定义模式继续使用化学式的原子计数和独立密度（如 H2O）；单独的元素原子数比表格编辑器尚未加入。
选择材料属于参数草稿，重新 Calculate 后用于前投与体积显示。

### 重建切片、CT 值与窗宽窗位

重建完成后自动显示右侧三幅切片（XY 轴向、XZ 冠状、YZ 矢状），并切换到 Results 页。
可选择各方向层号，图像比例遵循重建体素间距；数据使用原始重建网格，无体积降采样。
Results 的 Right panels 可切回两幅前投图，Load saved reconstruction 可读取当前配置的输出体积。
`[visualization].auto_load_reconstruction=true` 是显式选择启动/加载配置时显示已有结果；默认 false。
已有文件会检查输出网格、字节数、有限值，不会被当作新计算结果。

显示采用 `HU = 1000 * (mu / mu_water - 1)`，水为 0 HU，零衰减为 -1000 HU。
水取 H2O、密度 1 g/cm³，衰减系数与 BHC 水板拟合共用 `resolveWaterReference()`：
使用 `beam_hardening.reference_energy_keV`，为 0 时用校正能谱的光子加权平均能量。
XCOM 的水线性衰减为 cm^-1，转换为 mm^-1 后与重建输出比较，避免十倍单位误差。
界面显示实际参考能量及水的 mm^-1 衰减系数。未启用 BHC 时，仍以配置中的参考能量定义显示基准。

Window width / Window level 的单位为 HU，宽度必须大于 0；三幅切片共用同一窗口。
Auto window 使用全体积的稳健分位范围，保留当前层号。HU 换算与窗变换仅发生在切片显示，
重建原始 float32 文件和内存衰减数据保持不变，不重新计算投影或重建。
`multispectrum_reconstruction_display` 校验文件、三个方向和间距；
`multispectrum_water_reference` 校验 BHC 参考、cm/mm 单位换算及水/零衰减 HU 基准。

材料列表弹出宽度限定为 240..430 像素，并约束在参数栏可用宽度附近；
长名称采用省略号，悬停显示完整名称、ID、密度和组成，提示框也限制换行宽度。

### 页面布局与样式

左侧页签冻结在参数栏顶部，每个页签内容独立滚动；底部计算/重建按钮固定。
右侧图像顶部保留 Projections / Reconstruction 页签：前者显示两幅投影图，后者显示三幅重建切片。
切换保留各自图像缓存，不重新计算，也不删除投影区域。
主题统一蓝灰背景、青色强调、圆角控件和较宽间距；主字体 20 px，图像标题 19..20 px。
参数栏占窗口宽度 30%，三维区域与图像区域分别占 35%。

启用重建散射校正时，Correction air / Correction model 可分别选取空场 raw 和模型 JSON；
空值默认沿用前投 air_output_file / scatter.model_file，也可按 Use forward scatter inputs 显式同步。
参数序列化会保存两条路径及校正参数；执行校验会区分缺失空场、缺失模型和数值参数错误。
选择了前投模型并不代表其格式或标定范围一定适用于校正，实际模型校验仍由运行后端执行。

### 采样、余晖和校正高级参数

Effects 的 Ray sampling 控制每像素射线数与采样种子；量子噪声种子也可编辑。
ImGui 读取与运行均保留采样数，不再强制为 1，保存的配置记录实际使用值。
旧纯 VTK 预览路径仍保留其原有默认策略。

余晖可编辑比例、初始状态、帧时间、曝光时间、分量权重和时间常数。
可以增删分量或归一化权重；计算前要求权重和为 1、时间常数正值、曝光不超过帧时间。
BHC 可编辑多项式阶数、手动系数、截断上下限、参考能量及拟合水板最大长度/采样数。
截断作用于校正后的线积分，是无量纲范围；水板最大长度单位为 mm。

散射校正可选择 fixed / adaptive_experimental；固定迭代次数和阻尼独立于自适应设置。
引导开关及 U/V 平滑 sigma（像素）可编辑。
自适应可编辑迭代次数、初始/最小阻尼、阻尼衰减，以及低分位数、风险阈值、正性余量。
校正倍数、掩膜阈值和核能量评估分段数也已开放。JSON 中的大量离散核仍通过文件载入。
运行自适应需开启 `MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER=ON`；本机演示构建已开启。
回归覆盖这些值的 TOML 读写、8 射线/像素与余晖的 UI 全链路、散射引导及自适应求解。

### 图像抽屉与真实重建进度

几何画布始终占左侧参数栏之外的完整区域。右侧图像默认收在抽屉中，
通过 Projection images / Reconstruction images 打开；Close 收起。
抽屉叠加在右侧，不缩小几何画布；打开时仅平移观察中心以避开遮挡，保持视角与缩放。
打开后的页签仍可切换两幅投影图和三幅重建切片，底部可直接控制层号、HU 窗宽窗位。
加载已有结果时默认保持抽屉收起；新重建完成会打开结果抽屉。

重建进度按当前阶段报告实际完成数：BHC 拟合样本、投影校正视图、
平板圆扫描 FDK 已处理视图、迭代后端已完成迭代/子集更新，以及输出保存。
有计数的阶段显示完成数/总数和百分比，同时保留耗时；初始化等不可量化阶段显示 Working。
一次性执行的完整体积算法以执行开始/完成报告，不用耗时猜测其中进度。

公共 SExecutionRequest 添加可选进度回调尾部，通过 struct_size 判断；旧请求尺寸仍可使用。
回调在 execute 线程中、GPU 工作完成后同步调用，关闭回调时保持原异步内部算法策略。
回调数据只借用到 execute 返回，回调不得抛异常。GUI 回调仅更新线程安全进度状态，
不从工作线程调用 VTK。回归覆盖旧 ABI 尺寸、真实迭代计数与回调开启前后的数值一致性。
