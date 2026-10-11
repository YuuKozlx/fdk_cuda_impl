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
