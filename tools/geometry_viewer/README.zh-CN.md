# VTK 几何与体数据查看器原型

这是 `ai/vtk` 分支上的独立 C++ 查看器，使用本库实际几何 builder。默认关闭，主库不链接 VTK，也不引入 Qt。可单独构建，无需编译主库/CUDA 算法；CUDA Toolkit 仅提供现有几何头文件使用的向量类型。

## 能看什么

- Flat/Cyl Circular、Flat/Cyl Helical、SourceOrbit、NikonTilted 六种几何。
- 当前帧源点、真实平板或柱面探测器、完整源轨迹、中心/边缘射线和世界轴。
- 随帧更新的源与探测器局部坐标轴，支持 L 键隐藏/显示并随几何导出。
- 重建体积的实际外边界，包括体积中心与非等体素。
- 内置示意模体的 GPU 体绘制，或外部 float32 RAW 重建体积。
- 帧滑条、键盘模型切换、鼠标旋转/缩放/平移、PNG 截图和 VTK 文件导出。
- M 键切换“模体固定、机架运动”与“机架固定、模体运动”的等价显示。

示意模体用于查看器演示，不是重建结果。可视化几何不代表所有算法组合已在公共 Session 放行。几何计算调用 include/YKCBCT 的规则 builder 和 src/NonStandardFpBp 的非标准 builder；不在渲染代码中重新推导扫描轨迹。

## 构建

准备 VTK 9.3 或更新的 C++ 开发包和兼容的 C++17 工具链。只需本文 CMake 中列出的渲染、交互、图像/XML I/O 模块，不需要 Qt/Python 包装模块。官方接入说明：[C++ / CMake](https://docs.vtk.org/en/latest/getting_started/using_cpp.html)。

单独构建（在 VS x64 Developer PowerShell 中）：

```powershell
cmake -S tools/geometry_viewer -B out/build/vtk-viewer -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DVTK_DIR=G:/path/to/vtk/lib/cmake/vtk-9.6
cmake --build out/build/vtk-viewer --parallel 4
```

将 VTK 的运行 DLL 目录设置为 `$env:VTK_RUNTIME_DIR` 或加入 PATH。上面的 VTK_DIR 为占位路径，须替换为含 vtk-config.cmake 的实际目录。

主工程也可以显式打开 `-DBUILD_VTK_VIEWER=ON`，并只构建 geometry_viewer 目标；默认 OFF 时不会寻找 VTK，不改变主 DLL 的安装/导出依赖。

### 本机实验环境

本原型实际使用 VTK 9.6.2 的预编译包及配套 C++ Wheel SDK，避免本机从源码构建整个 VTK。SDK 用于头文件/CMake/链接，运行库使用配套安装包；官方说明见 [Wheel SDKs](https://docs.vtk.org/en/latest/advanced/wheel_sdks.html)。本机包和输出全部在忽略的 out/ 下，未写入全局 Python 或主工作区。

```powershell
python -m venv out/vtk-env
out/vtk-env/Scripts/python.exe -m pip install vtk==9.6.2
Invoke-WebRequest https://vtk.org/files/wheel-sdks/vtk-wheel-sdk-9.6.2-cp312-cp312-win_amd64.tar.xz `
  -OutFile out/vtk-sdk.tar.xz
New-Item -ItemType Directory -Path out/vtk-sdk -Force
tar -xf out/vtk-sdk.tar.xz -C out/vtk-sdk
cmake -S tools/geometry_viewer -B out/build/vtk-viewer -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DVTK_DIR="$PWD/out/vtk-sdk/vtk-wheel-sdk-9.6.2-cp312-cp312-win_amd64/vtk-9.6.2.data/headers/cmake"
cmake --build out/build/vtk-viewer --parallel 4
```

此包路径特定于 Windows x64 / Python 3.12。其他环境使用对应开发包，不应硬改成相同路径。运行辅助脚本会优先使用 VTK_RUNTIME_DIR；未设置时寻找本工作树 out/vtk-env 的 vtk.libs。

## 查看和导出

```powershell
./tools/geometry_viewer/run-viewer.ps1 -Geometry cyl-helical -Frame 30
./tools/geometry_viewer/run-viewer.ps1 -Geometry nikon-tilted
./tools/geometry_viewer/run-viewer.ps1 -Geometry cyl-helical -Frame 30 `
  -Screenshot out/viewer-preview/cyl-helical.png `
  -ExportDirectory out/viewer-preview/cyl-helical
```

原生程序完整参数见 geometry_viewer.exe --help。常用参数：

| 参数 | 定义 |
| --- | --- |
| --geometry | 六种几何名称，默认 cyl-helical |
| --views / --views-per-turn | 总视图数/每圈视图数；圆扫一圈时令两者相等 |
| --frame | 当前帧，从 0 开始 |
| --sid / --sdd / --du / --dv | mm；du 在 Cyl 中是弧长采样步长 |
| --radius / --pitch / --tilt | 柱面曲率半径 mm、螺旋 mm/圈、Nikon 初始倾斜角度（度） |
| --size nx ny nz | 体素数，原型每维 2..512，总量最多 128M float 元素 |
| --spacing dx dy dz / --center x y z | mm，均按 X/Y/Z 顺序 |
| --raw | 连续原生 little-endian float32 [z][y][x]，X 最快；尺寸必须精确匹配 |
| --box-only | 初始隐藏体数据，只显示几何与体积边界 |
| --screenshot / --export | PNG 文件路径/VTK 输出目录 |
| --offscreen | 完成渲染后退出，不进入交互循环 |
| --self-check | 调用实际模型切换、帧滑条、体数据开关回调检查交互 |
| --gantry-fixed | 从固定机架相对模式启动，也可在窗口中用 M 切换 |

快捷键：1..6 切换几何，左右键逐帧，M 切换相对模式，V 切换体数据，T 切换轨迹，L 切换两组局部坐标轴，R 重置相机，E 导出。鼠标使用 VTK trackball camera 操作。屏幕参数为当前几何，载入的 RAW 不会因切换扫描模型而变化。

### 固定机架相对模式

默认 fixed object 在物体世界坐标下显示源/探测器的运动。按 M 切为 fixed gantry：将当前帧的刚体机架姿态映射回第 0 帧，同时把模体、体积盒、射线和局部轴施加相同坐标变换，保持成像几何关系。圆轨迹中模体反向转动；螺旋中同时旋转和沿 Z 移动；NikonTilted 的初始倾斜保留，模体按对应相对姿态运动。

源和探测器局部轴在此模式下随固定机架保持不动，O-X/Y/Z 是随模体移动的原物体坐标轴。T 显示/隐藏模体中心轨迹；默认模式的 T 对应源轨迹。模式切换仅改变显示参考系，不修改实际 builder 输出或 RAW 标量值。

SourceOrbit 中源单独运动、探测器固定，两者没有共同的刚体机架姿态，因此不能同时固定源和探测器。它不提供此相对模式；选择模型 5 会自动回到默认模式，M 会显示说明并保留原几何。

```powershell
./tools/geometry_viewer/run-viewer.ps1 -Geometry cyl-helical -Frame 30 -GantryFixed
```

导出保存当前参考系：geometry.vtm 中几何与窗口一致；volume.vti 的 origin 和 Direction 矩阵保存模体显示姿态，spacing 和体素值不变。scene.txt 的 reference 为 world 或 gantry-fixed。相对 VTI 是有方向的网格，不能忽略 Direction 后当成未经旋转的 RAW 使用。

### 局部坐标系

探测器原点在采样阵列中心；柱面取表面中心而非圆柱轴心。D-U 随通道索引递增，D-V 随行索引递增，D-N=U×V 指向探测器内侧。三个方向分别为红、绿、蓝，对应局部 X/Y/Z。

源端原点在实际焦点。S-R（局部 Y）沿源→探测器中心，S-U（局部 X）是探测器 U 在射线垂直平面上的归一化投影，S-V（局部 Z）=U×R。源坐标采用 U/R/V 的右手序，颜色为红/绿/蓝；它是从射线几何派生的坐标架，不表示库中另有射线管物理朝向或独立 roll 参数。

两组坐标使用世界 mm，与体积和实际 builder 输出一致；改变帧或切换几何时同时更新。导出轴为三条带 AxisRGB 和 AxisLabels 的线段，方向/位置与窗口箭头相同。

导出 geometry.vtm 及其关联 .vtp（体积盒、源点球、当前帧探测器、源轨迹、源局部坐标轴和探测器局部坐标轴），另写 volume.vti 与 scene.txt。用 ParaView 打开 geometry.vtm 和 volume.vti，可分别设置几何表示与体绘制；移动文件时应保持关联目录。VTI origin 为第一个体素中心，体积盒边界额外包含半个体素，二者不是坐标错位。

## 原型边界

当前通过命令行参数生成实际 builder 几何；尚未读取应用 TOML 或任意校准 geometry 文件。源/探测器的 offset/tilt 高级字段未全部提供命令行入口。SourceOrbit 半径暂设为 SID 的 15%；规则轨迹默认 180 帧、每圈 90 帧。RAW 从磁盘读取到 Host，再由 VTK 上传渲染，不直接共享重建 CUDA 缓冲。

验证记录与自动检查见 tests/verify_exports.py：六种几何离屏截图和 XML 文件读回、像素中心/体积边界、柱面曲率、源位置、RAW 布局与错误输入拒绝。性能、极大体积和长时间交互尚未做专项测试。

## 本机验证

2026-10-11，Windows / MSVC 19.44 / CUDA Toolkit 12.6 / VTK 9.6.2：独立 C++ 构建及主工程可选目标构建均通过。六种模型的实际离屏渲染、源坐标、柱面半径、VTI/VTM 读回、非零体积中心和非等体素 RAW 布局均通过检查；错误尺寸/视图/距离/pitch 会拒绝。局部轴检查覆盖原点、单位长度、正交/右手性、源射线方向、探测器法向及不同帧更新；模型切换、帧滑条和体数据/局部轴开关使用真实 VTK 回调进行检查。

```powershell
$env:PATH = "$PWD/out/vtk-env/Lib/site-packages/vtk.libs;" + $env:PATH
out/vtk-env/Scripts/python.exe tools/geometry_viewer/tests/verify_exports.py `
  --exe out/build/vtk-viewer/geometry_viewer.exe --output out/viewer-checks
```

报告为 out/viewer-checks/verification.json，截图为同目录六份 PNG。这是可运行原型，尚未承诺读取现有应用 TOML、接收实时重建数据或 CUDA/VTK 零拷贝互操作。

相对模式检查见 tests/verify_relative.py：五种刚体几何的第 0/30/61 帧，验证源/探测器固定、模体姿态、统一刚体变换、射线到体素距离不变、非零体积中心与非等体素，以及 VTI Direction 读回和真实 M 回调。报告位于 out/viewer-relative-checks/relative-verification.json。
