# 多能前投客户端：构建与运行

使用者先准备编译器、CUDA 和原生 SDK；项目通过标准 CMake 查找依赖、编译目标，不负责安装工具链、下载第三方库或管理 SDK。下面命令在仓库根目录、已初始化的 VS x64 开发终端执行。

## 环境与 SDK 要求

| 依赖 | 要求 |
| --- | --- |
| CMake / Ninja | CMake ≥ 3.24，Ninja 在 PATH 中 |
| MSVC / CUDA | C++17、CUDA17，宿主编译器与 CUDA 版本兼容；验证组合为 VS 2022 MSVC 19.44 + CUDA 12.6 |
| NVIDIA 驱动 | 兼容所用 CUDA 及显卡 |
| VTK | 原生 C++ SDK，≥ 9.3；验证版本 9.6.2 |
| matio / zlib | matio ≥ 1.5.28，支持 zlib 压缩 MAT v5；验证为 matio 1.5.28 + zlib 1.3.1 |
| Git LFS | 仅获取仓库内体素示例数据时需要，编译和空 UI 不需要 |

Windows 客户端和 SDK 统一使用 x64 / Release / MD。SDK 已有 matio CMake 包时直接复用；只有标准头文件与库时，由独立的 Findmatio.cmake 适配，不要求使用本项目制作 SDK。matio 不依赖 MATLAB 或 Python。

SDK 布局示例：

```text
<SDK>/
  include/matio.h, matio_pubconf.h, zlib.h
  include/vtk-*/...
  lib/matio.lib 或 libmatio.lib
  lib/zlibstatic.lib
  lib/cmake/vtk-*/vtk-config.cmake
  bin/vtk*.dll
```

matio 建议以静态库放入同一个 SDK，启用 zlib，关闭不需要的 HDF5/MAT v7.3。SDK 的安装、更新与 ABI 管理由使用者或 SDK 维护者负责。程序源码自带 ImGui、toml++、JSON、XCOM 和能谱生成器，不需要额外安装它们。

## 1. 配置

下面假设准备好的 SDK 放在仓库的 sdk 目录；可以换成任意其他位置。

```powershell
$env:YKCBCT_SDK_ROOT = (Resolve-Path .\sdk).Path

cmake -S . -B out/build/ui -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
  -DCMAKE_CUDA_ARCHITECTURES=native `
  -DYKCBCT_SDK_ROOT="$env:YKCBCT_SDK_ROOT" `
  -DBUILD_MULTISPECTRUM_SIM=ON `
  -DBUILD_VTK_VISUALIZATION=ON -DBUILD_IMGUI_GUI=ON `
  -DBUILD_DLLTEST=OFF
```

native 为当前机器的显卡编译；面向其他 GPU 分发时，改成覆盖目标设备的架构列表，例如 75;80;86，并确认 CUDA 支持这些架构。修改编译器、CUDA 或 SDK ABI 后使用新的构建目录，避免混用缓存。

SDK 不在一起时，也可用标准 CMAKE_PREFIX_PATH 提供多个前缀，或分别设置 VTK_DIR、matio_DIR。不要把另一台机器的 CMakeCache.txt 或构建目录复制过来当作 SDK。

## 2. 编译

```powershell
cmake --build out/build/ui --target multispectrum_sim --parallel 8
```

输出为 out/build/ui/example/multispectrum_sim/multispectrum_sim.exe。需要 RAW → MAT 打包工具时，再构建 multispectrum_pack_mat 目标。

## 3. 运行

保证 SDK 的 bin 和 CUDA 的 bin 在 PATH 中。SDK 路径沿用配置时的值：

```powershell
$env:PATH = "$env:YKCBCT_SDK_ROOT\bin;$env:PATH"
$exe = '.\out\build\ui\example\multispectrum_sim\multispectrum_sim.exe'

& $exe --gui
# 或加载自己的配置
& $exe --gui .\path\to\scan.toml
# 或直接使用内置示例
& $exe --gui-example water-circle .\out\examples\water
& $exe --gui-example brainweb-normal .\out\examples\brainweb
& $exe --list-examples
```

空界面入口是 --gui；无参数运行只显示用法。选择 Example 后先加载资源，Compute / Forward projection 才开始计算。

Water 和几何模体由程序生成，不需要体素数据文件。BrainWeb/ICRP 需要对应的数据包，详见 [内置体素数据](../data/phantoms/README.zh-CN.md)。迁移时一起复制 data/materials 与 data/phantoms，并将 YK_EXAMPLE_DATA_ROOT 设置为包含它们的 data 根目录。数据不嵌入 exe。

示例输出写入指定的 out 目录。自定义 TOML 的相对路径按配置文件所在目录解析。

## 可选开发测试

配置时启用 BUILD_TESTING=ON（默认开启），然后按需编译和执行：

```powershell
cmake --build out/build/ui --parallel 8 --target `
  multispectrum_ui_config_test multispectrum_examples_test multispectrum_mat_volume_test `
  multispectrum_icrp_test multispectrum_mpr_test multispectrum_mpr_interaction_test

ctest --test-dir out/build/ui --output-on-failure `
  -R '^(multispectrum_ui_config|multispectrum_examples|multispectrum_mat_volume|multispectrum_icrp|multispectrum_icrp_source|multispectrum_mpr|multispectrum_mpr_interaction)$'
```

这七项检查不自动运行完整扫描。底层 CUDA 数值和集成测试见 [测试与验证](testing-and-validation.md)。

## CMake 职责

- 顶层 CMakeLists.txt：主库、功能开关、SDK 搜索前缀与子模块编排。
- example/multispectrum_sim/CMakeLists.txt：应用目标与功能模块组合。
- example/multispectrum_sim/cmake/Examples.cmake：内置示例、ICRP 和对应测试目标。
- example/multispectrum_sim/cmake/MatVolume.cmake：MAT 读取目标；Findmatio.cmake 只负责查找和适配 SDK。
- example/multispectrum_sim/visualization/CMakeLists.txt：VTK/ImGui 显示目标。

模块通过 CMake target 传递头文件与链接依赖。matio、VTK 仅属于应用层，关闭 BUILD_MULTISPECTRUM_SIM 的本体 DLL 构建不需要它们。扩展处理功能不应增加下载器、环境初始化脚本或另一套构建调度。

## 常见问题与验证范围

找不到包时检查 SDK 布局、前缀和 ABI；缺 DLL 时检查 SDK/CUDA 的 PATH、驱动和 MSVC x64 运行库；MAT 是 LFS pointer 时获取实际数据；exe 被占用时先关闭对应界面。Windows ImGui 输入桥不代表 Linux UI 已完成支持。

2026-10-17：上述 MSVC/CUDA/VTK/matio 组合在全新目录完成客户端构建，七项定向测试通过；BrainWeb 资源加载与界面启动通过。matio SDK 复制到含空格的新目录后，头文件、库查找和压缩读写通过。VTK 复用已准备的原生 SDK；未验证所有工具链或另一台物理机器。
