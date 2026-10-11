# 构建与接入

本篇负责本体库构建和外部接入。算法组合见 [算法](algorithms.md)，测试执行见 [测试与验证](testing-and-validation.md)。事实来源：[顶层 CMake](../CMakeLists.txt)、[预设](../CMakePresets.json)、[安装包配置](../cmake/YKCBCTConfig.cmake.in)。

## 工具链与依赖

| 项目 | 当前要求或用途 |
| --- | --- |
| CMake | 最低 3.24 |
| C++ / CUDA | 语言标准均为 17；需要 CUDA 编译器和兼容的宿主编译器 |
| CUDA Toolkit | 必需；主库公开依赖 cudart，私有依赖 cuFFT |
| OpenMP CXX | 必需；主库私有链接 |
| Ninja / Visual Studio | Ninja 为单配置；Visual Studio 为多配置 |
| GoogleTest | `BUILD_UNIT_TESTS=ON` 时需要 CMake 包 |
| toml++ | `BUILD_TESTS=ON` 时需要；可来自 CMake 包或指定 include 根目录 |
| CLI11 / spdlog / fmt | 当前仓库包含所需内部头文件 |
| Matplot++ / gnuplot | 仅绘图工具开关开启时需要 |

Windows 请从 VS x64 Developer PowerShell 执行，确保 `cl`、`link`、`rc` 可见。CUDA Toolkit 与 NVIDIA 驱动分别负责编译和 GPU 运行；本库没有 CPU 重建后端。这里不指定未经本机验证的“最低 CUDA 版本”。

## 功能开关

默认值以顶层 CMake 为准，预设或既有缓存可能覆盖它们。

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `YKCBCT_BUILD_HELICAL` | ON | 编入 wFBP 等螺旋解析源码；部分 Cyl/Heli 测试也受此开关控制 |
| `YKCBCT_BUILD_CVP` | OFF | 实验 CVP 正投；不增加公共 `EProjectionModel` 选项 |
| `BUILD_TESTS` | OFF | `ykcbct_manual_tests` 内部 CUDA/数值/集成测试 |
| `BUILD_UNIT_TESTS` | OFF | `YKCBCT_unit_test`，当前主要测试 Logger |
| `BUILD_DLLTEST` | ON | `dlltest`、`dlltest_shepp_logan` 公共接口测试 |
| `BUILD_MULTISPECTRUM_SIM` | ON | 独立多能谱应用；只构建本体库时显式关闭 |
| `BUILD_MULTISPECTRUM_PLOT_TOOLS` | OFF | Matplot++ 分析工具 |
| `BUILD_VTK_VIEWER` | OFF | 独立 VTK 几何/体数据查看器；仅打开时寻找 VTK，不向主库增加链接依赖 |
| `BUILD_VTK_VISUALIZATION` | OFF | 独立实时显示模块，接入多能前投；不依赖查看器源码，不向主库增加 VTK 依赖 |

`CMAKE_CUDA_ARCHITECTURES` 未指定时默认 `75;80;86`。仓库 Release 预设指定 `86`，不是运行时自动适配所有 GPU。Ninja 未指定构建类型时默认为 Release。MSVC Release 使用 AVX2；GNU/Clang Release 使用 `-march=native`，跨机器分发时需检查 CPU 兼容性。CUDA Release 使用 fast math，数值比较应保留浮点容差。

关闭 `YKCBCT_BUILD_HELICAL` 会去除公共 WFBP 后端，但不能据此推断所有螺旋迭代入口均不可用；应结合实际后端与测试配置判断。

## 三种构建方式

### 最小本体库

```powershell
cmake -S . -B out/build/core -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DBUILD_MULTISPECTRUM_SIM=OFF -DBUILD_DLLTEST=OFF
cmake --build out/build/core --target YKCBCT --parallel 4
cmake --install out/build/core --prefix install/core
```

### 仓库预设

`x64-Release`、`x64-RelWithDebInfo`、`x64-Debug` 使用 Ninja 和 `$env:VCPKG_ROOT` 下的 toolchain。先设置该变量，再运行：

```powershell
cmake --preset x64-Release -DBUILD_MULTISPECTRUM_SIM=OFF
cmake --build out/build/x64-Release --target YKCBCT
```

预设定义的是 configure preset，当前没有同名 build preset，因此不能把上述第二行直接替换为 `cmake --build --preset x64-Release`。已有构建目录可能服务于其他开发工作；变更开关时优先使用独立目录。

`vcpkg.json` 包含 gtest、tinytiff、tomlplusplus，绘图通过 `plotting` feature 提供。最小主库并不需要把 manifest 中所有依赖都当成算法运行依赖。

### 完整本体测试构建

下面命令假设依赖 CMake 包可被发现。可通过 vcpkg toolchain 或 `CMAKE_PREFIX_PATH` 指向已安装的依赖前缀。

```powershell
cmake -S . -B out/build/docs-check -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DYKCBCT_BUILD_HELICAL=ON -DBUILD_TESTS=ON `
  -DBUILD_UNIT_TESTS=ON -DBUILD_DLLTEST=ON -DBUILD_MULTISPECTRUM_SIM=OFF
cmake --build out/build/docs-check `
  --target YKCBCT ykcbct_manual_tests YKCBCT_unit_test dlltest dlltest_shepp_logan `
  --parallel 4
```

没有 toml++ CMake 包时，设置 `YKCBCT_TOMLPLUSPLUS_INCLUDE_DIR`，其下必须存在 `toml++/toml.hpp`。GoogleTest 仍需独立找到。

如果通过已安装的依赖前缀发现动态链接版 GTest，Windows **构建阶段**就需要该前缀的 `bin/` 在 PATH 中：gtest_discover_tests 会在链接后立即执行程序。否则可能报 0xc0000135，即使链接已经成功。本次验证使用仓库 `thirdparty/vcpkg_installed/x64-windows` 作为 CMAKE_PREFIX_PATH，并把它的 bin 加入当前进程 PATH；该目录只有在依赖已安装时才有效。

`scripts/build-all.ps1` 可导入 VS 环境并集中设置测试开关，但它不关闭默认多能谱应用，也不是上述最小本体构建的同义命令。应先检查脚本参数及实际缓存。

Visual Studio 生成器请在 build/install/CTest 阶段增加 `--config Release` 或 `-C Release`；可执行文件通常位于子目录的 `Release/` 中。本文其余路径均采用 Ninja，避免混用这两种产物布局。

## 安装与外部工程

安装后包含：`include/YKCBCT/`、`bin/` 中的 Windows DLL、`lib/` 中的导入库/平台库、`lib/cmake/YKCBCT/` 中的包配置。默认安装前缀为仓库 `install/`，命令行 `--prefix` 可覆盖。

外部项目 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.24)
project(cbct_client LANGUAGES CXX)
find_package(YKCBCT CONFIG REQUIRED)
add_executable(cbct_client main.cpp)
target_compile_features(cbct_client PRIVATE cxx_std_17)
target_link_libraries(cbct_client PRIVATE YKCBCT::YKCBCT)
if(MSVC)
    target_compile_options(cbct_client PRIVATE /utf-8 /Zc:__cplusplus)
endif()
```

`main.cpp` 可使用 [公共 API](public-api.md#完整公共调用示例) 的完整示例。配置时将安装前缀加入 `CMAKE_PREFIX_PATH`，或将 `YKCBCT_DIR` 指向安装后的 `lib/cmake/YKCBCT`。包配置会寻找 CUDAToolkit；公开系统几何仍使用 CUDA 向量类型，消费方需要 Toolkit 头文件，即使其缓冲区全部位于 Host。

MSVC 的 `/utf-8` 对包含中文注释的 SDK 头文件同样必要；仅把主程序保存为 UTF-8 不会自动改变编译器源字符集。`/Zc:__cplusplus` 使标准版本宏反映实际 C++17 配置。

```powershell
cmake -S path/to/client -B out/build/client -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=G:/path/to/install/core
cmake --build out/build/client
```

以上安装路径是占位示例，需换为实际绝对路径。第三方只能依赖安装头文件和导出目标，不能加入 `src/` 搜索路径来补齐 SDK。

Windows 运行前，将安装 `bin/` 加入当前进程 `PATH`，或把所需 DLL 放在程序旁；CUDA/编译器运行库也必须可加载。仓库 DLL 测试构建会复制 `YKCBCT.dll`，该操作不等于外部工程自动部署全部依赖。Linux 检查共享库搜索路径和 CUDA 运行库。

## 构建故障定位

| 现象 | 检查项 |
| --- | --- |
| nvcc 找不到宿主编译器 | VS x64 环境、CUDA 支持的宿主工具链 |
| 找不到 OpenMP | C++ 工具链与 OpenMP 支持，不要仅配置 CUDA 路径 |
| 找不到 GTest/toml++ | 测试开关、toolchain、依赖前缀或 toml++ include 根目录 |
| 编译后无法启动 | DLL/共享库及运行库搜索路径、GPU 驱动、目标架构 |
| 改开关仍沿用旧配置 | 检查 `CMakeCache.txt`；使用新目录验证 |
| 改头文件却未重新编译依赖源文件 | 检查 Ninja 的 MSVC showIncludes 前缀与编译器输出语言/编码是否匹配；本次遇到编码损坏导致依赖记录为空，需修正构建缓存并干净重建 |
| CUDA device-link 失败 | 正式 CUDA 源码是否归属主目标；见 [架构](architecture-and-development.md) |

不要通过修改开发机绝对路径到仓库源码来修复依赖发现问题。

## 可选 VTK 查看器

`tools/geometry_viewer` 提供独立 C++ 原型，直接使用本库几何 builder，不链接主 DLL 或 CUDA 算法。它也能作为 CXX-only 小工程单独配置；CUDA Toolkit 只提供几何头文件使用的向量类型。主库 BUILD_VTK_VIEWER 默认 OFF，普通构建不需要 VTK。

准备 VTK C++ 包后，可在主工程添加 `-DBUILD_VTK_VIEWER=ON -DVTK_DIR=<包配置目录>` 并构建 geometry_viewer 目标。也可直接以 tools/geometry_viewer 为源码目录配置，跳过主库编译。运行需要 VTK 对应的 DLL/共享库，操作、RAW 布局、导出与验证见 [查看器说明](../tools/geometry_viewer/README.zh-CN.md)。

多能前投的内置实时显示使用独立 `BUILD_VTK_VISUALIZATION` 开关和 `ykcbct_visualization` 模块，配置、完整体数据与投影贴图约定见 [实时显示层](../visualization/README.zh-CN.md)。两个开关均关闭时不寻找 VTK。
