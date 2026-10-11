# YKCBCT（fdk-test）

YKCBCT 是 C++17/CUDA CT 投影与重建动态库，提供平板/柱面探测器、规则圆轨迹/螺旋轨迹及两类非标准平板轨迹。公共调用入口是 `YKCBCT/interface/YkReconstructionApi.hpp` 中的 Session；安装目标为 `YKCBCT::YKCBCT`。

## 开发文档

本体库只维护以下六份详细文档。安装头文件、实际后端和测试注册共同决定当前行为；声明了参数或存在内部实现，不代表公共 DLL 已完整支持。

| 文档 | 内容 |
| --- | --- |
| [构建与接入](doc/build-and-integration.md) | 依赖、构建开关、安装、外部工程、运行环境 |
| [公共 API](doc/public-api.md) | 完整调用示例、参数消费情况、缓冲区、生命周期、分包、错误 |
| [几何与数据](doc/geometry-and-data.md) | 坐标、单位、像素/体素中心、轨迹、姿态、数据布局 |
| [算法](doc/algorithms.md) | 公共支持矩阵、实际算法路径、必要公式、近似和限制 |
| [架构与开发](doc/architecture-and-development.md) | 模块边界、资源与算子、CUDA 约定、新算法接入 |
| [测试与验证](doc/testing-and-validation.md) | 测试入口、TOML、误差指标、验证证据、待解决问题 |

## 最小本体库构建

在可找到 C++ 编译器、Ninja 和 CUDA Toolkit 的环境中运行。`86` 是本例 GPU 架构，其他设备需调整。

```powershell
cmake -S . -B out/build/core -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 `
  -DBUILD_MULTISPECTRUM_SIM=OFF -DBUILD_DLLTEST=OFF
cmake --build out/build/core --target YKCBCT --parallel 4
cmake --install out/build/core --prefix install/core
```

完整公共调用示例见 [公共 API](doc/public-api.md)。选择算法之前先查 [支持矩阵](doc/algorithms.md#公共支持矩阵)，特别是 CFDK、WFBP、非标准轨迹和设备缓冲区限制。

内部逐帧/逐包重建已覆盖 Flat FDK、SourceOrbit、NikonTilted 和柱面 FDK；xFDK 保留已有带邻域缓存的流式路径。C-FDK、螺旋 wFBP 和迭代重建保持完整输入，不纳入此次自然流式扩展。具体入口和公共/内部区别见 [流式范围](doc/algorithms.md#流式范围与当前完成状态)。

## 独立应用和工具

- [多能谱仿真](example/multispectrum_sim/README.zh-CN.md)：应用层物理模型、预处理、配置与输出。
- [模体生成器](example/phantom_generator/README.zh-CN.md)。
- [几何标定工具](tools/cbct_calibration/README.zh-CN.md)。
- [VTK 几何与体数据查看器原型](tools/geometry_viewer/README.zh-CN.md)：可单独构建，默认关闭，支持六种几何、帧切换、RAW 体绘制与文件导出。

这些目录各自维护使用说明；本体库文档不承诺其应用层 TOML 与 Yktest TOML 相同。第三方授权与来源保留在 `LICENSES/`、`thirdparty/` 及相关源码中。

## 文档维护

公共 API、几何、算法开放状态、构建或测试入口发生变化时，同一变更必须更新对应文档。一个事实只在一份详细文档中定义，其他位置链接引用。测试结果必须注明代码版本、构建配置、硬件、输入和指标；历史结果不能作为当前版本通过验证的证据。
