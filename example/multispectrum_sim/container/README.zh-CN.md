# Linux CUDA 容器构建

此目录提供多能投影和重建的命令行容器构建入口。VTK、ImGui 客户端不在这个构建范围内。Windows 原生构建仍见 [构建文档](../../../doc/build-ui.md)。

需要 Docker、NVIDIA Container Toolkit、兼容的 NVIDIA 驱动，以及为 Linux 准备的 matio SDK（项目要求 1.5.28 或更新，含 CMake 包配置及运行库）。镜像不会下载或构建该 SDK。源码和输入输出目录挂载到容器，不打包进镜像。

以下在仓库根目录的 PowerShell 中执行，将 `<linux-matio-sdk>` 替换成 SDK 的实际目录。Linux SDK 不能使用 Windows 的 DLL 或 import library 替代。

```powershell
docker build -f example/multispectrum_sim/container/Dockerfile -t fdk-test-multispectrum:cuda12.6 example/multispectrum_sim/container
docker run --rm --gpus all -v "${PWD}:/workspace" -v "<linux-matio-sdk>:/opt/matio:ro" -w /workspace fdk-test-multispectrum:cuda12.6 bash -lc 'cmake -S . -B out/build/linux-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/matio -DBUILD_MULTISPECTRUM_SIM=ON -DBUILD_VTK_VISUALIZATION=OFF -DBUILD_IMGUI_GUI=OFF -DBUILD_TESTING=OFF && cmake --build out/build/linux-cuda --target multispectrum_sim -j2'
```

`86` 是 GPU 架构示例，按目标显卡调整。运行时同样需要挂载 SDK；若 matio 是共享库，需将其运行库目录加入 `LD_LIBRARY_PATH`：

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" -v "<linux-matio-sdk>:/opt/matio:ro" -w /workspace -e LD_LIBRARY_PATH=/opt/matio/lib fdk-test-multispectrum:cuda12.6 out/build/linux-cuda/example/multispectrum_sim/multispectrum_sim example/multispectrum_sim/configs/project-and-reconstruct/analytic/water-circle.toml
```

配置中的模体需预先准备，TOML 相对路径始终相对于 TOML 文件。参见[配置入口](../configs/README.zh-CN.md)和[内置体素数据说明](../../../data/phantoms/README.zh-CN.md)。也可替换命令末尾的 TOML，分别执行纯投影、纯重建或完整流程。散射模型需自行提供，并配置容器内可访问的路径。

该容器入口从旧分支迁移并按当前目录整理；尚未在当前版本完成 Linux 容器构建或 GPU 运行验证。
