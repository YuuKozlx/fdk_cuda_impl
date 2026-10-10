# 标定回归容器

该镜像是 Python 标定链和独立 C++ 标定项目的统一 Linux 测试环境。源码与 RAW 数据从宿主机只读或读写挂载，镜像内包含 Python 3.11、NumPy、SciPy、pytest、CMake、Ninja、GCC 和 Eigen。

容器只负责构建与测试，不改变标定算法，也不要求 Windows 安装 Python、Visual Studio 或 vcpkg。构建产物统一放到 `out/cbct-calibration-cpp-linux/`，不要写入源码目录。

在仓库根目录构建镜像：

```powershell
docker build -f tools/cbct_calibration/container/Dockerfile `
  -t fdk-test-cbct-calibration:test .
```

完整 Python 回归：

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace `
  fdk-test-cbct-calibration:test `
  python -m pytest -q tools/cbct_calibration
```

C++ 独立项目从空目录配置、构建并运行 CTest：

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace `
  fdk-test-cbct-calibration:test sh -lc `
  "cmake -S tools/cbct_calibration_cpp -B out/cbct-calibration-cpp-linux -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build out/cbct-calibration-cpp-linux && ctest --test-dir out/cbct-calibration-cpp-linux --output-on-failure"
```

回归包含 Python 全链路、C++ DLT/联合拟合、Cho 12+12 与 Yang 6+6 逐点掩码，以及合成 RAW 的粘连后恢复。仓库没有对应真实 RAW 时，只报告合成测试结果，不用缓存或伪造真实数据结论。
