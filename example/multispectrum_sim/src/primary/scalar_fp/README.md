# 独立标量前投

`multispectrum_scalar_fp` 为多能前投应用拥有的独立 CUDA 静态库。仅链接 CUDA runtime，不引用底层 DLL 内部头文件、源码、Session 或调度状态。项目拆分时可整体移走本目录及 `include/primary/ScalarProjector.hpp`。

## 来源与范围

初始内核来自 fdk-test `f63a8fe`：

- Siddon：`src/FlatFpBp/FP/kernels/YkFPSiddonKernel.cu` 的标量遍历。
- Joseph：`src/FlatFpBp/FP/kernels/YkFPKernels.cu` 的纹理采样内核。
- 三轴坐标映射：`src/common/cuda/operators/YkOperatorKernelTypes.cuh`。

保留原有数值计算；去掉底层类型、策略和日志依赖，独立 launch 按视图主轴分发并支持调用者分批。后续由应用独立维护，不自动跟随底层变化。对照测试仅通过 DLL 公共接口生成参考。

支持平板逐视图向量几何、Siddon 和 Joseph。圆扫、螺旋、倾斜轨迹由向量数组表达。圆柱探测器、footprint、BP、标签材料积分及多能积分不在此模块内。现有骨硬化和 EBHC 默认执行路径未切换。

## 坐标约定

`View` 为源位置、探测器位置、列步进 u、行步进 v；单位均为 mm，u/v 长度包含像素间距。默认探测器位置是整幅平板中心，对应 ASTRA cone_vec。`Pixel00` 可直接输入首像素中心，避免中心转换的浮点往返误差。

`Grid.center` 是体积几何中心；内存按 `[z][y][x]` 连续，spacing 可非等距。体积有姿态时，调用者将所有源/探测器点及步进向量逆变换到体积局部坐标。源应在体积外、探测器在体积另一侧。

`prepareViews` 在 CPU 上生成设备记录；Siddon 保持物理坐标，Joseph 转为居中体素坐标。记录必须与相同方法、体积网格及探测器尺寸一起使用，不可混用。

## 资源与调用

1. 调用 `prepareViews`，由调用者上传记录与标量体积。
2. Joseph 由调用者创建 float32 CUDA array 纹理：非归一化坐标、线性插值、各轴零边界。Siddon 使用设备线性体积指针。
3. 调用 `launch`；输出按 `[view][v][u]` 连续。默认覆盖，`accumulate=true` 累加。分批时偏移记录及输出指针，并传入本批帧数。
4. 调用者负责流同步、异步错误检查与所有资源寿命。launch 不分配或传输数据，也不隐式同步；输入、输出和几何缓冲不能重叠。

Joseph 支持每体素 1/2/4 步，Siddon 为精确体素交段、只接受 steps=1。Joseph 的零边界纹理和超采样会产生原内核相同的边界积分偏差，不能视为 Siddon 的精确体素模型；也不承诺与任意 BP 严格伴随。没有 detector/focal supersampling。

测试见 `tests/test_scalar_fp.cpp`，含独立解析/交点参考、分批一致性及可选 DLL 对照。

## 完整体积性能基准

Siddon 支持 `SiddonMemory::Linear`（默认，线性设备缓冲读取）与 `SiddonMemory::Texture`（三维点采样纹理读取）。后者不做插值，不改变遍历或积分公式；调用者提供非归一化坐标、float32、point filter、zero border 的三维纹理，可不提供线性体积指针。Joseph 仍使用 linear filter，不能复用点采样纹理。两类纹理对象可共享同一个 CUDA array，避免重复存储体积。

DLL 的正式 Siddon 算子走点采样纹理路径，旧的独立版只包含线性读取路径。因此不能拿旧独立版的性能直接代表 DLL；尤其是规则体积尺寸下，两种读取路径的缓存行为可能差异很大。`scalar_fp_siddon_texture` 测试覆盖独立交点参考、线性/纹理数值一致、分批和累加；完整头部基准还逐批比较全部纹理结果和线性结果。

手动构建 `multispectrum_benchmark_scalar_fp` 目标，运行：

```text
multispectrum_benchmark_scalar_fp <projection.toml>
```

该工具读取 TOML 的原始 uint8 RAW 标签、材料和网格，将标签转换为 70 keV 的浮点线衰减体积（mm⁻¹），沿相同的平板圆扫/螺旋几何测 Siddon 和 Joseph。它不是多能仿真：每像素一条中心射线，无焦点、探测器随机采样、能谱积分或计数噪声。暂不支持 MAT 输入、圆柱/planar/tilted 几何。

工具保留原体积尺寸；每次仅使用最多 64 MiB 投影工作区，不保存全套投影。一次预热后三次扫描的 CUDA event 中位数作为 `kernel_median_ms`；另行报告体积上传、Joseph 纹理初始化及全套投影下载耗时。计时包含 launch 内部的输出清零和调度，不包含数据准备、文件读取或主机输出统计。所有下载结果检查有限值和非负性；这不替代解析与 DLL 数值回归测试。对比多能前投时必须注明材料数、能量数、随机采样数和计算任务的不同，不能将两者输出视为等价。
