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

测试见 `tests/test_scalar_fp.cpp`，含独立解析/交点参考、分批一致性及可选 DLL 对照。性能尚未做完整头部基准。
