# 重建前处理开发说明

## 投影域平滑与引导滤波

`reconstruction.preprocessing.filters` 是有序的逐帧二维滤波 Stage 列表，默认空列表。
复用现有 `image::makeProjectionStage` 的 CUDA 均值、高斯和自引导滤波，不混合相邻角度：

```toml
[reconstruction.preprocessing]
enabled = true

[[reconstruction.preprocessing.filters]]
operation = "gaussian"
radius_pixels = 5
sigma_pixels = 1.5

[[reconstruction.preprocessing.filters]]
operation = "guided"
radius_pixels = 4
epsilon = 0.0001
```

启用散射时处理顺序为 `p → 恢复能量 → 散射扣除 → 除以空场 → filters → 乘回空场 → -log(T) → BHC`。
未启用散射时则为 `p → exp(-p) → filters → -log(T) → BHC`。
滤波使用归一化透射率，`epsilon` 是透射率平方的量纲，不随 eV、keV 或每历史能量的绝对比例改变。
半径、sigma 以探测器像素计，sigma 只影响高斯滤波；引导滤波为标准 self-guided，
并非拿无噪声 primary 作为外部引导。转换回对数时采用 `T >= 1e-8` 的数值下限。
`mean`、`gaussian`、`guided` 是当前支持的 operation，顺序由数组次序决定。

该列表平滑扣除后的测量投影，避免直接对噪声很大的残差取对数；原有 `scatter.guide` 则只平滑散射估计的输入，
最终残差仍扣自测量图像，两者含义不同。滤波不是标定，也不保证消除模型失配或光子饥饿。
滤波会改变空间分辨率，比较图像时应给 primary 对照施加相同滤波。

全链在 GPU 上复用工作区，不增加逐 Stage 的 CPU 往返。单独启用 filters 无需能谱或空场。
顺序和参数均纳入校正缓存；UI 保存/加载 TOML 保留这些设置，目前尚无专用 UI 编辑控件。

## 文件职责

- `PreprocessPipeline.cpp`：阶段编排、数据域检查、Host/Device 状态转换、文件 I/O、进度和计时。
- `Stages.cpp`：内置能量恢复、散射、转对数/BHC 阶段；固定、自适应和引导散射均通过设备接口连接。
- `CudaPixelOps.cu`：驻留帧工作缓冲、能量/对数转换、双精度 Horner 与截断、有效性检查。
- `input/calibration/WaterBhcCalibration.cpp`：独立 CPU 水板标定任务，不由前处理调用。`WaterBhcTable` 加载完成的标定表，逐像素应用仍在 CUDA 完成。
- `DeviceMemoryPool.cpp`：进程内、按 CUDA 设备隔离、线程安全的缓冲池。无 VTK 依赖。

应用入口 `prepareReconstructionProjection()` 内部调用 `Pipeline`。原来的校正数值定义不变：恢复 `air*exp(-log)`，散射正值下限 `air*1e-8`，转对数比例下限 `FLT_MIN`，BHC 使用双精度多项式并按配置截断；多项式溢出必须报错，不能被截断掩盖。

## 插件约定

实现 `IStage` 后通过 `Pipeline::append(std::unique_ptr<IStage>)` 加入，执行位置在内置校正链之后。阶段通过统一的 `contract()` 声明输入/输出数值域（LogAttenuation 或 EnergySignal）及 Host/Device 位置；内置校正明确声明 Device。前处理输出必须为 LogAttenuation，域不匹配在读取帧前报错。

`prepare()` 每次任务一次，用于常量上传和工作区准备；`process()` 每帧一次；`reset() noexcept` 在成功及异常退出时调用。reset 必须可处理尚未准备或准备到一半的对象，不得依赖保存过的 Context。

Frame 的位置表示**当前权威副本**，另一个副本可能已过时。流水线按阶段要求才进行传输；阶段执行后按声明更新状态。CPU 阶段修改 `*host_data`，GPU 阶段修改 `device_data`；不得更换存储指针、改变帧大小、保留跨帧指针。不同输出位置的阶段必须自己完成该次位置转换。GPU 指针使用当前 CUDA 设备及默认流；其他流上的处理必须在返回前与默认流建立依赖。

Context 和 Frame 均为借用对象，只在回调期间使用。Pipeline 本身不支持并发；每个重建工作线程使用独立实例。显存池在不同实例之间共享，不共享数据内容。

## 显存池和 VTK 预留

`yk::spectral::cuda_resources::initializeDevicePool(sizes)` 同时预留所列缓冲；重复调用可复用缓存。`poolAllocate/poolRelease` 成对使用，活动缓冲不会再次借出。归还时记录默认流完成事件，下一借用方等待该事件，防止跨任务覆盖尚未完成的 GPU 运算。所有新借用者都必须重新初始化数据，池不保证内容为零。

新分配按实际可用显存检查，默认留下 `max(512 MiB, 总显存/5)`，小显卡最多预留一半；池活动与缓存之和最多为总显存 60%，空闲缓存上限 256 MiB。这里是池的占用约束，其他库仍可使用预留空间，并非 GPU 驱动级的专属分区。VTK 和重建 DLL 的显存不由此池分配。

尺寸或配置变化时复用容量足够的空闲块，缺少的部分才新增；不足时先释放当前设备的空闲缓存，再检查预算。超预算报错而非占用预留区。`yk::spectral::cuda_resources::trimDevicePool()` 主动清理当前设备空闲缓存，活动借用保持有效。`yk::spectral::cuda_resources::devicePoolStatistics()` 提供真实 cudaMalloc 次数、复用次数及活动/缓存字节。FFT 显式关闭 cuFFT 自动工作区分配，并从池提供工作区；自适应排序临时缓冲同样使用池。

## 传输及性能边界

一般 CUDA 校正链每帧一次上传、一次最终下载；单帧 air 文件只上传一次并复用于所有角度，多帧 air 随角度更新。正式 fixed 的掩码及边缘窗口均在 GPU 生成，不下载掩码。实验 adaptive 目前仍下载 uint8 掩码并在 CPU 生成边缘窗口；自适应求解回传小规模收敛统计，不回传完整浮点图像。

重建入口通过 `ProjectionCache` 将校正结果持久保存到输入旁的 `.corrected.raw`，配套 JSON 验证输入/空气场/散射模型/能谱文件元数据、探测器和校正参数。匹配时直接复用，改变重建算法或网格不重复校正。文件先写入独立 partial 文件，成功关闭并验证输入未改变后发布；失败的 partial 文件自动清理。JSON 只在完整 RAW 发布后写入。缓存版本需在内部校正数值语义变化时递增；每个输入仅保留最新一份结果。缓存命中仍由重建 DLL 读取并上传，这一边界没有改为直接传递显存。VTK 图像显示也保留自己的上传过程。

日志分别记录初始化、拟合、读写、传输、各阶段耗时与整帧传输次数，以及池的分配/复用次数。阶段计时同步默认 GPU 工作，避免把异步提交时间当成计算耗时。生成能谱按模型、电压、角度、bin 宽及全部过滤器参数缓存最多 8 项；文件能谱仍每次读取，参数改动不能命中旧能谱。缓存仅在同一进程有效，首次生成仍需计算。

## 验证

`multispectrum_preprocessing` 验证多项式、EID 水板拟合、引导零散射、Host/Device 权威状态及传输次数、0～16 阶数值、溢出后重试、池重复初始化、活动缓冲隔离和 VTK 显存预算。`multispectrum_scatter` 对比设备固定校正和引导接口的实际模型结果；`multispectrum_adaptive_scatter` 验证自适应迭代及设备掩码分位数。构建后从 `out/build/viewer-integrated/example/multispectrum_sim` 运行相应 CTest。

## 公共执行层

处理与输出使用 [统一流水线契约](../../../include/processing/stage/README.md)。业务接口 `IProjectionPreprocessStage` 绑定顶层 `pipeline::IStage<Frame, Context, Domain>`；通用接口不属于前处理。状态标记由执行器管理，插件不应修改；辅助输入要求通过 `validateContext()` 检查。

## 独立标定文件

水 BHC 现在分为两步，正式校正不会隐式拟合：

```powershell
multispectrum_sim.exe --calibrate-water-bhc scan.toml calibration/water.bhc.toml
multispectrum_sim.exe calibration/water.bhc.toml.config.toml
```

第一条只计算水板标定，保存系数、截断范围、参考能量、水衰减系数、标定水厚度范围和来源，并生成引用该表的完整配置。它不读取模体或做投影、重建；其当前物理模型是理想能量积分探测器，不能自动替代实机标定。

```toml
[reconstruction.preprocessing.beam_hardening]
enabled = true
calibration_file = "calibration/water.bhc.toml"
```

路径相对扫描 TOML；表中的系数、阶数、截断和参考覆盖扫描配置里的同名数值。应用已有表不要求原始能谱存在；HU 的水基准也直接读取该表。显式填写 coefficients 的调用仍可用。`fit_from_water_plate=true` 现在只表示尚待执行的标定配方，不能在正式校正中触发拟合；旧配置需要先运行上述标定命令。选中已有表时，以表为准。

UI 的 Reconstruction → Projection corrections → Beam hardening correction 提供表路径与 Generate water BHC table 按钮。生成成功后选中该表，再保存扫描 TOML 以持久化选择。重新生成是明确操作，不会因每次点击重建自动发生。

几何标定完成后在诊断报告旁保存 `.geometry-correction.toml`；可在几何标定窗口加载表并使用校正几何重建，不必再次提取钢珠和拟合。表目前限整周圆轨道平板 CT，并要求相同视角数、探测器行列和像素间距。

HU 校正的程序接口：

```cpp
auto result = postprocess::fitAirWaterHu(measured_air, measured_water);
result.save("air-water-hu.toml"); // 独立标定作业
// 后续任务：不再测量 ROI，不再拟合
pipeline.append(postprocess::makeHuCorrectionStage(
    std::filesystem::path("air-water-hu.toml"), postprocess::Location::Device));
```

HU 表也可用于正式重建，在 UI 的重建输出设置中启用 **Apply HU calibration** 并选择文件，或配置：

```toml
[reconstruction.postprocessing.hu_correction]
enabled = true
calibration_file = "calibration/air-water-hu.toml"
```

路径相对扫描 TOML。正式流程只读取标定结果，不会重新选择 ROI 或拟合。最终重建体积先按水基准转换为 HU，应用 `HU_corrected = scale * HU + offset`，再转回衰减系数（mm⁻¹）保存，因此 RAW 和现有显示模块的单位约定保持一致。水基准优先来自水 BHC 表；未配置该表时沿用参考能量/能谱计算。请确保标定时和校正时使用相同的水基准。

该处理只应用于最终输出，骨硬化循环的中间重建不会重复应用；不会使投影校正缓存失效。程序指定的其他后处理 Stage 在此校正之后运行。缺失或非法的表会在重建计算前报错。
