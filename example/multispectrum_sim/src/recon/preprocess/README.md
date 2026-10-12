# 重建前处理开发说明

## 文件职责

- `PreprocessPipeline.cpp`：阶段编排、数据域检查、Host/Device 状态转换、文件 I/O、进度和计时。
- `Stages.cpp`：内置能量恢复、散射、转对数/BHC 阶段；固定、自适应和引导散射均通过设备接口连接。
- `CudaPixelOps.cu`：驻留帧工作缓冲、能量/对数转换、双精度 Horner 与截断、有效性检查。
- `BeamHardeningFit.cpp`：小规模 CPU 水板标定拟合。逐像素应用在 CUDA 完成。
- `DeviceMemoryPool.cpp`：进程内、按 CUDA 设备隔离、线程安全的缓冲池。无 VTK 依赖。

外部入口 `prepareReconstructionProjection()` 保留兼容，内部调用 `Pipeline`。原来的校正数值定义不变：恢复 `air*exp(-log)`，散射正值下限 `air*1e-8`，转对数比例下限 `FLT_MIN`，BHC 使用双精度多项式并按配置截断；多项式溢出必须报错，不能被截断掩盖。

## 插件约定

实现 `IStage` 后通过 `Pipeline::append(std::unique_ptr<IStage>)` 加入，执行位置在内置校正链之后。阶段声明 `inputDomain/outputDomain`（LogAttenuation 或 EnergySignal）以及 `inputLocation/outputLocation`（Host 或 Device，默认 Device）。前处理输出必须为 LogAttenuation，域不匹配在读取帧前报错。

`prepare()` 每次任务一次，用于常量上传和工作区准备；`process()` 每帧一次；`reset() noexcept` 在成功及异常退出时调用。reset 必须可处理尚未准备或准备到一半的对象，不得依赖保存过的 Context。

Frame 的位置表示**当前权威副本**，另一个副本可能已过时。流水线按阶段要求才进行传输；阶段执行后按声明更新状态。CPU 阶段修改 `*host_data`，GPU 阶段修改 `device_data`；不得更换存储指针、改变帧大小、保留跨帧指针。不同输出位置的阶段必须自己完成该次位置转换。GPU 指针使用当前 CUDA 设备及默认流；其他流上的处理必须在返回前与默认流建立依赖。

Context 和 Frame 均为借用对象，只在回调期间使用。Pipeline 本身不支持并发；每个重建工作线程使用独立实例。显存池在不同实例之间共享，不共享数据内容。

## 显存池和 VTK 预留

`initializeDevicePool(sizes)` 同时预留所列缓冲；重复调用可复用缓存。`poolAllocate/poolRelease` 成对使用，活动缓冲不会再次借出。归还时记录默认流完成事件，下一借用方等待该事件，防止跨任务覆盖尚未完成的 GPU 运算。所有新借用者都必须重新初始化数据，池不保证内容为零。

新分配按实际可用显存检查，默认留下 `max(512 MiB, 总显存/5)`，小显卡最多预留一半；池活动与缓存之和最多为总显存 60%，空闲缓存上限 256 MiB。这里是池的占用约束，其他库仍可使用预留空间，并非 GPU 驱动级的专属分区。VTK 和重建 DLL 的显存不由此池分配。

尺寸或配置变化时复用容量足够的空闲块，缺少的部分才新增；不足时先释放当前设备的空闲缓存，再检查预算。超预算报错而非占用预留区。`trimDevicePool()` 主动清理当前设备空闲缓存，活动借用保持有效。`devicePoolStatistics()` 提供真实 cudaMalloc 次数、复用次数及活动/缓存字节。FFT 显式关闭 cuFFT 自动工作区分配，并从池提供工作区；自适应排序临时缓冲同样使用池。

## 传输及性能边界

一般 CUDA 校正链每帧一次上传、一次最终下载；单帧 air 文件只上传一次并复用于所有角度，多帧 air 随角度更新。正式 fixed 的掩码及边缘窗口均在 GPU 生成，不下载掩码。实验 adaptive 目前仍下载 uint8 掩码并在 CPU 生成边缘窗口；自适应求解回传小规模收敛统计，不回传完整浮点图像。

当前重建入口仍将校正结果写入临时文件，再由重建 DLL 读取并上传。这一边界没有改为直接传递显存，不能宣称前处理与重建完全零拷贝。VTK 图像显示也保留自己的上传过程。

日志分别记录初始化、拟合、读写、传输、各阶段耗时与整帧传输次数，以及池的分配/复用次数。阶段计时同步默认 GPU 工作，避免把异步提交时间当成计算耗时。生成能谱按模型、电压、角度、bin 宽及全部过滤器参数缓存最多 8 项；文件能谱仍每次读取，参数改动不能命中旧能谱。缓存仅在同一进程有效，首次生成仍需计算。

## 验证

`multispectrum_preprocessing` 验证多项式、EID 水板拟合、引导零散射、Host/Device 权威状态及传输次数、0～16 阶数值、溢出后重试、池重复初始化、活动缓冲隔离和 VTK 显存预算。`multispectrum_scatter` 对比设备固定校正和引导接口的实际模型结果；`multispectrum_adaptive_scatter` 验证自适应迭代及设备掩码分位数。构建后从 `out/build/viewer-integrated/example/multispectrum_sim` 运行相应 CTest。
