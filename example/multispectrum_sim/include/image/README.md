# 图像处理 Stage（CPU / CUDA）

供投影图和重建体积共用的 float32 标量图像处理核心，使用现有 `pipeline::IStage`、元信息和工厂注册表。无 OpenCV 依赖。CUDA 后端使用现有设备显存池，CPU 后端用于数值对照和显式回退。

## 算法与维度

| 操作 | 实现 | 主要参数 |
|---|---|---|
| Resample | 分轴最近邻／线性插值；线性降采样默认按像素覆盖面积平均 | 输出尺寸、插值、抗混叠 |
| Mean | 分轴均值 | 各轴半径 |
| Gaussian | 分轴有限支持高斯 | 各轴半径、物理标准差 mm |
| Bilateral | 空间与强度联合权重的直接邻域算法 | 半径、空间标准差 mm、强度标准差 |
| Guided | 自引导滤波，以输入本身为引导图；局部均值／方差及系数均值 | 半径、epsilon（输入单位的平方） |

`Neighborhood::Slice2D` 表示沿 Z 逐层进行 XY 二维处理，层间不混合；`Spatial3D` 表示真正的 XYZ 邻域。两者均可用于三维体积。默认是 Spatial3D，调用者应显式选择所需方式。逐层重采样保持 Z 层数。

投影的布局为 `[batch][1][v][u]`：角度放在 `batches`，不能放在 Z。体积布局为 `[batch][z][y][x]`，X 连续。投影始终逐帧处理，不会跨角度滤波。均值和高斯采用可分离计算；双边采用完整邻域，三维大半径的计算量仍然较高。

边界采用端点复制，有限半径截断。滤波不改变数据的物理单位，不会自动把衰减系数转为 HU。对 HU 和 mm^-1 数据应分别选择强度参数。引导滤波目前是自引导，尚未提供独立的外部引导图接口。

## 使用现有处理链

```cpp
#include "image/ImageStages.hpp"
using namespace yk::spectral;

image::Options filter;
filter.operation = image::Operation::Gaussian;
filter.neighborhood = image::Neighborhood::Slice2D; // 三维体积逐层滤波
filter.radius = {2, 2, 0};
filter.sigma_mm = {.8, .8, 1.};

// 投影校正链：默认 Device、输入输出均为 LogAttenuation。
preprocess::Pipeline correction;
correction.append(image::makeProjectionStage(filter));

// 重建后的体积链：默认 Device、输入输出均为 AttenuationMmInv。
postprocess::Pipeline post;
post.append(image::makeVolumeStage(filter));
image::VolumeTransfer bridge;
image::Workspace scratch(image::Location::Device);
postprocess::Context context;
context.transfer = &bridge;
context.image_workspace = &scratch; // 相邻 Stage 共用临时缓冲
// post.run(volume, context);
```

`reconstructWithLibraryFdk` 的 `PostprocessConfigurer` 回调可以添加 `makeVolumeStage`；标准 RAW/BMP 导出在其后。迭代重建的设备输出直接进入后处理，先销毁重建 Session 释放权重等工作区，再执行图像 Stage。Host 输出的分析重建在进入第一个 CUDA Stage 时上传一次。连续 CUDA Stage 不会反复上传／下载，最终 CPU 导出时下载一次。

前处理仍维持原有的固定探测器尺寸。`makeProjectionStage` 支持上述四种滤波，明确拒绝 Resample。投影重采样必须通过下述图像链取得新的几何信息，再更新下游重建的探测器行列、像素间距和必要的坐标换算；不能在校正链中静默更改尺寸。体积 Stage 支持全部五种操作。

## 投影批次重采样／独立图像链

```cpp
image::Image frames;
frames.domain = image::Domain::Projection;
frames.grid.dimensions = {1024, 1024, 1};
frames.grid.spacing_mm = {.4, .4, 1.};
frames.batches = 8;
frames.location = image::Location::Device;
frames.data = device_projection_batch; // 借用输入；调用期间有效

image::Options resize;
resize.operation = image::Operation::Resample;
resize.neighborhood = image::Neighborhood::Slice2D;
resize.output_dimensions = {512, 512, 1};
image::Pipeline images;
images.append(image::makeStage(resize, image::Domain::Projection));
images.append(image::makeStage(filter, image::Domain::Projection));
image::Workspace workspace(image::Location::Device);
images.run(frames, workspace);
// frames.grid 现在为 512 x 512，间距 .8 mm；batches 仍为 8。
// frames.owner 持有输出；data 可直接交给下一个设备消费者。
// 需要 Host 数据时才调用 image::ensureLocation(frames, Location::Host)。
```

重采样保持体素单元覆盖的物理范围、中心和方向，同时更新 origin（首体素中心）与 spacing。它不改变批次数量。最近邻适用于要求保留标签值的输入；线性插值配合降采样面积平均用于连续图像。逐轴中间结果也是池化缓冲。

RAW 体积输出附带 `<raw>.grid` 元数据，包含实际尺寸、间距、原点与方向。UI 的重建结果读取器优先使用这个文件的尺寸和间距；没有 sidecar 的旧输出仍根据 TOML 读取。三视图显示沿输出网格轴切片，sidecar 中的方向／原点保留给后续空间定位或导出，并未解释为 DICOM 患者坐标。

## 内存、分块与线程

- 输入允许借用 Host/Device 指针；输出用共享所有权持有，Stage reset 或 workspace 销毁不会使结果失效。Host/Device 状态由流水线负责，数值 Stage 不改变状态标志。
- 同一个 worker 的 CUDA Stage 使用同一个 workspace；该对象不支持并发调用。当前只支持默认 CUDA stream，与已有显存池的完成事件约定一致；非默认 stream 明确报错。
- 默认临时缓冲逻辑预算为 256 MiB（按所需容量计算，显存池复用的块可能更大）。滤波沿 Z 切块，均值／高斯／双边保留 radius 的重叠边界，自引导保留 2×radius。批次、尾块、外边界均独立处理。二维模式不需要 Z halo。
- 预算仅包含滤波 scratch；输入、完整输出和重采样中间图像另计。它不是超大体积的磁盘／Host 换入换出方案。若一层及其必要 halo 就超过预算，明确报错，可由调用者增加预算。
- 池的 VTK 预留规则沿用现有实现。输出替换、运行结束及异常退出均通过 RAII 归还显存。重复同尺寸调用经预热后复用已分配块。
- CPU fallback 需显式选择 Host 后端；不在显存不足或参数错误时悄悄切换执行位置。

## 元信息与验证

`pipeline::builtinPluginDescriptions()` 包含 CPU/CUDA × 投影/体积 × 五种操作共 20 个描述。`image::makeRegistry(domain, location)` 返回该领域／后端的五个可创建 Stage，查询不分配图像缓冲。带自定义参数时使用 `makeStage(options, ...)`。元信息表示算法能力，投影校正链的固定尺寸限制仍由适配器检查。

`Pipeline::timings()` 给出每级处理和传输耗时，`transfers()` 统计执行位置切换次数；`VolumeTransfer` 单独统计上传和下载次数。CUDA 处理结束同步默认 stream，因此处理计时包含设备执行完成，而不是只有启动时间。

新增测试覆盖 CPU/CUDA 对照、独立的朴素均值和引导参考、二维／三维、层间及角度隔离、缩放抗混叠、旋转网格中心不变、强制小预算分块与尾块、输出所有权、工厂元信息、相邻 Stage 显存驻留与预热后显存复用。现有校正链测试验证新滤波 Stage 不增加逐帧 PCIe 往返次数，显示测试验证重采样后读取新尺寸。

本次提供 C++ Stage/API 与链路接入；尚未新增 TOML 字段或 UI 滤波面板。

## 实测示例

RTX 3070 Laptop 8 GiB，256×256×64 float32 体积，3D 半径 1，预热后单次调用：

| 滤波 | CPU 参考实现 | CUDA |
|---|---:|---:|
| 均值 | 95.6 ms | 4.8 ms |
| 高斯 | 267.0 ms | 13.4 ms |
| 双边 | 1092.9 ms | 50.8 ms |
| 自引导 | 406.5 ms | 17.6 ms |

计时包含处理和输出缓冲申请／复用、默认流同步，不含输入上传和结果下载。CPU 是本库单线程参考实现，不是优化后的 OpenCV 对照。这里是一次实测，不保证不同半径、数据尺寸或硬件有相同加速比。可用 `multispectrum_image_test.exe --benchmark` 复现。
