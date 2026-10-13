# 强度转换、HU 标定／校正与阈值分割

这些 Stage 使用现有后处理链，均提供 CPU 和 CUDA 路径。源文件与接口按用途分别放在 `calibration/`、`correction/`、`segmentation/`。标定只计算系数；校正才修改图像。当前提供 C++ API，尚未添加 TOML 字段和 UI 控件。

## μ 与 HU 的可变基准转换

`correction/HuTransforms.hpp`：

- `makeMuToHuStage(reference, location)`：`AttenuationMmInv -> HounsfieldUnits`。
- `makeHuToMuStage(reference, location)`：`HounsfieldUnits -> AttenuationMmInv`。
- `MuHuReference` 包含水／空气的衰减基准和对应 HU 锚点。水衰减必须由调用者明确给出，通常采用 BHC 拟合能量对应的水衰减系数，单位 mm^-1。
- `makeHuConversionStage(mu_water)` 保留为水单基准的 CPU 便捷入口，数值执行已归入校正目录的共用实现。

常用设置为空气衰减 0、水 HU=0、空气 HU=-1000：

```text
HU = 1000 × (μ / μ_water - 1)
μ  = μ_water × (1 + HU / 1000)
```

使用四个可变锚点时：

```text
HU = HU_air + (μ - μ_air) × (HU_water - HU_air) / (μ_water - μ_air)
μ  = μ_air + (HU - HU_air) × (μ_water - μ_air) / (HU_water - HU_air)
```

要求水衰减大于空气衰减、空气衰减非负、水 HU 大于空气 HU。转换不截断 HU 或负的输入／输出 μ，以免破坏有噪声数据及往返关系。CPU/CUDA 都检查 NaN、Inf 和 float32 溢出；发现错误时不对图像做部分修改。

```cpp
using namespace yk::spectral::postprocess;
MuHuReference reference;
reference.water_mu_mm_inv = .019285; // 示例；实际值取所选能量的水衰减
Pipeline p;
p.append(makeMuToHuStage(reference)); // 默认 CUDA
// 后续输入为 HU，可以接校正或分割。
```

## 空气／水双材料 HU 线性标定

`calibration/HuCalibration.hpp`：

```text
scale  = (target_water_HU - target_air_HU) / (measured_water_HU - measured_air_HU)
offset = target_air_HU - scale × measured_air_HU
HU_corrected = scale × HU_measured + offset
```

默认目标为空气 -1000 HU、水 0 HU。`AirWaterTargets` 可调整目标值和测量均值的最小间隔（默认 1 HU）。反向、重合或间隔过小的测量会被拒绝。

已知两种材料的测量均值时，可直接计算：

```cpp
auto fitted = fitAirWaterHu(-900., 100.);
// scale=1, offset=-100；-900 -> -1000，100 -> 0。
auto correction = makeHuCorrectionStage(fitted.coefficients);
```

也可插入标定 Stage，在当前 HU 体积中计算两个 ROI 的均值：

```cpp
auto calibration = std::make_shared<HuCalibrationResult>();
CalibrationRoi air   {{0, 0, 0}, {16, 16, 8}};
CalibrationRoi water {{32, 32, 16}, {48, 48, 24}};
Pipeline p;
p.append(makeMuToHuStage(reference));
p.append(makeAirWaterHuCalibrationStage(air, water, calibration));
p.append(makeHuCorrectionStage(calibration));
p.append(makeHuToMuStage(reference)); // 恢复 μ，供标准衰减 RAW 导出
```

ROI 是当前体素网格上的半开索引盒 `[begin,end)`，与物体世界坐标区分。两 ROI 必须非空、在网格内且不重叠；不会自动识别材料区域。Stage 输入必须是 HU，因此原始 μ 体积需先转换。CUDA 在设备端求和，只下载最多两组 1024 个 double 部分和及错误标志，不下载完整体积。

结果含有效标志、系数、测量均值、目标锚点和 ROI 体素数。成功结果在 Stage reset 后仍可读取，系数可复制到后续任务。新一次标定准备／执行时先清除旧结果，失败不会留下有效的旧系数。同一结果对象不可并发读写。

## HU 校正

`makeHuCorrectionStage` 输入输出都是 HU，可接受：

1. 固定 `HuLinearCoefficients{scale, offset}`。
2. `shared_ptr<const HuCalibrationResult>`，在执行时读取已完成的标定结果，因此可紧接同链中的标定 Stage，也可应用上一次任务的结果。

系数要求正的有限 scale 和有限 offset。不自动夹紧到 -1000，不重采样，不改变几何，也不额外分配整份体积。未成功标定时，依赖结果的校正会显式失败。

## 多级 HU 阈值与标签

`segmentation/HuThresholds.hpp`：

```cpp
HuThresholdSegmentation segments;
segments.thresholds_hu = {-500., 200.};
segments.classes = {{0, "air"}, {10, "soft tissue"}, {20, "bone"}};
Pipeline labels;
labels.append(makeHuThresholdStage(segments));
// 输入 HU：[-inf,-500) -> 0；[-500,200) -> 10；[200,+inf) -> 20。
```

阈值必须严格升序且有限；N 个阈值对应 N+1 个区间，等号归入较高值区间。标签包含整数 ID 和可选名称；未提供 classes 时自动使用 0..N。允许多个区间使用同一 ID，但其名称必须一致。零个阈值意味着整幅有限值图像归入同一类。

输出标记为 `Domain::Labels`，并在 `Volume::label_mapping` / `ReadOnlyVolume::label_mapping` 中保留阈值、ID、名称。底层沿用 float32 存储，ID 限制为 0..16777216，以保证精确表示；这不是 uint16 标签文件。标签数据不能接 HU/μ 转换，也不能接均值、双边等连续值滤波。图像处理适配器只允许标签使用最近邻重采样。

默认衰减 RAW 导出器拒绝 HU 和 Labels，避免外部显示程序把它们误当作 μ。分割结果由独立标签分支／消费者读取；当前没有新增标签 RAW/DICOM 导出或 UI 标签着色。需要标准衰减输出的 HU 校正链应以 `makeHuToMuStage` 结束。

## 执行与查询

CUDA Stage 默认使用与设备显存池一致的默认 stream。相邻 GPU 转换、标定、校正、分割不进行体积级上传下载；少量参数、错误标志和 ROI 统计量可跨 CPU/GPU。原位校正与分割先验证整幅输入，再进行写入。CPU 后端通过传入 `Location::Host` 选择；无 CUDA 的独立构建也可使用。

上述十个 CPU/CUDA 描述可通过 `pipeline::builtinPluginDescriptions()` 或 `intensityPluginDescriptions()` 查询；所有实例支持 `info()`。描述类别分别为 calibration、correction、segmentation。工厂参数和元信息独立，不依赖 GUI 或图像数据。

测试覆盖可变锚点及负值往返、ROI 均值／计数、同链标定校正、跨任务系数复用、失败结果失效、标签等号边界与名称、CPU/CUDA 数值、设备常驻、数据域拒绝和 NaN／Inf／溢出不产生部分写入。
