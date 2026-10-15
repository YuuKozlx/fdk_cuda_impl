# 双材料骨硬化插件

`makeTwoMaterialBhcStage(model, provider)` 返回现有前处理 `IStage`，可通过 `preprocess::Pipeline::append()` 插入。它不调用重建、不读取能谱，也不强制采用特定骨分割或前投算法。目录分工：

- `calibration/TwoMaterialCalibration.hpp`：从规则水/骨厚度网格的标定投影生成单调反函数表，并保存/加载 JSON 模型。可接收合成或实测标定面。
- `calibration/EmpiricalTwoMaterialFit.hpp`：只接收两种材料路径厚度和测量对数投影，拟合五参数双指数曲面；不接收能谱。PMMA/铝模体用法见 `../calibration/README.md`。
- `correction/TwoMaterialBhcStage.hpp`：CUDA 查表校正及可替换的 `IBoneThicknessProvider`。
- `app/experiments/TwoMaterialBoneCorrection.cu`：当前头部验证工具，用已有能谱和材料合成标定面，并用分割体积生成确定性骨厚度。

## 与论文的关系

采用 Martinez 等 2022 2DCalBH 的“固定第二材料厚度，对实测投影进行线性化”结构。现有两种离线建表入口：合成水/骨标定面使用指定参考能量（头部实验为 70 keV）；新增经验拟合使用五参数双指数和零厚度切线目标，不赋予单能能量（`reference_energy_keV=0`）。两者最终均生成单调反函数 LUT，而非论文每行的二次多项式，不能称为论文原版逐项复现。

## 插入方式

```cpp
using namespace yk::spectral;
auto model = preprocess::TwoMaterialCalibration::load("calibration.json");
// provider implements IBoneThicknessProvider; it may project a mask or supply cached paths.
std::shared_ptr<preprocess::IBoneThicknessProvider> provider = makeMyBoneProvider();
config.reconstruction.preprocessing.enabled = true;
config.reconstruction.preprocessing.beam_hardening.enabled = false;
preprocess::Pipeline pipeline;
pipeline.append(preprocess::makeTwoMaterialBhcStage(std::move(model), provider));
std::string diagnostic;
if (!pipeline.run(config, inputLogFile, correctedLogFile, diagnostic))
    throw std::runtime_error(diagnostic);
```

开启散射校正时，内置链先恢复能量、减散射并转回对数，再运行插件。已有散射校正的投影必须关闭散射步骤。禁止同时开启内置水 BHC，避免重复线性化。

`provider->writeDevice(view,width,height,destination)` 在默认 CUDA stream 上输出当前视图的骨厚度，单位 **mm**，长度为 `width*height`；插件拥有 destination，不允许提供器替换或释放。提供器的重投影必须使用与投影完全一致的几何和体积坐标。模型、体积缓存和生命周期由调用方配置，插件不依赖外部源码。

PMMA/铝标定时，提供器必须输出等效铝厚度，不能直接把骨厚度送入铝表。实验工具提供显式厚度比例，未标定的比例不能视为材料等效成立。加载表的路径不查询能谱或 XCOM。

## 数据与生命周期

输入和输出都是 Device 上的无量纲对数投影。插件只原地修改像素，不改变 Frame 的域、位置和存储指针。GPU 表和骨厚度缓冲使用现有显存池，在 prepare 时分配，reset 后可复用；不进行逐帧投影上传下载。提供器自行决定读取缓存还是计算路径。

骨厚度在相邻标定行之间线性插值。本实验骨步距 0.25 mm，小于重建体素 0.5 mm。投影值在每行均匀采样，端点外采用端点斜率线性外推，不把噪声直接裁平；调用方应保证校准水厚度范围覆盖对象，不能把大范围外推当作经过验证的校正。超出骨标定范围、负骨厚度或非有限数值会生成无效值，由前处理管线的输出检查拒绝。CPU 参考接口直接抛出异常。

## 头部实验

显式构建 `multispectrum_bone_bhc_2dcal`，然后执行：

```text
multispectrum_bone_bhc_2dcal baseline.toml segmented-u8.raw scatter-corrected-log.raw output-directory
```

该工具支持平板圆扫描、平板/柱面螺旋；分割标签 2 表示第二材料，0/1 不计入第二材料路径。头部实验分割必须来自初步重建，不能输入原始真值标签。默认合成模式以能量积分探测器、水密度 1.03、ncat_skull 密度 1.61 建表，拒绝附加探测器响应；`--apply-table` 则直接加载已有表。更换材料、能谱、滤过或探测器响应需重新标定。正式 Stage 不受这些实验限制。

输出 `calibration.json`、`bone-corrected.raw` 和 `2dcalbh.json`。输出目录已有 corrected RAW 时拒绝覆盖。当前不自动修改 UI/TOML 默认 BHC，不替换旧模型差值算法，不自动提交。

## 本次头部验证结果

使用已有 2500 帧、5 mAs/view 的全物理仿真投影，重新执行散射校正，使用初步水 BHC 重建的分割结果。全部 400 层均评估，重建为 512×512×400、0.5 mm、WFBP RamLak；校正后没有图像降噪。比较参考为独立 70 keV、每像素 50 射线、相同 WFBP 的重建，仅用于评价。

| 指标（HU） | 水 BHC | 旧模型差值骨 BHC | 本插件 |
|---|---:|---:|---:|
| 颅底骨间 ROI 低频暗偏差 RMS | 8.873 | 2.306 | 2.242 |
| 颅底非骨区域低频误差 RMS | 8.949 | 3.347 | 3.207 |
| 全体积非骨区域低频误差 RMS | 6.821 | 4.225 | 4.095 |
| 颅底骨间 ROI 高频误差 RMS | 27.024 | 26.946 | 27.371 |
| 骨内部原始误差 RMS | 51.310 | 25.058 | 25.598 |

低频误差是对误差场作 2 mm XY 高斯分析，显示及导出体积没有这一步滤波。骨间 ROI 索引为 x=[200,312)、y=[230,360)、z=[36,116)，剔除骨与空气并侵蚀 2 mm。不能把这些指标当作全部细条纹消除的证明；新旧骨 BHC 效果接近，新方法的高频误差略增。

GPU 骨路径＋查表约 4.08 s，含投影读写约 16.64 s（不含散射校正、离线标定和最终重建）。骨步距 0.25 mm、水标定范围 0–350 mm，骨范围按体积对角线扩展至 414 mm。独立标定测试点最大投影误差 4.92e-6。脚本 `scripts/evaluate_two_material_bhc.py` 记录全部切片指标，图像由原生 C++ 工具导出。
