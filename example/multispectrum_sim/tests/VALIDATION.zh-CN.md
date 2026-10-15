# 多能谱重建回归记录

日期：2026-09-20。环境：Windows、VS2022 Release、CUDA 12.6、RTX 3080 10 GB。
本轮没有实现 wFBP 流式分批，也没有修改其滤波或反投影公式。

## 接口与配置

- CTest 三项全部通过：wFBP 缓冲区组合、配置校验、XCOM 水。
- wFBP 输入/输出分别取 Host/Device，共四种组合；首次、重复、reset 后共十二次执行，相对第一次 Host/Host 的最大绝对差全部为 0。
- 测试检查非零、有限输出，并验证半批视图返回 StreamingStateError；不是只比较两个全零体积。
- 配置测试覆盖非法管线/滤波器/类型/数值范围及不支持的几何组合，避免拼错参数后静默降级。
- Python ROI 三项单元测试通过，覆盖邻域腐蚀、非有限值、参考容差通过/失败、固定水 ROI 与网格偏移不一致。

## BrainWeb 回归

中心 33 层已有投影通过 `workflow.mode = "reconstruct"` 重建。FDK 与 wFBP pitch4 的 RAW 均与本轮改动前逐字节一致：

| 算法 | RAW SHA256 |
| --- | --- |
| FDK | 3B79870877769C95D5E0DF96B32B672024360665532659BF647C6603F269C0D0 |
| wFBP | 2DF6209B89C64EEA7BD176CF06E9BF896FC56A0B8813EA95F2F5F337C71C4836 |

`reference-brainweb-center33-*.json` 是实现回归基线，不是人体组织的物理真值。
ROI 使用一体素立方邻域腐蚀，排除首尾各四层；不同腐蚀定义的旧统计不可直接比较。
四种主要组织均通过 1e-5 mm^-1 绝对容差。空 ROI 不统计；单体素 ROI 不作为定量结论。

## 小锥角水模

使用已有两个 `water-small-cone-*-calibration.toml`，120 kV、150 能量采样点、水密度 1 g/cm3。
体积 181×217×33、1 mm 体素；水圆柱半径 72.4 mm。
探测器 768×112、0.630124 mm 像素，SID=709 mm、SDD=1143.699951 mm。
FDK 为 360 帧圆扫；wFBP 为 1800 帧、360 帧/圈、pitch=8 mm/圈、start_z=-20 mm。
FDK 使用 Ram-Lak；wFBP 使用 FreeCT ramp，cutoff=1、apodization=1。

固定 XY 中心半径 20 mm 的圆形 ROI，z=-12…12 mm，共 25 层：

| 指标（mm^-1） | FDK | wFBP |
| --- | ---: | ---: |
| ROI 均值 | 0.0205154462 | 0.0208266815 |
| 各层均值的标准差 | 4.44651e-8 | 3.59978e-7 |
| 各层均值的极差 | 1.42241e-7 | 1.05175e-6 |
| 全水标签内部 ROI 均值 | 0.0208811857 | 0.0211783937 |

wFBP 固定 ROI 比 FDK 高 0.0003112354 mm^-1，约 1.5171%。两者均纵向稳定，
但整体偏差尚在；本轮没有用经验比例强行抹平差异。多能谱、束硬化、不同滤波及重排
都会影响数值，本测试不能单独定位偏差来源，也不能认定 FDK 为绝对真值。
全水标签 ROI 和固定中心 ROI 不同，不可混用。

结果位于 `../outputs/calibration/`：两套 projection.raw、volume.raw、三方向 BMP、
volume.roi.json 和 `water-roi-comparison.png`。对比图上栏为绝对值，下栏为各自去均值后的
纵向波动；下栏不能用来判断算法间的绝对偏差。结果保留本机，不纳入 Git。

## 复现

从仓库根目录运行，下面用 `$exe` 指代本轮 DLL 构建的程序：

```powershell
$exe = 'out/top-multispectrum-dll-vs/example/multispectrum_sim/Release/multispectrum_sim.exe'
$base = 'example/multispectrum_sim'
& $exe "$base/configs/calibration/reconstruction/water-small-cone-fdk.toml"
& $exe "$base/configs/calibration/reconstruction/water-small-cone-heli-cyl-wfbp.toml"
python "$base/scripts/report_material_roi.py" "$base/configs/calibration/reconstruction/water-small-cone-fdk.toml" --fixed-radius-mm 20
python "$base/scripts/report_material_roi.py" "$base/configs/calibration/reconstruction/water-small-cone-heli-cyl-wfbp.toml" --fixed-radius-mm 20
python "$base/scripts/plot_water_roi.py" "$base/outputs/calibration/water-fdk-volume.roi.json" "$base/outputs/calibration/water-wfbp-volume.roi.json" --output "$base/outputs/calibration/water-roi-comparison.png"
ctest --test-dir out/top-multispectrum-dll-vs/example/multispectrum_sim -C Release --output-on-failure
python -m unittest discover -s "$base/scripts" -p test_report_material_roi.py
```

Python 需要 3.11+、numpy；绘图另需 matplotlib。

## 限制

- wFBP 仍一次处理完整投影。该水模输入 619315200 字节，输出体积 5184564 字节，另有内部工作区。
- WDDM 实测空闲物理显存查询为 0 时仍可分配并完成重建，因此容量检查只作警告，不硬性拒绝。不能由此承诺任意尺寸可运行或不会分页。
- 既有 CUDA 分配失败仍可能经底层检查宏终止进程；本轮没有将整个库的错误处理改为可恢复机制。
- `workflow.mode = "reconstruct"` 检查 RAW 尺寸，但没有几何指纹；同尺寸不同几何的投影仍需调用者确认。
- 本轮运行验证限于 Windows，未新增 Linux 实机运行结论。

## 2026-10-15：Stage 与标定辅助函数测试覆盖

测试可执行文件的数量不等于测试覆盖率。以下按实际处理过程列出断言范围；未做分支覆盖率统计，不宣称全库充分覆盖，也不能用单元测试通过替代图像质量验收。

| 模块 | 数值或行为依据 | 测试文件 |
|---|---|---|
| 均值滤波 | 独立三维邻域求和，复制边界、各向异性网格、逐层/三维、单体素和多批次；CPU/CUDA | test_image_oracles.cpp、test_image_processing.cpp |
| 高斯滤波 | 独立完整邻域高斯卷积，对照实现中的可分离计算；非等距体素和不同轴 sigma；CPU/CUDA | test_image_oracles.cpp |
| 双边滤波 | 独立空间权重×灰度权重公式；边界、批次隔离、逐层/三维；CPU/CUDA | test_image_oracles.cpp |
| 引导滤波 | 独立局部均值/方差/线性系数两遍计算；非平凡输入；CPU/CUDA | test_image_oracles.cpp、test_image_processing.cpp |
| 升降采样 | 手算非整数缩放、最近邻标签、面积降采样；旋转网格物理中心；CPU/CUDA | test_image_oracles.cpp、test_image_processing.cpp |
| 图像 Stage 管理 | 连续设备常驻、分块 halo 与尾块、输出所有权、工作区释放后结果有效、非法参数 | test_image_processing.cpp、test_image_oracles.cpp |
| μ↔HU | 可变水/空气基准、负值、往返转换、组合链、溢出失败不部分修改 | test_intensity_stages.cpp |
| 空气/水 HU 标定及校正 | ROI 均值/计数、三维多块归约、复用系数、失败后失效、ROI 越界/空/重叠、最小测量间隔 | test_intensity_stages.cpp |
| HU 阈值分割 | 自定义与自动标签、阈值及左右相邻浮点值、标签元信息、NaN、冲突名称与错误数据域 | test_intensity_stages.cpp |
| 水板多项式拟合 | 1–16 阶单能恒等；3–16 阶独立双能 EID 响应的未采样厚度对照；参考光子均能；参数拒绝、进度及异常回调后复用 | test_calibration_fits.cpp |
| CUDA 像素辅助处理 | 0–16 阶多项式与独立幂求和对照、上下限裁剪、能量/对数往返、引导残差与下限、阈值掩码、513 像素尾块、错误帧后的复用 | test_pixel_helpers.cpp |
| 双材料经验拟合 | 已知双指数模型的独立厚度验证、两种切线斜率、无单能能量语义、退化厚度/非有限数据拒绝 | test_calibration_fits.cpp |
| EBHC 鲁棒系数拟合 | 合成已知系数、样本重排、离群点、拟合误差统计、共线基底/非法 Huber 尺度拒绝 | test_calibration_fits.cpp、test_empirical_bhc.cpp |
| 双材料标定表 | 独立线性逆映射、插值/线性外推、端点、非单调数据、序列化与损坏文件/版本/域拒绝 | test_two_material_calibration.cpp |
| 双材料 CUDA Stage | 独立非线性解析留出点、CPU/CUDA 一致、重复 prepare/reset、重复水 BHC 拒绝、厚度越界信号 | test_two_material_bhc.cpp |
| 统一骨硬化策略 | 固定原始投影、重建反馈、数据域、迭代停止/不停止/等号边界、可选误差、异常清理、不发布半成品、进度顺序 | test_bone_bhc.cpp、test_bone_bhc_contract.cpp |
| 实验 EBHC 三项 Stage | 已知三系数恢复、TV 独立 CPU 对照、Joseph 几何与 CPU/CUDA 对照、重复生命周期 | test_ebhc_paper.cu |

### 独立运行

在仓库根目录、已配置 C++ 编译器的终端中，不需要 VTK 或 UI：

```text
cmake -S example/multispectrum_sim/tests/cpu -B out/build/stage-tests -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/stage-tests --config Release
ctest --test-dir out/build/stage-tests -C Release --output-on-failure
```

需要同时运行 GPU 滤波及强度 Stage 测试时，使用另一个构建目录并增加 `-DSTAGE_TEST_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`（架构按实际 GPU 调整）。纯 CPU 入口包含真实 XCOM/能谱依赖源码，不使用虚构衰减系数替代生产拟合。

完整应用构建也通过 CTest 注册新增拟合和图像 oracle 测试。EBHC、双材料 CUDA 等既有测试仍由完整应用构建提供。

### 尚不能据此保证的内容

- 多项式高阶病态拟合（尤其接近最高 16 阶）没有获得普遍稳定性保证；本轮已改为自变量缩放与 Householder QR，仍需避免区间外高阶外推。
- 实测数据、严重残余散射及材料失配下的校正效果需要独立质量实验。
- EBHC 与两遍锥束伪影实验仍未通过头部图像质量验收；数值测试通过不会改变实验性状态。
- 不涵盖所有重建算法、全部几何和 UI 自动化，也不声称已测得行/分支覆盖率。

### 扩展数值族与独立标量 FP

固定随机种子使数据可复现；每个 CTest 条目内部包含多组数值样例。

| 数值族 | 样例范围 |
|---|---|
| 水板拟合 | 90 组：35/70/120 keV × 10/100/500 mm × 1/3/6/10/16 阶 × 33/101 个拟合点；每组另验 101 个非采样点；另保留双能 EID 与全部 1–16 阶测试 |
| 双材料经验拟合 | 12 组混合比例、衰减尺度与噪声组合；随机缺失 20% 内部测量点，每组另验 150 个厚度点，分别约束 RMSE 和峰值误差 |
| EBHC 鲁棒拟合 | 54 组系数、幅值、基底相关性及 0/5/15% 离群点组合，每组 800 个样本 |
| 每种滤波 | 504 组：7 种图案 × 3 种半径 × 3 种幅值 × 普通/单体素 × 逐层/三维 × CPU/CUDA；每组含 2 批，防止批间混合 |
| 升降采样 | 每个后端额外 24 组非立方/单维度输入、混合升降采样、最近邻/线性及抗混叠开关；独立三轴张量积权重参考 |
| μ/HU 与校正 | 每个后端 12 组参考基准，每组 513 个值及往返检验；9 组空气/水测量锚点校正 |
| 标量 Siddon/Joseph | 三主轴、非等距体素、体积偏移、单体素与跨 256 层边界、Joseph 1/2/4 步；12 组几何的分批/整批/累加；Siddon 独立双精度平面交点参考 |
| DLL 对照 | 圆扫/螺旋 × Siddon/Joseph × 3 种偏移/倾斜设置，共 12 组；仅对照测试依赖 DLL 的公共 SDK |

引导滤波参考采用双精度完整邻域求和。误差判定同时包含输出相对误差和输入幅值对应的 float 舍入项，避免经过抵消的近零输出使用不合理的纯相对容差。

新增 FP 在 `include/primary/ScalarProjector.hpp`，由上层独立维护，生产目标仅依赖 CUDA，不引用底层内部源码。`src/primary/scalar_fp/README.md` 说明坐标和资源所有权。原有多能融合投影、骨硬化和 EBHC 默认执行路径尚未替换；本轮无速度收益结论。

可选 DLL 对照：独立 GPU 测试配置增加 `SCALAR_FP_DLL_LIBRARY`（导入库）与 `SCALAR_FP_DLL_INCLUDE_DIR`（公共头文件目录），运行时确保 DLL 及其依赖位于 PATH。完整工程存在 `YKCBCT` 目标时自动注册。未提供 DLL 时不注册此项，其余 FP 测试仍可运行。

实测结果（2026-10-15，Windows / RTX 3070 Laptop）：独立 CPU/CUDA 测试 31/31 通过，耗时 5.46 秒，含可选 DLL 对照；正式应用的相关 CPU 测试 13/13 通过，耗时 1.07 秒。两套入口存在重复用例，不能相加当作 44 个独立测试。测试覆盖数值和生命周期，不代表头部骨伪影或全部扫描几何通过质量验收。

数值族暴露并修复了短厚度高阶水板拟合误报奇异：改为在 [-1,1] 上构造设计矩阵并用 Householder QR 拟合，最后换回 Stage 所需的原始多项式系数。没有新增运行库依赖。既有单能、双能及新增短厚度样例均通过。
