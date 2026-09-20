# 多能谱重建回归记录

日期：2026-09-20。环境：Windows、VS2022 Release、CUDA 12.6、RTX 3080 10 GB。
本轮没有实现 wFBP 流式分批，也没有修改其滤波或反投影公式。

## 接口与配置

- CTest 四项全部通过：wFBP 缓冲区组合、配置校验、XCOM 水、路径缓存布局。
- wFBP 输入/输出分别取 Host/Device，共四种组合；首次、重复、reset 后共十二次执行，相对第一次 Host/Host 的最大绝对差全部为 0。
- 测试检查非零、有限输出，并验证半批视图返回 StreamingStateError；不是只比较两个全零体积。
- 配置测试覆盖非法管线/滤波器/类型/数值范围及不支持的几何组合，避免拼错参数后静默降级。
- Python ROI 三项单元测试通过，覆盖邻域腐蚀、非有限值、参考容差通过/失败、固定水 ROI 与网格偏移不一致。

## BrainWeb 回归

中心 33 层已有投影通过 `--reconstruct-only` 重建。FDK 与 wFBP pitch4 的 RAW 均与本轮改动前逐字节一致：

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
& $exe --water-cylinder "$base/configs/water-small-cone-fdk-calibration.toml"
& $exe --water-cylinder "$base/configs/water-small-cone-heli-cyl-wfbp-calibration.toml"
& $exe --export-water-cylinder "$base/configs/water-small-cone-fdk-calibration.toml" "$base/outputs/calibration/water-labels.raw"
python "$base/scripts/report_material_roi.py" "$base/configs/water-small-cone-fdk-calibration.toml" --labels "$base/outputs/calibration/water-labels.raw" --fixed-radius-mm 20
python "$base/scripts/report_material_roi.py" "$base/configs/water-small-cone-heli-cyl-wfbp-calibration.toml" --labels "$base/outputs/calibration/water-labels.raw" --fixed-radius-mm 20
python "$base/scripts/plot_water_roi.py" "$base/outputs/calibration/water-fdk-volume.roi.json" "$base/outputs/calibration/water-wfbp-volume.roi.json" --output "$base/outputs/calibration/water-roi-comparison.png"
ctest --test-dir out/top-multispectrum-dll-vs/example/multispectrum_sim -C Release --output-on-failure
python -m unittest discover -s "$base/scripts" -p test_report_material_roi.py
```

Python 需要 3.11+、numpy；绘图另需 matplotlib。

## 限制

- wFBP 仍一次处理完整投影。该水模输入 619315200 字节，输出体积 5184564 字节，另有内部工作区。
- WDDM 实测空闲物理显存查询为 0 时仍可分配并完成重建，因此容量检查只作警告，不硬性拒绝。不能由此承诺任意尺寸可运行或不会分页。
- 既有 CUDA 分配失败仍可能经底层检查宏终止进程；本轮没有将整个库的错误处理改为可恢复机制。
- `--reconstruct-only` 检查 RAW 尺寸，但没有几何指纹；同尺寸不同几何的投影仍需调用者确认。
- 本轮运行验证限于 Windows，未新增 Linux 实机运行结论。
