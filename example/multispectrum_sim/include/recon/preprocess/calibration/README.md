# PMMA／铝模体：离线标定与在线校正

## 分工

1. `pmma_al_calibration_experiment.py` 本地生成两个分离圆柱：PMMA 半径 110 mm、密度 1.19 g/cm³；纯铝半径 25 mm、密度 2.7 g/cm³。不是 AL6082，也不是论文模体形状的精确复制。
2. 正式 `multispectrum_sim` 扫描并 FDK 重建。演示协议继承头部的 120 kVp、6 mm Al、SID 709、SDD 1143；圆扫描 360 帧，1024×32、0.6 mm 探测器。第一轮关闭散射、噪声、焦点和探测器后效，隔离标定算法误差。
3. `--collect-phantom` 将模体标签重投影成两种材料路径，与扫描对数投影配对。支持 CAD 标签对照，以及初步 FDK 重建按 0.009 / 0.04 mm⁻¹ 阈值分割的标签。这些阈值只适用于当前数字模体。
4. `multispectrum_fit_material_phantom` 离线拟合并保存 JSON 表。输入仅为 `view,first_mm,second_mm,measured_log` CSV，**没有能谱输入**；可选同名 `.csv.json` 保存样本来源。将每五个角度中的一个完整留出验证，其余角度按 5×2 mm 厚度格子均衡取样，每格最多 12 点。
5. `--apply-table` 加载表，通过可插拔 `TwoMaterialBhcStage` 在 CUDA 上校正，再调用正常重建。它不调用能谱或衰减数据库；输入为散射已处理、尚未水 BHC 的对数投影。

## 拟合及输出含义

经验响应为：

`F(x,y) = -log(a*exp(-b*x-c*y) + (1-a)*exp(-d*x-f*y))`

采用有界参数变换、三组初始混合比例和阻尼 Gauss–Newton。线性目标为：

`L(x,y) = [a*b+(1-a)*d]*x + [a*c+(1-a)*f]*y`

这是零厚度切线目标，不是指定 70 keV。`reference_energy_keV=0` 表达没有赋予单能能量，不能直接套用旧的 70 keV 水衰减转 HU。后续 HU 应另做水／空气标定。

将拟合曲面按第一材料 ≤0.5 mm、第二材料 ≤0.1 mm 采样，构造每行 1025 点反函数表。当前结构的 `water_*` 字段表示第一材料，`bone_*` 表示第二材料；对于本实验即 PMMA／铝。第二材料厚度由提供器确定，不应把未知骨材料直接当成等厚度铝。表覆盖的矩形包含拟合曲面的插值或外推区域，两个轴的最大厚度不代表所有联合厚度都获得同等测量支持。

## 复现

构建目标：`multispectrum_sim`、`multispectrum_fit_material_phantom`、`multispectrum_bone_bhc_2dcal`。以下 `$case` 是新的输出目录，`$baseline` 是已有头部扫描 TOML；Python 仅生成模体、配置和统计，不承担标定拟合或绘图。

```powershell
python example/multispectrum_sim/scripts/pmma_al_calibration_experiment.py $baseline $case
multispectrum_sim "$case/scan.toml"
multispectrum_bone_bhc_2dcal --collect-phantom "$case/scan.toml" "$case/samples.csv"
multispectrum_fit_material_phantom "$case/samples.csv" "$case/fitted"
multispectrum_bone_bhc_2dcal --apply-table "$case/scan.toml" "$case/phantom-u8.raw" "$case/projection.raw" "$case/fitted/calibration.json" 1 "$case/applied"
multispectrum_sim "$case/reconstruct-corrected.toml"
python example/multispectrum_sim/scripts/pmma_al_calibration_experiment.py $baseline $case --evaluate
```

最后一步生成重建分割和 `segmentation-paths.toml`。随后用此 TOML 收集新样本，拟合到另一个目录，再将 `--apply-table` 的标签替换为 `reconstructed-segmentation-u8.raw`。重建 TOML 的输入和输出也应改到新目录，保留 CAD 对照。

样本采集器目前要求每像素一条仿真射线、焦点与散焦关闭。为配对测量值，它复用仿真的随机探测器落点；在线校正始终使用确定性中心射线。实测采集不能依赖仿真随机种子，应提供与实际积分像素模型匹配的路径估计。不要将多射线路径均值直接代入指数模型。

绘图由原生 C++ / Matplot++ 工具完成：

```text
multispectrum_plot_bone_bhc --calibration-map calibration.json calibration-map.png
multispectrum_plot_bone_bhc --phantom-profiles case-directory
```

## 2026-10-14 验证

结果目录：`G:/Code/fanproj/fdk-test/out/pmma-al-empirical-calibration-20261014`。

- `fitted-segmentation/calibration.json`：基于重建分割的标定表；`fit.json` 记录样本来源、系数及验证误差。
- 独立留出角度 47,813 个样本，投影 RMSE 0.002612、最大误差 0.02724；拟合输入不含能谱。
- PMMA／铝目标斜率分别为 0.022944 / 0.078275 mm⁻¹。
- `corrected-segmentation.raw`：加载该表，经 CUDA Stage 校正后 FDK 重建。
- 360 帧路径＋查表 GPU 约 78 ms，含投影 I/O 约 0.27 s。只计校正，不含标定、前投、分割或重建。
- 有完整角度覆盖的中央四层（索引 14–17），PMMA 中心／边缘相对偏差由 1.156% 变为 −0.169%。中央分割标签一致率 99.996%，仅用于数字实验评价。
- PMMA 校正后中心均值约 0.02196 mm⁻¹，仍低于切线目标约 4.3%；局部波动未消失，中心标准差由约 0.00121 增至 0.00153 mm⁻¹。不能只凭杯状偏差改善宣称整体图像更好。

该模体沿 Z 为 32 mm，但短探测器不能完整重建全部 32 层，外围层不用于质量结论。`evaluation.json` 明确记录实际评价范围。

此轮验证了两阶段原型，不是实测工程定标，也尚未证明 PMMA／纯铝表能消除头部骨间暗条纹。应用于头部前仍需确定材料等效厚度及独立 HU 标定，检验残余散射、像素积分、分割偏差以及相同采集协议下的独立对象。未改 UI 默认 BHC 或现有头部结果。

### 实际厚度覆盖

`scripts/analyze_calibration_coverage.py samples.csv coverage.json` 对真实样本计数；原生绘图工具 `--calibration-coverage coverage.json coverage.png` 展示格子中心及空缺，不对计数插值。

重建分割样本共 239,051 条，训练候选 191,238 条，验证 47,813 条。5×2 mm 分格、225×52 mm 外包矩形共 1170 格，最外侧格子仅部分落入实际厚度范围。全部角度覆盖 1115 格（95.30%）；训练角度覆盖 1099 格（93.93%），其中 247 格只有 1–3 条训练候选射线。最终每格最多取 12 条，共 9080 条参与拟合。

训练的 71 个空格均在铝 2–30 mm、PMMA 5–125 mm 内；铝 2–10 mm 段覆盖率仅 81.67%。按 1×1 mm 重新分格，训练覆盖率为 50.70%，因此粗网格覆盖率不能当作连续厚度覆盖证明。全部有效射线中，79.07% 仅穿 PMMA、8.96% 仅穿铝、11.97% 同时穿过两者。图中红叉为空格，着色为 log10(候选射线数)，左右图共用色标。

### 与解析谱积分的对照

`multispectrum_bone_bhc_2dcal --export-material-reference scan.toml analytic-reference.json` 导出同协议的归一化能量权重和两材料线性衰减，**仅供评价，不送入经验拟合器**。`scripts/compare_empirical_analytic_calibration.py case-directory` 在 PMMA 0–220 mm、Al 0–50 mm 上，以 0.25×0.1 mm 网格独立计算解析谱积分，共 441,381 点。此解析参照仍是理想仿真模型，不是实测真值。

重建分割拟合表相对解析响应：全域对数投影 RMSE 0.003916，已覆盖格子内 0.003282、空格子内 0.009037。铝 <10 mm 区域 RMSE 0.007061，铝 ≥30 mm 区域 0.002185。全域最大绝对差 0.01920，位于 PMMA=0、Al=6.3 mm。排除解析对数投影 <0.1 的近空气点，相对响应 RMS 0.4466%、95% 分位绝对相对误差 0.6994%、最差 8.01%。透射强度相对误差 RMS 0.3934%、最差 1.938%。

将**同一线性目标系数**用于两表，实际 LUT 校正误差 RMS 0.004098，目标值 ≥0.1 时相对 RMS 0.4578%。这项比较只隔离响应曲面与反演误差，不包含目标系数差异。

更大的问题是目标斜率：解析零厚度斜率 PMMA=0.02315541、Al=0.08649488 mm⁻¹；经验斜率分别为 0.02294398、0.07827463，偏差 −0.913% / −9.504%。若比较两者**各自的原点切线目标**，实际校正相对解析目标 RMS 达 5.269%（目标 ≥0.1），不能用“响应误差不到 1%”掩盖此差异。CAD 精确路径拟合的铝斜率也偏低 9.325%，说明分割不是主要原因；当前五参数拟合及采样加权未准确恢复薄材料附近的导数。

当前结论：厚度覆盖大部分成立，响应面总体接近解析，但薄铝误差与线性目标标定仍需改进；还不能把此经验表视为解析表的等精度替代。原有表、头部数据与默认流程均未因本次评价改动。详见 `analytic-comparison.json` 和 `analytic-error-map.png`。
