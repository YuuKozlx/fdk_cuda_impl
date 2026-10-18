# 外部仿真数据：从 MC-GPU 到重建

重建工作流不依赖本库前投，不要求原标签体积或材料清单。这里提供两个处理模板：

- [total → FDK](mcgpu-total-fdk.toml)：散射校正、水 BHC、骨校正的配置位置齐全；未取得匹配标定时均关闭。
- [primary → OS-SART + Huber-TV](mcgpu-primary-ossart.toml)：不扣除模体散射，可配置水 BHC。

模板的 360 帧、1024×256、SID=709 mm / SDD=1143 mm **只是填写示例**，不是某次 MC-GPU 扫描的实参。
建议用 MC-GPU 导出的几何创建配置，再加入这些模板的处理部分。
全流程为：

```text
MC 物体能量堆栈 + 匹配空场
  → 导入：空场归一化、负对数、几何与单位转换
  → 可选散射校正（内部恢复到能量域后处理，再取对数）
  → 可选水 BHC
  → FDK / 迭代重建
  → 可选模型骨校正及重建更新
  → RAW 重建体、可选切片 / UI 的 HU 窗宽窗位显示
```

骨校正实际位置由方法决定，并非所有方法都放在重建之后。上面的可选骨校正对应
`physical_after_water`：使用初始体积建立材料模型、修正投影并重新重建。
当前通用后处理 Stage 并不是每项都已有 TOML 入口；此处没有声称自动完成 DICOM 导出或任意降噪。

## 推荐：由 MC-GPU 导出实际扫描配置

本次参考的是 `mcgpu-modern` 工作区中新增、尚未提交的 `mcgpu_export_fdk` 应用工具，
其源码为 `src/app/FdkExport.cpp` 和 `FdkExportCli.cpp`。不向本库引入 MC-GPU 源码依赖。
确认使用包含该目标的 MC-GPU 构建；普通旧版发行包可能没有此工具。

物体、空场均需以 `--output-stack` 完成，保留 `<prefix>_stack.json` 和 total/primary 堆栈。
物体扫描 TOML 必须对应这些投影。以下命令中的前缀不带 `_total.stack.raw` 后缀：

```powershell
& '<mcgpu-bin>/mcgpu_export_fdk.exe' scan.toml --projection-prefix results/object --air-prefix results/air --output results/reconstruction-input --volume 256 256 128 --voxel-mm 1 1 1
& '<fdk-bin>/multispectrum_sim.exe' results/reconstruction-input/reconstruct_total.toml
& '<fdk-bin>/multispectrum_sim.exe' results/reconstruction-input/reconstruct_primary.toml
```

导出包包含 `total/primary.line_integrals.raw`、`.energy.raw`、`.air.raw`、两份重建 TOML 和扫描几何 JSON。
生成的 TOML 可直接执行**不附加校正的 FDK**，不必先使用这里的固定几何模板。
想用 OS-SART 时保留导出的全部 `[geometry]`，将 `[reconstruction]` 的类型及算法参数改成迭代模板的内容。
不要把模板中的输入/输出路径覆盖到导出包配置中；导出包采用自身目录内的相对路径。

当前导出工具仅支持完整均匀 360°、无轴向平移、单源单平板圆扫描。
它核对实际逐帧源/探测器基向量，完成 cm→mm、角度方向转换，不翻转或转置投影。
螺旋、柱面和多源扫描不能用这条导出路径；本库支持这些重建几何，不代表当前 MC 导入器已支持。
`--volume` 与 `--voxel-mm` 必须同时提供，指定世界坐标重建网格；默认使用带姿态模体的轴对齐包围盒。
导出目标必须是新目录。空场可一帧广播，也可与各视图逐帧配对。

## 信号与物理标定

输入重建的是 float32 `[view][v][u]`、U 最快的**负对数投影**，不是能量 RAW。
导出工具计算 `-log(clamp(I/I0, 1e-6, 1))`；因此比空场更亮的像素被截至 0，严重光子饥饿也被截断。
这不是去噪或散射校正。不要重复归一化或再次取对数。
MC 原能量和复制的空场保持 `eV/cm²/history`，不是本库仿真的绝对 mAs 信号。
配对空场必须具有相同能谱、照野、探测器响应、附件和每历史归一化，所有有效像素空场需严格大于 0。

`total` 含模体散射，`primary` 已排除模体散射；后者仍可能有多能硬化、有限焦点或探测器效应。
散射校正模型必须匹配 MC 的响应与归一化约定。仅把历史 fASKS JSON 路径填进去并不构成匹配标定。
单帧空场广播也要求扫描中空场分布不随视角改变。

水 BHC 有两种接法：

1. **已有实验/MC 水板标定多项式**：`fit_from_water_plate=false`，填写从常数项开始的
   `coefficients=[a0,a1,...]`，再同时开启总预处理与 BHC。无需伪造前投能谱或 mAs。
2. **按谱拟合水板**：设置 `fit_from_water_plate=true`，提供本库支持的匹配光子数谱及参考能量。
   当前水板拟合按能量积分探测器响应建模，不能自动复现任意 MC 闪烁体输运、计数模式或附件响应。
   简单的相同 kVp 不代表等效谱相同；不要用本库默认 120 kVp 谱替代未知 MC 实际响应。

直接复用 MC 能谱文件前应核对其格式；只有本库明确支持的格式（例如有
`Format: MCGPU-SPEC 1.0` 标记的边界谱）才按相应协议读取，不能把任意 MC 采样概率表当绝对光子数谱。
本工作区的 MC 导出器没有自动转换能谱或探测器标定。

骨校正还需要匹配的材料模型、能谱和 HU/衰减基准。初次对照建议先比较未经额外校正的 total 与 primary，
再逐项开启已标定的处理，避免把几何、散射和硬化误差混在一起。

## 使用本目录的模板

### 噪声较高时：投影平滑与引导滤波

在 `[reconstruction.preprocessing] enabled=true` 的配置中添加：

```toml
[[reconstruction.preprocessing.filters]]
operation = "gaussian"
radius_pixels = 5
sigma_pixels = 1.5

[[reconstruction.preprocessing.filters]]
operation = "guided"
radius_pixels = 4
epsilon = 0.0001
```

该有序列表逐帧在 CUDA 上处理归一化透射率；启用散射时在扣除散射之后、取对数之前执行，
未启用散射时由输入 `exp(-p)` 转换后执行，再进入 BHC。
引导滤波为自引导，epsilon 使用透射率平方的单位；没有使用 MC primary 作引导。
原来的 `scatter.guide` 只影响散射估计，不能代替这个作用于测量投影的滤波列表。
滤波会改变分辨率；比较时应对 primary 施加相同滤波。
参数是本次测试起点，不是适用于所有投影或剂量的默认值。设置变更会使校正缓存失效。

### 输入与路径

把导出的线积分及空场放到仓库 `out/config-examples/imported/mcgpu/`，
或修改模板中的路径；随后用导出的真实几何替换模板 `[geometry]`。
模板输出到仓库 `out/config-examples/mcgpu-*/`。外部数据不要求标签体或本库前投材料清单。
默认关闭的校正不是已经标定的算法参数；在未提供系数或模型前，不能把开关打开即视为完成标定。
