# ICRP145 全材料示例

UI 示例位于 `../ui/icrp/`：男女各五种，分别是 head、chest、chest_no_arm、abdomen、abdomen_no_arm。头部没有另造去手臂版本。

## 材料定义

可供其他仿真读取的独立源数据位于仓库的
[`data/materials/icrp145/`](../../../../data/materials/icrp145/README.zh-CN.md)。
以下 materials TOML 和 organs CSV 均指该目录中的文件。
内置 C++ 组成表由其中的 `media.csv` 生成，运行客户端不需要生成工具。

- 源体积来自 [vox_vol_interconvert](https://github.com/IIIIIsak/vox_vol_interconvert/tree/f78ed9178380ef18038e26818ce695f7b6d05f7f/ICRP145-mat)，使用该版本的 `map.xlsx` 行顺序建立 uint8 标签。170 个唯一器官标签的名字、原始 organ_ID、介质编号和男女密度保存在 `organs-female.csv`、`organs-male.csv`。
- **不使用 MC-GPU 的 38 组代理材料，也不对器官密度取平均。** `materials-female.toml`、`materials-male.toml` 按原介质编号及实际密度区分，男女各 53 项。相同组成、相同密度的器官共享物理材料；不同松质骨、眼组织、乳腺组织等保持区分。
- 元素质量分数来自 [ICRP Publication 145](https://doi.org/10.1177/0146645319893605) 附录 B 表 B.1/B.2，印刷页 89–92；存入 `src/BuiltinIcrpMaterials.inc`，由现有 XCOM 接口查询衰减。男性 43 号是前列腺，女性 43 号是子宫。发布表的质量分数只有有限小数，查询时由 XCOM 归一化。
- 附录给出 1–52 号介质。体外空气 0 号沿用本库既有 `air` 组成及源表密度 0.001205 g/cm³；体内空气 51 号使用附录组成。示例的实际密度以 `map.xlsx` 为准，不覆盖为通用组织的默认密度。
- 每个性别有三个密度为零的异性器官标签，不生成材料；实际十个 RAW 已逐体素检查，这些未映射标签均未出现。遇到任何未映射标签会报错。
- 明确修正上游一处自相矛盾的编号：organ_ID=20000 / voxel_id=169 的 `BB` 被标为 51（空气），但同一行给出 soft_tissue、密度 1.031；附录 B 的 2 号介质明确包含 BB。因此该标签改用 2 号介质，保留原密度。organs CSV 同时保存 `source_medium_id=51` 和修正后的 `medium_id=2`，不修改原始工作簿。
- 原始器官 RAW 保留在 upstream 中。导入后的 RAW 是紧凑的物理材料标签，不再是独立器官编号；需器官分割时使用原始 RAW 与 organs CSV。

这里的“全材料”指按源表的介质和密度完整建模，并非为每个器官虚构一套独立元素组成。模体是上游将 ICRP145 网格栅格化得到的 0.5 mm 体积，并不恢复网格中的微米级薄层。

## 本地数据与导入工具

数据目录默认为 `example/multispectrum_sim/inputs/icrp145/`，已忽略，不随源码分发：

```text
inputs/icrp145/upstream/source/ICRP145-mat/  原始 MAT、map.xlsx
inputs/icrp145/upstream/phantom/            原流程生成的器官 uint8 RAW
inputs/icrp145/prepared/inputs/             本库全材料标签 RAW
out/icrp145/<示例名>/                       前投和重建结果
```

首次将 MAT 转成器官 RAW 时可使用原 ICRP 项目的准备脚本。后续导入、客户端加载和计算使用本库 C++ 工具，不链接、引用该项目的源码，也不需要 Python 运行时。导入器所需的输入仅为 `upstream/phantom/` 中的十个 RAW，材料表已纳入本仓库。

在仓库根目录执行（已生成的数据和配置不会被覆盖，需要选择新的输出目录或由使用者处理旧结果）：

```powershell
cmake --build out/build/main-ui --target multispectrum_icrp_examples multispectrum_sim
& ./out/build/main-ui/example/multispectrum_sim/multispectrum_icrp_examples.exe `
  example/multispectrum_sim/inputs/icrp145/upstream `
  example/multispectrum_sim/inputs/icrp145/prepared `
  --results out/icrp145 --configs example/multispectrum_sim/configs/ui/icrp
```

`--list` 列出十个示例；两个目录后也可指定一个示例 ID。`--materials <目录>` 可选择另一套含 `materials-female.toml`、`materials-male.toml` 的映射表。构建时默认定位到仓库内的表；单独分发导入工具时应同时分发此目录，并显式指定 `--materials`。

TOML 采用相对路径，移动整个仓库和数据后仍可加载。请遵守上游模体数据的使用与再分发条款；源码不附带原始 MAT/RAW 或参考 PDF。

## 扫描与重建默认值

十个配置均用平板螺旋扫描和解析 WFBP，Hann 滤波。SID 709 mm，SDD 1143.699951 mm，探测器 768×256，每圈 360 帧。横向扇角按完整体积包围圆计算，带手臂版本使用更大的物理视野；螺距取等中心纵向覆盖的一半，并在两端留覆盖余量。

原模体不降采样，体素均为 0.5 mm；重建默认 512×512，Z 间距 1 mm，完整覆盖源体积高度。前投使用 120 kVp 生成谱，Al 2 mm、Ti 0.6 mm 滤过，5 mAs/帧，4 条子射线，开启量子噪声和几何通量；散射关闭。重建前开启水板 BHC，参考能量 70 keV。此处未默认启用骨硬化或散射校正。

UI 使用 File → Open configuration 加载 TOML，会同时加载体积和材料；也可以直接启动：

```powershell
& ./out/build/main-ui/example/multispectrum_sim/multispectrum_sim.exe --gui `
  example/multispectrum_sim/configs/ui/icrp/icrp145-female-head.toml
```

不加 `--gui` 则按 TOML 执行计算。完整默认配置有多圈投影，不等同于缩小投影与重建网格的冒烟验证。

## 验证范围

`multispectrum_icrp` 检查 104 套男女介质组成、53 项映射、密度保留、标签重映射、输出保护、十个配置的往返读写，以及带手臂体积的扇角和螺旋覆盖。GPU 回归夹具还提供 38、53、64 材料场景，其中 64 材料使用 256 个能量 bin 和 33 条子射线，覆盖系数表全局内存回退及跨 warp 批次采样。

全材料会增加路径数组和谱积分的工作量；路径几何遍历仍只做一遍，不是按材料重复射线遍历。不能将“没有蒙特卡洛逐光子输运开销”理解成材料数增加完全免费。
