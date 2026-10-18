# 当前配置入口

所有 TOML 使用当前 `schema_version = 2`，按**运行内容**分为三类：

| 目录 | workflow.mode | 用途 |
|---|---|---|
| `projection/` | `project` | 标签体积 → 多能投影，保存投影、能量信号和空场 |
| `reconstruction/` | `reconstruct` | 读取已有投影，执行所选前处理、重建与骨硬化校正 |
| `project-and-reconstruct/` | `project_and_reconstruct` | 先仿真，再使用同一采集配置重建 |

子目录只区分解析、迭代、几何标定、质量评价或实验性方法，不改变运行方式。
完整文件及输入要求见 [配置索引](CATALOG.zh-CN.md)。旧 `ui/`、`pipelines/`、顶层 `quality/`、
`calibration/`、`icrp145/` 不再作为入口。UI 与命令行共用这些配置。

来自 MC-GPU 或其他程序的投影也可直接进入纯重建工作流，无需先运行本库前投。
见 [外部仿真数据全链路](reconstruction/external/README.zh-CN.md)：包括 MC-GPU 新导出工具的用法、
total/primary 模板、校正标定条件及几何转换边界。

## 水 BHC 标定准备

带 `fit_from_water_plate = true` 的配置包含标定配方，重建会在 PrepareCalibrate 阶段拟合并将系数留在内存，前处理随后应用。也可提前导出校正表供多次任务复用：

```powershell
multispectrum_sim.exe --calibrate-water-bhc <扫描配置.toml> <输出目录/water.bhc.toml>
multispectrum_sim.exe <输出目录/water.bhc.toml.config.toml>
```

已有校正表则直接设置 `reconstruction.preprocessing.beam_hardening.calibration_file`；路径相对配置文件。
UI 也可在 Beam hardening correction 下生成或选择表。ASKS JSON、双材料骨硬化表和空气 RAW
仍通过原配置路径引用，处理链只加载，不重新标定。表只适用于匹配的能谱、响应与数值域。

## 先跑哪个

| 数据 | 纯投影 | 纯重建 | 投影＋重建 |
|---|---|---|---|
| 20 cm 水模圆扫描 | [water-circle](projection/water-circle.toml) | [FDK](reconstruction/analytic/water-circle.toml) | [完整流程](project-and-reconstruct/analytic/water-circle.toml) |
| 20 cm 水模螺旋 | [water-helix](projection/water-helix.toml) | [WFBP](reconstruction/analytic/water-helix.toml) | [完整流程](project-and-reconstruct/analytic/water-helix.toml) |
| BrainWeb 全头 | [BrainWeb](projection/brainweb-full-fdk.toml) | [FDK](reconstruction/analytic/brainweb-full-fdk.toml) | [完整流程](project-and-reconstruct/analytic/brainweb-full-fdk.toml) |
| ICRP 女性头部 | [female-head](projection/icrp145-female-head.toml) | [WFBP](reconstruction/analytic/icrp145-female-head.toml) | [完整流程](project-and-reconstruct/icrp/icrp145-female-head.toml) |
| ICRP 男性头部 | [male-head](projection/icrp145-male-head.toml) | [WFBP](reconstruction/analytic/icrp145-male-head.toml) | [完整流程](project-and-reconstruct/icrp/icrp145-male-head.toml) |

配套配置写读同一个 `out/config-examples/<示例名>/projection.raw`。运行完整流程会覆盖该示例已有结果；
不同重建结果通常按配置名区分。要并行对比多次实验，请为每次实验另设输出目录。

## 准备内置数据

以下命令均从仓库根目录运行。按 [构建文档](../../../doc/build-ui.md) 编译客户端及
`multispectrum_pack_mat`；将示例中的 `<bin>` 替换为可执行文件目录。
二者使用同一 SDK 运行环境，不需要 Python 或 MC-GPU 源码目录。

首次下载仓库需获取 LFS 数据：

```powershell
git lfs pull
```

BrainWeb 全头直接读取 `data/phantoms/brainweb/normal-1mm.mat`，无需转换。
水模、钢珠模体及 ICRP 的命令行配置需要一次性生成标签 MAT：

```powershell
& '<bin>/multispectrum_pack_mat.exe' --example water-circle out/prepared-examples/water-circle/labels.mat
& '<bin>/multispectrum_pack_mat.exe' --example water-helix out/prepared-examples/water-helix/labels.mat
& '<bin>/multispectrum_pack_mat.exe' --example brainweb-normal out/prepared-examples/brainweb-center33/labels.mat --center-slices 33
& '<bin>/multispectrum_pack_mat.exe' --example icrp145-female-head out/prepared-examples/icrp145-female-head/labels.mat
& '<bin>/multispectrum_pack_mat.exe' --example icrp145-male-head out/prepared-examples/icrp145-male-head/labels.mat
```

工具只准备标签数据并读回校验，不运行正投。已有目标文件会拒绝覆盖；需要重新生成时先移走旧文件。
`--center-slices` 保持原体素中心，截取层数必须与原 Z 层数同奇偶。
程序化模体支持重复 `--set 参数 值`，沿用 ExampleStage 的参数校验；人体数据不支持这些模体参数。

ICRP 原 MAT 存储的是器官编号，**不能直接当材料标签体加载**。以上命令使用仓库中的器官映射、
53 类介质组成与对应男女密度，转换为前投材料标签。其余八个 ICRP 示例同样执行
`--example icrp145-<female|male>-<chest|chest_no_arm|abdomen|abdomen_no_arm>`，
输出到同名 `out/prepared-examples/<id>/labels.mat`。保留源体积 0.5 mm 网格，当前重建预设为
256×256、Z 按物理范围降采样，投影为 1024×256；螺旋圈数和探测器宽度按各部位范围设置。
数据来源及使用条件见 [内置体素数据说明](../../../data/phantoms/README.zh-CN.md)。

几何模体对应关系：

| 配置名 | `--example` |
|---|---|
| tungsten-bead-line-3mm-spacing10 | bead-line |
| tungsten-bead-double-ring | bead-double-ring |
| tungsten-bead-double-ring-cylinder-marker | bead-cylinder-marker |
| tungsten-bead-spiral | bead-spiral |
| tungsten-bead-synchronized-ring-6-marker | bead-sync-6 |
| tungsten-bead-synchronized-ring-12-marker | bead-sync-12 |

生成到 `out/prepared-examples/<example-id>/labels.mat`。这些配置模拟带偏移的采集；
配置里的 `calibration` 分类不代表程序会自动执行珠心检测或标定拟合。
`reconstruction/calibration/` 中读取 `imported/` 的配置则需要原实验的标定投影，不能用任意新模体数据替代。

`water-small-cone-*` 的标签准备命令为：

```powershell
& '<bin>/multispectrum_pack_mat.exe' --example water-circle out/prepared-examples/water-small-cone/labels.mat --set nx 181 --set ny 217 --set nz 33 --set voxel 1 --set support_diameter 160 --set support_height 30
```

## 运行

纯投影与纯重建可分开运行：

```powershell
& '<bin>/multispectrum_sim.exe' example/multispectrum_sim/configs/projection/water-circle.toml
& '<bin>/multispectrum_sim.exe' example/multispectrum_sim/configs/reconstruction/analytic/water-circle.toml
```

或一次运行完整流程，也可加载到 UI 后修改参数：

```powershell
& '<bin>/multispectrum_sim.exe' example/multispectrum_sim/configs/project-and-reconstruct/analytic/water-circle.toml
& '<bin>/multispectrum_sim.exe' --gui example/multispectrum_sim/configs/project-and-reconstruct/analytic/brainweb-full-fdk.toml
```

所有文件路径相对于 **TOML 所在目录**解析；命令行传入的 TOML 路径相对于当前目录。
配置的计算输出统一位于仓库 `out/config-examples/`，准备数据位于 `out/prepared-examples/`。
配置中关闭自动弹出旧 VTK 窗口，显式使用 `--gui` 打开客户端。

投影 RAW 为 float32、`[view][v][u]`、U 最快，保存负对数衰减；能量信号与空场另存。
纯重建不需要标签体积，但需要与投影完全一致的轨迹、探测器、信号约定和必要的空场/校正数据。
水 BHC 拟合还需匹配的能谱；不能直接用任意配置处理未知能谱的实测数据。
预处理校正后缓存的投影也不能再套用一次相同校正。

## 默认值与专用实验

常规平板预设采用 1024×256 投影；改尺寸时保持探测器物理覆盖范围。
读取已有实验 RAW 的标定重建保留原 1024×1024 尺寸，不能将历史数据重新解释为另一种布局。
源标签体积不降采样。前投使用当前默认 kernel 分派，批次缓存上限 256 MiB。
前投示例使用内置能谱生成器；已有生成谱的过滤层和曝光量保留各自实验设置。
外部纯重建不强制指定能谱，只有需要相应物理校正时才提供匹配的谱或标定系数。
内部迭代的正则项只在对应算法中配置，不保留无效的 TIGRE TV 参数。
Huber-TV 的内部近端步数由当前实现自动确定。

金箔、钨丝、低对比度、带 PMMA 壳水模与 PCB 配置属于专用数值实验，保留其测量网格和物理参数。
对应标签体未随仓库提供时，必须按 TOML 的源网格与材料编号准备至 `out/prepared-examples/custom/`，
也可修改 `projection.label_volume` 指向自己的匹配数据，不能拿不同形状的内置水模代替。
配置中钨丝数据文件名的 `0p05` 是历史输入命名，不能据此认定其真实直径。

启用散射的案例需要用户提供与当前几何、谱和探测器采样匹配的
`out/calibration/scatter/runtime-model.json`。仓库没有附带通用散射标定核；未准备模型时用基础示例，
或同时关闭仿真散射及对应散射校正。修改探测器尺寸后必须重新确认核的适用性。
`experimental/` 中的统计优化、散射自适应与 transmission guard 仍是实验性配置。

高级头部配置按实际用途命名：`brainweb-circular-ossart` 和 `brainweb-helical-ossart`。
后者保留 5 圈、每圈 500 帧、40 mm/圈、512×512×400 / 0.5 mm 重建和 50 子集，
其完整物理效应与散射模型要求使它不适合作为首次运行示例。

目录测试使用当前解析器检查所有 TOML、工作流归类、相对路径、输出目录，以及配对投影的几何尺寸。
专用模体与散射数据的真实有效性仍需相应实验数据验证。
