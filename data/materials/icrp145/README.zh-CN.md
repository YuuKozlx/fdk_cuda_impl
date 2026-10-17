# ICRP145 材料源数据

这个目录独立于 CUDA、VTK、TOML 客户端和 MC-GPU，可直接给其他解析投影、蒙特卡洛或材料分析程序使用。

## 文件和单位

| 文件 | 内容 |
| --- | --- |
| `media.csv` | 104 套男女介质组成（每性别 52 套），以及一套体外空气；材料 ID、性别、介质编号、名称、密度、来源标识和逐元素质量分数 |
| `organs-male.csv`、`organs-female.csv` | 原始器官标签、organ_ID、器官名称、上游介质编号、实际采用编号、性别相关器官密度 |
| `materials-male.toml`、`materials-female.toml` | 本库导入器使用的标签到材料预设映射，每个性别 53 项 |

密度单位为 **g/cm³**。`media.csv` 中 H、C、N、O、Na、Mg、P、S、Cl、Ar、K、Ca、Fe、I 列均为 **质量分数 0–1**，不是原子数比，也不是质量百分数；零表示没有该元素。原子序数分别为 1、6、7、8、11、12、15、16、17、18、19、20、26、53。

发布表有小数舍入，质量分数和可略偏离 1。使用时以各项除以总和归一化，不应将缺失的尾差任意补给某个元素。本库 XCOM 接口负责归一化。

`media.csv` 的密度是发布表的介质参考密度；具体模体请优先使用 organs CSV 中的实际器官密度。同组成且同密度的多个器官可共享物理材料，仍可保留独立器官标签用于统计。

## 来源与修正

- `ICRP145_B1`：ICRP Publication 145，Annex B，Table B.1，男性，印刷页 89–90。
- `ICRP145_B2`：同书 Table B.2，女性，印刷页 91–92。
- [论文 DOI / 原始发布页](https://doi.org/10.1177/0146645319893605)。这里只记录数值数据及来源，不包含论文全文或原始模体。
- `existing_air_preset`：本仓库已有 `air` 预设的体外空气组成，不冒充附录 B 数据。体内空气使用附录 51 号介质。
- 器官对应表来自 [vox_vol_interconvert 的固定版本](https://github.com/IIIIIsak/vox_vol_interconvert/tree/f78ed9178380ef18038e26818ce695f7b6d05f7f/ICRP145-mat) 中的 `map.xlsx`；uint8 标签按唯一 organ_ID 首次出现顺序编号。
- 上游 BB / organ_ID=20000 / voxel_id=169 存在矛盾：介质编号写成空气 51，但名称、soft_tissue 分类、1.031 密度，以及附录 B 对 BB 的定义均指向 2 号软组织。本目录显式改用介质 2，同时保留 `source_medium_id=51` 以便追溯；没有修改原工作簿。
- 每个性别密度为零的异性器官不加入物理材料映射。

其他仿真可以只读取 CSV：按性别和 `medium_id` 查组成，再由器官映射取实际密度。应遵守原始数据来源的适用条款；本目录不额外声明获得原模体或原出版物的再分发许可。

## 本库如何使用

**组成的唯一维护来源是 `media.csv`。** 修改后在仓库根目录运行：

```powershell
powershell -NoProfile -File tools/generate_icrp_materials.ps1
powershell -NoProfile -File tools/generate_icrp_materials.ps1 -Check
```

脚本仅依赖 PowerShell，生成 `example/multispectrum_sim/src/BuiltinIcrpMaterials.inc`，校验 ID、性别、介质覆盖、密度、来源和质量分数。生成的 C++ 表随库编译，运行时不需要 CSV 解析器或生成脚本。

标签与密度对应关系维护在 organs CSV 和映射 TOML 中；修改介质归属时要同步对应映射并重新导入体积。示例导入、扫描与运行方式见 [ICRP145 示例说明](../../../example/multispectrum_sim/configs/icrp145/README.zh-CN.md)。
