# 内置体素示例数据

客户端 Example 菜单直接读取压缩 MAT 数据，不需要 MATLAB 或 Python，也不需要先生成 TOML。选择示例会同时应用体积尺寸、体素间距、材料映射、扫描和重建参数；此后可以在界面中调整扫描参数。

## 数据布局

| 目录 | 内容 | 网格与材料 |
| --- | --- | --- |
| `brainweb/normal-1mm.mat` | BrainWeb 正常头部、离散标签 | 181 × 217 × 181，1 mm，10 种材料 |
| `icrp145/female/data_*.mat` | 女性头部、胸部、腹部及胸腹无手臂版本 | 保留原始 0.5 mm 网格，53 种物理材料 |
| `icrp145/male/data_*.mat` | 男性对应的五个版本 | 保留原始 0.5 mm 网格，53 种物理材料 |

MAT 第一维为 X，内存顺序是 X 最快，随后 Y、Z；加载时不交换轴，也不翻转。ICRP MAT 保留原始器官编号，由 `data/materials/icrp145/organs-*.csv` 与 `materials-*.toml` 映射到材料标签。不能把原始 ICRP 器官编号当成 uint8 材料编号直接加载。

普通 Label volume 文件入口支持单个三维实数数组的 MAT v5（含 zlib 压缩）或 RAW uint8。通用入口要求已配置的 X/Y/Z 与文件一致，且数值是 0–255 的整数标签。MAT v7.3/HDF5 暂不支持。包含多个三维数组的文件会报歧义，避免错误选择。保存内置示例配置时，客户端会另外导出已映射的压缩 uint8 MAT，使 TOML 可以独立重新加载。

## 运行

从构建目录运行 `example/multispectrum_sim/multispectrum_sim.exe`，在 Example → BrainWeb 或 ICRP145 选择数据。

也可以使用 `multispectrum_sim --gui-example brainweb-normal out/examples/brainweb`，或将示例 ID 换为 `icrp145-female-head`、`icrp145-male-chest_no_arm` 等。`--list-examples` 列出完整目录。头部以外的数据较大，解压和首次建立显示体积需要等待。

ICRP 默认采用完整材料的多能正投与螺旋 WFBP，探测器宽度随体积范围匹配；BrainWeb 默认圆扫描与 FDK。上述配置用于演示，可以在计算前修改，加载示例不会自动运行正投。

默认采用较小的计算尺寸，原始模体与材料标签保持不变：BrainWeb 投影为 1024×256、U/V 间距 0.5/1.5 mm，重建为 91×109×91；ICRP 投影为 1024×256，重建为 256×256×ceil(原始层数/4)。像素和重建体素间距按原物理范围匹配，扫描帧数、螺距、每像素采样数不变。重建体素数约减少 87.5%；投影尺寸按当前 1024×256 预设计算，运行时间需以实际测量为准。需要精细图像时可在 UI 调大尺寸并相应调整间距。已有 TOML 按其原参数加载，不自动缩小。

数据根目录优先使用环境变量 `YK_EXAMPLE_DATA_ROOT`，其下应有 `materials/` 和 `phantoms/`；其次查找当前目录的 `data/`，最后使用构建时记录的仓库 `data/`。复制客户端时可以把该目录一起复制，并设置环境变量。

## Git LFS

仅 `data/phantoms/**/*.mat` 使用 LFS。源码、材料 CSV/TOML 和运行结果不受这条规则影响。克隆后运行 `git lfs install`、`git lfs pull` 下载实际数据。缺少 LFS 实体时，加载器会给出明确提示。`SHA256SUMS.txt` 记录实体文件校验值。

## 数据来源与再分发状态

ICRP145 材料组成来自 ICRP Publication 145，原始体素 MAT 从本地已有 ICRP 示例资料整理，保持原文件内容。材料说明见 `../materials/icrp145/README.zh-CN.md`。ICRP 原始资料的本地 LICENSE_NOTES.md 明确说明：脚本仓库不授予原始数据的再分发权。因此，本地可用于已有授权下的验证；公开发布本数据包之前，需要核实相应授权，项目源码许可证不覆盖这些模体数据。

BrainWeb 数据来自 [BrainWeb](https://brainweb.bic.mni.mcgill.ca/brainweb/) 的 `phantom_1.0mm_normal_crisp`，仅做无损 RAW → MAT 打包。RAW SHA-256 为 `8693f9fde4f2b233237f2c5d0c4e4e2aac705bb08efd1e13ae70079ab7456c54`。引用应包含 Cocosco 等（1997）、Kwan 等（1999）、Collins 等（1998）的 BrainWeb 工作。当前未确认公开再分发许可，远程发布前应核实。

LFS 配置本身不构成数据授权，当前数据包先完成本地组织与验证。
