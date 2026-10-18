# 多能谱 CT 仿真与重建

本应用提供标签体积多能前投、探测器效应仿真、投影校正、CT 重建及体积后处理。CLI 与 ImGui 客户端共用配置和计算入口，VTK 提供几何、投影及 MPR 显示。

## 从哪里开始

| 目的 | 文档 |
| --- | --- |
| 准备 SDK、编译和启动 UI | [构建与运行](../../doc/build-ui.md) |
| 选择纯投影、纯重建或完整流程 | [配置入口](configs/README.zh-CN.md)及[配置索引](configs/CATALOG.zh-CN.md) |
| 使用 BrainWeb / ICRP 数据 | [内置体素数据](../../data/phantoms/README.zh-CN.md) |
| 导入 MC-GPU 投影 | [外部数据重建](configs/reconstruction/external/README.zh-CN.md) |
| 理解源码归属和处理流程 | [架构与框图](ARCHITECTURE.zh-CN.md) |
| 生成和读取校正表 | [标定产物](CALIBRATION-ARTIFACTS.zh-CN.md) |
| 查看测试范围 | [验证说明](tests/VALIDATION.zh-CN.md) |

按构建文档准备环境后，从仓库根目录运行（`<bin>` 为可执行文件目录）：

```powershell
# 不带 TOML 时加载内置水模预设，可在 Example 菜单切换
& '<bin>/multispectrum_sim.exe' --gui

# 命令行执行已准备好输入的配置
& '<bin>/multispectrum_sim.exe' '<配置文件.toml>'
```

TOML 内相对路径以配置文件所在目录为基准。Example 菜单直接准备内置模体和预设，不要求先生成 TOML；命令行配置所需输入按配置入口文档准备。

## 处理能力与数据约定

- **前投**：独立 CUDA 后端执行标签材料路径积分、多能谱积分、焦点采样和探测器响应；可选散射、量子噪声、串扰、余晖和 DAS。浮点体积前投另有独立接口，供骨硬化重投影等使用。
- **前处理**：空气归一化、散射校正、透射域滤波、水 BHC 和校正缓存。骨硬化按方法要求接入水校正前或后，不能任意交换顺序。
- **重建**：经底层 DLL 执行 FDK、WFBP、内部代数、TIGRE 及统计重建；可用几何与算法组合由公开能力校验。内部 TV 参数只绑定支持它的内部算法，TIGRE TV 使用自己的参数。
- **后处理**：正式配置支持读取 HU 标定表；API 提供 μ/HU 转换、阈值分割、重采样与滤波，支持体积逐层 2D 或空间 3D。并非所有 API 都有 TOML/UI 编辑入口。
- **客户端**：参数编辑、扫描/重建作业、几何相对模式、投影显示、窗宽窗位与可旋转 MPR。计算核心不依赖 UI。

投影 RAW 为连续 float32，顺序 `[view][row][column]`，列最快。负对数投影为 `-log(E/E0)`；能量输出为未归一化的能量积分信号，两者不能混用。体积输出通常为衰减系数 mm⁻¹；显示 HU 需要匹配的水衰减基准。具体空气参考布局和校正资源要求见标定产物文档。

UI 的 `Calibrate` 页签（PrepareCalibrate）仅用于独立标定、查看及导出结果，不自动改写重建配置。当前 UI 重建采用扫描几何或手动几何；水 BHC 使用内置系数或当前能谱在线拟合，不读标定文件；依赖外部表的 2D BHC 暂不接入 UI 重建。导入 TOML 中的几何/BHC/2D BHC 表也不会在 UI 重建中应用。命令行和库 API 的读表能力保留。页面入口位于菜单下方固定的全宽工具栏；窄窗口自动换行，左侧仅显示当前页参数，各页保留独立滚动位置。

## 扩展入口

| 扩展内容 | 说明 |
| --- | --- |
| Stage 契约、元信息和执行器 | [Stage 设计](include/processing/stage/README.md) |
| 前处理 Stage 示例 | [BHC 示例](include/recon/preprocess/examples/README.md) |
| 图像算子与 2D/3D 处理 | [图像处理](include/processing/image/README.md) |
| HU 转换、校正与分割 | [强度 Stage](include/recon/postprocess/stages/README.md) |
| 骨硬化策略与迭代作业 | [BoneBhcStage](include/recon/bhc/README.md) |
| 双材料标定与校正 | [标定](include/recon/preprocess/calibration/README.md)、[校正](include/recon/preprocess/correction/README.md) |
| EBHC / 锥束实验 | [EBHC](include/recon/preprocess/correction/EBHC.md)、[两遍锥束实验](app/experiments/CONE_TWO_PASS.md) |
| 独立模块与可选显示 | [模块边界](modules/README.zh-CN.md)、[VTK 显示](modules/visualization/README.zh-CN.md) |
| 快速绘图 | [线图、二维图与三维图工具](app/plot/QUICK_PLOT.md) |

骨硬化和锥束实验的接口可用不代表所有条件下的效果已验证；以对应模块的实验标记和验证说明为准。历史试验截图、单机耗时与临时输出路径不作为使用说明维护。

## 输出管理

运行前检查配置中的输出目录；大尺寸投影、能量图、校正缓存和重建体积会迅速占用磁盘。只需数值结果时保持 `reconstruction.save_slices=false`。

结束对比实验后可删除对应输出及校正缓存，下次重建会重新计算校正。保留原始模体、外部采集数据、空气参考与标定表；不要将 `out/` 整体删除，其中还可能有构建、SDK 和运行依赖。删除结果后，旧缓存描述文件本身不能代替已删除的数据。

### 角度和运行日志

UI 与新导出的扫描 TOML、几何校正 TOML、几何标定 JSON 均以度显示/保存角度，字段显式使用 `_deg`。内部计算仍为弧度。读取 TOML 时继续识别旧 `_rad` 字段，但同一角度同时出现 `_deg` 与 `_rad` 会报错；几何校正表的 `angle_unit` 必须与字段一致。单排钢珠模型仅估计面内角，UI 对 U/V 倾角标注未估计。

顶部 **Logs** 或 **View → Runtime log** 打开可移动、可缩放的只读日志面板，支持筛选、复制可见内容、清空显示和自动跟随。面板显示应用日志与任务状态，保留最近 2000 行；直接写入终端的第三方库输出不在此范围内。
