# TOML 配置化重建与前投手册

## 运行方式

一个 TOML 文件可包含多个 `[[case]]`。不传 `--case` 时顺序执行全部任务：

```powershell
ykcbct_manual_tests --config test-configs/fdk.toml
ykcbct_manual_tests --config test-configs/ossart.toml --case catphan-ossart-tv
ykcbct_manual_tests --config test-configs/victre-breast-fdk.toml --case victre-breast-fdk-360
ykcbct_manual_tests --config test-configs/forward-circular.toml
ykcbct_manual_tests --config test-configs/forward-planar.toml --case planar-offset-source-joseph
ykcbct_manual_tests --config test-configs/cyl-fp-bp.toml
ykcbct_manual_tests --config test-configs/wfbp-flat.toml
```

相对路径始终以 TOML 文件所在目录为基准，不受 Visual Studio 或终端工作目录
影响。输出目录自动创建。所有可执行 TOML 都集中在仓库根目录的 `test-configs/`，
不与 `Yktest` C++ 源码混放；`Yktest/config/` 仅包含 TOML 执行器实现，不是配置文件。

配置按算法拆分：

| 文件 | pipeline |
|---|---|
| `fdk.toml` | 普通 FDK |
| `sirt.toml` | SIRT |
| `sart.toml` | SART |
| `ossart.toml` | OSSART 与可选 smoothed-TV |
| `cgls.toml` | CGLS |
| `tigre-gradient-local.toml` | 本地复现的 TIGRE 梯度/POCS/PCSD 系列 |
| `pwls.toml` | 通用 PWLS |
| `fdk-ossart.toml` | FDK 初值加 OSSART 精修 |
| `fdk-cgls.toml` | FDK 初值加 CGLS 精修 |
| `victre-breast-fdk.toml` | VICTRE 360/720 真实数据 FDK |
| `BrainWeb-fdk.toml` | BrainWeb 720/360 投影真实数据 FDK，依赖本地 `example` 数据 |
| `ICRP-fdk.toml` | ICRP 胸部/头部真实数据 FDK，依赖本地 `example` 数据 |
| `projection-raw.example.toml` | 通用 raw 投影输入模板 |
| `forward-circular.toml` | 圆扫描 Joseph/Siddon 前投 |
| `forward-helical.toml` | 平面探测器螺旋前投 |
| `forward-planar.toml` | 固定平面探测器与椭圆源轨迹前投 |
| `forward-cylindrical.toml` | 实验性圆柱弧面探测器前投 |
| `cyl-fp-bp.toml` | BrainWeb 圆柱探测器匹配 FP/BP |
| `wfbp-flat.toml` | BrainWeb 平板探测器螺旋 wFBP |
| `wfbp-equiangular.toml` | BrainWeb 焦点中心等扇角弧面螺旋 wFBP |
| `wfbp-cylindrical.toml` | BrainWeb 一般圆柱探测器螺旋 wFBP |
| `helical-pwls-flat.toml` | 同一 BrainWeb 平板螺旋几何的普通全批量 PWLS 定量对照 |

## 前投任务

前投配置使用 `task = "forward-projection"`，仍通过相同的 `--config` 和
`--case` 参数运行。输出投影是 float32 little-endian raw，布局固定为
`[view][v][u]`。四类几何分别放在独立配置文件中，避免把不相关参数混在一起。

### 前投通用参数

| 位置/参数 | 必需 | 默认值 | 含义 |
|---|---:|---:|---|
| 顶层 `task` | 是 | - | 必须是 `forward-projection`，否则不会进入前投执行器。 |
| `case.name` | 是 | - | case 唯一名称，可由 `--case` 单独选择。 |
| `case.device` | 否 | `0` | CUDA device id。 |
| `scan.nu`, `scan.nv` | 是 | - | 探测器 U 列数、V 行数。 |
| `scan.views` | 是 | - | 视图总数，输出元素数为 `views*nv*nu`。 |
| `scan.du_mm`, `scan.dv_mm` | 否 | `1.0` | 探测器像素物理尺寸，单位 mm。 |
| `scan.offset_u_mm`, `scan.offset_v_mm` | 否 | `0.0` | 探测器物理偏移，单位 mm；平面 CT 模型当前不支持。 |
| `scan.source_offset_x_mm`, `source_offset_y_mm`, `source_offset_z_mm` | 否 | `0.0` | 源端在扫描仪初始局部坐标系中的校准偏移，单位 mm；builder 会将其烘焙进逐视图 `src`。 |
| `scan.sid_mm` | 是 | - | 源到等中心距离，单位 mm，必须大于 0。 |
| `scan.sdd_mm` | 是 | - | 源到探测器距离，单位 mm，必须大于 SID。 |
| `scan.start_angle_deg` | 否 | `0.0` | 第一视图角度，单位度。 |
| `scan.scan_range_deg` | 否 | `360.0` | 全部视图覆盖的角度范围，不重复采集终点。 |
| `scan.direction` | 否 | `1` | `1` 为角度递增，`-1` 为角度递减。 |
| `scan.tilt_u_deg`, `tilt_n_deg`, `tilt_v_deg` | 否 | `0.0` | 平面探测器三个旋转分量，单位度；只用于 circular/helical。 |
| `volume.nx`, `ny`, `nz` | 是 | - | 输入体 X/Y/Z 体素数。 |
| `volume.voxel_mm` | 是 | - | `[x,y,z]` 体素尺寸，单位 mm。 |
| `volume.offset_mm` | 否 | `[0,0,0]` | 输入体中心的 `[x,y,z]` 物理偏移，单位 mm。 |

### 前投几何参数

| `geometry.model` | 实际几何与能力边界 |
|---|---|
| `circular` | 普通圆锥束圆轨迹，支持探测器 U/V offset 和三个 tilt。 |
| `helical` | 平面探测器螺旋轨迹，`pitch_mm` 表示每整圈 Z 位移，`start_z_mm` 是角度 0 处 Z；支持 offset/tilt。 |
| `planar` | 固定平面探测器加横向旋转源轨迹，`source_lateral_mm` 控制源点横向半径。当前 builder 不支持探测器 offset/tilt，配置了会直接报错。 |
| `planar-ellipse` | 固定平面探测器加椭圆源轨迹，两个半轴由 `source_axis_x_mm`、`source_axis_z_mm` 指定；同样不支持 offset/tilt。 |
| `cylindrical` | 实验性圆柱弧面逐视图几何，只能配 `cylindrical-joseph`；它不是平面探测器插值，也不是完整圆柱重建器。 |

| 参数 | 适用模型 | 必需/默认值 | 含义 |
|---|---|---|---|
| `geometry.pitch_mm` | helical, cylindrical | 默认 `0.0` | 每旋转一整圈的 Z 位移，单位 mm；0 表示圆轨迹高度不变。 |
| `geometry.start_z_mm` | helical, cylindrical | 默认 `0.0` | 角度 0 处源点和探测器的 Z 坐标，单位 mm。 |
| `geometry.views_per_rotation` | helical, cylindrical | 是 | 每圈视图数，仅校验 scan 角步长，不生成另一份角度。 |
| `geometry.source_lateral_mm` | planar | 默认 `0.0` | 源点横向旋转半径，单位 mm；0 表示源点固定。 |
| `geometry.source_axis_x_mm`, `source_axis_z_mm` | planar-ellipse | 是 | 椭圆源轨迹两个半轴，单位 mm，均须大于 0。 |
| `geometry.curvature_radius_mm` | cylindrical | 默认 SDD | 圆柱探测器曲率半径，单位 mm，可与 SDD 不同。 |
| `geometry.channel_angle_step_rad` | cylindrical | 默认 `0.0` | 相邻通道角步长；0 时由 `du_mm/radius` 推导。 |

### 前投算子、输入和输出

普通 `circular/helical/planar/planar-ellipse` 可选
`projector.model = "joseph" | "siddon"`，二者都通过公共
`SConeProjGeomVec` 前投接口。圆柱模型使用独立的匹配离散算子；
`curvature_radius_mm` 是曲率半径，和 SDD 可以不同，
`channel_angle_step_rad = 0` 时按 `du_mm / curvature_radius_mm` 推导，
`samples_per_voxel` 控制射线采样密度。

| 参数 | 必需/默认值 | 含义 |
|---|---|---|
| `projector.model` | 是 | 普通几何选 `joseph` 或 `siddon`；圆柱几何选 `cylindrical-joseph` 或 `cylindrical-siddon`。后者只提供 Siddon FP，BP 仍由 `operator.backprojector` 选择。 |
| `projector.samples_per_voxel` | 圆柱默认 `1.0` | 圆柱射线采样密度，必须大于 0；提高会增加耗时。 |
| `input.mode` | 默认 `phantom` | `phantom` 或 `volume-raw`。 |
| `input.phantom` | 默认 `catphan` | 内置 `basic`、`catphan`、`arrow`。 |
| `input.volume` | volume-raw 必需 | raw 路径，布局 `[z][y][x]`，尺寸必须严格匹配。 |
| `input.scalar_type` | 默认 `float32` | `float32` 或 `uint8`。 |
| `input.scale` | 默认 `1.0` | 读取体数据后统一乘的比例；BrainWeb 材料标签示例使用 `0.01`。 |
| `output.projection` | 是 | float32 raw 输出路径，布局 `[view][v][u]`。 |
| `output.preview` | 否 | 指定后输出单视图 BMP；仅用于目视检查。 |
| `output.preview_view` | 默认 `0` | BMP 对应视图序号，范围 `0..views-1`。 |

前投输入有两种：`mode = "phantom"` 支持 `basic`、`catphan`、`arrow`；
`mode = "volume-raw"` 时必须提供 `volume`，布局为 `[z][y][x]`。文件字节数
必须与 `scalar_type` 和体素数严格匹配。`[case.output]` 必须提供 `projection`，可选
`preview` 和 `preview_view` 输出指定视图的 BMP。所有相对路径仍以 TOML 所在目录
为基准。

当前前投配置只生成均匀角度序列。`views_per_rotation` 不生成角度，只校验
`scan_range_deg/views` 是否与每圈视图数一致；角度仍由 `views`、
`start_angle_deg`、`scan_range_deg`、`direction` 唯一生成。两者不一致时直接报错，
不会静默采用其中一份。

### CylFpBp 正反投

`cyl-fp-bp.toml` 使用 `task = "cyl-fp-bp"`，并要求
`geometry.model = "cylindrical"` 与
`projector.model = "cylindrical-joseph"`。`[case.operator]` 支持：

| `operator.mode` | 行为 |
|---|---|
| `forward` | 只执行圆柱探测器 FP，与 `forward-cylindrical.toml` 等价。 |
| `backproject` | 从 `input.mode = "projection-raw"` 的 float32 `[view][v][u]` 投影执行匹配 BP；必须设置 `output.volume`。 |
| `forward-backproject` | 先 FP，再执行 `operator.backprojector` 选择的 BP；除投影外还必须设置 `output.volume`。 |

`operator.backprojector` 选择圆柱 BP，省略时保持 `v3`：

| 值 | 行为与限制 |
|---|---|
| `v3` | 默认体素驱动近似 BP，支持一般圆柱 `R!=SDD`，适合带归一化的迭代更新。 |
| `fdk` | 圆柱 FDK 空间权重，只接受源中心等角弧面 `R=SDD`。 |
| `fdk-matched` | 带曲面 Jacobian 和体素/像素尺度的 FDK-matched 权重，只接受 `R=SDD`。 |

`v3` 不是默认纹理 FP 的严格离散转置。`fdk` 和 `fdk-matched` 也不是完整
解析重建流程：它们不包含滤波、短扫描/冗余权重和角度归一化，因此
`forward-backproject` 的体数据不能直接当作定量重建结果。对于 `R!=SDD`
投影，必须先由上游重映射到 `R=SDD`，再选择两种解析 BP。

## 螺旋 wFBP 任务

wFBP 使用独立的 `task = "wfbp-reconstruction"`。构建 Yktest 时必须启用
`-DYKCBCT_BUILD_HELICAL=ON`；关闭时执行器会直接报告功能未编译。示例采用
`example/phantom/brainweb_normal_1mm_181x217x181_uint8.raw`，探测器为
`1024x256`、`SID=710 mm`、`SDD=1170 mm`、像素尺寸 `0.35 mm`。

### wFBP 扫描与输入

`[case.scan]` 和 `[case.volume]` 的尺寸、距离、offset 语义与前投任务相同。
`[case.geometry]` 必须提供：

| 参数 | 含义 |
|---|---|
| `pitch_mm` | 每整圈的 Z 位移，必须大于 0。 |
| `start_z_mm` | 第一视图的源点/探测器 Z 位置。 |
| `views_per_rotation` | 原始采集每圈帧数，必须与 `scan_range_deg/views` 严格一致。 |

`input.mode = "projection-raw"` 时读取 float32 线积分投影，布局为
`[view][v][u]`。`input.mode = "volume-raw"` 是自测路径：先用与探测器类型匹配
的 FP 生成投影，再执行 wFBP，支持 `float32`/`uint8` 和 `scale`。该路径不伪造
交错焦点，因此 FFS 必须使用真实 `projection-raw`。

### `[case.wfbp]`

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `input_detector` | `flat-panel` | `flat-panel`、`equiangular-arc` 或 `cylindrical-arc`。 |
| `focal_spot_mode` | `none` | `none`、`phi`、`z`、`phi-z`。圆柱适配当前不支持 FFS。 |
| `arc_channel_angle_step_rad` | `0` | 等扇角弧面输入的相邻通道扇角；该输入模式必须显式给出正值。 |
| `arc_principal_channel` | `-1` | 弧面主射线通道；负值表示由中心/offset 推导。 |
| `arc_curvature_radius_mm` | `0` | 一般圆柱输入的物理曲率半径，必须大于 0，不等同于 SDD。 |
| `redundancy_flat` | `0.6` | FreeCT `W(q)` 权重的平顶区，范围 `[0,1)`。 |
| `angle_tolerance` | `1e-3` | 原始视图均匀角步长的相对容差。 |
| `anode_angle_deg` | `0` | z-FFS/联合 FFS 的阳极角；启用对应模式时必须为 `(0,90)`。 |
| `reverse_row_interleave` | `false` | 交换 z-FFS 两个焦点的行交错顺序。 |
| `filter_cutoff` | `1.0` | FreeCT ramp 截止比例，范围 `(0,1]`。 |
| `filter_apodization` | `1.0` | FreeCT 窗混合参数，范围 `[0,1]`。 |

wFBP 内部始终转换成等扇角表示后执行 fan-to-parallel 重排、滤波和加权反投。
平板输入会先做平板到等扇角插值；一般圆柱输入会根据曲率半径做圆柱表面到
焦点中心等扇角插值。输出是 float32 `[z][y][x]` raw。存在真值时，BMP 从左到右
依次为真值、原始重建值和绝对误差；前两幅使用同一窗宽，不做最佳拟合缩放。

### `[case.validation]`

| 参数 | 含义 |
|---|---|
| `maximum_absolute_nrmse` | 原始重建与真值的 `||recon-truth||2 / ||truth||2` 上限。 |
| `maximum_mae` | 原始重建的平均绝对误差上限，单位与输入衰减系数一致。 |
| `maximum_absolute_bias` | 全体素平均值偏差绝对值上限。 |
| `maximum_material_bias` | `uint8` 标签模体各标签重建均值偏差的最大值上限。 |
| `maximum_background_mean` | 标签 0 区域重建均值绝对值上限，用于拦截背景抬升。 |
| `minimum_correlation` | Pearson 相关系数下限，只辅助检查结构，不代替绝对误差。 |
| `minimum_slice_correlation` | 中心层 cosine 下限，只辅助检查空间结构。 |

日志中的 `fit-scale` 和 `fit-NRMSE` 仅用于判断是否存在单一比例误差，绝不参与
PASS/FAIL。三份 BrainWeb 示例使用 720 视图覆盖两圈、每圈 360 视图、`20 mm`
螺距，并裁取完整模体的中心 41 层，以控制 10 GiB 显卡上的回归测试显存。
它们是带绝对数值门槛的回归用例，但不等于已经验证完整 181 mm 轴向覆盖；真实
扫描仍应按实际床速设置圈数、`pitch_mm`、`start_z_mm` 和重建 Z 范围。

`helical-pwls-flat.toml` 通过 `task = "helical-reconstruction"` 和
`reconstructor = "pwls"` 在同一平板螺旋 geometry 上运行普通 PWLS。为保证数据项
确实使用 `A^T` 梯度，该定量用例固定 `subsets = 1`，默认采用数值伴随误差约
`1e-4` 的 `FP_Joseph/BP_Joseph`，不使用误差约 `0.72` 的 `BP_Joseph_v3`。
材料日志同时给出全部标签体素和六邻域腐蚀一层后的 `core-mean`；前者包含边缘
部分容积效应，后者更适合判断材料内部是否恢复到真值。

配置产生的 `test-configs/output/` 内容都是可复现测试产物，已由同目录 `.gitignore`
排除，不属于源代码。需要重新生成时直接运行对应 TOML；不要把 raw/BMP 提交到仓库。

## 顶层参数

| 参数 | 必需 | 默认值 | 含义与选择建议 |
|---|---:|---:|---|
| `name` | 是 | - | case 唯一名称，用于 `--case`。同一文件不能重名。 |
| `pipeline` | 是 | - | 支持 `fdk`、`sirt`、`sart`、`ossart`、`cgls`、`tigre-gradient-local`、`pwls`、`fdk-ossart`、`fdk-cgls`。完整圆轨迹快速重建优先 FDK；数据不完整、要加约束时选择迭代算法。 |
| `device` | 否 | `0` | CUDA device id。单卡通常保持 0。 |
| `batch_views` | 否 | `chunk_views` | raw FDK 每次从磁盘读取并在线提交的视图数。增大可减少 IO 调用，但两个 pinned buffer 各占 `nu*nv*batch_views*4` 字节。一般取 8-32。 |
| `chunk_views` | 否 | `32` | FDK pipeline 内部工作区和 kernel 分块，不等于在线 batch。当前 FDK 上限为 32；显存紧张取 8/16，否则取 32。 |

## `[case.scan]` 扫描几何

| 参数 | 必需 | 默认值 | 单位/含义 |
|---|---:|---:|---|
| `nu`, `nv` | 是 | - | 探测器 U 列数和 V 行数，必须与 raw 每帧尺寸一致。 |
| `views` | 是 | - | 投影视图数，raw 文件必须恰好含这么多帧。 |
| `du_mm`, `dv_mm` | 否 | `1.0` | 探测器像素物理尺寸，单位 mm。应使用校准值，不能用等中心折算后的像素尺寸。 |
| `offset_u_mm`, `offset_v_mm` | 否 | `0.0` | 探测器中心相对几何中心的物理偏移，单位 mm。来自几何标定；正负方向遵循库的 U/V vector。 |
| `sid_mm` | 是 | - | source-isocenter distance，单位 mm。 |
| `sdd_mm` | 是 | - | source-detector distance，单位 mm，必须大于 SID。 |
| `start_angle_deg` | 否 | `0.0` | 第一帧角度，单位度。 |
| `scan_range_deg` | 否 | `360.0` | 整个采集覆盖范围，不是最后一帧角度。均匀 360 帧完整扫描会生成 0° 到 359°。 |
| `short_scan` | 否 | `false` | 是否启用 Parker 短扫描加权。只有扫描覆盖满足短扫描条件且几何确实是短扫时才设为 true。 |
| `direction` | 否 | `1` | `1` 表示角度递增，`-1` 表示递减。应按采集顺序设置，不要通过翻转投影数据补偿。 |

重建 TOML runner 当前仍只生成均匀圆轨迹 vector，角度是唯一来源。前投 runner
另外支持本手册“前投任务”列出的螺旋和平面 builder。二者都还没有定义外部逐视图
`SConeProjGeomVec` 文件格式，因此不能表达非等间隔角度、摆动或任意轨迹，也不会
读取数据集附带的说明 TXT。需要这类几何时，应先扩展明确的 vector 文件格式，不能
静默退回圆轨迹。

## `[case.volume]` 重建体

| 参数 | 必需 | 默认值 | 含义与选择建议 |
|---|---:|---:|---|
| `nx`, `ny`, `nz` | 是 | - | X/Y/Z 方向体素数。总显存和计算量近似随三者乘积增长。 |
| `voxel_mm` | 是 | - | `[x,y,z]` 体素尺寸，单位 mm。通常先按目标分辨率和有效 FOV 计算，避免仅为得到大矩阵而过采样。 |
| `offset_mm` | 否 | `[0,0,0]` | 重建体中心相对等中心的 `[x,y,z]` 偏移，单位 mm。偏心对象应改这里，不要改探测器 offset 代替。 |

## `[case.input]` 输入

`mode = "phantom"`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `phantom` | `catphan` | `basic` 或 `catphan`。runner 使用配置几何和 `[algorithm].forward` 生成投影。 |

`mode = "projection-raw"`：

| 参数 | 必需 | 默认值 | 含义 |
|---|---:|---:|---|
| `projection` | 是 | - | float32 little-endian raw，布局 `[view][v][u]`。字节数必须为 `views*nv*nu*4`。 |
| `air` | 否 | 空 | 单帧 float32 空气场，布局 `[v][u]`。提供后输入被视为强度并转为线积分；不提供则假定 projection 已是线积分。 |
| `minimum_ratio` | 否 | `1e-6` | 空气校正时 `I/I0` 的下限，范围 `(0,1]`。过大会压低高衰减区域，过小会放大暗场噪声；一般保持 `1e-6`。 |

FDK raw 输入采用双 pinned-buffer 流式读取，只保留 `2*batch_views` 帧；迭代算法
需要反复访问全部测量，仍会把完整投影放入显存。

## `[case.filter]` FDK 滤波

该表只对 FDK 生效。

| 参数 | 默认值 | 可选值/选择建议 |
|---|---:|---|
| `kernel` | `ramlak` | `ramlak` 分辨率最高、噪声也最高；`hann`/`hamming` 更平滑；还支持 `none`、`shepp-logan`、`cosine`、`blackman`、`butterworth`、`kaiser`、`tukey`。 |
| `source` | `discrete-fft` | `discrete-fft` 使用离散 Ram-Lak 空域核 FFT，适合与当前 FDK 数值基线一致；`analytic` 直接构造频域权重。 |
| `cutoff` | `0.5` | 归一化截止频率，合法范围 `(0,0.5]`。降低会抑制高频噪声并损失空间分辨率。 |
| `gain` | `1.0` | 滤波整体增益。除非做过定量标定，否则保持 1。 |
| `order` | `2.0` | Butterworth 阶数；越高过渡越陡。 |
| `beta` | `8.6` | Kaiser beta；越大旁瓣越低、主瓣越宽。 |
| `tukey_alpha` | `0.5` | Tukey 窗锥形比例，范围通常为 `[0,1]`。 |

## `[case.algorithm]` 迭代算法

该表主要对 SIRT/SART/OSSART/CGLS 生效。内置模体的测量生成也会使用 `forward`。

| 参数 | 默认值 | 适用范围与选择建议 |
|---|---:|---|
| `iterations` | `1` | 外循环次数。先用 1-5 次检查数据和方向，再按残差/图像质量增加。 |
| `subsets` | `2` | 仅 OSSART 使用，范围 `1..views`。较大收敛初期快但更易振荡；常用 5-20，且最好能较均匀分配视图。 |
| `relaxation` | `0.2` | SIRT/SART/OSSART 初始松弛因子。发散或条纹振荡时降低；收敛过慢时谨慎提高。 |
| `relaxation_reduction` | `1.0` | 每个完整外循环后的松弛衰减系数。`1` 不衰减，常用 `0.95-1.0`。 |
| `epsilon` | `1e-6` | 归一化/除法保护阈值；CGLS 示例可用 `1e-8`。通常无需调节。 |
| `use_min`, `minimum` | `true`, `0` | 启用下界及其值。衰减系数重建一般保持非负约束。 |
| `use_max`, `maximum` | `false`, `1e30` | 可选上界。只有已知物理范围或要抑制异常值时启用。 |
| `forward` | `joseph` | `joseph` 或 `siddon`。Joseph 通常更平滑、速度较好；Siddon 是射线精确路径长度模型。 |
| `back` | `joseph-v3` | `joseph`、`joseph-v2`、`joseph-v3`、`siddon-ray`、`siddon-voxel`。应优先选择已通过对应 FP/BP 共轭与数值测试的组合。 |

代数法还支持嵌套的 `[case.algorithm.regularization]`：`type` 可取 `none` 或
`smoothed-tv`，并通过 `strength`、`inner_iterations`、`dimensionality = "2d"|"3d"`、
`epsilon`、`strength_reduction` 控制 TV。嵌套的 `[case.algorithm.convergence]` 支持
`relative_residual`、`relative_update`、`relative_improvement`、`minimum_iterations`、
`check_interval`、`patience`；三个浮点阈值均为 0 时关闭提前停止。

CGLS 的 `strategy` 可取 `robust-restart` 或 `astra-classic`。后者不支持上述收敛
阈值；配置加载时会直接拒绝这种组合。`fdk-ossart` 和 `fdk-cgls` 先按
`[case.filter]` 完成 FDK，再在同一 device volume 上继续迭代。

## `[case.tigre]` TIGRE 梯度族

仅供 `pipeline = "tigre-gradient-local"` 使用。完整示例见
`tigre-gradient-local.toml`。这是仓库内的本地复现实现，不链接官方 TIGRE，
当前验证范围是本仓库的冒烟/数值测试，不能视为官方逐项一致。
`method` 支持 `sart`、`os-sart`、`sirt`、`asd-pocs`、`os-asd-pocs`、
`b-asd-pocs-beta`、`pcsd`、`os-pcsd`、`aw-pcsd`、`os-aw-pcsd`、
`aw-asd-pocs`、`os-aw-asd-pocs`。`block_size` 是每个块的视图数，不是子集数；
`initialization` 可取 `zero` 或 `fdk`；`relaxation_mode` 可取 `scalar` 或
`nesterov`。POCS/PCSD 的 TV、beta、L2 条件分别由该表中的同名字段控制，不能
等同于普通 OSSART 的后处理 TV。

## `[case.pwls]` PWLS

仅供 `pipeline = "pwls"` 使用，对应通用入口
`Iter::PwlsReconstructor`。`regularizer` 可取
`none`、`quadratic`、`huber`；
`regularization` 是正则强度，`huber_delta` 是二次区与线性区的转折点，单位与体素
值相同。`subsets = 1` 是全批量并行更新，大于 1 时按视图序号构造交错子集。
`lower_bound`/`upper_bound` 是硬约束。当前实现采用同步全体素更新，不是串行逐体素
ICD。用于平板螺旋扫描时仍是同一个通用 PWLS，只有显式传入的逐视图 geometry
改为螺旋轨迹。

## 当前未纳入 TOML runner 的功能

以下功能在仓库中没有对应的通用配置 pipeline，不能通过本目录 TOML 直接运行：

- 单独 BP 任务、平面 CT 专用重建、外部 `SConeProjGeomVec` 文件输入；
- 非等间隔角度和任意轨迹的 TOML 表达；当前前投与重建配置都只生成均匀角度；
- 基于 `CylFpBp` 的完整迭代重建；TOML 已支持匹配 FP/BP，但尚未把它接入
  PWLS/CGLS 等迭代框架；
- 平板螺旋迭代重建尚未接入 TOML runner；wFBP（含真实投影 FFS 参数）已接入；
- `BP_FDK_matched` 单独 pipeline 和 XFDK。前者是可供迭代算子选择的反投 kernel，
  不是完整重建器；XFDK 当前未实现；
- 官方 TIGRE、官方 PWLS 或其他第三方实现的直接调用。

这些功能已有的 C++ 测试仍保留在 `Yktest`，但不应根据测试文件存在就推断它们
已经进入配置化公共接口。

## `[case.output]` 输出

| 参数 | 必需 | 默认值 | 含义 |
|---|---:|---:|---|
| `volume` | 是 | - | float32 raw 重建体，布局 `[z][y][x]`。 |
| `preview` | 否 | 空 | 中心 Z 层 BMP 快速预览；空值表示不生成。它只用于目视检查，定量分析应读取 raw。 |

每项任务都会检查输出是否全部为有限值且包含非零动态范围，并打印投影/体数据统计
和总耗时。FP/BP 共轭性、滤波频谱、异步 fence、在线分包一致性等数值契约测试
仍保留为独立 C++ 测试，因为它们不是改变配置参数即可充分覆盖的行为。
