# TOML 配置化重建与前投手册

## 运行方式

一个 TOML 文件可包含多个 `[[case]]`。不传 `--case` 时顺序执行全部任务：

```powershell
ykcbct_manual_tests --config test-configs/fdk.toml
ykcbct_manual_tests --config test-configs/ossart.toml --case catphan-ossart-tv
ykcbct_manual_tests --config test-configs/victre-breast-fdk.toml --case victre-breast-fdk-360
ykcbct_manual_tests --config test-configs/forward-circular.toml
ykcbct_manual_tests --config test-configs/forward-planar.toml --case planar-offset-source-joseph
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
| `parallel-pwls.toml` | 并行 PWLS |
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
| `projector.model` | 是 | 普通几何选 `joseph` 或 `siddon`；圆柱几何必须选 `cylindrical-joseph`。 |
| `projector.samples_per_voxel` | 圆柱默认 `1.0` | 圆柱射线采样密度，必须大于 0；提高会增加耗时。 |
| `input.mode` | 默认 `phantom` | `phantom` 或 `volume-raw`。 |
| `input.phantom` | 默认 `catphan` | 内置 `basic`、`catphan`、`arrow`。 |
| `input.volume` | volume-raw 必需 | float32 raw 路径，布局 `[z][y][x]`，尺寸必须严格匹配。 |
| `output.projection` | 是 | float32 raw 输出路径，布局 `[view][v][u]`。 |
| `output.preview` | 否 | 指定后输出单视图 BMP；仅用于目视检查。 |
| `output.preview_view` | 默认 `0` | BMP 对应视图序号，范围 `0..views-1`。 |

前投输入有两种：`mode = "phantom"` 支持 `basic`、`catphan`、`arrow`；
`mode = "volume-raw"` 时必须提供 `volume`，布局为 `[z][y][x]`，文件字节数
必须严格等于 `nx*ny*nz*4`。`[case.output]` 必须提供 `projection`，可选
`preview` 和 `preview_view` 输出指定视图的 BMP。所有相对路径仍以 TOML 所在目录
为基准。

当前前投配置只生成均匀角度序列。`views_per_rotation` 不生成角度，只校验
`scan_range_deg/views` 是否与每圈视图数一致；角度仍由 `views`、
`start_angle_deg`、`scan_range_deg`、`direction` 唯一生成。两者不一致时直接报错，
不会静默采用其中一份。

配置产生的 `test-configs/output/` 内容都是可复现测试产物，已由同目录 `.gitignore`
排除，不属于源代码。需要重新生成时直接运行对应 TOML；不要把 raw/BMP 提交到仓库。

## 顶层参数

| 参数 | 必需 | 默认值 | 含义与选择建议 |
|---|---:|---:|---|
| `name` | 是 | - | case 唯一名称，用于 `--case`。同一文件不能重名。 |
| `pipeline` | 是 | - | 支持 `fdk`、`sirt`、`sart`、`ossart`、`cgls`、`tigre-gradient-local`、`parallel-pwls`、`fdk-ossart`、`fdk-cgls`。完整圆轨迹快速重建优先 FDK；数据不完整、要加约束时选择迭代算法。 |
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

## `[case.parallel_pwls]` 并行 PWLS

仅供 `pipeline = "parallel-pwls"` 使用。对应实现是
`Iter::ParallelPwlsReconstructor`，不是通用 PWLS 入口。`regularizer` 可取
`none`、`quadratic`、`huber`；
`regularization` 是正则强度，`huber_delta` 是二次区与线性区的转折点，单位与体素
值相同。`subsets = 1` 是全批量并行更新，大于 1 时按视图序号构造交错子集。
`lower_bound`/`upper_bound` 是硬约束。当前实现是同步并行 PWLS，不是串行逐体素
PWLS，也不是 ICD；螺旋 `IcdReconstructor` 目前只是这个并行实现的兼容包装。

## 当前未纳入 TOML runner 的功能

以下功能在仓库中没有对应的通用配置 pipeline，不能通过本目录 TOML 直接运行：

- 单独 BP 任务、平面 CT 专用重建、外部 `SConeProjGeomVec` 文件输入；
- 非等间隔角度和任意轨迹的 TOML 表达；当前前投与重建配置都只生成均匀角度；
- `CylFpBp` 圆柱探测器重建；TOML 现只接入实验性圆柱前投，不代表已有完整重建；
- 螺旋 wFBP/FFS、螺旋 ICD；wFBP 是实验性 FreeCT 风格实现，ICD 名称是并行
  PWLS 兼容包装；
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
