# 多能谱 CBCT 仿真示例

本例通过 `YKCBCT` 公共 DLL 完成材料路径正投影。TOML 指定标签模体、材料（标签、化学式、密度）、能谱、XCOM 数据目录和四类几何名称：`flat_cbct`、`flat_helical`、`cyl_cbct`、`cyl_helical`。几何 offset 等系统参数由公共 geometry builder 生成。

当前已实现标签/材料到 XCOM 质量衰减系数，再到多能量透射积分和逐视图 CBCT 投影输出的闭环。投影按照 `view、v、u` 顺序写入 float32 二进制文件；`runToFile()` 按视图流式写盘，避免完整投影长期驻留内存。路径长度单位是 cm，质量衰减系数单位是 cm²/g；每个材料的线性系数为 `density * mu_over_rho`。

配置中的 `path_cache_file` 指定材料路径积分缓存文件。程序采用两阶段流式流程：第一阶段逐视图读取标签体并写入缓存，第二阶段逐视图读取缓存、执行多能谱积分并写出最终投影，因此不会同时在内存中保存完整投影或全部视图的材料路径。

缓存布局为：固定头 `PathCacheHeader`，随后按视图排列；每个视图依次存储每种材料的一张 `detector_v × detector_u` `float32` 路径积分图。头部包含 magic、版本、视图数、探测器尺寸和材料数，读取时会严格校验这些字段。缓存是中间产物，可以在后续实现 GPU 路径积分时直接复用。

当前已实现的物理过程是材料路径积分、能谱指数衰减和常数探测器效率。散射、电子/光学串扰、余晖和电子噪声保留配置字段与扩展位置，但尚未宣称已经生效。

在顶层工程开启 `BUILD_MULTISPECTRUM_SIM=ON` 并同时设置 CMake 选项 `MULTISPECTRUM_USE_LIBRARY_FP=ON` 时，配置中的 `use_library_fp = true` 会链接 `YKCBCT` 的公开导入库，并在运行时调用 `YKCBCT.dll` 的 `ReconstructionSessionFactory` 和 `IReconstructionSession`。适配器为每种材料构造二值标签体，通过公共 `ForwardProjection` Session 生成路径投影，再写入同样的路径缓存；它不访问 `src` 内部 kernel、geometry 或内存类。Flat/Cyl、Circular/Helical 四种组合均使用同一套 `SSystemSpec` 构造。适配器使用 Host buffer，CUDA 工作区由 DLL 管理。独立构建或未开启该 CMake 选项时，启用配置会明确报错，不会静默退回 CPU。默认关闭该选项，保证常规示例构建稳定。

探测器效率、散射、光学串扰、余晖、电子噪声均已保留配置字段。当前仅效率常数参与积分，其余字段明确为关闭占位，后续通过可组合效果接口实现。

构建：`cmake -S example/multispectrum_sim -B out/multispectrum_sim`，然后 `cmake --build out/multispectrum_sim --config Release`。相对路径以 TOML 所在目录为基准。能谱 CSV 每个有效数据行由 `energy_keV,relative_photons` 构成，读取器接受任意正数个采样点，不固定为 150。

示例文件按职责分开保存：

- `configs/`：仅 TOML 运行配置；
- `spectra/`：原始能谱和转换后的 CSV；
- `inputs/`：外部标签体；
- `outputs/path-cache/`：材料路径积分缓存；
- `outputs/projections/`：最终 `-log(I/I0)` 投影；
- `outputs/reconstructions/`：float32 重建体；
- `outputs/images/`：用于观察的中心切片图。

普通调用会从 TOML 的 `simulation.label_volume` 读取外部 `uint8` 标签体，布局为
`[z][y][x]`，文件元素数必须严格等于 `volume_x * volume_y * volume_z`。内部水模只是一种
测试输入，也可先导出后再按外部体积流程运行：

```powershell
multispectrum_sim --export-water-cylinder configs/water-cylinder-120kv-fullfov.toml `
  inputs/water-cylinder-512x512x256-u8.raw
multispectrum_sim configs/water-cylinder-120kv-fullfov.toml
```

## CUDA 多能谱积分

顶层 DLL 构建支持 `simulation.use_cuda_spectral = true`。CUDA 积分器让任意长度的
能谱权重和各材料 `density * mu_over_rho` 表常驻显存，每次只上传当前视图的
`[material][v][u]` 路径并回传一张投影，不缓存完整扫描。配置关闭该开关时仍使用 CPU
实现，便于回归比较；独立无 CUDA 构建若开启该选项会明确报错。

`water-cylinder-120kv-fullfov.toml` 使用从原始 `.spc` 转换得到的 150 点 120 kV 谱。
150 不是程序限制，CSV 中任意正数个有效 `(energy_keV, relative_photons)` 数据行均可读取。

XCOM 核心及 `data/MDATX3.*` 来自 `nist-xcom-portable`，保留 GPL-3.0-or-later 和第三方声明。

## 典型水模测试

仓库只保留 `configs/water-cylinder-120kv-fullfov.toml` 这一份完整示例。它使用
`512 x 512 x 256` 标签体、`1024 x 1024` 平板探测器、360 个视图和 120 kV 能谱，
随后通过 DLL 的 Flat Circular FDK 重建 `512 x 512 x 256` 体积。投影、路径缓存、
重建体和 BMP 均写入 `outputs/`，该目录不纳入版本控制。

### 几何偏移参数

`[geometry_config]` 中的偏移均以 mm 表示：`phantom_offset_x/y/z_mm` 定义标签模体
中心，`reconstruction_offset_x/y/z_mm` 定义重建网格中心，二者分别只作用于正投和重建。
`source_offset_x/y/z_mm` 定义焦点相对标称轨迹的扫描架局部偏移，该偏移随旋转架一起旋转。探测器支持局部
`U/N/V` 三方向偏移（`offset_u_mm`、`offset_n_mm`、`offset_v_mm`）；探测器姿态偏转
暂不在仿真配置中开放，因为不同投影/重建模型对偏转的支持范围不同。`views_per_turn`
和 `rotation_direction` 控制规则角度采样；这些值最终交给公共 geometry builder，正投和
重建不会各自解释一份几何。

多能谱 TOML 保留 `reconstruction.pipeline` 字段。材料路径积分固定调用公共
`ForwardProjection`；当前示例的 DLL 重建实现仍只执行 `pipeline = "fdk"`。其他管线的
配置字段可以继续保留给后续接入，但未接入时会明确报错，不会静默改用 FDK。

offset 的处理按算法能力执行：Flat FDK、FP 和迭代算子保留配置中的 offset；XFDK、
C-FDK、Cyl Analytic FDK、wFBP 等要求规范采集几何的管线，在 DLL Session 初始化时将
后端几何副本中不支持的源端/探测器 offset 分量强制置为 0，并通过日志输出中文提示。
wFBP 保留其支持的 U/V 主点 offset，仅清除源端和 N 分量。调用方配置、正投影几何和
体积中心 offset 均不改变。探测器 tilt 仍未在 TOML 中开放。

标签体不随仓库提交，可先由同一个可执行文件生成，再按外部体积输入流程运行：

```powershell
out/top-multispectrum-dll-vs/example/multispectrum_sim/Release/multispectrum_sim.exe `
  --export-water-cylinder example/multispectrum_sim/configs/water-cylinder-120kv-fullfov.toml `
  example/multispectrum_sim/inputs/water-cylinder-512x512x256-u8.raw

out/top-multispectrum-dll-vs/example/multispectrum_sim/Release/multispectrum_sim.exe `
  example/multispectrum_sim/configs/water-cylinder-120kv-fullfov.toml
```

也可用 `--water-cylinder` 直接在内存中生成纯水圆柱。该模式以横向短边的 40% 作为
圆柱半径，并使用 `name = "water"` 或 `formula = "H2O"` 对应的材料标签。

示例输出包括 `[view][v][u]` 排列的 float32 投影、材料路径缓存、`[z][y][x]`
排列的 float32 FDK 重建体，以及轴位、冠状位和矢状位中心切片 BMP。BMP 仅用于观察，
数值分析应读取重建体 raw。
