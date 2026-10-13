# 测试与验证

本篇负责测试入口、配置与证据。算法公开状态以 [支持矩阵](algorithms.md#公共支持矩阵) 为准；内部测试成功不能绕过 DLL 放行检查。构建方法见 [完整本体测试构建](build-and-integration.md#完整本体测试构建)。

## 三类测试

| 目标 | 职责 | 事实来源 |
| --- | --- | --- |
| `YKCBCT_unit_test` | 不依赖算法 device-link 的轻量组件；当前 Logger | [unit_test](../unit_test/CMakeLists.txt) |
| `ykcbct_manual_tests` | 内部 CUDA、几何、算法、数值回归和大规模测试 | [注册表](../Yktest/testmain.cpp)、[构建](../Yktest/CMakeLists.txt) |
| `dlltest` / `dlltest_shepp_logan` | 仅从公共头文件/导出目标验证 API 与 FP→FDK | [dlltest 构建](../dlltest/CMakeLists.txt)、[定量示例](../dlltest/test_shepp_logan.cpp) |

Yktest 不是所有测试都自动纳入 CTest。unit_test 通过 gtest_discover_tests 注册；DLL Shepp-Logan 仅在 BUILD_TESTING 为真时 add_test，且需工程启用测试发现。DLL 程序直接执行更明确，不应把一个空的 CTest 报告当作全库通过。

## 运行方式

下面采用 Ninja 的 `out/build/docs-check`，VS 生成器需在目标子目录增加 Release，并给 CTest 加 `-C Release`。

```powershell
ctest --test-dir out/build/docs-check --output-on-failure
out/build/docs-check/dlltest/dlltest.exe
out/build/docs-check/dlltest/dlltest_shepp_logan.exe
out/build/docs-check/Yktest/ykcbct_manual_tests.exe list
out/build/docs-check/Yktest/ykcbct_manual_tests.exe filter
out/build/docs-check/Yktest/ykcbct_manual_tests.exe fdk/batch-consistency
```

Yktest 接受完整测试名、分类名、list 或 all-local；CLI11 帮助用 `--help`。`--log-level info` 控制日志。unit_test 接受 `--yk-log-level off` 并转交 GoogleTest 参数，如 `--gtest_filter=LoggerTest.LevelFiltersLowerSeverity`。

`needs_real_data=true` 的测试只能按完整测试名显式运行，分类/all-local 会跳过它们。all-local 仍包括大模体与性能测试，可能耗时或消耗大量显存，不能把“local”理解为轻量。

真实数据路径和输出目录由具体测试/配置决定；不要在正式代码增加开发机绝对路径。数值产物保存到独立 output 或 out/test-artifacts，不应提交大 RAW。

## 按改动选择验证

测试名以实际 `list` 为准，以下入口均来自源码注册表；Cyl/Heli 部分在 YKCBCT_BUILD_HELICAL 开启时才注册。

| 改动 | 关键入口 |
| --- | --- |
| 公共 API / 缓冲区 / 版本 / 状态 | dlltest、dlltest_shepp_logan；核对公开参数实际消费 |
| 几何 builder / 刚体 / 体积位置 | framework/geometry-builder-four-modes、framework/rigid-geometry、framework/external-geometry、fp/volume-pose-api |
| 滤波 / 物理尺度 | filter/spatial-ramp、filter/discrete-ramlak-dc、fdk/depth-denominator |
| 普通 FDK / 分包 / 同步 | fdk/batch-consistency、fdk/synchronous-batch、fdk/calibrated-geometry、fdk/system-request |
| FP/BP / 纹理 / 资源 | framework/operator-matrix、framework/operator-release-fence、fp/siddon-uniform-center、fp/siddon-single-voxel、fpcyl/adjoint、fpcyl/fdk-adjoint |
| Cyl 解析 / 几何映射 | fpcyl/cfdk-reconstruction、fpcyl/fdk-backprojectors；同时测试公共 FDK 分派 |
| xFDK | fdk/xfdk-smoke、fdk/xfdk-streaming-consistency；大锥角入口作为专项扩展 |
| C-FDK | fdk/curve-filtered-piecewise-mapping、fdk/curve-filtered-reconstruction；large-cone/large-water 为专项 |
| 代数 / TV / 收敛 | recon/algebraic-ex、recon/ossart-tv、recon/convergence |
| CGLS / TIGRE | recon/cgls-unified、recon/tigre-gradient-family |
| Cyl PWLS / wFBP | heli/cyl-pwls-static-helical、fpcyl/wfbp-flat-vs-equiangular、fpcyl/wfbp-compare |
| DLT | dlt/adjoint、dlt/algebraic-methods |
| 非标准轨迹 | geometry/planar-fp-bp、recon/source-orbit-tomography、recon/nikon-tilted-pcb-reconstruction；公共 DLL 覆盖见 dlltest |
| 非标准内部流式 | recon/nonstandard-fdk-streaming；两个 SourceOrbit 映射模式与 NikonTilted 的 Host/Device、不规则包、尾包、容量/状态拒绝、取消/重启、累加与输入复用 |
| 柱面内部流式 | recon/cyl-fdk-streaming；R=SDD、R≠SDD、V 偏移、旧门面、Host/Device、单帧/不规则包、累加及取消/重启 |

框架 operator-matrix 验证内部组合，不代表任意模型配对公开可选。入口中的 cfdk/FDK 名称也不应代替实际 detector/pipeline 的判断。

## Yktest TOML

```powershell
out/build/docs-check/Yktest/ykcbct_manual_tests.exe `
  --config test-configs/reconstruction-static-flat.toml --case catphan-fdk
```

case 名先查看所用文件中的 `name`；不同文件不能共用一个假定名称。相对路径以 TOML 所在目录为基准，不取决于启动工作目录。

| 文件 | 任务和覆盖 |
| --- | --- |
| [reconstruction-static-flat.toml](../test-configs/reconstruction-static-flat.toml) | reconstruction；FDK/C-FDK/xFDK、代数/CGLS/TIGRE/PWLS、FDK 初值组合 |
| [reconstruction-static-cyl.toml](../test-configs/reconstruction-static-cyl.toml) | cylindrical-reconstruction；Cyl 解析与迭代 |
| [reconstruction-heli-flat.toml](../test-configs/reconstruction-heli-flat.toml) | helical-reconstruction；内部 wFBP 与 Flat 螺旋迭代 |
| [reconstruction-heli-cyl.toml](../test-configs/reconstruction-heli-cyl.toml) | cylindrical-reconstruction；Cyl 螺旋迭代 |
| [forward-projection.toml](../test-configs/forward-projection.toml) | forward-projection；四种规则几何 |

这是内部测试配置，不能直接传给公共 C++ Session，也不是 `example/multispectrum_sim` 的配置格式。内部 Flat 螺旋 wFBP/C-FDK 可运行不代表对应公共算法放行。

核心字段示例（必须放在完整 case 中，并配套具体任务和算法字段）：

```toml
[case.scan]
nu = 48
nv = 32
views = 64
du_mm = 1.0
dv_mm = 1.0
sid_mm = 80.0
sdd_mm = 160.0
scan_range_deg = 720.0

[case.volume]
nx = 32
ny = 32
nz = 24
voxel_mm = [1.0, 1.0, 1.0]

[case.geometry]
pitch_mm = 8.0
start_z_mm = -8.0
views_per_rotation = 32
curvature_radius_mm = 160.0

[case.input]
mode = "phantom"
phantom = "catphan"

[case.output]
volume = "output/reconstruction.raw"
preview = "output/reconstruction.bmp"
```

TOML scan_range_deg 使用度，SDK 使用弧度；TOML pitch_mm=0 的任务解释与 SDK 的显式 Circular/Helical 枚举也不能混用。任务解析源码位于 `Yktest/config/`，完整可执行格式以仓库配置为准。

Flat 重建和内部 wFBP 可用 `input.mode="projection-raw"`、`input.projection="data/projection.raw"` 读取 float32 连续线积分。当前 Cyl 配置化重建只接受内置 Catphan，与公共 Cyl Session 可使用外部 Buffer 是两个边界。

Flat Heli 的 SIRT/SART/OS-SART/CGLS/PWLS 使用各自算法表；Cyl 使用 algorithm 选择内部模型，PWLS 表有 projection_weight_scale/data_model 等测试层参数。它们未必一一对应公开 SReconstructionSpec 字段。

## 误差与验收

### 伴随性

记录 `<Ax,y>`、`<x,BPy>` 和：

```text
relative = abs(<Ax,y>-<x,BPy>) / max(abs(<Ax,y>),abs(<x,BPy>))
```

随机种子、几何、采样、纹理、硬件、输入分布均需相同。内积接近零时分母会放大误差，测试需处理退化输入。严格离散模型仍允许浮点 atomic 顺序误差；硬件 Linear 插值与软件权重不能按严格逐元素相等验收。

Point 纹理读回是否精确、两次原子累加差异、FP/BP 是否伴随是三项独立验证。FDK BP 不满足 Siddon/Joseph 的离散转置，不能套用同一阈值。

### 图像与尺度

NRMSE=`||x-truth||/||truth||`，MAE=`mean(abs(x-truth))`，余弦相似度只衡量方向，不证明材料绝对值正确。若使用最佳缩放 NRMSE、中央层 ROI 或排除轴向端部，必须同时说明，不能和全体积未缩放指标混比。

DLL Shepp-Logan 的固定 smoke 验收为有限输出、cosine>0.85、NRMSE<0.65、全量/47-view 分包最大绝对差<1e-5。它输出 comparison.bmp 与 metrics.txt，计时包含初始化/传输，不是纯 kernel 性能，也不是高精度图像质量承诺。

大锥角测试应同时看中心/端部材料均值、轴向 profile、不同方向切片与分辨率；不要只展示相关系数高的拼图。有限高水柱硬端面不能单独替代解析椭球的论文级均匀性复现。

### 性能

kernel 使用 CUDA event，说明重复次数、预热、尺寸、模型和是否包含纹理更新。完整 pipeline 另记录初始化、传输、FFT、显存峰值与总耗时。性能比较使用同一输入/几何/设备/编译选项；不同测试规模的计时不可拼成统一排行榜。

## 验证记录

首次文档重建以 2026-10-11 的工作树为依据，基础提交 `0480d29`。文档改动外还存在应用目录未提交修改；本体检查关闭多能谱应用，使用独立构建目录。以下表格记录内部流式改动前完成的检查，不把旧文档数字当作本次结果；后续流式变更的验证单独记录。

| 环境 | 本次配置 |
| --- | --- |
| 平台 | Windows，MSVC 19.44，CUDA Toolkit 12.6 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU，驱动 561.17 |
| 构建 | Ninja Release，CUDA 架构 86，Helical/Manual/Unit/DLL 开启，Multispectrum 关闭 |
| 目录 | `out/build/docs-check` |

| 检查 | 本次结果 |
| --- | --- |
| 本体库和全部三类测试目标构建 | 通过；未修改算法源码 |
| 安装 SDK → 独立外部工程 | 通过；直接提取 public-api 的完整 C++ 示例和接入文档 CMake 示例，使用 find_package 编译运行，球心输出 0.0196773（输入 0.02），输出均有限 |
| CTest | 3 个 Logger 测试全部通过；此配置没有自动注册 DLL 定量示例，因此 DLL 程序另行直接执行 |
| dlltest | 通过；覆盖公共组合策略、容量/执行校验、reset、姿态、非标准 FP/FDK、普通 FDK 分包、XFDK、Cyl FDK、TIGRE、WFBP |
| dlltest_shepp_logan | 通过；cosine=0.982495，NRMSE=0.221270，MAE=0.000256790，全量/分包最大绝对差=7.45058e-9 |
| Yktest 定向选择 | 执行 28 次选择（filter 分类含两个测试）；27 次退出 0，fpcyl/fdk-backprojectors 退出 1，详情见下文 |
| TOML | reconstruction-static-flat.toml 的 catphan-fdk case 通过 |
| 文档检查 | 六份详细文档、根入口及八份目录导航共 15 文件；94 个本地链接/锚点、36 个测试引用名称检查通过；git diff --check 通过 |

定向执行包括本篇表中几何、滤波、FDK 分包/同步/深度/校准、xFDK 基础与流式、C-FDK 映射与基础重建、Cyl 伴随与解析、代数/收敛/TV/CGLS/TIGRE、平面几何及 DLT 伴随入口。精确选择和退出码保存在本地 `out/build/docs-check/manual-validation.json`，每个选择的日志位于同目录；构建、外部示例、CTest 和 DLL 日志也保存在该目录。未运行 all-local 和大规模/真实数据全套，未列出的组合及论文级质量不视为本次已验证。

`fpcyl/fdk-backprojectors` 的普通 Cyl FDK BP 中心值 32 与期望一致，但离中心点相对误差 1.172e-3 超过源码 1e-4 阈值，导致整项失败。matched BP 的中心相对误差 5.595e-5、离中心相对误差 1.313e-7 在各自阈值内；日志两行共享同一累计 ok 状态，因此两行均打印 FAIL，不能据此把 matched 的数值也判为超阈值。该项测试源码未因文档整理改变，待排查解析权重与测试期望的差异。

`fpcyl/fdk-adjoint` 的退出 0 只证明执行完成，没有严格伴随误差阈值。本次 Joseph FP 配普通 Cyl FDK BP 的 relative=0.171618，配 FDK-pseudo-matched BP 的 relative=0.735591；后者在该测试中并未更接近离散转置。不要把 pseudo-matched 名称理解为严格离散伴随。独立 `fpcyl/adjoint` 中，Joseph 配对误差约 2.535e-3，Siddon RayDriven 配对约 3.716e-7，均通过该测试自己的阈值。

本次外部示例首次缺少 MSVC /utf-8 导致 SDK 中文注释被按本机代码页解释，补入接入文档的编译选项后成功；unit_test 首次发现测试时因 GTest 运行 DLL 不在 PATH 报 0xc0000135，加入依赖前缀的 bin 后构建与 CTest 成功。这些环境要求分别记入接入与运行说明，不通过改动 SDK 或测试源码规避。

### 非标准 FDK 内部流式验证（2026-10-11）

在同一环境中增加 SourceOrbit/NikonTilted 内部流式接口，干净重建主库、Yktest、unit_test 与 DLL 测试目标后执行 `recon/nonstandard-fdk-streaming`。新增测试使用 13 个视图、内部工作区 4 视图；输入包为 1/7/5 视图以及逐视图提交，故包含跨计算块、拆计算块、短尾包和大于内部工作区的输入包。

| 流式路径 | 相对完整 Device 输入的最大差 / 参考体积最大绝对值 |
| --- | --- |
| SourceOrbit InPlaneRotated | 1.944e-7 |
| SourceOrbit PerspectiveOrthogonal | 2.058e-7 |
| NikonTilted（含逐视图 du 变化） | 2.186e-7 |

三项均通过 1e-5 相对阈值，分别覆盖 Host/Device 输入、非零体积中心及非等体素。测试在完整输入基线完成后释放整套 Device 投影，流式 Device 包仅复用 7 视图缓冲，SourceOrbit Host 暂存检查为 4 视图；每次 Host 提交返回后立即改写包缓冲。输出还检验 clear_output=false 累加、容量不足/非法位置/空包/跳帧/重复/越界拒绝、不完整 complete、cancel 后重启及活动重建释放。

新增接口是同步消费逐包数据，本次不声明双缓冲传输计算重叠或性能加速。完整 geometry、map 和派生量仍在 prepare 时常驻；节省的是完整原始投影缓存。公共 Session 尚未导出这项分包能力。

相关回归全部通过：recon/source-orbit-tomography、recon/source-orbit-pcb、recon/nikon-tilted-pcb-reconstruction、geometry/planar-fp-bp、fdk/batch-consistency、fdk/synchronous-batch、fdk/xfdk-streaming-consistency，另直接运行 dlltest、dlltest_shepp_logan 通过，CTest 的 3 个 Logger 测试通过。DLL 定量 smoke 的数值与上表的首次验证保持一致。新增回归源码为 [test_nonstandard_streaming.cpp](../Yktest/test_nonstandard_streaming.cpp)；本次没有重跑上节已记录失败的独立 Cyl 解析 BP 专项，也没有把它标记为已修复。

本机旧 Ninja/MSVC 缓存的 showIncludes 前缀编码损坏，使已有 C++ 对象的头文件依赖记录为空。流式验证采用 clean-first 完整重建，避免旧/新内部类布局混用；没有把仅增量链接的结果作为验收。日志保存到 `out/build/docs-check/nonstandard-streaming-clean-build.log` 及 `streaming-regression-*`，汇总退出码为 `nonstandard-streaming-regression.json`。

### 柱面 FDK 内部流式验证（2026-10-11）

干净重建全部本体测试目标后，执行新增 [test_cyl_fdk_streaming.cpp](../Yktest/test_cyl_fdk_streaming.cpp) 注册的 recon/cyl-fdk-streaming。每项使用 13 个视图、4 视图工作区、Host/Device 的 1/7/5 视图包和逐帧复用的 Host 输入；完整输入参考完成后释放全量 Device 投影。

| 柱面配置 | 流式与完整输入的最大绝对差 / 参考体积最大绝对值 |
| --- | --- |
| R=SDD | 1.675e-7 |
| R<SDD，含 V 偏移 | 2.237e-7 |
| R>SDD，含 V 偏移 | 2.235e-7 |
| 旧 CylFdkPipeline 门面 | 2.237e-7 |

全部通过 1e-5 阈值，同时检验容量/位置/视图范围拒绝、活动流重复 begin 或完整 reconstruct 拒绝、提前/重复 complete、clear_output=false 累加、cancel/restart 和活动资源释放。该实现逐包独立映射并累加，不需要跨视图 halo 或历史投影缓存；新增测试不是图像质量或吞吐性能承诺。

相关回归均通过：fpcyl/cfdk-reconstruction 的定量图像验证、example/system-reconstruction-four-geometries、fdk/batch-consistency、fdk/synchronous-batch、fdk/calibrated-geometry、fdk/depth-denominator、xFDK 流式一致性、非标准流式与 SourceOrbit 冒烟、TIGRE 梯度族、dlltest、dlltest_shepp_logan 及 3 个 Logger CTest。构建日志为 `out/build/docs-check/cyl-streaming-build.log`，新测试日志为 `cyl-streaming-test.log`，相关选择及退出码保存在 `cyl-streaming-regression.json`；其余日志使用 `cyl-regression-*` 前缀。

本次同时核对并修正文档中的公共 Flat FDK 缓冲区描述：Session 当前仍只接受 Host projection，Device 批次新增于内部 Flat FDK，供柱面映射直接衔接；公共 Cyl Session 仍要求完整序列。此前独立 Cyl 解析 BP 的离中心回归失败未因本次改动而修复。

## 待验证与实现限制

这些事项来自代码接线审计或旧实验的未完成条件；文档整理不修改算法：

- Cyl SART 边界放行与实际后端接线不一致；需要接通并补公共 DLL 回归。
- 内部 fpcyl/fdk-backprojectors 当前有离中心解析值回归失败；FDK-pseudo-matched 的名称也不能代替数值伴随结论，详见本次验证记录。
- Flat CGLS 的公开停止/策略字段、TIGRE block_size、PWLS 逐射线权重尚未按声明完整消费。需决定接通或收缩契约，并增加参数生效测试。
- XFDK 内部 pinned-host 流式能力未向公共 Session 导出；公共完整 Device 输入与内部 streaming 分别验证。
- C-FDK 论文级复现仍缺解析椭球、3×3 子射线和论文原始模体的同条件实验；重排插值强度/分辨率损失、大尺寸性能及中央条纹需量化。
- Flat 螺旋 wFBP 与 CFDK 公共接口未放行；内部测试入口的存在不改变此状态。
- 校准 FDK 的逐视图 du 变化仍使用 chunk 共享频谱，需要说明近似误差。
- 跨平台、关闭 Helical、其他 GPU 架构及任意非标准采集质量不由本次单机验证覆盖。

旧验证曾报告普通 FDK/ASTRA 高相关、C-FDK 水模端部改善、Flat/Cyl kernel 耗时和迭代提前停止收益，但缺少与本次工作树对应的完整环境/产物。本次不继承这些数字为当前基线；需要时从 Git 历史取实验条件后重跑。ASTRA 对比工具保留在 [compare_astra_fdk.py](../scripts/compare_astra_fdk.py)，版本和 geometry 必须写入新结果。

## 文档整理范围与维护检查

旧接口指南和内部使用示例分别合并到 public-api 与 architecture；FDK/Ram-Lak/ART/TIGRE/CFDK/xFDK/正则化内容合并到 algorithms；几何样例合并到 geometry；TOML、模块数值与状态记录合并到本篇。源码和测试目录仅留导航链接到统一文档。长篇论文翻译、重复规范和开发过程通过 Git 历史保存。

`helical_ct_geometry_designer_v2.html` 和已有 SVG 作为参考材料/工具保留，不是第七份详细开发规范，也不作为当前代码行为的权威说明。独立应用、标定工具和第三方授权文档保留原位置。

提交前检查所有相对链接、引用文件、测试名、构建开关默认值和 SDK 示例；同一修改同步更新支持矩阵。验证证据记日期和代码版本，不能通过只修改“最后更新时间”宣称重新验证。
