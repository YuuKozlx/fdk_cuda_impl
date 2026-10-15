# BoneBhcStage：统一骨硬化策略

`BoneBhcStage` 是完整投影集级 Stage，内部持有一个 `IBoneBhcAlgorithm`。
没有枚举分发执行器、注册中心或工厂体系；具体对象在组装流程时直接注入。
EBHC 保持独立，不进入这个框架。

```cpp
using namespace yk::spectral::bone_bhc;
auto algorithm = std::make_unique<PhysicalBoneBhc>(
    config, Domain::WaterCorrectedLog, fittedWaterBhc);
auto result = std::make_shared<Result>();
BoneBhcStage stage(std::move(algorithm), result);
// Context 提供底层重建回调，避免回调完整 pipeline 造成递归校正。
context.options.max_iterations = 1;
stage.process(dataset, context);
// 或替换为：std::make_unique<TwoMaterialLutBhc>(config, calibration)
```

## 职责与契约

Stage 管理固定基础投影、初始重建、逐轮校正和重建反馈、进度、检查点、资源清理。
具体策略负责分割、材料路径和校正算法。两种策略共用分割辅助函数；单轮 BoneBHC
和迭代 BHC 使用同一个 PhysicalBoneBhc，只有执行轮数不同。

内部算法接口提供 id、inputDomain、errorMetric、supportsIteration、prepare、correct、reset。
Stage 对外沿用公共 IStage 契约，元信息 ID 是 `preprocess.bone_bhc`。
输入 Domain 与所选算法不匹配时拒绝执行。数据域转换由调用者/runtime 管理。
每轮都校正固定基础投影，不累加完整补偿到上一轮投影。

| 策略／输入 | 公式 | 误差判据 |
|---|---|---|
| PhysicalBoneBhc／水 BHC 前 | p + mono - poly | 归一化强度 MSE |
| PhysicalBoneBhc／水 BHC 后 | W(p) + mono - W(poly) | 水校正后的 log MSE |
| TwoMaterialLutBhc／水 BHC 前 | LUT(p, 骨等效厚度) | 无物理模型误差 |

水 BHC 后输入禁止再执行初始水校正。前置输入可以先做水校正以生成初始重建；
后续补偿仍从固定前置输入开始。物理策略的两种模式不宣称数学等价。
双材料策略直接复用现有 CUDA TwoMaterialBhcStage 的插值及边界处理，不重写查表公式。
骨厚度超标定范围会报错。迭代查表时重新分割并计算骨路径，不重复校正上轮结果。

默认一轮；物理模式多轮时从第4轮按最近两轮/前两轮误差和的比值停止，另有轮数上限。
后置物理模式采用同形式停止规则，但其 log 误差不是论文原来的强度误差。
LUT 无对应模型误差，记录 null，按显式 max_iterations 执行，不伪造收敛判据。
策略异常时也调用 reset；阶段末尾才发布最终输出路径。

## 原生运行入口

```text
multispectrum_sim.exe --bone-bhc job.json
```

物理法，推荐现有水校正链条使用的单轮配置：

```json
{
  "method": "physical",
  "config": "baseline.toml",
  "input_projection": "scatter-and-water-corrected.raw",
  "input_domain": "scatter_and_water_corrected_log",
  "output_directory": "bone-bhc-result",
  "soft_material": {"preset": "compound.water", "density_g_cm3": 1.03},
  "bone_material": {"preset": "tissue.ncat_skull", "density_g_cm3": 1.61},
  "air_threshold_hu": -500,
  "bone_threshold_hu": 300,
  "max_iterations": 1,
  "stop_ratio": 0.78
}
```

若用前置投影，input_domain 改为 `scatter_corrected_log_before_water_bhc`，
同时换成相应输入文件。后置模式的水多项式及截断参数必须与输入实际采用的校正一致。

双材料法用 `method: "two_material_lut"`、`calibration: "calibration.json"`、
可选 `bone_thickness_scale: 1.0`，输入仅接受前置 log，不需要 soft_material/bone_material。
表的第二材料基准必须与提供的等效骨厚度匹配；比例由用户明确指定，不自动猜测。
当前自动 HU 分割循环要求表的参考能量为正且与配置一致；未指定参考能量的经验
PMMA/Al 切线表不能直接投入这个自动循环，仍可用独立逐帧查表接口和明确的厚度提供者。

路径相对 job。输出目录必须不存在、父目录必须存在。CLI 用 TOML 几何、参考能量、
解析 FDK/WFBP 参数，重建回调关闭所有前处理，避免递归和重复校正。旧试验入口
`--iterative-bhc` 已改为 `--bone-bhc`，不保留临时兼容入口；旧 job 需添加 method。

## 数据与资源

分割采用 -500/300 HU 默认阈值，可修改；这不是论文指定参数。物理模型使用真空背景、
显式两种材料的固定密度。CUDA 中心射线 Siddon、单能与多能共用路径，没有随机采样。
能谱采用理想能量积分型权重，当前拒绝配置启用分层探测器响应的物理模型。

正投/查表使用 CUDA，分割、误差统计、文件 I/O 在 CPU。正投大缓冲按256 MiB批量限制，
查表还需少量标定表存储；各轮模型缓冲在重建前归还显存池。保持现有重建入口的池修剪。
保留 initial.raw、iteration-N 下的投影、体积、model.json，以及 run.json 中的策略、
误差类型和耗时。当前仍为文件检查点方式，不宣称跨重建设备常驻。

## 已有数值依据

改造前的全分辨率物理前置模式在512×512×400头部、2500张螺旋投影上：
水 BHC 骨间低频暗误差 RMS 8.87 HU，第1轮2.32 HU，第4轮2.32 HU；
骨内部 RMSE 51.31→25.26→25.47 HU。主要改善来自第1轮。
原始真值与单能参考仅用于全400层评价，不参与校正。

统一策略框架后，同一头部物理前置模式第1、2轮的投影和体积与旧循环逐值一致；
物理后置模式全流程跑通。LUT 使用相同初始分割和标定表时，与原逐帧入口的
校正投影逐值一致。历史 2dcalbh-full 使用另一份初始重建的分割，不能直接把
它与新框架自动重建后产生的分割混用来判断查表数值回归。

实现文件：BoneBhcStage.cpp 为公共调度，BoneBhcAlgorithms.cu 为内部算法策略，
BoneBhcJob.cpp 只负责命令行配置和组装。调用内部策略无需 JSON 或前端枚举。

频率分离加速尚未实现，不称为商业 Siemens iBHC 的复现。
参考：https://doi.org/10.1117/12.2007808 。模型误差停止不等于质量验收。
