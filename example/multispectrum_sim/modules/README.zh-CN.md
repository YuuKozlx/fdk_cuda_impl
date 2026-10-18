# 模块与构建目标

这里集中管理由项目维护的组件，主工程按需接入。模块是职责边界，不等于静态库；每个模块的构建形式按实际代码决定，不按是否具有第三方来源或是否新增目录来决定。

| 目录 | 构建目标 | 职责 |
| --- | --- | --- |
| `spectrum` | `spek_spectrum::core` | 本地能谱计算、滤过、内嵌谱表 |
| `attenuation` | `nist_xcom_core` | XCOM 衰减核心、项目 C API 与内嵌数据 |
| `scalar_projection` | `multispectrum_scalar_fp` | 浮点体积前投 |
| `visualization` | `ykcbct_visualization` | 可选 VTK 显示；启用 ImGui 时同时构建界面桥接静态库 |

这些组件具有明确的数值 API 或可选显示边界。能谱和材料衰减包含 `.cpp` 核心及内嵌数据表，可独立调用与验证，适合保留静态库。

Stage 接口、显存池、图像处理和自适应散射属于应用内部设施。Stage 直接保留在 `include/processing/stage/`，无需单独构建目标；其余实现位于 `src/processing/device/`、`image/`、`scatter/`。`cmake/ProcessingSources.cmake` 用三个显式函数将源码加入使用它们的程序或内部目标，不为它们建立独立静态库。自适应散射仍由实验开关控制。

业务流程、客户端装配、处理策略和示例按原来的目标直接编译，仍放在 `src/` 或 `app/`，不一律拆库。是否使用 `STATIC`、`SHARED` 或 `INTERFACE` 是构建决策，应与目录归属分别考虑。

应用公共接口集中保留在 `multispectrum_sim/include/` 的五个分区；数值组件 `spectrum`、`attenuation` 保留各自原生 API 布局。具有外部来源的实现保留来源说明及许可证，项目集成说明与组件一起保存。目录迁移不重新生成数据表。

这些组件是多能前投应用的组成部分，底层重建 DLL 不反向依赖它们。`spectrum`、`visualization` 与 `app/plot` 保留独立配置入口；其他组件由应用或独立 Stage 测试工程提供 CUDA、公共接口及依赖后加入。

新增目录应对应实际库目标与明确职责。业务装配继续放在 `src/`，可执行程序放在 `app/`，接口示例放在接口所属目录，不因新增单个辅助类就建立一个模块。
