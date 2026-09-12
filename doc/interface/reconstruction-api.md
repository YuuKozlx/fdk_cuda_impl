# 重建 DLL 新接口

`YkReconstructionApi.hpp` 是新的 DLL 入口模型。它不延续旧
`SessionDesc` 将扫描、体积、逐视图 geometry、GPU 和算法细节放入一个结构体的
做法。

## 公开对象

调用方只需处理三个对象：

```cpp
SSystemSpec system;              // 系统、体积、算法和设备
SExecutionRequest execution;     // 一次投影/重建的数据缓冲与分包范围
IReconstructionSession* session; // 生命周期
```

`SSystemSpec::geometry` 是唯一的 `SSystemConfig`。调用方通过
`detector = Flat/Cylindrical` 与 `trajectory = Circular/Helical` 组合四种系统，
再填写对应的 circular/helical 轨迹和 flat/cylindrical 探测器参数。扫描轨迹、探测器
和体积仍聚合在同一个系统对象中；逐视图 geometry、角度数组和算法预计算全部由内部
builder 生成。角度不会再同时出现在初始化对象和执行请求中。

## 边界

新接口不公开 CUDA 纹理、kernel、workspace 或 launch 配置。投影布局固定为
`[view][v][u]`，体积布局固定为 `[z][y][x]`；内存位置由 `Buffer::location` 指定，
连续 float 容量由 `Buffer::element_count` 指定。执行请求当前为 API v2，容量不足会在
进入 CUDA 后端之前返回 `InvalidBuffer`。
`struct_size` 和 `api_version` 用于未来 ABI 演进。

`initialize()` 或 `execute()` 返回 `false` 后，通过 `lastError()` 和
`lastErrorMessage()` 读取稳定错误分类和中文诊断。配置/几何不合法、算法组合未接入、
缓冲区错误、流式状态错误和 CUDA 后端失败互不混淆。下一次调用会先清除旧错误。

在线分包通过 `view_offset/view_count` 表示，系统 geometry 和算法配置在初始化后不可
变。当前只有 Flat Circular FDK 支持分包，且 `view_offset` 必须连续递增，只有首包可
设置 `clear_output=true`。其余算法必须从 offset 0 一次提交完整投影，接口不会接受后
再让后端忽略 offset。`execute()` 当前是同步契约：返回后本次输入缓冲可以复用，输出
可以读取。库没有对应后端的组合会失败，不会降级为另一种几何或算法。

## DLL 放行范围

公开枚举描述库中的全部算法，但 DLL 只放行已经完成数值验证的组合。暂未放行的组合
仍保留对应内部调用代码，`initialize()` 返回 `AlgorithmUnderTest`，错误信息为“该算法
或几何组合仍在测试中”，不会降级到另一种几何或算法。

| Pipeline | Flat Circular | Flat Helical | Cyl Circular | Cyl Helical |
| :--- | :---: | :---: | :---: | :---: |
| ForwardProjection | 放行 | 放行 | 放行 | 放行 |
| FDK | 放行 | 测试中 | 测试中 | 测试中 |
| XFDK | 放行 | 测试中 | 测试中 | 测试中 |
| SIRT / OSSART / CGLS / PWLS | 放行 | 放行 | 放行 | 放行 |
| TigreGradient | 放行 | 放行 | 测试中 | 测试中 |
| CylAnalyticFDK | 测试中 | 测试中 | 放行 | 测试中 |
| WFBP | 测试中 | 测试中 | 测试中 | 放行 |
| CFDK | 测试中 | 测试中 | 测试中 | 测试中 |

`WFBP` 还要求构建时启用 `YKCBCT_BUILD_HELICAL=ON`。关闭该选项的 DLL 会返回
`UnsupportedCombination` 并提示当前构建缺少 wFBP 后端，这与算法仍在测试中的状态
分开报告。

## 已验证的 DLL 执行回归

`dlltest` 不仅检查 `initialize()` 返回值，还会为每条已放行路径提交非零投影，
同步 CUDA stream，拷回重建体并检查所有体素为有限值且总能量非零。当前已验证：

- Flat Circular FDK
- Flat Circular FDK 离线与在线分包结果一致性
- Flat Circular XFDK
- Cyl Circular analytic FDK
- Flat Circular TIGRE Gradient
- Cyl Helical wFBP

测试使用 `32 x 24` 探测器和小体积，解析管线使用 32 个等角视图，wFBP 使用 120
个螺旋视图。它们是接口级连通和数值写出回归，不替代大尺寸模体的精度评估。

## FDK 探测器尺寸约束

当前 Flat FDK 反投影为每个视图创建 `pitch2D` texture。行字节数
`Nu * sizeof(float)` 必须满足设备的 `texturePitchAlignment`，单视图字节数
`Nu * Nv * sizeof(float)` 必须满足 `textureAlignment`。初始化阶段会检查这两个条件；
不满足时返回 `InvalidGeometry/UnsupportedCombination`，不会让 CUDA runtime 的底层
检查宏直接终止 DLL 进程。选择常见的 `Nu=32`、`Nu=64` 等对齐尺寸可避免该限制。

## 螺旋 SID 定义

wFBP 从逐视图源点派生 `sid` 时只取源点到旋转轴的 XY 径向距离：

```text
sid = hypot(source.x, source.y)
```

不能使用包含源点 z 高度的三维模长；螺旋扫描中 z 随视图变化，但 SID 的物理定义
仍是源到旋转轴的径向距离。否则规则螺旋会被误判为 SID 漂移并在初始化阶段失败。

DLL 主路径直接将 `SSystemConfig + SReconstructionSpec + device` 交给内部
`IExecutionBackend`，`PreparedGeometry` 只构造一次。`Session` 不再构造或保存旧
`SessionDesc` 副本；旧结构仅保留给尚未迁移的内部测试/适配入口使用，不应继续扩展为
第二套公开配置。

## 探测器 offset 与偏转的能力边界

公共 geometry builder 和具体算法是两个层次。builder 能将探测器局部 U/N/V offset
以及绕 U/V/N 的 tilt 展开成逐视图向量，并不表示每条解析重建公式都能消费这些自由度。
公共接口不得因为 builder 构造成功就宣称算法支持，也不得在后端静默清零姿态参数；
若某算法只能消费规范采集参数，必须在进入后端前明确记录归零提示。

| 算法族 | detector offset | detector tilt | 当前接口语义 |
| :--- | :---: | :---: | :--- |
| Flat Joseph/Siddon FP、BP | 支持 | 支持 | 使用完整逐视图 vector geometry |
| Flat SIRT/OSSART/CGLS/PWLS/TIGRE | 支持 | 支持 | 能力随所选 Flat FP/BP 算子 |
| Flat FDK | 支持 | 支持 | 使用完整 geometry；非理想轨迹属于近似 FDK，并输出诊断 |
| XFDK、C-FDK | 不支持 | 不支持 | 论文实现只接受理想平面圆轨迹 |
| Cyl Joseph/Siddon FP、BP 与迭代 | 支持 | 暂未放行 | offset 使用柱面逐视图 geometry；tilt 尚缺完整回归 |
| Cyl Analytic FDK | 不支持 | 不支持 | 只接受规范同轴圆柱；非标准曲率先 map 到 `R=SDD` |
| wFBP | 支持 U/V；N 和源端不支持 | 不支持 | 公共 DLL 入口仅归零不支持的分量 |

多能谱仿真 TOML 公开 `reconstruction.pipeline`，但探测器 tilt 仍封闭，只开放 U/N/V
offset。材料路径投影固定使用公共 FP。对于明确要求理想 offset 的 XFDK、C-FDK、
Cyl Analytic FDK 和 wFBP，公共 Session 不直接拒绝；进入后端前会复制系统几何、将
不支持的 offset 分量置零并输出提示，调用方原配置不会被修改。wFBP 保留其支持的
U/V 主点偏移，仅归零源端和 N 分量。体积中心 offset 始终保留。Flat FDK、FP 和迭代
算子则使用原始 offset。尚未接入示例的 pipeline 会返回明确错误，不会隐式降级成其他算法。
