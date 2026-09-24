# 重建接口

第三方包含 `YKCBCT/interface/YkReconstructionApi.hpp`，使用
`SSystemSpec`、`SExecutionRequest`、`IReconstructionSession`。
本次整理直接替换配置，不新增接口版本或旧参数兼容入口；需要使用匹配的头文件重新编译调用方。

## 参数目录

```text
include/YKCBCT/
  geometry/                         物理系统、探测器、轨迹、体素网格
  algorithms/
    analytic/YkFdkParams.hpp         FDK 滤波和 Parker 参数
    analytic/YkWfbpParams.hpp        wFBP 冗余权重和滤波参数，无 FFS
    iterative/YkIterationParams.hpp 通用迭代次数、松弛系数、子集
    iterative/YkCglsParams.hpp       CGLS 停止条件
    iterative/YkPwlsParams.hpp       PWLS 数据约束和正则项
    iterative/YkTigreParams.hpp      TIGRE 方法和 TV 参数
    projection/YkProjectionParams.hpp  Joseph/Siddon 模型
  interface/
    YkReconstructionTypes.hpp        公共算法枚举、缓冲类型和参数头聚合
    YkSystemReconstruction.hpp       算法配置聚合
    YkReconstructionApi.hpp          Session 入口
```

每个算法头可以独立包含，不依赖 CUDA。配置通过 `reconstruction.fdk`、
`wfbp`、`cgls`、`pwls`、`tigre` 区分；公共迭代控制放在 `iterative`。
Parker 位于 `reconstruction.fdk.parker`，仅用于当前支持的 Flat 圆轨迹 FDK。
选择 `pipeline` 决定消费哪组参数，不要求填其它算法的参数。
完整系统几何接口仍依赖 CUDA 向量类型，本次未转换为 C ABI。

## 执行路径

- Flat/Cyl 圆轨迹统一选择 FDK，柱面到平板重采样由内部完成。
- Cyl FDK 支持探测器 V 方向偏移，重采样后由平板主点保留；源端偏移、探测器 U/N 偏移及三个轴的 tilt 均拒绝。
- WFBP 当前公共入口仅放行 Cyl Helical，固定无 FFS；Flat 输入仍处于内部验证阶段。
- 不支持的几何组合或 offset 明确报错，绝不清零几何后继续执行。
- Flat FDK 支持连续分包；Cyl FDK 一次提交完整投影，支持 Host/Device 缓冲。
- 投影布局为 [view][v][u]，体积为 [z][y][x]；容量以 float 元素个数填写。
- execute 返回后 GPU 工作已完成。分包仅首包 clear_output=true，完成后 reset 再开始下一次。

旧 SessionDesc、AlgorithmDesc、GPURes、CylAnalyticFDK 和 ISession 别名已删除。
ETask 只在 src/common 中定义，不属于 SDK。内部算法和 kernel 保持独立实现。

## 多能仿真

解析选择 `pipeline="fdk"` 或 `"wfbp"`，物理探测器类型由 geometry 决定。
迭代使用 `projection_model="joseph"` 或 `"siddon"`，
不再接受 forward_projector/back_projector 的独立组合。
