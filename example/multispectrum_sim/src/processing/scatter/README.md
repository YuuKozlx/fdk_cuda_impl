# 实验性动态散射求解插件

固定迭代仍为正式默认流程。此模块控制已有散射预测器的安全更新步长并记录迭代过程，
不拟合 gamma、散射幅度或图像平坦度，也不需要独立 DLL 加载器。

CUDA 顶层构建时启用：

```text
-DMULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER=ON
```

配置示例；从 `scatter.correction` 删除固定流程的 `iterations` 和 `damping`，
保留 `scatter_scale` 与 `object_mask_threshold`：

```toml
[reconstruction.preprocessing.scatter.solver]
type = "adaptive_experimental"

[reconstruction.preprocessing.scatter.solver.adaptive]
iterations = 10
initial_damping = 0.7
min_damping = 0.1

[reconstruction.preprocessing.scatter.solver.adaptive.transmission_guard]
low_quantile = 0.01
risk_threshold = 0.02
positivity_margin = 0.9
```

预测前统计物体掩膜区域测量透射率 `I/I0` 的低分位数，判断低透射率风险。
单个最小像素不作为全图判断依据；剂量等比例变化不影响透射率。
默认阈值 0.02 仅用于实验风险提示，不是通用标定值。
未传入掩膜时统计整幅图像；空物体掩膜不报告低透射率风险。

回代残差为 `r = I - P - scatter_scale * max(S(P), 0)`。
更新使用 `P_new = P + damping * r`，不再把负目标 `I-scale*S` 硬截断。
对于所有下降像素，CUDA 归约得到整帧正值安全步长上限：

```text
damping <= positivity_margin * (P - I0*1e-8) / (-r)
```

默认余量 0.9 保证更新后的初级信号仍高于正值下限。
即使测量透射率未触发风险提示，也检查该上限，因为散射可能掩盖低初级透射率。
正值下限只作为约束，不作为迭代目标。

候选步骤若使残差上升，就减半步长并从最后接受的解重试。
下一次更新允许步长恢复至上次的 1.25 倍，但不能超过初始值或新的正值上限。
正值保护可能要求步长低于 min_damping；当前重试的最低值取两者中较小者。
最低步长仍不能降低残差时报告 stalled；耗尽正值空间时报告 positivity_stalled。
正常执行指定次数后报告 completed，仅表示已完成次数，不表示物理准确或已收敛。
只有无法得到安全下降步骤时才提前终止，并明确报告上述异常状态和实际次数。

按照 GGEMS ASKS 的方式，用 iterations 指定接受的更新次数，不设置容差提前停止。
初始状态和每轮更新都记录回代残差相对测量信号的 L2 范数，以及初级信号相对上一轮
的变化量。它们只作为观察曲线，不换算成 HU，也不用于宣称已收敛。
每次尝试需要一次散射预测，接受的候选预测会在下一轮复用。
CUDA 负责像素更新、误差归约和安全步长归约；CPU 负责小规模汇总、分位数、
回退和次数控制。设备缓冲区跨帧复用。

重建预处理使用设备预测接口，初级信号、空气场、候选解、散射预测和已接受的散射
常驻 GPU。每轮只回传归约统计量，结束后下载最终初级信号；主机预测接口仍可用于
独立验证。散射源幅度及核间线性混合在 CUDA 中生成，不再在 CPU 建立整组源图并逐核
上传。边界距离保持原精确算法，但掩膜内容不变时复用缓存；掩膜变化会重新计算。
每个核的源图仍执行正向 FFT，频域乘积求和后只执行一次逆 FFT。
非零 gamma 的普通项与 WET 加权项分别走这条路径，仍保留两个独立卷积结果。

日志逐帧记录状态、接受次数、预测次数、回退次数、实际步长、残差、变化量、
低分位透射率、风险标志、正值限步次数和耗时。
回代残差只衡量与当前散射模型的一致性，不代表物理准确或图像平坦。
掩膜仍由原预处理阈值估计，散射模型和 BHC 的误差需要独立验证。

## 可选平滑引导与步长衰减

步长衰减仅用于自适应求解器；平滑引导为两种求解器共用，默认关闭：

```toml
[reconstruction.preprocessing.scatter.solver.adaptive]
iterations = 10
initial_damping = 0.7
min_damping = 0.0001
damping_decay = 0.9 # 默认1；第k次更新的步长上限为 initial_damping * decay^(k-1)

[reconstruction.preprocessing.scatter.guide]
enabled = true # 默认false
sigma_u_pixels = 2.0
sigma_v_pixels = 2.0
```

引导模式在 CUDA 上对测量能量图和空气能量图做归一化的可分离高斯平滑，再从
平滑透射率估计掩膜并求解初级信号。最终散射由引导初级信号预测，扣除对象仍是
未平滑的原始测量能量图；日志 `scatter_guide` 记录最终触及正值下限的像素数。
标准差单位为探测器像素，允许范围为 `(0,32]`，边界使用最近像素延拓。

`damping_decay` 只衰减更新步长上限，不衰减 `scatter_scale`，不会逐轮改变散射
校正目标。范围为 `(0,1]`，不引入容差停止。减小步长不能保证消除模型过估计。

平滑引导不等于噪声恢复：弱信号下，原始测量减去估计散射仍可能为负，截断后取
对数仍可能产生严重条纹。简单高斯平滑也可能影响物体边缘，因此该选项暂仅用于
实验，不应作为低曝光问题已解决的依据。
