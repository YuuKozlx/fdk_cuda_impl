# YKCBCT 文档索引

## 使用入口

- [TOML 测试配置说明](toml_test_config_zh.md)：外部测试配置、任务类型、几何字段、输入输出和算法参数。
- [迭代重建指南](iterative_reconstruction_guide.md)：SIRT、SART、OS-SART、CGLS、PWLS 及收敛配置。
- [CT 迭代代数重建推导](ART_algebraic_reconstruction_derivation_zh.md)：从离散投影模型推导 ART，并说明噪声、约束与 SART 的关系。
- [螺旋模块说明](../src/Heli/README.md)：Heli 解析 wFBP 与 Flat/Cyl 迭代门面的职责边界。

## 几何与算子

- [FDK 完整推导](FDK_Full_Derivation.md)
- [C-FDK 算法与验证状态](cfdk_algorithm.md)
- [xFDK 算法说明](xfdk_algorithm.md)
- [C-FDK 论文公式翻译](cfdk_paper_formula_translation_zh.md)
- [xFDK 论文公式翻译](xfdk_paper_formula_translation_zh.md)
- [Flat/Cyl FP-BP 对比](cyl_flat_fp_bp_comparison.md)
- [FP/BP Kernel 约定](fp_bp_kernel_conventions.md)

## 其他算法

- [代数正则化规范](algebraic_regularization_spec.md)
- [TIGRE 梯度族算法](tigre_gradient_algorithms.md)
- [Ram-Lak 离散化](RamLak_Discretization.md)

`doc/cv*` 下的 CVP 资料属于实验性参考，不是当前正式 Flat/Cyl/Heli API 的使用说明。
