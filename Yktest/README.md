# Yktest 测试组织

`ykcbct_manual_tests` 是项目的 CUDA、数值回归和算法集成测试入口。测试数据分为两类：

- `synthetic/local`：测试内部生成数据，可在开发机直接运行。
- `real-data`：依赖开发者本地 RAW 数据，只能按完整测试名显式运行。

常用命令：

```powershell
# 列出测试
ykcbct_manual_tests.exe list

# 运行单项
ykcbct_manual_tests.exe filter/discrete-ramlak-dc

# 运行一个分类中的所有本地测试
ykcbct_manual_tests.exe filter
ykcbct_manual_tests.exe fp

# 运行所有不依赖真实数据的测试
ykcbct_manual_tests.exe all-local
```

## 分类约定

- `framework`：资源、几何及算子数据流。
- `filter`：滤波核和频谱数值回归。
- `fp` / `operator` / `geometry`：正反投影与伴随性质。
- `fdk` / `recon` / `iter`：重建流水线与迭代算法。
- `helical` / `fpcyl`：螺旋和圆柱探测器算法。
- `phantom`：测试模体生成器。

新增 CUDA、算法或大规模数值测试时应放入 Yktest。`unit_test` 只保留不依赖完整算法源码和 CUDA device-link 的轻量组件测试，目前用于 Logger。

真实数据测试必须将 `needs_real_data` 注册为 `true`，确保分类运行和 `all-local` 不会意外读取硬编码的本地文件。
