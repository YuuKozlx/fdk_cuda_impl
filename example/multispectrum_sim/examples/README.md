# BHC Stage 示例

[PolynomialBhcStage.hpp](PolynomialBhcStage.hpp) 是可以通过现有前处理 Pipeline 运行的 CPU 示例，使用统一接口，不需要直接操作模板参数。

输入为已经取过负对数的投影 p，输出为多项式 `c0 + c1*p + c2*p² + ...`，仍是投影对数衰减域。双精度 Horner 求值后检查溢出，再截断输出；截断不能隐藏溢出。插件只修改像素数值，不修改 domain/location 标记。

仅实现三项：`info()`、`contract()`、`process()`。这里不持有设备工作区，因此使用公共接口默认的 prepare/reset。系数和截断范围在构造时传入，构造期间验证参数。

## 接入真实流水线

```cpp
#include "examples/PolynomialBhcStage.hpp" // 编译时把 multispectrum_sim 根目录加入 include path

// config 是当前 SimulationConfig；input_log_raw 为匹配探测器尺寸/帧数的 float32 对数投影。
auto& pp=config.reconstruction.preprocessing;
pp.enabled=true;
pp.scatter.enabled=false;
pp.beam_hardening.enabled=false; // 由示例完成 BHC，避免内置 BHC 再应用一次。

yk::spectral::preprocess::Pipeline pipeline;
pipeline.append(std::make_unique<yk::spectral::examples::PolynomialBhcStage>(
    std::vector<double>{0.,1.,.02},0.,.8));
std::string diagnostic;
if(!pipeline.run(config,input_log_raw,corrected_raw,diagnostic))
    throw std::runtime_error(diagnostic);
```

上述系数只用于验证接口，真实 BHC 系数应来自水板拟合/标定。本示例不负责拟合；正式内置 BHC 仍使用 CUDA 像素实现。示例配置关闭散射是为了单独观察 BHC 行为。

阶段声明 Host -> Host，输入输出同域；现有流水线读取的帧本来就在 Host，因此这条示例链无需上传或下载图像。若放在设备 Stage 后面，公共执行器会在实际位置边界安排传输。前处理 Pipeline 本身仍初始化现有 CUDA 工作区，不能因此把完整应用称为不需要 CUDA 的程序。

数值、输出截断、重复运行及零帧传输次数由 `multispectrum_preprocessing` 测试中的 `testPreprocessState()` 验证。与其他原地处理 Stage 一样，异常时不会自动回滚已经写入的像素。

示例 info() 返回内置静态描述 `example.preprocess.bhc`，包含用途、参数含义、文档路径和 Experimental 状态。可通过 `pipeline::findBuiltinPluginInfo("example.preprocess.bhc")` 在不创建插件的情况下查看。名称不再通过独立 name() 维护。
