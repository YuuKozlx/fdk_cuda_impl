# 公共 API

本篇只描述安装 SDK。内部重建器见 [架构与开发](architecture-and-development.md)，算法/缓冲区支持组合见 [算法矩阵](algorithms.md#公共支持矩阵)。主要依据：[公共头](../include/YKCBCT/interface/YkReconstructionApi.hpp)、[边界校验](../src/interface/YkReconstructionApi.cpp)、[后端实现](../src/interface/YkSession.cpp)。

## 入口和生命周期

包含 `YKCBCT/interface/YkReconstructionApi.hpp`。使用 `ReconstructionSessionFactory::create()` 创建 Session，并用同一 factory 的 `destroy()` 销毁；建议以自定义 deleter 的 `unique_ptr` 管理。

1. `SSystemSpec{}` 设置物理几何、算法和 GPU 编号。
2. `initialize(system)` 准备几何与 GPU 资源；重新 initialize 会释放旧资源。
3. `SExecutionRequest{}` 提交调用方持有的连续 float 数组。
4. `execute(request)` 返回成功时，本次 GPU 工作已完成。
5. `reset()` 清除执行/分包状态，保留已准备资源；换系统或算法需重新 initialize。
6. `release()` 释放后端资源，使 Session 不再初始化；factory destroy 最终销毁对象。

不要假设同一 Session 支持并发 execute。API 没有暴露用户 stream、完成事件或线程安全契约；调用方应串行操作，并在跨 stream 提交 Device 输入前自行保证生产者已完成。

## 完整公共调用示例

下面程序生成球形体积，通过 DLL 正投，再通过另一 Session 做 Flat 圆轨迹 FDK。它只使用安装头文件，不使用内部模体或重建器。

```cpp
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "YKCBCT/interface/YkReconstructionApi.hpp"

using Session = std::unique_ptr<YK::IReconstructionSession,
    decltype(&YK::ReconstructionSessionFactory::destroy)>;

Session makeSession() {
    Session s(YK::ReconstructionSessionFactory::create(),
              &YK::ReconstructionSessionFactory::destroy);
    if (!s) throw std::runtime_error("create failed");
    return s;
}
void check(bool ok, YK::IReconstructionSession& s) {
    if (!ok) throw std::runtime_error(s.lastErrorMessage());
}
YK::Buffer host(std::vector<float>& a) {
    return {a.data(), YK::EMemoryLocation::Host, a.size()};
}
int main() {
    try {
        YK::SSystemSpec system{};
        auto& g = system.geometry;
        g.detector = YK::EDetectorKind::Flat;
        g.trajectory = YK::ETrajectoryKind::Circular;
        g.circular.total_views = 360;
        g.circular.views_per_turn = 360;
        g.circular.sid_mm = 200.f;
        g.circular.sdd_mm = 400.f;
        g.flat_detector.channels = 128;
        g.flat_detector.rows = 128;
        g.flat_detector.channel_size_mm = 0.8f;
        g.flat_detector.row_size_mm = 0.8f;
        g.volume = {64, 64, 64, 0.75f, 0.75f, 0.75f, {0.f, 0.f, 0.f}};
        std::vector<float> truth(64u * 64 * 64), volume(truth.size());
        std::vector<float> projection(360u * 128 * 128);
        for (int z = 0; z < 64; ++z)
            for (int y = 0; y < 64; ++y)
                for (int x = 0; x < 64; ++x) {
                    float dx = (x - 31.5f) * 0.75f;
                    float dy = (y - 31.5f) * 0.75f;
                    float dz = (z - 31.5f) * 0.75f;
                    truth[(size_t(z) * 64 + y) * 64 + x] =
                        dx*dx + dy*dy + dz*dz < 100.f ? 0.02f : 0.f;
                }
        auto fp = makeSession();
        system.reconstruction.pipeline = YK::EPipeline::ForwardProjection;
        check(fp->initialize(system), *fp);
        YK::SExecutionRequest request{};
        request.projection = host(projection);
        request.volume = host(truth);
        request.view_count = 360;
        check(fp->execute(request), *fp);
        fp.reset(); // unique_ptr reset：销毁 FP Session

        auto recon = makeSession();
        system.reconstruction.pipeline = YK::EPipeline::FDK;
        check(recon->initialize(system), *recon);
        request.volume = host(volume);
        check(recon->execute(request), *recon);
        for (float value : volume)
            if (!std::isfinite(value)) throw std::runtime_error("non-finite output");
        std::cout << "center=" << volume[(32u * 64 + 32) * 64 + 32] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
```

构建方法见 [外部工程接入](build-and-integration.md#安装与外部工程)。本例的 finite 检查只验证基本执行；量纲与质量验收需运行定量测试。

## 系统参数

`SSystemSpec` 由 `geometry`、`reconstruction`、`device` 和可选尾部 `nonstandard` 组成。规则圆轨迹读取 `geometry.circular`，螺旋读取 `geometry.helical`；两类非标准轨迹读取相应 `nonstandard` 分支。只填写所选系统需要的参数。

`SReconstructionSpec` 用 `pipeline` 选择算法，`projection_model` 选择 Joseph/Siddon 配对。默认 FDK、Joseph。当前后端将 Joseph 映射到 Joseph FP + Joseph-v3 BP，将 Siddon 映射到 Siddon FP + Siddon RayDriven BP。公共 API 不接受 FP/BP 的任意独立组合。

| 参数组 | 声明默认值与消费情况 |
| --- | --- |
| `fdk` | Parker Auto，RamLak，cutoff=0.5，gain=1；Butterworth order=2、Kaiser beta=8.6、Tukey alpha=0.5；相关滤波路径消费 |
| `iterative` | iterations=10，relaxation=1，subsets=1；各迭代后端读取适用字段，CGLS 不按 SART 松弛更新 |
| `cgls` | minimum_iterations=0、check_interval=1、patience=1、relative_residual=0、robust_restart=true；当前 Cyl 通用迭代路径传入前四项，Flat CGLS Session 未映射该组，robust_restart 也未映射 |
| `pwls` | Quadratic，regularization=1e-3、huber_delta=3e-3、epsilon=1e-6、lower_bound=0、upper_bound=float 最大值；use_projection_weights=false |
| `tigre` | method=OsSart，block_size=20，lambda_reduction=1，Nesterov=false，FDK 初值=false，非负=true；TV 次数=20、alpha=0.002、衰减=0.95、maximum_update_ratio=0.95、max_l2_error=-1 |
| `wfbp` | redundancy_flat=0.6，filter_cutoff=1，filter_apodization=1；与普通 FDK cutoff 的归一化语义不同 |

重要消费限制：

- Session 的 TIGRE `block_size` 实际由 `iterative.subsets` 与视图数换算，未直接读取公开的 `tigre.block_size`。SART 为每块一视图，SIRT 为全视图一块。
- 通用 SART/SIRT/OS-SART 路径也读取 `tigre.non_negative`，即使 pipeline 不是 TigreGradient。
- `pwls.use_projection_weights` 已声明，但公共执行请求没有权重数组字段，Session 也没有把该开关映射到后端。当前公共调用使用默认单位权重，不能通过此字段实现逐射线权重。
- 内部平滑 TV、权重模型、子集顺序和完整收敛统计并未全部导出到公共 API；不要直接把内部 Config 示例改名后当作 SDK 示例。

参数完整定义位于 [analytic](../include/YKCBCT/algorithms/analytic/YkFdkParams.hpp)、[iterative](../include/YKCBCT/algorithms/iterative/YkIterationParams.hpp)、[CGLS](../include/YKCBCT/algorithms/iterative/YkCglsParams.hpp)、[PWLS](../include/YKCBCT/algorithms/iterative/YkPwlsParams.hpp)、[TIGRE](../include/YKCBCT/algorithms/iterative/YkTigreParams.hpp)、[WFBP](../include/YKCBCT/algorithms/analytic/YkWfbpParams.hpp)。新增字段时同时核对其后端消费路径。

## 请求与缓冲区

| 字段 | 当前含义 |
| --- | --- |
| `projection` / `volume` | 非空连续 float 指针、Host/Device 位置、float 元素容量 |
| `view_offset` | 本包在初始化序列中的起点，默认 0；不会自动偏移传入的数据指针 |
| `view_count` | 正值为本包视图数；非正值按整个初始化序列处理，不是“剩余视图” |
| `clear_output` | 默认 true；FDK 清零/累加控制，不能作为所有迭代算法的通用初值开关 |
| `iteration_count` | 默认 0；当前只在 Flat SIRT、Flat OS-SART execute 分支用于正值覆盖，其他路径按初始化配置执行 |

投影容量至少 `view_count * rows * channels`，体积容量至少 `nx * ny * nz`，以元素个数填写，不能填字节数。投影 `[view][v][u]`，体积 `[z][y][x]`，见 [数据布局](geometry-and-data.md#数组与物理单位)。

正投时 volume 是输入，projection 是输出；重建时相反。输入输出均由调用方分配并持有，至少保持有效到 execute 返回；Device 指针必须属于所选 GPU。不要依赖输入原样保留或缓冲区别名，部分内部解析路径会在设备投影上进行处理。

普通 FP、Cyl FDK、非标准 FDK 和公共 Cyl Helical WFBP 支持 Host/Device 传输路径。公共 Flat FDK 的 projection 当前只接受 Host，volume 可为 Host/Device；内部 Flat FDK 已有 Device 批次入口，但 Session 尚未映射该能力。XFDK 与迭代重建要求双方都是 Device。组合限制以 [算法矩阵](algorithms.md#公共支持矩阵) 为准。

## Flat 圆轨迹 FDK 分包

只有公共普通 Flat 圆轨迹 FDK 支持连续视图分包。XFDK 的内部流式实现不是公共 Session 分包契约；非标准轨迹 FDK 也要求完整输入。

非标准 FDK 内部管线已支持 Host/Device 逐包输入，详见 [内部流式契约](algorithms.md#非标准-fdk-内部流式契约)。公共 execute 仍不接受部分序列，Host 输入也会先完整上传到 Device；使用内部能力不会自动改变本篇的 SDK 契约。

柱面 FDK 内部 Reconstruction 也已支持逐包映射、滤波、反投累加，详见 [柱面内部流式契约](algorithms.md#柱面-fdk-内部流式契约)。公共 Cyl Session 仍要求完整输入。

```cpp
recon->reset();
const size_t per_view = 128u * 128;
for (int offset = 0; offset < 360; offset += 47) {
    int count = (360 - offset < 47) ? 360 - offset : 47;
    YK::SExecutionRequest batch{};
    batch.projection = {projection.data() + size_t(offset) * per_view,
                        YK::EMemoryLocation::Host, size_t(count) * per_view};
    batch.volume = host(volume);
    batch.view_offset = offset;
    batch.view_count = count;
    batch.clear_output = (offset == 0);
    check(recon->execute(batch), *recon);
}
```

包必须从 offset=0 开始，连续无遗漏/重复；只有首包允许 clear_output=true。体积缓冲区应保持同一个累加结果，不要换成未初始化的数组。完整执行也是一次包；同一 Session 下一次重建前 reset。API 没有独立 complete 方法，调用方负责提交全部视图。

迭代续算/初值行为取决于对应重建器，`clear_output` 在 Session 迭代分支没有传给重建器。SDK 用户应使用初始化配置中的 iterations 完成一次请求；需要内部增量状态控制时先查具体实现。

## ABI 与错误

`SSystemSpec`、`SExecutionRequest` 均默认 api_version=2，并带 struct_size。使用 `{}` 初始化，保留自动设置的版本与尺寸。边界检查必需字段是否包含在尺寸内；非标准轨迹额外要求 `nonstandard` 尾部存在。较小的旧 v2 系统结构在无需该尾部的场景可通过尺寸检查，这不意味着任意旧布局都兼容。

这是一套 C++ DLL 接口，包含虚函数及 CUDA 类型，不是稳定的 C ABI。应使用匹配的安装头文件、编译器 ABI 和运行库；不要跨边界自行 delete factory 创建的对象。

| `EApiErrorCode` | 常见触发 |
| --- | --- |
| `InvalidConfig` | 枚举、版本、尺寸或视图范围不合法 |
| `InvalidGeometry` | 系统/算法初始化失败；具体后端原因还需查日志 |
| `UnsupportedCombination` | 已检测出的 offset/tilt 不支持，或 WFBP 后端未编入 |
| `AlgorithmUnderTest` | 该算法/几何组合未获公共边界放行 |
| `InvalidBuffer` | 空指针、位置枚举非法或元素容量不足 |
| `StreamingStateError` | 未初始化、分包不连续、后续包清零、非流式算法提交部分序列 |
| `CudaError` | execute 后端失败；也可能包含后端拒绝当前缓冲区位置等原因 |

失败后立即读取 `lastError()` 和 `lastErrorMessage()`；消息指针不应跨后续 Session 操作长期保存。公开错误分类比底层日志粗，不能仅凭 `CudaError` 断定驱动故障。
