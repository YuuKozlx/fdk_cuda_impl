#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "FDK/YkFdkPipeline.hpp"
#include "util/YkDeviceTensorDumper.hpp"

namespace YK {

// 诊断数据的选择掩码。它只控制独立 dumper 要观察哪些已有缓冲，不会改变
// FDK 的计算顺序、同步策略或内存分配。
enum class EFdkDumpStage : uint32_t {
    None        = 0,
    Input       = 1u << 0,
    Preweighted = 1u << 1,
    Filtered    = 1u << 2,
};

inline EFdkDumpStage operator|(EFdkDumpStage lhs, EFdkDumpStage rhs)
{
    return static_cast<EFdkDumpStage>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline bool hasDumpStage(EFdkDumpStage mask, EFdkDumpStage stage)
{
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(stage)) != 0;
}

struct FdkDumpConfig {
    std::string output_directory;
    EFdkDumpStage stages = EFdkDumpStage::Filtered;
    bool enabled = false;
};

// FDK 的独立观察/落盘组件。
//
// 使用方在 processBatch() 返回后将 pipeline.lastStage() 传给 dump()。该类
// 自己完成 D2H、等待所属 stream 和文件写入；FdkPipeline 不持有它、也不会
// 调用它。文件命名明确包含全局首视图和 chunk 大小，例如：
//   fdk_filtered_v000128_k000032.raw
class FdkStageDumper {
public:
    bool initialize(const FdkDumpConfig& config)
    {
        config_ = config;
        return dump_.initialize({ config.output_directory, config.enabled, true });
    }

    bool dump(const FdkStageView& view, int detector_u, int detector_v)
    {
        if (!config_.enabled) return true;
        if (!view.valid() || detector_u <= 0 || detector_v <= 0) return false;

        bool ok = true;
        if (hasDumpStage(config_.stages, EFdkDumpStage::Input))
            ok = dump_.dump({ fileName_("input", view), view.d_input,
                { view.count, detector_v, detector_u }, view.stream }) && ok;
        if (hasDumpStage(config_.stages, EFdkDumpStage::Preweighted))
            ok = dump_.dump({ fileName_("preweighted", view), view.d_preweighted,
                { view.count, detector_v, detector_u }, view.stream }) && ok;
        if (hasDumpStage(config_.stages, EFdkDumpStage::Filtered))
            ok = dump_.dump({ fileName_("filtered", view), view.d_filtered,
                { view.count, detector_v, detector_u }, view.stream }) && ok;
        return ok;
    }

private:
    static std::string fileName_(const char* stage, const FdkStageView& view)
    {
        char name[96]{};
        std::snprintf(name, sizeof(name), "fdk_%s_v%06d_k%06d", stage,
            view.first_view, view.count);
        return name;
    }

    FdkDumpConfig config_{};
    IO::DeviceTensorDumper dump_{};
};

} // namespace YK
