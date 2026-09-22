#pragma once
#include "config/SimConfig.hpp"
#include "SpectralModel.hpp"
#include <vector>
#include <filesystem>
#include <string>
#include "PathCache.hpp"

namespace yk::spectral {
// 通过 YKCBCT 公共 DLL FP 生成材料路径缓存。没有链接 YKCBCT 时由独立示例
// 提供回退实现；适配器不访问 src 内部 kernel、geometry 或内存类。
bool generatePathCacheWithLibraryFp(const SimulationConfig&, const std::vector<std::uint8_t>&,
                                    const std::filesystem::path&, std::string& diagnostic);
bool reconstructWithLibraryFdk(const SimulationConfig&, const std::filesystem::path&,
                               const std::filesystem::path&, const std::filesystem::path&,
                               std::string& diagnostic);
}
namespace yk::spectral {
class ProjectionSimulator {
public:
    ProjectionSimulator(const SimulationConfig& config, const SpectralTransmissionModel& model)
        : config_(config), model_(model) {}
    // 输出按 view、v、u 排列的 -log(I/I0) 投影。
    std::vector<float> run(const std::vector<std::uint8_t>& labels) const;
    bool runToFile(const std::vector<std::uint8_t>& labels,
                   const std::filesystem::path& output_file,
                   std::string& error) const;
    bool generatePathCache(const std::vector<std::uint8_t>& labels,
                           const std::filesystem::path& cache_file,
                           std::string& error) const;
    bool runFromPathCache(const std::filesystem::path& cache_file,
                          const std::filesystem::path& output_file,
                          std::string& error) const;
private:
    SimulationConfig config_; const SpectralTransmissionModel& model_;
};
}
