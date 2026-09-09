#include "ProjectionSimulator.hpp"
#include "LibraryGeometry.hpp"

#include <algorithm>
#include <stdexcept>

namespace yk::spectral {
namespace {

void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

bool generatePathCacheWithLibraryFp(const SimulationConfig& config,
    const std::vector<std::uint8_t>& labels,
    const std::filesystem::path& cache_file, std::string& diagnostic)
{
    YK::IReconstructionSession* session = nullptr;
    PathCacheWriter writer;
    try {
        const auto& g = config.geometry_config;
        const size_t volume_elements = static_cast<size_t>(g.volume_x) *
            g.volume_y * g.volume_z;
        const size_t pixels_per_view = static_cast<size_t>(g.detector_u) *
            g.detector_v;
        const size_t projection_elements = static_cast<size_t>(g.views) *
            pixels_per_view;
        check(labels.size() == volume_elements,
            "标签体素数量与配置尺寸不一致");

        const YK::SSystemSpec system = makeLibrarySystem(
            config, YK::EPipeline::ForwardProjection,
            YK::ETask::FP_Joseph, YK::EFdkFilter::RamLak,
            GeometryUse::PhantomProjection);
        session = YK::ReconstructionSessionFactory::create();
        check(session != nullptr, "无法创建 YKCBCT DLL Session");
        if (!session->initialize(system)) {
            const std::string error = session->lastErrorMessage();
            YK::ReconstructionSessionFactory::destroy(session);
            session = nullptr;
            throw std::runtime_error("YKCBCT FP 初始化失败: " + error);
        }

        std::vector<float> host_volume(volume_elements, 0.f);
        std::vector<float> host_projection(projection_elements, 0.f);
        check(writer.open(cache_file, g.views, g.detector_u, g.detector_v,
            static_cast<std::uint32_t>(config.materials.size()), diagnostic),
            "无法创建 DLL FP 路径缓存");
        std::vector<float> view_paths(config.materials.size() * pixels_per_view,
            0.f);

        for (size_t material = 0; material < config.materials.size(); ++material) {
            std::fill(host_volume.begin(), host_volume.end(), 0.f);
            for (size_t i = 0; i < volume_elements; ++i)
                host_volume[i] = labels[i] == config.materials[material].label
                    ? 1.f : 0.f;
            YK::SExecutionRequest request{};
            request.view_count = g.views;
            request.projection = {host_projection.data(),
                YK::EMemoryLocation::Host, projection_elements};
            request.volume = {host_volume.data(), YK::EMemoryLocation::Host,
                volume_elements};
            if (!session->execute(request))
                throw std::runtime_error("YKCBCT FP 执行失败: " +
                    std::string(session->lastErrorMessage()));
            for (int view = 0; view < g.views; ++view) {
                const float* begin = host_projection.data() +
                    static_cast<size_t>(view) * pixels_per_view;
                // YKCBCT FP 返回毫米，SpectralTransmissionModel 的路径单位为厘米。
                std::transform(begin, begin + pixels_per_view,
                    view_paths.begin() + material * pixels_per_view,
                    [](float value) { return value * 0.1f; });
                if (material + 1 == config.materials.size())
                    check(writer.writeView(view_paths, diagnostic),
                        "写入 DLL FP 路径缓存失败");
            }
        }
        writer.close();
        YK::ReconstructionSessionFactory::destroy(session);
        session = nullptr;
        return true;
    } catch (const std::exception& e) {
        writer.close();
        if (session) YK::ReconstructionSessionFactory::destroy(session);
        diagnostic = e.what();
        return false;
    }
}

} // namespace yk::spectral
