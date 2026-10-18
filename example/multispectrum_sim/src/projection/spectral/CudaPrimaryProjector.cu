#include "projection/spectral/CudaPrimaryProjector.hpp"
#include "projection/spectral/SpectralProjectionPipeline.hpp"
#include "CudaProjectionBackend.hpp"
#include <stdexcept>

namespace yk::spectral {
bool CudaPrimaryProjector::run(const SimulationConfig& c,
    const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model,
    const std::filesystem::path& output, std::string& error, IProjectionObserver* observer) const {
    try {
        SpectralProjectionPipeline pipeline(
            primary_detail::makeCudaProjectionSourceRegistry().create("projection.source.multi_spec_fp"));
        pipeline.setObserver(observer);
        return pipeline.run(c, labels, model, output, error);
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace yk::spectral

