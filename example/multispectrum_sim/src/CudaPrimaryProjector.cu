#include "CudaPrimaryProjector.hpp"
#include "SpectralProjectionPipeline.hpp"
#include "primary/CudaProjectionBackend.hpp"
#include <stdexcept>

namespace yk::spectral {
bool CudaPrimaryProjector::run(const SimulationConfig& c,
    const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model,
    const std::filesystem::path& output, std::string& error) const {
    try {
        SpectralProjectionPipeline pipeline(
            std::make_unique<primary_detail::CudaProjectionBackend>());
        return pipeline.run(c, labels, model, output, error);
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace yk::spectral

