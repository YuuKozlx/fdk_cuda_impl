#include "CudaPrimaryProjector.hpp"
#include "SpectralProjectionPipeline.hpp"
#include "primary/CudaProjectionBackend.hpp"
#ifdef YK_SPECTRAL_VTK
#include "InteractiveProjection.hpp"
#endif
#include <stdexcept>

namespace yk::spectral {
bool CudaPrimaryProjector::run(const SimulationConfig& c,
    const std::vector<std::uint8_t>& labels, const SpectralTransmissionModel& model,
    const std::filesystem::path& output, std::string& error) const {
    try {
#ifdef YK_SPECTRAL_VTK
        if(c.visualization.enabled&&c.visualization.interactive)
            return runInteractiveProjection(c,labels,model,output,error);
#endif
        SpectralProjectionPipeline pipeline(
            std::make_unique<primary_detail::CudaProjectionBackend>());
        return pipeline.run(c, labels, model, output, error);
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace yk::spectral

