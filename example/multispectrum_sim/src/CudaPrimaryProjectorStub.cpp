#include "CudaPrimaryProjector.hpp"
namespace yk::spectral {
bool CudaPrimaryProjector::run(const SimulationConfig&,
    const std::vector<std::uint8_t>&, const SpectralTransmissionModel&,
    const std::filesystem::path&, std::string& error) const
{
    error = "Projection requires the CUDA-enabled parent build (BUILD_MULTISPECTRUM_SIM=ON)";
    return false;
}
}
