#include "ProjectionSimulator.hpp"
#include "CudaPrimaryProjector.hpp"
namespace yk::spectral {
bool ProjectionSimulator::runToFile(const std::vector<std::uint8_t>& labels,
                                    const std::filesystem::path& output_file,
                                    std::string& error) const {
    if (config_.projection.engine == "pixel_local_random" || config_.projection.engine == "detector_global_random")
        return CudaPrimaryProjector{}.run(config_, labels, model_, output_file, error);
    error = "projection.engine 仅支持 pixel_local_random 或 detector_global_random";
    return false;
}
}
