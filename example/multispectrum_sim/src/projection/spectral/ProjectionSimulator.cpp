#include "projection/spectral/ProjectionSimulator.hpp"
#include "projection/spectral/CudaPrimaryProjector.hpp"
namespace yk::spectral {
bool ProjectionSimulator::runToFile(const std::vector<std::uint8_t>& labels,
                                    const std::filesystem::path& output_file,
                                    std::string& error, IProjectionObserver* observer) const {
    return CudaPrimaryProjector{}.run(config_, labels, model_, output_file, error, observer);
}
}
