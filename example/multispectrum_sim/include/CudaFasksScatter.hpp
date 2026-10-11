#pragma once

#include <memory>
#include <cstddef>
#include <vector>

namespace yk::spectral {

class FasksRuntimeModel;

class CudaFasksScatter {
public:
    CudaFasksScatter(int width, int height, double pixel_u_mm,
                     double pixel_v_mm, const FasksRuntimeModel& model,
                     int transmission_segments = 0);
    ~CudaFasksScatter();

    CudaFasksScatter(const CudaFasksScatter&) = delete;
    CudaFasksScatter& operator=(const CudaFasksScatter&) = delete;

    void predict(const std::vector<float>& primary,
                 const std::vector<float>& air_primary,
                 const std::vector<unsigned char>& object_mask,
                 std::vector<float>& scatter);

    void apply(std::vector<float>& primary,
               const std::vector<float>& air_primary,
               const std::vector<unsigned char>& object_mask);

    // Device buffers use the current CUDA device and default stream. The mask
    // stays on the host and its edge window is reused while its contents match.
    void predictDevice(const float* primary, const float* air_primary,
                       const std::vector<unsigned char>& object_mask, float* scatter);

    // Same fixed damping and per-pixel floor as the host correction. Images
    // remain on the GPU between iterations; only the final primary is returned.
    void correctFixed(const std::vector<float>& measured,
                      const std::vector<float>& air_primary,
                      const std::vector<unsigned char>& object_mask,
                      int iterations, double damping, double scatter_scale,
                      std::vector<float>& primary);

    // Gaussian guidance in the energy domain; normalized, nearest boundary.
    void smoothGuide(const std::vector<float>& input, double sigma_u_pixels,
                     double sigma_v_pixels, std::vector<float>& output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
