#pragma once

#include <memory>
#include <vector>

namespace yk::spectral {

class FasksRuntimeModel;

class CudaFasksScatter {
public:
    CudaFasksScatter(int width, int height, double pixel_u_mm,
                     double pixel_v_mm, const FasksRuntimeModel& model);
    ~CudaFasksScatter();

    CudaFasksScatter(const CudaFasksScatter&) = delete;
    CudaFasksScatter& operator=(const CudaFasksScatter&) = delete;

    void apply(std::vector<float>& primary,
               const std::vector<float>& air_primary,
               const std::vector<unsigned char>& object_mask);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
