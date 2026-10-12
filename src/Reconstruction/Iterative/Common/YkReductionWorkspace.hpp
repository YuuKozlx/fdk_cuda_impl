#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime_api.h>
namespace YK::Iter {
constexpr int scalar_reduction_blocks = 1024;
class ReductionWorkspace {
    double* device_ = nullptr;
    std::array<double, scalar_reduction_blocks> host_ {};

public:
    ~ReductionWorkspace() { release(); }
    bool prepare()
    {
        release();
        return cudaMalloc(reinterpret_cast<void**>(&device_), sizeof(double) * scalar_reduction_blocks) == cudaSuccess;
    }
    void release()
    {
        if (device_)
            cudaFree(device_);
        device_ = nullptr;
    }
    double* data() { return device_; }
    bool read(cudaStream_t stream, double& value, bool maximum = false)
    {
        if (cudaMemcpyAsync(host_.data(), device_, sizeof(host_), cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess)
            return false;
        value = 0;
        for (double x : host_) {
            if (!std::isfinite(x))
                return false;
            value = maximum ? std::max(value, x) : value + x;
        }
        return std::isfinite(value);
    }
};
}
