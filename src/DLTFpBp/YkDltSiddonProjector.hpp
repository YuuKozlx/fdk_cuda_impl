#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "DLTFpBp/kernels/YkDltSiddonLaunch.cuh"
#include "global/YkMacro.hpp"
#include "global/YkMem3d.hpp"

namespace YK::DltFpBp {

class DltSiddonProjector {
public:
    DltSiddonProjector() = default;
    ~DltSiddonProjector() = default;
    DltSiddonProjector(const DltSiddonProjector&) = delete;
    DltSiddonProjector& operator=(const DltSiddonProjector&) = delete;

    bool prepare(const std::vector<SDltRayGeometry>& geometry,
        const SVolGeom& volume_geometry, int channels, int rows,
        int device_id = 0)
    {
        release();
        if (geometry.empty() || channels <= 0 || rows <= 0 ||
            volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0 || !(volume_geometry.vox_x > 0.f) ||
            !(volume_geometry.vox_y > 0.f) || !(volume_geometry.vox_z > 0.f))
            return false;
        device_id_ = device_id;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        volume_geometry_ = volume_geometry;
        Mem::PodDataController memory;
        device_geometry_ = memory.allocateAndUpload(geometry, device_id_);
        prepared_ = static_cast<bool>(device_geometry_);
        return prepared_;
    }

    bool forward(const float* device_volume, float* device_projection,
        bool accumulate, cudaStream_t stream) const
    {
        if (!prepared_ || !device_volume || !device_projection) return false;
        YK_CUDA_CHECK(cudaSetDevice(device_id_));
        dltSiddonForwardLaunch(device_volume, device_projection,
            device_geometry_.data(), volume_geometry_, channels_, rows_, views_,
            accumulate, stream);
        return true;
    }

    bool backproject(const float* device_projection, float* device_volume,
        bool clear_volume, cudaStream_t stream) const
    {
        if (!prepared_ || !device_projection || !device_volume) return false;
        YK_CUDA_CHECK(cudaSetDevice(device_id_));
        if (clear_volume) {
            const size_t count = static_cast<size_t>(volume_geometry_.Nx) *
                volume_geometry_.Ny * volume_geometry_.Nz;
            YK_CUDA_CHECK(cudaMemsetAsync(device_volume, 0,
                count * sizeof(float), stream));
        }
        dltSiddonBackLaunch(device_projection, device_volume,
            device_geometry_.data(), volume_geometry_, channels_, rows_, views_,
            stream);
        return true;
    }

    void release()
    {
        device_geometry_.reset();
        volume_geometry_ = {};
        channels_ = rows_ = views_ = 0;
        prepared_ = false;
    }

    int channels() const { return channels_; }
    int rows() const { return rows_; }
    int views() const { return views_; }

private:
    Mem::DeviceLinearBuffer<SDltRayGeometry> device_geometry_{};
    SVolGeom volume_geometry_{};
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
    bool prepared_ = false;
};

} // namespace YK::DltFpBp
