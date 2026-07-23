#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkMacro.hpp"

namespace YK {

// Immutable geometry shared by pipelines and operators.  It is deliberately
// separate from algorithm workspace: a caller can reuse one geometry while
// selecting a different FP/BP implementation.
class GeometryContext {
public:
    // Internal iterative solvers already work with SCBCTParams.  This overload
    // lets them adopt the shared context without changing their math loops.
    bool initialize(const SCBCTParams& params)
    {
        if (params.iPU <= 0 || params.iPV <= 0 || params.iPAngTotal <= 0 ||
            params.iVX <= 0 || params.iVY <= 0 || params.iVZ <= 0)
            return false;
        base_ = params;
        all_angles_ = params.angle_list;
        return true;
    }

    bool initialize(const SessionDesc& desc)
    {
        if (desc.scan.Nu <= 0 || desc.scan.Nv <= 0 || desc.scan.NAng <= 0 ||
            desc.volume.Nx <= 0 || desc.volume.Ny <= 0 || desc.volume.Nz <= 0)
            return false;

        base_ = {};
        base_.iPU = desc.scan.Nu; base_.iPV = desc.scan.Nv;
        base_.iPAngTotal = desc.scan.NAng;
        base_.du_mm = desc.scan.du_mm; base_.dv_mm = desc.scan.dv_mm;
        base_.offsetU_mm = desc.scan.offsetU_mm; base_.offsetV_mm = desc.scan.offsetV_mm;
        base_.tiltn_angle_rad = desc.scan.tiltN_rad;
        base_.tiltu_angle_rad = desc.scan.tiltU_rad;
        base_.tiltv_angle_rad = desc.scan.tiltV_rad;
        base_.SID = desc.scan.SOD_mm; base_.SDD = desc.scan.SDD_mm;
        base_.scan_range_rad = desc.scan.scanRangeRad;
        base_.scan_start_angle_rad = desc.scan.startAngleRad;
        base_.bShortScan = desc.scan.shortScan; base_.nDirSign = desc.scan.nDirSign;
        base_.iVX = desc.volume.Nx; base_.iVY = desc.volume.Ny; base_.iVZ = desc.volume.Nz;
        base_.vox_x_mm = desc.volume.voxX_mm; base_.vox_y_mm = desc.volume.voxY_mm;
        base_.vox_z_mm = desc.volume.voxZ_mm;
        base_.vol_offset_x_mm = desc.volume.offsetX_mm;
        base_.vol_offset_y_mm = desc.volume.offsetY_mm; base_.vol_offset_z_mm = desc.volume.offsetZ_mm;
        all_angles_ = desc.angles;
        return true;
    }

    SCBCTParams batch(const float* angles, int count) const
    {
        SCBCTParams p = base_;
        p.iPAng = count;
        p.angle_list.assign(angles, angles + count);
        return p;
    }

    const SCBCTParams& base() const { return base_; }
    const std::vector<float>& allAngles() const { return all_angles_; }
    SVolGeom volumeGeometry() const
    {
        SVolGeom g = SVolGeom::make_centered(base_.iVX, base_.iVY, base_.iVZ,
            base_.vox_x_mm, base_.vox_y_mm, base_.vox_z_mm);
        g.center = make_float3(base_.vol_offset_x_mm, base_.vol_offset_y_mm, base_.vol_offset_z_mm);
        return g;
    }

private:
    SCBCTParams base_{};
    std::vector<float> all_angles_{};
};

// Owns the CUDA resources shared by every operator in one session.
class ResourceContext {
public:
    ~ResourceContext() { release(); }
    bool initialize(int device)
    {
        release(); device_ = device;
        YK_CUDA_CHECK(cudaSetDevice(device_));
        YK_CUDA_CHECK(cudaStreamCreate(&stream_));
        return true;
    }
    // A solver receives a stream from its caller.  Borrow it rather than
    // creating a second stream; release() will not destroy borrowed streams.
    void attach(cudaStream_t stream, int device)
    {
        release(); stream_ = stream; device_ = device; owns_stream_ = false;
    }
    void release()
    {
        if (stream_ && owns_stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
        stream_ = nullptr; owns_stream_ = true;
    }
    cudaStream_t stream() const { return stream_; }
    int device() const { return device_; }
private:
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
    bool owns_stream_ = true;
};

} // namespace YK
