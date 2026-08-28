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
    // Internal iterative solvers already work with SReconstructionParams.  This overload
    // lets them adopt the shared context without changing their math loops.
    bool initialize(const SReconstructionParams& params)
    {
        if (params.scan.Nu <= 0 || params.scan.Nv <= 0 || params.scan.totalViews <= 0 ||
            params.volume.Nx <= 0 || params.volume.Ny <= 0 || params.volume.Nz <= 0)
            return false;
        base_ = params;
        all_angles_ = params.scan.angles;
        return true;
    }

    // Iterative algorithms with calibrated per-view geometry use this overload.
    // The vector is copied once during prepare and remains the only geometry
    // source for the lifetime of the operator set.
    bool initialize(const SReconstructionParams& params, const std::vector<SConeProjGeomVec>& geometry)
    {
        if (!initialize(params) || static_cast<int>(geometry.size()) != params.scan.NAng)
            return false;
        all_geometry_ = geometry;
        all_angles_.resize(geometry.size());
        for (size_t i = 0; i < geometry.size(); ++i)
            all_angles_[i] = geometry[i].angle.x;
        return true;
    }

    bool initialize(const SessionDesc& desc)
    {
        if (desc.scan.Nu <= 0 || desc.scan.Nv <= 0 || desc.scan.NAng <= 0 ||
            desc.volume.Nx <= 0 || desc.volume.Ny <= 0 || desc.volume.Nz <= 0)
            return false;

        base_ = {};
        base_.scan.Nu = desc.scan.Nu; base_.scan.Nv = desc.scan.Nv;
        base_.scan.NAng = desc.scan.NAng;
        base_.scan.totalViews = desc.scan.NAng;
        base_.scan.du_mm = desc.scan.du_mm; base_.scan.dv_mm = desc.scan.dv_mm;
        base_.scan.offsetU_mm = desc.scan.offsetU_mm; base_.scan.offsetV_mm = desc.scan.offsetV_mm;
        base_.scan.sourceOffsetX_mm = desc.scan.sourceOffsetX_mm;
        base_.scan.sourceOffsetY_mm = desc.scan.sourceOffsetY_mm;
        base_.scan.sourceOffsetZ_mm = desc.scan.sourceOffsetZ_mm;
        base_.scan.tiltN_rad = desc.scan.tiltN_rad;
        base_.scan.tiltU_rad = desc.scan.tiltU_rad;
        base_.scan.tiltV_rad = desc.scan.tiltV_rad;
        base_.scan.sid_mm = desc.scan.SOD_mm; base_.scan.sdd_mm = desc.scan.SDD_mm;
        base_.scan.range_rad = desc.scan.scanRangeRad;
        base_.scan.start_angle_rad = desc.scan.startAngleRad;
        base_.scan.short_scan = desc.scan.shortScan; base_.scan.direction = desc.scan.nDirSign;
        base_.volume.Nx = desc.volume.Nx; base_.volume.Ny = desc.volume.Ny; base_.volume.Nz = desc.volume.Nz;
        base_.volume.voxelX_mm = desc.volume.voxX_mm; base_.volume.voxelY_mm = desc.volume.voxY_mm;
        base_.volume.voxelZ_mm = desc.volume.voxZ_mm;
        base_.volume.centerX_mm = desc.volume.offsetX_mm;
        base_.volume.centerY_mm = desc.volume.offsetY_mm; base_.volume.centerZ_mm = desc.volume.offsetZ_mm;
        all_angles_ = desc.angles;
        all_geometry_ = desc.geometry;
        if (!all_geometry_.empty() &&
            static_cast<int>(all_geometry_.size()) != desc.scan.NAng)
            return false;
        if (!desc.objectFromScanner.empty() &&
            desc.objectFromScanner.size() != 1 &&
            static_cast<int>(desc.objectFromScanner.size()) != desc.scan.NAng)
            return false;
        for (const auto& transform : desc.objectFromScanner)
            if (!transform.isRigid()) return false;

        // 姿态只能作用于明确的逐视图几何。圆轨迹输入先在 Scanner 坐标系
        // 生成 vector，再统一烘焙到固定 Object 坐标系。
        if (!desc.objectFromScanner.empty() && all_geometry_.empty()) {
            if (static_cast<int>(all_angles_.size()) != desc.scan.NAng)
                return false;
            const auto rad2deg = [](float radians) {
                return radians * 180.f / CUDA_PI;
            };
            buildCircularConeGeometry(all_geometry_, all_angles_,
                desc.scan.NAng, desc.scan.Nu, desc.scan.Nv,
                desc.scan.du_mm, desc.scan.dv_mm,
                desc.scan.SOD_mm, desc.scan.SDD_mm - desc.scan.SOD_mm,
                f3(desc.scan.offsetU_mm, 0.f, desc.scan.offsetV_mm),
                f3(rad2deg(desc.scan.tiltU_rad), rad2deg(desc.scan.tiltN_rad),
                    rad2deg(desc.scan.tiltV_rad)),
                f3(desc.scan.sourceOffsetX_mm, desc.scan.sourceOffsetY_mm,
                    desc.scan.sourceOffsetZ_mm));
        }
        if (!desc.objectFromScanner.empty()) {
            for (size_t i = 0; i < all_geometry_.size(); ++i) {
                const SRigidTransform& objectFromScanner =
                    desc.objectFromScanner.size() == 1
                    ? desc.objectFromScanner.front()
                    : desc.objectFromScanner[i];
                all_geometry_[i] = transformProjectionGeometry(
                    all_geometry_[i], objectFromScanner);
            }
        }
        if (!all_geometry_.empty()) {
            all_angles_.resize(all_geometry_.size());
            for (size_t i = 0; i < all_geometry_.size(); ++i)
                all_angles_[i] = all_geometry_[i].angle.x;
        }
        return true;
    }

    SReconstructionParams batch(const float* angles, int count) const
    {
        SReconstructionParams p = base_;
        p.scan.NAng = count;
        p.scan.angles.assign(angles, angles + count);
        return p;
    }

    const SReconstructionParams& base() const { return base_; }
    const std::vector<float>& allAngles() const { return all_angles_; }
    const std::vector<SConeProjGeomVec>& allGeometry() const { return all_geometry_; }
    bool hasExternalGeometry() const { return !all_geometry_.empty(); }
    SVolGeom volumeGeometry() const
    {
        SVolGeom g = SVolGeom::make_centered(base_.volume.Nx, base_.volume.Ny, base_.volume.Nz,
            base_.volume.voxelX_mm, base_.volume.voxelY_mm, base_.volume.voxelZ_mm);
        g.center = make_float3(base_.volume.centerX_mm, base_.volume.centerY_mm, base_.volume.centerZ_mm);
        return g;
    }

private:
    SReconstructionParams base_{};
    std::vector<float> all_angles_{};
    std::vector<SConeProjGeomVec> all_geometry_{};
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
