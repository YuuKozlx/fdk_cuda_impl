#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkMacro.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"

namespace YK {

// Immutable geometry shared by pipelines and operators.  It is deliberately
// separate from algorithm workspace: a caller can reuse one geometry while
// selecting a different FP/BP implementation.
class PreparedGeometry {
public:
    // 统一系统配置入口：builder 在这里一次性生成规范 geometry，后续算子
    // 只从本对象读取，不再分别转换 Flat/Cyl 或 Circular/Helical 参数。
    bool initialize(const SSystemConfig& system)
    {
        releaseGeometry_();
        std::vector<SConeProjGeomVec> flat;
        std::vector<SCylConeProjGeomVec> cyl;
        SVolGeom volume{};
        if (!buildSystemGeometry(system, flat, cyl, volume)) return false;
        system_ = system;
        all_geometry_ = std::move(flat);
        cyl_geometry_ = std::move(cyl);
        base_ = {};
        const bool helical = system.trajectory == ETrajectoryKind::Helical;
        const bool cylindrical = system.detector == EDetectorKind::Cylindrical;
        constexpr float kTwoPi = 6.28318530717958647692f;
        if (helical) {
            base_.scan.NAng = system.helical.total_views;
            base_.scan.totalViews = system.helical.total_views;
            base_.scan.sid_mm = system.helical.sid_mm;
            base_.scan.sdd_mm = system.helical.sdd_mm;
            base_.scan.start_angle_rad = system.helical.start_angle_rad;
            base_.scan.direction = system.helical.rotation_direction;
            base_.scan.sourceOffsetX_mm = system.helical.source_offset_mm.x;
            base_.scan.sourceOffsetY_mm = system.helical.source_offset_mm.y;
            base_.scan.sourceOffsetZ_mm = system.helical.source_offset_mm.z;
            base_.scan.range_rad = kTwoPi * system.helical.total_views /
                static_cast<float>(system.helical.views_per_turn);
        } else {
            base_.scan.NAng = system.circular.total_views;
            base_.scan.totalViews = system.circular.total_views;
            base_.scan.sid_mm = system.circular.sid_mm;
            base_.scan.sdd_mm = system.circular.sdd_mm;
            base_.scan.start_angle_rad = system.circular.start_angle_rad;
            base_.scan.direction = system.circular.rotation_direction;
            base_.scan.sourceOffsetX_mm = system.circular.source_offset_mm.x;
            base_.scan.sourceOffsetY_mm = system.circular.source_offset_mm.y;
            base_.scan.sourceOffsetZ_mm = system.circular.source_offset_mm.z;
            base_.scan.range_rad = kTwoPi * system.circular.total_views /
                static_cast<float>(system.circular.views_per_turn);
        }
        base_.scan.Nu = cylindrical ? system.cylindrical_detector.channels : system.flat_detector.channels;
        base_.scan.Nv = cylindrical ? system.cylindrical_detector.rows : system.flat_detector.rows;
        base_.scan.du_mm = cylindrical ? system.cylindrical_detector.channel_arc_mm : system.flat_detector.channel_size_mm;
        base_.scan.dv_mm = cylindrical ? system.cylindrical_detector.row_size_mm : system.flat_detector.row_size_mm;
        const auto& pose = cylindrical ? system.cylindrical_detector.pose : system.flat_detector.pose;
        base_.scan.offsetU_mm = pose.offset_unv_mm.x;
        base_.scan.offsetV_mm = pose.offset_unv_mm.z;
        base_.volume.Nx = volume.Nx; base_.volume.Ny = volume.Ny; base_.volume.Nz = volume.Nz;
        base_.volume.voxelX_mm = volume.vox_x; base_.volume.voxelY_mm = volume.vox_y;
        base_.volume.voxelZ_mm = volume.vox_z;
        base_.volume.centerX_mm = volume.center.x; base_.volume.centerY_mm = volume.center.y;
        base_.volume.centerZ_mm = volume.center.z;
        all_angles_.resize(cylindrical ? cyl_geometry_.size() : all_geometry_.size());
        if (cylindrical) for (size_t i = 0; i < cyl_geometry_.size(); ++i) all_angles_[i] = cylViewAngle(cyl_geometry_[i]);
        else for (size_t i = 0; i < all_geometry_.size(); ++i) all_angles_[i] = all_geometry_[i].angle.x;
        return true;
    }

    // Internal iterative solvers already work with SReconstructionParams.  This overload
    // lets them adopt the shared context without changing their math loops.
    bool initialize(const SReconstructionParams& params)
    {
        releaseGeometry_();
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
        releaseGeometry_();
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
            SCircularTrajectorySpec trajectory{};
            trajectory.angles_rad = all_angles_;
            trajectory.sid_mm = desc.scan.SOD_mm;
            trajectory.sdd_mm = desc.scan.SDD_mm;
            trajectory.source_offset_mm = make_float3(
                desc.scan.sourceOffsetX_mm, desc.scan.sourceOffsetY_mm,
                desc.scan.sourceOffsetZ_mm);
            SFlatDetectorSpec detector{};
            detector.channels = desc.scan.Nu;
            detector.rows = desc.scan.Nv;
            detector.channel_size_mm = desc.scan.du_mm;
            detector.row_size_mm = desc.scan.dv_mm;
            detector.pose.offset_unv_mm = make_float3(
                desc.scan.offsetU_mm, 0.f, desc.scan.offsetV_mm);
            detector.pose.tilt_u_rad = desc.scan.tiltU_rad;
            detector.pose.tilt_v_rad = desc.scan.tiltV_rad;
            detector.pose.tilt_n_rad = desc.scan.tiltN_rad;
            if (!buildProjectionGeometry(trajectory, detector, all_geometry_))
                return false;
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
    const std::vector<SCylConeProjGeomVec>& cylGeometry() const { return cyl_geometry_; }
    bool isCylindrical() const { return system_.detector == EDetectorKind::Cylindrical; }
    bool hasExternalGeometry() const { return !all_geometry_.empty(); }
    SVolGeom volumeGeometry() const
    {
        SVolGeom g = SVolGeom::make_centered(base_.volume.Nx, base_.volume.Ny, base_.volume.Nz,
            base_.volume.voxelX_mm, base_.volume.voxelY_mm, base_.volume.voxelZ_mm);
        g.center = make_float3(base_.volume.centerX_mm, base_.volume.centerY_mm, base_.volume.centerZ_mm);
        return g;
    }

private:
    void releaseGeometry_()
    {
        base_ = {};
        system_ = {};
        all_geometry_.clear();
        cyl_geometry_.clear();
        all_angles_.clear();
    }
    SReconstructionParams base_{};
    SSystemConfig system_{};
    std::vector<float> all_angles_{};
    std::vector<SConeProjGeomVec> all_geometry_{};
    std::vector<SCylConeProjGeomVec> cyl_geometry_{};
};

// 旧名称仅供尚未迁移的内部算子和测试使用；新代码统一使用 PreparedGeometry。
using GeometryContext = PreparedGeometry;

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
