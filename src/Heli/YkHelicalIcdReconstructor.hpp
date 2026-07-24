#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHeliCTParams.h"
#include "Heli/kernels/YkHelicalIcdLaunch.cuh"
#include "Iter/kernels/YkIterLaunch.cuh"
#include "common/YkProjectionOperators.hpp"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "global/YkMem3d.hpp"

namespace YK::Helical {

struct IcdConfig {
    int iterations = 10;
    float relaxation = 0.8f;
    float regularization = 1e-3f;
    float epsilon = 1e-6f;
    float lower_bound = 0.f;
    float upper_bound = std::numeric_limits<float>::max();
    ETask fp_task = ETask::FP_Joseph;
    ETask bp_task = ETask::BP_Joseph;
};

// 螺旋 PWLS-ICD 重建器。
//
// 数据项使用逐视角 vector geometry：
//   0.5 * ||A x - b||^2
// 正则项使用 6 邻域二次势函数：
//   0.5 * beta * sum_(j in N(i)) (x_i - x_j)^2
//
// GPU 上采用可分离二次代理的并行 ICD 更新。data_curvature 由
// A^T(A1) 预计算，避免逐体素显式构造系统矩阵列；这与逐体素串行
// ICD 的执行顺序不同，但保留了逐体素曲率、邻域先验和非负约束。
class IcdReconstructor {
public:
    ~IcdReconstructor() { release(); }
    IcdReconstructor() = default;
    IcdReconstructor(const IcdReconstructor&) = delete;
    IcdReconstructor& operator=(const IcdReconstructor&) = delete;

    bool prepare(const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const IcdConfig& config, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(params, geometry, config, stream)) return false;
        params_ = params;
        geometry_ = geometry;
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;

        const size_t volume_count = volumeCount_();
        const size_t projection_count = projectionCount_();
        d_forward_ = memory_.allocateDevice3D<float>(params_.iPU, params_.iPV,
            params_.iPAng, device_id_);
        d_residual_ = memory_.allocateDevice3D<float>(params_.iPU, params_.iPV,
            params_.iPAng, device_id_);
        d_gradient_ = memory_.allocateDevice3D<float>(params_.iVX, params_.iVY,
            params_.iVZ, device_id_);
        d_curvature_ = memory_.allocateDevice3D<float>(params_.iVX, params_.iVY,
            params_.iVZ, device_id_);
        d_ones_ = memory_.allocateDevice3D<float>(params_.iVX, params_.iVY,
            params_.iVZ, device_id_);

        if (!fp_.init(params_, geometry_, config_.fp_task, device_id_, stream_) ||
            !bp_.init(params_, geometry_, config_.bp_task, device_id_, stream_)) {
            release();
            return false;
        }

        // A^T(A1) 是数据 Hessian 行和，可作为非负系统矩阵的对角主化项。
        Iter::fill_ones_launch(d_ones_.data(), volume_count, stream_);
        if (!fp_.run(d_ones_.data(), params_, d_forward_.data(), stream_) ||
            !bp_.run(d_forward_.data(), params_, d_curvature_.data(), stream_, true)) {
            release();
            return false;
        }
        Iter::clamp_min_launch(d_curvature_.data(), volume_count,
            config_.epsilon, stream_);
        YK_CUDA_CHECK(cudaMemsetAsync(d_residual_.data(), 0,
            projection_count * sizeof(float), stream_));
        YK_CUDA_CHECK(cudaMemsetAsync(d_gradient_.data(), 0,
            volume_count * sizeof(float), stream_));
        prepared_ = true;
        return true;
    }

    bool prepare(const SHeliCTParam& params, const IcdConfig& config,
        cudaStream_t stream, int device_id = 0)
    {
        std::vector<SConeProjGeomVec> geometry;
        build_helical_vec_geometry(geometry, params);
        return prepare(toCbctParams_(params), geometry, config, stream, device_id);
    }

    bool reconstruct(const float* d_measured_projection, float* d_volume)
    {
        if (!prepared_ || !d_measured_projection || !d_volume) return false;
        const size_t volume_count = volumeCount_();
        const size_t projection_count = projectionCount_();

        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            if (!fp_.run(d_volume, params_, d_forward_.data(), stream_)) return false;
            Iter::residual_launch(d_measured_projection, d_forward_.data(),
                d_residual_.data(), projection_count, stream_);
            if (!bp_.run(d_residual_.data(), params_, d_gradient_.data(),
                stream_, true)) return false;
            helical_icd_update_launch(d_volume, d_gradient_.data(),
                d_curvature_.data(), params_.iVX, params_.iVY, params_.iVZ,
                config_.relaxation, config_.regularization, config_.epsilon,
                config_.lower_bound, config_.upper_bound, stream_);
        }
        return true;
    }

    void release()
    {
        fp_.release();
        bp_.release();
        d_forward_ = {};
        d_residual_ = {};
        d_gradient_ = {};
        d_curvature_ = {};
        d_ones_ = {};
        geometry_.clear();
        params_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    static bool validate_(const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const IcdConfig& config, cudaStream_t stream)
    {
        if (!stream || params.iPU <= 0 || params.iPV <= 0 || params.iPAng <= 0 ||
            params.iPAngTotal < params.iPAng || params.iVX <= 0 || params.iVY <= 0 ||
            params.iVZ <= 0 || static_cast<int>(geometry.size()) != params.iPAng ||
            config.iterations <= 0 || config.relaxation <= 0.f ||
            config.regularization < 0.f || config.epsilon <= 0.f ||
            config.lower_bound > config.upper_bound) {
            YK_LOGE("[Helical::IcdReconstructor] invalid configuration");
            return false;
        }
        for (size_t i = 0; i < geometry.size(); ++i) {
            if (!std::isfinite(geometry[i].angle.x) ||
                (i > 0 && geometry[i].angle.x <= geometry[i - 1].angle.x)) {
                YK_LOGE("[Helical::IcdReconstructor] geometry angles must increase");
                return false;
            }
        }
        return true;
    }

    static SCBCTParams toCbctParams_(const SHeliCTParam& h)
    {
        SCBCTParams p{};
        p.angle_list = h.angle_list;
        p.iPU = h.iPU; p.iPV = h.iPV;
        p.iPAng = static_cast<int>(h.angle_list.size());
        p.iPAngTotal = p.iPAng;
        p.du_mm = h.du_mm; p.dv_mm = h.dv_mm;
        p.offsetU_mm = h.offsetU_mm; p.offsetV_mm = h.offsetV_mm;
        p.tiltu_angle_rad = h.tiltu_angle_rad;
        p.tiltn_angle_rad = h.tiltn_angle_rad;
        p.tiltv_angle_rad = h.tiltv_angle_rad;
        p.SID = h.SID; p.SDD = h.SDD;
        p.iVX = h.iVX; p.iVY = h.iVY; p.iVZ = h.iVZ;
        p.vox_x_mm = h.vox_x_mm; p.vox_y_mm = h.vox_y_mm;
        p.vox_z_mm = h.vox_z_mm;
        p.vol_offset_x_mm = h.vol_offset_x_mm;
        p.vol_offset_y_mm = h.vol_offset_y_mm;
        p.vol_offset_z_mm = h.vol_offset_z_mm;
        p.scan_start_angle_rad = h.angle_list.empty() ? 0.f : h.angle_list.front();
        p.scan_range_rad = h.angle_list.size() < 2 ? 0.f :
            h.angle_list.back() - h.angle_list.front();
        return p;
    }

    size_t volumeCount_() const
    { return static_cast<size_t>(params_.iVX) * params_.iVY * params_.iVZ; }
    size_t projectionCount_() const
    { return static_cast<size_t>(params_.iPAng) * params_.iPU * params_.iPV; }

    SCBCTParams params_{};
    std::vector<SConeProjGeomVec> geometry_{};
    IcdConfig config_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
    ForwardOperatorAdapter fp_{};
    BackOperatorAdapter bp_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_forward_{};
    Mem::DeviceLinearBuffer3D<float> d_residual_{};
    Mem::DeviceLinearBuffer3D<float> d_gradient_{};
    Mem::DeviceLinearBuffer3D<float> d_curvature_{};
    Mem::DeviceLinearBuffer3D<float> d_ones_{};
};

} // namespace YK::Helical
