#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "Iter/kernels/YkParallelPwlsLaunch.cuh"
#include "Iter/kernels/YkIterLaunch.cuh"
#include "common/YkProjectionOperators.hpp"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "global/YkMem3d.hpp"

namespace YK::Iter {

enum class EPwlsRegularizer : int {
    None = 0,
    Quadratic = 1,
    Huber = 2
};

struct PwlsConfig {
    int iterations = 10;
    // 1 表示全批量更新；大于 1 时按 view_index % subset_count 构造
    // 交错 ordered subsets。iterations 表示完整遍历全部子集的外循环数。
    int subset_count = 1;
    float relaxation = 0.8f;
    EPwlsRegularizer regularizer = EPwlsRegularizer::Quadratic;
    float regularization = 1e-3f;
    // Huber 转折点，单位与体素值相同。|xi-xj| 小于该值时采用二次
    // 惩罚，大于该值时梯度截断，从而减少对真实材料边缘的平滑。
    float huber_delta = 3e-3f;
    float epsilon = 1e-6f;
    float lower_bound = 0.f;
    float upper_bound = std::numeric_limits<float>::max();
    ETask fp_task = ETask::FP_Joseph;
    // PWLS 的 residual BP 是数据项梯度 A^T(Ax-b)，必须选择 FP 的伴随。
    // 数值内积测试中 Joseph/Joseph 的相对误差约 1e-4，而 Joseph-v3
    // 约为 0.72；后者适合经验性体素反投，但不能作为 PWLS 默认梯度。
    ETask bp_task = ETask::BP_Joseph;
};

// 通用 PWLS 重建器。
//
//   min_x 0.5 ||Ax-b||^2 + 0.5 beta sum_(j in N(i)) (x_i-x_j)^2
//
// 每轮使用全部投影计算 A^T(b-Ax)，再以 A^T(A1) 为对角曲率主化项
// 同步更新全部体素。它执行的是 SIRT 风格的全批量并行更新，不是逐体素
// 立即传播残差的串行 ICD。几何由调用者显式传入，可用于圆轨迹、螺旋轨迹
// 或其他由当前 FP/BP 算子支持的 cone_vec 几何。
class PwlsReconstructor {
public:
    ~PwlsReconstructor() { release(); }
    PwlsReconstructor() = default;
    PwlsReconstructor(const PwlsReconstructor&) = delete;
    PwlsReconstructor& operator=(const PwlsReconstructor&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const PwlsConfig& config, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(params, geometry, config, stream)) return false;
        params_ = params;
        geometry_ = geometry;
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;

        const size_t volume_count = volumeCount_();
        buildSubsets_();
        const size_t max_subset_projection_count = static_cast<size_t>(params_.scan.Nu) *
            params_.scan.Nv * max_subset_views_;
        // 曲率预计算 A^T(A1) 仍使用完整投影，因此 forward 缓冲保留全尺寸；
        // measured/residual 仅需容纳最大的单个子集。
        d_forward_ = memory_.allocateDevice3D<float>(params_.scan.Nu, params_.scan.Nv,
            params_.scan.NAng, device_id_);
        d_residual_ = memory_.allocateDevice3D<float>(params_.scan.Nu, params_.scan.Nv,
            max_subset_views_, device_id_);
        d_measured_subset_ = memory_.allocateDevice3D<float>(params_.scan.Nu,
            params_.scan.Nv, max_subset_views_, device_id_);
        d_gradient_ = memory_.allocateDevice3D<float>(params_.volume.Nx, params_.volume.Ny,
            params_.volume.Nz, device_id_);
        d_curvature_ = memory_.allocateDevice3D<float>(params_.volume.Nx, params_.volume.Ny,
            params_.volume.Nz, device_id_);
        d_ones_ = memory_.allocateDevice3D<float>(params_.volume.Nx, params_.volume.Ny,
            params_.volume.Nz, device_id_);

        if (!fp_.init(params_, geometry_, config_.fp_task, device_id_, stream_) ||
            !bp_.init(params_, geometry_, config_.bp_task, device_id_, stream_)) {
            release();
            return false;
        }

        // A^T(A1) 是非负系统矩阵的数据 Hessian 行和，用作逐体素曲率主化。
        fill_ones_launch(d_ones_.data(), volume_count, stream_);
        if (!fp_.run(d_ones_.data(), params_, d_forward_.data(), stream_) ||
            !bp_.run(d_forward_.data(), params_, d_curvature_.data(), stream_, true)) {
            release();
            return false;
        }
        clamp_min_launch(d_curvature_.data(), volume_count,
            config_.epsilon, stream_);
        YK_CUDA_CHECK(cudaMemsetAsync(d_residual_.data(), 0,
            max_subset_projection_count * sizeof(float), stream_));
        YK_CUDA_CHECK(cudaMemsetAsync(d_gradient_.data(), 0,
            volume_count * sizeof(float), stream_));
        prepared_ = true;
        return true;
    }

    bool reconstruct(const float* d_measured_projection, float* d_volume)
    {
        if (!prepared_ || !d_measured_projection || !d_volume) return false;
        const size_t view_size = static_cast<size_t>(params_.scan.Nu) * params_.scan.Nv;
        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            for (size_t subset = 0; subset < subset_indices_.size(); ++subset) {
                const auto& indices = subset_indices_[subset];
                const SReconstructionParams& subset_params = subset_params_[subset];
                const size_t projection_count = indices.size() * view_size;
                for (size_t local_view = 0; local_view < indices.size(); ++local_view) {
                    YK_CUDA_CHECK(cudaMemcpyAsync(
                        d_measured_subset_.data() + local_view * view_size,
                        d_measured_projection + static_cast<size_t>(indices[local_view]) *
                            view_size, view_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_));
                }
                if (!fp_.run(d_volume, subset_params, d_forward_.data(), stream_))
                    return false;
                residual_launch(d_measured_subset_.data(), d_forward_.data(),
                    d_residual_.data(), projection_count, stream_);
                if (!bp_.run(d_residual_.data(), subset_params, d_gradient_.data(),
                    stream_, true)) return false;
                const float subset_scale = 1.f /
                    static_cast<float>(subset_indices_.size());
                parallel_pwls_update_launch(d_volume, d_gradient_.data(),
                    d_curvature_.data(), params_.volume.Nx, params_.volume.Ny, params_.volume.Nz,
                    config_.relaxation, subset_scale,
                    static_cast<int>(config_.regularizer),
                    config_.regularization * subset_scale, config_.huber_delta,
                    config_.epsilon, config_.lower_bound, config_.upper_bound,
                    stream_);
            }
        }
        return true;
    }

    void reset()
    {
        if (!prepared_) return;
        YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        const size_t residual_count = static_cast<size_t>(params_.scan.Nu) *
            params_.scan.Nv * max_subset_views_;
        YK_CUDA_CHECK(cudaMemsetAsync(d_residual_.data(), 0,
            residual_count * sizeof(float), stream_));
        YK_CUDA_CHECK(cudaMemsetAsync(d_gradient_.data(), 0,
            volumeCount_() * sizeof(float), stream_));
    }

    void release()
    {
        if (stream_)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        fp_.release();
        bp_.release();
        d_forward_ = {};
        d_residual_ = {};
        d_measured_subset_ = {};
        d_gradient_ = {};
        d_curvature_ = {};
        d_ones_ = {};
        geometry_.clear();
        subset_indices_.clear();
        subset_params_.clear();
        max_subset_views_ = 0;
        params_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    static bool validate_(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const PwlsConfig& config, cudaStream_t stream)
    {
        if (!stream || params.scan.Nu <= 0 || params.scan.Nv <= 0 || params.scan.NAng <= 0 ||
            params.scan.totalViews < params.scan.NAng || params.volume.Nx <= 0 || params.volume.Ny <= 0 ||
            params.volume.Nz <= 0 || static_cast<int>(geometry.size()) != params.scan.NAng ||
            config.iterations <= 0 || config.relaxation <= 0.f ||
            config.subset_count <= 0 || config.subset_count > params.scan.NAng ||
            config.regularization < 0.f || config.epsilon <= 0.f ||
            (config.regularizer == EPwlsRegularizer::Huber &&
                config.huber_delta <= 0.f) ||
            config.lower_bound > config.upper_bound) {
            YK_LOGE("[Iter::PwlsReconstructor] invalid configuration");
            return false;
        }
        for (const auto& view : geometry) {
            if (!std::isfinite(view.angle.x)) {
                YK_LOGE("[Iter::PwlsReconstructor] non-finite geometry");
                return false;
            }
        }
        return true;
    }

    size_t volumeCount_() const
    { return static_cast<size_t>(params_.volume.Nx) * params_.volume.Ny * params_.volume.Nz; }
    size_t projectionCount_() const
    { return static_cast<size_t>(params_.scan.NAng) * params_.scan.Nu * params_.scan.Nv; }

    void buildSubsets_()
    {
        subset_indices_.assign(config_.subset_count, {});
        subset_params_.resize(config_.subset_count);
        max_subset_views_ = 0;
        for (int subset = 0; subset < config_.subset_count; ++subset) {
            auto& indices = subset_indices_[subset];
            for (int view = subset; view < params_.scan.NAng; view += config_.subset_count)
                indices.push_back(view);
            max_subset_views_ = std::max(max_subset_views_,
                static_cast<int>(indices.size()));
            auto& subset_params = subset_params_[subset];
            subset_params = params_;
            subset_params.scan.NAng = static_cast<int>(indices.size());
            subset_params.scan.angles.resize(indices.size());
            for (size_t i = 0; i < indices.size(); ++i)
                subset_params.scan.angles[i] = params_.scan.angles[indices[i]];
            if (subset_params.scan.angles.size() > 1) {
                subset_params.scan.start_angle_rad = subset_params.scan.angles.front();
                subset_params.scan.range_rad = subset_params.scan.angles.back() -
                    subset_params.scan.angles.front();
            }
        }
    }

    SReconstructionParams params_{};
    std::vector<SConeProjGeomVec> geometry_{};
    PwlsConfig config_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
    int max_subset_views_ = 0;
    std::vector<std::vector<int>> subset_indices_{};
    std::vector<SReconstructionParams> subset_params_{};
    ForwardOperatorAdapter fp_{};
    BackOperatorAdapter bp_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_forward_{};
    Mem::DeviceLinearBuffer3D<float> d_residual_{};
    Mem::DeviceLinearBuffer3D<float> d_measured_subset_{};
    Mem::DeviceLinearBuffer3D<float> d_gradient_{};
    Mem::DeviceLinearBuffer3D<float> d_curvature_{};
    Mem::DeviceLinearBuffer3D<float> d_ones_{};
};

} // namespace YK::Iter
