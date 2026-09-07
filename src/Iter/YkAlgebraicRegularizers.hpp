#pragma once

#include "Iter/kernels/YkIterLaunch.cuh"
#include "Iter/kernels/YkTvRegularizationLaunch.cuh"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

#include <cuda_runtime.h>
#include <global/YkCBCTParams.h>

#include <limits>
#include <memory>

namespace YK::Iter {

// 正则化与数据更新算法、权重模型和几何入口相互独立。后续可增加
// ASD-POCS、Chambolle-Pock、FISTA-TV prox、ADMM/Split-Bregman，
// 而不需要复制或修改 SIRT/SART/OSSART 后端。
enum class EAlgebraicRegularizer : int {
    None = 0,
    SmoothedTv = 1
};

enum class ETvDimensionality : int {
    Slice2D = 2,
    Volume3D = 3
};

struct AlgebraicRegularizationConfig {
    EAlgebraicRegularizer type = EAlgebraicRegularizer::None;
    float strength = 0.f;
    int inner_iterations = 5;
    ETvDimensionality tv_dimensionality = ETvDimensionality::Volume3D;
    float epsilon = 1e-6f;
    float strength_reduction = 1.f;
};

// 调度器传给正则器的外循环上下文。data_update_l2 专门为 ASD-POCS
// 等需要约束“正则化步长/数据更新幅度”比例的策略预留。
struct AlgebraicRegularizationContext {
    int outer_iteration = 0;
    int completed_subset_updates = 0;
    float data_update_l2 = std::numeric_limits<float>::quiet_NaN();
    float data_relaxation = 1.f;
};

class IAlgebraicRegularizer {
public:
    virtual ~IAlgebraicRegularizer() = default;
    virtual bool prepare(const SReconstructionParams& params,
        const AlgebraicRegularizationConfig& config,
        cudaStream_t stream, int device_id) = 0;
    virtual bool apply(float* d_volume,
        const AlgebraicRegularizationContext& context) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual bool requiresDataUpdateNorm() const { return false; }
};

class SmoothedTvRegularizer final : public IAlgebraicRegularizer {
public:
    ~SmoothedTvRegularizer() override { release(); }

    bool prepare(const SReconstructionParams& params,
        const AlgebraicRegularizationConfig& config,
        cudaStream_t stream, int device_id) override
    {
        release();
        params_ = params;
        config_ = config;
        stream_ = stream;
        strength_ = config.strength;
        volume_elements_ = static_cast<size_t>(params.volume.Nx) *
            params.volume.Ny * params.volume.Nz;
        if (cudaSetDevice(device_id) != cudaSuccess) {
            release();
            return false;
        }
        gradient_ = memory_.allocateDevice3D<float>(params.volume.Nx,
            params.volume.Ny, params.volume.Nz, device_id, false);
        if (!gradient_) { release(); return false; }
        prepared_ = true;
        return true;
    }

    bool apply(float* d_volume,
        const AlgebraicRegularizationContext& context) override
    {
        if (!prepared_ || !d_volume) return false;
        for (int iteration = 0; iteration < config_.inner_iterations; ++iteration) {
            tv_gradient_launch(d_volume, gradient_.data(),
                params_.volume.Nx, params_.volume.Ny, params_.volume.Nz,
                params_.volume.voxelX_mm, params_.volume.voxelY_mm, params_.volume.voxelZ_mm,
                config_.epsilon,
                static_cast<int>(config_.tv_dimensionality), stream_);
            axpy_launch(d_volume, gradient_.data(), -strength_,
                volume_elements_, stream_);
        }
        YK_LOGD("[SmoothedTvRegularizer] outer={} inner={} strength={:.4e}",
            context.outer_iteration + 1, config_.inner_iterations, strength_);
        strength_ *= config_.strength_reduction;
        return true;
    }

    void reset() override { strength_ = config_.strength; }

    void release() override
    {
        gradient_ = {};
        volume_elements_ = 0;
        stream_ = nullptr;
        strength_ = 0.f;
        prepared_ = false;
    }

private:
    SReconstructionParams params_{};
    AlgebraicRegularizationConfig config_{};
    cudaStream_t stream_ = nullptr;
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> gradient_{};
    size_t volume_elements_ = 0;
    float strength_ = 0.f;
    bool prepared_ = false;
};

// 正则化策略门面。状态型求解器可以在自己的实现中持有对偶变量、
// 辅助变量或历史体积；统一代数重建器只负责固定的外循环调度协议。
class AlgebraicRegularizer {
public:
    bool prepare(const SReconstructionParams& params,
        const AlgebraicRegularizationConfig& config,
        cudaStream_t stream, int device_id)
    {
        release();
        if (config.type == EAlgebraicRegularizer::None) return true;
        if (config.type == EAlgebraicRegularizer::SmoothedTv)
            implementation_ = std::make_unique<SmoothedTvRegularizer>();
        if (!implementation_) return false;
        if (!implementation_->prepare(params, config, stream, device_id)) {
            release();
            return false;
        }
        return true;
    }

    bool apply(float* d_volume,
        const AlgebraicRegularizationContext& context)
    { return !implementation_ || implementation_->apply(d_volume, context); }

    bool enabled() const { return implementation_ != nullptr; }
    bool requiresDataUpdateNorm() const
    { return implementation_ && implementation_->requiresDataUpdateNorm(); }
    void reset() { if (implementation_) implementation_->reset(); }
    void release()
    {
        if (implementation_) implementation_->release();
        implementation_.reset();
    }

private:
    std::unique_ptr<IAlgebraicRegularizer> implementation_{};
};

} // namespace YK::Iter
