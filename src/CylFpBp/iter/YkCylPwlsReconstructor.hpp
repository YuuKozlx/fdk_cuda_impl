#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include "CylFpBp/bp/YkCylBackOperator.hpp"
#include "CylFpBp/fp/YkCylForwardOperator.hpp"
#include "CylFpBp/kernels/YkCylPwlsLaunch.cuh"
#include "Iter/YkPwlsReconstructor.hpp"
#include "Iter/kernels/YkIterLaunch.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp {

enum class ECylPwlsDataModel {
    // Point Joseph 参考 FP + Joseph BP：高伴随，适合定量 PWLS。
    JosephMatched,
    // Siddon FP + Siddon ray-driven BP：严格路径长度伴随，但较慢。
    Siddon
};

// Cyl 几何上的惩罚加权最小二乘（PWLS/PLS）重建。它是迭代算法，不经过
// Cyl-FDK 的曲率 map、余弦预加权、滤波或解析深度权重；R!=SDD 也不构成
// 限制。当前实现使用完整逐视图批量更新，静态和螺旋 geometry 均可使用。
//
// 目标函数：0.5 (Ax-b)^T W (Ax-b) + 0.5 beta R(x)。默认 W=I；可选
// 设备端 [view][row][channel] 数组在 prepare() 时复制为重建器自有数据。
struct CylPwlsConfig {
    int iterations = 10;
    float relaxation = 0.8f;
    Iter::EPwlsRegularizer regularizer = Iter::EPwlsRegularizer::Quadratic;
    float regularization = 1e-3f;
    float huber_delta = 3e-3f;
    float epsilon = 1e-6f;
    float lower_bound = 0.f;
    float upper_bound = std::numeric_limits<float>::max();
    ECylPwlsDataModel data_model = ECylPwlsDataModel::JosephMatched;
    // 未提供逐射线数组时，每个投影元素都使用该值，默认严格为 1。
    // 提供数组时，有效权重为 projection_weight_scale * weights[i]。
    float projection_weight_scale = 1.f;
    // 可选设备指针。数组元素必须有限且非负；prepare() 返回后调用方可
    // 释放原数组，因为重建器已经完成 D2D 拷贝。
    const float* projection_weights = nullptr;
};

class CylPwlsReconstructor {
public:
    ~CylPwlsReconstructor() { release(); }
    CylPwlsReconstructor() = default;
    CylPwlsReconstructor(const CylPwlsReconstructor&) = delete;
    CylPwlsReconstructor& operator=(const CylPwlsReconstructor&) = delete;

    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const CylPwlsConfig& config, const Config& operator_config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || channels < 2 || rows < 2 || geometry.empty() ||
            volume.Nx <= 0 || volume.Ny <= 0 || volume.Nz <= 0 ||
            config.iterations <= 0 || config.relaxation <= 0.f ||
            config.regularization < 0.f || config.epsilon <= 0.f ||
            !(config.projection_weight_scale > 0.f) ||
            !std::isfinite(config.projection_weight_scale) ||
            config.lower_bound > config.upper_bound ||
            (config.regularizer == Iter::EPwlsRegularizer::Huber &&
                config.huber_delta <= 0.f)) return false;

        auto prepared = detail::prepareJosephGeometry(volume, channels, rows,
            geometry, device_id);
        if (!prepared) return false;
        if (!forward_.prepare(prepared, operator_config) ||
            !back_.prepare(prepared, operator_config)) return false;

        volume_ = volume; channels_ = channels; rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        volume_count_ = static_cast<size_t>(volume.Nx) * volume.Ny * volume.Nz;
        projection_count_ = static_cast<size_t>(channels) * rows * views_;
        config_ = config; stream_ = stream; device_id_ = device_id;
        weight_scale_ = config.projection_weight_scale;
        if (config.projection_weights) {
            d_projection_weights_ = memory_.allocateDevice3D<float>(channels,
                rows, views_, device_id);
            YK_CUDA_CHECK(cudaMemcpyAsync(d_projection_weights_.data(),
                config.projection_weights, projection_count_ * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_));
            weights_ = d_projection_weights_.data();
        }
        YK_LOGI("[CylPWLS] projection weights: {} scale={:.6g}",
            weights_ ? "per-ray" : "uniform", weight_scale_);

        d_forward_ = memory_.allocateDevice3D<float>(channels, rows, views_, device_id);
        d_residual_ = memory_.allocateDevice3D<float>(channels, rows, views_, device_id);
        d_weighted_forward_ = memory_.allocateDevice3D<float>(channels, rows,
            views_, device_id);
        d_gradient_ = memory_.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, device_id);
        d_curvature_ = memory_.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, device_id);
        d_ones_ = memory_.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, device_id);
        shared_volume_texture_ = Mem::TextureController::createEmptyTex3D(
            volume.Nx, volume.Ny, volume.Nz, cudaFilterModePoint,
            cudaAddressModeBorder);
        shared_projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels, rows, views_, cudaFilterModePoint, cudaAddressModeBorder);
        if (!d_forward_ || !d_residual_ || !d_weighted_forward_ ||
            !d_gradient_ || !d_curvature_ || !d_ones_) {
            release(); return false;
        }

        Iter::fill_ones_launch(d_ones_.data(), volume_count_, stream_);
        uploadVolume_(d_ones_.data());
        if (!forwardVolume_(d_forward_.data())) { release(); return false; }
        detail::launch_cyl_pwls_apply_weights(d_forward_.data(), weights_,
            weight_scale_, d_weighted_forward_.data(), projection_count_, stream_);
        if (!uploadProjection_(d_weighted_forward_.data()) ||
            !backproject_(d_curvature_.data())) { release(); return false; }
        Iter::clamp_min_launch(d_curvature_.data(), volume_count_,
            config_.epsilon, stream_);
        prepared_ = true;
        return true;
    }

    bool reconstruct(const float* measured_projection, float* volume)
    {
        if (!prepared_ || !measured_projection || !volume) return false;
        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            uploadVolume_(volume);
            if (!forwardVolume_(d_forward_.data())) return false;
            detail::launch_cyl_pwls_residual(measured_projection,
                d_forward_.data(), weights_, weight_scale_, d_residual_.data(),
                projection_count_, stream_);
            if (!uploadProjection_(d_residual_.data()) ||
                !backproject_(d_gradient_.data())) return false;
            Iter::parallel_pwls_update_launch(volume, d_gradient_.data(),
                d_curvature_.data(), volume_.Nx, volume_.Ny, volume_.Nz,
                config_.relaxation, 1.f,
                static_cast<int>(config_.regularizer), config_.regularization,
                config_.huber_delta, config_.epsilon, config_.lower_bound,
                config_.upper_bound, stream_);
        }
        return true;
    }

    void release()
    {
        if (stream_) YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        forward_.release(); back_.release();
        d_forward_ = {}; d_residual_ = {}; d_weighted_forward_ = {};
        d_gradient_ = {};
        d_curvature_ = {}; d_ones_ = {}; d_projection_weights_ = {};
        shared_volume_texture_ = {}; shared_projection_texture_ = {};
        volume_ = {}; channels_ = rows_ = views_ = 0;
        volume_count_ = projection_count_ = 0; weights_ = nullptr;
        weight_scale_ = 1.f;
        stream_ = nullptr; prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    void uploadVolume_(const float* volume)
    {
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_volume_texture_,
            volume, volume_.Nx, volume_.Ny, volume_.Nz, stream_);
    }

    bool forwardVolume_(float* projection)
    {
        return config_.data_model == ECylPwlsDataModel::Siddon
            ? forward_.forwardSiddon(shared_volume_texture_, projection, stream_)
            : forward_.forwardMatchedReference(shared_volume_texture_, projection,
                stream_);
    }

    bool backproject_(float* volume)
    {
        return config_.data_model == ECylPwlsDataModel::Siddon
            ? back_.backprojectSiddon(shared_projection_texture_, volume,
                stream_, false)
            : back_.backproject(shared_projection_texture_, volume,
                stream_, false);
    }

    bool uploadProjection_(const float* projection)
    {
        if (!projection || shared_projection_texture_.tex == 0) return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_projection_texture_,
            projection, channels_, rows_, views_, stream_);
        return true;
    }

    SVolGeom volume_{};
    int channels_ = 0, rows_ = 0, views_ = 0, device_id_ = 0;
    size_t volume_count_ = 0, projection_count_ = 0;
    CylPwlsConfig config_{};
    const float* weights_ = nullptr;
    float weight_scale_ = 1.f;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
    CylForwardOperator forward_{};
    CylBackOperator back_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_forward_{};
    Mem::DeviceLinearBuffer3D<float> d_residual_{};
    Mem::DeviceLinearBuffer3D<float> d_weighted_forward_{};
    Mem::DeviceLinearBuffer3D<float> d_gradient_{};
    Mem::DeviceLinearBuffer3D<float> d_curvature_{};
    Mem::DeviceLinearBuffer3D<float> d_ones_{};
    Mem::DeviceLinearBuffer3D<float> d_projection_weights_{};
    Mem::Tex3DHandle shared_volume_texture_{};
    Mem::Tex3DHandle shared_projection_texture_{};
};

} // namespace YK::CylFpBp
