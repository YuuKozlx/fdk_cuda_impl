#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "CylFpBp/YkCylBackOperator.hpp"
#include "CylFpBp/YkCylForwardOperator.hpp"
#include "CylFpBp/YkCylVoxelDrivenBackprojector.hpp"
#include "Iter/kernels/YkIterLaunch.cuh"
#include "global/YkCudaTextureController.hpp"
#include "global/YkLog.h"

namespace YK::CylFpBp {

// 圆柱探测器的统一迭代方法。SIRT/SART/OS-SART 使用相同的 R/C
// 归一化更新，仅由每个外循环的子集数量区分。CGLS 使用硬件 Linear 纹理
// FP 和浮点 Joseph BP；该组合因 FP 插值权重量化而不是严格 matched。
enum class EIterativeMethod {
    Sirt,
    Sart,
    Ossart,
    Cgls
};

struct IterativeConfig {
    EIterativeMethod method = EIterativeMethod::Ossart;
    int iterations = 10;
    int subset_count = 20; // 仅 OS-SART 使用；SIRT/SART 会自动覆盖。
    float relaxation = 0.2f;
    float relaxation_reduction = 1.f; // 每个完整外循环衰减一次。
    float epsilon = 1e-6f;
    bool nonnegative = true;
    bool use_max = false;
    float maximum = 1e30f;
    // 代数迭代使用纹理 Joseph FP；V3 BP 可关闭以执行浮点 Joseph BP 对照。
    bool use_voxel_driven_backprojector = true;
};

// 该类只依赖 Cyl 专有 IForwardOperator/IBackOperator，不经过平板几何适配器，
// 因而不会在迭代过程中丢失探测器曲率。所有工作区均由公共分配器持有。
class IterativeReconstructor {
public:
    ~IterativeReconstructor() { release(); }
    IterativeReconstructor() = default;
    IterativeReconstructor(const IterativeReconstructor&) = delete;
    IterativeReconstructor& operator=(const IterativeReconstructor&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const IterativeConfig& config, const Config& operator_config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || geometry.empty() || channels <= 0 || rows <= 0 ||
            config.iterations <= 0 || config.relaxation <= 0.f ||
            config.relaxation_reduction <= 0.f || config.epsilon <= 0.f)
            return false;

        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;
        operator_config_ = operator_config;
        volume_count_ = static_cast<size_t>(volume_geometry.Nx) *
            volume_geometry.Ny * volume_geometry.Nz;
        view_count_ = static_cast<size_t>(channels) * rows;

        if (config_.method == EIterativeMethod::Cgls) {
            const auto prepared_geometry = detail::prepareJosephGeometry(
                volume_geometry_, channels_, rows_, geometry, device_id_);
            full_forward_ = makeForwardOperator();
            full_back_ = makeBackOperator();
            if (!prepared_geometry ||
                !full_forward_->prepare(prepared_geometry, operator_config_) ||
                !full_back_->prepare(prepared_geometry, operator_config_)) {
                release();
                return false;
            }
            allocateCgls_();
            allocateSharedTextures_(views_);
        }
        else {
            int subsets = config_.subset_count;
            if (config_.method == EIterativeMethod::Sirt) subsets = 1;
            if (config_.method == EIterativeMethod::Sart) subsets = views_;
            if (subsets <= 0 || subsets > views_) return false;
            if (!buildSubsets_(geometry, subsets)) {
                release();
                return false;
            }
            allocateAlgebraic_();
            allocateSharedTextures_(maximum_subset_views_);
        }
        prepared_ = true;
        return true;
    }

    bool reconstruct(const float* measured_projection, float* volume)
    {
        if (!prepared_ || !measured_projection || !volume) return false;
        return config_.method == EIterativeMethod::Cgls
            ? reconstructCgls_(measured_projection, volume)
            : reconstructAlgebraic_(measured_projection, volume);
    }

    void release()
    {
        if (stream_) YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        subset_forward_operators_.clear();
        subset_back_operators_.clear();
        subset_v3_backprojectors_.clear();
        subsets_.clear();
        full_forward_.reset();
        full_back_.reset();
        shared_volume_texture_ = {};
        shared_projection_texture_ = {};
        d_measured_subset_ = {};
        d_forward_ = {};
        d_residual_ = {};
        d_row_weight_ = {};
        d_backprojection_ = {};
        d_column_weight_ = {};
        d_ones_volume_ = {};
        d_cgls_r_ = {};
        d_cgls_w_ = {};
        d_cgls_p_ = {};
        d_cgls_z_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

private:
    bool buildSubsets_(const std::vector<SCylConeProjGeomVec>& geometry,
        int subset_count)
    {
        subsets_.resize(subset_count);
        subset_forward_operators_.reserve(subset_count);
        subset_back_operators_.reserve(subset_count);
        maximum_subset_views_ = 0;
        for (int subset = 0; subset < subset_count; ++subset) {
            auto& indices = subsets_[subset];
            std::vector<SCylConeProjGeomVec> subset_geometry;
            for (int view = subset; view < views_; view += subset_count) {
                indices.push_back(view);
                subset_geometry.push_back(geometry[view]);
            }
            maximum_subset_views_ = std::max(maximum_subset_views_,
                static_cast<int>(indices.size()));
            const auto prepared_geometry = detail::prepareJosephGeometry(
                volume_geometry_, channels_, rows_, subset_geometry, device_id_);
            auto forward = makeForwardOperator();
            auto back = makeBackOperator();
            if (!prepared_geometry ||
                !forward->prepare(prepared_geometry, operator_config_) ||
                !back->prepare(prepared_geometry, operator_config_)) return false;
            subset_forward_operators_.push_back(std::move(forward));
            subset_back_operators_.push_back(std::move(back));
            if (config_.use_voxel_driven_backprojector) {
                auto v3 = std::make_unique<VoxelDrivenBackprojectorV3>();
                if (!v3->prepare(volume_geometry_, channels_, rows_,
                        subset_geometry, stream_, device_id_)) return false;
                subset_v3_backprojectors_.push_back(std::move(v3));
            }
        }
        return true;
    }

    bool forwardSubset_(int subset, const float* volume, float* projection)
    {
        auto& op = *subset_forward_operators_[subset];
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_volume_texture_,
            volume, volume_geometry_.Nx, volume_geometry_.Ny,
            volume_geometry_.Nz, stream_);
        return op.forward(shared_volume_texture_, projection, stream_, false);
    }

    bool backprojectSubset_(int subset, const float* projection, float* volume)
    {
        if (!config_.use_voxel_driven_backprojector)
        {
            Mem::TextureController::updateTex3DFromDeviceAsync(
                shared_projection_texture_, projection, channels_, rows_,
                static_cast<int>(subsets_[subset].size()), stream_);
            return subset_back_operators_[subset]->backproject(
                shared_projection_texture_, volume, stream_, false);
        }
        auto& v3 = *subset_v3_backprojectors_[subset];
        return v3.uploadProjection(projection) && v3.backproject(volume, false);
    }

    void allocateAlgebraic_()
    {
        const int maximum_projection_count = static_cast<int>(
            view_count_ * maximum_subset_views_);
        d_measured_subset_ = allocator_.allocate<float>(maximum_projection_count,
            device_id_);
        d_forward_ = allocator_.allocate<float>(maximum_projection_count, device_id_);
        d_residual_ = allocator_.allocate<float>(maximum_projection_count, device_id_);
        d_row_weight_ = allocator_.allocate<float>(maximum_projection_count, device_id_);
        d_backprojection_ = allocator_.allocate<float>(
            static_cast<int>(volume_count_), device_id_);
        d_column_weight_ = allocator_.allocate<float>(
            static_cast<int>(volume_count_), device_id_);
        d_ones_volume_ = allocator_.allocate<float>(
            static_cast<int>(volume_count_), device_id_);
    }

    void allocateCgls_()
    {
        const int projection_count = static_cast<int>(view_count_ * views_);
        d_cgls_r_ = allocator_.allocate<float>(projection_count, device_id_);
        d_cgls_w_ = allocator_.allocate<float>(projection_count, device_id_);
        d_cgls_p_ = allocator_.allocate<float>(static_cast<int>(volume_count_), device_id_);
        d_cgls_z_ = allocator_.allocate<float>(static_cast<int>(volume_count_), device_id_);
    }

    void allocateSharedTextures_(int projection_views)
    {
        // 体积使用 Linear 纹理；投影使用不改变离散样本的 Point 纹理。
        // FP 的硬件插值会量化小数权重，因此与浮点 Joseph BP 组成的是
        // 高伴随度组合，不宣称严格 matched。
        // 两个纹理在全部子集间复用，显存不会随 subset_count 增长。
        shared_volume_texture_ = Mem::TextureController::createEmptyTex3D(
            volume_geometry_.Nx, volume_geometry_.Ny, volume_geometry_.Nz,
            cudaFilterModeLinear, cudaAddressModeBorder);
        shared_projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels_, rows_, projection_views, cudaFilterModePoint,
            cudaAddressModeBorder);
    }

    bool gatherSubset_(const float* measured, int subset)
    {
        const auto& indices = subsets_[subset];
        for (size_t i = 0; i < indices.size(); ++i) {
            YK_CUDA_CHECK(cudaMemcpyAsync(d_measured_subset_.data() + i * view_count_,
                measured + static_cast<size_t>(indices[i]) * view_count_,
                view_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }
        return true;
    }

    bool reconstructAlgebraic_(const float* measured, float* volume)
    {
        Iter::fill_ones_launch(d_ones_volume_.data(), volume_count_, stream_);
        const int subset_count = static_cast<int>(subsets_.size());
        // SIRT 始终使用同一个完整数据子集，R/C 权重只与几何有关，可只算一次。
        // 多子集算法仍逐子集复用工作区，避免保存 subset_count 份体积列权重。
        if (subset_count == 1) {
            const size_t projection_count = view_count_ * subsets_.front().size();
            if (!forwardSubset_(0, d_ones_volume_.data(), d_row_weight_.data()))
                return false;
            Iter::fill_ones_launch(d_residual_.data(), projection_count, stream_);
            if (!backprojectSubset_(0, d_residual_.data(), d_column_weight_.data()))
                return false;
        }
        float relaxation = config_.relaxation;
        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            for (int subset = 0; subset < subset_count; ++subset) {
                const size_t projection_count = view_count_ * subsets_[subset].size();
                gatherSubset_(measured, subset);

                // R=A_s*1 与 C=B_s*1 使用当前圆柱子集。B_s 在 V3 模式下
                // 是近似 BP，不要求等于 A_s^T，但同一 B_s 同时参与列归一化
                // 和残差反投影，因此不会引入未经归一化的固定增益。
                if (subset_count != 1) {
                    if (!forwardSubset_(subset, d_ones_volume_.data(),
                            d_row_weight_.data())) return false;
                    Iter::fill_ones_launch(d_residual_.data(), projection_count, stream_);
                    if (!backprojectSubset_(subset, d_residual_.data(),
                            d_column_weight_.data())) return false;
                }

                if (!forwardSubset_(subset, volume, d_forward_.data())) return false;
                Iter::residual_launch(d_measured_subset_.data(), d_forward_.data(),
                    d_residual_.data(), projection_count, stream_);
                Iter::divide_launch(d_residual_.data(), d_row_weight_.data(),
                    config_.epsilon, projection_count, stream_);
                if (!backprojectSubset_(subset, d_residual_.data(),
                        d_backprojection_.data())) return false;
                Iter::update_launch(volume, d_backprojection_.data(),
                    d_column_weight_.data(), relaxation, config_.epsilon,
                    volume_count_, stream_);
                if (config_.nonnegative)
                    Iter::clamp_min_launch(volume, volume_count_, 0.f, stream_);
                if (config_.use_max)
                    Iter::clamp_max_launch(volume, volume_count_, config_.maximum,
                        stream_);
            }
            relaxation *= config_.relaxation_reduction;
            YK_LOGI("[CylIterative] outer {}/{} subsets={} lambda={:.6e}",
                iteration + 1, config_.iterations, subset_count, relaxation);
        }
        return true;
    }

    bool reconstructCgls_(const float* measured, float* volume)
    {
        const size_t projection_count = view_count_ * views_;
        auto& forward = *full_forward_;
        auto& back = *full_back_;
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_volume_texture_,
            volume, volume_geometry_.Nx, volume_geometry_.Ny,
            volume_geometry_.Nz, stream_);
        forward.forward(shared_volume_texture_, d_cgls_w_.data(), stream_, false);
        Iter::residual_launch(measured, d_cgls_w_.data(), d_cgls_r_.data(),
            projection_count, stream_);
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_projection_texture_,
            d_cgls_r_.data(), channels_, rows_, views_, stream_);
        back.backproject(shared_projection_texture_, d_cgls_p_.data(), stream_, false);
        float gamma = 0.f;
        Iter::dot_launch(d_cgls_p_.data(), d_cgls_p_.data(), volume_count_,
            &gamma, stream_);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream_));

        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            Mem::TextureController::updateTex3DFromDeviceAsync(shared_volume_texture_,
                d_cgls_p_.data(), volume_geometry_.Nx, volume_geometry_.Ny,
                volume_geometry_.Nz, stream_);
            forward.forward(shared_volume_texture_, d_cgls_w_.data(), stream_, false);
            float ww = 0.f;
            Iter::dot_launch(d_cgls_w_.data(), d_cgls_w_.data(), projection_count,
                &ww, stream_);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (!(ww > config_.epsilon) || !std::isfinite(ww)) break;
            const float alpha = gamma / ww;
            Iter::axpy_launch(volume, d_cgls_p_.data(), alpha, volume_count_, stream_);
            Iter::axpy_launch(d_cgls_r_.data(), d_cgls_w_.data(), -alpha,
                projection_count, stream_);
            Mem::TextureController::updateTex3DFromDeviceAsync(
                shared_projection_texture_, d_cgls_r_.data(), channels_, rows_,
                views_, stream_);
            back.backproject(shared_projection_texture_, d_cgls_z_.data(),
                stream_, false);
            float next_gamma = 0.f;
            Iter::dot_launch(d_cgls_z_.data(), d_cgls_z_.data(), volume_count_,
                &next_gamma, stream_);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (!std::isfinite(next_gamma)) return false;
            const float beta = next_gamma / std::max(gamma, config_.epsilon);
            Iter::scale_launch(d_cgls_p_.data(), beta, volume_count_, stream_);
            Iter::axpy_launch(d_cgls_p_.data(), d_cgls_z_.data(), 1.f,
                volume_count_, stream_);
            gamma = next_gamma;
            YK_LOGI("[CylIterative:CGLS] iter {}/{} gamma={:.6e}",
                iteration + 1, config_.iterations, gamma);
        }
        return true;
    }

    SVolGeom volume_geometry_{};
    Config operator_config_{};
    IterativeConfig config_{};
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int maximum_subset_views_ = 0;
    int device_id_ = 0;
    size_t volume_count_ = 0;
    size_t view_count_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;

    Mem::PodDataController allocator_{};
    std::vector<std::vector<int>> subsets_;
    std::vector<std::unique_ptr<IForwardOperator>> subset_forward_operators_;
    std::vector<std::unique_ptr<IBackOperator>> subset_back_operators_;
    std::vector<std::unique_ptr<VoxelDrivenBackprojectorV3>>
        subset_v3_backprojectors_;
    std::unique_ptr<IForwardOperator> full_forward_;
    std::unique_ptr<IBackOperator> full_back_;
    Mem::Tex3DHandle shared_volume_texture_{};
    Mem::Tex3DHandle shared_projection_texture_{};
    Mem::DeviceLinearBuffer<float> d_measured_subset_{};
    Mem::DeviceLinearBuffer<float> d_forward_{};
    Mem::DeviceLinearBuffer<float> d_residual_{};
    Mem::DeviceLinearBuffer<float> d_row_weight_{};
    Mem::DeviceLinearBuffer<float> d_backprojection_{};
    Mem::DeviceLinearBuffer<float> d_column_weight_{};
    Mem::DeviceLinearBuffer<float> d_ones_volume_{};
    Mem::DeviceLinearBuffer<float> d_cgls_r_{};
    Mem::DeviceLinearBuffer<float> d_cgls_w_{};
    Mem::DeviceLinearBuffer<float> d_cgls_p_{};
    Mem::DeviceLinearBuffer<float> d_cgls_z_{};
};

} // namespace YK::CylFpBp
