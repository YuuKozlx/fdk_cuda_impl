#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <limits>
#include <numeric>
#include <vector>

#include "CylFpBp/bp/YkCylBackOperator.hpp"
#include "CylFpBp/bp/YkCylBackProjection.hpp"
#include "CylFpBp/fp/YkCylForwardOperator.hpp"
#include "CylFpBp/iter/YkCylIterativeWeights.hpp"
#include "Iter/kernels/YkIterLaunch.cuh"
#include "Iter/YkIterativeConvergence.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkLog.h"

namespace YK::CylFpBp {

// 圆柱探测器的统一迭代方法。SIRT/SART/OS-SART 使用相同的 R/C
// 归一化更新，仅由每个外循环的子集数量区分。CGLS 默认可选快速的
// JosephV3 BP；该组合不是 FP 的严格转置，因此属于工程近似 CGLS。
enum class EIterativeMethod {
    Sirt,
    Sart,
    Ossart,
    Cgls
};

// 迭代器可选的圆柱算子。所有迭代方法都允许自由组合 FP/BP；FDK 风格 BP
// 不等于 A^T，因此配置和日志均明确将其标记为工程近似。
enum class EIterativeForwardModel {
    Joseph,
    Siddon,
    JosephMatchedReference
};

enum class EIterativeBackprojectorModel {
    Joseph,
    Siddon,
    SiddonV2,
    SiddonV3,
    SiddonRayDriven,
    // Cyl Joseph_V3：体素驱动实现，采用 Joseph 插值语义的几何权重。
    JosephV3,
    // FDK 风格解析反投影；允许所有迭代方法使用，但不是 A^T。
    Fdk,
    // FDK 风格的匹配权重版本；相对 FP 更接近，但仍非严格转置。
    FdkMatched
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
    EIterativeForwardModel forward_model = EIterativeForwardModel::Joseph;
    EIterativeBackprojectorModel backprojector_model =
        EIterativeBackprojectorModel::JosephV3;
    Iter::IterativeConvergenceConfig convergence{};
    IterativeWeightConfig weighting{};
};

// 该类只依赖 Cyl 专有 ICylForwardOperator/ICylBackOperator，不经过平板几何适配器，
// 因而不会在迭代过程中丢失探测器曲率。所有工作区均由公共分配器持有。
class AlgebraicReconstructor {
public:
    // 每次调用需要把 view_indices 指定的投影按相同顺序写入
    // destination。数据源必须支持跨迭代轮次重放，可来自主机内存、磁盘
    // 或采集缓存；回调中的异步操作必须提交到给定 stream。
    using ProjectionSubsetLoader = std::function<bool(int iteration,
        int subset, const std::vector<int>& view_indices, float* destination,
        cudaStream_t stream)>;

    ~AlgebraicReconstructor() { release(); }
    AlgebraicReconstructor() = default;
    AlgebraicReconstructor(const AlgebraicReconstructor&) = delete;
    AlgebraicReconstructor& operator=(const AlgebraicReconstructor&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const IterativeConfig& config, const Config& operator_config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || geometry.empty() || channels <= 0 || rows <= 0 ||
            config.iterations <= 0 || config.relaxation <= 0.f ||
            config.relaxation_reduction <= 0.f || config.epsilon <= 0.f ||
            !Iter::validConvergenceConfig(config.convergence))
            return false;
        volume_geometry_ = volume_geometry;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;
        resources_.attach(stream_, device_id_);
        if (config_.method == EIterativeMethod::Cgls) {
            const char* fp_name = config_.forward_model ==
                    EIterativeForwardModel::Joseph ? "Joseph" :
                config_.forward_model == EIterativeForwardModel::Siddon ?
                    "Siddon" : "JosephMatchedReference";
            const char* bp_name = config_.backprojector_model ==
                    EIterativeBackprojectorModel::JosephV3 ? "JosephV3" :
                config_.backprojector_model == EIterativeBackprojectorModel::Siddon ?
                    "SiddonRayDriven" : config_.backprojector_model ==
                    EIterativeBackprojectorModel::SiddonV2 ? "SiddonV2" :
                config_.backprojector_model == EIterativeBackprojectorModel::SiddonV3 ?
                    "SiddonV3" :
                config_.backprojector_model == EIterativeBackprojectorModel::SiddonRayDriven ?
                    "SiddonRayDriven" : config_.backprojector_model ==
                    EIterativeBackprojectorModel::Fdk ? "Fdk" :
                    config_.backprojector_model == EIterativeBackprojectorModel::FdkMatched ?
                    "FdkMatched" : "Joseph";
            YK_LOGI("[CylIterative:CGLS] FP={} BP={}；默认快速组合为 "
                "Joseph + JosephV3", fp_name, bp_name);
            if (config_.forward_model == EIterativeForwardModel::Siddon &&
                (config_.backprojector_model == EIterativeBackprojectorModel::Siddon ||
                 config_.backprojector_model == EIterativeBackprojectorModel::SiddonRayDriven)) {
                YK_LOGI("[CylIterative:CGLS] 当前为严格 Siddon 路径长度伴随组合，"
                    "但射线遍历和原子累加使其速度较慢");
            }
            else if (config_.forward_model ==
                    EIterativeForwardModel::JosephMatchedReference &&
                config_.backprojector_model == EIterativeBackprojectorModel::Joseph) {
                YK_LOGI("[CylIterative:CGLS] 当前为高伴随性 Joseph 参考组合，"
                    "但速度最慢（Point FP + Joseph BP）");
            }
            else if (config_.backprojector_model == EIterativeBackprojectorModel::Fdk ||
                     config_.backprojector_model == EIterativeBackprojectorModel::FdkMatched) {
                YK_LOGW("[CylIterative:CGLS] FDK BP 是解析重建权重，非 A^T；"
                    "当前按工程近似 CGLS 使用");
            }
            else {
                YK_LOGW("[CylIterative:CGLS] 当前组合为工程近似，"
                    "严格伴随组合是 Siddon + Siddon；高伴随性参考组合是 "
                    "JosephMatchedReference + Joseph，两者均较慢");
            }
        }
        operator_config_ = operator_config;
        volume_count_ = static_cast<size_t>(volume_geometry.Nx) *
            volume_geometry.Ny * volume_geometry.Nz;
        view_count_ = static_cast<size_t>(channels) * rows;
        const auto projection_weights = buildIterativeProjectionWeights(
            volume_geometry_, channels_, rows_, geometry, config_.weighting);
        if (projection_weights.size() != view_count_ * views_) {
            release();
            return false;
        }
        d_projection_weights_ = allocator_.allocateAndUpload(projection_weights,
            device_id_);

        if (config_.method == EIterativeMethod::Cgls) {
            const auto prepared_geometry = detail::prepareJosephGeometry(
                volume_geometry_, channels_, rows_, geometry, device_id_);
            full_forward_ = makeCylForwardOperator();
            if (!prepared_geometry ||
                !full_forward_->prepare(prepared_geometry, operator_config_)) {
                release();
                return false;
            }
            full_back_projection_ = makeBackProjection(backProjectionModel_());
            if (!full_back_projection_ || !full_back_projection_->prepare(
                    volume_geometry_, channels_, rows_, geometry,
                    operator_config_, resources_)) {
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
            if (needsProjectionConvergence_()) {
                const auto prepared_geometry = detail::prepareJosephGeometry(
                    volume_geometry_, channels_, rows_, geometry, device_id_);
                full_forward_ = makeCylForwardOperator();
                if (!prepared_geometry || !full_forward_->prepare(
                        prepared_geometry, operator_config_)) {
                    release();
                    return false;
                }
                d_convergence_projection_ = allocator_.allocate<float>(
                    static_cast<int>(view_count_ * views_), device_id_);
            }
        }
        if (config_.convergence.relative_update_tolerance > 0.f)
            d_previous_volume_ = allocator_.allocate<float>(
                static_cast<int>(volume_count_), device_id_);
        convergence_statistics_ = {};
        previous_residual_ = std::numeric_limits<float>::quiet_NaN();
        convergence_patience_ = 0;
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

    // 以调用方提供的设备体积作为初值。该入口可直接承接 Cyl FDK 的
    // 输出，也可承接外部上一阶段迭代结果；不在此处重新解释几何或复制
    // 主机数据，保证初值与重建体积使用同一分配器和 CUDA 上下文。
    bool reconstructWithInitialVolume(const float* measured_projection,
        const float* initial_volume, float* volume)
    {
        if (!prepared_ || !measured_projection || !initial_volume || !volume)
            return false;
        if (initial_volume != volume) {
            YK_CUDA_CHECK(cudaMemcpyAsync(volume, initial_volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }
        return reconstruct(measured_projection, volume);
    }

    // 迭代算法需要在每一轮重新访问投影，因此分包输入只负责异步汇集，
    // 不会像 FDK 那样在一次反投后丢弃该包。全部视图到齐后才能开始迭代。
    bool beginProjectionBatches()
    {
        if (!prepared_) return false;
        d_uploaded_projection_ = allocator_.allocate<float>(
            static_cast<int>(view_count_ * views_), device_id_);
        uploaded_views_.assign(static_cast<size_t>(views_), false);
        uploaded_view_count_ = 0;
        return static_cast<bool>(d_uploaded_projection_);
    }

    bool uploadProjectionBatch(const float* projection, int first_view,
        int view_count, bool source_is_device)
    {
        if (!prepared_ || !d_uploaded_projection_ || !projection ||
            first_view < 0 || view_count <= 0 || first_view + view_count > views_)
            return false;
        const size_t offset = static_cast<size_t>(first_view) * view_count_;
        const size_t count = static_cast<size_t>(view_count) * view_count_;
        YK_CUDA_CHECK(cudaMemcpyAsync(d_uploaded_projection_.data() + offset,
            projection, count * sizeof(float), source_is_device ?
                cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream_));
        for (int view = first_view; view < first_view + view_count; ++view) {
            if (!uploaded_views_[view]) {
                uploaded_views_[view] = true;
                ++uploaded_view_count_;
            }
        }
        return true;
    }

    bool reconstructUploaded(float* volume)
    {
        if (!prepared_ || !d_uploaded_projection_ ||
            uploaded_view_count_ != views_) return false;
        return reconstruct(d_uploaded_projection_.data(), volume);
    }

    // 真正的子集流式入口：GPU 只保存最大子集，不保存完整 measured
    // projection。SART/OS-SART 可直接使用；SIRT 的唯一子集本身就是完整
    // 投影，因此不会节省投影工作区。CGLS 需要不同的分批递推实现，当前
    // 明确拒绝，避免把完整缓存接口误称为流式 CGLS。
    bool reconstructStreaming(const ProjectionSubsetLoader& loader, float* volume)
    {
        if (!prepared_ || !loader || !volume ||
            config_.method == EIterativeMethod::Cgls) return false;
        return reconstructAlgebraic_(nullptr, volume, &loader);
    }

    const Iter::IterativeConvergenceStatistics& convergenceStatistics() const
    { return convergence_statistics_; }

    void release()
    {
        if (stream_) YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        subset_forward_operators_.clear();
        subset_back_projections_.clear();
        subsets_.clear();
        full_forward_.reset();
        full_back_projection_.reset();
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
        d_convergence_projection_ = {};
        d_previous_volume_ = {};
        d_uploaded_projection_ = {};
        d_projection_weights_ = {};
        d_subset_weights_ = {};
        d_weighted_projection_ = {};
        uploaded_views_.clear();
        uploaded_view_count_ = 0;
        resources_.release();
        stream_ = nullptr;
        prepared_ = false;
    }

private:
    bool buildSubsets_(const std::vector<SCylConeProjGeomVec>& geometry,
        int subset_count)
    {
        subsets_.resize(subset_count);
        subset_forward_operators_.reserve(subset_count);
        subset_back_projections_.reserve(subset_count);
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
            auto forward = makeCylForwardOperator();
            if (!prepared_geometry ||
                !forward->prepare(prepared_geometry, operator_config_)) return false;
            subset_forward_operators_.push_back(std::move(forward));
            auto back = makeBackProjection(backProjectionModel_());
            if (!back || !back->prepare(volume_geometry_, channels_, rows_,
                    subset_geometry, operator_config_, resources_)) return false;
            subset_back_projections_.push_back(std::move(back));
        }
        return true;
    }

    bool forwardSubset_(int subset, const float* volume, float* projection)
    {
        auto& op = *subset_forward_operators_[subset];
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_volume_texture_,
            volume, volume_geometry_.Nx, volume_geometry_.Ny,
            volume_geometry_.Nz, stream_);
        switch (config_.forward_model) {
        case EIterativeForwardModel::Joseph:
            return op.forward(shared_volume_texture_, projection, stream_, false);
        case EIterativeForwardModel::Siddon:
            return op.forwardSiddon(shared_volume_texture_, projection, stream_, false);
        case EIterativeForwardModel::JosephMatchedReference:
            return op.forwardMatchedReference(shared_volume_texture_, projection,
                stream_, false);
        }
        return false;
    }

    bool forwardFull_(const float* volume, float* projection)
    {
        if (!full_forward_) return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(shared_volume_texture_,
            volume, volume_geometry_.Nx, volume_geometry_.Ny,
            volume_geometry_.Nz, stream_);
        switch (config_.forward_model) {
        case EIterativeForwardModel::Joseph:
            return full_forward_->forward(shared_volume_texture_, projection,
                stream_, false);
        case EIterativeForwardModel::Siddon:
            return full_forward_->forwardSiddon(shared_volume_texture_, projection,
                stream_, false);
        case EIterativeForwardModel::JosephMatchedReference:
            return full_forward_->forwardMatchedReference(shared_volume_texture_,
                projection, stream_, false);
        }
        return false;
    }

    bool backprojectFull_(const float* projection, float* volume)
    {
        return full_back_projection_ && full_back_projection_->apply(projection,
            volume, false, resources_);
    }

    bool backprojectSubset_(int subset, const float* projection, float* volume)
    {
        return subset >= 0 && subset < static_cast<int>(subset_back_projections_.size()) &&
            subset_back_projections_[subset]->apply(projection, volume, false,
                resources_);
    }

    EBackProjection backProjectionModel_() const
    {
        switch (config_.backprojector_model) {
        case EIterativeBackprojectorModel::Joseph: return EBackProjection::Joseph;
        case EIterativeBackprojectorModel::Siddon: return EBackProjection::Siddon;
        case EIterativeBackprojectorModel::SiddonV2: return EBackProjection::SiddonV2;
        case EIterativeBackprojectorModel::SiddonV3: return EBackProjection::SiddonV3;
        case EIterativeBackprojectorModel::SiddonRayDriven:
            return EBackProjection::SiddonRayDriven;
        case EIterativeBackprojectorModel::JosephV3: return EBackProjection::JosephV3;
        case EIterativeBackprojectorModel::Fdk: return EBackProjection::Fdk;
        case EIterativeBackprojectorModel::FdkMatched:
            return EBackProjection::FdkMatched;
        }
        return EBackProjection::Joseph;
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
        d_subset_weights_ = allocator_.allocate<float>(maximum_projection_count,
            device_id_);
    }

    void allocateCgls_()
    {
        const int projection_count = static_cast<int>(view_count_ * views_);
        d_cgls_r_ = allocator_.allocate<float>(projection_count, device_id_);
        d_cgls_w_ = allocator_.allocate<float>(projection_count, device_id_);
        d_cgls_p_ = allocator_.allocate<float>(static_cast<int>(volume_count_), device_id_);
        d_cgls_z_ = allocator_.allocate<float>(static_cast<int>(volume_count_), device_id_);
        d_weighted_projection_ = allocator_.allocate<float>(projection_count,
            device_id_);
    }

    void allocateSharedTextures_(int projection_views)
    {
        // 体积使用 Linear 纹理；投影使用不改变离散样本的 Point 纹理。
        // FP 的硬件插值会量化小数权重，因此与浮点 Joseph BP 组成的是
        // 高伴随度组合，不宣称严格 matched。
        // 两个纹理在全部子集间复用，显存不会随 subset_count 增长。
        const auto volume_filter = config_.forward_model ==
            EIterativeForwardModel::Joseph ? cudaFilterModeLinear :
            cudaFilterModePoint;
        shared_volume_texture_ = Mem::TextureController::createEmptyTex3D(
            volume_geometry_.Nx, volume_geometry_.Ny, volume_geometry_.Nz,
            volume_filter, cudaAddressModeBorder);
        shared_projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels_, rows_, projection_views, cudaFilterModePoint,
            cudaAddressModeBorder);
    }

    bool needsProjectionConvergence_() const
    {
        return config_.convergence.relative_residual_tolerance > 0.f ||
            config_.convergence.relative_improvement_tolerance > 0.f;
    }

    float norm_(const float* data, size_t count) const
    {
        float squared = 0.f;
        Iter::dot_launch(data, data, count, &squared, stream_);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return std::sqrt(std::max(0.f, squared));
    }

    bool shouldCheckConvergence_(int completed_iterations) const
    {
        return Iter::convergenceEnabled(config_.convergence) &&
            (completed_iterations % config_.convergence.check_interval == 0 ||
             completed_iterations == config_.iterations);
    }

    bool checkConvergence_(const float* measured, float* volume,
        const float* known_residual = nullptr)
    {
        const auto& cfg = config_.convergence;
        auto& stats = convergence_statistics_;
        ++stats.convergence_checks;
        bool residual_satisfied = false;
        bool update_satisfied = false;
        bool stagnation_satisfied = false;

        if (needsProjectionConvergence_()) {
            const float* residual = known_residual;
            if (!residual) {
                if (!forwardFull_(volume, d_convergence_projection_.data()))
                    return false;
                Iter::residual_launch(measured, d_convergence_projection_.data(),
                    d_convergence_projection_.data(), view_count_ * views_, stream_);
                residual = d_convergence_projection_.data();
            }
            const float residual_norm = norm_(residual, view_count_ * views_);
            const float measured_norm = norm_(measured, view_count_ * views_);
            stats.projection_residual_l2 = residual_norm;
            stats.relative_projection_residual = residual_norm /
                std::max(measured_norm, config_.epsilon);
            residual_satisfied = cfg.relative_residual_tolerance > 0.f &&
                stats.relative_projection_residual <= cfg.relative_residual_tolerance;
            if (std::isfinite(previous_residual_)) {
                stats.relative_residual_improvement =
                    (previous_residual_ - residual_norm) /
                    std::max(previous_residual_, config_.epsilon);
                stagnation_satisfied = cfg.relative_improvement_tolerance > 0.f &&
                    stats.relative_residual_improvement >= 0.f &&
                    stats.relative_residual_improvement <=
                        cfg.relative_improvement_tolerance;
            }
            previous_residual_ = residual_norm;
        }
        if (config_.convergence.relative_update_tolerance > 0.f) {
            const float previous_norm = norm_(d_previous_volume_.data(), volume_count_);
            Iter::subtract_launch(d_previous_volume_.data(), volume,
                d_previous_volume_.data(), volume_count_, stream_);
            stats.relative_volume_update = norm_(d_previous_volume_.data(),
                volume_count_) / std::max(previous_norm, config_.epsilon);
            update_satisfied = stats.relative_volume_update <=
                cfg.relative_update_tolerance;
            YK_CUDA_CHECK(cudaMemcpyAsync(d_previous_volume_.data(), volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }
        YK_LOGI("[CylIterative] iter={} relative-residual={:.6e} "
            "relative-update={:.6e}", stats.completed_iterations,
            stats.relative_projection_residual, stats.relative_volume_update);
        const bool satisfied = stats.completed_iterations >= cfg.minimum_iterations &&
            (residual_satisfied || update_satisfied || stagnation_satisfied);
        convergence_patience_ = satisfied ? convergence_patience_ + 1 : 0;
        if (convergence_patience_ < cfg.patience) return false;
        stats.stopped_by_relative_residual = residual_satisfied;
        stats.stopped_by_relative_update = update_satisfied;
        stats.stopped_by_stagnation = stagnation_satisfied;
        return true;
    }

    bool gatherSubset_(const float* measured, int subset)
    {
        const auto& indices = subsets_[subset];
        for (size_t i = 0; i < indices.size(); ++i) {
            if (measured) {
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_measured_subset_.data() + i * view_count_,
                    measured + static_cast<size_t>(indices[i]) * view_count_,
                    view_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            }
            YK_CUDA_CHECK(cudaMemcpyAsync(d_subset_weights_.data() + i * view_count_,
                d_projection_weights_.data() +
                    static_cast<size_t>(indices[i]) * view_count_,
                view_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }
        return true;
    }

    bool reconstructAlgebraic_(const float* measured, float* volume,
        const ProjectionSubsetLoader* loader = nullptr)
    {
        if (d_previous_volume_)
            YK_CUDA_CHECK(cudaMemcpyAsync(d_previous_volume_.data(), volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        Iter::fill_ones_launch(d_ones_volume_.data(), volume_count_, stream_);
        const int subset_count = static_cast<int>(subsets_.size());
        // SIRT 始终使用同一个完整数据子集，R/C 权重只与几何有关，可只算一次。
        // 多子集算法仍逐子集复用工作区，避免保存 subset_count 份体积列权重。
        if (subset_count == 1) {
            const size_t projection_count = view_count_ * subsets_.front().size();
            gatherSubset_(nullptr, 0);
            if (!forwardSubset_(0, d_ones_volume_.data(), d_row_weight_.data()))
                return false;
            // C = B(W*1)，列归一化使用加权的单位投影。
            Iter::fill_ones_launch(d_residual_.data(), projection_count, stream_);
            Iter::mul_launch(d_residual_.data(), d_subset_weights_.data(),
                projection_count, stream_);
            if (!backprojectSubset_(0, d_residual_.data(), d_column_weight_.data()))
                return false;
        }
        float relaxation = config_.relaxation;
        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            for (int subset = 0; subset < subset_count; ++subset) {
                const size_t projection_count = view_count_ * subsets_[subset].size();
                if (loader) {
                    if (!(*loader)(iteration, subset, subsets_[subset],
                            d_measured_subset_.data(), stream_)) return false;
                    // 权重与投影采用完全相同的视图顺序。
                    gatherSubset_(nullptr, subset);
                }
                else {
                    gatherSubset_(measured, subset);
                }

                // R=A_s*1 与 C=B_s*1 使用当前圆柱子集。B_s 在 V3 模式下
                // 是近似 BP，不要求等于 A_s^T，但同一 B_s 同时参与列归一化
                // 和残差反投影，因此不会引入未经归一化的固定增益。
                if (subset_count != 1) {
                    if (!forwardSubset_(subset, d_ones_volume_.data(),
                            d_row_weight_.data())) return false;
                    Iter::fill_ones_launch(d_residual_.data(), projection_count, stream_);
                    Iter::mul_launch(d_residual_.data(), d_subset_weights_.data(),
                        projection_count, stream_);
                    if (!backprojectSubset_(subset, d_residual_.data(),
                            d_column_weight_.data())) return false;
                }

                if (!forwardSubset_(subset, volume, d_forward_.data())) return false;
                Iter::residual_launch(d_measured_subset_.data(), d_forward_.data(),
                    d_residual_.data(), projection_count, stream_);
                Iter::divide_launch(d_residual_.data(), d_row_weight_.data(),
                    config_.epsilon, projection_count, stream_);
                Iter::mul_launch(d_residual_.data(), d_subset_weights_.data(),
                    projection_count, stream_);
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
            convergence_statistics_.completed_iterations = iteration + 1;
            // 流式数据源没有完整 measured 指针；收敛残差需要另一次全数据
            // 重放，当前只对常驻投影启用。
            if (!loader && shouldCheckConvergence_(iteration + 1) &&
                checkConvergence_(measured, volume))
                break;
            YK_LOGI("[CylIterative] outer {}/{} subsets={} lambda={:.6e}",
                iteration + 1, config_.iterations, subset_count, relaxation);
        }
        return true;
    }

    bool reconstructCgls_(const float* measured, float* volume)
    {
        if (d_previous_volume_)
            YK_CUDA_CHECK(cudaMemcpyAsync(d_previous_volume_.data(), volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        const size_t projection_count = view_count_ * views_;
        if (!forwardFull_(volume, d_cgls_w_.data())) return false;
        Iter::residual_launch(measured, d_cgls_w_.data(), d_cgls_r_.data(),
            projection_count, stream_);
        YK_CUDA_CHECK(cudaMemcpyAsync(d_weighted_projection_.data(),
            d_cgls_r_.data(), projection_count * sizeof(float),
            cudaMemcpyDeviceToDevice, stream_));
        Iter::mul_launch(d_weighted_projection_.data(), d_projection_weights_.data(),
            projection_count, stream_);
        if (!backprojectFull_(d_weighted_projection_.data(), d_cgls_p_.data()))
            return false;
        float gamma = 0.f;
        Iter::dot_launch(d_cgls_p_.data(), d_cgls_p_.data(), volume_count_,
            &gamma, stream_);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream_));

        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            if (!forwardFull_(d_cgls_p_.data(), d_cgls_w_.data())) return false;
            float ww = 0.f;
            YK_CUDA_CHECK(cudaMemcpyAsync(d_weighted_projection_.data(),
                d_cgls_w_.data(), projection_count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_));
            Iter::mul_launch(d_weighted_projection_.data(),
                d_projection_weights_.data(), projection_count, stream_);
            Iter::dot_launch(d_cgls_w_.data(), d_weighted_projection_.data(),
                projection_count, &ww, stream_);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            if (!(ww > config_.epsilon) || !std::isfinite(ww)) break;
            const float alpha = gamma / ww;
            Iter::axpy_launch(volume, d_cgls_p_.data(), alpha, volume_count_, stream_);
            Iter::axpy_launch(d_cgls_r_.data(), d_cgls_w_.data(), -alpha,
                projection_count, stream_);
            YK_CUDA_CHECK(cudaMemcpyAsync(d_weighted_projection_.data(),
                d_cgls_r_.data(), projection_count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_));
            Iter::mul_launch(d_weighted_projection_.data(),
                d_projection_weights_.data(), projection_count, stream_);
            if (!backprojectFull_(d_weighted_projection_.data(), d_cgls_z_.data()))
                return false;
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
            convergence_statistics_.completed_iterations = iteration + 1;
            if (shouldCheckConvergence_(iteration + 1) &&
                checkConvergence_(measured, volume, d_cgls_r_.data()))
                break;
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
    std::vector<std::unique_ptr<ICylForwardOperator>> subset_forward_operators_;
    std::vector<std::unique_ptr<IBackProjection>> subset_back_projections_;
    std::unique_ptr<ICylForwardOperator> full_forward_;
    std::unique_ptr<IBackProjection> full_back_projection_;
    ResourceContext resources_{};
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
    Mem::DeviceLinearBuffer<float> d_convergence_projection_{};
    Mem::DeviceLinearBuffer<float> d_previous_volume_{};
    Mem::DeviceLinearBuffer<float> d_uploaded_projection_{};
    Mem::DeviceLinearBuffer<float> d_projection_weights_{};
    Mem::DeviceLinearBuffer<float> d_subset_weights_{};
    Mem::DeviceLinearBuffer<float> d_weighted_projection_{};
    std::vector<bool> uploaded_views_{};
    int uploaded_view_count_ = 0;
    Iter::IterativeConvergenceStatistics convergence_statistics_{};
    float previous_residual_ = std::numeric_limits<float>::quiet_NaN();
    int convergence_patience_ = 0;
};

} // namespace YK::CylFpBp



