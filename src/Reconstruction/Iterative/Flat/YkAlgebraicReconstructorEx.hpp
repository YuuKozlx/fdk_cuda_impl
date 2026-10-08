#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "Reconstruction/Iterative/Flat/YkAlgebraicBackends.hpp"
#include "Reconstruction/Iterative/Flat/YkAlgebraicRegularizers.hpp"
#include "Reconstruction/Iterative/Common/YkIterativeConvergence.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace YK::Iter {

// 统一代数重建模式。三种经典方法由每轮使用的子集数量统一表达：
//   SIRT   : 1 个子集（每次使用全部视角）；
//   SART   : Na 个子集（每次使用单个视角）；
//   OSSART : 介于两者之间的交错子集。
// OssartTigre 保留 TIGRE 风格的连续分块及粗网格行/列归一化后端。
enum class EAlgebraicMethod : int {
    Sirt = 0,
    Sart = 1,
    Ossart = 2,
    // 仅为旧源码兼容保留；新代码请使用 Ossart + TigreApprox。
    OssartTigre = 3
};

enum class EAlgebraicWeightModel : int {
    DetailedSubset = 0, // 每个子集实时计算 R=A_s*1、C=A_s^T*1
    TigreApprox = 1     // 2x2x2、1.1/0.9 的 TIGRE 近似权重
};

enum class EAlgebraicSubsetOrder : int {
    Sequential = 0,
    GoldenRatio = 1
};

struct AlgebraicReconstructionConfig {
    EAlgebraicMethod method = EAlgebraicMethod::Ossart;
    EAlgebraicWeightModel weight_model = EAlgebraicWeightModel::DetailedSubset;
    int iterations = 10;       // 完整遍历全部子集的外循环数
    int subset_count = 10;     // 仅 Ossart/OssartTigre 使用
    // Detailed OSSART 的子集遍历顺序；TIGRE 模式固定为连续分块。
    EAlgebraicSubsetOrder subset_order = EAlgebraicSubsetOrder::Sequential;
    float relaxation = 1.f;
    // 统一定义为每完成一个外循环后衰减一次，而不是每个子集衰减。
    float relaxation_reduction = 1.f;
    float epsilon = 1e-6f;
    // 正则化在每轮完整子集扫描后执行；None 保持原代数重建行为。
    AlgebraicRegularizationConfig regularization{};
    IterativeConvergenceConfig convergence{};
    bool use_min = false;
    float min_constraint = 0.f;
    bool use_max = false;
    float max_constraint = 1e30f;
    ETask fp_task = ETask::FP_Joseph;
    ETask bp_task = ETask::BP_Joseph_v3;
};

// 显式 geometry 版本的统一代数重建器。
//
// SIRT/SART/OSSART 共用 OSSARTEx 的任意 cone-vector geometry 数据流，
// 只改变子集数，因此其投影、归一化、约束和生命周期完全一致。
// TIGRE 风格后端目前仍只支持 SReconstructionParams 可表达的圆轨迹；prepare() 会
// 校验传入 geometry 与该圆轨迹一致，拒绝静默丢弃任意几何信息。
class AlgebraicReconstructorEx {
public:
    using Config = AlgebraicReconstructionConfig;
    ~AlgebraicReconstructorEx() { release(); }
    AlgebraicReconstructorEx() = default;
    AlgebraicReconstructorEx(const AlgebraicReconstructorEx&) = delete;
    AlgebraicReconstructorEx& operator=(const AlgebraicReconstructorEx&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const AlgebraicReconstructionConfig& config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(params, geometry, config, stream)) return false;
        params_ = params;
        config_ = config;
        stream_ = stream;
        geometry_ = geometry;
        subset_count_ = resolvedSubsetCount_(params, config);

        use_tigre_weights_ = config.weight_model == EAlgebraicWeightModel::TigreApprox ||
            config.method == EAlgebraicMethod::OssartTigre;
        if (use_tigre_weights_) {
            AlgebraicTigreBackend::Config legacy{};
            legacy.n_iter = config.iterations;
            legacy.n_subset = subset_count_;
            legacy.lambda = config.relaxation;
            legacy.lambda_red = config.relaxation_reduction;
            legacy.eps = config.epsilon;
            legacy.use_min = config.use_min;
            legacy.min_constraint = config.min_constraint;
            legacy.use_max = config.use_max;
            legacy.max_constraint = config.max_constraint;
            legacy.fp_task = config.fp_task;
            legacy.bp_task = config.bp_task;
            prepared_ = tigre_.init(params_, legacy, geometry, stream_, device_id);
            if (prepared_) actual_subset_count_ = tigre_.actualSubsetCount();
        }
        else {
            AlgebraicDetailedWeightBackend::Config ex{};
            ex.n_iter = config.iterations;
            ex.n_subset = subset_count_;
            ex.lambda = config.relaxation;
            // OSSARTEx 在每个子集后衰减。开 subset_count 次方根，使公开
            // 配置的 reduction 保持“每个完整外循环一次”的统一语义。
            ex.lambda_red = std::pow(config.relaxation_reduction,
                1.f / static_cast<float>(subset_count_));
            ex.eps = config.epsilon;
            ex.use_min = config.use_min;
            ex.min_constraint = config.min_constraint;
            ex.use_max = config.use_max;
            ex.max_constraint = config.max_constraint;
            ex.fp_task = config.fp_task;
            ex.bp_task = config.bp_task;
            ex.golden_subset_order =
                config.subset_order == EAlgebraicSubsetOrder::GoldenRatio;
            prepared_ = detailed_weights_.init(
                params_, ex, geometry, stream_, device_id);
            if (prepared_)
                actual_subset_count_ = detailed_weights_.actualSubsetCount();
        }
        if (prepared_)
            prepared_ = regularizer_.prepare(params_, config.regularization,
                stream_, device_id);
        if (prepared_ && convergenceEnabled(config_.convergence))
            prepared_ = prepareConvergence_(geometry, device_id);
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* d_measured_projection, float* d_volume)
    {
        if (!prepared_ || !d_measured_projection || !d_volume) return false;
        convergence_statistics_ = {};
        convergence_patience_ = 0;
        previous_residual_ = std::numeric_limits<float>::quiet_NaN();
        if (d_previous_volume_) {
            YK_CUDA_CHECK(cudaMemcpyAsync(d_previous_volume_.data(), d_volume,
                volumeCount_() * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }
        for (int outer = 0; outer < config_.iterations; ++outer) {
            if (!iterateDataSubsets_(d_measured_projection, d_volume,
                    static_cast<unsigned int>(actual_subset_count_)))
                return false;

            AlgebraicRegularizationContext context{};
            context.outer_iteration = outer;
            context.completed_subset_updates = actual_subset_count_;
            context.data_relaxation = config_.relaxation *
                std::pow(config_.relaxation_reduction,
                    static_cast<float>(outer));
            if (!regularizer_.apply(d_volume, context)) return false;
            applyConstraints_(d_volume);
            convergence_statistics_.completed_iterations = outer + 1;
            if (shouldCheckConvergence_(outer + 1) &&
                checkConvergence_(d_measured_projection, d_volume))
                break;
        }
        return true;
    }

    // 兼容旧类的增量接口：iterations 表示子集级更新次数。
    bool iterateSubsetUpdates(const float* d_measured_projection,
        float* d_volume, unsigned int iterations)
    {
        if (!prepared_ || !d_measured_projection || !d_volume) return false;
        // 增量接口保持“纯子集数据更新”语义，不隐式触发一次不完整的
        // 正则化周期。需要正则化时使用 reconstruct() 的完整外循环。
        return iterateDataSubsets_(d_measured_projection, d_volume, iterations);
    }

    void reset()
    {
        if (use_tigre_weights_) tigre_.reset();
        else detailed_weights_.reset();
        regularizer_.reset();
    }

    void release()
    {
        // reconstruct()/iterate() 保持异步；只有销毁或重新 prepare 时才等待，
        // 防止下方工作区在最后一批 kernel 尚未完成时被释放。
        if (stream_)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        detailed_weights_.release();
        tigre_.release();
        regularizer_.release();
        convergence_fp_.release();
        d_projection_residual_ = {};
        d_previous_volume_ = {};
        params_ = {};
        config_ = {};
        stream_ = nullptr;
        geometry_.clear();
        subset_count_ = 0;
        actual_subset_count_ = 0;
        use_tigre_weights_ = false;
        prepared_ = false;
        convergence_statistics_ = {};
    }

    bool isPrepared() const { return prepared_; }
    int subsetCount() const { return subset_count_; }
    int actualSubsetCount() const { return actual_subset_count_; }
    unsigned int totalSubsetUpdates() const
    {
        if (use_tigre_weights_)
            return tigre_.totalIterations();
        return detailed_weights_.totalIterations();
    }
    const IterativeConvergenceStatistics& convergenceStatistics() const
    { return convergence_statistics_; }

private:
    size_t volumeCount_() const
    {
        return static_cast<size_t>(params_.volume.Nx) * params_.volume.Ny * params_.volume.Nz;
    }

    size_t projectionCount_() const
    {
        return static_cast<size_t>(params_.scan.NAng) * params_.scan.Nu * params_.scan.Nv;
    }

    bool prepareConvergence_(const std::vector<SConeProjGeomVec>& geometry,
        int device_id)
    {
        const auto& convergence = config_.convergence;
        if (convergence.relative_residual_tolerance > 0.f ||
            convergence.relative_improvement_tolerance > 0.f) {
            d_projection_residual_ = memory_.allocateDevice3D<float>(
                params_.scan.Nu, params_.scan.Nv, params_.scan.NAng,
                device_id, false);
            if (!d_projection_residual_) return false;
            if (!convergence_fp_.init(params_, geometry, config_.fp_task,
                    device_id, stream_)) return false;
        }
        if (convergence.relative_update_tolerance > 0.f) {
            d_previous_volume_ = memory_.allocateDevice3D<float>(
                params_.volume.Nx, params_.volume.Ny, params_.volume.Nz,
                device_id, false);
            if (!d_previous_volume_) return false;
        }
        return true;
    }

    bool shouldCheckConvergence_(int completed_iterations) const
    {
        if (!convergenceEnabled(config_.convergence)) return false;
        return completed_iterations % config_.convergence.check_interval == 0 ||
            completed_iterations == config_.iterations;
    }

    float norm_(const float* data, size_t count) const
    {
        float squared = 0.f;
        YK::Iter::dot_launch(data, data, count, &squared, stream_);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return std::sqrt(std::max(0.f, squared));
    }

    bool checkConvergence_(const float* measured, float* volume)
    {
        const auto& cfg = config_.convergence;
        auto& stats = convergence_statistics_;
        ++stats.convergence_checks;
        bool residual_satisfied = false;
        bool update_satisfied = false;
        bool stagnation_satisfied = false;

        if (d_projection_residual_) {
            YK_CUDA_CHECK(cudaMemsetAsync(d_projection_residual_.data(), 0,
                projectionCount_() * sizeof(float), stream_));
            if (!convergence_fp_.run(volume, params_, d_projection_residual_.data(), stream_))
                return false;
            YK::Iter::residual_launch(measured, d_projection_residual_.data(),
                d_projection_residual_.data(), projectionCount_(), stream_);
            const float residual = norm_(d_projection_residual_.data(), projectionCount_());
            const float measured_norm = norm_(measured, projectionCount_());
            stats.projection_residual_l2 = residual;
            stats.relative_projection_residual = residual /
                std::max(measured_norm, config_.epsilon);
            YK_LOGI("[AlgebraicReconstructorEx] 第 {} 轮相对投影残差 {:.6e}",
                stats.completed_iterations, stats.relative_projection_residual);
            residual_satisfied = cfg.relative_residual_tolerance > 0.f &&
                stats.relative_projection_residual <= cfg.relative_residual_tolerance;
            if (std::isfinite(previous_residual_)) {
                stats.relative_residual_improvement =
                    (previous_residual_ - residual) /
                    std::max(previous_residual_, config_.epsilon);
                stagnation_satisfied = cfg.relative_improvement_tolerance > 0.f &&
                    stats.relative_residual_improvement >= 0.f &&
                    stats.relative_residual_improvement <=
                        cfg.relative_improvement_tolerance;
            }
            previous_residual_ = residual;
        }

        if (d_previous_volume_) {
            const float previous_norm = norm_(d_previous_volume_.data(), volumeCount_());
            YK::Iter::residual_launch(volume, d_previous_volume_.data(),
                d_previous_volume_.data(), volumeCount_(), stream_);
            stats.relative_volume_update = norm_(d_previous_volume_.data(), volumeCount_()) /
                std::max(previous_norm, config_.epsilon);
            update_satisfied = stats.relative_volume_update <=
                cfg.relative_update_tolerance;
            YK_CUDA_CHECK(cudaMemcpyAsync(d_previous_volume_.data(), volume,
                volumeCount_() * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        }

        const bool eligible = stats.completed_iterations >= cfg.minimum_iterations;
        const bool satisfied = eligible &&
            (residual_satisfied || update_satisfied || stagnation_satisfied);
        convergence_patience_ = satisfied ? convergence_patience_ + 1 : 0;
        if (convergence_patience_ < cfg.patience) return false;
        stats.stopped_by_relative_residual = residual_satisfied;
        stats.stopped_by_relative_update = update_satisfied;
        stats.stopped_by_stagnation = stagnation_satisfied;
        YK_LOGI("[AlgebraicReconstructorEx] 第 {} 轮满足收敛条件，提前停止",
            stats.completed_iterations);
        return true;
    }

    bool iterateDataSubsets_(const float* d_measured_projection,
        float* d_volume, unsigned int iterations)
    {
        if (use_tigre_weights_)
            return tigre_.iterate(d_measured_projection, d_volume, params_,
                stream_, iterations);
        return detailed_weights_.iterate(
            d_measured_projection, d_volume, stream_, iterations);
    }

    void applyConstraints_(float* d_volume)
    {
        const size_t count = static_cast<size_t>(params_.volume.Nx) *
            params_.volume.Ny * params_.volume.Nz;
        if (config_.use_min)
            YK::Iter::clamp_min_launch(
                d_volume, count, config_.min_constraint, stream_);
        if (config_.use_max)
            YK::Iter::clamp_max_launch(
                d_volume, count, config_.max_constraint, stream_);
    }

    static int resolvedSubsetCount_(const SReconstructionParams& params,
        const AlgebraicReconstructionConfig& config)
    {
        if (config.method == EAlgebraicMethod::Sirt) return 1;
        if (config.method == EAlgebraicMethod::Sart) return params.scan.NAng;
        return config.subset_count;
    }

    static bool validate_(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const AlgebraicReconstructionConfig& config, cudaStream_t stream)
    {
        const int subsets = resolvedSubsetCount_(params, config);
        if (!stream || params.scan.NAng <= 0 || params.scan.totalViews < params.scan.NAng ||
            static_cast<int>(geometry.size()) != params.scan.NAng ||
            config.iterations <= 0 || subsets <= 0 || subsets > params.scan.NAng ||
            config.relaxation <= 0.f || config.relaxation_reduction <= 0.f ||
            config.epsilon <= 0.f || config.min_constraint > config.max_constraint ||
            !validConvergenceConfig(config.convergence)) {
            YK_LOGE("[AlgebraicReconstructorEx] 无效配置");
            return false;
        }
        const auto& regularization = config.regularization;
        if (regularization.type != EAlgebraicRegularizer::None &&
            (regularization.strength <= 0.f ||
             regularization.inner_iterations <= 0 ||
             regularization.epsilon <= 0.f ||
             regularization.strength_reduction <= 0.f ||
             params.volume.voxelX_mm <= 0.f || params.volume.voxelY_mm <= 0.f ||
             params.volume.voxelZ_mm <= 0.f)) {
            YK_LOGE("[AlgebraicReconstructorEx] 无效正则化配置");
            return false;
        }
        return true;
    }

    SReconstructionParams params_{};
    AlgebraicReconstructionConfig config_{};
    cudaStream_t stream_ = nullptr;
    std::vector<SConeProjGeomVec> geometry_{};
    int subset_count_ = 0;
    int actual_subset_count_ = 0;
    bool prepared_ = false;
    bool use_tigre_weights_ = false;
    AlgebraicDetailedWeightBackend detailed_weights_{};
    AlgebraicTigreBackend tigre_{};
    AlgebraicRegularizer regularizer_{};
    ForwardOperatorAdapter convergence_fp_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_projection_residual_{};
    Mem::DeviceLinearBuffer3D<float> d_previous_volume_{};
    float previous_residual_ = std::numeric_limits<float>::quiet_NaN();
    int convergence_patience_ = 0;
    IterativeConvergenceStatistics convergence_statistics_{};
};

// 标准圆轨迹便捷入口。它只负责由 SReconstructionParams 生成逐视角 geometry，实际
// 重建仍由 Ex 类完成，因此普通版与显式几何版不会形成两套算法实现。
class AlgebraicReconstructor {
public:
    using Config = AlgebraicReconstructionConfig;

    bool prepare(const SReconstructionParams& params, const Config& config,
        cudaStream_t stream, int device_id = 0)
    {
        std::vector<SConeProjGeomVec> geometry;
        detail::buildCircularViews(params, geometry);
        return implementation_.prepare(params, geometry, config, stream, device_id);
    }

    bool reconstruct(const float* d_measured_projection, float* d_volume)
    {
        return implementation_.reconstruct(d_measured_projection, d_volume);
    }

    bool iterateSubsetUpdates(const float* d_measured_projection,
        float* d_volume, unsigned int iterations)
    {
        return implementation_.iterateSubsetUpdates(
            d_measured_projection, d_volume, iterations);
    }

    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }
    bool isPrepared() const { return implementation_.isPrepared(); }
    int subsetCount() const { return implementation_.subsetCount(); }
    int actualSubsetCount() const { return implementation_.actualSubsetCount(); }
    unsigned int totalSubsetUpdates() const
    { return implementation_.totalSubsetUpdates(); }
    const IterativeConvergenceStatistics& convergenceStatistics() const
    { return implementation_.convergenceStatistics(); }

private:
    AlgebraicReconstructorEx implementation_{};
};

} // namespace YK::Iter
