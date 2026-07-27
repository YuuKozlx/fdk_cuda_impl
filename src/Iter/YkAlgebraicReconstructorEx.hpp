#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "Iter/YkAlgebraicBackends.hpp"
#include "Iter/YkAlgebraicRegularizers.hpp"
#include "global/YkLog.h"

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
// TIGRE 风格后端目前仍只支持 SCBCTParams 可表达的圆轨迹；prepare() 会
// 校验传入 geometry 与该圆轨迹一致，拒绝静默丢弃任意几何信息。
class AlgebraicReconstructorEx {
public:
    using Config = AlgebraicReconstructionConfig;
    ~AlgebraicReconstructorEx() { release(); }
    AlgebraicReconstructorEx() = default;
    AlgebraicReconstructorEx(const AlgebraicReconstructorEx&) = delete;
    AlgebraicReconstructorEx& operator=(const AlgebraicReconstructorEx&) = delete;

    bool prepare(const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const AlgebraicReconstructionConfig& config,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(params, geometry, config, stream)) return false;
        params_ = params;
        config_ = config;
        stream_ = stream;
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
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* d_measured_projection, float* d_volume)
    {
        if (!prepared_ || !d_measured_projection || !d_volume) return false;
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
        detailed_weights_.release();
        tigre_.release();
        regularizer_.release();
        params_ = {};
        config_ = {};
        stream_ = nullptr;
        subset_count_ = 0;
        actual_subset_count_ = 0;
        use_tigre_weights_ = false;
        prepared_ = false;
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

private:
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
        const size_t count = static_cast<size_t>(params_.iVX) *
            params_.iVY * params_.iVZ;
        if (config_.use_min)
            YK::Iter::clamp_min_launch(
                d_volume, count, config_.min_constraint, stream_);
        if (config_.use_max)
            YK::Iter::clamp_max_launch(
                d_volume, count, config_.max_constraint, stream_);
    }

    static int resolvedSubsetCount_(const SCBCTParams& params,
        const AlgebraicReconstructionConfig& config)
    {
        if (config.method == EAlgebraicMethod::Sirt) return 1;
        if (config.method == EAlgebraicMethod::Sart) return params.iPAng;
        return config.subset_count;
    }

    static bool validate_(const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const AlgebraicReconstructionConfig& config, cudaStream_t stream)
    {
        const int subsets = resolvedSubsetCount_(params, config);
        if (!stream || params.iPAng <= 0 || params.iPAngTotal < params.iPAng ||
            static_cast<int>(geometry.size()) != params.iPAng ||
            config.iterations <= 0 || subsets <= 0 || subsets > params.iPAng ||
            config.relaxation <= 0.f || config.relaxation_reduction <= 0.f ||
            config.epsilon <= 0.f || config.min_constraint > config.max_constraint) {
            YK_LOGE("[AlgebraicReconstructorEx] 无效配置");
            return false;
        }
        const auto& regularization = config.regularization;
        if (regularization.type != EAlgebraicRegularizer::None &&
            (regularization.strength <= 0.f ||
             regularization.inner_iterations <= 0 ||
             regularization.epsilon <= 0.f ||
             regularization.strength_reduction <= 0.f ||
             params.vox_x_mm <= 0.f || params.vox_y_mm <= 0.f ||
             params.vox_z_mm <= 0.f)) {
            YK_LOGE("[AlgebraicReconstructorEx] 无效正则化配置");
            return false;
        }
        return true;
    }

    SCBCTParams params_{};
    AlgebraicReconstructionConfig config_{};
    cudaStream_t stream_ = nullptr;
    int subset_count_ = 0;
    int actual_subset_count_ = 0;
    bool prepared_ = false;
    bool use_tigre_weights_ = false;
    AlgebraicDetailedWeightBackend detailed_weights_{};
    AlgebraicTigreBackend tigre_{};
    AlgebraicRegularizer regularizer_{};
};

// 标准圆轨迹便捷入口。它只负责由 SCBCTParams 生成逐视角 geometry，实际
// 重建仍由 Ex 类完成，因此普通版与显式几何版不会形成两套算法实现。
class AlgebraicReconstructor {
public:
    using Config = AlgebraicReconstructionConfig;

    bool prepare(const SCBCTParams& params, const Config& config,
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

private:
    AlgebraicReconstructorEx implementation_{};
};

} // namespace YK::Iter
