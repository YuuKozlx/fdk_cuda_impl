#pragma once

#include <algorithm>
#include <cmath>

#include "Iter/YkAlgebraicReconstructorEx.hpp"

namespace YK {

namespace detail {
inline Iter::AlgebraicReconstructionConfig makeLegacyAlgebraicConfig(
    Iter::EAlgebraicMethod method, int n_iter, int n_subset,
    float lambda, float lambda_red, float eps,
    bool use_min, float min_constraint, bool use_max, float max_constraint,
    ETask fp_task, ETask bp_task, bool reduction_was_per_subset)
{
    Iter::AlgebraicReconstructionConfig out{};
    out.method = method;
    out.iterations = n_iter;
    out.subset_count = n_subset;
    out.relaxation = lambda;
    out.relaxation_reduction = reduction_was_per_subset
        ? std::pow(lambda_red, static_cast<float>(std::max(n_subset, 1)))
        : lambda_red;
    out.epsilon = eps;
    out.use_min = use_min;
    out.min_constraint = min_constraint;
    out.use_max = use_max;
    out.max_constraint = max_constraint;
    out.fp_task = fp_task;
    out.bp_task = bp_task;
    return out;
}
} // namespace detail

class OSSART_TIGRE {
public:
    struct Config {
        int n_iter = 10;
        int n_subset = 5;
        float lambda = 1.f;
        float lambda_red = 1.f;
        float eps = 1e-6f;
        bool use_min = false;
        float min_constraint = 0.f;
        bool use_max = false;
        float max_constraint = 1e30f;
        ETask fp_task = ETask::FP_Joseph;
        ETask bp_task = ETask::BP_Joseph_v2;
    };

    bool init(const SCBCTParams& params, const Config& cfg,
        cudaStream_t stream, int device_id = 0)
    {
        auto unified = detail::makeLegacyAlgebraicConfig(
            Iter::EAlgebraicMethod::Ossart, cfg.n_iter, cfg.n_subset,
            cfg.lambda, cfg.lambda_red, cfg.eps, cfg.use_min, cfg.min_constraint,
            cfg.use_max, cfg.max_constraint, cfg.fp_task, cfg.bp_task, false);
        unified.weight_model = Iter::EAlgebraicWeightModel::TigreApprox;
        return implementation_.prepare(params, unified, stream, device_id);
    }
    bool iterate(const float* measured, float* volume, const SCBCTParams&,
        cudaStream_t, unsigned int iterations)
    { return implementation_.iterateSubsetUpdates(measured, volume, iterations); }
    bool run(const float* measured, float* volume, const SCBCTParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }
    unsigned int totalIterations() const
    { return implementation_.totalSubsetUpdates(); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }

private:
    Iter::AlgebraicReconstructor implementation_{};
};

class OSSART {
public:
    struct Config {
        int n_iter = 10;
        int n_subset = 20;
        float lambda = 1.f;
        float lambda_red = 1.f;
        float eps = 1e-6f;
        bool use_min = false;
        float min_constraint = 0.f;
        bool use_max = false;
        float max_constraint = 1e30f;
        ETask fp_task = ETask::FP_Joseph;
        ETask bp_task = ETask::BP_Joseph_v2;
    };

    bool init(const SCBCTParams& params, const Config& cfg,
        cudaStream_t stream, int device_id = 0)
    {
        auto unified = detail::makeLegacyAlgebraicConfig(
            Iter::EAlgebraicMethod::Ossart, cfg.n_iter, cfg.n_subset,
            cfg.lambda, cfg.lambda_red, cfg.eps, cfg.use_min, cfg.min_constraint,
            cfg.use_max, cfg.max_constraint, cfg.fp_task, cfg.bp_task, true);
        unified.subset_order = Iter::EAlgebraicSubsetOrder::GoldenRatio;
        return implementation_.prepare(params, unified, stream, device_id);
    }
    bool iterate(const float* measured, float* volume, const SCBCTParams&,
        cudaStream_t, unsigned int iterations)
    { return implementation_.iterateSubsetUpdates(measured, volume, iterations); }
    bool run(const float* measured, float* volume, const SCBCTParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }
    unsigned int totalIterations() const
    { return implementation_.totalSubsetUpdates(); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }

private:
    Iter::AlgebraicReconstructor implementation_{};
};

class OSSARTEx {
public:
    using Config = OSSART::Config;

    bool init(const SCBCTParams& params, const Config& cfg,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        auto unified = detail::makeLegacyAlgebraicConfig(
            Iter::EAlgebraicMethod::Ossart, cfg.n_iter, cfg.n_subset,
            cfg.lambda, cfg.lambda_red, cfg.eps, cfg.use_min, cfg.min_constraint,
            cfg.use_max, cfg.max_constraint, cfg.fp_task, cfg.bp_task, true);
        unified.subset_order = Iter::EAlgebraicSubsetOrder::Sequential;
        return implementation_.prepare(params, geometry, unified, stream, device_id);
    }
    bool iterate(const float* measured, float* volume, cudaStream_t,
        unsigned int iterations)
    { return implementation_.iterateSubsetUpdates(measured, volume, iterations); }
    bool run(const float* measured, float* volume, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }
    unsigned int totalIterations() const
    { return implementation_.totalSubsetUpdates(); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }

private:
    Iter::AlgebraicReconstructorEx implementation_{};
};

inline bool ossart_tigre_reconstruct(const float* measured, float* volume,
    const SCBCTParams& params, cudaStream_t stream, OSSART_TIGRE::Config cfg = {})
{
    OSSART_TIGRE recon;
    return recon.init(params, cfg, stream) &&
        recon.run(measured, volume, params, stream);
}

inline bool ossart_reconstruct(const float* measured, float* volume,
    const SCBCTParams& params, cudaStream_t stream, OSSART::Config cfg = {})
{
    OSSART recon;
    return recon.init(params, cfg, stream) &&
        recon.run(measured, volume, params, stream);
}

inline bool ossart_reconstruct_ex(const float* measured, float* volume,
    const SCBCTParams& params, const std::vector<SConeProjGeomVec>& geometry,
    cudaStream_t stream, OSSARTEx::Config cfg = {})
{
    OSSARTEx recon;
    return recon.init(params, cfg, geometry, stream) &&
        recon.run(measured, volume, stream);
}

} // namespace YK
