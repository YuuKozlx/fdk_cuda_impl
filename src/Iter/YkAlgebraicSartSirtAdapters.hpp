#pragma once

// SART/SIRT 旧接口适配器。算法执行统一转发到 AlgebraicReconstructor，
// 不再维护独立的投影、归一化、显存和迭代实现。
#include "Iter/YkAlgebraicReconstructor.hpp"

namespace YK {

class SART {
public:
    struct Config {
        int n_iter = 10;
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
    { return init(params, cfg, std::vector<SConeProjGeomVec>{}, stream, device_id); }

    bool init(const SCBCTParams& params, const Config& cfg,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        Iter::AlgebraicReconstructor::Config unified{};
        unified.method = Iter::EAlgebraicMethod::Sart;
        unified.iterations = cfg.n_iter;
        unified.relaxation = cfg.lambda;
        unified.relaxation_reduction = cfg.lambda_red;
        unified.epsilon = cfg.eps;
        unified.use_min = cfg.use_min;
        unified.min_constraint = cfg.min_constraint;
        unified.use_max = cfg.use_max;
        unified.max_constraint = cfg.max_constraint;
        unified.fp_task = cfg.fp_task;
        unified.bp_task = cfg.bp_task;
        std::vector<SConeProjGeomVec> resolved = geometry;
        if (resolved.empty()) YK::detail::buildCircularViews(params, resolved);
        return implementation_.prepare(params, resolved, unified, stream, device_id);
    }

    bool run(const float* measured, float* volume,
        const SCBCTParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }

    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }

private:
    Iter::AlgebraicReconstructorEx implementation_{};
};

inline bool sart_reconstruct(const float* measured, float* volume,
    const SCBCTParams& params, cudaStream_t stream, SART::Config cfg = {})
{
    SART recon;
    return recon.init(params, cfg, stream) &&
        recon.run(measured, volume, params, stream);
}

class SIRT {
public:
    struct Config {
        int n_iter = 50;
        float lambda = 1.f;
        float lambda_red = 1.f;
        float eps = 1e-6f;
        bool use_min = false;
        float min_constraint = 0.f;
        bool use_max = false;
        float max_constraint = 1e30f;
        // 以下字段只为源代码兼容保留。统一 SIRT 固定为一个全角度子集，
        // 内部显存调度不再由调用方控制批次数或调试落盘。
        int n_batch = 8;
        bool dump_debug = false;
        int row_w_down = 2;
        ETask fp_task = ETask::FP_Joseph;
        ETask bp_task = ETask::BP_Joseph_v3;
    };

    bool init(const SCBCTParams& params, const Config& cfg,
        cudaStream_t stream, int device_id = 0)
    { return init(params, cfg, std::vector<SConeProjGeomVec>{}, stream, device_id); }

    bool init(const SCBCTParams& params, const Config& cfg,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        Iter::AlgebraicReconstructor::Config unified{};
        unified.method = Iter::EAlgebraicMethod::Sirt;
        unified.iterations = cfg.n_iter;
        unified.relaxation = cfg.lambda;
        unified.relaxation_reduction = cfg.lambda_red;
        unified.epsilon = cfg.eps;
        unified.use_min = cfg.use_min;
        unified.min_constraint = cfg.min_constraint;
        unified.use_max = cfg.use_max;
        unified.max_constraint = cfg.max_constraint;
        unified.fp_task = cfg.fp_task;
        unified.bp_task = cfg.bp_task;
        std::vector<SConeProjGeomVec> resolved = geometry;
        if (resolved.empty()) YK::detail::buildCircularViews(params, resolved);
        return implementation_.prepare(params, resolved, unified, stream, device_id);
    }

    bool iterate(const float* measured, float* volume,
        const SCBCTParams&, cudaStream_t, unsigned int iterations)
    { return implementation_.iterateSubsetUpdates(measured, volume, iterations); }

    bool run(const float* measured, float* volume,
        const SCBCTParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }

    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }

private:
    Iter::AlgebraicReconstructorEx implementation_{};
};

inline bool sirt_reconstruct(const float* measured, float* volume,
    const SCBCTParams& params, cudaStream_t stream, SIRT::Config cfg = {})
{
    SIRT recon;
    return recon.init(params, cfg, stream) &&
        recon.run(measured, volume, params, stream);
}

} // namespace YK
