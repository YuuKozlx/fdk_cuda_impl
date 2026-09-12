#pragma once

#include "Iter/YkCglsReconstructor.hpp"

namespace YK {

struct CglsLegacyConfig {
    int n_iter = 50;
    float eps = 1e-8f;
    bool restart = true;
    bool use_min = false;
    float min_constraint = 0.f;
    bool use_max = false;
    float max_constraint = 1e30f;
    ETask fp_task = ETask::FP_Joseph;
    ETask bp_task = ETask::BP_Joseph_v2;
};

inline Iter::CglsReconstructionConfig makeCglsConfig(
    const CglsLegacyConfig& old, Iter::ECglsStrategy strategy)
{
    Iter::CglsReconstructionConfig config{};
    config.strategy = strategy;
    config.iterations = old.n_iter;
    config.epsilon = old.eps;
    config.restart_on_divergence = old.restart;
    config.use_min = old.use_min;
    config.min_constraint = old.min_constraint;
    config.use_max = old.use_max;
    config.max_constraint = old.max_constraint;
    config.fp_task = old.fp_task;
    config.bp_task = old.bp_task;
    return config;
}

class CGLS {
public:
    using Config = CglsLegacyConfig;
    bool init(const SReconstructionParams& params, const Config& config,
        cudaStream_t stream, int device_id = 0)
    { return init(params, config, std::vector<SConeProjGeomVec>{}, stream, device_id); }

    bool init(const SReconstructionParams& params, const Config& config,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        const auto unified = makeCglsConfig(config, Iter::ECglsStrategy::RobustRestart);
        std::vector<SConeProjGeomVec> resolved = geometry;
        if (resolved.empty()) YK::detail::buildCircularViews(params, resolved);
        return implementation_.prepare(params, resolved, unified, stream, device_id);
    }
    bool run(const float* measured, float* volume,
        const SReconstructionParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }
private:
    Iter::CglsReconstructorEx implementation_{};
};

class CGLSEx {
public:
    using Config = CglsLegacyConfig;
    bool init(const SReconstructionParams& params, const Config& config,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        return implementation_.prepare(params, geometry,
            makeCglsConfig(config, Iter::ECglsStrategy::RobustRestart),
            stream, device_id);
    }
    bool run(const float* measured, float* volume,
        const SReconstructionParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }
private:
    Iter::CglsReconstructorEx implementation_{};
};

class CGLSAstra {
public:
    struct Config {
        int n_iter = 50;
        float eps = 1e-8f;
        bool use_min = false;
        float min_constraint = 0.f;
        bool use_max = false;
        float max_constraint = 1e30f;
        ETask fp_task = ETask::FP_Joseph;
        ETask bp_task = ETask::BP_FDK_matched;
    };
    bool init(const SReconstructionParams& params, const Config& old,
        cudaStream_t stream, int device_id = 0)
    {
        CglsLegacyConfig common{};
        common.n_iter = old.n_iter; common.eps = old.eps;
        common.use_min = old.use_min; common.min_constraint = old.min_constraint;
        common.use_max = old.use_max; common.max_constraint = old.max_constraint;
        common.fp_task = old.fp_task; common.bp_task = old.bp_task;
        return implementation_.prepare(params,
            makeCglsConfig(common, Iter::ECglsStrategy::AstraClassic),
            stream, device_id);
    }
    bool run(const float* measured, float* volume,
        const SReconstructionParams&, cudaStream_t)
    { return implementation_.reconstruct(measured, volume); }
    void reset() { implementation_.reset(); }
    void release() { implementation_.release(); }
private:
    Iter::CglsReconstructor implementation_{};
};

inline bool cgls_reconstruct(const float* measured, float* volume,
    const SReconstructionParams& params, cudaStream_t stream, CGLS::Config config = {})
{
    CGLS recon;
    return recon.init(params, config, stream) &&
        recon.run(measured, volume, params, stream);
}
inline bool cgls_ex_reconstruct(const float* measured, float* volume,
    const SReconstructionParams& params, const std::vector<SConeProjGeomVec>& geometry,
    cudaStream_t stream, CGLSEx::Config config = {})
{
    CGLSEx recon;
    return recon.init(params, config, geometry, stream) &&
        recon.run(measured, volume, params, stream);
}
inline bool cgls_astra_reconstruct(const float* measured, float* volume,
    const SReconstructionParams& params, cudaStream_t stream, CGLSAstra::Config config = {})
{
    CGLSAstra recon;
    return recon.init(params, config, stream) &&
        recon.run(measured, volume, params, stream);
}

} // namespace YK
