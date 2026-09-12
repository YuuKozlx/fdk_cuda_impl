#pragma once

#include <vector>

#include "Iter/YkCglsBackends.hpp"
#include "Iter/YkIterativeConvergence.hpp"

namespace YK::Iter {

enum class ECglsStrategy : int {
    RobustRestart = 0, // 每轮检查 ||b-Ax||，发散时回退并可重启
    AstraClassic = 1  // 直接递推残差，接近 ASTRA CGLS 流程
};

struct CglsReconstructionConfig {
    ECglsStrategy strategy = ECglsStrategy::RobustRestart;
    int iterations = 50;
    float epsilon = 1e-8f;
    bool restart_on_divergence = true;
    bool use_min = false;
    float min_constraint = 0.f;
    bool use_max = false;
    float max_constraint = 1e30f;
    ETask fp_task = ETask::FP_Joseph;
    ETask bp_task = ETask::BP_Joseph_v2;
    IterativeConvergenceConfig convergence{};
};

// 显式逐视角 geometry 的 CGLS 门面。Ex 只描述 geometry 来源，和
// RobustRestart/AstraClassic 数值策略相互独立。
class CglsReconstructorEx {
public:
    using Config = CglsReconstructionConfig;

    ~CglsReconstructorEx() { release(); }
    CglsReconstructorEx() = default;
    CglsReconstructorEx(const CglsReconstructorEx&) = delete;
    CglsReconstructorEx& operator=(const CglsReconstructorEx&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const Config& config, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || params.scan.NAng <= 0 ||
            static_cast<int>(geometry.size()) != params.scan.NAng ||
            config.iterations <= 0 || config.epsilon <= 0.f ||
            config.min_constraint > config.max_constraint ||
            !validConvergenceConfig(config.convergence) ||
            (config.strategy == ECglsStrategy::AstraClassic &&
             convergenceEnabled(config.convergence))) return false;
        params_ = params;
        config_ = config;
        stream_ = stream;

        if (config.strategy == ECglsStrategy::AstraClassic) {
            CglsAstraBackend::Config backend{};
            backend.n_iter = config.iterations;
            backend.eps = config.epsilon;
            backend.use_min = config.use_min;
            backend.min_constraint = config.min_constraint;
            backend.use_max = config.use_max;
            backend.max_constraint = config.max_constraint;
            backend.fp_task = config.fp_task;
            backend.bp_task = config.bp_task;
            prepared_ = astra_.init(params_, backend, geometry, stream_, device_id);
        }
        else {
            CglsRobustBackend::Config backend{};
            backend.n_iter = config.iterations;
            backend.eps = config.epsilon;
            backend.restart = config.restart_on_divergence;
            backend.use_min = config.use_min;
            backend.min_constraint = config.min_constraint;
            backend.use_max = config.use_max;
            backend.max_constraint = config.max_constraint;
            backend.fp_task = config.fp_task;
            backend.bp_task = config.bp_task;
            backend.convergence = config.convergence;
            prepared_ = robust_.init(params_, backend, geometry, stream_, device_id);
        }
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* measured_projection, float* volume)
    {
        if (!prepared_ || !measured_projection || !volume) return false;
        if (config_.strategy == ECglsStrategy::AstraClassic)
            return astra_.run(measured_projection, volume, params_, stream_);
        return robust_.run(measured_projection, volume, params_, stream_);
    }

    void reset()
    {
        if (!prepared_) return;
        if (config_.strategy == ECglsStrategy::AstraClassic)
            astra_.reset(stream_);
        else
            robust_.reset(stream_);
    }

    void release()
    {
        if (stream_)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        robust_.release();
        astra_.release();
        params_ = {};
        config_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }
    const IterativeConvergenceStatistics& convergenceStatistics() const
    { return robust_.statistics(); }

private:
    SReconstructionParams params_{};
    Config config_{};
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
    CglsRobustBackend robust_{};
    CglsAstraBackend astra_{};
};

} // namespace YK::Iter
