#pragma once
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "Reconstruction/Iterative/Common/kernels/YkIterLaunch.cuh"
#include "Reconstruction/Iterative/Statistical/ReductionWorkspace.hpp"
#include "Reconstruction/Iterative/Statistical/StatisticalContracts.hpp"
#include "common/YkDeviceWorkspace.hpp"
#include "common/YkExecutionProgress.hpp"
namespace YK::Iter::Statistical {
class GradientDescent final : public IOptimizer {
    DeviceWorkspaceF32 gradient_, candidate_, regularization_gradient_;
    ReductionWorkspace reduction_;
    size_t n_ = 0;
    cudaStream_t stream_ = nullptr;
    SStatisticalOptimizerSpec config_ {};
    SStatisticalConstraintSpec constraints_ {};
    OptimizerStatistics statistics_ {};
    std::string error_;
    bool evaluate(ReconstructionProblem& p, const float* x, float* gradient, double& objective)
    {
        if (!p.data.evaluate(x, gradient, &objective)) {
            error_ = p.data.error();
            return false;
        }
        if (p.regularization && p.strength > 0) {
            double penalty = 0;
            if (!p.regularization->value(x, penalty)) {
                error_ = "regularization value evaluation failed";
                return false;
            }
            objective += p.strength * penalty;
            if (gradient) {
                if (!p.regularization->gradient(x, regularization_gradient_.data())) {
                    error_ = "regularization gradient evaluation failed";
                    return false;
                }
                axpy_launch(gradient, regularization_gradient_.data(), p.strength, n_, stream_);
            }
        }
        return std::isfinite(objective);
    }

public:
    ~GradientDescent() { release(); }
    static size_t estimateWorkspaceBytes(size_t n) { return 3 * n * sizeof(float) + 8192; }
    OptimizerRequirements requirements() const override { return {}; }
    size_t workspaceBytes(size_t n) const override { return estimateWorkspaceBytes(n); }
    bool prepare(size_t n, const SStatisticalOptimizerSpec& cfg, const SStatisticalConstraintSpec& constraints, cudaStream_t stream, int device) override
    {
        release();
        if (!n || cfg.kind != EStatisticalOptimizer::GradientDescent || cfg.iterations <= 0 || !std::isfinite(cfg.step_size) || cfg.step_size < 0 || cfg.max_backtracking < 0 || !std::isfinite(cfg.step_shrink) || cfg.step_shrink <= 0 || cfg.step_shrink >= 1 || !std::isfinite(cfg.relative_tolerance) || cfg.relative_tolerance < 0 || cfg.minimum_iterations <= 0 || !std::isfinite(constraints.lower_bound) || !std::isfinite(constraints.upper_bound) || constraints.lower_bound > constraints.upper_bound)
            return false;
        if (!detail::fitsReconstructionBudget(workspaceBytes(n), "statistical optimizer"))
            return false;
        n_ = n;
        config_ = cfg;
        constraints_ = constraints;
        stream_ = stream;
        gradient_.allocate(n, device);
        candidate_.allocate(n, device);
        regularization_gradient_.allocate(n, device);
        return reduction_.prepare();
    }
    bool run(ReconstructionProblem& p, float* x) override
    {
        reset();
        if (!x)
            return false;
        if (p.regularization && p.strength > 0 && (!p.regularization->capabilities().gradient || !p.regularization->capabilities().value)) {
            error_ = "Gradient Descent requires regularization value and gradient capabilities";
            return false;
        }
        // Project the initial volume once; every later accepted update respects bounds.
        clamp_launch(x, n_, constraints_.lower_bound, constraints_.upper_bound, stream_);
        if(p.observer){
            if(cudaStreamSynchronize(stream_)!=cudaSuccess)return false;
            if(!p.observer->continueIteration(statistics_)){statistics_.cancelled=true;return true;}
        }
        double step = config_.step_size;
        if (step == 0) {
            double data_bound = 0;
            if (!p.data.estimateLipschitz(candidate_.data(), gradient_.data(), data_bound)) {
                error_ = p.data.error();
                return false;
            }
            step = 1 / std::max(1e-12, data_bound + p.strength * p.regularization_lipschitz);
        }
        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            double current = 0;
            if (!evaluate(p, x, gradient_.data(), current))
                return false;
            if (iteration == 0)
                statistics_.initial_objective = current;
            bool accepted = false;
            double trial_value = current, trial_step = step;
            const int attempts = config_.backtracking ? config_.max_backtracking + 1 : 1;
            for (int attempt = 0; attempt < attempts; ++attempt) {
                if (!std::isfinite(trial_step) || trial_step <= 0 || float(trial_step) == 0) {
                    error_ = "statistical step underflow";
                    return false;
                }
                gradient_update_launch(candidate_.data(), x, gradient_.data(), n_, float(trial_step), constraints_.lower_bound, constraints_.upper_bound, stream_);
                if (!evaluate(p, candidate_.data(), nullptr, trial_value))
                    return false;
                // Empirical monotonic acceptance also works with approximate BP.
                // It is not an Armijo/FISTA convergence guarantee.
                if (!config_.backtracking || trial_value <= current + 1e-12 * std::max(1.0, std::abs(current))) {
                    accepted = true;
                    break;
                }
                trial_step *= config_.step_shrink;
                ++statistics_.total_backtracking;
            }
            if (!accepted) {
                error_ = "backtracking found no decreasing step; check FP/BP, weights and scaling";
                return false;
            }
            if (config_.relative_tolerance > 0) {
                double change = 0, norm = 0;
                squared_change_launch(candidate_.data(), x, n_, reduction_.data(), stream_);
                if (!reduction_.read(stream_, change))
                    return false;
                if (cudaMemsetAsync(regularization_gradient_.data(), 0, n_ * sizeof(float), stream_) != cudaSuccess)
                    return false;
                squared_change_launch(x, regularization_gradient_.data(), n_, reduction_.data(), stream_);
                if (!reduction_.read(stream_, norm))
                    return false;
                statistics_.relative_update = std::sqrt(change / std::max(norm, 1e-24));
            }
            if (cudaMemcpyAsync(x, candidate_.data(), n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            step = trial_step;
            statistics_.last_step = step;
            statistics_.final_objective = trial_value;
            statistics_.completed_iterations = iteration + 1;
            if (!detail::reportExecutionProgress(iteration + 1, config_.iterations, stream_))
                return false;
            if(p.observer){
                if(cudaStreamSynchronize(stream_)!=cudaSuccess)return false;
                if(!p.observer->continueIteration(statistics_)){statistics_.cancelled=true;return true;}
            }
            if (config_.relative_tolerance > 0 && iteration + 1 >= config_.minimum_iterations && statistics_.relative_update <= config_.relative_tolerance) {
                statistics_.converged = true;
                break;
            }
        }
        return cudaPeekAtLastError() == cudaSuccess;
    }
    void reset() override
    {
        statistics_ = {};
        error_.clear();
    }
    void release() override
    {
        if (stream_)
            cudaStreamSynchronize(stream_);
        gradient_.reset();
        candidate_.reset();
        regularization_gradient_.reset();
        reduction_.release();
        stream_ = nullptr;
        n_ = 0;
        reset();
    }
    const OptimizerStatistics& statistics() const override { return statistics_; }
    const std::string& error() const override { return error_; }
};
}
