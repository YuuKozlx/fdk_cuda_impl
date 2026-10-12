#pragma once
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "Reconstruction/Iterative/Common/kernels/YkIterLaunch.cuh"
#include "Reconstruction/Iterative/Statistical/ReductionWorkspace.hpp"
#include "Reconstruction/Iterative/Statistical/StatisticalContracts.hpp"
#include "common/YkDeviceWorkspace.hpp"
#include "common/YkExecutionProgress.hpp"
namespace YK::Iter::Statistical {
// Alternating normalized algebraic data update and regularization gradient.
// It is not the gradient of the displayed WLS objective.
class RegularizedAlgebraic final : public IOptimizer {
    DeviceWorkspaceF32 data_volume_, candidate_, regularization_gradient_;
    ReductionWorkspace reduction_;
    size_t n_ = 0;
    cudaStream_t stream_ = nullptr;
    SStatisticalOptimizerSpec config_ {};
    SRegularizationCompositionSpec composition_ {};
    SStatisticalConstraintSpec constraints_ {};
    OptimizerStatistics statistics_ {};
    std::string error_;
    bool objective(ReconstructionProblem& p, const float* x, double& value)
    {
        if (!p.data.evaluate(x, nullptr, &value)) {
            error_ = p.data.error();
            return false;
        }
        if (p.regularization && p.strength > 0) {
            double r = 0;
            if (!p.regularization->value(x, r))
                return false;
            value += p.strength * r;
        }
        return std::isfinite(value);
    }
    bool notify(ReconstructionProblem& p)
    {
        if (!p.observer)
            return true;
        if (cudaStreamSynchronize(stream_) != cudaSuccess)
            return false;
        if (!p.observer->continueIteration(statistics_))
            statistics_.cancelled = true;
        return true;
    }

public:
    ~RegularizedAlgebraic() { release(); }
    bool configureComposition(const SRegularizationCompositionSpec& c) override
    {
        if ((c.strategy != ERegularizationStrategy::AlternatingGradient && c.strategy != ERegularizationStrategy::SynchronousGradient && c.strategy != ERegularizationStrategy::Proximal) || c.prox_iterations <= 0 || !std::isfinite(c.prox_step) || c.prox_step < 0)
            return false;
        composition_ = c;
        return true;
    }
    OptimizerRequirements requirements() const override
    {
        OptimizerRequirements r;
        r.data_gradient = false;
        r.data_algebraic_update = true;
        r.regularization_prox = composition_.strategy == ERegularizationStrategy::Proximal;
        r.regularization_gradient = !r.regularization_prox;
        return r;
    }
    size_t workspaceBytes(size_t n) const override { return 3 * n * sizeof(float) + 8192; }
    bool prepare(size_t n, const SStatisticalOptimizerSpec& cfg, const SStatisticalConstraintSpec& bounds, cudaStream_t stream, int device) override
    {
        release();
        if (!n || cfg.kind != EStatisticalOptimizer::RegularizedAlgebraic || cfg.iterations <= 0 || !std::isfinite(cfg.step_size) || cfg.step_size < 0 || cfg.max_backtracking < 0 || cfg.step_shrink <= 0 || cfg.step_shrink >= 1 || !std::isfinite(cfg.step_shrink) || !std::isfinite(cfg.relative_tolerance) || cfg.relative_tolerance < 0 || cfg.minimum_iterations <= 0 || !std::isfinite(bounds.lower_bound) || !std::isfinite(bounds.upper_bound) || bounds.lower_bound > bounds.upper_bound)
            return false;
        if (!detail::fitsReconstructionBudget(workspaceBytes(n), "regularized algebraic optimizer"))
            return false;
        n_ = n;
        stream_ = stream;
        config_ = cfg;
        constraints_ = bounds;
        data_volume_.allocate(n, device);
        candidate_.allocate(n, device);
        regularization_gradient_.allocate(n, device);
        return reduction_.prepare();
    }
    bool run(ReconstructionProblem& p, float* x) override
    {
        reset();
        if (!x || !p.data.capabilities().algebraic_update) {
            error_ = "data provider has no algebraic update";
            return false;
        }
        if (p.strength > 0 && (!p.regularization || !p.regularization->capabilities().value || (requirements().regularization_prox ? !p.regularization->capabilities().prox : !p.regularization->capabilities().gradient)))
            return false;
        clamp_launch(x, n_, constraints_.lower_bound, constraints_.upper_bound, stream_);
        if (!notify(p))
            return false;
        if (statistics_.cancelled)
            return true;
        if (!objective(p, x, statistics_.initial_objective))
            return false;
        double regularization_step = config_.step_size;
        if (p.strength > 0 && regularization_step == 0) {
            double data_bound = 0;
            if (!p.data.estimateLipschitz(data_volume_.data(), candidate_.data(), data_bound)) {
                error_ = p.data.error();
                return false;
            }
            regularization_step = 1 / std::max(1e-12, data_bound + p.strength * p.regularization_lipschitz);
        }
        for (int iteration = 0; iteration < config_.iterations; ++iteration) {
            if (cudaMemcpyAsync(data_volume_.data(), x, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            if (!p.data.algebraicUpdate(data_volume_.data(), 1.f)) {
                error_ = p.data.error();
                return false;
            }
            if(composition_.strategy != ERegularizationStrategy::SynchronousGradient || p.strength == 0)
                clamp_launch(data_volume_.data(), n_, constraints_.lower_bound, constraints_.upper_bound, stream_);
            const float* accepted = data_volume_.data();
            if (p.strength > 0 && composition_.strategy == ERegularizationStrategy::Proximal) {
                const float gamma = float(regularization_step * p.strength);
                if (!std::isfinite(gamma) || gamma <= 0 || !p.regularization->prox(data_volume_.data(), candidate_.data(), gamma)) {
                    error_ = "regularization prox failed";
                    return false;
                }
                clamp_launch(candidate_.data(), n_, constraints_.lower_bound, constraints_.upper_bound, stream_);
                accepted = candidate_.data();
                statistics_.last_step = regularization_step;
            } else if (p.strength > 0 && composition_.strategy == ERegularizationStrategy::SynchronousGradient) {
                // Both increments are based on x. x remains unchanged until acceptance.
                if (!p.regularization->gradient(x, regularization_gradient_.data()))
                    return false;
                subtract_launch(data_volume_.data(), data_volume_.data(), x, n_, stream_);
                axpy_launch(data_volume_.data(), regularization_gradient_.data(), -float(regularization_step * p.strength), n_, stream_);
                double before = 0;
                if (!objective(p, x, before))
                    return false;
                float fraction = 1.f;
                bool found = false;
                for (int attempt = 0; attempt < (config_.backtracking ? config_.max_backtracking + 1 : 1); ++attempt) {
                    linear_combination_launch(candidate_.data(), x, data_volume_.data(), fraction, n_, stream_);
                    clamp_launch(candidate_.data(), n_, constraints_.lower_bound, constraints_.upper_bound, stream_);
                    double after = 0;
                    if (!objective(p, candidate_.data(), after))
                        return false;
                    if (!config_.backtracking || after <= before + 1e-12 * std::max(1., std::abs(before))) {
                        found = true;
                        break;
                    }
                    fraction *= config_.step_shrink;
                    ++statistics_.total_backtracking;
                }
                if (!found) {
                    error_ = "synchronous combined direction did not decrease objective";
                    return false;
                }
                accepted = candidate_.data();
                statistics_.last_step = regularization_step * fraction;
            } else if (p.strength > 0) {
                double before = 0;
                if (!p.regularization->value(data_volume_.data(), before) || !p.regularization->gradient(data_volume_.data(), regularization_gradient_.data()))
                    return false;
                bool found = false;
                double step = regularization_step;
                for (int attempt = 0; attempt < (config_.backtracking ? config_.max_backtracking + 1 : 1); ++attempt) {
                    if (!std::isfinite(step) || step <= 0 || float(step * p.strength) == 0) {
                        error_ = "regularization step underflow";
                        return false;
                    }
                    gradient_update_launch(candidate_.data(), data_volume_.data(), regularization_gradient_.data(), n_, float(step * p.strength), constraints_.lower_bound, constraints_.upper_bound, stream_);
                    double after = 0;
                    if (!p.regularization->value(candidate_.data(), after))
                        return false;
                    // Backtracking applies only to the regularization substep.
                    // It never shortens/rejects the normalized algebraic update.
                    if (!config_.backtracking || after <= before + 1e-12 * std::max(1., std::abs(before))) {
                        found = true;
                        break;
                    }
                    step *= config_.step_shrink;
                    ++statistics_.total_backtracking;
                }
                if (!found) {
                    error_ = "regularization backtracking found no decreasing step";
                    return false;
                }
                accepted = candidate_.data();
                regularization_step = step;
                statistics_.last_step = step;
            } else
                statistics_.last_step = 1;
            if (config_.relative_tolerance > 0) {
                double change = 0, norm = 0;
                squared_change_launch(accepted, x, n_, reduction_.data(), stream_);
                if (!reduction_.read(stream_, change))
                    return false;
                cudaMemsetAsync(regularization_gradient_.data(), 0, n_ * sizeof(float), stream_);
                squared_change_launch(x, regularization_gradient_.data(), n_, reduction_.data(), stream_);
                if (!reduction_.read(stream_, norm))
                    return false;
                statistics_.relative_update = std::sqrt(change / std::max(norm, 1e-24));
            }
            if (!objective(p, accepted, statistics_.final_objective))
                return false;
            if (cudaMemcpyAsync(x, accepted, n_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            statistics_.completed_iterations = iteration + 1;
            if (!detail::reportExecutionProgress(iteration + 1, config_.iterations, stream_) || !notify(p))
                return false;
            if (statistics_.cancelled)
                return true;
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
        data_volume_.reset();
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
