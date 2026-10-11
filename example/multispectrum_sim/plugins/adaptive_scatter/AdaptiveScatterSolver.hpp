#pragma once

#include "config/SimTypes.hpp"
#include <functional>
#include <vector>

namespace yk::spectral::experimental {

struct AdaptiveScatterIteration {
    int iteration = 0;
    double damping = 0.0;
    double relative_residual = 0.0;
    double relative_change = 0.0;
};

struct AdaptiveScatterReport {
    int iterations = 0;
    int predictions = 0;
    int rejected_steps = 0;
    int positivity_limited_steps = 0;
    double low_transmission = 1.0;
    bool low_transmission_risk = false;
    double damping = 0.0;
    double relative_residual = 0.0;
    double relative_change = 0.0;
    const char* status = "completed";
    std::vector<AdaptiveScatterIteration> history;
};

// The predictor supplies S(P). Its model, mask and scatter scale stay fixed.
using ScatterPredictor = std::function<void(const std::vector<float>&,
                                           std::vector<float>&)>;
using DeviceScatterPredictor = std::function<void(const float*, const float*, float*)>;

class AdaptiveScatterSolver {
public:
    explicit AdaptiveScatterSolver(size_t pixels);
    ~AdaptiveScatterSolver();
    AdaptiveScatterSolver(const AdaptiveScatterSolver&) = delete;
    AdaptiveScatterSolver& operator=(const AdaptiveScatterSolver&) = delete;

    AdaptiveScatterReport solve(const std::vector<float>& measured,
        const std::vector<float>& air, std::vector<float>& primary,
        const ScatterCorrectionConfig::SolverConfig::AdaptiveConfig& options,
        double scatter_scale, const ScatterPredictor& predict,
        const std::vector<unsigned char>* object_mask = nullptr,
        const DeviceScatterPredictor& device_predict = {});

private:
    size_t pixels_;
    float *current_ = nullptr, *candidate_ = nullptr, *measured_ = nullptr;
    float *air_ = nullptr, *scatter_ = nullptr;
    float* accepted_scatter_ = nullptr;
    double* statistics_ = nullptr;
};

}
