#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace YK {
enum class EStatisticalOptimizer : int32_t { GradientDescent = 0,
    Fista = 1,
    RegularizedAlgebraic = 2 };
enum class EDataFidelityKind : int32_t { WeightedLeastSquares = 0 };
struct SDataFidelitySpec {
    EDataFidelityKind kind = EDataFidelityKind::WeightedLeastSquares;
    bool use_projection_weights = false;
    bool normalize_by_ray_count = false;
    bool require_verified_adjoint = false;
    float scale = 1.f;
};
struct SStatisticalRegularizationSpec {
    // Borrowed through initialize; copied by the backend.
    const char* plugin_id = "huber_tv_prox";
    float strength = 0.f;
    float shape = 1e-3f;
    int dimensions = 3;
};
struct SStatisticalOptimizerSpec {
    EStatisticalOptimizer kind = EStatisticalOptimizer::RegularizedAlgebraic;
    int iterations = 10;
    float step_size = 0.f; // zero estimates an initial step from the operators
    bool backtracking = true;
    int max_backtracking = 20;
    float step_shrink = 0.5f;
    float relative_tolerance = 0.f; // zero disables early stopping
    int minimum_iterations = 1;
};
struct SStatisticalConstraintSpec {
    float lower_bound = 0.f;
    float upper_bound = std::numeric_limits<float>::max();
};
struct SStatisticalWorkspaceEstimate {
    size_t optimizer_bytes = 0, data_fidelity_bytes = 0, regularization_bytes = 0;
    size_t caller_volume_bytes = 0, projection_staging_bytes = 0, weight_staging_bytes = 0;
    // Texture and geometry headroom are included in data_fidelity_bytes.
    size_t solverBytes() const { return optimizer_bytes + data_fidelity_bytes + regularization_bytes; }
};
struct SStatisticalReconstructionSpec {
    SDataFidelitySpec data_fidelity {};
    SStatisticalRegularizationSpec regularization {};
    SStatisticalOptimizerSpec optimizer {};
    SStatisticalConstraintSpec constraints {};
};
}
