#pragma once
#include "Reconstruction/Iterative/Regularization/RegularizerPlugin.hpp"
#include "YKCBCT/algorithms/iterative/YkStatisticalParams.hpp"
#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "common/YkOperatorTypes.hpp"
#include <string>
namespace YK::detail {
class ProjectionInput;
}
namespace YK::Iter::Statistical {
// Internal experiment controls: never serialized or exposed by the SDK.
enum class ERegularizationStrategy : int32_t { AlternatingGradient = 0, SynchronousGradient = 1, Proximal = 2 };
struct SRegularizationCompositionSpec {
    ERegularizationStrategy strategy = ERegularizationStrategy::AlternatingGradient;
    int prox_iterations = 20;
    float prox_step = 0.f;
};
inline SRegularizationCompositionSpec defaultComposition(const SStatisticalReconstructionSpec& c) {
    SRegularizationCompositionSpec result;
    // Huber-TV defaults to the independent proximal subproblem; standard TV
    // also requires Prox. Raw GD retains its joint-gradient definition.
    const std::string id=c.regularization.plugin_id?c.regularization.plugin_id:"none";
    if(c.optimizer.kind==EStatisticalOptimizer::RegularizedAlgebraic &&
        (id=="huber_tv"||id=="huber_tv_prox"||id=="tv"))
        result.strategy=ERegularizationStrategy::Proximal;
    return result;
}

enum class AdjointStatus { Unverified,
    Approximate,
    Verified };
struct DataFidelityCapabilities {
    bool value = true, gradient = true, diagonal_weights = true, batched_input = true;
    AdjointStatus adjoint = AdjointStatus::Unverified;
    bool algebraic_update = false;
};
// x and gradient are device buffers. A null output requests only the other
// capability. Every value is a host scalar after completion on the bound stream.
struct DataFidelitySetup {
    const SReconstructionParams& params;
    const std::vector<SConeProjGeomVec>& geometry;
    const SDataFidelitySpec& config;
    ETask fp_task, bp_task;
    cudaStream_t stream;
    int device;
    bool algebraic_enabled = false;
};
class IProjectionDataSource {
public:
    virtual ~IProjectionDataSource() = default;
    virtual int preferredViews() const = 0;
    virtual bool prepare(int, cudaStream_t) = 0;
    // Returned device pointer remains valid until the next fetch/release.
    virtual const float* fetch(const std::vector<int>&, cudaStream_t) = 0;
    virtual const std::string& error() const = 0;
};
struct ProjectionDataSet {
    const float* projection_device = nullptr;
    IProjectionDataSource* projection_source = nullptr;
    const float* weights_device = nullptr;
    IProjectionDataSource* weights_source = nullptr;
};
class IDataFidelity {
public:
    virtual ~IDataFidelity() = default;
    virtual DataFidelityCapabilities capabilities() const = 0;
    virtual bool prepare(const DataFidelitySetup&) = 0;
    virtual size_t workspaceBytes(const SReconstructionParams&) const = 0;
    virtual bool bind(const ProjectionDataSet&) = 0;
    virtual void unbind() { } // borrowed sources end with this execution
    virtual size_t algebraicWorkspaceBytes(const SReconstructionParams&) const { return 0; }
    // Full-view SIRT update; normalization belongs to the data provider.
    virtual bool algebraicUpdate(float*, float) { return false; }
    virtual bool evaluate(const float* x, float* gradient, double* value) = 0;
    // Conservative bound for nonnegative matched operators; only a heuristic
    // when adjoint status is not Verified. Caller supplies two volume buffers.
    virtual bool estimateLipschitz(float* ones, float* curvature, double& bound) = 0;
    virtual const std::string& error() const = 0;
};
struct OptimizerStatistics;
class IOptimizerObserver {
public:
    virtual ~IOptimizerObserver() = default;
    // False requests a clean stop; the last accepted device volume is retained.
    // Callback must not throw. Called only after GPU completion when enabled.
    virtual bool continueIteration(const OptimizerStatistics&) = 0;
};
struct ReconstructionProblem {
    IDataFidelity& data;
    IAlgebraicRegularizer* regularization = nullptr;
    float strength = 0;
    double regularization_lipschitz = 0;
    IOptimizerObserver* observer = nullptr;
};
struct OptimizerStatistics {
    int completed_iterations = 0, total_backtracking = 0;
    double initial_objective = 0, final_objective = 0, last_step = 0, relative_update = 0;
    bool converged = false, cancelled = false;
};
struct OptimizerRequirements {
    bool data_value = true, data_gradient = true, regularization_value = true;
    bool regularization_gradient = true, regularization_prox = false, verified_adjoint = false;
    bool data_algebraic_update = false;
};
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    virtual bool configureComposition(const SRegularizationCompositionSpec& c) { return c.strategy == ERegularizationStrategy::AlternatingGradient; }
    virtual OptimizerRequirements requirements() const = 0;
    virtual size_t workspaceBytes(size_t) const = 0;
    virtual bool prepare(size_t, const SStatisticalOptimizerSpec&, const SStatisticalConstraintSpec&, cudaStream_t, int) = 0;
    virtual bool run(ReconstructionProblem&, float*) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual const OptimizerStatistics& statistics() const = 0;
    virtual const std::string& error() const = 0;
};
YK_API std::unique_ptr<IDataFidelity> createDataFidelity(EDataFidelityKind);
YK_API std::unique_ptr<IOptimizer> createOptimizer(EStatisticalOptimizer);
}
