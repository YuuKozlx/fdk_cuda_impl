#pragma once
#include "YKCBCT/global/YkExport.hpp"
#include "global/YkCBCTParams.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <limits>
#include <memory>
#include <string>
namespace YK::Iter {
enum class EAlgebraicRegularizer : int { None = 0,
    SmoothedTv = 1,
    HuberTvProx = 2 };
enum class ETvDimensionality : int { Slice2D = 2,
    Volume3D = 3 };
struct AlgebraicRegularizationConfig {
    EAlgebraicRegularizer type = EAlgebraicRegularizer::None;
    float strength = 0.f;
    int inner_iterations = 5;
    ETvDimensionality tv_dimensionality = ETvDimensionality::Volume3D;
    float epsilon = 1e-6f, strength_reduction = 1.f;
    // A registered ID takes precedence over the compatibility enum.
    std::string plugin_id;
    float huber_delta = 1e-3f, step_size = 0.f;
    bool non_negative = true;
    bool require_prox = false;
    float lower_bound = -std::numeric_limits<float>::max(), upper_bound = std::numeric_limits<float>::max();
    bool gradient_only = false; // statistical solvers own their volume workspaces
};
struct AlgebraicRegularizationContext {
    int outer_iteration = 0, completed_subset_updates = 0;
    float data_update_l2 = std::numeric_limits<float>::quiet_NaN(), data_relaxation = 1.f;
};
struct RegularizerCapabilities {
    bool value = false, gradient = false, prox = false;
};
class IAlgebraicRegularizer {
public:
    virtual ~IAlgebraicRegularizer() = default;
    virtual bool prepare(const SReconstructionParams&, const AlgebraicRegularizationConfig&, cudaStream_t, int) = 0;
    virtual bool apply(float*, const AlgebraicRegularizationContext&) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual RegularizerCapabilities capabilities() const { return {}; }
    // Explicit Prox: caller supplies fixed z, output and current gamma (e.g.
    // step * lambda for FISTA). Legacy apply() may maintain its own schedule.
    virtual bool prox(const float*, float*, float) { return false; }
    virtual bool value(const float*, double&) { return false; }
    virtual bool gradient(const float*, float*) { return false; }
    virtual double gradientLipschitz() const { return 0; }
    virtual bool requiresDataUpdateNorm() const { return false; }
    virtual size_t workspaceBytes(const SReconstructionParams&, const AlgebraicRegularizationConfig&) const { return 0; }
};
using RegularizerFactory = std::function<std::unique_ptr<IAlgebraicRegularizer>()>;
YK_API bool registerRegularizerPlugin(const std::string&, RegularizerFactory);
YK_API std::unique_ptr<IAlgebraicRegularizer> createRegularizerPlugin(const std::string&);
YK_API std::string regularizerPluginId(const AlgebraicRegularizationConfig&);
// Each prepare gets a fresh plugin instance; immutable factory registration is
// process-wide, while all CUDA buffers and schedules are session-owned.
class AlgebraicRegularizer {
    std::unique_ptr<IAlgebraicRegularizer> implementation_;

public:
    ~AlgebraicRegularizer() { release(); }
    bool prepare(const SReconstructionParams& p, const AlgebraicRegularizationConfig& c, cudaStream_t stream, int device)
    {
        release();
        const auto id = regularizerPluginId(c);
        if (id.empty())
            return !c.plugin_id.empty() || c.type == EAlgebraicRegularizer::None;
        try {
            implementation_ = createRegularizerPlugin(id);
            if (!implementation_ || !implementation_->prepare(p, c, stream, device)) {
                release();
                return false;
            }
        } catch (...) {
            release();
            return false;
        }
        return true;
    }
    bool apply(float* volume, const AlgebraicRegularizationContext& c)
    {
        try {
            return !implementation_ || implementation_->apply(volume, c);
        } catch (...) {
            return false;
        }
    }
    bool enabled() const { return bool(implementation_); }
    bool requiresDataUpdateNorm() const { return implementation_ && implementation_->requiresDataUpdateNorm(); }
    void reset()
    {
        if (implementation_)
            implementation_->reset();
    }
    void release()
    {
        if (implementation_)
            implementation_->release();
        implementation_.reset();
    }
};
}
