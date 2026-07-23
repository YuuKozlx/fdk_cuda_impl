#pragma once

#include <memory>

#include "BP/YkSiddonBPRunner.hpp"
#include "FP/YkFPRunner.hpp"
#include "common/YkExecutionContext.hpp"

namespace YK {

// Strongly typed common operator contract.  Solvers never need to know which
// Joseph/Siddon implementation is selected, nor manage CUDA streams directly.
class IForwardOperator {
public:
    virtual ~IForwardOperator() = default;
    virtual bool prepare(const GeometryContext&, ResourceContext&) = 0;
    virtual bool apply(const float* d_volume, const SCBCTParams& batch,
        float* d_projection, ResourceContext&) = 0;
    virtual void release() = 0;
};

class IBackOperator {
public:
    virtual ~IBackOperator() = default;
    virtual bool prepare(const GeometryContext&, ResourceContext&) = 0;
    virtual bool apply(const float* d_projection, const SCBCTParams& batch,
        float* d_volume, bool clear_volume, ResourceContext&) = 0;
    virtual void release() = 0;
};

class ForwardOperator final : public IForwardOperator {
public:
    explicit ForwardOperator(ETask kind) : kind_(kind) {}
    bool prepare(const GeometryContext& geometry, ResourceContext& resources) override
    { return runner_.init(geometry.base(), kind_, resources.device()); }
    bool apply(const float* d_volume, const SCBCTParams& batch,
        float* d_projection, ResourceContext& resources) override
    { return runner_.run(d_volume, batch, d_projection, resources.stream(), resources.device()); }
    void release() override { runner_.release(); }
private:
    ETask kind_;
    ConeProjector runner_;
};

class BackOperator final : public IBackOperator {
public:
    explicit BackOperator(ETask kind) : kind_(kind) {}
    bool prepare(const GeometryContext& geometry, ResourceContext& resources) override
    { return runner_.init(geometry.base(), kind_, resources.device()); }
    bool apply(const float* d_projection, const SCBCTParams& batch,
        float* d_volume, bool clear_volume, ResourceContext& resources) override
    { return runner_.run(d_projection, batch, d_volume, resources.stream(), clear_volume, resources.device()); }
    void release() override { runner_.release(); }
private:
    ETask kind_;
    ConeBackprojector runner_;
};

inline std::unique_ptr<IForwardOperator> makeForwardOperator(ETask kind)
{ return std::make_unique<ForwardOperator>(kind); }
inline std::unique_ptr<IBackOperator> makeBackOperator(ETask kind)
{ return std::make_unique<BackOperator>(kind); }

} // namespace YK
