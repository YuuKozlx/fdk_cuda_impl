#pragma once

#include <memory>
#include <vector>

#include "Heli/iter/YkHelicalIterativeTypes.hpp"
#include "Iter/YkAlgebraicReconstructorEx.hpp"
#include "Iter/YkCglsReconstructorEx.hpp"
#include "Iter/YkPwlsReconstructor.hpp"
#include "global/YkLog.h"

namespace YK::Helical::Iterative {

struct FlatConfig {
    EMethod method = EMethod::Ossart;
    Iter::AlgebraicReconstructionConfig algebraic{};
    Iter::CglsReconstructionConfig cgls{};
    Iter::PwlsConfig pwls{};
};

// 平板螺旋迭代门面。逐视图 SConeProjGeomVec 是唯一几何来源；本类只
// 选择通用迭代实现，不从 SID、角度等标称参数重新生成圆轨迹。
class FlatReconstructor {
public:
    ~FlatReconstructor() { release(); }
    FlatReconstructor() = default;
    FlatReconstructor(const FlatReconstructor&) = delete;
    FlatReconstructor& operator=(const FlatReconstructor&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const FlatConfig& config, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || params.scan.NAng <= 0 ||
            geometry.size() != static_cast<size_t>(params.scan.NAng))
            return false;

        params_ = params;
        geometry_ = geometry;
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;

        switch (config_.method) {
        case EMethod::Pwls:
            pwls_ = std::make_unique<Iter::PwlsReconstructor>();
            prepared_ = pwls_->prepare(params_, geometry_, config_.pwls,
                stream_, device_id_);
            break;
        case EMethod::Cgls:
            cgls_ = std::make_unique<Iter::CglsReconstructorEx>();
            prepared_ = cgls_->prepare(params_, geometry_, config_.cgls,
                stream_, device_id_);
            break;
        case EMethod::Sirt:
        case EMethod::Sart:
        case EMethod::Ossart: {
            algebraic_ = std::make_unique<Iter::AlgebraicReconstructorEx>();
            auto algorithm = config_.algebraic;
            algorithm.method = algebraicMethod_(config_.method);
            prepared_ = algebraic_->prepare(params_, geometry_, algorithm,
                stream_, device_id_);
            break;
        }
        }
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* projection, float* volume)
    {
        if (!prepared_ || !projection || !volume) return false;
        if (pwls_) return pwls_->reconstruct(projection, volume);
        if (cgls_) return cgls_->reconstruct(projection, volume);
        return algebraic_ && algebraic_->reconstruct(projection, volume);
    }

    void release()
    {
        if (stream_) YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (pwls_) pwls_->release();
        if (cgls_) cgls_->release();
        if (algebraic_) algebraic_->release();
        pwls_.reset();
        cgls_.reset();
        algebraic_.reset();
        geometry_.clear();
        params_ = {};
        config_ = {};
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    static Iter::EAlgebraicMethod algebraicMethod_(EMethod method)
    {
        switch (method) {
        case EMethod::Sirt: return Iter::EAlgebraicMethod::Sirt;
        case EMethod::Sart: return Iter::EAlgebraicMethod::Sart;
        case EMethod::Ossart: return Iter::EAlgebraicMethod::Ossart;
        default:
            // CGLS/PWLS 在 prepare() 中走独立分支，不允许降级为 SIRT。
            return Iter::EAlgebraicMethod::Ossart;
        }
    }

    SReconstructionParams params_{};
    std::vector<SConeProjGeomVec> geometry_{};
    FlatConfig config_{};
    std::unique_ptr<Iter::AlgebraicReconstructorEx> algebraic_{};
    std::unique_ptr<Iter::CglsReconstructorEx> cgls_{};
    std::unique_ptr<Iter::PwlsReconstructor> pwls_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
};

} // namespace YK::Helical::Iterative
