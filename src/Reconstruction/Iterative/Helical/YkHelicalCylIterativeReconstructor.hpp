#pragma once

#include <vector>

#include "CylFpBp/Iter/YkCylAlgebraicReconstructor.hpp"
#include "CylFpBp/Iter/YkCylPwlsReconstructor.hpp"
#include "Reconstruction/Iterative/Helical/YkHelicalIterativeTypes.hpp"

namespace YK::Helical::Iterative {

struct CylConfig {
    EMethod method = EMethod::Ossart;
    CylFpBp::IterativeConfig algebraic{};
    CylFpBp::CylPwlsConfig pwls{};
    CylFpBp::Config operators{};
};

// 柱面螺旋迭代门面。SCylConeProjGeomVec 原样交给 Cyl 专用算子，因而
// 保留曲率、探测器姿态和逐视图 z 位置；迭代路径不要求 R=SDD，也不经过
// 解析重建使用的曲率 map、预加权和深度加权。
class CylReconstructor {
public:
    ~CylReconstructor() { release(); }
    CylReconstructor() = default;
    CylReconstructor(const CylReconstructor&) = delete;
    CylReconstructor& operator=(const CylReconstructor&) = delete;

    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const CylConfig& config, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || channels <= 0 || rows <= 0 || geometry.empty())
            return false;
        config_ = config;
        stream_ = stream;

        if (config_.method == EMethod::Pwls) {
            prepared_ = pwls_.prepare(volume, channels, rows, geometry,
                config_.pwls, config_.operators, stream, device_id);
        } else {
            auto algorithm = config_.algebraic;
            algorithm.method = iterativeMethod_(config_.method);
            prepared_ = algebraic_.prepare(volume, channels, rows, geometry,
                algorithm, config_.operators, stream, device_id);
        }
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* projection, float* volume)
    {
        if (!prepared_ || !projection || !volume) return false;
        return config_.method == EMethod::Pwls
            ? pwls_.reconstruct(projection, volume)
            : algebraic_.reconstruct(projection, volume);
    }

    // Cyl 通用迭代器已经支持子集回放和分批上传；Heli 门面只转发该能力，
    // 不额外持有全量投影。PWLS/CGLS 当前需要全批量数据，底层会明确拒绝。
    bool reconstructStreaming(
        const CylFpBp::AlgebraicReconstructor::ProjectionSubsetLoader& loader,
        float* volume)
    {
        return prepared_ && config_.method != EMethod::Pwls &&
            algebraic_.reconstructStreaming(loader, volume);
    }

    bool beginProjectionBatches()
    {
        return prepared_ && config_.method != EMethod::Pwls &&
            algebraic_.beginProjectionBatches();
    }

    bool uploadProjectionBatch(const float* projection, int first_view,
        int view_count, bool source_is_device)
    {
        return prepared_ && config_.method != EMethod::Pwls &&
            algebraic_.uploadProjectionBatch(projection, first_view,
                view_count, source_is_device);
    }

    bool reconstructUploaded(float* volume)
    {
        return prepared_ && config_.method != EMethod::Pwls &&
            algebraic_.reconstructUploaded(volume);
    }

    const Iter::IterativeConvergenceStatistics& convergenceStatistics() const
    {
        return algebraic_.convergenceStatistics();
    }

    void release()
    {
        if (stream_) YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        pwls_.release();
        algebraic_.release();
        config_ = {};
        stream_ = nullptr;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    static CylFpBp::EIterativeMethod iterativeMethod_(EMethod method)
    {
        switch (method) {
        case EMethod::Sirt: return CylFpBp::EIterativeMethod::Sirt;
        case EMethod::Sart: return CylFpBp::EIterativeMethod::Sart;
        case EMethod::Ossart: return CylFpBp::EIterativeMethod::Ossart;
        case EMethod::Cgls: return CylFpBp::EIterativeMethod::Cgls;
        default:
            return CylFpBp::EIterativeMethod::Ossart;
        }
    }

    CylConfig config_{};
    CylFpBp::AlgebraicReconstructor algebraic_{};
    CylFpBp::CylPwlsReconstructor pwls_{};
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
};

} // namespace YK::Helical::Iterative
