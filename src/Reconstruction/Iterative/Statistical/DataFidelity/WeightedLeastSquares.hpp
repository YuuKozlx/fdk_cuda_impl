#pragma once
#include "Reconstruction/Iterative/Common/YkIterationWeightCache.hpp"
#include "Reconstruction/Iterative/Common/YkProjectionInput.hpp"
#include "Reconstruction/Iterative/Common/YkReconstructionMemoryBudget.hpp"
#include "Reconstruction/Iterative/Common/kernels/YkIterLaunch.cuh"
#include "Reconstruction/Iterative/Statistical/ReductionWorkspace.hpp"
#include "Reconstruction/Iterative/Statistical/StatisticalContracts.hpp"
#include "common/YkProjectionOperators.hpp"
#include <numeric>
namespace YK::Iter::Statistical {
class WeightedLeastSquares final : public IDataFidelity {
    SReconstructionParams params_ {};
    SDataFidelitySpec config_ {};
    cudaStream_t stream_ = nullptr;
    ForwardOperatorAdapter fp_;
    BackOperatorAdapter bp_;
    DeviceWorkspaceF32 prediction_;
    ReductionWorkspace reduction_;
    DeviceWorkspaceF32 row_, column_, update_, ones_;
    detail::IterationWeightCache row_cache_;
    std::vector<size_t> row_counts_;
    bool algebraic_enabled_ = false, column_ready_ = false;
    const float *measured_ = nullptr, *weights_ = nullptr;
    IProjectionDataSource *source_ = nullptr, *weight_source_ = nullptr;
    int batch_views_ = 1;
    size_t pixels_ = 0, volume_count_ = 0;
    double scale_ = 1;
    std::vector<SReconstructionParams> batches_;
    std::vector<std::vector<int>> indices_;
    std::string error_;
    bool evaluate_(const float* x, float* gradient, double* value, bool curvature)
    {
        if (!x || (!curvature && !measured_ && !source_)) {
            error_ = "missing statistical projection source";
            return false;
        }
        if (config_.use_projection_weights && !weights_ && !weight_source_) {
            error_ = "diagonal W requires a weight source";
            return false;
        }
        if (gradient && cudaMemsetAsync(gradient, 0, volume_count_ * sizeof(float), stream_) != cudaSuccess)
            return false;
        double sum = 0;
        for (size_t k = 0; k < batches_.size(); ++k) {
            const auto& ids = indices_[k];
            const float* b = curvature ? nullptr : source_ ? source_->fetch(ids, stream_)
                                                           : measured_ + size_t(ids[0]) * pixels_;
            const float* w = config_.use_projection_weights ? (weight_source_ ? weight_source_->fetch(ids, stream_) : weights_ + size_t(ids[0]) * pixels_) : nullptr;
            if ((!curvature && !b) || (config_.use_projection_weights && !w)) {
                error_ = "statistical projection/weight read failed";
                return false;
            }
            if (!fp_.run(x, batches_[k], prediction_.data(), stream_)) {
                error_ = "statistical FP failed";
                return false;
            }
            weighted_residual_launch(prediction_.data(), b, w, ids.size() * pixels_, reduction_.data(), stream_);
            double partial = 0;
            if (!reduction_.read(stream_, partial)) {
                error_ = "nonfinite projection/residual or nonfinite/negative W";
                return false;
            }
            sum += partial;
            if (gradient && !bp_.run(prediction_.data(), batches_[k], gradient, stream_, false)) {
                error_ = "statistical BP failed";
                return false;
            }
        }
        if (gradient)
            scale_launch(gradient, float(scale_), volume_count_, stream_);
        if (value)
            *value = sum * scale_;
        return cudaPeekAtLastError() == cudaSuccess && std::isfinite(sum * scale_);
    }

public:
    ~WeightedLeastSquares() { release(); }
    DataFidelityCapabilities capabilities() const override { return { true, true, true, true, AdjointStatus::Approximate, true }; }
    const std::string& error() const override { return error_; }
    static size_t estimateWorkspaceBytes(const SReconstructionParams& p)
    {
        const size_t pixels = size_t(p.scan.Nu) * p.scan.Nv, volume = size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
        // Includes projector texture capacity, residual/projection textures and metadata.
        return (volume + 3 * pixels * detail::iterativeBatchViews(pixels, p.scan.NAng)) * sizeof(float) + size_t(32) * 1024 * 1024;
    }
    size_t workspaceBytes(const SReconstructionParams& p) const override { return estimateWorkspaceBytes(p); }
    size_t algebraicWorkspaceBytes(const SReconstructionParams& p) const override
    {
        return (3 * size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz + size_t(detail::iterativeBatchViews(size_t(p.scan.Nu) * p.scan.Nv, p.scan.NAng)) * p.scan.Nu * p.scan.Nv) * sizeof(float);
    }
    bool prepare(const DataFidelitySetup& s) override
    {
        if (!prepare(s.params, s.geometry, s.config, s.fp_task, s.bp_task, s.stream, s.device))
            return false;
        algebraic_enabled_ = s.algebraic_enabled;
        if (algebraic_enabled_) {
            if (!detail::fitsReconstructionBudget(algebraicWorkspaceBytes(s.params), "algebraic data update"))
                return false;
            row_.allocate(size_t(batch_views_) * pixels_, s.device);
            column_.allocate(volume_count_, s.device);
            update_.allocate(volume_count_, s.device);
            ones_.allocate(volume_count_, s.device);
            fill_ones_launch(ones_.data(), volume_count_, stream_);
        }
        return true;
    }
    bool bind(const ProjectionDataSet& data) override { return bind(data.projection_device, data.projection_source, data.weights_device, data.weights_source); }
    bool prepare(const SReconstructionParams& p, const std::vector<SConeProjGeomVec>& geometry, const SDataFidelitySpec& cfg,
        ETask fp, ETask bp, cudaStream_t stream, int device)
    {
        release();
        if (cfg.kind != EDataFidelityKind::WeightedLeastSquares || !std::isfinite(cfg.scale) || cfg.scale <= 0 || cfg.require_verified_adjoint || p.scan.Nu <= 0 || p.scan.Nv <= 0 || p.scan.NAng <= 0 || p.volume.Nx <= 0 || p.volume.Ny <= 0 || p.volume.Nz <= 0 || p.scan.angles.size() != size_t(p.scan.NAng) || geometry.size() != size_t(p.scan.NAng))
            return false;
        params_ = p;
        config_ = cfg;
        stream_ = stream;
        pixels_ = size_t(p.scan.Nu) * p.scan.Nv;
        volume_count_ = size_t(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
        scale_ = cfg.scale / (cfg.normalize_by_ray_count ? double(pixels_) * p.scan.NAng : 1);
        batch_views_ = detail::iterativeBatchViews(pixels_, p.scan.NAng);
        if (!detail::fitsReconstructionBudget(estimateWorkspaceBytes(p), "statistical data fidelity"))
            return false;
        prediction_.allocate(size_t(batch_views_) * pixels_, device);
        return reduction_.prepare() && fp_.init(p, geometry, fp, device, stream) && bp_.init(p, geometry, bp, device, stream);
    }
    bool bind(const float* measured, IProjectionDataSource* source, const float* weights, IProjectionDataSource* weight_source)
    {
        error_.clear();
        measured_ = measured;
        source_ = source;
        weights_ = weights;
        weight_source_ = weight_source;
        int count = batch_views_;
        if (source)
            count = std::min(count, source->preferredViews());
        if (weight_source)
            count = std::min(count, weight_source->preferredViews());
        if (count <= 0) {
            error_ = "projection source preferredViews must be positive";
            return false;
        }
        if (source && !source->prepare(count, stream_)) {
            error_ = source->error();
            return false;
        }
        if (weight_source && !weight_source->prepare(count, stream_)) {
            error_ = weight_source->error();
            return false;
        }
        column_ready_ = false;
        batches_.clear();
        indices_.clear();
        for (int first = 0; first < params_.scan.NAng; first += count) {
            int n = std::min(count, params_.scan.NAng - first);
            std::vector<int> ids(n);
            std::iota(ids.begin(), ids.end(), first);
            indices_.push_back(std::move(ids));
            auto p = params_;
            p.scan.NAng = n;
            p.scan.angles.assign(params_.scan.angles.begin() + first, params_.scan.angles.begin() + first + n);
            batches_.push_back(std::move(p));
        }
        if (algebraic_enabled_) {
            std::vector<size_t> counts;
            for (const auto& ids : indices_)
                counts.push_back(ids.size() * pixels_);
            if (counts != row_counts_) {
                int device = 0;
                cudaGetDevice(&device);
                row_cache_.configure(counts, device, stream_);
                row_counts_ = std::move(counts);
            }
        }
        return true;
    }
    void unbind() override
    {
        measured_ = weights_ = nullptr;
        source_ = weight_source_ = nullptr;
    }
    bool algebraicUpdate(float* x, float relaxation) override
    {
        if (!algebraic_enabled_ || !x || !std::isfinite(relaxation) || relaxation <= 0 || (!measured_ && !source_))
            return false;
        if (config_.use_projection_weights && !weights_ && !weight_source_) {
            error_ = "diagonal W requires a source";
            return false;
        }
        auto weight = [&](size_t k) -> const float* {
            if (!config_.use_projection_weights)
                return nullptr;
            const auto& ids = indices_[k];
            const float* w = weight_source_ ? weight_source_->fetch(ids, stream_) : weights_ + size_t(ids[0]) * pixels_;
            if (!w) {
                error_ = "algebraic W read failed";
                return nullptr;
            }
            fill_ones_launch(prediction_.data(), ids.size() * pixels_, stream_);
            weighted_residual_launch(prediction_.data(), nullptr, w, ids.size() * pixels_, reduction_.data(), stream_);
            double value = 0;
            if (!reduction_.read(stream_, value)) {
                error_ = "nonfinite or negative W";
                return nullptr;
            }
            return w;
        };
        // C=B(W*1); rebuild at bind, reuse within the complete reconstruction.
        if (!column_ready_) {
            for (size_t k = 0; k < batches_.size(); ++k) {
                const auto* w = weight(k);
                if (config_.use_projection_weights && !w)
                    return false;
                if (!w)
                    fill_ones_launch(prediction_.data(), indices_[k].size() * pixels_, stream_);
                // Weighted validation above leaves W in prediction_.
                if (!bp_.run(prediction_.data(), batches_[k], column_.data(), stream_, k == 0))
                    return false;
            }
            column_ready_ = true;
        }
        for (size_t k = 0; k < batches_.size(); ++k) {
            const auto& ids = indices_[k];
            const size_t elements = ids.size() * pixels_;
            const float* row = row_cache_.get(k, row_.data(), [&](float* out) { return fp_.run(ones_.data(), batches_[k], out, stream_); });
            const float* b = source_ ? source_->fetch(ids, stream_) : measured_ + size_t(ids[0]) * pixels_;
            const float* w = weight(k);
            if (!row || !b || (config_.use_projection_weights && !w)) {
                error_ = "algebraic projection/weight read failed";
                return false;
            }
            if (!fp_.run(x, batches_[k], prediction_.data(), stream_))
                return false;
            residual_launch(b, prediction_.data(), prediction_.data(), elements, stream_);
            divide_launch(prediction_.data(), row, 1e-6f, elements, stream_);
            if (w)
                multiply_launch(prediction_.data(), w, elements, stream_);
            if (!bp_.run(prediction_.data(), batches_[k], update_.data(), stream_, k == 0))
                return false;
        }
        // Exactly one volume update after every full angle sequence.
        update_launch(x, update_.data(), column_.data(), relaxation, 1e-6f, volume_count_, stream_);
        row_cache_.planGpu();
        return cudaPeekAtLastError() == cudaSuccess;
    }
    bool evaluate(const float* x, float* gradient, double* value) override { return evaluate_(x, gradient, value, false); }
    bool estimateLipschitz(float* ones, float* curvature, double& bound) override
    {
        fill_ones_launch(ones, volume_count_, stream_);
        if (!evaluate_(ones, curvature, nullptr, true))
            return false;
        max_absolute_launch(curvature, volume_count_, reduction_.data(), stream_);
        if (!reduction_.read(stream_, bound, true))
            return false;
        bound = std::max(bound, 1e-12);
        return true;
    }
    void release()
    {
        if (stream_)
            cudaStreamSynchronize(stream_);
        fp_.release();
        bp_.release();
        prediction_.reset();
        row_cache_.clear();
        row_counts_.clear();
        row_.reset();
        column_.reset();
        update_.reset();
        ones_.reset();
        algebraic_enabled_ = column_ready_ = false;
        reduction_.release();
        source_ = weight_source_ = nullptr;
        measured_ = weights_ = nullptr;
        column_ready_ = false;
        batches_.clear();
        indices_.clear();
        stream_ = nullptr;
    }
};
}
