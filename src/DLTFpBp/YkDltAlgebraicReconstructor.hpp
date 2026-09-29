#pragma once

// Independent algebraic reconstruction driven only by SDltRayGeometry.
// Public inputs may be DLT matrices, canonical DLT rays, or the library's
// SConeProjGeomVec. All are converted once to the same ray camera; subset
// FP/BP never routes through the generic Iter reconstruction backends.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "DLTFpBp/YkDltProjectionGeometry.hpp"
#include "DLTFpBp/YkDltSiddonProjector.hpp"
#include "DLTFpBp/kernels/YkDltAlgebraicLaunch.cuh"
#include "common/YkDeviceWorkspace.hpp"

namespace YK::DltFpBp {

enum class EDltAlgebraicMethod {
    Sirt,
    Sart,
    Ossart
};

enum class EDltAlgebraicWeightModel {
    // R=A_s*1 and C=A_s^T*1 use the full-resolution DLT operator.
    ExactSubset,
    // C is averaged along z to a 2D weight image; R remains exact.
    TigreApprox
};

struct SDltAlgebraicConfig {
    EDltAlgebraicMethod method = EDltAlgebraicMethod::Ossart;
    EDltAlgebraicWeightModel weight_model =
        EDltAlgebraicWeightModel::ExactSubset;
    int ossart_subset_size = 9;
    float relaxation = 1.f;
    float epsilon = 1.e-6f;
    bool nonnegative = true;
};

class DltAlgebraicReconstructor {
public:
    DltAlgebraicReconstructor() = default;
    ~DltAlgebraicReconstructor() { release(); }
    DltAlgebraicReconstructor(const DltAlgebraicReconstructor&) = delete;
    DltAlgebraicReconstructor& operator=(const DltAlgebraicReconstructor&) = delete;

    bool prepare(const std::vector<SDltRayGeometry>& geometry,
        const SVolGeom& volume_geometry, int channels, int rows,
        const SDltAlgebraicConfig& config = {},
        cudaStream_t stream = nullptr, int device_id = 0)
    {
        return prepareCanonical_(geometry, volume_geometry, channels, rows,
            config, stream, device_id);
    }

    bool prepare(const std::vector<SDltProjectionMatrix>& projections,
        const SVolGeom& volume_geometry, int channels, int rows,
        const SDltAlgebraicConfig& config = {},
        cudaStream_t stream = nullptr, int device_id = 0)
    {
        std::vector<SDltRayGeometry> geometry;
        if (!buildRayGeometry(projections, channels, rows,
                volume_geometry.center, geometry)) return false;
        return prepareCanonical_(geometry, volume_geometry, channels, rows,
            config, stream, device_id);
    }

    bool prepare(const std::vector<SConeProjGeomVec>& geometry,
        const SVolGeom& volume_geometry, int channels, int rows,
        const SDltAlgebraicConfig& config = {},
        cudaStream_t stream = nullptr, int device_id = 0)
    {
        std::vector<SDltRayGeometry> rays;
        if (!buildRayGeometry(geometry, channels, rows,
                volume_geometry.center, rays)) return false;
        return prepareCanonical_(rays, volume_geometry, channels, rows,
            config, stream, device_id);
    }

    // One sweep visits every subset once. SIRT therefore performs one update,
    // SART one update per view, and OS-SART one update per subset.
    bool iterate(const float* device_measured_projection,
        float* device_volume, int sweeps, cudaStream_t stream = nullptr)
    {
        if (!prepared_ || !device_measured_projection || !device_volume ||
            sweeps < 0 || (stream && stream != stream_)) return false;
        const unsigned long long update_count =
            static_cast<unsigned long long>(sweeps) * subset_count_;
        if (update_count > std::numeric_limits<unsigned int>::max()) return false;
        for (unsigned int update = 0; update < update_count; ++update) {
            if (!iterateSubset_(device_measured_projection, device_volume))
                return false;
        }
        return true;
    }

    void reset()
    {
        update_index_ = 0;
        relaxation_ = config_.relaxation;
    }

    void release()
    {
        if (stream_) cudaStreamSynchronize(stream_);
        projectors_.clear();
        d_measured_subset_.reset();
        d_forward_.reset();
        d_residual_.reset();
        d_row_weight_.reset();
        d_backprojection_.reset();
        d_ones_volume_.reset();
        d_column_weight_.reset();
        d_column_weight_2d_.reset();
        geometry_.clear();
        subsets_.clear();
        if (owned_stream_) cudaStreamDestroy(owned_stream_);
        owned_stream_ = nullptr;
        stream_ = nullptr;
        volume_geometry_ = {};
        config_ = {};
        channels_ = rows_ = views_ = subset_size_ = subset_count_ = 0;
        update_index_ = 0;
        relaxation_ = 1.f;
        device_id_ = 0;
        prepared_ = false;
    }

    bool prepared() const { return prepared_; }
    EDltAlgebraicMethod method() const { return config_.method; }
    EDltAlgebraicWeightModel weightModel() const { return config_.weight_model; }
    int subsetSize() const { return subset_size_; }
    int subsetCount() const { return subset_count_; }
    unsigned int completedSubsetUpdates() const { return update_index_; }

private:
    static bool finite4xyz_(const float4& value)
    {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
            std::isfinite(value.z);
    }

    static bool validRay_(const SDltRayGeometry& ray)
    {
        if (!finite4xyz_(ray.source) || !finite4xyz_(ray.ray00) ||
            !finite4xyz_(ray.rayU) || !finite4xyz_(ray.rayV)) return false;
        const double ux = ray.rayU.x, uy = ray.rayU.y, uz = ray.rayU.z;
        const double vx = ray.rayV.x, vy = ray.rayV.y, vz = ray.rayV.z;
        const double uu = ux * ux + uy * uy + uz * uz;
        const double uv = ux * vx + uy * vy + uz * vz;
        const double vv = vx * vx + vy * vy + vz * vz;
        return uu > 0.0 && vv > 0.0 && uu * vv - uv * uv > 1e-24;
    }

    bool prepareCanonical_(const std::vector<SDltRayGeometry>& geometry,
        const SVolGeom& volume_geometry, int channels, int rows,
        const SDltAlgebraicConfig& config, cudaStream_t stream, int device_id)
    {
        release();
        if (geometry.empty() || channels <= 0 || rows <= 0 ||
            volume_geometry.Nx <= 0 || volume_geometry.Ny <= 0 ||
            volume_geometry.Nz <= 0 || !(volume_geometry.vox_x > 0.f) ||
            !(volume_geometry.vox_y > 0.f) || !(volume_geometry.vox_z > 0.f) ||
            !std::isfinite(config.relaxation) || !(config.relaxation > 0.f) ||
            !std::isfinite(config.epsilon) || !(config.epsilon > 0.f) ||
            (config.method == EDltAlgebraicMethod::Ossart &&
                (config.ossart_subset_size <= 0 ||
                 config.ossart_subset_size > static_cast<int>(geometry.size()))))
            return false;
        for (const auto& ray : geometry)
            if (!validRay_(ray)) return false;

        if (cudaSetDevice(device_id) != cudaSuccess) return false;
        if (!stream) {
            if (cudaStreamCreate(&owned_stream_) != cudaSuccess) return false;
            stream = owned_stream_;
        }
        stream_ = stream;
        device_id_ = device_id;
        geometry_ = geometry;
        volume_geometry_ = volume_geometry;
        config_ = config;
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        subset_size_ = config.method == EDltAlgebraicMethod::Sirt ? views_ :
            config.method == EDltAlgebraicMethod::Sart ? 1 :
            config.ossart_subset_size;
        subset_count_ = (views_ + subset_size_ - 1) / subset_size_;
        relaxation_ = config.relaxation;

        subsets_.resize(subset_count_);
        std::vector<std::vector<SDltRayGeometry>> subset_geometry(subset_count_);
        size_t maximum_subset_views = 0;
        for (int subset = 0; subset < subset_count_; ++subset) {
            auto& indices = subsets_[subset];
            auto& rays = subset_geometry[subset];
            for (int view = subset; view < views_; view += subset_count_) {
                indices.push_back(view);
                rays.push_back(geometry_[view]);
            }
            maximum_subset_views = std::max(maximum_subset_views, rays.size());
        }

        const size_t view_elements = static_cast<size_t>(channels_) * rows_;
        const size_t maximum_projection_elements =
            maximum_subset_views * view_elements;
        const size_t volume_elements = static_cast<size_t>(volume_geometry_.Nx) *
            volume_geometry_.Ny * volume_geometry_.Nz;
        d_measured_subset_.allocate(maximum_projection_elements, device_id_);
        d_forward_.allocate(maximum_projection_elements, device_id_);
        d_residual_.allocate(maximum_projection_elements, device_id_);
        d_row_weight_.allocate(maximum_projection_elements, device_id_);
        d_backprojection_.allocate(volume_elements, device_id_);
        d_column_weight_.allocate(volume_elements, device_id_);
        if (!d_measured_subset_ || !d_forward_ || !d_residual_ ||
            !d_row_weight_ || !d_backprojection_ || !d_column_weight_) {
            release();
            return false;
        }

        projectors_.reserve(subset_count_);
        for (int subset = 0; subset < subset_count_; ++subset) {
            auto projector = std::make_unique<DltSiddonProjector>();
            if (!projector->prepare(subset_geometry[subset], volume_geometry_,
                    channels_, rows_, device_id_)) {
                release();
                return false;
            }
            projectors_.push_back(std::move(projector));
        }

        d_ones_volume_.allocate(volume_elements, device_id_);
        if (!d_ones_volume_) { release(); return false; }
            dltFillOnesLaunch(d_ones_volume_.data(), volume_elements, stream_);
        if (config_.weight_model == EDltAlgebraicWeightModel::TigreApprox &&
            !prepareApproximateWeights_()) {
            release();
            return false;
        }

        prepared_ = true;
        return true;
    }

    bool prepareApproximateWeights_()
    {
        d_column_weight_2d_.allocate(
            static_cast<size_t>(volume_geometry_.Nx) * volume_geometry_.Ny,
            device_id_);
        if (!d_column_weight_2d_) return false;
        return true;
    }

    bool iterateSubset_(const float* measured, float* volume)
    {
        const int subset = static_cast<int>(update_index_ % subset_count_);
        const auto& indices = subsets_[subset];
        const int subset_views = static_cast<int>(indices.size());
        const size_t view_elements = static_cast<size_t>(channels_) * rows_;
        const size_t projection_elements =
            static_cast<size_t>(subset_views) * view_elements;
        const size_t volume_elements = static_cast<size_t>(volume_geometry_.Nx) *
            volume_geometry_.Ny * volume_geometry_.Nz;

        if (config_.weight_model == EDltAlgebraicWeightModel::ExactSubset) {
            if (!projectors_[subset]->forward(d_ones_volume_.data(),
                    d_row_weight_.data(), false, stream_)) return false;
            dltFillOnesLaunch(d_residual_.data(), projection_elements, stream_);
            if (!projectors_[subset]->backproject(d_residual_.data(),
                    d_column_weight_.data(), true, stream_)) return false;
        }
        else {
            // Keep the exact ray row norm here. The approximation is in the
            // inexpensive 2-D column image below.
            if (!projectors_[subset]->forward(d_ones_volume_.data(),
                    d_row_weight_.data(), false, stream_)) return false;
            dltFillOnesLaunch(d_residual_.data(), projection_elements, stream_);
            if (!projectors_[subset]->backproject(d_residual_.data(),
                    d_column_weight_.data(), true, stream_)) return false;
            dltMeanZLaunch(d_column_weight_.data(),
                d_column_weight_2d_.data(), volume_geometry_.Nx,
                volume_geometry_.Ny, volume_geometry_.Nz, stream_);
            dltThresholdInfLaunch(d_column_weight_2d_.data(), 0.f,
                static_cast<size_t>(volume_geometry_.Nx) * volume_geometry_.Ny,
                stream_);
        }

        for (int local_view = 0; local_view < subset_views; ++local_view) {
            if (cudaMemcpyAsync(d_measured_subset_.data() +
                    static_cast<size_t>(local_view) * view_elements,
                    measured + static_cast<size_t>(indices[local_view]) * view_elements,
                    view_elements * sizeof(float), cudaMemcpyDeviceToDevice,
                    stream_) != cudaSuccess) return false;
        }
        if (!projectors_[subset]->forward(volume, d_forward_.data(), false,
                stream_)) return false;
        dltResidualLaunch(d_measured_subset_.data(), d_forward_.data(),
            d_residual_.data(), projection_elements, stream_);
        dltDivideLaunch(d_residual_.data(), d_row_weight_.data(),
            config_.epsilon, projection_elements, stream_);
        if (!projectors_[subset]->backproject(d_residual_.data(),
                d_backprojection_.data(), true, stream_)) return false;

        if (config_.weight_model == EDltAlgebraicWeightModel::ExactSubset) {
            dltUpdateLaunch(volume, d_backprojection_.data(),
                d_column_weight_.data(), relaxation_, config_.epsilon,
                volume_elements, stream_);
        }
        else {
            dltUpdate2dLaunch(volume, d_backprojection_.data(),
                d_column_weight_2d_.data(), relaxation_, config_.epsilon,
                volume_geometry_.Nx, volume_geometry_.Ny,
                volume_geometry_.Nz, stream_);
        }
        if (config_.nonnegative)
            dltClampMinLaunch(volume, 0.f, volume_elements, stream_);
        ++update_index_;
        return true;
    }

    SDltAlgebraicConfig config_{};
    SVolGeom volume_geometry_{};
    std::vector<SDltRayGeometry> geometry_{};
    std::vector<std::vector<int>> subsets_{};
    std::vector<std::unique_ptr<DltSiddonProjector>> projectors_{};

    DeviceWorkspaceF32 d_measured_subset_{};
    DeviceWorkspaceF32 d_forward_{};
    DeviceWorkspaceF32 d_residual_{};
    DeviceWorkspaceF32 d_row_weight_{};
    DeviceWorkspaceF32 d_backprojection_{};
    DeviceWorkspaceF32 d_ones_volume_{};
    DeviceWorkspaceF32 d_column_weight_{};
    DeviceWorkspaceF32 d_column_weight_2d_{};

    cudaStream_t stream_ = nullptr;
    cudaStream_t owned_stream_ = nullptr;
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int subset_size_ = 0;
    int subset_count_ = 0;
    int device_id_ = 0;
    unsigned int update_index_ = 0;
    float relaxation_ = 1.f;
    bool prepared_ = false;
};

} // namespace YK::DltFpBp
