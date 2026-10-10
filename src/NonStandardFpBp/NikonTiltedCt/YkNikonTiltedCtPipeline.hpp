#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "NonStandardFpBp/Geometry/YkNonStandardFdkDerivedGeometry.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkBackProjectProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKGpuContext.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKFilterProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKPreWeightProcessor.hpp"
#include "global/YkCBCTParams.h"

namespace YK::NonStandardFpBp::NikonTiltedCt {

// Analytic tilted-CT scheduler. Projections stay on their acquired ConeVec
// detector grids: cosine preweighting, U-direction ramp filtering and exact
// ConeVec backprojection are applied without a standard-FDK rebin.
class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline() { release(); }
    Pipeline(const Pipeline&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!stream || params.scan.Nu <= 0 || params.scan.Nv <= 0 ||
            params.scan.totalViews <= 0 ||
            static_cast<int>(geometry.size()) != params.scan.totalViews)
            return false;

        params_ = params;
        geometry_ = geometry;
        stream_ = stream;
        chunk_size_ = std::min(params.scan.totalViews, kMaxChunkAng);
        if (!buildNonStandardFdkDerivedGeometry(params.scan.Nu, params.scan.Nv,
                params.scan.range_rad,
                make_float3(params.volume.centerX_mm,
                    params.volume.centerY_mm, params.volume.centerZ_mm),
                geometry_, derived_)) {
            release();
            return false;
        }

        const SProjDims dims{params.scan.Nu, params.scan.Nv, chunk_size_};
        gpu_.init(dims, params.scan.totalViews, stream_, device_id);
        gpu_.uploadGeoIncremental(geometry_, derived_, 0,
            params.scan.totalViews, stream_);
        PreweightConfig preweight_config{};
        preweight_config.dims = dims;
        if (!preweight_.prepare(preweight_config)) { release(); return false; }
        FdkFilterConfig filter_config{};
        filter_config.dims = dims;
        filter_config.desc = params.reconstruction.filter;
        filter_config.stream = stream_;
        if (!filter_.prepare(filter_config)) { release(); return false; }
        BpConfig bp_config{};
        bp_config.vol_geom = SVolGeom::make_centered(params.volume.Nx,
            params.volume.Ny, params.volume.Nz, params.volume.voxelX_mm,
            params.volume.voxelY_mm, params.volume.voxelZ_mm);
        bp_config.vol_geom.center = make_float3(params.volume.centerX_mm,
            params.volume.centerY_mm, params.volume.centerZ_mm);
        bp_config.use_precomputed = true;
        bp_config.max_chunk_views = chunk_size_;
        if (!backproject_.prepare(bp_config)) { release(); return false; }
        prepared_ = true;
        return true;
    }

    bool process(const float* d_projection, float* d_volume,
        bool clear_output = true)
    {
        if (!prepared_ || !d_projection || !d_volume) return false;
        const size_t volume_elements = static_cast<size_t>(params_.volume.Nx) *
            params_.volume.Ny * params_.volume.Nz;
        if (clear_output && cudaMemsetAsync(d_volume, 0,
                volume_elements * sizeof(float), stream_) != cudaSuccess)
            return false;
        const size_t view_elements = static_cast<size_t>(params_.scan.Nu) *
            params_.scan.Nv;
        for (int base = 0; base < params_.scan.totalViews; base += chunk_size_) {
            const int count = std::min(chunk_size_, params_.scan.totalViews - base);
            if (cudaMemcpyAsync(gpu_.proj.chunk_in.data(),
                    d_projection + static_cast<size_t>(base) * view_elements,
                    static_cast<size_t>(count) * view_elements * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)
                return false;
            const auto* d_geo = gpu_.geo.d_geo() + base;
            const auto* d_derived = gpu_.geo.d_gv() + base;
            if (!preweight_.apply(gpu_.proj.chunk_in.data(),
                    gpu_.proj.chunk_pw.data(), {d_geo, d_derived, count}, stream_))
                return false;
            if (!filter_.apply(gpu_.proj.chunk_pw.data(),
                    gpu_.proj.chunk_flt.data(), {derived_.data() + base, count},
                    stream_)) return false;
            gpu_.geo.uploadCoeffsChunk(gpu_.geo.d_coeffs() + base, count, stream_);
            if (!backproject_.apply(gpu_.proj.d_texObjs(), d_volume,
                    {d_geo, d_derived, count}, stream_)) return false;
        }
        return cudaStreamSynchronize(stream_) == cudaSuccess;
    }

    const std::vector<SConeProjGeomVec>& geometry() const { return geometry_; }

    void release()
    {
        if (stream_) cudaStreamSynchronize(stream_);
        preweight_.release();
        filter_.release();
        backproject_.release();
        gpu_.release();
        params_ = {};
        geometry_.clear();
        derived_.clear();
        chunk_size_ = 0;
        prepared_ = false;
        stream_ = nullptr;
    }

private:
    FdkGpuContext gpu_{};
    Fdk::PreweightProcessor preweight_{};
    Fdk::FilterProcessor filter_{};
    Fdk::BpProcessor backproject_{};
    SReconstructionParams params_{};
    std::vector<SConeProjGeomVec> geometry_{};
    std::vector<SFDKGeoParamPerView> derived_{};
    int chunk_size_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
};

} // namespace YK::NonStandardFpBp::NikonTiltedCt
