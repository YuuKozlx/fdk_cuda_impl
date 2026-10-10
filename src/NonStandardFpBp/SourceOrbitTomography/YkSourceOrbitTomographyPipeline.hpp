#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomography.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitDerivedGeometry.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/kernels/YkSourceOrbitResampleKernels.cuh"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkBackProjectProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKGpuContext.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKFilterProcessor.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKPreWeightProcessor.hpp"
#include "global/YkCBCTParams.h"

namespace YK::NonStandardFpBp {

// Fixed-detector/source-orbit reconstruction scheduler.  The projection
// buffer is [view][raw_v][raw_u] on device. Each fixed-detector view is mapped
// to a common local grid before the existing analytic stages are invoked.
class SourceOrbitTomographyPipeline {
public:
    SourceOrbitTomographyPipeline() = default;
    ~SourceOrbitTomographyPipeline() { release(); }
    SourceOrbitTomographyPipeline(const SourceOrbitTomographyPipeline&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& raw_geometry,
        const std::vector<ProjectionWindow>& windows,
        int raw_u, int raw_v, cudaStream_t stream, int device_id = 0)
    {
        return prepare(params, raw_geometry, windows, raw_u, raw_v,
            SourceOrbitResampleMode::InPlaneRotated, stream, device_id);
    }

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& raw_geometry,
        const std::vector<ProjectionWindow>& windows,
        int raw_u, int raw_v, SourceOrbitResampleMode mode,
        cudaStream_t stream, int device_id = 0)
    {
        release();
        if (raw_u <= 0 || raw_v <= 0 || params.scan.Nu <= 0 ||
            params.scan.Nv <= 0 || params.scan.totalViews <= 0 ||
            static_cast<int>(raw_geometry.size()) != params.scan.totalViews ||
            static_cast<int>(windows.size()) != params.scan.totalViews ||
            !stream) return false;
        for (const auto& window : windows) {
            if (window.width != params.scan.Nu || window.height != params.scan.Nv ||
                window.origin_u < 0 || window.origin_v < 0 ||
                window.origin_u + window.width > raw_u ||
                window.origin_v + window.height > raw_v) return false;
        }

        geometry_ = raw_geometry;
        std::vector<SourceOrbitResampleMap> maps;
        maps.reserve(windows.size());
        const float reference_angle = raw_geometry.front().angle.x;
        const float3 reconstruction_center = make_float3(
            params.volume.centerX_mm, params.volume.centerY_mm,
            params.volume.centerZ_mm);
        const auto xyz = [](float4 value) {
            return make_float3(value.x, value.y, value.z);
        };
        const auto dot = [](float3 a, float3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        };
        const auto add = [](float3 a, float3 b) {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        };
        const auto sub = [](float3 a, float3 b) {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        };
        const auto mul = [](float3 a, float value) {
            return make_float3(a.x * value, a.y * value, a.z * value);
        };
        const auto cross = [](float3 a, float3 b) {
            return make_float3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        };
        const auto normalized = [&](float3 value) {
            const float length = std::sqrt(dot(value, value));
            return length > 1e-12f ? mul(value, 1.f / length) :
                make_float3(0.f, 0.f, 0.f);
        };
        for (size_t i = 0; i < windows.size(); ++i) {
            const auto& w = windows[i];
            const auto& raw = raw_geometry[i];
            auto& mapped = geometry_[i];
            const float du = std::sqrt(raw.detU.x * raw.detU.x +
                raw.detU.y * raw.detU.y + raw.detU.z * raw.detU.z);
            const float dv = std::sqrt(raw.detV.x * raw.detV.x +
                raw.detV.y * raw.detV.y + raw.detV.z * raw.detV.z);
            if (!(du > 0.f) || !(dv > 0.f)) return false;
            SProjectionFrame physical_frame{};
            if (!deriveProjectionFrame(raw, raw_u, raw_v, physical_frame))
                return false;
            const float3 source = xyz(raw.src);
            const float3 raw_start = xyz(raw.detS);
            const float3 raw_u_step = xyz(raw.detU);
            const float3 raw_v_step = xyz(raw.detV);
            const float uu = dot(raw_u_step, raw_u_step);
            const float vv = dot(raw_v_step, raw_v_step);
            const float uv = dot(raw_u_step, raw_v_step);
            const float gram = uu * vv - uv * uv;
            if (!(gram > 1e-20f)) return false;
            const auto raw_coordinates = [&](float3 vector) {
                const float vu = dot(vector, raw_u_step);
                const float vv_dot = dot(vector, raw_v_step);
                return make_float2((vu * vv - vv_dot * uv) / gram,
                    (vv_dot * uu - vu * uv) / gram);
            };
            const float delta = raw.angle.x - reference_angle;
            const float c = std::cos(delta), s = std::sin(delta);
            const float center_u = w.origin_u + 0.5f * (w.width - 1);
            const float center_v = w.origin_v + 0.5f * (w.height - 1);
            const float3 raw_u_unit = mul(raw_u_step, 1.f / du);
            const float3 raw_v_unit = mul(raw_v_step, 1.f / dv);
            const float3 candidate_u = add(mul(raw_u_unit, c),
                mul(raw_v_unit, s));
            const float3 candidate_v = add(mul(raw_u_unit, -s),
                mul(raw_v_unit, c));
            if (mode == SourceOrbitResampleMode::InPlaneRotated) {
                const float3 mapped_center = add(raw_start,
                    add(mul(raw_u_step, center_u), mul(raw_v_step, center_v)));
                const float3 mapped_u = mul(candidate_u, du);
                const float3 mapped_v = mul(candidate_v, dv);
                const float3 mapped_start = sub(mapped_center,
                    add(mul(mapped_u, 0.5f * (w.width - 1)),
                        mul(mapped_v, 0.5f * (w.height - 1))));
                mapped.detS = make_float4(mapped_start.x, mapped_start.y,
                    mapped_start.z, 0.f);
                mapped.detU = make_float4(mapped_u.x, mapped_u.y, mapped_u.z, 0.f);
                mapped.detV = make_float4(mapped_v.x, mapped_v.y, mapped_v.z, 0.f);
            } else {
                const float3 radial = normalized(sub(reconstruction_center, source));
                const float radial_plane_dot = dot(radial,
                    physical_frame.detectorNormal);
                if (!(radial_plane_dot > 1e-8f)) return false;
                const float distance = physical_frame.planeDistance /
                    radial_plane_dot;
                const float3 mapped_center = add(source, mul(radial, distance));
                float3 mapped_u_unit = normalized(sub(candidate_u,
                    mul(radial, dot(candidate_u, radial))));
                float3 mapped_v_unit = normalized(cross(mapped_u_unit, radial));
                if (dot(mapped_v_unit, candidate_v) < 0.f)
                    mapped_v_unit = mul(mapped_v_unit, -1.f);
                if (dot(mapped_u_unit, mapped_u_unit) < 0.99f ||
                    dot(mapped_v_unit, mapped_v_unit) < 0.99f) return false;
                const auto center_mapping_scale = [&](float3 direction) {
                    const float3 physical_delta = sub(direction, mul(radial,
                        dot(direction, physical_frame.detectorNormal) /
                            radial_plane_dot));
                    const float2 pixels_per_mm = raw_coordinates(physical_delta);
                    return std::hypot(pixels_per_mm.x, pixels_per_mm.y);
                };
                const float scale_u = center_mapping_scale(mapped_u_unit);
                const float scale_v = center_mapping_scale(mapped_v_unit);
                if (!(scale_u > 1e-8f) || !(scale_v > 1e-8f)) return false;
                const float3 mapped_u = mul(mapped_u_unit, 1.f / scale_u);
                const float3 mapped_v = mul(mapped_v_unit, 1.f / scale_v);
                const float3 mapped_start = sub(mapped_center,
                    add(mul(mapped_u, 0.5f * (w.width - 1)),
                        mul(mapped_v, 0.5f * (w.height - 1))));
                mapped.detS = make_float4(mapped_start.x, mapped_start.y,
                    mapped_start.z, 0.f);
                mapped.detU = make_float4(mapped_u.x, mapped_u.y, mapped_u.z, 0.f);
                mapped.detV = make_float4(mapped_v.x, mapped_v.y, mapped_v.z, 0.f);
            }

            const float3 mapped_start = xyz(mapped.detS);
            const float3 mapped_u = xyz(mapped.detU);
            const float3 mapped_v = xyz(mapped.detV);
            const float3 ray_start = sub(mapped_start, source);
            const float2 source_coordinates = raw_coordinates(sub(source, raw_start));
            const auto map_coefficient = [&](float3 ray_component) {
                const float denominator = dot(ray_component,
                    physical_frame.detectorNormal);
                const float2 ray_coordinates = raw_coordinates(ray_component);
                return make_float3(denominator,
                    source_coordinates.x * denominator +
                        physical_frame.planeDistance * ray_coordinates.x,
                    source_coordinates.y * denominator +
                        physical_frame.planeDistance * ray_coordinates.y);
            };
            const float3 coefficient_0 = map_coefficient(ray_start);
            const float3 coefficient_u = map_coefficient(mapped_u);
            const float3 coefficient_v = map_coefficient(mapped_v);
            SourceOrbitResampleMap map{};
            map.denominator = make_float3(coefficient_0.x,
                coefficient_u.x, coefficient_v.x);
            map.numerator_u = make_float3(coefficient_0.y,
                coefficient_u.y, coefficient_v.y);
            map.numerator_v = make_float3(coefficient_0.z,
                coefficient_u.z, coefficient_v.z);
            const auto map_pixel = [&](float u, float v) {
                const float denominator = map.denominator.x +
                    u * map.denominator.y + v * map.denominator.z;
                return make_float2((map.numerator_u.x + u * map.numerator_u.y +
                    v * map.numerator_u.z) / denominator,
                    (map.numerator_v.x + u * map.numerator_v.y +
                    v * map.numerator_v.z) / denominator);
            };
            for (int corner_v : {0, w.height - 1}) {
                for (int corner_u : {0, w.width - 1}) {
                    const float2 raw_corner = map_pixel(
                        static_cast<float>(corner_u), static_cast<float>(corner_v));
                    if (!std::isfinite(raw_corner.x) || !std::isfinite(raw_corner.y) ||
                        raw_corner.x < -1e-4f ||
                        raw_corner.x > raw_u - 1.f + 1e-4f ||
                        raw_corner.y < -1e-4f ||
                        raw_corner.y > raw_v - 1.f + 1e-4f)
                        return false;
                }
            }
            maps.push_back(map);
        }

        stream_ = stream;
        mode_ = mode;
        raw_u_ = raw_u; raw_v_ = raw_v;
        params_ = params;
        chunk_size_ = std::min(params.scan.totalViews, kMaxChunkAng);
        if (!buildSourceOrbitDerivedGeometry(params.scan.Nu, params.scan.Nv,
                params.scan.range_rad, make_float3(params.volume.centerX_mm,
                    params.volume.centerY_mm, params.volume.centerZ_mm),
                geometry_, derived_)) return false;
        if (cudaMalloc(&d_maps_, maps.size() * sizeof(SourceOrbitResampleMap)) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync(d_maps_, maps.data(),
                maps.size() * sizeof(SourceOrbitResampleMap), cudaMemcpyHostToDevice, stream_) != cudaSuccess)
            return false;

        const SProjDims dims{params.scan.Nu, params.scan.Nv, chunk_size_};
        gpu_.init(dims, params.scan.totalViews, stream_, device_id);
        gpu_.uploadGeoIncremental(geometry_, derived_, 0,
            params.scan.totalViews, stream_);
        PreweightConfig preweight_config{};
        preweight_config.dims = dims;
        if (!preweight_.prepare(preweight_config)) return false;
        FdkFilterConfig filter_config{};
        filter_config.dims = dims;
        filter_config.desc = params.reconstruction.filter;
        filter_config.stream = stream_;
        if (!filter_.prepare(filter_config)) return false;
        BpConfig bp_config{};
        bp_config.vol_geom = SVolGeom::make_centered(params.volume.Nx,
            params.volume.Ny, params.volume.Nz, params.volume.voxelX_mm,
            params.volume.voxelY_mm, params.volume.voxelZ_mm);
        bp_config.vol_geom.center = make_float3(params.volume.centerX_mm,
            params.volume.centerY_mm, params.volume.centerZ_mm);
        bp_config.use_precomputed = true;
        bp_config.max_chunk_views = chunk_size_;
        if (!backproject_.prepare(bp_config)) return false;
        prepared_ = true;
        return true;
    }

    bool process(const float* d_raw_projection, float* d_volume,
        bool clear_output = true)
    {
        if (!prepared_ || !d_raw_projection || !d_volume) return false;
        if (clear_output) {
            const size_t bytes = static_cast<size_t>(params_.volume.Nx) *
                params_.volume.Ny * params_.volume.Nz * sizeof(float);
            if (cudaMemsetAsync(d_volume, 0, bytes, stream_) != cudaSuccess)
                return false;
        }
        for (int base = 0; base < params_.scan.totalViews; base += chunk_size_) {
            const int count = std::min(chunk_size_, params_.scan.totalViews - base);
            if (!resample(d_raw_projection, gpu_.proj.chunk_in.data(),
                    base, count)) return false;
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

    // Resample one contiguous range into [view][local_v][local_u]. The source
    // remains the full [view][raw_v][raw_u] acquisition buffer.
    bool resample(const float* d_raw_projection, float* d_resampled,
        int first_view = 0, int view_count = -1) const
    {
        if (!prepared_ || !d_raw_projection || !d_resampled ||
            first_view < 0 || first_view >= params_.scan.totalViews)
            return false;
        if (view_count < 0) view_count = params_.scan.totalViews - first_view;
        if (view_count <= 0 || first_view + view_count > params_.scan.totalViews)
            return false;
        return launchSourceOrbitResample(d_raw_projection, d_resampled,
            raw_u_, raw_v_, params_.scan.Nu, params_.scan.Nv, first_view,
            view_count, d_maps_, stream_);
    }

    const std::vector<SConeProjGeomVec>& mappedGeometry() const
    { return geometry_; }

    SourceOrbitResampleMode resampleMode() const { return mode_; }

    void release()
    {
        if (stream_) cudaStreamSynchronize(stream_);
        preweight_.release();
        filter_.release();
        backproject_.release();
        gpu_.release();
        if (d_maps_) cudaFree(d_maps_);
        d_maps_ = nullptr;
        geometry_.clear(); derived_.clear();
        params_ = {};
        raw_u_ = raw_v_ = chunk_size_ = 0;
        mode_ = SourceOrbitResampleMode::InPlaneRotated;
        prepared_ = false; stream_ = nullptr;
    }

private:
    FdkGpuContext gpu_{};
    Fdk::PreweightProcessor preweight_{};
    Fdk::FilterProcessor filter_{};
    Fdk::BpProcessor backproject_{};
    SReconstructionParams params_{};
    std::vector<SConeProjGeomVec> geometry_{};
    std::vector<SFDKGeoParamPerView> derived_{};
    int raw_u_ = 0, raw_v_ = 0, chunk_size_ = 0;
    cudaStream_t stream_ = nullptr;
    SourceOrbitResampleMap* d_maps_ = nullptr;
    SourceOrbitResampleMode mode_ = SourceOrbitResampleMode::InPlaneRotated;
    bool prepared_ = false;
};

} // namespace YK::NonStandardFpBp
