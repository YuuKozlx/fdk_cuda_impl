#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtGeometry.hpp"
#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtPipeline.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomographyPipeline.hpp"
#include "YKCBCT/geometry/YkPlanarGeometryBuilder.hpp"

namespace {
using namespace YK;
using namespace YK::NonStandardFpBp;

SReconstructionParams params(int nu, int nv)
{
    SReconstructionParams p{};
    p.scan.Nu = nu; p.scan.Nv = nv;
    p.scan.NAng = p.scan.totalViews = 13;
    p.scan.du_mm = p.scan.dv_mm = 1.f;
    p.scan.sid_mm = 80.f; p.scan.sdd_mm = 160.f;
    p.scan.range_rad = 2.f * CUDA_PI;
    for (int i = 0; i < p.scan.totalViews; ++i)
        p.scan.angles.push_back(p.scan.range_rad * i / p.scan.totalViews);
    p.volume.Nx = 16; p.volume.Ny = 12; p.volume.Nz = 8;
    p.volume.voxelX_mm = 0.8f; p.volume.voxelY_mm = 0.9f;
    p.volume.voxelZ_mm = 1.1f;
    p.volume.centerX_mm = 0.3f; p.volume.centerY_mm = -0.2f;
    p.volume.centerZ_mm = 0.1f;
    p.reconstruction.filter = SFilterKernelDesc::RamLak();
    return p;
}

std::vector<float> projections(int nu, int nv, int views)
{
    std::vector<float> result(static_cast<size_t>(nu) * nv * views);
    for (int i = 0; i < views; ++i)
        for (int v = 0; v < nv; ++v)
            for (int u = 0; u < nu; ++u) {
                const float x = (u - 0.5f * (nu - 1)) / nu;
                const float y = (v - 0.5f * (nv - 1)) / nv;
                result[(static_cast<size_t>(i) * nv + v) * nu + u] =
                    std::exp(-8.f * (x*x + y*y)) * (1.f + 0.07f * i) +
                    0.1f * std::sin(0.17f * u + 0.11f * v + 0.31f * i);
            }
    return result;
}

bool stagingIsBounded(const SourceOrbitTomographyPipeline& pipeline,
    size_t view_elements, int max_chunk)
{
    return pipeline.hostProjectionStagingElements() ==
        view_elements * static_cast<size_t>(max_chunk);
}
bool stagingIsBounded(const NikonTiltedCt::Pipeline&, size_t, int)
{
    return true; // Nikon uses its existing bounded chunk_in workspace.
}

template<class Pipeline>
bool checkStreaming(Pipeline& pipeline, const SReconstructionParams& p,
    int raw_u, int raw_v, cudaStream_t stream, const char* label)
{
    constexpr int max_chunk = 4;
    const int views = p.scan.totalViews;
    const size_t per_view = static_cast<size_t>(raw_u) * raw_v;
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const auto input = projections(raw_u, raw_v, views);
    Mem::MemoryController memory;
    auto d_full = memory.allocateDevice3D<float>(raw_u, raw_v, views, 0);
    auto d_packet = memory.allocateDevice3D<float>(raw_u, raw_v, 7, 0);
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    bool ok = cudaMemcpy(d_full.data(), input.data(), input.size() * sizeof(float),
        cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && pipeline.projectionWorkspaceViews() == max_chunk &&
        pipeline.process(d_full.data(), d_volume.data());
    std::vector<float> reference(volume_n), result(volume_n);
    ok = ok && cudaMemcpy(reference.data(), d_volume.data(), volume_n * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess;
    // Streaming below cannot read a whole acquisition from the device.
    d_full = {};
    double reference_max = 0.;
    for (float value : reference) {
        ok = ok && std::isfinite(value);
        reference_max = std::max(reference_max, std::abs(double(value)));
    }
    ok = ok && reference_max > 1e-6;
    double maximum_relative = 0.;
    auto compare = [&](float scale) {
        if (cudaMemcpy(result.data(), d_volume.data(), volume_n * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        double difference = 0.;
        for (size_t i = 0; i < volume_n; ++i) {
            if (!std::isfinite(result[i])) return false;
            difference = std::max(difference,
                std::abs(double(result[i]) - double(scale) * reference[i]));
        }
        const double relative = difference / (reference_max * scale);
        maximum_relative = std::max(maximum_relative, relative);
        return relative < 1e-5;
    };
    const NonStandardFdkProjectionBatch first{input.data(), 0, 1,
        EProjectionBatchLocation::Host, per_view};
    ok = ok && !pipeline.enqueueBatch(first) && !pipeline.completeStreaming();
    // Cut canonical chunks, cross multiple chunks per packet, and end short.
    for (const auto location : {EProjectionBatchLocation::Host,
            EProjectionBatchLocation::Device}) {
        ok = ok && pipeline.beginStreaming(d_volume.data()) &&
            !pipeline.beginStreaming(d_volume.data()) &&
            !pipeline.process(d_packet.data(), d_volume.data()) &&
            !pipeline.completeStreaming();
        auto invalid = first;
        invalid.first_view = 1;
        ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.element_count = per_view - 1;
        ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.view_count = 0;
        ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.view_count = views + 1;
        ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.data = nullptr;
        ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.location = static_cast<EProjectionBatchLocation>(-1);
        ok = ok && !pipeline.enqueueBatch(invalid) &&
            pipeline.receivedStreamingViews() == 0;
        int offset = 0;
        for (int count : {1, 7, 5}) {
            const float* source = input.data() + static_cast<size_t>(offset) * per_view;
            if (location == EProjectionBatchLocation::Device) {
                ok = ok && cudaMemcpyAsync(d_packet.data(), source,
                    static_cast<size_t>(count) * per_view * sizeof(float),
                    cudaMemcpyHostToDevice, stream) == cudaSuccess;
                source = d_packet.data();
            }
            const NonStandardFdkProjectionBatch batch{source, offset, count,
                location, static_cast<size_t>(count) * per_view};
            ok = ok && pipeline.enqueueBatch(batch) &&
                pipeline.receivedStreamingViews() == offset + count;
            // Duplicate/skipped packet must not consume views or alter output.
            ok = ok && !pipeline.enqueueBatch(batch);
            if (offset + count < views) ok = ok && !pipeline.completeStreaming();
            offset += count;
        }
        ok = ok && pipeline.completeStreaming() && !pipeline.streamingActive() &&
            !pipeline.completeStreaming() && !pipeline.enqueueBatch(first) && compare(1.f);
        ok = ok && stagingIsBounded(pipeline, per_view, max_chunk);
    }
    // clear_output=false preserves the caller's existing volume exactly once.
    ok = ok && pipeline.beginStreaming(d_volume.data(), false);
    const NonStandardFdkProjectionBatch all{input.data(), 0, views,
        EProjectionBatchLocation::Host, input.size()};
    ok = ok && pipeline.enqueueBatch(all) && pipeline.completeStreaming() && compare(2.f);
    // Cancellation retains preparation and permits a fresh acquisition.
    ok = ok && pipeline.beginStreaming(d_volume.data()) && pipeline.enqueueBatch(first) &&
        pipeline.cancelStreaming() && !pipeline.streamingActive() &&
        pipeline.receivedStreamingViews() == 0 &&
        pipeline.beginStreaming(d_volume.data()) && pipeline.enqueueBatch(all) &&
        pipeline.completeStreaming() && compare(1.f);
    // One-view packets exercise safe input reuse and every map/geometry offset.
    ok = ok && pipeline.beginStreaming(d_volume.data());
    std::vector<float> reusable(per_view);
    for (int i = 0; i < views; ++i) {
        std::copy_n(input.data() + static_cast<size_t>(i) * per_view,
            per_view, reusable.data());
        const NonStandardFdkProjectionBatch batch{reusable.data(), i, 1,
            EProjectionBatchLocation::Host, per_view};
        ok = ok && pipeline.enqueueBatch(batch);
        std::fill(reusable.begin(), reusable.end(), -999.f);
    }
    ok = ok && pipeline.completeStreaming() && compare(1.f);
    ok = ok && pipeline.beginStreaming(d_volume.data()) && pipeline.enqueueBatch(first);
    pipeline.release();
    ok = ok && !pipeline.streamingActive() && pipeline.receivedStreamingViews() == 0 &&
        !pipeline.beginStreaming(d_volume.data());
    std::printf("nonstandard streaming %s: max-relative=%.3e %s\n",
        label, maximum_relative, ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace

int main_nonstandard_fdk_streaming()
{
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    bool ok = true;
    for (const auto mode : {SourceOrbitResampleMode::InPlaneRotated,
            SourceOrbitResampleMode::PerspectiveOrthogonal}) {
        auto p = params(64, 64);
        constexpr int raw_u = 160, raw_v = 160;
        std::vector<SConeProjGeomVec> geometry;
        bool case_ok = buildPlanarRotatingSourceGeometry(p.scan.angles, raw_u,
            raw_v, 1.f, 1.f, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
            12.f, geometry);
        std::vector<ProjectionWindow> windows;
        for (int i = 0; i < p.scan.totalViews; ++i)
            windows.push_back({64, 64, 48 + (i % 2), 48});
        SourceOrbitTomographyPipeline pipeline;
        case_ok = case_ok && pipeline.prepare(p, geometry, windows, raw_u,
            raw_v, mode, stream, 0, 4);
        case_ok = case_ok && checkStreaming(pipeline, p, raw_u, raw_v, stream,
            mode == SourceOrbitResampleMode::InPlaneRotated ? "source-orbit/in-plane" :
                "source-orbit/perspective");
        ok = case_ok && ok;
    }
    auto p = params(64, 48);
    NikonTiltedCt::GeometryConfig config{};
    config.detector_u = p.scan.Nu; config.detector_v = p.scan.Nv;
    config.detector_pixel_u_mm = config.detector_pixel_v_mm = 1.f;
    config.sid_mm = p.scan.sid_mm; config.sdd_mm = p.scan.sdd_mm;
    config.tilt_rad = 0.35f; config.tilt_axis_azimuth_rad = 0.2f;
    config.source = make_float3(0.f, -p.scan.sid_mm, 0.f);
    std::vector<NikonTiltedCt::ViewPose> poses;
    std::vector<SConeProjGeomVec> geometry;
    bool case_ok = NikonTiltedCt::buildTiltedAxisGeometry(config,
        p.scan.angles, poses, geometry);
    // Changing per-view du must not make packet boundaries change filter scale.
    for (size_t i = 0; i < geometry.size(); ++i) {
        const float scale = 1.f + 0.01f * static_cast<float>(i);
        geometry[i].detU.x *= scale; geometry[i].detU.y *= scale;
        geometry[i].detU.z *= scale;
    }
    NikonTiltedCt::Pipeline pipeline;
    case_ok = case_ok && pipeline.prepare(p, geometry, stream, 0, 4) &&
        checkStreaming(pipeline, p, p.scan.Nu, p.scan.Nv, stream, "nikon-tilted");
    ok = case_ok && ok;
    pipeline.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
