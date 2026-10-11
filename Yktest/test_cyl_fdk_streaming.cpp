#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "CylFpBp/Analytic/YkCylFdkPipeline.hpp"

namespace {
using namespace YK;
using namespace YK::CylFpBp::Analytic;
constexpr int channels = 64, rows = 48, views = 13, chunk = 4;

std::vector<float> inputProjection()
{
    std::vector<float> result(size_t(channels) * rows * views);
    for (int i = 0; i < views; ++i)
        for (int v = 0; v < rows; ++v)
            for (int u = 0; u < channels; ++u) {
                const float x = (u - 31.5f) / channels, y = (v - 23.5f) / rows;
                result[(size_t(i) * rows + v) * channels + u] =
                    std::exp(-8.f * (x*x+y*y)) * (1.f+0.07f*i) +
                    0.1f * std::sin(0.17f*u+0.11f*v+0.31f*i);
            }
    return result;
}

template<class Pipeline>
bool checkPackets(Pipeline& pipeline, const SVolGeom& volume,
    cudaStream_t stream, const char* label)
{
    const auto input = inputProjection();
    const size_t per_view = size_t(channels) * rows;
    const size_t volume_n = size_t(volume.Nx) * volume.Ny * volume.Nz;
    Mem::MemoryController memory;
    auto d_full = memory.allocateDevice3D<float>(channels, rows, views, 0);
    auto d_packet = memory.allocateDevice3D<float>(channels, rows, 7, 0);
    auto d_volume = memory.allocateDevice3D<float>(volume.Nx, volume.Ny, volume.Nz, 0);
    bool ok = cudaMemcpy(d_full.data(), input.data(), input.size()*sizeof(float),
        cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && pipeline.projectionWorkspaceViews() == chunk &&
        pipeline.reconstruct(d_full.data(), d_volume.data());
    std::vector<float> reference(volume_n), result(volume_n);
    ok = ok && cudaMemcpy(reference.data(), d_volume.data(), volume_n*sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess;
    d_full = {}; // All subsequent calls use only packet-sized device input.
    double peak = 0., max_relative = 0.;
    for (float value : reference) {
        ok = ok && std::isfinite(value);
        peak = std::max(peak, std::abs(double(value)));
    }
    ok = ok && peak > 1e-6;
    auto compare = [&](float scale) {
        if (cudaMemcpy(result.data(), d_volume.data(), volume_n*sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        double error = 0.;
        for (size_t i = 0; i < volume_n; ++i) {
            if (!std::isfinite(result[i])) return false;
            error = std::max(error, std::abs(double(result[i])-scale*reference[i]));
        }
        const double relative = error/(scale*peak);
        max_relative = std::max(max_relative, relative);
        return relative < 1e-5;
    };
    const CylFdkProjectionBatch first{input.data(), 0, 1,
        EMemoryLocation::Host, per_view};
    ok = ok && !pipeline.enqueueBatch(first) && !pipeline.completeStreaming();
    for (const auto location : {EMemoryLocation::Host, EMemoryLocation::Device}) {
        ok = ok && pipeline.beginStreaming(d_volume.data()) &&
            !pipeline.beginStreaming(d_volume.data()) &&
            !pipeline.reconstruct(d_packet.data(), d_volume.data()) &&
            !pipeline.completeStreaming();
        auto invalid = first;
        invalid.first_view = 1; ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.element_count = per_view-1;
        ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.view_count = 0; ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.view_count = views+1; ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.data = nullptr; ok = ok && !pipeline.enqueueBatch(invalid);
        invalid = first; invalid.location = static_cast<EMemoryLocation>(-1);
        ok = ok && !pipeline.enqueueBatch(invalid) && pipeline.receivedStreamingViews() == 0;
        int offset = 0;
        for (int count : {1, 7, 5}) {
            const float* source = input.data()+size_t(offset)*per_view;
            if (location == EMemoryLocation::Device) {
                ok = ok && cudaMemcpyAsync(d_packet.data(), source,
                    size_t(count)*per_view*sizeof(float), cudaMemcpyHostToDevice,
                    stream) == cudaSuccess;
                source = d_packet.data();
            }
            const CylFdkProjectionBatch batch{source, offset, count, location,
                size_t(count)*per_view};
            ok = ok && pipeline.enqueueBatch(batch) &&
                pipeline.receivedStreamingViews() == offset+count &&
                !pipeline.enqueueBatch(batch);
            offset += count;
            if (offset < views) ok = ok && !pipeline.completeStreaming();
        }
        ok = ok && pipeline.completeStreaming() && !pipeline.streamingActive() &&
            !pipeline.completeStreaming() && !pipeline.enqueueBatch(first) && compare(1.f);
    }
    const CylFdkProjectionBatch all{input.data(), 0, views,
        EMemoryLocation::Host, input.size()};
    ok = ok && pipeline.beginStreaming(d_volume.data(), false) &&
        pipeline.enqueueBatch(all) && pipeline.completeStreaming() && compare(2.f);
    ok = ok && pipeline.beginStreaming(d_volume.data()) && pipeline.enqueueBatch(first) &&
        pipeline.cancelStreaming() && pipeline.receivedStreamingViews() == 0 &&
        !pipeline.streamingActive() && pipeline.beginStreaming(d_volume.data());
    std::vector<float> reusable(per_view);
    for (int i = 0; i < views; ++i) {
        std::copy_n(input.data()+size_t(i)*per_view, per_view, reusable.data());
        const CylFdkProjectionBatch batch{reusable.data(), i, 1,
            EMemoryLocation::Host, per_view};
        ok = ok && pipeline.enqueueBatch(batch);
        std::fill(reusable.begin(), reusable.end(), -999.f);
    }
    ok = ok && pipeline.completeStreaming() && compare(1.f);
    ok = ok && pipeline.beginStreaming(d_volume.data()) && pipeline.enqueueBatch(first);
    pipeline.release();
    ok = ok && !pipeline.streamingActive() && !pipeline.isPrepared() &&
        pipeline.projectionWorkspaceViews() == 0 && !pipeline.beginStreaming(d_volume.data());
    std::printf("Cyl FDK streaming %s: max-relative=%.3e %s\n", label,
        max_relative, ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace

int main_cyl_fdk_streaming()
{
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    auto volume = SVolGeom::make_centered(16, 12, 8, 0.8f, 0.9f, 1.1f);
    volume.center = make_float3(0.3f, -0.2f, 0.1f);
    bool ok = true;
    for (float radius : {160.f, 120.f, 220.f}) {
        SCircularTrajectorySpec trajectory{};
        trajectory.sid_mm = 80.f; trajectory.sdd_mm = 160.f;
        for (int i = 0; i < views; ++i)
            trajectory.angles_rad.push_back(2.f*CUDA_PI*i/views);
        SCylDetectorSpec detector{};
        detector.channels = channels; detector.rows = rows;
        detector.channel_arc_mm = detector.row_size_mm = 1.f;
        detector.curvature_radius_mm = radius;
        detector.pose.offset_unv_mm.z = radius == 160.f ? 0.f : 1.6f;
        std::vector<SCylConeProjGeomVec> geometry;
        bool case_ok = buildProjectionGeometry(trajectory, detector, geometry);
        ReconstructionConfig config{};
        config.source_to_detector_mm = 160.f; config.max_chunk_views = chunk;
        Reconstruction pipeline;
        case_ok = case_ok && pipeline.prepare(volume, channels, rows, geometry,
            config, SFilterKernelDesc::RamLak(), stream) &&
            checkPackets(pipeline, volume, stream, radius == 160.f ? "R=SDD" :
                radius == 120.f ? "R<SDD,V-offset" : "R>SDD,V-offset");
        pipeline.release();
        ok = case_ok && ok;
        if (radius == 120.f) {
            CylFpBp::CylFdkPipeline legacy;
            bool legacy_ok = legacy.prepare(volume, channels, rows, geometry,
                SFilterKernelDesc::RamLak(), stream, 0, chunk) &&
                checkPackets(legacy, volume, stream, "legacy-facade");
            legacy.release();
            ok = legacy_ok && ok;
        }
    }
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
