#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include "YkTestPhantoms.hpp"
#include "YkTestImage.hpp"
#include "common/YkProjectionOperators.hpp"
#include "YKCBCT/geometry/YkPlanarGeometryBuilder.hpp"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomographyPipeline.hpp"

namespace {
using namespace YK;

SReconstructionParams makePropertyParams()
{
    SReconstructionParams p{};
    p.scan.Nu = 32; p.scan.Nv = 24; p.scan.NAng = 12; p.scan.totalViews = 12;
    p.volume.Nx = 16; p.volume.Ny = 16; p.volume.Nz = 12;
    p.scan.du_mm = p.scan.dv_mm = 1.f;
    p.volume.voxelX_mm = p.volume.voxelY_mm = p.volume.voxelZ_mm = 1.f;
    p.scan.sid_mm = 80.f; p.scan.sdd_mm = 160.f; p.scan.range_rad = 2.f * CUDA_PI;
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i) p.scan.angles[i] = 2.f * CUDA_PI * i / p.scan.NAng;
    return p;
}
double dot(const std::vector<float>& a, const std::vector<float>& b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) result += static_cast<double>(a[i]) * b[i];
    return result;
}

bool finite(const std::vector<float>& values)
{
    for (float value : values) if (!std::isfinite(value)) return false;
    return true;
}

bool projectAndBackproject(const SReconstructionParams& p, const std::vector<SConeProjGeomVec>& geometry,
    ETask fp_task, ETask bp_task, const std::vector<float>& h_volume,
    const std::vector<float>& h_sino, std::vector<float>& out_projection,
    std::vector<float>& out_volume)
{
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t sino_n = static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv;
    cudaStream_t stream = nullptr;
    Mem::MemoryController memory;
    // All test allocations use the library allocator: ownership and the
    // canonical [z][y][x] / [view][v][u] layouts stay identical to production.
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
    auto d_sino_input = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0);
    auto d_backprojection = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
    bool ok = cudaStreamCreate(&stream) == cudaSuccess &&
        cudaMemcpyAsync(d_volume.data(), h_volume.data(), volume_n * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
        cudaMemcpyAsync(d_sino_input.data(), h_sino.data(), sino_n * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;

    GeometryContext context;
    ResourceContext resources;
    ok = ok && context.initialize(p, geometry);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(fp_task);
    auto bp = makeBackOperator(bp_task);
    Util::CudaEventTimer timer;
    timer.start(stream);
    ok = ok && fp->prepare(context, resources) && bp->prepare(context, resources) &&
        fp->apply(d_volume.data(), p, d_projection.data(), resources) &&
        bp->apply(d_sino_input.data(), p, d_backprojection.data(), true, resources);
    timer.stop(stream);
    const float elapsed_ms = ok ? timer.elapsed_ms() : NAN;
    std::printf("operator pair GPU time: %.3f ms\n", elapsed_ms);
    out_projection.resize(sino_n); out_volume.resize(volume_n);
    ok = ok && cudaMemcpy(out_projection.data(), d_projection.data(), sino_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(out_volume.data(), d_backprojection.data(), volume_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    fp->release(); bp->release();
    if (stream) cudaStreamDestroy(stream);
    return ok;
}

} // namespace

// Complete FP/BP operator matrix.  This is intentionally a numerical
// characterization test: every supported pair must execute and produce finite
// data; the reported inner-product mismatch becomes the regression baseline.
// Only an explicitly documented strict-adjoint pair should later receive a
// separate tight assertion.
int main_operator_matrix_smoke()
{
    const SReconstructionParams p = makePropertyParams();
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const auto x = TestPhantom::makeBasic(p);
    std::vector<float> y(static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv);
    for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<float>((i * 19 + 7) % 41) / 41.f - 0.5f;
    struct Pair { const char* name; ETask fp; ETask bp; };
    const Pair pairs[] = {
        { "joseph/siddon-ray", ETask::FP_Joseph, ETask::BP_Siddon_RayDriven },
        { "joseph/siddon-voxel", ETask::FP_Joseph, ETask::BP_Siddon_VoxDriven },
        { "joseph/siddon-voxel-v2", ETask::FP_Joseph,
            ETask::BP_Siddon_VoxDriven_v2 },
        { "joseph/siddon-voxel-v3", ETask::FP_Joseph,
            ETask::BP_Siddon_VoxDriven_v3 },
        { "joseph/joseph", ETask::FP_Joseph, ETask::BP_Joseph },
        { "joseph/joseph-v2", ETask::FP_Joseph, ETask::BP_Joseph_v2 },
        { "joseph/joseph-v3", ETask::FP_Joseph, ETask::BP_Joseph_v3 },
        { "joseph/fdk", ETask::FP_Joseph, ETask::BP_FDK },
        { "joseph/fdk-matched", ETask::FP_Joseph, ETask::BP_FDK_matched },
        { "siddon/siddon-ray", ETask::FP_Siddon, ETask::BP_Siddon_RayDriven },
        { "siddon/siddon-voxel", ETask::FP_Siddon, ETask::BP_Siddon_VoxDriven },
        { "siddon/siddon-voxel-v2", ETask::FP_Siddon,
            ETask::BP_Siddon_VoxDriven_v2 },
        { "siddon/siddon-voxel-v3", ETask::FP_Siddon,
            ETask::BP_Siddon_VoxDriven_v3 },
        { "siddon/joseph", ETask::FP_Siddon, ETask::BP_Joseph },
        { "siddon/joseph-v2", ETask::FP_Siddon, ETask::BP_Joseph_v2 },
        { "siddon/joseph-v3", ETask::FP_Siddon, ETask::BP_Joseph_v3 },
        { "siddon/fdk", ETask::FP_Siddon, ETask::BP_FDK },
        { "siddon/fdk-matched", ETask::FP_Siddon, ETask::BP_FDK_matched },
    };

    bool ok = true;
    for (const Pair& pair : pairs) {
        std::vector<float> ax, aty;
        const bool executed = projectAndBackproject(p, geometry, pair.fp, pair.bp, x, y, ax, aty);
        const double lhs = executed ? dot(ax, y) : NAN;
        const double rhs = executed ? dot(x, aty) : NAN;
        const double rel = std::fabs(lhs - rhs) /
            std::max({ 1e-8, std::fabs(lhs), std::fabs(rhs) });
        const bool pair_ok = executed && finite(ax) && finite(aty) && std::isfinite(rel);
        std::printf("%-23s <Ax,y>=%.6e <x,By>=%.6e relative=%.6f %s\n",
            pair.name, lhs, rhs, rel, pair_ok ? "OK" : "FAIL");
        ok = ok && pair_ok;
    }
    return ok ? 0 : 1;
}

// Fixed detector plane + rotating off-axis source trajectory.  This validates
// that external vector geometry, rather than circular assumptions, reaches
// both FP and BP and produces a finite non-zero result.
int main_planar_geometry_operator_smoke()
{
    const SReconstructionParams p = makePropertyParams();
    std::vector<SConeProjGeomVec> planar;
    buildPlanarRotatingSourceGeometry(p.scan.angles, p.scan.Nu, p.scan.Nv,
        p.scan.du_mm, p.scan.dv_mm, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
        12.f, planar);
    const auto x = TestPhantom::makeCatphanLike(p);
    std::vector<float> y(static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv, 0.1f);
    std::vector<float> projection, backprojection;
    const bool ok = projectAndBackproject(p, planar, ETask::FP_Joseph, ETask::BP_Joseph_v2,
        x, y, projection, backprojection) && finite(projection) && finite(backprojection) &&
        std::fabs(dot(projection, projection)) > 1e-8 && std::fabs(dot(backprojection, backprojection)) > 1e-8;
    std::printf("planar geometry FP/BP: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

std::vector<float> makePcbPhantom(const SReconstructionParams& p)
{
    std::vector<float> volume(static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz, 0.f);
    constexpr int samples = 4;
    for (int z = 0; z < p.volume.Nz; ++z) {
        for (int y = 0; y < p.volume.Ny; ++y) {
            for (int x = 0; x < p.volume.Nx; ++x) {
                float sum = 0.f;
                for (int sz = 0; sz < samples; ++sz) {
                    for (int sy = 0; sy < samples; ++sy) {
                        for (int sx = 0; sx < samples; ++sx) {
                            const float wx = (x + (sx + 0.5f) / samples -
                                0.5f * p.volume.Nx) * p.volume.voxelX_mm;
                            const float wy = (y + (sy + 0.5f) / samples -
                                0.5f * p.volume.Ny) * p.volume.voxelY_mm;
                            const float wz = (z + (sz + 0.5f) / samples -
                                0.5f * p.volume.Nz) * p.volume.voxelZ_mm;
                            // A compact tungsten PCB carrier with a thin PMMA-like
                            // board and sub-millimetre voids inside selected pads.
                            float value = std::fabs(wx) <= 7.f &&
                                std::fabs(wz) <= 5.f && std::fabs(wy) <= 0.25f ?
                                0.03f : 0.f;
                            for (int iz = -2; iz <= 2; ++iz) {
                                for (int ix = -2; ix <= 2; ++ix) {
                                    const float dx = wx - 2.4f * ix;
                                    const float dz = wz - 2.4f * iz;
                                    const bool in_pad = dx * dx + dz * dz <= 0.60f * 0.60f &&
                                        wy >= 0.25f && wy <= 0.65f;
                                    if (in_pad) value = 0.60f;

                                    // Deliberate voids: two asymmetric pad defects,
                                    // large enough to span multiple 0.1 mm voxels.
                                    const bool defect_a = ix == -1 && iz == 1 &&
                                        (dx - 0.12f) * (dx - 0.12f) +
                                        (dz + 0.08f) * (dz + 0.08f) <= 0.24f * 0.24f;
                                    const bool defect_b = ix == 1 && iz == -1 &&
                                        (dx + 0.10f) * (dx + 0.10f) +
                                        (dz - 0.10f) * (dz - 0.10f) <= 0.18f * 0.18f;
                                    if ((defect_a || defect_b) &&
                                        wy >= 0.25f && wy <= 0.65f)
                                        value = 0.f;
                                }
                            }
                            if (std::fabs(wx) <= 3.5f && std::fabs(wz) <= 0.22f &&
                                wy >= 0.25f && wy <= 0.42f) value = 0.60f;
                            sum += value;
                        }
                    }
                }
                volume[TestPhantom::index(p, x, y, z)] =
                    sum / static_cast<float>(samples * samples * samples);
            }
        }
    }
    return volume;
}

float positiveDisplayMaximum(const std::vector<float>& values)
{
    std::vector<float> positive;
    positive.reserve(values.size());
    for (float value : values)
        if (std::isfinite(value) && value > 0.f) positive.push_back(value);
    if (positive.empty()) return 1.f;
    const size_t index = std::min(positive.size() - 1,
        static_cast<size_t>(0.995 * positive.size()));
    std::nth_element(positive.begin(), positive.begin() + index, positive.end());
    return std::max(positive[index], 1e-8f);
}

int main_source_orbit_tomography_smoke()
{
    SReconstructionParams p = makePropertyParams();
    constexpr int raw_u = 160, raw_v = 160;
    p.scan.Nu = raw_u; p.scan.Nv = raw_v;
    p.scan.NAng = p.scan.totalViews = 12;
    p.volume.Nx = 16; p.volume.Ny = 16; p.volume.Nz = 12;
    std::vector<float> angles(p.scan.totalViews);
    for (int i = 0; i < p.scan.totalViews; ++i)
        angles[i] = 2.f * CUDA_PI * i / p.scan.totalViews;
    std::vector<SConeProjGeomVec> geometry;
    const bool geometry_ok = buildPlanarRotatingSourceGeometry(angles, raw_u, raw_v, p.scan.du_mm,
        p.scan.dv_mm, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
        12.f, geometry);
    SReconstructionParams local = p;
    local.scan.Nu = 64; local.scan.Nv = 64;
    std::vector<SFDKGeoParamPerView> derived_check;
    const bool derived_ok = NonStandardFpBp::buildSourceOrbitDerivedGeometry(
        local.scan.Nu, local.scan.Nv, local.scan.range_rad,
        make_float3(0.f, 0.f, 0.f), geometry, derived_check);
    std::vector<NonStandardFpBp::ProjectionWindow> windows;
    for (int i = 0; i < p.scan.totalViews; ++i)
        windows.push_back({local.scan.Nu, local.scan.Nv, 48 + (i & 1), 48});
    const auto phantom = TestPhantom::makeBasic(p);
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
    auto d_raw = memory.allocateDevice3D<float>(raw_u, raw_v, p.scan.totalViews, 0);
    auto d_recon = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, true);
    cudaStream_t stream = nullptr;
    bool ok = cudaStreamCreate(&stream) == cudaSuccess;
    GeometryContext context;
    ResourceContext resources;
    if (ok) ok = context.initialize(p, geometry);
    resources.attach(stream, 0);
    auto forward = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && cudaMemcpyAsync(d_truth.data(), phantom.data(), volume_n * sizeof(float),
        cudaMemcpyHostToDevice, stream) == cudaSuccess;
    ok = ok && forward->prepare(context, resources) &&
        forward->apply(d_truth.data(), p, d_raw.data(), resources);
    NonStandardFpBp::SourceOrbitTomographyPipeline pipeline;
    const bool prepared_orbit = ok && pipeline.prepare(local, geometry, windows,
        raw_u, raw_v, stream, 0);
    const bool processed_orbit = prepared_orbit &&
        pipeline.process(d_raw.data(), d_recon.data(), true);
    ok = geometry_ok && derived_ok && processed_orbit;
    ok = ok && cudaStreamSynchronize(stream) == cudaSuccess;
    std::vector<float> result(volume_n);
    ok = ok && cudaMemcpy(result.data(), d_recon.data(), volume_n * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess;
    bool finite_nonzero = false;
    for (float value : result)
        finite_nonzero = finite_nonzero ||
            (std::isfinite(value) && std::abs(value) > 1e-7f);
    std::printf("source-orbit tomography pipeline: %s\n", ok && finite_nonzero ? "PASS" : "FAIL");
    pipeline.release();
    forward->release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    return ok && finite_nonzero ? 0 : 1;
}

int main_source_orbit_pcb_reconstruction()
{
    namespace fs = std::filesystem;
    constexpr int raw_u = 512, raw_v = 512, local_u = 192, local_v = 192;
    constexpr int views = 180;
    SReconstructionParams raw = makePropertyParams();
    raw.scan.Nu = raw_u; raw.scan.Nv = raw_v;
    raw.scan.NAng = raw.scan.totalViews = views;
    raw.scan.du_mm = raw.scan.dv_mm = 0.25f;
    raw.scan.sid_mm = 220.f; raw.scan.sdd_mm = 400.f;
    raw.scan.range_rad = 2.f * CUDA_PI;
    NonStandardFpBp::SourceTrajectorySampling sampling{};
    sampling.views = views;
    sampling.start_angle_rad = 0.f;
    sampling.range_rad = 2.f * CUDA_PI;
    sampling.direction = 1.f;
    bool ok = NonStandardFpBp::buildUniformTrajectoryAngles(
        sampling, raw.scan.angles);
    raw.volume.Nx = 192; raw.volume.Ny = 48; raw.volume.Nz = 192;
    raw.volume.voxelX_mm = raw.volume.voxelY_mm = raw.volume.voxelZ_mm = 0.1f;

    NonStandardFpBp::SourceOrbitGeometryConfig config{};
    config.detector_u = raw_u; config.detector_v = raw_v;
    config.detector_pixel_u_mm = config.detector_pixel_v_mm = 0.25f;
    config.sid_mm = raw.scan.sid_mm;
    config.detector_y_mm = raw.scan.sdd_mm - raw.scan.sid_mm;
    config.source_orbit_radius_mm = 15.f;
    std::vector<SConeProjGeomVec> geometry;
    ok = ok && NonStandardFpBp::buildSourceOrbitGeometry(config,
        raw.scan.angles, geometry);

    std::vector<SFDKGeoParamPerView> raw_derived;
    ok = ok && NonStandardFpBp::buildSourceOrbitDerivedGeometry(raw_u, raw_v,
        raw.scan.range_rad, make_float3(0.f, 0.f, 0.f), geometry, raw_derived);
    std::vector<NonStandardFpBp::ProjectionWindow> windows;
    for (const auto& view : raw_derived) {
        const float center_u = 0.5f * (raw_u - 1) + view.offsetU_pix;
        const float center_v = 0.5f * (raw_v - 1) + view.offsetV_pix;
        const auto safe = NonStandardFpBp::makeSafeWindow(raw_u, raw_v,
            local_u, local_v, center_u, center_v);
        if (!safe.valid) { ok = false; break; }
        windows.push_back(safe.window);
    }

    SReconstructionParams local = raw;
    local.scan.Nu = local_u; local.scan.Nv = local_v;
    const auto truth = makePcbPhantom(raw);
    const size_t volume_n = truth.size();
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(raw.volume.Nx,
        raw.volume.Ny, raw.volume.Nz, 0);
    auto d_raw = memory.allocateDevice3D<float>(raw_u, raw_v, views, 0);
    auto d_resampled_inplane = memory.allocateDevice3D<float>(local_u, local_v, views, 0);
    auto d_resampled_perspective = memory.allocateDevice3D<float>(local_u, local_v, views, 0);
    auto d_recon_inplane = memory.allocateDevice3D<float>(raw.volume.Nx,
        raw.volume.Ny, raw.volume.Nz, 0, true);
    auto d_recon_perspective = memory.allocateDevice3D<float>(raw.volume.Nx,
        raw.volume.Ny, raw.volume.Nz, 0, true);
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    GeometryContext context;
    ResourceContext resources;
    ok = ok && context.initialize(raw, geometry);
    resources.attach(stream, 0);
    auto forward = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_n * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    Util::CudaEventTimer fp_timer;
    fp_timer.start(stream);
    ok = ok && forward->prepare(context, resources) &&
        forward->apply(d_truth.data(), raw, d_raw.data(), resources);
    fp_timer.stop(stream);
    const float fp_ms = ok ? fp_timer.elapsed_ms() : NAN;

    NonStandardFpBp::SourceOrbitTomographyPipeline inplane_pipeline;
    Util::CudaEventTimer inplane_timer;
    inplane_timer.start(stream);
    ok = ok && inplane_pipeline.prepare(local, geometry, windows,
        raw_u, raw_v, stream, 0) &&
        inplane_pipeline.resample(d_raw.data(), d_resampled_inplane.data()) &&
        inplane_pipeline.process(d_raw.data(), d_recon_inplane.data(), true);
    inplane_timer.stop(stream);
    const float inplane_ms = ok ? inplane_timer.elapsed_ms() : NAN;

    NonStandardFpBp::SourceOrbitTomographyPipeline perspective_pipeline;
    Util::CudaEventTimer perspective_timer;
    perspective_timer.start(stream);
    ok = ok && perspective_pipeline.prepare(local, geometry, windows,
        raw_u, raw_v, NonStandardFpBp::SourceOrbitResampleMode::PerspectiveOrthogonal,
        stream, 0) &&
        perspective_pipeline.resample(d_raw.data(), d_resampled_perspective.data()) &&
        perspective_pipeline.process(d_raw.data(), d_recon_perspective.data(), true);
    perspective_timer.stop(stream);
    const float perspective_ms = ok ? perspective_timer.elapsed_ms() : NAN;
    std::vector<float> recon_inplane(volume_n), recon_perspective(volume_n);
    std::vector<float> raw_projection(static_cast<size_t>(raw_u) * raw_v * views);
    std::vector<float> resampled_inplane(
        static_cast<size_t>(local_u) * local_v * views);
    std::vector<float> resampled_perspective(
        static_cast<size_t>(local_u) * local_v * views);
    ok = ok && cudaMemcpy(recon_inplane.data(), d_recon_inplane.data(),
        volume_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(recon_perspective.data(), d_recon_perspective.data(),
        volume_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(raw_projection.data(), d_raw.data(),
            raw_projection.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(resampled_inplane.data(), d_resampled_inplane.data(),
            resampled_inplane.size() * sizeof(float),
            cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(resampled_perspective.data(), d_resampled_perspective.data(),
            resampled_perspective.size() * sizeof(float),
            cudaMemcpyDeviceToHost) == cudaSuccess &&
        finite(recon_inplane) && finite(recon_perspective) &&
        finite(resampled_inplane) && finite(resampled_perspective);
    bool inplane_nonzero = false, perspective_nonzero = false;
    for (float value : recon_inplane)
        inplane_nonzero = inplane_nonzero || std::abs(value) > 1e-7f;
    for (float value : recon_perspective)
        perspective_nonzero = perspective_nonzero || std::abs(value) > 1e-7f;
    ok = ok && inplane_nonzero && perspective_nonzero;
    const auto relative_rms = [](const std::vector<float>& reference,
        const std::vector<float>& comparison) {
        double difference2 = 0.0, reference2 = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double difference = comparison[i] - reference[i];
            difference2 += difference * difference;
            reference2 += static_cast<double>(reference[i]) * reference[i];
        }
        return std::sqrt(difference2 / std::max(reference2, 1e-30));
    };
    const double projection_relative_rms = relative_rms(resampled_inplane,
        resampled_perspective);
    const double reconstruction_relative_rms = relative_rms(recon_inplane,
        recon_perspective);
    const auto maximum_center_ray_tilt_degrees = [&](const auto& mapped_geometry) {
        float maximum = 0.f;
        for (const auto& view : mapped_geometry) {
            SProjectionFrame frame{};
            if (!deriveProjectionFrame(view, local_u, local_v, frame))
                return NAN;
            const float dx = raw.volume.centerX_mm - view.src.x;
            const float dy = raw.volume.centerY_mm - view.src.y;
            const float dz = raw.volume.centerZ_mm - view.src.z;
            const float inverse_length = 1.f / std::sqrt(dx * dx + dy * dy + dz * dz);
            const float cosine = std::clamp((dx * frame.detectorNormal.x +
                dy * frame.detectorNormal.y + dz * frame.detectorNormal.z) *
                inverse_length, -1.f, 1.f);
            maximum = std::max(maximum, std::acos(cosine) * 180.f / CUDA_PI);
        }
        return maximum;
    };
    const float inplane_tilt_degrees = maximum_center_ray_tilt_degrees(
        inplane_pipeline.mappedGeometry());
    const float perspective_tilt_degrees = maximum_center_ray_tilt_degrees(
        perspective_pipeline.mappedGeometry());
    ok = ok && std::isfinite(inplane_tilt_degrees) &&
        std::isfinite(perspective_tilt_degrees) && perspective_tilt_degrees < 0.05f;

    const fs::path output_dir = fs::path(YKCBCT_TEST_SOURCE_DIR) / "output" /
        "source_orbit_tomography";
    fs::create_directories(output_dir);
    const auto write_raw = [&](const fs::path& path,
        const std::vector<float>& values) {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        return output.good();
    };
    ok = write_raw(output_dir / "pcb_truth_f32.raw", truth) && ok;
    ok = write_raw(output_dir / "pcb_reconstruction_f32.raw", recon_inplane) && ok;
    ok = write_raw(output_dir / "pcb_reconstruction_perspective_f32.raw",
        recon_perspective) && ok;
    ok = write_raw(output_dir / "pcb_raw_projection_f32.raw", raw_projection) && ok;
    ok = write_raw(output_dir / "pcb_resampled_projection_f32.raw",
        resampled_inplane) && ok;
    ok = write_raw(output_dir / "pcb_resampled_perspective_f32.raw",
        resampled_perspective) && ok;
    const float projection_max = std::max(positiveDisplayMaximum(raw_projection),
        std::max(positiveDisplayMaximum(resampled_inplane),
            positiveDisplayMaximum(resampled_perspective)));
    const int comparison_views[] = {0, views / 4, views / 2, 3 * views / 4};
    std::vector<TestImage::GrayPanel> raw_projection_panels;
    std::vector<TestImage::GrayPanel> inplane_projection_panels;
    std::vector<TestImage::GrayPanel> perspective_projection_panels;
    for (int view : comparison_views) {
        raw_projection_panels.push_back({&raw_projection, raw_u, raw_v, views,
            view, 1.f, 0.f, projection_max});
        inplane_projection_panels.push_back({&resampled_inplane,
            local_u, local_v, views, view, 1.f, 0.f, projection_max});
        perspective_projection_panels.push_back({&resampled_perspective,
            local_u, local_v, views, view, 1.f, 0.f, projection_max});
    }
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "pcb_raw_projection_views_0_90_180_270.bmp",
        raw_projection_panels, 2, 8, 1) && ok;
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "pcb_resampled_projection_views_0_90_180_270.bmp",
        inplane_projection_panels, 2, 8, 2) && ok;
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "pcb_resampled_perspective_views_0_90_180_270.bmp",
        perspective_projection_panels, 2, 8, 2) && ok;
    const float recon_max = std::max(positiveDisplayMaximum(recon_inplane),
        positiveDisplayMaximum(recon_perspective));
    const std::vector<TestImage::GrayPanel> face_panels = {
        {&truth,raw.volume.Nx,raw.volume.Ny,raw.volume.Nz,raw.volume.Nz / 2,
            1.f,0.f,0.60f},
        {&recon_inplane,raw.volume.Nx,raw.volume.Ny,raw.volume.Nz,raw.volume.Nz / 2,
            1.f,0.f,recon_max},
        {&recon_perspective,raw.volume.Nx,raw.volume.Ny,raw.volume.Nz,raw.volume.Nz / 2,
            1.f,0.f,recon_max}
    };
    ok = TestImage::writeGrayMontageBmp(output_dir / "pcb_xy_truth_recon.bmp",
        face_panels, 3, 8, 3) && ok;
    std::vector<float> truth_xz(static_cast<size_t>(raw.volume.Nx) * raw.volume.Nz);
    std::vector<float> inplane_xz(truth_xz.size()), perspective_xz(truth_xz.size());
    // Inspect the pad layer (board mid-plane is at y=0; pads start at 0.25 mm).
    const int center_y = raw.volume.Ny / 2 + 4;
    for (int z = 0; z < raw.volume.Nz; ++z) {
        for (int x = 0; x < raw.volume.Nx; ++x) {
            const size_t source = (static_cast<size_t>(z) * raw.volume.Ny +
                center_y) * raw.volume.Nx + x;
            truth_xz[static_cast<size_t>(z) * raw.volume.Nx + x] = truth[source];
            inplane_xz[static_cast<size_t>(z) * raw.volume.Nx + x] =
                recon_inplane[source];
            perspective_xz[static_cast<size_t>(z) * raw.volume.Nx + x] =
                recon_perspective[source];
        }
    }
    const std::vector<TestImage::GrayPanel> edge_panels = {
        {&truth_xz,raw.volume.Nx,raw.volume.Nz,1,0,1.f,0.f,0.60f},
        {&inplane_xz,raw.volume.Nx,raw.volume.Nz,1,0,1.f,0.f,recon_max},
        {&perspective_xz,raw.volume.Nx,raw.volume.Nz,1,0,1.f,0.f,recon_max}
    };
    ok = TestImage::writeGrayMontageBmp(output_dir / "pcb_xz_truth_recon.bmp",
        edge_panels, 3, 8, 3) && ok;
    std::ofstream metadata(output_dir / "metadata.txt");
    metadata << "volume=192x48x192\nvoxel_mm=0.1\nraw_detector=512x512\n"
        << "raw_pixel_mm=0.25\nlocal_window=192x192\nviews=180\n"
        << "sid_mm=220\nsdd_mm=400\nsource_orbit_radius_mm=15\n"
        << "pcb_size_mm=14x10\npad_diameter_mm=1.2\n"
        << "pad_voids_mm=0.48,0.36\n"
        << "projection_layout=[view][v][u], u_fastest\n"
        << "inplane_max_center_ray_tilt_deg=" << inplane_tilt_degrees << '\n'
        << "perspective_max_center_ray_tilt_deg=" << perspective_tilt_degrees << '\n'
        << "projection_relative_rms=" << projection_relative_rms << '\n'
        << "reconstruction_relative_rms=" << reconstruction_relative_rms << '\n'
        << "forward_ms=" << fp_ms << "\ninplane_reconstruction_ms=" <<
        inplane_ms << "\nperspective_reconstruction_ms=" << perspective_ms << '\n';
    ok = metadata.good() && ok;
    std::printf("source-orbit PCB: %s FP=%.3f ms inplane=%.3f ms "
        "perspective=%.3f ms tilt=%.4f/%.4f deg projection-rms=%.6f "
        "recon-rms=%.6f output=%s\n", ok ? "PASS" : "FAIL", fp_ms,
        inplane_ms, perspective_ms, inplane_tilt_degrees,
        perspective_tilt_degrees, projection_relative_rms,
        reconstruction_relative_rms, output_dir.string().c_str());
    inplane_pipeline.release();
    perspective_pipeline.release();
    forward->release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
