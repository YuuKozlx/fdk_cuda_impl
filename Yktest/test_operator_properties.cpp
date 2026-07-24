#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "util/YkCudaTimer.hpp"

namespace {
using namespace YK;

SCBCTParams makePropertyParams()
{
    SCBCTParams p{};
    p.iPU = 32; p.iPV = 24; p.iPAng = 12; p.iPAngTotal = 12;
    p.iVX = 16; p.iVY = 16; p.iVZ = 12;
    p.du_mm = p.dv_mm = 1.f;
    p.vox_x_mm = p.vox_y_mm = p.vox_z_mm = 1.f;
    p.SID = 80.f; p.SDD = 160.f; p.scan_range_rad = 2.f * CUDA_PI;
    p.angle_list.resize(p.iPAng);
    for (int i = 0; i < p.iPAng; ++i) p.angle_list[i] = 2.f * CUDA_PI * i / p.iPAng;
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

bool projectAndBackproject(const SCBCTParams& p, const std::vector<SConeProjGeomVec>& geometry,
    ETask fp_task, ETask bp_task, const std::vector<float>& h_volume,
    const std::vector<float>& h_sino, std::vector<float>& out_projection,
    std::vector<float>& out_volume)
{
    const size_t volume_n = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t sino_n = static_cast<size_t>(p.iPAng) * p.iPU * p.iPV;
    cudaStream_t stream = nullptr;
    Mem::MemoryController memory;
    // All test allocations use the library allocator: ownership and the
    // canonical [z][y][x] / [view][v][u] layouts stay identical to production.
    auto d_volume = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
    auto d_sino_input = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
    auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
    auto d_backprojection = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
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
    const SCBCTParams p = makePropertyParams();
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const auto x = TestPhantom::makeBasic(p);
    std::vector<float> y(static_cast<size_t>(p.iPAng) * p.iPU * p.iPV);
    for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<float>((i * 19 + 7) % 41) / 41.f - 0.5f;
    struct Pair { const char* name; ETask fp; ETask bp; };
    const Pair pairs[] = {
        { "joseph/siddon-ray", ETask::FP_Joseph, ETask::BP_Siddon_RayDriven },
        { "joseph/siddon-voxel", ETask::FP_Joseph, ETask::BP_Siddon_VoxDriven },
        { "joseph/joseph", ETask::FP_Joseph, ETask::BP_Joseph },
        { "joseph/joseph-v2", ETask::FP_Joseph, ETask::BP_Joseph_v2 },
        { "joseph/joseph-v3", ETask::FP_Joseph, ETask::BP_Joseph_v3 },
        { "joseph/fdk", ETask::FP_Joseph, ETask::BP_FDK },
        { "joseph/fdk-matched", ETask::FP_Joseph, ETask::BP_FDK_matched },
        { "siddon/siddon-ray", ETask::FP_Siddon, ETask::BP_Siddon_RayDriven },
        { "siddon/siddon-voxel", ETask::FP_Siddon, ETask::BP_Siddon_VoxDriven },
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
    const SCBCTParams p = makePropertyParams();
    std::vector<SConeProjGeomVec> planar;
    build_planar_ct_vec_geometry(planar, p.angle_list, p.iPAng, p.iPU, p.iPV,
        p.du_mm, p.dv_mm, p.SID, p.SDD - p.SID, 12.f);
    const auto x = TestPhantom::makeCatphanLike(p);
    std::vector<float> y(static_cast<size_t>(p.iPAng) * p.iPU * p.iPV, 0.1f);
    std::vector<float> projection, backprojection;
    const bool ok = projectAndBackproject(p, planar, ETask::FP_Joseph, ETask::BP_Joseph_v2,
        x, y, projection, backprojection) && finite(projection) && finite(backprojection) &&
        std::fabs(dot(projection, projection)) > 1e-8 && std::fabs(dot(backprojection, backprojection)) > 1e-8;
    std::printf("planar geometry FP/BP: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
