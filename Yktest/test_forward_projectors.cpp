#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "FlatFpBp/FP/kernels/YkFPSiddonLaunch.cuh"
#include "global/YkGlobals.h"

namespace {

using namespace YK;
using namespace YK::Fp;

SConeProjGeomVec makeSingleView(int nu, int nv, float du, float dv)
{
    constexpr float sid = 500.f;
    constexpr float sdd = 1000.f;
    SConeProjGeomVec view{};
    view.src = make_float4(0.f, -sid, 0.f, 0.f);
    view.detU = make_float4(du, 0.f, 0.f, 0.f);
    view.detV = make_float4(0.f, 0.f, dv, 0.f);
    view.detS = make_float4(-nu * 0.5f * du, sdd - sid, -nv * 0.5f * dv, 0.f);
    return view;
}

bool runSiddonSingleView(
    const std::vector<float>& volume,
    const SVolGeom& geometry,
    int nu,
    int nv,
    std::vector<float>& projection)
{
    float* d_volume = nullptr;
    float* d_projection = nullptr;
    SConeProjGeomVec* d_view = nullptr;
    bool ok = cudaMalloc(&d_volume, volume.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_projection, projection.size() * sizeof(float)) == cudaSuccess &&
        cudaMalloc(&d_view, sizeof(SConeProjGeomVec)) == cudaSuccess;

    const SConeProjGeomVec view = makeSingleView(nu, nv, 1.f, 1.f);
    if (ok) {
        ok = cudaMemcpy(d_volume, volume.data(), volume.size() * sizeof(float),
                 cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_view, &view, sizeof(view), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemset(d_projection, 0, projection.size() * sizeof(float)) == cudaSuccess;
    }
    if (ok) {
        fp_siddon_launch(d_volume, d_projection, d_view, geometry, nu, nv, 1, false, 0);
        ok = cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(projection.data(), d_projection,
                projection.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    if (d_view) cudaFree(d_view);
    if (d_projection) cudaFree(d_projection);
    if (d_volume) cudaFree(d_volume);
    return ok;
}

} // namespace

int main_fp_siddon_uniform_center_length()
{
    constexpr int nx = 64, ny = 64, nz = 64;
    constexpr int nu = 64, nv = 64;
    constexpr float voxel = 1.f;
    const SVolGeom geometry = SVolGeom::make_centered(nx, ny, nz, voxel);
    std::vector<float> volume(static_cast<size_t>(nx) * ny * nz, 1.f);
    std::vector<float> projection(static_cast<size_t>(nu) * nv);

    const bool launched = runSiddonSingleView(volume, geometry, nu, nv, projection);
    const float center = launched ? projection[(nv / 2) * nu + nu / 2] : NAN;
    const float expected = ny * voxel;
    const float tolerance = expected * 0.01f;
    const bool ok = launched && std::isfinite(center) &&
        std::fabs(center - expected) <= tolerance;
    std::printf("Siddon uniform center: value=%.6f expected=%.6f tolerance=%.6f, %s\n",
        center, expected, tolerance, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main_fp_siddon_single_voxel_peak()
{
    constexpr int nx = 64, ny = 64, nz = 64;
    constexpr int nu = 64, nv = 64;
    constexpr float voxel = 1.f;
    constexpr float magnification = 2.f;
    const SVolGeom geometry = SVolGeom::make_centered(nx, ny, nz, voxel);
    std::vector<float> volume(static_cast<size_t>(nx) * ny * nz, 0.f);
    volume[(static_cast<size_t>(nz / 2) * ny + ny / 2) * nx + nx / 2] = 1.f;
    std::vector<float> projection(static_cast<size_t>(nu) * nv);

    const bool launched = runSiddonSingleView(volume, geometry, nu, nv, projection);
    const auto peak = launched
        ? std::max_element(projection.begin(), projection.end())
        : projection.end();
    const size_t peak_index = peak == projection.end() ? 0 :
        static_cast<size_t>(std::distance(projection.begin(), peak));
    const int peak_u = static_cast<int>(peak_index % nu);
    const int peak_v = static_cast<int>(peak_index / nu);
    const float peak_value = peak == projection.end() ? NAN : *peak;
    // With an even-sized volume, voxel index N/2 is centered at +0.5 voxel.
    // At SDD/SID=2 this maps to a one-pixel detector shift.
    const int expected_u = nu / 2 + static_cast<int>(0.5f * voxel * magnification);
    const int expected_v = nv / 2 + static_cast<int>(0.5f * voxel * magnification);
    const bool ok = launched && std::isfinite(peak_value) && peak_value > 0.f &&
        peak_u == expected_u && peak_v == expected_v;
    std::printf("Siddon single voxel: peak=(%d,%d) value=%.6f expected=(%d,%d), %s\n",
        peak_u, peak_v, peak_value, expected_u, expected_v, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
