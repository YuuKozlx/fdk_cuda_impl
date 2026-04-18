#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "test_common.hpp"
#include "FP/kernels/YkFPSiddonLaunch.cuh"
#include "FP/kernels/YkFPLaunch.cuh"
#include "FP/kernels/YkFPHelpers.cuh"
#include "FP/kernels/YkFPCVPLaunch.cuh"
#include "FP/YkFPRunner.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkGlobals.h"
#include "global/YkMem3d.hpp"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "util/YkCudaTimer.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"

using namespace YK;
using namespace YK::Fp;

// ================================================================
// SiddonTest
// ================================================================

// ----------------------------------------------------------------
// 均匀体单视角：中心像素路径长度验证
// ----------------------------------------------------------------
TEST(SiddonTest, UniformSingleView_CenterPixel)
{
    constexpr int   Nx = 64, Ny = 64, Nz = 64;
    constexpr float vox = 1.f;
    constexpr int   Nu = 64, Nv = 64;
    constexpr float du = 1.f, dv = 1.f;
    constexpr float SID = 500.f, SDD = 1000.f;

    cudaStream_t stream = 0;

    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);
    std::vector<float> h_vol(Nx * Ny * Nz, 1.f);

    float* d_vol = nullptr;
    ASSERT_EQ(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_vol, h_vol.data(), h_vol.size() * sizeof(float), cudaMemcpyHostToDevice);

    SConeProjGeomVec view;
    view.src  = make_float4(0.f, -SID, 0.f, 0.f);
    view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
    view.detU = make_float4(du, 0.f, 0.f, 0.f);
    view.detV = make_float4(0.f, 0.f, dv, 0.f);
    view.detS = make_float4(-Nu * 0.5f * du, SDD - SID, -Nv * 0.5f * dv, 0.f);
    view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

    SConeProjGeomVec* d_views = nullptr;
    ASSERT_EQ(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)), cudaSuccess);
    cudaMemcpy(d_views, &view, sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice);

    const size_t sino_elems = (size_t)Nv * Nu;
    float* d_sino = nullptr;
    ASSERT_EQ(cudaMalloc(&d_sino, sino_elems * sizeof(float)), cudaSuccess);
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));

    fp_siddon_launch(d_vol, d_sino, d_views, g, Nu, Nv, 1, false, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> h_sino(sino_elems);
    cudaMemcpy(h_sino.data(), d_sino, sino_elems * sizeof(float), cudaMemcpyDeviceToHost);

    const float center = h_sino[(Nv / 2) * Nu + Nu / 2];
    const float expect = (float)Ny * vox;
    EXPECT_NEAR(center, expect, expect * 0.01f)
        << "Center pixel path length incorrect";

    cudaFree(d_vol); cudaFree(d_views); cudaFree(d_sino);
}

// ----------------------------------------------------------------
// 单点体素：投影峰值在探测器中心
// ----------------------------------------------------------------
TEST(SiddonTest, SingleVoxel_PeakAtCenter)
{
    constexpr int   Nx = 64, Ny = 64, Nz = 64;
    constexpr float vox = 1.f;
    constexpr int   Nu = 64, Nv = 64;
    constexpr float SID = 500.f, SDD = 1000.f;

    cudaStream_t stream = 0;
    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox);

    std::vector<float> h_vol(Nx * Ny * Nz, 0.f);
    h_vol[(Nz/2)*Ny*Nx + (Ny/2)*Nx + (Nx/2)] = 1.f;

    float* d_vol = nullptr;
    ASSERT_EQ(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_vol, h_vol.data(), h_vol.size() * sizeof(float), cudaMemcpyHostToDevice);

    SConeProjGeomVec view;
    view.src  = make_float4(0.f, -SID, 0.f, 0.f);
    view.srcCR = make_float4(0.f, 1.f, 0.f, 0.f);
    view.detU = make_float4(1.f, 0.f, 0.f, 0.f);
    view.detV = make_float4(0.f, 0.f, 1.f, 0.f);
    view.detS = make_float4(-Nu * 0.5f, SDD - SID, -Nv * 0.5f, 0.f);
    view.angle = make_float4(0.f, 0.f, 0.f, 0.f);

    SConeProjGeomVec* d_views = nullptr;
    ASSERT_EQ(cudaMalloc(&d_views, sizeof(SConeProjGeomVec)), cudaSuccess);
    cudaMemcpy(d_views, &view, sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice);

    const size_t sino_elems = (size_t)Nv * Nu;
    float* d_sino = nullptr;
    ASSERT_EQ(cudaMalloc(&d_sino, sino_elems * sizeof(float)), cudaSuccess);
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));

    fp_siddon_launch(d_vol, d_sino, d_views, g, Nu, Nv, 1, false, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> h_sino(sino_elems);
    cudaMemcpy(h_sino.data(), d_sino, sino_elems * sizeof(float), cudaMemcpyDeviceToHost);

    int peak_u = -1, peak_v = -1;
    float peak_val = 0.f;
    for (int iv = 0; iv < Nv; ++iv)
        for (int iu = 0; iu < Nu; ++iu) {
            float val = h_sino[iv * Nu + iu];
            if (val > peak_val) { peak_val = val; peak_u = iu; peak_v = iv; }
        }

    EXPECT_EQ(peak_u, Nu / 2) << "Peak not at center U";
    EXPECT_EQ(peak_v, Nv / 2) << "Peak not at center V";
    EXPECT_GT(peak_val, 0.f)  << "No response from single voxel";

    cudaFree(d_vol); cudaFree(d_views); cudaFree(d_sino);
}

// ----------------------------------------------------------------
// 真实体积正投影（目视类，验证能跑通且非零）
// ----------------------------------------------------------------
TEST(SiddonTest, RealVolume_NonZeroOutput)
{
    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.25f, vox_z = 0.25f;
    constexpr int   Na = 360, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    if (!read_raw_float("fdk_vec_vol_offline.raw", h_vol)) {
        GTEST_SKIP() << "fdk_vec_vol_offline.raw not found, skipping";
    }

    cudaStream_t stream = 0;
    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);

    float* d_vol = nullptr;
    ASSERT_EQ(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_vol, h_vol.data(), h_vol.size() * sizeof(float), cudaMemcpyHostToDevice);
    h_vol.clear();

    std::vector<float> angles(Na);
    for (int i = 0; i < Na; ++i)
        angles[i] = 2.f * CUDA_PI * i / Na;

    std::vector<SConeProjGeomVec>    h_views(Na);
    std::vector<SFDKGeoParamPerView> h_gv(Na);
    build_circular_vec_geometry_from_theta(
        h_views, angles, Na, Nu, Nv, du, dv,
        SID, SDD - SID, f3(0.f, 0.f, 0.f), f3(0.f, 0.f, 0.f));

    SConeProjGeomVec* d_views = nullptr;
    ASSERT_EQ(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)), cudaSuccess);
    cudaMemcpy(d_views, h_views.data(), Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice);

    const size_t sino_elems = (size_t)Na * Nv * Nu;
    float* d_sino = nullptr;
    ASSERT_EQ(cudaMalloc(&d_sino, sino_elems * sizeof(float)), cudaSuccess);
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));

    fp_siddon_launch(d_vol, d_sino, d_views, g, Nu, Nv, Na, false, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> h_sino(sino_elems);
    cudaMemcpy(h_sino.data(), d_sino, sino_elems * sizeof(float), cudaMemcpyDeviceToHost);

    float maxv = *std::max_element(h_sino.begin(), h_sino.end());
    EXPECT_GT(maxv, 0.f) << "Siddon FP output is all zeros";

    write_raw_float("siddon_test_real.raw", h_sino.data(), sino_elems);

    cudaFree(d_vol); cudaFree(d_views); cudaFree(d_sino);
}

// ================================================================
// JosephTest
// ================================================================

TEST(JosephTest, RealVolume_NonZeroOutput)
{
    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.25f, vox_z = 0.25f;
    constexpr int   Na = 360, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    if (!read_raw_float("fdk_vec_vol_offline.raw", h_vol)) {
        GTEST_SKIP() << "fdk_vec_vol_offline.raw not found, skipping";
    }

    cudaStream_t stream = 0;
    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);

    std::vector<float> angles(Na);
    for (int i = 0; i < Na; ++i)
        angles[i] = 2.f * CUDA_PI * i / Na;

    std::vector<SConeProjGeomVec>    h_views(Na);
    std::vector<SFDKGeoParamPerView> h_gv(Na);
    build_circular_vec_geometry_from_theta(
        h_views, angles, Na, Nu, Nv, du, dv,
        SID, SDD - SID, f3(0.f, 0.f, 0.f), f3(0.f, 0.f, 0.f));

    h_views = Fp::normalizeToVoxelBatch(h_views, g);

    SConeProjGeomVec* d_views = nullptr;
    ASSERT_EQ(cudaMalloc(&d_views, Na * sizeof(SConeProjGeomVec)), cudaSuccess);
    cudaMemcpy(d_views, h_views.data(), Na * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice);

    auto volTex = Mem::TextureController::createTex3DFromHost(h_vol.data(), g);
    h_vol.clear();

    const size_t sino_elems = (size_t)Na * Nv * Nu;
    float* d_sino = nullptr;
    ASSERT_EQ(cudaMalloc(&d_sino, sino_elems * sizeof(float)), cudaSuccess);
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));

    fp_joseph_launch(volTex.tex, h_views, d_views, d_sino, g,
                     Na, Nu, Nv, false, stream, FpStepSuperSample::x1);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> h_sino(sino_elems);
    cudaMemcpy(h_sino.data(), d_sino, sino_elems * sizeof(float), cudaMemcpyDeviceToHost);

    float maxv = *std::max_element(h_sino.begin(), h_sino.end());
    EXPECT_GT(maxv, 0.f) << "Joseph FP output is all zeros";

    write_raw_float("rawtest4_real_sino.raw", h_sino.data(), sino_elems);

    cudaFree(d_views); cudaFree(d_sino);
}

// ================================================================
// CVPTest
// ================================================================

TEST(CVPTest, RealVolume_NonZeroOutput)
{
    constexpr int   Nx = 512, Ny = 512, Nz = 400;
    constexpr float vox_xy = 0.25f, vox_z = 0.25f;
    constexpr int   Na = 360, Nu = 1024, Nv = 1024;
    constexpr float du = 0.25f, dv = 0.25f;
    constexpr float SID = 500.f, SDD = 1000.f;

    std::vector<float> h_vol((size_t)Nx * Ny * Nz);
    if (!read_raw_float("fdk_vec_vol_offline.raw", h_vol)) {
        GTEST_SKIP() << "fdk_vec_vol_offline.raw not found, skipping";
    }

    cudaStream_t stream = 0;
    SVolGeom g = SVolGeom::make_centered(Nx, Ny, Nz, vox_xy, vox_z);

    float* d_vol = nullptr;
    ASSERT_EQ(cudaMalloc(&d_vol, h_vol.size() * sizeof(float)), cudaSuccess);
    cudaMemcpy(d_vol, h_vol.data(), h_vol.size() * sizeof(float), cudaMemcpyHostToDevice);
    h_vol.clear();

    std::vector<float> angles(Na);
    for (int i = 0; i < Na; ++i)
        angles[i] = 2.f * CUDA_PI * i / Na;

    std::vector<SConeProjGeomVec> h_views(Na);
    std::vector<SFDKGeoParamPerView> h_gv(Na);
    build_circular_vec_geometry_from_theta(
        h_views, angles, Na, Nu, Nv, du, dv,
        SID, SDD - SID, f3(0.f, 0.f, 0.f), f3(0.f, 0.f, 0.f));

    const size_t sino_elems = (size_t)Na * Nv * Nu;
    float* d_sino = nullptr;
    ASSERT_EQ(cudaMalloc(&d_sino, sino_elems * sizeof(float)), cudaSuccess);
    cudaMemset(d_sino, 0, sino_elems * sizeof(float));

    fp_cvp_launch(d_vol, d_sino, h_views.data(), g, Na, Nu, Nv, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> h_sino(sino_elems);
    cudaMemcpy(h_sino.data(), d_sino, sino_elems * sizeof(float), cudaMemcpyDeviceToHost);

    float maxv = *std::max_element(h_sino.begin(), h_sino.end());
    EXPECT_GT(maxv, 0.f) << "CVP FP output is all zeros";

    write_raw_float("cvp_test_real_sino.raw", h_sino.data(), sino_elems);

    cudaFree(d_vol); cudaFree(d_sino);
}

// ----------------------------------------------------------------
// CVP vs Joseph 相对误差对比
// ----------------------------------------------------------------
TEST(CVPTest, CompareWithJoseph_RelRms)
{
    const size_t Na = 360, Nu = 64, Nv = 64;
    const size_t sino_elems = Na * Nv * Nu;

    std::vector<float> h_joseph, h_cvp;
    if (!read_raw_float("rawtest2_multi_view.raw",  h_joseph) ||
        !read_raw_float("cvptest2_multi_view.raw",  h_cvp)) {
        GTEST_SKIP() << "Reference files not found, skipping";
    }
    if (h_joseph.size() != sino_elems || h_cvp.size() != sino_elems) {
        GTEST_SKIP() << "File size mismatch, skipping";
    }

    double diff2 = 0.0, ref2 = 0.0;
    for (size_t k = 0; k < sino_elems; ++k) {
        double d = h_cvp[k] - h_joseph[k];
        diff2 += d * d;
        ref2  += (double)h_joseph[k] * h_joseph[k];
    }
    const double rel_rms = std::sqrt(diff2 / (ref2 + 1e-30));
    EXPECT_LT(rel_rms, 0.05) << "CVP vs Joseph rel_rms too large: " << rel_rms;
}
