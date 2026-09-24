#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "BP/YkBPGpuContext_Siddon.hpp"
#include "BP/kernels/YkBPJosephLaunch.cuh"
#include "BP/kernels/YkBPSiddonLaunch.cuh"
#include "BP/kernels/YkBpFdkLaunch.cuh"
#include "YkTestGeometry.hpp"
#include "CylFpBp/bp/YkCylBackProjection.hpp"
#include "CylFpBp/bp/YkCylBackOperator.hpp"
#include "CylFpBp/fp/YkCylForwardProjection.hpp"
#include "CylFpBp/iter/YkCylAlgebraicReconstructor.hpp"
#include "CylFpBp/fp/YkCylForwardOperator.hpp"
#include "CylFpBp/bp/YkCylFdkBackprojector.hpp"
#include "CylFpBp/analytic/YkCylFdkPipeline.hpp"
#include "CylFpBp/iter/YkCylPwlsReconstructor.hpp"
#include "CylFpBp/analytic/YkCylAnalyticReconstruction.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"
#include "FDK/YkFdkPipeline.hpp"
#include "FP/YkFPGpuContext.hpp"
#include "FP/kernels/YkFPLaunch.cuh"

#include "Heli/analytic/wfbp/YkWfbpPipeline.hpp"
#include "Heli/iter/YkHelicalCylIterativeReconstructor.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "cuda/YkTextureTestKernels.cuh"
#include "global/YkMem3d.hpp"
#include "global/YkCudaTextureController.hpp"

namespace {

using namespace YK;

Mem::Tex3DHandle makeTexture(const float* source, int nx, int ny, int nz,
    cudaTextureFilterMode filter, cudaStream_t stream)
{
    auto texture = Mem::TextureController::createEmptyTex3D(nx, ny, nz,
        filter, cudaAddressModeBorder);
    Mem::TextureController::updateTex3DFromDeviceAsync(texture, source, nx, ny,
        nz, stream);
    return texture;
}

SReconstructionParams toCbct(const SHeliCTParam& h)
{
    SReconstructionParams p{};
    p.scan.angles = h.angle_list;
    p.scan.Nu = h.iPU; p.scan.Nv = h.iPV;
    p.scan.NAng = static_cast<int>(h.angle_list.size());
    p.scan.totalViews = p.scan.NAng;
    p.scan.du_mm = h.du_mm; p.scan.dv_mm = h.dv_mm;
    p.scan.offsetU_mm = h.offsetU_mm; p.scan.offsetV_mm = h.offsetV_mm;
    p.scan.sid_mm = h.SID; p.scan.sdd_mm = h.SDD;
    p.volume.Nx = h.iVX; p.volume.Ny = h.iVY; p.volume.Nz = h.iVZ;
    p.volume.voxelX_mm = h.vox_x_mm;
    p.volume.voxelY_mm = h.vox_y_mm;
    p.volume.voxelZ_mm = h.vox_z_mm;
    p.scan.range_rad = h.angle_list.back() - h.angle_list.front();
    return p;
}

SVolGeom volumeGeometry(const SHeliCTParam& h)
{
    SVolGeom g = SVolGeom::make_centered(h.iVX, h.iVY, h.iVZ,
        h.vox_x_mm, h.vox_y_mm, h.vox_z_mm);
    g.center = make_float3(h.vol_offset_x_mm, h.vol_offset_y_mm,
        h.vol_offset_z_mm);
    return g;
}

struct Metrics {
    double correlation = 0.0;
    double absolute_nrmse = 0.0;
    double fitted_nrmse = 0.0;
    double mae = 0.0;
    double bias = 0.0;
    float fitted_scale = 0.f;
};

Metrics compare(const std::vector<float>& truth, const std::vector<float>& recon)
{
    double xx = 0.0, yy = 0.0, xy = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        xx += static_cast<double>(truth[i]) * truth[i];
        yy += static_cast<double>(recon[i]) * recon[i];
        xy += static_cast<double>(truth[i]) * recon[i];
    }
    Metrics result{};
    result.fitted_scale = yy > 1e-30 ? static_cast<float>(xy / yy) : 0.f;
    double absolute_error = 0.0;
    double fitted_error = 0.0;
    double absolute_error_sum = 0.0;
    double bias_sum = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double raw_difference = static_cast<double>(recon[i]) - truth[i];
        const double fitted_difference =
            static_cast<double>(result.fitted_scale) * recon[i] - truth[i];
        absolute_error += raw_difference * raw_difference;
        fitted_error += fitted_difference * fitted_difference;
        absolute_error_sum += std::fabs(raw_difference);
        bias_sum += raw_difference;
    }
    result.correlation = xy / std::sqrt(std::max(xx * yy, 1e-30));
    result.absolute_nrmse = std::sqrt(absolute_error / std::max(xx, 1e-30));
    result.fitted_nrmse = std::sqrt(fitted_error / std::max(xx, 1e-30));
    result.mae = absolute_error_sum / truth.size();
    result.bias = bias_sum / truth.size();
    return result;
}

SHeliCTParam makeComparisonParams()
{
    SHeliCTParam h{};
    h.iPU = 96; h.iPV = 32;
    h.iVX = 64; h.iVY = 64; h.iVZ = 32;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 0.8f; h.vox_y_mm = 0.8f; h.vox_z_mm = 0.8f;
    h.SID = 160.f; h.SDD = 300.f;
    h.pitch_mm = 8.f;
    h.views_per_rot = 128;
    h.angle_list.resize(640);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / h.views_per_rot;
    h.start_z_mm = -0.5f * h.pitch_mm * h.angle_list.back() / (2.f * CUDA_PI);
    return h;
}

// Kernel 基准使用独立的大尺寸参数，避免改变其它功能/数值测试的运行规模。
// 该规模在常见 8 GB 显存上仍有余量，同时让单次 kernel 足够长，降低计时抖动。
SHeliCTParam makeKernelBenchmarkParams()
{
    SHeliCTParam h = makeComparisonParams();
    h.iPU = 256;
    h.iPV = 64;
    h.iVX = 128;
    h.iVY = 128;
    h.iVZ = 64;
    h.views_per_rot = 180;
    h.angle_list.resize(360);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / h.views_per_rot;
    h.start_z_mm = -0.5f * h.pitch_mm * h.angle_list.back() / (2.f * CUDA_PI);
    return h;
}

} // namespace

int main_fpcyl_adjoint()
{
    SHeliCTParam h{};
    h.iPU = 24; h.iPV = 12;
    h.iVX = 18; h.iVY = 16; h.iVZ = 14;
    h.du_mm = 1.f; h.dv_mm = 1.1f;
    h.vox_x_mm = 1.f; h.vox_y_mm = 0.9f; h.vox_z_mm = 1.2f;
    h.SID = 80.f; h.SDD = 150.f;
    h.pitch_mm = 3.f; h.start_z_mm = -2.f;
    h.views_per_rot = 16;
    h.angle_list.resize(24);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / h.views_per_rot;
    h.offsetU_mm = 0.35f;
    h.offsetV_mm = -0.7f;

    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    std::mt19937 generator(12345u);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::vector<float> x(volume_count), y(projection_count);
    for (float& value : x) value = distribution(generator);
    for (float& value : y) value = distribution(generator);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_x = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_at_y = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_y = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_ax = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    cudaMemcpyAsync(d_x.data(), x.data(), volume_count * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y.data(), y.data(), projection_count * sizeof(float),
        cudaMemcpyHostToDevice, stream);

    // 非理想圆弧：SDD=150 mm，而曲率半径 R=120 mm。圆柱轴线不经过焦点。
    const float curvature_radius_mm = 120.f;
    const auto geometry = TestGeometry::helicalCyl(
        h, curvature_radius_mm);
    std::vector<float> ax(projection_count), at_y(volume_count);
    CylFpBp::Config config{};
    const auto prepared_geometry = CylFpBp::detail::prepareJosephGeometry(
        volumeGeometry(h), h.iPU, h.iPV, geometry);
    CylFpBp::CylForwardOperator forward;
    CylFpBp::CylBackOperator back;
    auto volume_texture = makeTexture(d_x.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    auto projection_texture = makeTexture(d_y.data(), h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), cudaFilterModePoint, stream);
    bool ok = prepared_geometry && forward.prepare(prepared_geometry, config) &&
        back.prepare(prepared_geometry, config) &&
        forward.forward(volume_texture, d_ax.data(), stream) &&
        back.backproject(projection_texture, d_at_y.data(), stream) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    if (ok) {
        ok = cudaMemcpy(ax.data(), d_ax.data(), projection_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(at_y.data(), d_at_y.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    double lhs = 0.0, rhs = 0.0;
    for (size_t i = 0; i < projection_count; ++i)
        lhs += static_cast<double>(ax[i]) * y[i];
    for (size_t i = 0; i < volume_count; ++i)
        rhs += static_cast<double>(x[i]) * at_y[i];
    const double relative_error = std::fabs(lhs - rhs) /
        std::max({ std::fabs(lhs), std::fabs(rhs), 1e-30 });
    // BP 保留完整浮点 Joseph 权重；误差来源是 FP 的 Linear 纹理量化。
    ok = ok && std::isfinite(relative_error) && relative_error < 5e-3;
    YK_LOGI("[CylFpBp texture-FP-vs-joseph-BP] "
        "SDD={:.1f} R={:.1f} <Ax,y>={:.9e} <x,BPy>={:.9e} "
        "agreement={:.6f} rel={:.3e} {}",
        h.SDD, curvature_radius_mm, lhs, rhs, 1.0 - relative_error,
        relative_error, ok ? "PASS" : "FAIL");

    // Siddon FP/BP 共享同一条源到圆柱像素射线以及相同的体素段长度，
    // 因此除 atomicAdd 累加顺序造成的舍入外应满足离散转置关系。
    auto point_volume_texture = makeTexture(d_x.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModePoint, stream);
    bool siddon_ok = forward.forwardSiddon(point_volume_texture, d_ax.data(), stream) &&
        back.backprojectSiddon(projection_texture, d_at_y.data(), stream) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    if (siddon_ok) {
        siddon_ok = cudaMemcpy(ax.data(), d_ax.data(), projection_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(at_y.data(), d_at_y.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    double siddon_lhs = 0.0, siddon_rhs = 0.0;
    for (size_t i = 0; i < projection_count; ++i)
        siddon_lhs += static_cast<double>(ax[i]) * y[i];
    for (size_t i = 0; i < volume_count; ++i)
        siddon_rhs += static_cast<double>(x[i]) * at_y[i];
    const double siddon_relative = std::fabs(siddon_lhs - siddon_rhs) /
        std::max({std::fabs(siddon_lhs), std::fabs(siddon_rhs), 1e-30});
    siddon_ok = siddon_ok && std::isfinite(siddon_relative) &&
        siddon_relative < 1e-5;
    YK_LOGI("[CylFpBp Siddon-FP-vs-Siddon-BP] "
        "SDD={:.1f} R={:.1f} <Ax,y>={:.9e} <x,BPy>={:.9e} rel={:.3e} {}",
        h.SDD, curvature_radius_mm, siddon_lhs, siddon_rhs, siddon_relative,
        siddon_ok ? "PASS" : "FAIL");
    ok = ok && siddon_ok;
    forward.release();
    back.release();

    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_fdk_adjoint()
{
    // FDK 权重不是 Joseph FP 的离散转置；这里用同一 R=SDD 几何量化
    // 两种组合的内积偏差，避免把“matched”误解成 FP/BP 严格伴随。
    SHeliCTParam h{};
    h.iPU = 32; h.iPV = 16;
    h.iVX = 20; h.iVY = 18; h.iVZ = 12;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 1.f; h.vox_y_mm = 1.f; h.vox_z_mm = 1.f;
    h.SID = 160.f; h.SDD = 300.f;
    h.views_per_rot = 24;
    h.angle_list.resize(24);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * i / h.views_per_rot;

    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    std::mt19937 generator(87231u);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::vector<float> x(volume_count), y(projection_count);
    for (float& value : x) value = distribution(generator);
    for (float& value : y) value = distribution(generator);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_x = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_y = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_ax = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_bp = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_x.data(), x.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    YK_CUDA_CHECK(cudaMemcpyAsync(d_y.data(), y.data(),
        projection_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    const auto geometry = TestGeometry::staticCyl(h, h.SDD);
    auto volume_texture = makeTexture(d_x.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    CylFpBp::Config fp_config{};
    CylFpBp::CylForwardOperator fp;
    bool ok = fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
        fp_config) && fp.forward(volume_texture, d_ax.data(), stream) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    fp.release();

    std::vector<float> ax(projection_count), bp(volume_count);
    if (ok) {
        ok = cudaMemcpy(ax.data(), d_ax.data(), projection_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    double lhs = 0.0;
    for (size_t i = 0; i < projection_count; ++i)
        lhs += static_cast<double>(ax[i]) * y[i];

    auto measure = [&](bool matched, double& rhs, double& relative,
        double& ratio) {
        YK_CUDA_CHECK(cudaMemsetAsync(d_bp.data(), 0,
            volume_count * sizeof(float), stream));
        if (matched) {
            CylFpBp::FdkMatchedBackprojector backprojector;
            ok = ok && backprojector.prepare(volumeGeometry(h), h.iPU, h.iPV,
                geometry, stream) && backprojector.uploadProjection(d_y.data()) &&
                backprojector.backproject(d_bp.data(), false);
            backprojector.release();
        } else {
            CylFpBp::FdkBackprojector backprojector;
            ok = ok && backprojector.prepare(volumeGeometry(h), h.iPU, h.iPV,
                geometry, stream) && backprojector.uploadProjection(d_y.data()) &&
                backprojector.backproject(d_bp.data(), false);
            backprojector.release();
        }
        ok = ok && cudaStreamSynchronize(stream) == cudaSuccess &&
            cudaMemcpy(bp.data(), d_bp.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
        rhs = 0.0;
        for (size_t i = 0; i < volume_count; ++i)
            rhs += static_cast<double>(x[i]) * bp[i];
        relative = std::fabs(lhs - rhs) /
            std::max({std::fabs(lhs), std::fabs(rhs), 1e-30});
        ratio = std::fabs(lhs) > 1e-30 ? rhs / lhs : 0.0;
    };

    double fdk_rhs = 0.0, fdk_relative = 0.0, fdk_ratio = 0.0;
    double matched_rhs = 0.0, matched_relative = 0.0, matched_ratio = 0.0;
    if (ok) {
        measure(false, fdk_rhs, fdk_relative, fdk_ratio);
        measure(true, matched_rhs, matched_relative, matched_ratio);
    }
    YK_LOGI("[CylFpBp FP_Joseph + FDK BP] <Ax,y>={:.9e} <x,BPy>={:.9e} "
        "rel={:.6f} ratio={:.6f}", lhs, fdk_rhs, fdk_relative, fdk_ratio);
    YK_LOGI("[CylFpBp FP_Joseph + FDK-matched BP] <Ax,y>={:.9e} "
        "<x,BPy>={:.9e} rel={:.6f} ratio={:.6f}", lhs, matched_rhs,
        matched_relative, matched_ratio);
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_fdk_backprojectors()
{
    SHeliCTParam h{};
    h.iPU = 64; h.iPV = 48;
    h.iVX = 17; h.iVY = 15; h.iVZ = 9;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 1.f; h.vox_y_mm = 1.f; h.vox_z_mm = 1.f;
    h.SID = 160.f; h.SDD = 300.f;
    h.views_per_rot = 32;
    h.angle_list.resize(h.views_per_rot);
    for (int i = 0; i < h.views_per_rot; ++i)
        h.angle_list[i] = 2.f * CUDA_PI * i / h.views_per_rot;

    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    std::vector<float> projection(projection_count, 1.f);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_fdk = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_matched = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_projection.data(), projection.data(),
        projection_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    const auto geometry = TestGeometry::helicalCyl(h, h.SDD);
    CylFpBp::FdkBackprojector fdk;
    CylFpBp::FdkMatchedBackprojector matched;
    bool ok = fdk.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, stream) &&
        matched.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, stream) &&
        fdk.uploadProjection(d_projection.data()) &&
        matched.uploadProjection(d_projection.data()) &&
        fdk.backproject(d_fdk.data(), false) &&
        matched.backproject(d_matched.data(), false) &&
        cudaStreamSynchronize(stream) == cudaSuccess;

    std::vector<float> fdk_result(volume_count);
    std::vector<float> matched_result(volume_count);
    if (ok) {
        ok = cudaMemcpy(fdk_result.data(), d_fdk.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(matched_result.data(), d_matched.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    const size_t center = static_cast<size_t>(h.iVZ / 2) * h.iVX * h.iVY +
        static_cast<size_t>(h.iVY / 2) * h.iVX + h.iVX / 2;
    const float expected_fdk = static_cast<float>(h.angle_list.size());
    const float expected_matched = expected_fdk * h.SDD * h.SDD /
        (h.SID * h.SID) *
        (h.vox_x_mm * h.vox_y_mm * h.vox_z_mm) /
        (h.du_mm * h.dv_mm);
    const float fdk_relative_error = ok ?
        fabsf(fdk_result[center] - expected_fdk) / expected_fdk : INFINITY;
    const float matched_relative_error = ok ?
        fabsf(matched_result[center] - expected_matched) / expected_matched :
        INFINITY;
    // 圆轨迹由单精度 sin/cos 构造，matched 权重中的距离平方会累积约
    // 1e-5~1e-4 相对误差；阈值仍远小于错误 Jacobian 带来的量级偏差。
    ok = ok && fdk_relative_error < 1e-4f &&
        matched_relative_error < 1e-4f;

    // 离轴点覆盖圆弧 U、轴向 V 和两套空间权重。投影恒为 1，CPU 可直接
    // 对每个 view 的解析权重求和，不依赖 GPU kernel 的实现细节。
    const int test_x = h.iVX / 2 + 4;
    const int test_y = h.iVY / 2 + 2;
    const int test_z = h.iVZ / 2 + 2;
    const size_t off_center = static_cast<size_t>(test_z) * h.iVX * h.iVY +
        static_cast<size_t>(test_y) * h.iVX + test_x;
    const SVolGeom vg = volumeGeometry(h);
    const float3 world = make_float3(vg.origin().x + test_x * vg.vox_x,
        vg.origin().y + test_y * vg.vox_y,
        vg.origin().z + test_z * vg.vox_z);
    double expected_fdk_off_center = 0.0;
    double expected_matched_off_center = 0.0;
    for (const auto& view : geometry) {
        const float3 source = make_float3(view.source.x, view.source.y, view.source.z);
        SCylProjectionFrame frame{};
        ok = ok && deriveCylProjectionFrame(view, h.iPU, h.iPV, frame);
        const float3 radial = frame.radialUnit;
        const float3 axis = frame.axisUnit;
        const float3 d = make_float3(world.x - source.x, world.y - source.y,
            world.z - source.z);
        const float depth = d.x * radial.x + d.y * radial.y + d.z * radial.z;
        const float axial_direction = d.x * axis.x + d.y * axis.y + d.z * axis.z;
        const float distance_squared = d.x * d.x + d.y * d.y + d.z * d.z;
        const float transverse_squared = distance_squared -
            axial_direction * axial_direction;
        const float detector_axial = frame.radius_mm * axial_direction /
            std::sqrt(transverse_squared);
        const float detector_distance_squared = frame.radius_mm * frame.radius_mm +
            detector_axial * detector_axial;
        expected_fdk_off_center += h.SID * h.SID / (depth * depth);
        expected_matched_off_center +=
            detector_distance_squared * std::sqrt(detector_distance_squared) /
            (frame.radius_mm * distance_squared) *
            (h.vox_x_mm * h.vox_y_mm * h.vox_z_mm) /
            (h.du_mm * h.dv_mm);
    }
    const double fdk_off_center_error = ok ? std::fabs(
        fdk_result[off_center] - expected_fdk_off_center) /
        expected_fdk_off_center : INFINITY;
    const double matched_off_center_error = ok ? std::fabs(
        matched_result[off_center] - expected_matched_off_center) /
        expected_matched_off_center : INFINITY;
    ok = ok && fdk_off_center_error < 1e-4 &&
        matched_off_center_error < 1e-4;

    // 一般圆柱必须先映射到 R=SDD；解析 BP 不允许静默接受 R!=SDD。
    const auto unsupported = TestGeometry::helicalCyl(h, 240.f);
    CylFpBp::FdkBackprojector rejected;
    const bool rejects_general_cylinder = !rejected.prepare(volumeGeometry(h),
        h.iPU, h.iPV, unsupported, stream);
    ok = ok && rejects_general_cylinder;

    constexpr int repetitions = 20;
    cudaEvent_t start = nullptr, stop = nullptr;
    float fdk_ms = 0.f, matched_ms = 0.f;
    if (ok && cudaEventCreate(&start) == cudaSuccess &&
        cudaEventCreate(&stop) == cudaSuccess) {
        YK_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < repetitions; ++i)
            ok = ok && fdk.backproject(d_fdk.data(), false);
        YK_CUDA_CHECK(cudaEventRecord(stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&fdk_ms, start, stop));

        YK_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < repetitions; ++i)
            ok = ok && matched.backproject(d_matched.data(), false);
        YK_CUDA_CHECK(cudaEventRecord(stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&matched_ms, start, stop));
        fdk_ms /= repetitions;
        matched_ms /= repetitions;
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);

    YK_LOGI("[CylFdkBp] center={:.9e} expected={:.9e} rel={:.3e} "
        "off-center-rel={:.3e} time={:.3f} ms {}",
        fdk_result.empty() ? 0.f : fdk_result[center], expected_fdk,
        fdk_relative_error, fdk_off_center_error, fdk_ms,
        ok ? "PASS" : "FAIL");
    YK_LOGI("[CylFdkMatchedBp] center={:.9e} expected={:.9e} rel={:.3e} "
        "off-center-rel={:.3e} time={:.3f} ms R!=SDD-rejected={} {}",
        matched_result.empty() ? 0.f : matched_result[center], expected_matched,
        matched_relative_error, matched_off_center_error, matched_ms,
        rejects_general_cylinder,
        ok ? "PASS" : "FAIL");

    rejected.release();
    fdk.release();
    matched.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_cfdk_reconstruction()
{
    // 第一版 C-FDK 只验证其有明确定义的适用域：完整圆扫、源中心等角
    // 柱面且 R=SDD。投影由同一柱面几何下的 Joseph FP 生成，判定使用
    // 绝对衰减系数，相关系数不参与通过条件。
    SHeliCTParam h{};
    h.iPU = 160; h.iPV = 64;
    h.iVX = 64; h.iVY = 64; h.iVZ = 32;
    h.du_mm = 0.8f; h.dv_mm = 0.8f;
    h.vox_x_mm = 0.8f; h.vox_y_mm = 0.8f; h.vox_z_mm = 0.8f;
    h.SID = 160.f; h.SDD = 300.f;
    h.offsetU_mm = 1.2f;
    h.offsetV_mm = 1.6f;
    h.views_per_rot = 360;
    h.angle_list.resize(h.views_per_rot);
    for (int i = 0; i < h.views_per_rot; ++i)
        h.angle_list[i] = 2.f * CUDA_PI * i / h.views_per_rot;

    const SReconstructionParams params = toCbct(h);
    const std::vector<float> truth = TestPhantom::makeBasic(params);
    const size_t volume_count = truth.size();
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    const auto geometry = TestGeometry::helicalCyl(h, h.SDD);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_recon = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto truth_texture = makeTexture(d_truth.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    CylFpBp::Config fp_config{};
    fp_config.samples_per_voxel = 2.f;
    CylFpBp::CylForwardOperator fp;
    CylFpBp::FdkPipeline flat_fdk;
    bool ok = fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, fp_config) &&
        fp.forward(truth_texture, d_projection.data(), stream) &&
        flat_fdk.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
            SFilterKernelDesc::RamLak(), stream) &&
        flat_fdk.reconstruct(d_projection.data(), d_recon.data()) &&
        cudaStreamSynchronize(stream) == cudaSuccess;

    std::vector<float> reconstruction(volume_count);
    if (ok) {
        ok = cudaMemcpy(reconstruction.data(), d_recon.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    const Metrics metrics = compare(truth, reconstruction);
    double water_sum = 0.0, background_sum = 0.0;
    size_t water_count = 0, background_count = 0;
    const int z = h.iVZ / 2;
    for (int y = 0; y < h.iVY; ++y) {
        for (int x = 0; x < h.iVX; ++x) {
            const float dx = (x + 0.5f - 0.5f * h.iVX) * h.vox_x_mm;
            const float dy = (y + 0.5f - 0.5f * h.iVY) * h.vox_y_mm;
            const float radius = std::sqrt(dx * dx + dy * dy);
            const size_t index = (static_cast<size_t>(z) * h.iVY + y) * h.iVX + x;
            // basic phantom 的高衰减小球中心约在 r=5.7 mm；中心 ROI 收到
            // 3 mm，避免把 0.08 mm^-1 插入物混入 0.02 mm^-1 水区统计。
            if (radius < 3.f) { water_sum += reconstruction[index]; ++water_count; }
            if (radius > 23.f) { background_sum += reconstruction[index]; ++background_count; }
        }
    }
    const double water_mean = water_count ? water_sum / water_count : 0.0;
    const double background_mean = background_count
        ? background_sum / background_count : 0.0;
    YK_LOGI("[Flat-FDK] water expected=2.000000e-02 mean={:.6e} bias={:.6e} "
        "background={:.6e} MAE={:.6e} abs-NRMSE={:.6f} corr={:.6f}",
        water_mean, water_mean - 0.02, background_mean, metrics.mae,
        metrics.absolute_nrmse, metrics.correlation);

    // 保存水模体中心层，便于区分整体数值偏差和结构性变形。
    if (!reconstruction.empty()) {
        const auto artifact_dir = std::filesystem::path("out/test-artifacts");
        std::filesystem::create_directories(artifact_dir);
        std::vector<float> error(volume_count);
        for (size_t i = 0; i < volume_count; ++i)
            error[i] = reconstruction[i] - truth[i];
        TestImage::writeGrayMontageBmp(
            artifact_dir / "fpcyl_flat_fdk_water_center.bmp",
            {{&truth, h.iVX, h.iVY, h.iVZ, z, 1.f, 0.f, 0.1f, false},
             {&reconstruction, h.iVX, h.iVY, h.iVZ, z, 1.f, 0.f, 0.1f, false},
             {&error, h.iVX, h.iVY, h.iVZ, z, 1.f, -0.05f, 0.05f, false}},
            3, 2, 4);
        YK_LOGI("[Flat-FDK] water artifacts: {}",
            (artifact_dir / "fpcyl_flat_fdk_water_center.bmp").string());
    }

    // 初始阈值只排除 NaN、零输出和数量级错误；达到材料定量精度前，测试
    // 日志中的绝对值是继续修正滤波核与归一化的依据。
    ok = ok && std::isfinite(water_mean) && water_mean > 0.005 &&
        water_mean < 0.08 && std::fabs(background_mean) < 0.01;
    fp.release();
    flat_fdk.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_kernel_benchmark()
{
    // 性能项只覆盖算子执行，不包含 prepare、主机传输或任何重建流程。
    SHeliCTParam h = makeKernelBenchmarkParams();
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    YK_LOGI("[Cyl benchmark] detector={}x{} volume={}x{}x{} views={} "
        "rays={}", h.iPU, h.iPV, h.iVX, h.iVY, h.iVZ,
        h.angle_list.size(), static_cast<size_t>(h.iPU) * h.iPV *
            h.angle_list.size());
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_volume = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_backprojection = memory.allocateDevice3D<float>(
        h.iVX, h.iVY, h.iVZ, 0);
    auto d_reference_backprojection = memory.allocateDevice3D<float>(
        h.iVX, h.iVY, h.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_texture_projection_readback = memory.allocateDevice3D<float>(
        h.iPU, h.iPV, static_cast<int>(h.angle_list.size()), 0);
    std::vector<float> volume(volume_count, 0.02f);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_volume.data(), volume.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    const auto geometry = TestGeometry::helicalCyl(h, 240.f);
    // 先预热若干次，再对每次 launch 单独计时。中位数比简单总平均更能
    // 排除首次调度、动态升频和偶发系统抢占造成的离群值。
    constexpr int warmup_runs = 3;
    constexpr int timed_runs = 15;
    constexpr int repetitions = timed_runs; // 平板对照沿用同一计时次数
    CylFpBp::Config config{};

    // Cyl 专有 IForwardProjection 负责体积纹理的过滤模式与生命周期。
    // 三种模型共用同一个线性设备输入，调用方不再为 Joseph/Siddon 分别
    // 创建 Linear/Point 纹理。
    int device_id = 0;
    YK_CUDA_CHECK(cudaGetDevice(&device_id));
    ResourceContext forward_resources;
    forward_resources.attach(stream, device_id);
    bool interface_ok = true;
    for (const auto model : {
            CylFpBp::EForwardProjection::Joseph,
            CylFpBp::EForwardProjection::JosephMatchedReference,
            CylFpBp::EForwardProjection::Siddon }) {
        auto projection = CylFpBp::makeForwardProjection(model);
        interface_ok = interface_ok && projection &&
            projection->prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
                config, forward_resources) &&
            projection->apply(d_volume.data(), d_projection.data(), false,
                forward_resources) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        if (projection) projection->release();
    }
    YK_LOGI("[CylFpBp IForwardProjection] joseph/matched/siddon {}",
        interface_ok ? "PASS" : "FAIL");

    // IBackProjection 对外统一使用线性设备内存。Joseph/V3 支持一般曲率，
    // FDK 两种权重按其算法约束改用 R=SDD 的源中心等角几何。
    bool back_interface_ok = true;
    const auto fdk_geometry = TestGeometry::helicalCyl(h, h.SDD);
    for (const auto model : {
            CylFpBp::EBackProjection::Joseph,
            CylFpBp::EBackProjection::JosephV3,
            CylFpBp::EBackProjection::SiddonV2,
            CylFpBp::EBackProjection::Siddon,
            CylFpBp::EBackProjection::SiddonRayDriven,
            CylFpBp::EBackProjection::Fdk,
            CylFpBp::EBackProjection::FdkMatched }) {
        const bool needs_source_centered_arc =
            model == CylFpBp::EBackProjection::Fdk ||
            model == CylFpBp::EBackProjection::FdkMatched;
        const auto& model_geometry = needs_source_centered_arc
            ? fdk_geometry : geometry;
        auto backprojection = CylFpBp::makeBackProjection(model);
        back_interface_ok = back_interface_ok && backprojection &&
            backprojection->prepare(volumeGeometry(h), h.iPU, h.iPV,
                model_geometry, config, forward_resources) &&
            backprojection->apply(d_projection.data(),
                d_reference_backprojection.data(), false, forward_resources) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        if (backprojection) backprojection->release();
    }
    YK_LOGI("[CylFpBp IBackProjection] joseph/v3/siddon-v2/"
        "siddon-v3/siddon-ray-driven/fdk/fdk-matched {}",
        back_interface_ok ? "PASS" : "FAIL");

    const auto benchmark_geometry = CylFpBp::detail::prepareJosephGeometry(
        volumeGeometry(h), h.iPU, h.iPV, geometry);
    CylFpBp::CylForwardOperator op;
    CylFpBp::CylBackOperator bp;
    auto volume_texture = makeTexture(d_volume.data(), h.iVX, h.iVY,
        h.iVZ, cudaFilterModeLinear, stream);
    // Siddon 读取离散体素值，必须使用独立的 Point 纹理；不能把 Joseph 的
    // Linear 纹理复用于 Siddon，否则边界附近会发生隐式三线性插值。
    auto volume_point_texture = makeTexture(d_volume.data(), h.iVX, h.iVY,
        h.iVZ, cudaFilterModePoint, stream);
    auto projection_texture = Mem::TextureController::createEmptyTex3D(
        h.iPU, h.iPV, static_cast<int>(h.angle_list.size()),
        cudaFilterModePoint, cudaAddressModeBorder);
    auto wrong_linear_projection_texture =
        Mem::TextureController::createEmptyTex3D(h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), cudaFilterModeLinear,
            cudaAddressModeBorder);
    cudaEvent_t start = nullptr, stop = nullptr;
    bool ok = interface_ok && back_interface_ok &&
        cudaEventCreate(&start) == cudaSuccess &&
        cudaEventCreate(&stop) == cudaSuccess &&
        benchmark_geometry && op.prepare(benchmark_geometry, config) &&
        bp.prepare(benchmark_geometry, config) &&
        op.forward(volume_texture, d_projection.data(), stream) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    if (ok) {
        Mem::TextureController::updateTex3DFromDeviceAsync(projection_texture,
            d_projection.data(), h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), stream);
        ok = bp.backproject(projection_texture, d_backprojection.data(),
                stream) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
    }
    // Operator 的资源契约必须在 launch 前拒绝错误采样模式，避免把能运行但
    // 数值含义不同的纹理交给 kernel。三个调用均不得排入任何 GPU 工作。
    const bool sampling_contract_ok =
        !op.forward(volume_point_texture, d_projection.data(), stream) &&
        !op.forwardSiddon(volume_texture, d_projection.data(), stream) &&
        !bp.backproject(wrong_linear_projection_texture,
            d_backprojection.data(), stream);
    ok = ok && sampling_contract_ok;
    YK_LOGI("[CylFpBp texture-contract] wrong-filter-rejected={} {}",
        sampling_contract_ok, sampling_contract_ok ? "PASS" : "FAIL");
    float forward_ms = 0.f, backproject_ms = 0.f;
    if (ok) {
        for (int i = 0; i < warmup_runs; ++i)
            ok = ok && op.forward(volume_texture, d_projection.data(), stream);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        YK_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < repetitions; ++i)
            ok = ok && op.forward(volume_texture, d_projection.data(), stream);
        YK_CUDA_CHECK(cudaEventRecord(stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&forward_ms, start, stop));

        YK_CUDA_CHECK(cudaMemsetAsync(d_backprojection.data(), 0,
            volume_count * sizeof(float), stream));
        YK_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < repetitions; ++i)
            ok = ok && bp.backproject(projection_texture,
                d_backprojection.data(), stream, true);
        YK_CUDA_CHECK(cudaEventRecord(stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&backproject_ms, start, stop));
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    forward_ms /= repetitions;
    backproject_ms /= repetitions;
    YK_LOGI("[CylFpBp kernel:texture-main-axis] rays={} FP={:.3f} ms "
        "joseph-BP={:.3f} ms {}",
        static_cast<size_t>(h.iPU) * h.iPV * h.angle_list.size(), forward_ms,
        backproject_ms, ok ? "PASS" : "FAIL");
    op.release();
    bp.release();

    auto cylindrical_v3 = CylFpBp::makeBackProjection(
        CylFpBp::EBackProjection::JosephV3);
    cudaEvent_t cylindrical_v3_start = nullptr, cylindrical_v3_stop = nullptr;
    float cylindrical_v3_ms = 0.f;
    bool cylindrical_v3_ok = cudaEventCreate(&cylindrical_v3_start) ==
            cudaSuccess &&
        cudaEventCreate(&cylindrical_v3_stop) == cudaSuccess &&
        cylindrical_v3 && cylindrical_v3->prepare(volumeGeometry(h), h.iPU,
            h.iPV, geometry, config, forward_resources) &&
        cylindrical_v3->apply(d_projection.data(), d_backprojection.data(),
            false, forward_resources) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    if (cylindrical_v3_ok) {
        for (int i = 0; i < warmup_runs; ++i)
            cylindrical_v3_ok = cylindrical_v3_ok &&
                cylindrical_v3->apply(d_projection.data(),
                    d_backprojection.data(), true, forward_resources);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        YK_CUDA_CHECK(cudaEventRecord(cylindrical_v3_start, stream));
        for (int i = 0; i < repetitions; ++i)
            cylindrical_v3_ok = cylindrical_v3_ok &&
                cylindrical_v3->apply(d_projection.data(),
                    d_backprojection.data(), true, forward_resources);
        YK_CUDA_CHECK(cudaEventRecord(cylindrical_v3_stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(cylindrical_v3_stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&cylindrical_v3_ms,
            cylindrical_v3_start, cylindrical_v3_stop));
        cylindrical_v3_ms /= repetitions;

        // 性能循环使用 accumulate=true；数值比较前重新生成单次 V3 输出。
        cylindrical_v3_ok = cylindrical_v3->apply(d_projection.data(),
            d_backprojection.data(), false, forward_resources) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        std::vector<float> v3_result(volume_count);
        cylindrical_v3_ok = cylindrical_v3_ok &&
            cudaMemcpy(v3_result.data(), d_backprojection.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;

        CylFpBp::Config joseph_config{};
        CylFpBp::CylBackOperator joseph;
        auto joseph_projection_texture = makeTexture(d_projection.data(),
            h.iPU, h.iPV, static_cast<int>(h.angle_list.size()),
            cudaFilterModePoint, stream);
        cylindrical_v3_ok = cylindrical_v3_ok && joseph.prepare(
                volumeGeometry(h), h.iPU, h.iPV, geometry, joseph_config) &&
            joseph.backproject(joseph_projection_texture,
                d_backprojection.data(), stream, false) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        std::vector<float> joseph_result(volume_count);
        cylindrical_v3_ok = cylindrical_v3_ok &&
            cudaMemcpy(joseph_result.data(), d_backprojection.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        joseph.release();

        double vv = 0.0, mm = 0.0, vm = 0.0;
        for (size_t i = 0; i < volume_count; ++i) {
            cylindrical_v3_ok = cylindrical_v3_ok &&
                std::isfinite(v3_result[i]) && std::isfinite(joseph_result[i]);
            vv += static_cast<double>(v3_result[i]) * v3_result[i];
            mm += static_cast<double>(joseph_result[i]) * joseph_result[i];
            vm += static_cast<double>(v3_result[i]) * joseph_result[i];
        }
        const double correlation = vm / std::sqrt(std::max(vv * mm, 1e-30));
        const double norm_ratio = std::sqrt(vv / std::max(mm, 1e-30));
        const double fitted_scale = vm / std::max(vv, 1e-30);
        cylindrical_v3_ok = cylindrical_v3_ok && correlation > 0.95 &&
            vv > 0.0 && mm > 0.0;
        YK_LOGI("[CylFpBp v3-vs-joseph-BP] corr={:.6f} norm-ratio={:.6f} "
            "fit-scale={:.6f}", correlation, norm_ratio, fitted_scale);
    }
    if (cylindrical_v3_start) cudaEventDestroy(cylindrical_v3_start);
    if (cylindrical_v3_stop) cudaEventDestroy(cylindrical_v3_stop);
    YK_LOGI("[CylFpBp kernel:joseph-v3] rays={} BP={:.3f} ms {}",
        static_cast<size_t>(h.iPU) * h.iPV * h.angle_list.size(),
        cylindrical_v3_ms, cylindrical_v3_ok ? "PASS" : "FAIL");
    ok = ok && cylindrical_v3_ok;
    if (cylindrical_v3) cylindrical_v3->release();

    // 圆柱其余纹理算子：Siddon FP、FDK/FDK-matched BP。线性内存路径不统计。
    cudaEvent_t cyl_extra_start = nullptr, cyl_extra_stop = nullptr;
    const bool cyl_events = cudaEventCreate(&cyl_extra_start) == cudaSuccess &&
        cudaEventCreate(&cyl_extra_stop) == cudaSuccess;
    auto measure_cyl = [&](const char* name, const std::function<void()>& launch) {
        if (!cyl_events) return;
        for (int i = 0; i < warmup_runs; ++i) launch();
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> samples;
        samples.reserve(timed_runs);
        for (int i = 0; i < timed_runs; ++i) {
            YK_CUDA_CHECK(cudaEventRecord(cyl_extra_start, stream));
            launch();
            YK_CUDA_CHECK(cudaEventRecord(cyl_extra_stop, stream));
            YK_CUDA_CHECK(cudaEventSynchronize(cyl_extra_stop));
            float elapsed = 0.f;
            YK_CUDA_CHECK(cudaEventElapsedTime(&elapsed,
                cyl_extra_start, cyl_extra_stop));
            samples.push_back(elapsed);
        }
        std::sort(samples.begin(), samples.end());
        const float median = samples[samples.size() / 2];
        YK_LOGI("[Cyl kernel:{}] median={:.3f} ms min={:.3f} ms max={:.3f} ms "
            "(warmup={} samples={})", name, median, samples.front(),
            samples.back(), warmup_runs, timed_runs);
    };
    if (cyl_events) {
        CylFpBp::CylForwardOperator joseph_fp;
        if (joseph_fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, config)) {
            measure_cyl("joseph-fp", [&] {
                joseph_fp.forward(volume_texture, d_projection.data(), stream);
            });
            joseph_fp.release();
        } else {
            YK_LOGE("[Cyl kernel:joseph-fp] prepare failed");
            ok = false;
        }
        // Joseph 的 matched/reference FP 与普通 Joseph 使用同一几何，单独
        // 计时以便观察其额外的离散采样/权重开销，而不是把两者合并。
        CylFpBp::CylForwardOperator matched_fp;
        if (matched_fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, config)) {
            measure_cyl("joseph-matched-reference-fp", [&] {
                matched_fp.forwardMatchedReference(volume_point_texture,
                    d_projection.data(), stream);
            });
            matched_fp.release();
        } else {
            YK_LOGE("[Cyl kernel:joseph-matched-reference-fp] prepare failed");
            ok = false;
        }
        CylFpBp::CylForwardOperator siddon_fp;
        siddon_fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, config);
        measure_cyl("siddon-fp", [&] {
            siddon_fp.forwardSiddon(volume_point_texture, d_projection.data(), stream);
        });
        siddon_fp.release();
        auto siddon_v2 = CylFpBp::makeBackProjection(
            CylFpBp::EBackProjection::SiddonV2);
        ok = ok && siddon_v2 && siddon_v2->prepare(volumeGeometry(h), h.iPU,
            h.iPV, geometry, config, forward_resources);
        measure_cyl("siddon-v2-bp", [&] {
            siddon_v2->apply(d_projection.data(), d_backprojection.data(),
                true, forward_resources);
        });
        siddon_v2->release();
        auto siddon_v3 = CylFpBp::makeBackProjection(
            CylFpBp::EBackProjection::SiddonV3);
        ok = ok && siddon_v3 && siddon_v3->prepare(volumeGeometry(h), h.iPU,
            h.iPV, geometry, config, forward_resources);
        measure_cyl("siddon-v3-bp", [&] {
            siddon_v3->apply(d_projection.data(), d_backprojection.data(),
                true, forward_resources);
        });
        siddon_v3->release();
        CylFpBp::CylBackOperator joseph_bp;
        if (joseph_bp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, config)) {
            measure_cyl("joseph-bp", [&] {
                joseph_bp.backproject(projection_texture,
                    d_backprojection.data(), stream, true);
            });
            joseph_bp.release();
        } else {
            YK_LOGE("[Cyl kernel:joseph-bp] prepare failed");
            ok = false;
        }
        CylFpBp::CylBackOperator siddon_ray;
        siddon_ray.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, config);
        Mem::TextureController::updateTex3DFromDeviceAsync(projection_texture,
            d_projection.data(), h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), stream);
        measure_cyl("siddon-ray-driven-bp", [&] {
            siddon_ray.backprojectSiddon(projection_texture,
                d_backprojection.data(), stream, true);
        });
        siddon_ray.release();
        CylFpBp::FdkBackprojector fdk_bp;
        fdk_bp.prepare(volumeGeometry(h), h.iPU, h.iPV,
            TestGeometry::helicalCyl(h, h.SDD), stream);
        fdk_bp.uploadProjection(d_projection.data());
        measure_cyl("fdk-bp", [&] { fdk_bp.backproject(d_backprojection.data(), true); });
        fdk_bp.release();
        CylFpBp::FdkMatchedBackprojector matched_bp;
        matched_bp.prepare(volumeGeometry(h), h.iPU, h.iPV,
            TestGeometry::helicalCyl(h, h.SDD), stream);
        matched_bp.uploadProjection(d_projection.data());
        measure_cyl("fdk-matched-bp", [&] { matched_bp.backproject(d_backprojection.data(), true); });
        matched_bp.release();
    }
    if (cyl_extra_start) cudaEventDestroy(cyl_extra_start);
    if (cyl_extra_stop) cudaEventDestroy(cyl_extra_stop);

    // 平板对照使用同一螺旋轨迹、体积、探测器尺寸和视图数。直接调用底层
    // launch，纹理和派生几何均在计时前创建，避免公共算子 apply() 的资源
    // 重建开销破坏与 CylFpBp 的比较口径。
    const SReconstructionParams flat_params = toCbct(h);
    const SVolGeom flat_volume_geometry = volumeGeometry(h);
    std::vector<SConeProjGeomVec> flat_geometry;
    flat_geometry = TestGeometry::helicalFlat(h);
    std::vector<SFDKGeoParamPerView> flat_derived;
    bool flat_ok = GeoDerivedManagerVec{}.build_geo_params(h.iPU, h.iPV,
        flat_params.scan.range_rad, flat_geometry, flat_derived);
    Fp::FpGpuContext flat_fp;
    Fp::FpGpuContext flat_siddon_fp;
    Bp::BpSiddonGpuContext flat_bp;
    Mem::Tex3DHandle flat_point_projection_texture{};
    cudaEvent_t flat_start = nullptr, flat_stop = nullptr;
    float flat_fp_ms = 0.f, flat_matched_bp_ms = 0.f, flat_v3_bp_ms = 0.f;
    if (flat_ok) {
        flat_fp.init(d_volume.data(), flat_volume_geometry, flat_geometry, 0);
        flat_siddon_fp.init(d_volume.data(), flat_volume_geometry, flat_geometry,
            0, cudaFilterModePoint);
        Fp::fp_joseph_launch(flat_fp.volTex.tex, flat_fp.geo.h_views_vec(),
            flat_fp.geo.d_views_vox(), d_projection.data(), flat_volume_geometry,
            flat_params.scan.NAng, h.iPU, h.iPV, false, stream,
            Fp::FpStepSuperSample::x1);
        flat_ok = cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (flat_ok) {
        flat_bp.init(d_projection.data(), SProjDims{h.iPU, h.iPV,
            flat_params.scan.NAng}, flat_volume_geometry, flat_geometry, flat_derived,
            stream, 0);
        flat_point_projection_texture =
            Mem::TextureController::createTex3DFromDevice(d_projection.data(),
                h.iPU, h.iPV, flat_params.scan.NAng, cudaFilterModePoint,
                cudaAddressModeBorder);
        flat_ok = cudaEventCreate(&flat_start) == cudaSuccess &&
            cudaEventCreate(&flat_stop) == cudaSuccess;
    }
    if (flat_ok) {
        // 分别预热匹配的 ray-driven BP 和更快但非离散转置的 Joseph V3。
        Bp::joseph_bp_launch(flat_point_projection_texture.tex,
            flat_bp.geo.h_views_world_vec(), flat_bp.geo.d_views_vox(),
            d_backprojection.data(), flat_volume_geometry, flat_params.scan.NAng,
            h.iPU, h.iPV, true, stream, Bp::BpStepSuperSample::x1);
        Bp::joseph_bp_v3_launch(flat_bp.sinoTex.tex,
            flat_bp.geo.d_views_world(), flat_bp.geo.d_coeffs_data(),
            d_backprojection.data(), flat_volume_geometry, flat_params.scan.NAng,
            true, stream);
        flat_ok = cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (flat_ok) {
        YK_CUDA_CHECK(cudaEventRecord(flat_start, stream));
        for (int i = 0; i < repetitions; ++i)
            Fp::fp_joseph_launch(flat_fp.volTex.tex,
                flat_fp.geo.h_views_vec(), flat_fp.geo.d_views_vox(),
                d_projection.data(), flat_volume_geometry, flat_params.scan.NAng,
                h.iPU, h.iPV, false, stream, Fp::FpStepSuperSample::x1);
        YK_CUDA_CHECK(cudaEventRecord(flat_stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(flat_stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&flat_fp_ms, flat_start, flat_stop));

        // 公共 Linear 投影纹理供连续坐标 BP 使用；射线驱动 Joseph 只读取
        // 离散投影样本，必须使用独立的 Point 纹理。
        YK_CUDA_CHECK(cudaMemsetAsync(d_backprojection.data(), 0,
            volume_count * sizeof(float), stream));
        YK_CUDA_CHECK(cudaEventRecord(flat_start, stream));
        for (int i = 0; i < repetitions; ++i)
            Bp::joseph_bp_launch(flat_point_projection_texture.tex,
                flat_bp.geo.h_views_world_vec(), flat_bp.geo.d_views_vox(),
                d_backprojection.data(), flat_volume_geometry, flat_params.scan.NAng,
                h.iPU, h.iPV, true, stream, Bp::BpStepSuperSample::x1);
        YK_CUDA_CHECK(cudaEventRecord(flat_stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(flat_stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(
            &flat_matched_bp_ms, flat_start, flat_stop));

        YK_CUDA_CHECK(cudaMemsetAsync(d_backprojection.data(), 0,
            volume_count * sizeof(float), stream));
        YK_CUDA_CHECK(cudaEventRecord(flat_start, stream));
        for (int i = 0; i < repetitions; ++i)
            Bp::joseph_bp_v3_launch(flat_bp.sinoTex.tex,
                flat_bp.geo.d_views_world(), flat_bp.geo.d_coeffs_data(),
                d_backprojection.data(), flat_volume_geometry,
                flat_params.scan.NAng, true, stream);
        YK_CUDA_CHECK(cudaEventRecord(flat_stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(flat_stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&flat_v3_bp_ms, flat_start, flat_stop));

        // 其余平板算子逐项测量。Siddon 的 ray-driven、原始 voxel、V2
        // 和 V3 分开记录，避免把不同驱动方式误合并为一个耗时。
        auto measure_flat = [&](const char* name, const std::function<void()>& launch) {
            float elapsed = 0.f;
            YK_CUDA_CHECK(cudaMemsetAsync(d_backprojection.data(), 0,
                volume_count * sizeof(float), stream));
            YK_CUDA_CHECK(cudaEventRecord(flat_start, stream));
            for (int i = 0; i < repetitions; ++i) launch();
            YK_CUDA_CHECK(cudaEventRecord(flat_stop, stream));
            YK_CUDA_CHECK(cudaEventSynchronize(flat_stop));
            YK_CUDA_CHECK(cudaEventElapsedTime(&elapsed, flat_start, flat_stop));
            elapsed /= repetitions;
            YK_LOGI("[Flat kernel:{}] {:.3f} ms", name, elapsed);
        };
        measure_flat("siddon-fp", [&] {
            Fp::fp_siddon_launch(flat_siddon_fp.volTex.tex, d_projection.data(),
                flat_siddon_fp.geo.d_views(), flat_volume_geometry, h.iPU, h.iPV,
                flat_params.scan.NAng, false, stream);
        });
        // cudaArray 是显存的快照，不会跟随 d_projection 的后续写入。上面的
        // Siddon FP 已改写投影，因此在所有 Siddon BP 前显式刷新 Point 纹理。
        Mem::TextureController::updateTex3DFromDeviceAsync(
            flat_point_projection_texture, d_projection.data(), h.iPU, h.iPV,
            flat_params.scan.NAng, stream);
        Test::pointTextureReadback(
            flat_point_projection_texture.tex,
            d_texture_projection_readback.data(), h.iPU, h.iPV,
            flat_params.scan.NAng, stream);
        measure_flat("joseph-v2-bp", [&] {
            Bp::joseph_bp_v2_launch(flat_bp.sinoTex.tex,
                flat_bp.geo.d_views_world(), d_backprojection.data(),
                flat_volume_geometry, flat_params.scan.NAng, h.iPU, h.iPV,
                true, stream);
        });
        measure_flat("joseph-v3-bp", [&] {
            Bp::joseph_bp_v3_launch(flat_bp.sinoTex.tex,
                flat_bp.geo.d_views_world(), flat_bp.geo.d_coeffs_data(),
                d_backprojection.data(), flat_volume_geometry,
                flat_params.scan.NAng, true, stream);
        });
        measure_flat("fdk-bp", [&] {
            Bp::fdk_bp_launch(flat_bp.sinoTex.tex, flat_bp.geo.d_views_world(),
                flat_bp.geo.d_coeffs_data(), d_backprojection.data(),
                flat_volume_geometry, flat_params.scan.NAng, true, stream);
        });
        measure_flat("fdk-matched-bp", [&] {
            Bp::fdk_matched_bp_launch(flat_bp.sinoTex.tex,
                flat_bp.geo.d_views_world(), flat_bp.geo.d_coeffs_data(),
                d_backprojection.data(), flat_volume_geometry,
                flat_params.scan.NAng, true, stream);
        });
        measure_flat("siddon-bp-texture", [&] {
            Bp::bp_siddon_voxel_v2_launch(flat_bp.sinoTex.tex,
                d_backprojection.data(), flat_bp.geo.d_views_world(),
                flat_volume_geometry, h.iPU, h.iPV, flat_params.scan.NAng,
                true, stream);
        });
        measure_flat("siddon-voxel-bp-linear", [&] {
            Bp::bp_siddon_voxel_launch(d_projection.data(),
                d_backprojection.data(), flat_bp.geo.d_views_world(),
                flat_volume_geometry, h.iPU, h.iPV, flat_params.scan.NAng,
                true, stream);
        });
        measure_flat("siddon-voxel-v2-bp", [&] {
            Bp::bp_siddon_voxel_v2_launch(flat_point_projection_texture.tex,
                d_backprojection.data(), flat_bp.geo.d_views_world(),
                flat_volume_geometry, h.iPU, h.iPV, flat_params.scan.NAng,
                true, stream);
        });
        measure_flat("siddon-voxel-v3-bp", [&] {
            Bp::bp_siddon_voxel_v3_launch(flat_point_projection_texture.tex,
                d_backprojection.data(), flat_bp.geo.d_views_world(),
                flat_volume_geometry, h.iPU, h.iPV, flat_params.scan.NAng,
                true, stream);
        });
        measure_flat("siddon-ray-bp-texture", [&] {
            Bp::bp_siddon_launch(flat_point_projection_texture.tex,
                d_backprojection.data(),
                flat_bp.geo.d_views_world(), flat_volume_geometry, h.iPU, h.iPV,
                flat_params.scan.NAng, stream);
        });

        // Point 纹理采样先逐像素读回并要求位级一致；随后 BP 输出对照只量化
        // 两次独立 atomicAdd launch 的非确定累加误差。
        YK_CUDA_CHECK(cudaMemsetAsync(d_reference_backprojection.data(), 0,
            volume_count * sizeof(float), stream));
        YK_CUDA_CHECK(cudaMemsetAsync(d_backprojection.data(), 0,
            volume_count * sizeof(float), stream));
        Bp::bp_siddon_launch(d_projection.data(),
            d_reference_backprojection.data(), flat_bp.geo.d_views_world(),
            flat_volume_geometry, h.iPU, h.iPV, flat_params.scan.NAng, stream);
        Bp::bp_siddon_launch(flat_point_projection_texture.tex,
            d_backprojection.data(), flat_bp.geo.d_views_world(),
            flat_volume_geometry, h.iPU, h.iPV, flat_params.scan.NAng, stream);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> raw_bp(volume_count), texture_bp(volume_count);
        const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
            flat_params.scan.NAng;
        std::vector<float> raw_projection(projection_count);
        std::vector<float> texture_projection(projection_count);
        YK_CUDA_CHECK(cudaMemcpy(raw_bp.data(), d_reference_backprojection.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost));
        YK_CUDA_CHECK(cudaMemcpy(texture_bp.data(), d_backprojection.data(),
            volume_count * sizeof(float), cudaMemcpyDeviceToHost));
        YK_CUDA_CHECK(cudaMemcpy(raw_projection.data(), d_projection.data(),
            projection_count * sizeof(float), cudaMemcpyDeviceToHost));
        YK_CUDA_CHECK(cudaMemcpy(texture_projection.data(),
            d_texture_projection_readback.data(), projection_count * sizeof(float),
            cudaMemcpyDeviceToHost));
        double max_difference = 0.0, max_magnitude = 0.0;
        for (size_t i = 0; i < volume_count; ++i) {
            max_difference = std::max(max_difference,
                std::fabs(static_cast<double>(raw_bp[i]) - texture_bp[i]));
            max_magnitude = std::max(max_magnitude,
                std::fabs(static_cast<double>(raw_bp[i])));
        }
        const double relative_difference = max_difference /
            std::max(max_magnitude, 1e-30);
        double texture_read_difference = 0.0;
        for (size_t i = 0; i < projection_count; ++i) {
            texture_read_difference = std::max(texture_read_difference,
                std::fabs(static_cast<double>(raw_projection[i]) -
                    texture_projection[i]));
        }
        const bool texture_samples_exact = texture_read_difference == 0.0;
        const bool atomic_accumulation_stable = relative_difference < 5e-6;
        flat_ok = flat_ok && texture_samples_exact && atomic_accumulation_stable;
        YK_LOGI("[Flat Point texture readback] max-abs={:.3e} {}",
            texture_read_difference, texture_samples_exact ? "PASS" : "FAIL");
        YK_LOGI("[Flat Siddon RayDriven raw-vs-point-texture] max-abs={:.3e} "
            "max-relative={:.3e} atomic-tolerance=5e-6 {}", max_difference,
            relative_difference, atomic_accumulation_stable ? "PASS" : "FAIL");
    }
    if (flat_start) cudaEventDestroy(flat_start);
    if (flat_stop) cudaEventDestroy(flat_stop);
    flat_fp_ms /= repetitions;
    flat_matched_bp_ms /= repetitions;
    flat_v3_bp_ms /= repetitions;
    YK_LOGI("[Flat kernel:joseph] rays={} FP={:.3f} ms "
        "matched-BP={:.3f} ms voxel-BP-v3={:.3f} ms {}",
        static_cast<size_t>(h.iPU) * h.iPV * h.angle_list.size(), flat_fp_ms,
        flat_matched_bp_ms, flat_v3_bp_ms, flat_ok ? "PASS" : "FAIL");
    ok = ok && flat_ok;
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_wfbp_comparison()
{
    struct TestCase {
        const char* name;
        bool catphan;
        float offset_v;
        float curvature_radius_mm;
    };
    const TestCase cases[] = {
        { "basic-R=SDD-v0", false, 0.f, 300.f },
        { "basic-R240-v+1.5", false, 1.5f, 240.f },
        { "catphan-R360-v0", true, 0.f, 360.f },
        { "catphan-R240-v-1.5", true, -1.5f, 240.f }
    };
    struct Result {
        std::string name;
        std::vector<float> truth;
        std::vector<float> recon;
        std::vector<float> error;
        Metrics metrics;
        float fp_ms = 0.f;
        float recon_ms = 0.f;
    };

    SHeliCTParam base = makeComparisonParams();
    const size_t volume_count = static_cast<size_t>(base.iVX) * base.iVY * base.iVZ;
    const size_t projection_count = static_cast<size_t>(base.iPU) * base.iPV *
        base.angle_list.size();
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(base.iVX, base.iVY, base.iVZ, 0);
    auto d_recon = memory.allocateDevice3D<float>(base.iVX, base.iVY, base.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(base.iPU, base.iPV,
        static_cast<int>(base.angle_list.size()), 0);
    std::vector<Result> results;
    bool ok = true;

    for (const auto& test_case : cases) {
        SHeliCTParam h = base;
        h.offsetV_mm = test_case.offset_v;
        const SReconstructionParams p = toCbct(h);
        Result result{};
        result.name = test_case.name;
        result.truth = test_case.catphan ? TestPhantom::makeCatphanLike(p)
                                         : TestPhantom::makeBasic(p);
        result.recon.resize(volume_count);
        result.error.resize(volume_count);
        cudaMemcpyAsync(d_truth.data(), result.truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream);

        const float radius = test_case.curvature_radius_mm;
        const auto geometry = TestGeometry::helicalCyl(h, radius);
        CylFpBp::CylForwardOperator projector;
        CylFpBp::Config projector_config{};
        projector_config.samples_per_voxel = 2.f;
        Helical::Wfbp::Config config{};
        config.arc_principal_channel = 0.5f * (h.iPU - 1) - h.offsetU_mm / h.du_mm;
        // R=SDD 也走新适配层，验证一般圆柱映射能严格退化为等角弧面。
        config.input_detector = Helical::Wfbp::EInputDetector::CylindricalArc;
        config.arc_curvature_radius_mm = radius;
        Helical::Wfbp::Pipeline wfbp;
        cudaEvent_t start = nullptr, fp_stop = nullptr, recon_stop = nullptr;
        bool case_ok = cudaEventCreate(&start) == cudaSuccess &&
            cudaEventCreate(&fp_stop) == cudaSuccess &&
            cudaEventCreate(&recon_stop) == cudaSuccess &&
            projector.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
                projector_config) &&
            wfbp.prepare(TestGeometry::wfbpInput(h), TestGeometry::volume(h),
                config, stream);
        if (case_ok) {
            cudaEventRecord(start, stream);
            auto volume_texture = makeTexture(d_truth.data(), h.iVX, h.iVY,
                h.iVZ, cudaFilterModeLinear, stream);
            case_ok = projector.forward(volume_texture,
                d_projection.data(), stream);
            cudaEventRecord(fp_stop, stream);
            case_ok = case_ok && wfbp.reconstruct(d_projection.data(), d_recon.data());
            cudaEventRecord(recon_stop, stream);
            cudaEventSynchronize(recon_stop);
            cudaEventElapsedTime(&result.fp_ms, start, fp_stop);
            cudaEventElapsedTime(&result.recon_ms, fp_stop, recon_stop);
        }
        if (start) cudaEventDestroy(start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (recon_stop) cudaEventDestroy(recon_stop);
        if (case_ok) {
            cudaMemcpy(result.recon.data(), d_recon.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost);
            result.metrics = compare(result.truth, result.recon);
            for (size_t i = 0; i < volume_count; ++i)
                result.error[i] = result.recon[i] - result.truth[i];
            // 只用原始重建值判定；拟合比例和拟合 NRMSE 仅用于诊断单一
            // 增益误差，不能掩盖 CylFpBp 或 wFBP 的绝对幅值问题。
            case_ok = result.metrics.correlation > 0.90 &&
                result.metrics.absolute_nrmse < 0.25 &&
                result.metrics.mae < 0.01 && std::fabs(result.metrics.bias) < 0.005;
        }
        std::printf("CylFpBp -> wFBP %-22s corr %.6f abs-NRMSE %.6f "
            "MAE %.6e bias %.6e fit-scale %.6f fit-NRMSE %.6f "
            "FP %.3f ms recon %.3f ms %s\n", result.name.c_str(),
            result.metrics.correlation, result.metrics.absolute_nrmse,
            result.metrics.mae, result.metrics.bias, result.metrics.fitted_scale,
            result.metrics.fitted_nrmse, result.fp_ms, result.recon_ms,
            case_ok ? "PASS" : "FAIL");
        ok = ok && case_ok;
        projector.release();
        wfbp.release();
        results.push_back(std::move(result));
    }

    std::vector<TestImage::GrayPanel> panels;
    for (const auto& result : results) {
        const bool catphan = result.name.find("catphan") != std::string::npos;
        const float maximum = catphan ? 0.09f : 0.08f;
        panels.push_back({ &result.truth, base.iVX, base.iVY, base.iVZ,
            base.iVZ / 2, 1.f, 0.f, maximum, false });
        panels.push_back({ &result.recon, base.iVX, base.iVY, base.iVZ,
            base.iVZ / 2, 1.f, 0.f, maximum, false });
        panels.push_back({ &result.error, base.iVX, base.iVY, base.iVZ,
            base.iVZ / 2, 1.f, 0.f, 0.03f, true });
    }
    const auto artifact = std::filesystem::absolute(
        "out/test-artifacts/fpcyl_wfbp_comparison.bmp");
    const bool image_ok = TestImage::writeGrayMontageBmp(artifact, panels, 3, 8, 4);
    std::printf("CylFpBp wFBP image: %s (%s)\n", artifact.string().c_str(),
        image_ok ? "written" : "FAILED");
    cudaStreamDestroy(stream);
    return ok && image_ok ? 0 : 1;
}

int main_fpcyl_wfbp_flat_vs_equiangular()
{
    SHeliCTParam h = makeComparisonParams();
    const SReconstructionParams params = toCbct(h);
    const auto truth = TestPhantom::makeAstraSheppLogan3D(params,
        22.f, true, 0.02f);
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_flat_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_equi_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_flat_recon = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_equi_recon = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_flat_arc = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    if (!d_truth || !d_flat_projection || !d_equi_projection ||
        !d_flat_recon || !d_equi_recon || !d_flat_arc) {
        cudaStreamDestroy(stream);
        return 1;
    }
    bool ok = cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;

    const auto flat_geometry = TestGeometry::helicalFlat(h);
    ForwardOperatorAdapter flat_forward;
    CylFpBp::CylForwardOperator equiangular_forward;
    CylFpBp::Config cyl_fp_config{};
    cyl_fp_config.samples_per_voxel = 2.f;
    auto volume_texture = makeTexture(d_truth.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    Helical::Wfbp::Config flat_config{};
    flat_config.input_detector = Helical::Wfbp::EInputDetector::FlatPanel;
    Helical::Wfbp::Pipeline flat_wfbp;
    ok = ok && flat_forward.init(params, flat_geometry, ETask::FP_Joseph,
            0, stream) &&
        flat_wfbp.prepare(TestGeometry::wfbpInput(h), TestGeometry::volume(h),
            flat_config, stream);
    const float target_angle_step = ok ? flat_wfbp.geometry().fan_angle_step : 0.f;
    const auto equiangular_geometry = TestGeometry::helicalCyl(h, h.SDD,
        target_angle_step);
    ok = ok && flat_forward.run(d_truth.data(), params, d_flat_projection.data(), stream) &&
        equiangular_forward.prepare(volumeGeometry(h), h.iPU, h.iPV,
            equiangular_geometry, cyl_fp_config) &&
        equiangular_forward.forward(volume_texture, d_equi_projection.data(), stream);

    Helical::Wfbp::Config equiangular_config{};
    equiangular_config.input_detector = Helical::Wfbp::EInputDetector::EquiangularArc;
    equiangular_config.arc_channel_angle_step_rad = target_angle_step;
    equiangular_config.arc_principal_channel = 0.5f * (h.iPU - 1);
    Helical::Wfbp::Pipeline equiangular_wfbp;
    ok = ok && equiangular_wfbp.prepare(TestGeometry::wfbpInput(h),
            TestGeometry::volume(h), equiangular_config, stream);
    Helical::Wfbp::FlatToEquiangularArcProcessor flat_to_arc;
    ok = ok && flat_to_arc.prepare(flat_wfbp.geometry(), h.du_mm, flat_config) &&
        flat_to_arc.apply(d_flat_projection.data(), d_flat_arc.data(), stream);
    if (ok) {
        ok = flat_wfbp.reconstruct(d_flat_projection.data(), d_flat_recon.data()) &&
            equiangular_wfbp.reconstruct(d_equi_projection.data(),
                d_equi_recon.data()) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
    }

    std::vector<float> flat_recon(volume_count), equiangular_recon(volume_count);
    std::vector<float> flat_arc(projection_count), equiangular_projection(projection_count);
    if (ok) {
        ok = cudaMemcpy(flat_recon.data(), d_flat_recon.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(equiangular_recon.data(), d_equi_recon.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(flat_arc.data(), d_flat_arc.data(),
                projection_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(equiangular_projection.data(), d_equi_projection.data(),
                projection_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    if (ok) {
        const Metrics reconstruction_difference = compare(equiangular_recon,
            flat_recon);
        const Metrics projection_difference = compare(equiangular_projection,
            flat_arc);
        std::printf("[wFBP FlatPanel vs native EquiangularArc] Shepp-Logan, "
            "R=SDD=%.1f mm, dgamma=%.7f rad\n"
            "  arc-input projections: corr %.6f NRMSE %.6f MAE %.6e bias %.6e\n"
            "  reconstructed volumes: corr %.6f NRMSE %.6f MAE %.6e bias %.6e "
            "fit-NRMSE %.6f\n",
            h.SDD, equiangular_config.arc_channel_angle_step_rad,
            projection_difference.correlation, projection_difference.absolute_nrmse,
            projection_difference.mae, projection_difference.bias,
            reconstruction_difference.correlation,
            reconstruction_difference.absolute_nrmse,
            reconstruction_difference.mae, reconstruction_difference.bias,
            reconstruction_difference.fitted_nrmse);
        ok = std::all_of(flat_recon.begin(), flat_recon.end(),
                [](float value) { return std::isfinite(value); }) &&
            std::all_of(equiangular_recon.begin(), equiangular_recon.end(),
                [](float value) { return std::isfinite(value); });
    }

    flat_wfbp.release();
    equiangular_wfbp.release();
    flat_forward.release();
    equiangular_forward.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_iterative_comparison()
{
    SHeliCTParam h{};
    h.iPU = 1024; h.iPV = 256;
    h.iVX = 181; h.iVY = 217; h.iVZ = 41;
    h.du_mm = 0.35f; h.dv_mm = 0.35f;
    h.vox_x_mm = 1.f; h.vox_y_mm = 1.f; h.vox_z_mm = 1.f;
    h.SID = 710.f; h.SDD = 1170.f;
    h.views_per_rot = 360;
    h.angle_list.resize(360);
    for (int i = 0; i < 360; ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / 360.f;

    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    const std::filesystem::path phantom_path = std::filesystem::path(
        YKCBCT_TEST_SOURCE_DIR) / "example/phantom/brainweb_normal_1mm_181x217x181_uint8.raw";
    std::ifstream input(phantom_path, std::ios::binary);
    constexpr int source_nz = 181;
    constexpr int crop_z = 70;
    const size_t source_slice = static_cast<size_t>(h.iVX) * h.iVY;
    std::vector<unsigned char> source_labels(source_slice * source_nz);
    if (!input.read(reinterpret_cast<char*>(source_labels.data()),
            static_cast<std::streamsize>(source_labels.size()))) {
        YK_LOGE("无法读取 BrainWeb 模体: {}", phantom_path.string());
        return 1;
    }
    std::vector<unsigned char> labels(volume_count);
    std::copy_n(source_labels.data() + static_cast<size_t>(crop_z) * source_slice,
        volume_count, labels.data());
    std::vector<float> truth(volume_count);
    std::transform(labels.begin(), labels.end(), truth.begin(),
        [](unsigned char label) { return 0.01f * label; });

    struct TestCase {
        const char* name;
        CylFpBp::EIterativeMethod method;
        CylFpBp::EIterativeForwardModel forward_model;
        CylFpBp::EIterativeBackprojectorModel backprojector_model;
        int iterations;
        int subsets;
        float relaxation;
        float reduction;
    };
    // SART 的一次外循环已经包含 360 次单视图更新；其迭代数不能与
    // SIRT/CGLS 的全投影更新次数直接比较。
    const TestCase cases[] = {
        // 常用快速组合：Joseph FP + JosephV3 BP。
        { "sirt-joseph-joseph-v3", CylFpBp::EIterativeMethod::Sirt,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::JosephV3, 3, 1, 1.f, 1.f },
        { "sart-joseph-joseph-v3", CylFpBp::EIterativeMethod::Sart,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::JosephV3, 1, 360, 0.2f, 1.f },
        { "ossart-joseph-siddon-v3", CylFpBp::EIterativeMethod::Ossart,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::SiddonV3, 3, 20, 0.2f, 0.98f },
        // 两组高伴随组合必须参与：严格 Siddon 转置，以及 Point Joseph 参考组合。
        { "cgls-siddon-ray-driven", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::Siddon,
            CylFpBp::EIterativeBackprojectorModel::SiddonRayDriven, 2, 1, 1.f, 1.f },
        { "cgls-joseph-matched", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::JosephMatchedReference,
            CylFpBp::EIterativeBackprojectorModel::Joseph, 2, 1, 1.f, 1.f },
        // FDK 风格 BP 可与迭代方法自由组合，作为工程近似对照。
        { "cgls-joseph-fdk-matched", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::FdkMatched, 2, 1, 1.f, 1.f }
    };

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_recon = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 迭代组合包含 FDK 风格 BP；该 BP 的标准柱面约束为 R=SDD，
    // 因此这里使用标称 SDD 曲率，避免把不兼容的 R=900 几何误判为算法失败。
    const auto geometry = TestGeometry::helicalCyl(h, h.SDD);
    CylFpBp::Config operator_config{};
    operator_config.samples_per_voxel = 1.f;
    CylFpBp::CylForwardOperator forward;
    auto truth_texture = makeTexture(d_truth.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    bool ok = forward.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
        operator_config) && forward.forward(truth_texture,
        d_projection.data(), stream);
    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    forward.release();
    if (!ok) {
        cudaStreamDestroy(stream);
        return 1;
    }

    std::vector<TestImage::GrayPanel> panels;
    std::vector<std::vector<float>> panel_storage;
    panel_storage.reserve(std::size(cases) * 3);
    for (const auto& test : cases) {
        YK_CUDA_CHECK(cudaMemsetAsync(d_recon.data(), 0,
            volume_count * sizeof(float), stream));
        CylFpBp::IterativeConfig config{};
        config.method = test.method;
        config.iterations = test.iterations;
        config.subset_count = test.subsets;
        config.relaxation = test.relaxation;
        config.relaxation_reduction = test.reduction;
        config.forward_model = test.forward_model;
        config.backprojector_model = test.backprojector_model;
        config.nonnegative = test.method != CylFpBp::EIterativeMethod::Cgls;
        config.use_max = test.method != CylFpBp::EIterativeMethod::Cgls;
        config.maximum = 0.12f;

        const auto started = std::chrono::steady_clock::now();
        Helical::Iterative::CylConfig heli_config{};
        heli_config.method = test.method == CylFpBp::EIterativeMethod::Sirt ?
            Helical::Iterative::EMethod::Sirt :
            test.method == CylFpBp::EIterativeMethod::Sart ?
                Helical::Iterative::EMethod::Sart :
            test.method == CylFpBp::EIterativeMethod::Cgls ?
                Helical::Iterative::EMethod::Cgls :
                Helical::Iterative::EMethod::Ossart;
        heli_config.algebraic = config;
        heli_config.operators = operator_config;
        Helical::Iterative::CylReconstructor reconstructor;
        bool case_ok = reconstructor.prepare(volumeGeometry(h), h.iPU, h.iPV,
                geometry, heli_config, stream) &&
            reconstructor.reconstruct(d_projection.data(), d_recon.data());
        case_ok = case_ok && cudaStreamSynchronize(stream) == cudaSuccess;
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::vector<float> reconstruction(volume_count);
        if (case_ok)
            case_ok = cudaMemcpy(reconstruction.data(), d_recon.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        reconstructor.release();

        const Metrics metrics = compare(truth, reconstruction);
        std::array<double, 10> sums{};
        std::array<size_t, 10> counts{};
        std::array<double, 10> core_sums{};
        std::array<size_t, 10> core_counts{};
        for (size_t i = 0; i < volume_count; ++i) {
            const unsigned int label = labels[i];
            if (label < sums.size()) {
                sums[label] += reconstruction[i];
                ++counts[label];
            }
            case_ok = case_ok && std::isfinite(reconstruction[i]);
        }
        const size_t slice = static_cast<size_t>(h.iVX) * h.iVY;
        for (int z = 1; z + 1 < h.iVZ; ++z)
            for (int y = 1; y + 1 < h.iVY; ++y)
                for (int x = 1; x + 1 < h.iVX; ++x) {
                    const size_t i = (static_cast<size_t>(z) * h.iVY + y) *
                        h.iVX + x;
                    const unsigned int label = labels[i];
                    if (label >= core_sums.size() || labels[i - 1] != label ||
                        labels[i + 1] != label || labels[i - h.iVX] != label ||
                        labels[i + h.iVX] != label || labels[i - slice] != label ||
                        labels[i + slice] != label) continue;
                    core_sums[label] += reconstruction[i];
                    ++core_counts[label];
                }
        YK_LOGI("[CylIterative:{}] corr={:.6f} abs-NRMSE={:.6f} "
            "MAE={:.6e} bias={:.6e} time={:.3f} ms", test.name,
            metrics.correlation, metrics.absolute_nrmse, metrics.mae,
            metrics.bias, elapsed_ms);
        for (size_t label = 0; label < sums.size(); ++label) {
            const double mean = counts[label] ? sums[label] / counts[label] : 0.0;
            const double core_mean = core_counts[label]
                ? core_sums[label] / core_counts[label] : 0.0;
            YK_LOGI("[CylIterative:{}] material {} expected={:.6e} "
                "mean={:.6e} bias={:.6e} voxels={} core-mean={:.6e} "
                "core-bias={:.6e} core-voxels={}", test.name, label,
                0.01 * label, mean, mean - 0.01 * label, counts[label],
                core_mean, core_mean - 0.01 * label, core_counts[label]);
        }

        const auto raw_path = std::filesystem::absolute(
            std::filesystem::path("out/test-artifacts") /
            (std::string("fpcyl_") + test.name + "_brainweb.raw"));
        std::filesystem::create_directories(raw_path.parent_path());
        std::ofstream raw_output(raw_path, std::ios::binary);
        raw_output.write(reinterpret_cast<const char*>(reconstruction.data()),
            static_cast<std::streamsize>(reconstruction.size() * sizeof(float)));
        case_ok = case_ok && raw_output.good();

        panel_storage.push_back(truth);
        panel_storage.push_back(std::move(reconstruction));
        panel_storage.emplace_back(volume_count);
        for (size_t i = 0; i < volume_count; ++i)
            panel_storage.back()[i] = panel_storage[panel_storage.size() - 2][i] - truth[i];
        const size_t base = panel_storage.size() - 3;
        panels.push_back({ &panel_storage[base], h.iVX, h.iVY, h.iVZ,
            h.iVZ / 2, 1.f, 0.f, 0.09f, false });
        panels.push_back({ &panel_storage[base + 1], h.iVX, h.iVY, h.iVZ,
            h.iVZ / 2, 1.f, 0.f, 0.09f, false });
        panels.push_back({ &panel_storage[base + 2], h.iVX, h.iVY, h.iVZ,
            h.iVZ / 2, 1.f, 0.f, 0.03f, true });
        ok = ok && case_ok;
    }

    const auto artifact = std::filesystem::absolute(
        "out/test-artifacts/fpcyl_iterative_comparison.bmp");
    ok = ok && TestImage::writeGrayMontageBmp(artifact, panels, 3, 8, 4);
    YK_LOGI("[CylIterative] image: {}", artifact.string());
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// ASTRA modified 3-D Shepp-Logan 椭球模体上的 Cyl 重建矩阵。
// 同一份物理 R!=SDD 投影同时用于代数重建和解析重建：代数算子直接消费
// 物理 geometry，解析路径先 map 到 R=SDD 的虚拟等角柱面。
int main_fpcyl_astra_ellipse_matrix()
{
    SHeliCTParam h{};
    h.iPU = 192; h.iPV = 96;
    h.iVX = 96; h.iVY = 96; h.iVZ = 96;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 0.75f; h.vox_y_mm = 0.75f; h.vox_z_mm = 0.75f;
    h.SID = 160.f; h.SDD = 300.f;
    h.views_per_rot = 180;
    h.angle_list.resize(180);
    for (int i = 0; i < 180; ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / 180.f;
    constexpr float physical_radius = 240.f;
    const auto physical_geometry = TestGeometry::helicalCyl(
        h, physical_radius);
    const SVolGeom vg = volumeGeometry(h);
    const SReconstructionParams phantom_params = toCbct(h);
    const auto truth = TestPhantom::makeAstraSheppLogan3D(
        phantom_params, 32.f, true, 0.02f);
    const size_t volume_count = truth.size();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_reconstruction = memory.allocateDevice3D<float>(h.iVX, h.iVY,
        h.iVZ, 0);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    auto truth_texture = makeTexture(d_truth.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    CylFpBp::Config operator_config{};
    operator_config.samples_per_voxel = 2.f;
    CylFpBp::CylForwardOperator simulator;
    bool ok = simulator.prepare(vg, h.iPU, h.iPV, physical_geometry,
            operator_config) &&
        simulator.forward(truth_texture, d_projection.data(), stream) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    simulator.release();
    if (!ok) { cudaStreamDestroy(stream); return 1; }

    // 单独验证解析重建前端的曲率重排。仅比较 map 前后并不能判断重排
    // 是否正确，因此还在规范 R=SDD geometry 上重新正投同一模体，作为
    // 每条目标射线的参考值。输出图固定按“物理/map/规范参考”排列。
    CylFpBp::Analytic::GeometryCanonicalizer canonicalizer;
    CylFpBp::Analytic::CanonicalGeometry canonical{};
    CylFpBp::Analytic::GeometryCanonicalizationConfig canonical_config{};
    canonical_config.source_to_detector_mm = h.SDD;
    bool map_ok = canonicalizer.canonicalize(h.iPU, h.iPV, physical_geometry,
        canonical_config, canonical);
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    auto d_mapped_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_canonical_reference = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    CylFpBp::Analytic::ProjectionMapper projection_mapper;
    CylFpBp::CylForwardOperator canonical_simulator;
    map_ok = map_ok && projection_mapper.prepare(canonical.projection_map) &&
        projection_mapper.apply(d_projection.data(), d_mapped_projection.data(),
            stream) &&
        canonical_simulator.prepare(vg, h.iPU, h.iPV,
            canonical.canonical_geometry, operator_config) &&
        canonical_simulator.forward(truth_texture, d_canonical_reference.data(),
            stream) && cudaStreamSynchronize(stream) == cudaSuccess;

    std::vector<float> physical_projection(projection_count);
    std::vector<float> mapped_projection(projection_count);
    std::vector<float> canonical_reference(projection_count);
    std::vector<float> flat_reference(projection_count);
    if (map_ok) {
        map_ok = cudaMemcpy(physical_projection.data(), d_projection.data(),
                projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(mapped_projection.data(), d_mapped_projection.data(),
                projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess &&
            cudaMemcpy(canonical_reference.data(), d_canonical_reference.data(),
                projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                cudaSuccess;
    }
    projection_mapper.release();
    canonical_simulator.release();

    struct ProjectionStats {
        double mean = 0.0;
        double rms = 0.0;
        float maximum = 0.f;
        double nonzero_ratio = 0.0;
        bool finite = true;
    };
    const auto projection_stats = [](const std::vector<float>& projection) {
        ProjectionStats result{};
        size_t nonzero = 0;
        double square_sum = 0.0;
        for (const float value : projection) {
            result.finite = result.finite && std::isfinite(value);
            result.mean += value;
            square_sum += static_cast<double>(value) * value;
            result.maximum = std::max(result.maximum, std::fabs(value));
            nonzero += std::fabs(value) > 1e-8f;
        }
        if (!projection.empty()) {
            result.mean /= projection.size();
            result.rms = std::sqrt(square_sum / projection.size());
            result.nonzero_ratio = static_cast<double>(nonzero) /
                projection.size();
        }
        return result;
    };
    const ProjectionStats physical_stats = projection_stats(physical_projection);
    const ProjectionStats mapped_stats = projection_stats(mapped_projection);
    const ProjectionStats reference_stats = projection_stats(canonical_reference);
    const Metrics map_metrics = compare(canonical_reference, mapped_projection);
    YK_LOGI("[CylEllipse:projection-map] physical max={:.6e} mean={:.6e} "
        "rms={:.6e} nonzero={:.6f}", physical_stats.maximum,
        physical_stats.mean, physical_stats.rms, physical_stats.nonzero_ratio);
    YK_LOGI("[CylEllipse:projection-map] mapped   max={:.6e} mean={:.6e} "
        "rms={:.6e} nonzero={:.6f}", mapped_stats.maximum, mapped_stats.mean,
        mapped_stats.rms, mapped_stats.nonzero_ratio);
    YK_LOGI("[CylEllipse:projection-map] reference max={:.6e} mean={:.6e} "
        "rms={:.6e} nonzero={:.6f}", reference_stats.maximum,
        reference_stats.mean, reference_stats.rms, reference_stats.nonzero_ratio);
    YK_LOGI("[CylEllipse:projection-map] mapped-vs-reference corr={:.6f} "
        "NRMSE={:.6f} MAE={:.6e} bias={:.6e}", map_metrics.correlation,
        map_metrics.absolute_nrmse, map_metrics.mae, map_metrics.bias);
    map_ok = map_ok && physical_stats.finite && mapped_stats.finite &&
        reference_stats.finite && mapped_stats.maximum > 1e-8f &&
        reference_stats.maximum > 1e-8f;
    ok = ok && map_ok;

    // 对照：将同一份 rebin 后投影直接送入普通平板 FDK。这里把
    // canonical R=SDD 柱面的局部基向量展开为平板 detector geometry，
    // 以隔离 rebin 误差与 CylFdkPipeline 自身误差。
    auto d_flat_fdk = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
    auto d_flat_reference = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    std::vector<SConeProjGeomVec> flat_geometry;
    flat_geometry.reserve(canonical.canonical_geometry.size());
    for (const auto& cylindrical : canonical.canonical_geometry) {
        SCylProjectionFrame frame{};
        if (!deriveCylProjectionFrame(cylindrical, h.iPU, h.iPV, frame)) {
            map_ok = false;
            break;
        }
        const float du = frame.radius_mm * frame.channelStepRad;
        const float dv = frame.rowStepMm;
        const float3 s = make_float3(cylindrical.source.x,
            cylindrical.source.y, cylindrical.source.z);
        // 目标平板法向沿 radialUnit；其中心像素射线为 (SDD,0,0)，
        // 因此用 target principal 保留 detector offset。
        const float3 c = make_float3(
            s.x + frame.radialUnit.x * h.SDD,
            s.y + frame.radialUnit.y * h.SDD,
            s.z + frame.radialUnit.z * h.SDD);
        const float3 u = make_float3(frame.tangentUnit.x * du,
            frame.tangentUnit.y * du, frame.tangentUnit.z * du);
        const float3 v = make_float3(frame.axisUnit.x * dv,
            frame.axisUnit.y * dv, frame.axisUnit.z * dv);
        SConeProjGeomVec flat{};
        flat.src = make_float4(s.x, s.y, s.z, cylindrical.source.w);
        const float target_pu = canonical.projection_map.target_principal_u;
        const float target_pv = canonical.projection_map.target_principal_v;
        flat.detS = make_float4(c.x - target_pu * u.x - target_pv * v.x,
            c.y - target_pu * u.y - target_pv * v.y,
            c.z - target_pu * u.z - target_pv * v.z,
            cylindrical.detectorCenter.w);
        flat.detU = make_float4(u.x, u.y, u.z, cylindrical.detectorU.w);
        flat.detV = make_float4(v.x, v.y, v.z, cylindrical.detectorV.w);
        flat.angle = cylindrical.viewParameters;
        flat_geometry.push_back(flat);
    }
    bool flat_forward_ok = false;
    if (map_ok && flat_geometry.size() == canonical.canonical_geometry.size()) {
        Fp::FpGpuContext flat_context;
        flat_context.init(d_truth.data(), vg, flat_geometry, 0,
            cudaFilterModeLinear, stream);
        YK_CUDA_CHECK(cudaMemsetAsync(d_flat_reference.data(), 0,
            projection_count * sizeof(float), stream));
        fp_joseph_launch(flat_context.volTex.tex, flat_context.geo.h_views_vec(),
            flat_context.geo.d_views_vox(), d_flat_reference.data(), vg,
            static_cast<int>(flat_geometry.size()), h.iPU, h.iPV, false,
            stream, Fp::FpStepSuperSample::x1);
        flat_forward_ok = cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (flat_forward_ok) {
        cudaMemcpy(flat_reference.data(), d_flat_reference.data(),
            projection_count * sizeof(float), cudaMemcpyDeviceToHost);
        const Metrics flat_metrics = compare(mapped_projection, flat_reference);
        YK_LOGI("[CylEllipse:map-flat-forward] corr={:.6f} NRMSE={:.6f} "
            "MAE={:.6e} bias={:.6e}", flat_metrics.correlation,
            flat_metrics.absolute_nrmse, flat_metrics.mae, flat_metrics.bias);
    } else {
        map_ok = false;
    }
    bool flat_fdk_ok = map_ok && flat_geometry.size() == canonical.canonical_geometry.size();
    std::vector<float> flat_fdk_reconstruction(volume_count, 0.f);
    if (flat_fdk_ok) {
        SReconstructionParams flat_params = toCbct(h);
        flat_params.reconstruction.filter = SFilterKernelDesc::RamLak();
        FdkPipeline flat_fdk;
        flat_fdk_ok = flat_fdk.prepareWithGeometry(flat_params, flat_geometry,
            64, stream) &&
            flat_fdk.enqueueBatch({d_mapped_projection.data(), nullptr,
                nullptr, static_cast<int>(flat_geometry.size())},
                d_flat_fdk.data(), true, nullptr);
        if (flat_fdk_ok) flat_fdk_ok = cudaStreamSynchronize(stream) == cudaSuccess;
        if (flat_fdk_ok) flat_fdk_ok = cudaMemcpy(flat_fdk_reconstruction.data(),
            d_flat_fdk.data(), volume_count * sizeof(float),
            cudaMemcpyDeviceToHost) == cudaSuccess;
        flat_fdk.release();
    }
    const Metrics flat_fdk_metrics = compare(truth, flat_fdk_reconstruction);
    YK_LOGI("[CylEllipse:flat-fdk-reference] corr={:.6f} NRMSE={:.6f} "
        "MAE={:.6e} bias={:.6e} {}", flat_fdk_metrics.correlation,
        flat_fdk_metrics.absolute_nrmse, flat_fdk_metrics.mae,
        flat_fdk_metrics.bias, flat_fdk_ok ? "PASS" : "FAIL");
    ok = ok && flat_fdk_ok;

    const float projection_window_max = std::max({physical_stats.maximum,
        mapped_stats.maximum, reference_stats.maximum, 1e-6f});
    const std::vector<TestImage::GrayPanel> projection_panels = {
        {&physical_projection, h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), 37, 1.f, 0.f,
            projection_window_max, false},
        {&mapped_projection, h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), 37, 1.f, 0.f,
            projection_window_max, false},
        {&canonical_reference, h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), 37, 1.f, 0.f,
            projection_window_max, false}
    };
    const auto projection_output = std::filesystem::absolute(
        "out/test-artifacts/fpcyl_projection_map_comparison.bmp");
    map_ok = TestImage::writeGrayMontageBmp(projection_output,
        projection_panels, 3, 8, 3) && map_ok;
    YK_LOGI("[CylEllipse:projection-map] montage physical/map/reference: {}",
        projection_output.string());
    ok = ok && map_ok;

    struct IterativeCase {
        const char* name;
        CylFpBp::EIterativeMethod method;
        CylFpBp::EIterativeForwardModel fp;
        CylFpBp::EIterativeBackprojectorModel bp;
    };
    const IterativeCase cases[] = {
        {"sart-joseph-joseph-v3", CylFpBp::EIterativeMethod::Sart,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::JosephV3},
        {"sart-joseph-siddon-v3", CylFpBp::EIterativeMethod::Sart,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::SiddonV3},
        {"sart-siddon-ray-driven", CylFpBp::EIterativeMethod::Sart,
            CylFpBp::EIterativeForwardModel::Siddon,
            CylFpBp::EIterativeBackprojectorModel::SiddonRayDriven},
        {"sart-joseph-matched", CylFpBp::EIterativeMethod::Sart,
            CylFpBp::EIterativeForwardModel::JosephMatchedReference,
            CylFpBp::EIterativeBackprojectorModel::Joseph},
        {"cgls-joseph-joseph-v3", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::JosephV3},
        {"cgls-joseph-siddon-v3", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::Joseph,
            CylFpBp::EIterativeBackprojectorModel::SiddonV3},
        {"cgls-siddon-ray-driven", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::Siddon,
            CylFpBp::EIterativeBackprojectorModel::SiddonRayDriven},
        {"cgls-joseph-matched", CylFpBp::EIterativeMethod::Cgls,
            CylFpBp::EIterativeForwardModel::JosephMatchedReference,
            CylFpBp::EIterativeBackprojectorModel::Joseph}
    };

    std::vector<std::vector<float>> images;
    std::vector<TestImage::GrayPanel> panels;
    images.reserve(std::size(cases) + 3);
    images.push_back(truth);
    images.push_back(flat_fdk_reconstruction);
    for (const auto& test : cases) {
        YK_CUDA_CHECK(cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream));
        CylFpBp::IterativeConfig config{};
        config.method = test.method;
        // CGLS 前几轮主要恢复低频分量；提高到 10 轮后再观察边缘
        // 清晰度和绝对量级，避免将“3 轮尚未收敛”误判为算法模糊。
        config.iterations = test.method == CylFpBp::EIterativeMethod::Sart ? 1 : 10;
        config.subset_count = test.method == CylFpBp::EIterativeMethod::Sart ?
            static_cast<int>(h.angle_list.size()) : 1;
        config.relaxation = test.method == CylFpBp::EIterativeMethod::Sart ? 0.2f : 1.f;
        config.nonnegative = test.method == CylFpBp::EIterativeMethod::Sart;
        config.forward_model = test.fp;
        config.backprojector_model = test.bp;
        const auto start = std::chrono::steady_clock::now();
        Helical::Iterative::CylConfig heli_config{};
        heli_config.method = test.method == CylFpBp::EIterativeMethod::Sart ?
            Helical::Iterative::EMethod::Sart :
            Helical::Iterative::EMethod::Cgls;
        heli_config.algebraic = config;
        heli_config.operators = operator_config;
        Helical::Iterative::CylReconstructor reconstruction;
        bool case_ok = reconstruction.prepare(vg, h.iPU, h.iPV,
                physical_geometry, heli_config, stream) &&
            reconstruction.reconstruct(d_projection.data(),
                d_reconstruction.data()) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        images.emplace_back(volume_count);
        if (case_ok) case_ok = cudaMemcpy(images.back().data(),
            d_reconstruction.data(), volume_count * sizeof(float),
            cudaMemcpyDeviceToHost) == cudaSuccess;
        const Metrics metrics = compare(truth, images.back());
        float truth_maximum = 0.f, reconstruction_maximum = 0.f;
        double truth_sum = 0.0, reconstruction_sum = 0.0;
        size_t truth_nonzero = 0;
        for (size_t index = 0; index < volume_count; ++index) {
            truth_maximum = std::max(truth_maximum, std::fabs(truth[index]));
            reconstruction_maximum = std::max(reconstruction_maximum,
                std::fabs(images.back()[index]));
            if (truth[index] > 1e-6f) {
                truth_sum += truth[index];
                reconstruction_sum += images.back()[index];
                ++truth_nonzero;
            }
        }
        const double roi_truth_mean = truth_nonzero
            ? truth_sum / truth_nonzero : 0.0;
        const double roi_reconstruction_mean = truth_nonzero
            ? reconstruction_sum / truth_nonzero : 0.0;
        YK_LOGI("[CylEllipse:{}] corr={:.6f} NRMSE={:.6f} MAE={:.6e} "
            "bias={:.6e} fit-scale={:.6f} roi-mean={:.6e}/{:.6e} "
            "max={:.6e}/{:.6e} time={:.3f} ms {}", test.name,
            metrics.correlation, metrics.absolute_nrmse, metrics.mae,
            metrics.bias, metrics.fitted_scale, roi_reconstruction_mean,
            roi_truth_mean, reconstruction_maximum, truth_maximum, elapsed,
            case_ok ? "PASS" : "FAIL");
        ok = ok && case_ok;
        reconstruction.release();
    }

    // 解析路径故意输入 R=240 的物理投影，验证前端 map 到 R=SDD=300
    // 后才进入 CylFdkPipeline，而不是绕过规范化直接执行解析 BP。
    YK_CUDA_CHECK(cudaMemsetAsync(d_reconstruction.data(), 0,
        volume_count * sizeof(float), stream));
    const auto analytic_start = std::chrono::steady_clock::now();
    CylFpBp::Analytic::Reconstruction analytic;
    bool analytic_ok = analytic.prepare(vg, h.iPU, h.iPV, physical_geometry,
            h.SDD, stream) &&
        analytic.reconstruct(d_projection.data(), d_reconstruction.data(), true) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    const double analytic_elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - analytic_start).count();
    images.emplace_back(volume_count);
    if (analytic_ok) analytic_ok = cudaMemcpy(images.back().data(),
        d_reconstruction.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess;
    const Metrics analytic_metrics = compare(truth, images.back());
    float analytic_maximum = 0.f;
    bool analytic_finite = true;
    for (const float value : images.back()) {
        analytic_finite = analytic_finite && std::isfinite(value);
        analytic_maximum = std::max(analytic_maximum, std::fabs(value));
    }
    // 调用成功不代表解析结果有效；至少排除 NaN/Inf 和静默全零。
    // 绝对量级偏差保留在 NRMSE/MAE/bias 中，后续归一化修正不能被拟合
    // 比例或相关系数掩盖。
    analytic_ok = analytic_ok && analytic_finite && analytic_maximum > 1e-8f;
    YK_LOGI("[CylEllipse:cyl-map-flat-fdk] physical-R={:.1f} mapped-R={:.1f} "
        "corr={:.6f} NRMSE={:.6f} MAE={:.6e} bias={:.6e} time={:.3f} ms {}",
        physical_radius, h.SDD, analytic_metrics.correlation,
        analytic_metrics.absolute_nrmse, analytic_metrics.mae,
        analytic_metrics.bias, analytic_elapsed, analytic_ok ? "PASS" : "FAIL");
    ok = ok && analytic_ok;
    analytic.release();

    for (auto& image : images)
        panels.push_back({&image, h.iVX, h.iVY, h.iVZ, h.iVZ / 2,
            1.f, 0.f, 0.022f, false});
    const auto output = std::filesystem::absolute(
        "out/test-artifacts/fpcyl_astra_ellipse_matrix.bmp");
    std::filesystem::create_directories(output.parent_path());
    // truth + flat FDK 对照 + 8 个迭代组合 + 1 个解析结果。
    ok = TestImage::writeGrayMontageBmp(output, panels, 5, 8, 4) && ok;
    YK_LOGI("[CylEllipse] montage: {}", output.string());
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// Cyl PWLS 的静态/螺旋几何回归。每种轨迹分别验证高伴随 Joseph 和严格
// Siddon 数据项，并提供非均匀 W，确保执行的是 PWLS 而非未加权 PLS。
int main_fpcyl_pwls_static_helical()
{
    SHeliCTParam base{};
    base.iPU = 96; base.iPV = 48;
    base.iVX = 48; base.iVY = 48; base.iVZ = 48;
    base.du_mm = 1.f; base.dv_mm = 1.f;
    base.vox_x_mm = 0.75f; base.vox_y_mm = 0.75f; base.vox_z_mm = 0.75f;
    base.SID = 160.f; base.SDD = 300.f;
    base.views_per_rot = 90;
    // toCbct() 会从角度数组派生范围；模体本身不依赖视图数，但这里仍需
    // 提供有效的占位圆扫，避免在测试轨迹循环前读取空数组。
    base.angle_list = {0.f, 2.f * CUDA_PI};
    constexpr float radius = 240.f;

    struct Trajectory { const char* name; int views; float pitch; float start_z; };
    const Trajectory trajectories[] = {
        {"static", 90, 0.f, 0.f},
        {"helical", 180, 16.f, -16.f}
    };
    struct Model { const char* name; CylFpBp::ECylPwlsDataModel kind; };
    const Model models[] = {
        {"joseph-matched", CylFpBp::ECylPwlsDataModel::JosephMatched},
        {"siddon", CylFpBp::ECylPwlsDataModel::Siddon}
    };

    const SReconstructionParams phantom_params = toCbct(base);
    const auto truth = TestPhantom::makeAstraSheppLogan3D(
        phantom_params, 16.f, true, 0.02f);
    const size_t volume_count = truth.size();
    const SVolGeom vg = volumeGeometry(base);
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(base.iVX, base.iVY,
        base.iVZ, 0);
    auto d_volume = memory.allocateDevice3D<float>(base.iVX, base.iVY,
        base.iVZ, 0);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    auto truth_texture = makeTexture(d_truth.data(), base.iVX, base.iVY,
        base.iVZ, cudaFilterModePoint, stream);

    bool ok = true;
    std::vector<std::vector<float>> images;
    images.push_back(truth);
    CylFpBp::Config operator_config{};
    operator_config.samples_per_voxel = 2.f;
    for (const auto& trajectory : trajectories) {
        SHeliCTParam h = base;
        h.pitch_mm = trajectory.pitch;
        h.start_z_mm = trajectory.start_z;
        h.angle_list.resize(trajectory.views);
        for (int view = 0; view < trajectory.views; ++view)
            h.angle_list[view] = 2.f * CUDA_PI * view / h.views_per_rot;
        const auto geometry = trajectory.pitch == 0.f
            ? TestGeometry::staticCyl(h, radius)
            : TestGeometry::helicalCyl(h, radius);
        const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
            trajectory.views;
        auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
            trajectory.views, 0);
        std::vector<float> weights(projection_count);
        for (int view = 0; view < trajectory.views; ++view) {
            for (int row = 0; row < h.iPV; ++row) {
                for (int channel = 0; channel < h.iPU; ++channel) {
                    const float normalized = std::fabs((channel + 0.5f) /
                        h.iPU - 0.5f) * 2.f;
                    const size_t index = (static_cast<size_t>(view) * h.iPV +
                        row) * h.iPU + channel;
                    weights[index] = 1.f - 0.35f * normalized;
                }
            }
        }
        auto d_weights = memory.allocateDevice3D<float>(h.iPU, h.iPV,
            trajectory.views, 0);
        YK_CUDA_CHECK(cudaMemcpyAsync(d_weights.data(), weights.data(),
            projection_count * sizeof(float), cudaMemcpyHostToDevice, stream));

        for (const auto& model : models) {
            CylFpBp::CylForwardOperator simulator;
            bool case_ok = simulator.prepare(vg, h.iPU, h.iPV, geometry,
                operator_config);
            case_ok = case_ok && (model.kind ==
                CylFpBp::ECylPwlsDataModel::Siddon
                ? simulator.forwardSiddon(truth_texture, d_projection.data(), stream)
                : simulator.forwardMatchedReference(truth_texture,
                    d_projection.data(), stream));
            simulator.release();
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume.data(), 0,
                volume_count * sizeof(float), stream));
            CylFpBp::CylPwlsConfig config{};
            config.iterations = 20;
            config.relaxation = 0.8f;
            config.regularizer = Iter::EPwlsRegularizer::Huber;
            config.regularization = 1e-4f;
            config.huber_delta = 2e-3f;
            config.data_model = model.kind;
            config.projection_weights = d_weights.data();
            Helical::Iterative::CylConfig heli_config{};
            heli_config.method = Helical::Iterative::EMethod::Pwls;
            heli_config.pwls = config;
            heli_config.operators = operator_config;
            Helical::Iterative::CylReconstructor reconstructor;
            const auto started = std::chrono::steady_clock::now();
            case_ok = case_ok && reconstructor.prepare(vg, h.iPU, h.iPV,
                    geometry, heli_config, stream) &&
                reconstructor.reconstruct(d_projection.data(), d_volume.data()) &&
                cudaStreamSynchronize(stream) == cudaSuccess;
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            images.emplace_back(volume_count);
            if (case_ok) case_ok = cudaMemcpy(images.back().data(),
                d_volume.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
            const Metrics metrics = compare(truth, images.back());
            const float maximum = *std::max_element(images.back().begin(),
                images.back().end());
            case_ok = case_ok && std::all_of(images.back().begin(),
                images.back().end(), [](float value) { return std::isfinite(value); }) &&
                maximum > 1e-6f && metrics.correlation > 0.75;
            YK_LOGI("[CylPWLS:{}:{}] corr={:.6f} NRMSE={:.6f} "
                "MAE={:.6e} bias={:.6e} fit-scale={:.6f} max={:.6e} "
                "time={:.3f} ms {}", trajectory.name, model.name,
                metrics.correlation, metrics.absolute_nrmse, metrics.mae,
                metrics.bias, metrics.fitted_scale, maximum, elapsed,
                case_ok ? "PASS" : "FAIL");
            ok = ok && case_ok;
            reconstructor.release();
        }
    }
    std::vector<TestImage::GrayPanel> panels;
    for (auto& image : images)
        panels.push_back({&image, base.iVX, base.iVY, base.iVZ,
            base.iVZ / 2, 1.f, 0.f, 0.022f, false});
    const auto output = std::filesystem::absolute(
        "out/test-artifacts/fpcyl_pwls_static_helical.bmp");
    ok = TestImage::writeGrayMontageBmp(output, panels, 5, 8, 4) && ok;
    YK_LOGI("[CylPWLS] montage truth/static-joseph/static-siddon/"
        "helical-joseph/helical-siddon: {}", output.string());
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
