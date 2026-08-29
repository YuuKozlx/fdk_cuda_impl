#include <algorithm>
#include <array>
#include <cmath>
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
#include "CylFpBp/YkCylFpBpGeometry.hpp"
#include "CylFpBp/YkCylBackProjection.hpp"
#include "CylFpBp/YkCylBackOperator.hpp"
#include "CylFpBp/YkCylForwardProjection.hpp"
#include "CylFpBp/YkCylIterativeReconstructor.hpp"
#include "CylFpBp/YkCylForwardOperator.hpp"
#include "CylFpBp/YkCylFdkBackprojector.hpp"
#include "CylFpBp/YkCylFdkPipeline.hpp"
#include "CylFpBp/YkCylVoxelDrivenBackprojector.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"
#include "FP/YkFPGpuContext.hpp"
#include "FP/kernels/YkFPLaunch.cuh"
#include "Heli/YkHelicalGeo.hpp"
#include "Heli/wfbp/YkWfbpPipeline.hpp"
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
    const auto geometry = CylFpBp::buildCylindricalArcGeometry(
        h, curvature_radius_mm);
    std::vector<float> ax(projection_count), at_y(volume_count);
    CylFpBp::Config config{};
    const auto prepared_geometry = CylFpBp::detail::prepareJosephGeometry(
        volumeGeometry(h), h.iPU, h.iPV, geometry);
    CylFpBp::ForwardOperator forward;
    CylFpBp::BackOperator back;
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

    const auto geometry = CylFpBp::buildFreeCtArcGeometry(h);
    auto volume_texture = makeTexture(d_x.data(), h.iVX, h.iVY, h.iVZ,
        cudaFilterModeLinear, stream);
    CylFpBp::Config fp_config{};
    CylFpBp::ForwardOperator fp;
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

    const auto geometry = CylFpBp::buildFreeCtArcGeometry(h);
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
    const auto unsupported = CylFpBp::buildCylindricalArcGeometry(h, 240.f);
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
    const auto geometry = CylFpBp::buildFreeCtArcGeometry(h);

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
    CylFpBp::ForwardOperator fp;
    CylFpBp::FdkPipeline cfdk;
    bool ok = fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, fp_config) &&
        fp.forward(truth_texture, d_projection.data(), stream) &&
        cfdk.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
            SFilterKernelDesc::RamLak(), stream) &&
        cfdk.reconstruct(d_projection.data(), d_recon.data()) &&
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
    YK_LOGI("[C-FDK] water expected=2.000000e-02 mean={:.6e} bias={:.6e} "
        "background={:.6e} MAE={:.6e} abs-NRMSE={:.6f} corr={:.6f}",
        water_mean, water_mean - 0.02, background_mean, metrics.mae,
        metrics.absolute_nrmse, metrics.correlation);

    // 初始阈值只排除 NaN、零输出和数量级错误；达到材料定量精度前，测试
    // 日志中的绝对值是继续修正滤波核与归一化的依据。
    ok = ok && std::isfinite(water_mean) && water_mean > 0.005 &&
        water_mean < 0.08 && std::fabs(background_mean) < 0.01;
    fp.release();
    cfdk.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_kernel_benchmark()
{
    // 性能项只覆盖算子执行，不包含 prepare、主机传输或任何重建流程。
    SHeliCTParam h = makeComparisonParams();
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
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

    const auto geometry = CylFpBp::buildCylindricalArcGeometry(h, 240.f);
    // 单次 kernel 仅数毫秒，增加重复次数以减小 GPU 动态升频和事件分辨率
    // 对结果的影响；日志输出仍为每次算子的平均耗时。
    constexpr int repetitions = 20;
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
    const auto fdk_geometry = CylFpBp::buildFreeCtArcGeometry(h);
    for (const auto model : {
            CylFpBp::EBackProjection::Joseph,
            CylFpBp::EBackProjection::VoxelDrivenV3,
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
    YK_LOGI("[CylFpBp IBackProjection] joseph/v3/fdk/fdk-matched {}",
        back_interface_ok ? "PASS" : "FAIL");

    const auto benchmark_geometry = CylFpBp::detail::prepareJosephGeometry(
        volumeGeometry(h), h.iPU, h.iPV, geometry);
    CylFpBp::ForwardOperator op;
    CylFpBp::BackOperator bp;
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

    CylFpBp::VoxelDrivenBackprojectorV3 cylindrical_v3;
    cudaEvent_t cylindrical_v3_start = nullptr, cylindrical_v3_stop = nullptr;
    float cylindrical_v3_ms = 0.f;
    bool cylindrical_v3_ok = cudaEventCreate(&cylindrical_v3_start) ==
            cudaSuccess &&
        cudaEventCreate(&cylindrical_v3_stop) == cudaSuccess &&
        cylindrical_v3.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry,
            stream) &&
        cylindrical_v3.uploadProjection(d_projection.data()) &&
        cylindrical_v3.backproject(d_backprojection.data(), false) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
    if (cylindrical_v3_ok) {
        YK_CUDA_CHECK(cudaEventRecord(cylindrical_v3_start, stream));
        for (int i = 0; i < repetitions; ++i)
            cylindrical_v3_ok = cylindrical_v3_ok &&
                cylindrical_v3.backproject(d_backprojection.data(), true);
        YK_CUDA_CHECK(cudaEventRecord(cylindrical_v3_stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(cylindrical_v3_stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&cylindrical_v3_ms,
            cylindrical_v3_start, cylindrical_v3_stop));
        cylindrical_v3_ms /= repetitions;

        // 性能循环使用 accumulate=true；数值比较前重新生成单次 V3 输出。
        cylindrical_v3_ok = cylindrical_v3.backproject(
            d_backprojection.data(), false) &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        std::vector<float> v3_result(volume_count);
        cylindrical_v3_ok = cylindrical_v3_ok &&
            cudaMemcpy(v3_result.data(), d_backprojection.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;

        CylFpBp::Config joseph_config{};
        CylFpBp::BackOperator joseph;
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
    YK_LOGI("[CylFpBp kernel:voxel-driven-v3] rays={} BP={:.3f} ms {}",
        static_cast<size_t>(h.iPU) * h.iPV * h.angle_list.size(),
        cylindrical_v3_ms, cylindrical_v3_ok ? "PASS" : "FAIL");
    ok = ok && cylindrical_v3_ok;
    cylindrical_v3.release();

    // 圆柱其余纹理算子：Siddon FP、FDK/FDK-matched BP。线性内存路径不统计。
    cudaEvent_t cyl_extra_start = nullptr, cyl_extra_stop = nullptr;
    const bool cyl_events = cudaEventCreate(&cyl_extra_start) == cudaSuccess &&
        cudaEventCreate(&cyl_extra_stop) == cudaSuccess;
    auto measure_cyl = [&](const char* name, const std::function<void()>& launch) {
        if (!cyl_events) return;
        float elapsed = 0.f;
        YK_CUDA_CHECK(cudaEventRecord(cyl_extra_start, stream));
        for (int i = 0; i < repetitions; ++i) launch();
        YK_CUDA_CHECK(cudaEventRecord(cyl_extra_stop, stream));
        YK_CUDA_CHECK(cudaEventSynchronize(cyl_extra_stop));
        YK_CUDA_CHECK(cudaEventElapsedTime(&elapsed, cyl_extra_start, cyl_extra_stop));
        YK_LOGI("[Cyl kernel:{}] {:.3f} ms", name, elapsed / repetitions);
    };
    if (cyl_events) {
        CylFpBp::ForwardOperator siddon_fp;
        siddon_fp.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry, config);
        measure_cyl("siddon-fp", [&] {
            siddon_fp.forwardSiddon(volume_point_texture, d_projection.data(), stream);
        });
        siddon_fp.release();
        CylFpBp::FdkBackprojector fdk_bp;
        fdk_bp.prepare(volumeGeometry(h), h.iPU, h.iPV,
            CylFpBp::buildFreeCtArcGeometry(h), stream);
        fdk_bp.uploadProjection(d_projection.data());
        measure_cyl("fdk-bp", [&] { fdk_bp.backproject(d_backprojection.data(), true); });
        fdk_bp.release();
        CylFpBp::FdkMatchedBackprojector matched_bp;
        matched_bp.prepare(volumeGeometry(h), h.iPU, h.iPV,
            CylFpBp::buildFreeCtArcGeometry(h), stream);
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
    build_helical_vec_geometry(flat_geometry, h);
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
        // 分别预热匹配的 ray-driven BP 和更快但非离散转置的 voxel-driven v3。
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

        // 其余平板纹理算子逐项测量；旧的线性内存 Siddon 路径不纳入统计。
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
        const auto geometry = CylFpBp::buildCylindricalArcGeometry(h, radius);
        CylFpBp::ForwardOperator projector;
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
            wfbp.prepare(h, config, stream);
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
        int iterations;
        int subsets;
        float relaxation;
        float reduction;
    };
    // SART 的一次外循环已经包含 360 次单视图更新；其迭代数不能与
    // SIRT/CGLS 的全投影更新次数直接比较。
    const TestCase cases[] = {
        { "sirt", CylFpBp::EIterativeMethod::Sirt, 10, 1, 1.f, 1.f },
        { "sart", CylFpBp::EIterativeMethod::Sart, 1, 360, 0.2f, 1.f },
        { "ossart", CylFpBp::EIterativeMethod::Ossart, 10, 20, 0.2f, 0.98f },
        { "cgls", CylFpBp::EIterativeMethod::Cgls, 10, 1, 1.f, 1.f }
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

    const auto geometry = CylFpBp::buildCylindricalArcGeometry(h, 900.f);
    CylFpBp::Config operator_config{};
    operator_config.samples_per_voxel = 1.f;
    CylFpBp::ForwardOperator forward;
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
        config.nonnegative = test.method != CylFpBp::EIterativeMethod::Cgls;
        config.use_max = test.method != CylFpBp::EIterativeMethod::Cgls;
        config.maximum = 0.12f;

        const auto started = std::chrono::steady_clock::now();
        CylFpBp::IterativeReconstructor reconstructor;
        bool case_ok = reconstructor.prepare(volumeGeometry(h), h.iPU, h.iPV,
            geometry, config, operator_config, stream) &&
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
