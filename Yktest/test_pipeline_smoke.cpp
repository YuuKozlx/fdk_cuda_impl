#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include <global/YkGlobals.h>
#include <utility>
#include "FDK/CFDK/YkCurveFilteredFdkPipeline.hpp"
#include "FDK/YkFdkPipeline.hpp"
#include "FDK/XFDK/YkXfdkPipeline.hpp"
#include "FDK/kernels/YkFDKBpPrecompute.cuh"
#include "YkTestPhantoms.hpp"
#include "YkTestImage.hpp"
#include "common/YkProjectionOperators.hpp"
#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "test_common.hpp"
#include "util/YkVecOperation.hpp"

namespace {

    using namespace YK;

    void buildTestCircularGeometry(std::vector<SConeProjGeomVec>& geometry,
        const std::vector<float>& angles, int nu, int nv, float du, float dv,
        float sid, float idd, float3 detector_offset,
        float3 detector_tilt_degrees,
        float3 source_offset = make_float3(0.f, 0.f, 0.f))
    {
        SCircularTrajectorySpec trajectory{};
        trajectory.angles_rad = angles;
        trajectory.sid_mm = sid;
        trajectory.sdd_mm = sid + idd;
        trajectory.source_offset_mm = source_offset;
        SFlatDetectorSpec detector{};
        detector.channels = nu; detector.rows = nv;
        detector.channel_size_mm = du; detector.row_size_mm = dv;
        detector.pose.offset_unv_mm = detector_offset;
        constexpr float kDegToRad = 3.14159265358979323846f / 180.f;
        detector.pose.tilt_u_rad = detector_tilt_degrees.x * kDegToRad;
        detector.pose.tilt_n_rad = detector_tilt_degrees.y * kDegToRad;
        detector.pose.tilt_v_rad = detector_tilt_degrees.z * kDegToRad;
        if (!buildProjectionGeometry(trajectory, detector, geometry)) geometry.clear();
    }

    void buildTestCircularGeometry(std::vector<SConeProjGeomVec>& geometry,
        const std::vector<float>& angles, int view_count, int nu, int nv,
        float du, float dv, float sid, float idd, float3 detector_offset,
        float3 detector_tilt_degrees,
        float3 source_offset = make_float3(0.f, 0.f, 0.f))
    {
        std::vector<float> selected(angles.begin(), angles.begin() +
            std::min<size_t>(angles.size(), static_cast<size_t>(view_count)));
        buildTestCircularGeometry(geometry, selected, nu, nv, du, dv, sid, idd,
            detector_offset, detector_tilt_degrees, source_offset);
    }

    SReconstructionParams makeSmallParams(int views = 12)
    {
        SReconstructionParams p{};
        p.scan.Nu = 48; p.scan.Nv = 36;
        p.scan.NAng = views; p.scan.totalViews = views;
        p.volume.Nx = 32; p.volume.Ny = 32; p.volume.Nz = 24;
        p.scan.du_mm = 1.f; p.scan.dv_mm = 1.f;
        p.volume.voxelX_mm = 1.f; p.volume.voxelY_mm = 1.f; p.volume.voxelZ_mm = 1.f;
        p.scan.sid_mm = 100.f; p.scan.sdd_mm = 200.f;
        p.scan.range_rad = 2.f * CUDA_PI;
        p.scan.start_angle_rad = 0.f;
        p.scan.short_scan = false;
        p.scan.angles.resize(views);
        for (int i = 0; i < views; ++i)
            p.scan.angles[i] = 2.f * CUDA_PI * static_cast<float>(i) / views;
        return p;
    }

    SReconstructionParams makeFdkSmokeParams()
    {
        SReconstructionParams p = makeSmallParams();
        // FDK 分包回归只验证在线状态和 chunk 边界；保持最小已验证尺寸，避免
        // 测试本身把 FFT 工作区扩张成性能/显存压力测试。
        p.scan.Nu = 32; p.scan.Nv = 24;
        p.volume.Nx = 16; p.volume.Ny = 16; p.volume.Nz = 12;
        return p;
    }

    bool checkCuda(cudaError_t status, const char* what)
    {
        if (status == cudaSuccess) return true;
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        return false;
    }

    float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
    {
        float result = 0.f;
        for (size_t i = 0; i < a.size(); ++i)
            result = std::max(result, std::fabs(a[i] - b[i]));
        return result;
    }

    bool hasSignal(const std::vector<float>& data)
    {
        return std::any_of(data.begin(), data.end(), [](float value) {
            return std::isfinite(value) && std::fabs(value) > 1e-6f;
            });
    }

} // namespace

// 论文版 C-FDK 的端到端数值回归。投影由同一逐视图平板圆几何生成，测试
// 直接报告 ASTRA 官方 Shepp-Logan 模体的材料值误差，避免用相关系数
// 掩盖整体比例错误。
int main_curve_filtered_fdk_reconstruction()
{
    SReconstructionParams p = makeSmallParams(90);
    p.scan.Nu = 128; p.scan.Nv = 128;
    p.scan.du_mm = 0.8f; p.scan.dv_mm = 0.8f;
    p.scan.sid_mm = 100.f; p.scan.sdd_mm = 200.f;
    p.volume.Nx = 64; p.volume.Ny = 64; p.volume.Nz = 64;
    p.volume.voxelX_mm = 0.5f;
    p.volume.voxelY_mm = 0.5f;
    p.volume.voxelZ_mm = 0.5f;
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = 2.f * CUDA_PI * i / p.scan.NAng;

    std::vector<SConeProjGeomVec> views;
    detail::buildCircularViews(p, views);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) *
        p.scan.Nu * p.scan.Nv;
    std::vector<float> phantom = TestPhantom::makeAstraSheppLogan3D(
        p, 16.0f, true, 0.02f);
    const SVolGeom vg = SVolGeom::make_centered(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, p.volume.voxelX_mm, p.volume.voxelY_mm,
        p.volume.voxelZ_mm);
    const float3 origin = vg.origin();

    Mem::MemoryController memory;
    auto d_phantom = memory.allocateDevice3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, 0, false);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu,
        p.scan.Nv, p.scan.NAng, 0, false);
    auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, 0, false);
    auto h_reconstruction = memory.allocateCpu3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, false);
    auto h_phantom = memory.allocateCpu3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, false);
    std::copy(phantom.begin(), phantom.end(), h_phantom.data());
    memory.upload3D(d_phantom, h_phantom);

    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create C-FDK stream") &&
        checkCuda(cudaEventCreate(&begin), "create C-FDK start event") &&
        checkCuda(cudaEventCreate(&end), "create C-FDK stop event");
    GeometryContext geometry;
    ResourceContext resources;
    ok = ok && geometry.initialize(p, views);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && fp->prepare(geometry, resources) &&
        fp->apply(d_phantom.data(), p, d_projection.data(), resources);

    Fdk::CurveFilteredFdkPipeline cfdk;
    if (ok) {
        ok = cfdk.prepare(p, views, stream);
        if (ok) {
            cudaEventRecord(begin, stream);
            ok = cfdk.reconstruct(d_projection.data(),
                d_reconstruction.data(), true);
            cudaEventRecord(end, stream);
            ok = ok && cfdk.wait();
        }
    }
    float elapsed_ms = 0.f;
    if (ok) {
        cudaEventElapsedTime(&elapsed_ms, begin, end);
        memory.download3D(h_reconstruction, d_reconstruction);
    }

    double sum = 0.0, truth_sum = 0.0, abs_error = 0.0, sq_error = 0.0, truth_sq = 0.0;
    size_t roi_count = 0;
    if (ok) {
        for (int z = 0; z < p.volume.Nz; ++z) {
            const float wz = origin.z + z * vg.vox_z;
            for (int y = 0; y < p.volume.Ny; ++y) {
                const float wy = origin.y + y * vg.vox_y;
                for (int x = 0; x < p.volume.Nx; ++x) {
                    const size_t i = (static_cast<size_t>(z) * p.volume.Ny + y) * p.volume.Nx + x;
                    const float truth = phantom[i];
                    if (truth <= 0.f) continue;
                    const float value = h_reconstruction.cdata()[i];
                    ok = ok && std::isfinite(value);
                    sum += value;
                    truth_sum += truth;
                    abs_error += std::fabs(value - truth);
                    sq_error += static_cast<double>(value - truth) * (value - truth);
                    truth_sq += static_cast<double>(truth) * truth;
                    ++roi_count;
                }
            }
        }
    }
    const double mean = roi_count ? sum / roi_count : 0.0;
    const double mae = roi_count ? abs_error / roi_count : INFINITY;
    const double truth_mean = roi_count ? truth_sum / roi_count : 0.0;
    const double bias = mean - truth_mean;
    const double nrmse = truth_sq > 0.0 ? std::sqrt(sq_error / truth_sq) : INFINITY;
    ok = ok && roi_count > 0 && std::isfinite(nrmse) && nrmse < 1.0;
    YK_LOGI("[CurveFilteredFdk] ASTRA Shepp-Logan truth_mean={:.6e} ROI mean={:.6e} "
        "bias={:.6e} MAE={:.6e} NRMSE={:.6e} time={:.3f} ms",
        truth_mean, mean, bias, mae, nrmse, elapsed_ms);

    // 探测器倾斜已超出论文推导范围，必须显式拒绝，不能静默退回近似 FDK。
    std::vector<SConeProjGeomVec> tilted;
    buildTestCircularGeometry(tilted, p.scan.angles, p.scan.NAng,
        p.scan.Nu, p.scan.Nv, p.scan.du_mm, p.scan.dv_mm,
        p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
        make_float3(0.f, 0.f, 0.f), make_float3(1.f, 0.f, 0.f));
    Fdk::CurveFilteredFdkPipeline invalid;
    ok = ok && !invalid.prepare(p, tilted, stream);
    YK_LOGI("[CurveFilteredFdk] absolute-value reconstruction: {}",
        ok ? "PASS" : "FAIL");

    invalid.release();
    cfdk.release();
    fp->release();
    resources.release();
    if (end) cudaEventDestroy(end);
    if (begin) cudaEventDestroy(begin);
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// xFDK 的最小回归：先检查论文权重的三类区域和 180 度归一化，再用
// Joseph FP 生成小型 Shepp-Logan 投影，验证重排、滤波和扩展反投影能完成
// 一次端到端执行。该测试只要求有限值和非零信号，不把近似 FP 的绝对量级
// 当作 xFDK 数学正确性的唯一判据。
int main_xfdk_smoke()
{
    Fdk::detail::SXfdkGeometry wg{};
    wg.sid_mm = 100.f;
    wg.c_cot_gamma = 4.f;
    wg.reconstruction_radius_mm = 30.f;
    wg.delta_r_mm = wg.reconstruction_radius_mm * wg.reconstruction_radius_mm /
        (2.f * wg.sid_mm);
    const float full = Fdk::detail::xfdkCoverageHalfRange(wg, 5.f, 0.f);
    const float partial = Fdk::detail::xfdkCoverageHalfRange(wg, 45.f, 15.f);
    const float boundary_z = (wg.sid_mm * wg.sid_mm - 45.f * 45.f) /
        (wg.sid_mm * wg.c_cot_gamma);
    const float boundary = Fdk::detail::xfdkCoverageHalfRange(wg, 45.f, boundary_z);
    const float outside = Fdk::detail::xfdkCoverageHalfRange(wg, 10.f, 30.f);
    bool ok = full > 1.5f && full < 1.6f && partial > 0.f &&
        partial < 1.6f && std::fabs(boundary) < 1e-4f && outside < 0.f &&
        Fdk::detail::xfdkPartialWeight(0.f, 0.f, 0.f) == 2.f &&
        Fdk::detail::xfdkPartialWeight(CUDA_PI, 0.f, 0.f) == 0.f;
    const float delta = 0.7f;
    float weight_integral = 0.f;
    constexpr int samples = 200000;
    for (int i = 0; i < samples; ++i) {
        const float theta = -CUDA_PI + (2.f * CUDA_PI) *
            (static_cast<float>(i) + 0.5f) / samples;
        weight_integral += Fdk::detail::xfdkPartialWeight(theta, 0.f, delta);
    }
    weight_integral *= 2.f * CUDA_PI / samples;
    // w_PS 本身积分为 2pi；反投影公式外部的 1/2 将其归一到 pi。
    ok = ok && std::fabs(weight_integral - 2.f * CUDA_PI) < 2e-3f;

    SReconstructionParams p = makeSmallParams(48);
    p.scan.Nu = 48; p.scan.Nv = 32;
    p.volume.Nx = 24; p.volume.Ny = 24; p.volume.Nz = 20;
    p.scan.du_mm = 0.8f; p.scan.dv_mm = 0.8f;
    p.volume.voxelX_mm = 0.8f; p.volume.voxelY_mm = 0.8f; p.volume.voxelZ_mm = 0.8f;
    std::vector<SConeProjGeomVec> views;
    detail::buildCircularViews(p, views);
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    std::vector<float> phantom = TestPhantom::makeAstraSheppLogan3D(p, 7.f, true, 0.02f);
    Mem::MemoryController memory;
    auto d_phantom = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0, false);
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto h_phantom = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    std::copy(phantom.begin(), phantom.end(), h_phantom.data());
    memory.upload3D(d_phantom, h_phantom);
    cudaStream_t stream = nullptr;
    ok = ok && checkCuda(cudaStreamCreate(&stream), "create xFDK stream");
    GeometryContext geometry;
    ResourceContext resources;
    ok = ok && geometry.initialize(p, views);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && fp->prepare(geometry, resources) &&
        fp->apply(d_phantom.data(), p, d_projection.data(), resources) &&
        checkCuda(cudaStreamSynchronize(stream), "synchronize xFDK FP");
    Fdk::XfdkPipeline pipeline;
    ok = ok && pipeline.prepare(p, views, stream) &&
        pipeline.reconstruct(d_projection.data(), d_volume.data(), true) &&
        pipeline.wait();
    std::vector<float> output(volume_n);
    if (ok) {
        auto h_output = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
        memory.download3D(h_output, d_volume);
        std::copy(h_output.cdata(), h_output.cdata() + volume_n, output.begin());
        ok = std::all_of(output.begin(), output.end(), [](float value) {
            return std::isfinite(value);
        }) && hasSignal(output);
    }
    std::vector<SConeProjGeomVec> tilted;
    buildTestCircularGeometry(tilted, p.scan.angles, p.scan.NAng,
        p.scan.Nu, p.scan.Nv, p.scan.du_mm, p.scan.dv_mm,
        p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
        make_float3(0.f, 0.f, 0.f), make_float3(1.f, 0.f, 0.f));
    Fdk::XfdkPipeline invalid;
    ok = ok && !invalid.prepare(p, tilted, stream);
    YK_LOGI("[xFDK] weight integral={:.6f}, full/partial/outside=({:.5f},{:.5f},{:.5f}), {}",
        weight_integral, full, partial, outside, ok ? "PASS" : "FAIL");
    pipeline.release();
    invalid.release();
    fp->release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// 验证流式入口与离线入口的数值一致性。测试只把逐视图投影放入 pinned
// host 缓冲，再按不规则小批次 enqueue；设备端没有完整投影副本。
int main_xfdk_streaming_consistency()
{
    SReconstructionParams p = makeSmallParams(48);
    p.scan.Nu = 48; p.scan.Nv = 32;
    p.volume.Nx = 24; p.volume.Ny = 24; p.volume.Nz = 20;
    p.scan.du_mm = 0.8f; p.scan.dv_mm = 0.8f;
    p.volume.voxelX_mm = 0.8f; p.volume.voxelY_mm = 0.8f;
    p.volume.voxelZ_mm = 0.8f;
    std::vector<SConeProjGeomVec> views;
    detail::buildCircularViews(p, views);
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny *
        p.volume.Nz;
    const size_t projection_n = static_cast<size_t>(p.scan.Nu) * p.scan.Nv *
        p.scan.NAng;
    const size_t view_n = static_cast<size_t>(p.scan.Nu) * p.scan.Nv;

    Mem::MemoryController memory;
    auto h_phantom = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, false);
    const auto phantom = TestPhantom::makeAstraSheppLogan3D(p, 7.f, true,
        0.02f);
    std::copy(phantom.begin(), phantom.end(), h_phantom.data());
    auto d_phantom = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0, false);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng, 0, false);
    auto d_offline = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0, false);
    auto d_streaming = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0, false);
    memory.upload3D(d_phantom, h_phantom);

    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream),
        "create xFDK streaming consistency stream");
    GeometryContext geometry;
    ResourceContext resources;
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && geometry.initialize(p, views) && fp &&
        fp->prepare(geometry, resources) &&
        fp->apply(d_phantom.data(), p, d_projection.data(), resources) &&
        checkCuda(cudaStreamSynchronize(stream),
            "synchronize xFDK streaming consistency FP");

    Fdk::XfdkPipeline offline;
    Fdk::XfdkPipeline streaming;
    ok = ok && offline.prepareBatched(p, views, 11, stream) &&
        offline.reconstruct(d_projection.data(), d_offline.data(), true) &&
        offline.wait();

    auto h_projection = memory.allocatePinnedCpu3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng);
    if (ok) memory.download3D(h_projection, d_projection);
    ok = ok && streaming.prepareBatched(p, views, 11, stream) &&
        streaming.beginStreaming(d_streaming.data(), true);
    const int batches[] = {3, 7, 2, 13, 5, 11, 7};
    int offset = 0;
    for (const int count : batches) {
        if (!ok || offset + count > p.scan.NAng) break;
        ok = streaming.enqueueBatch({h_projection.data() +
            static_cast<size_t>(offset) * view_n, count});
        offset += count;
    }
    if (ok && offset < p.scan.NAng)
        ok = streaming.enqueueBatch({h_projection.data() +
            static_cast<size_t>(offset) * view_n, p.scan.NAng - offset});
    ok = ok && streaming.completeStreaming() && streaming.wait();

    std::vector<float> offline_values(volume_n), streaming_values(volume_n);
    if (ok) {
        auto h_offline = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, false);
        auto h_streaming = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, false);
        memory.download3D(h_offline, d_offline);
        memory.download3D(h_streaming, d_streaming);
        std::copy(h_offline.cdata(), h_offline.cdata() + volume_n,
            offline_values.begin());
        std::copy(h_streaming.cdata(), h_streaming.cdata() + volume_n,
            streaming_values.begin());
    }
    const float max_diff = maxAbsDiff(offline_values, streaming_values);
    ok = ok && offset == p.scan.NAng && std::isfinite(max_diff) &&
        max_diff < 2e-5f;
    YK_LOGI("[xFDK streaming] theta chunk={} source window={} prefix={} "
        "views={} max diff={} : {}", streaming.batchLayout().theta_chunk_views,
        streaming.batchLayout().source_window_views,
        streaming.batchLayout().periodic_prefix_views, offset, max_diff,
        ok ? "PASS" : "FAIL");
    streaming.release();
    offline.release();
    if (fp) fp->release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    (void)projection_n;
    return ok ? 0 : 1;
}

// xFDK 扩展范围的端到端对照。三条路径使用同一体模和同一圆轨迹：
//   1. 小探测器普通 FDK；
//   2. 小探测器 xFDK；
//   3. 加高探测器普通 FDK，作为小探测器缺失数据问题的参考。
//
// ASTRA Shepp-Logan 沿 z 正方向平移，使真实材料跨过普通 FDK 的完整覆盖
// 边界并进入 xFDK 的扩展区。评价只在真实材料体素内完成，既报告相对参考
// 误差，也保留 mm^-1 绝对值，避免相关系数掩盖整体幅值偏差。
int main_xfdk_large_cone_comparison()
{
    namespace fs = std::filesystem;
    SReconstructionParams small = makeSmallParams(360);
    // 小探测器完整锥角约 22.5 度（半锥角约 11.25 度），约为论文
    // Varian OBI 11 度完整锥角的两倍，用于观察更强缺失数据条件。
    small.scan.Nu = 256; small.scan.Nv = 192;
    small.scan.du_mm = 1.25f; small.scan.dv_mm = 1.25f;
    small.scan.sid_mm = 300.f; small.scan.sdd_mm = 600.f;
    small.volume.Nx = 160; small.volume.Ny = 160; small.volume.Nz = 160;
    small.volume.voxelX_mm = 1.f;
    small.volume.voxelY_mm = 1.f;
    small.volume.voxelZ_mm = 1.f;
    small.scan.angles.resize(small.scan.NAng);
    for (int i = 0; i < small.scan.NAng; ++i)
        small.scan.angles[i] = 2.f * CUDA_PI * i / small.scan.NAng;

    SReconstructionParams large = small;
    // 参考探测器只增加纵向像素数，SID、SDD、像素尺寸和横向扇角不变。
    large.scan.Nv = 320;
    std::vector<SConeProjGeomVec> small_views, large_views;
    detail::buildCircularViews(small, small_views);
    detail::buildCircularViews(large, large_views);

    const size_t volume_count = static_cast<size_t>(small.volume.Nx) *
        small.volume.Ny * small.volume.Nz;
    const size_t small_projection_count = static_cast<size_t>(small.scan.NAng) *
        small.scan.Nu * small.scan.Nv;
    const size_t large_projection_count = static_cast<size_t>(large.scan.NAng) *
        large.scan.Nu * large.scan.Nv;
    const SVolGeom vg = SVolGeom::make_centered(small.volume.Nx,
        small.volume.Ny, small.volume.Nz, small.volume.voxelX_mm,
        small.volume.voxelY_mm, small.volume.voxelZ_mm);
    const float3 origin = vg.origin();

    std::vector<float> centered = TestPhantom::makeAstraSheppLogan3D(
        small, 70.f, true, 0.02f);
    std::vector<float> truth(volume_count, 0.f);
    constexpr int z_shift_voxels = 48;
    for (int z = z_shift_voxels; z < small.volume.Nz; ++z) {
        const size_t destination = static_cast<size_t>(z) * small.volume.Ny *
            small.volume.Nx;
        const size_t source = static_cast<size_t>(z - z_shift_voxels) *
            small.volume.Ny * small.volume.Nx;
        std::copy_n(centered.data() + source,
            static_cast<size_t>(small.volume.Nx) * small.volume.Ny,
            truth.data() + destination);
    }

    Mem::MemoryController memory;
    auto h_truth = memory.allocateCpu3D<float>(small.volume.Nx,
        small.volume.Ny, small.volume.Nz, false);
    std::copy(truth.begin(), truth.end(), h_truth.data());
    auto d_truth = memory.allocateDevice3D<float>(small.volume.Nx,
        small.volume.Ny, small.volume.Nz, 0, false);
    auto d_small_projection = memory.allocateDevice3D<float>(small.scan.Nu,
        small.scan.Nv, small.scan.NAng, 0, false);
    auto d_large_projection = memory.allocateDevice3D<float>(large.scan.Nu,
        large.scan.Nv, large.scan.NAng, 0, false);
    auto d_fdk = memory.allocateDevice3D<float>(small.volume.Nx,
        small.volume.Ny, small.volume.Nz, 0, false);
    auto d_xfdk = memory.allocateDevice3D<float>(small.volume.Nx,
        small.volume.Ny, small.volume.Nz, 0, false);
    auto d_reference = memory.allocateDevice3D<float>(small.volume.Nx,
        small.volume.Ny, small.volume.Nz, 0, false);
    memory.upload3D(d_truth, h_truth);

    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream),
        "create xFDK extended-range stream");
    ResourceContext resources;
    resources.attach(stream, 0);
    auto run_fp = [&](const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& views, float* projection) {
        GeometryContext geometry;
        auto fp = makeForwardOperator(ETask::FP_Joseph);
        const bool result = geometry.initialize(params, views) &&
            fp && fp->prepare(geometry, resources) &&
            fp->apply(d_truth.data(), params, projection, resources) &&
            checkCuda(cudaStreamSynchronize(stream),
                "synchronize xFDK comparison FP");
        if (fp) fp->release();
        return result;
    };
    ok = ok && run_fp(small, small_views, d_small_projection.data());
    std::vector<float> host_small_projection(small_projection_count);
    if (ok) ok = checkCuda(cudaMemcpy(host_small_projection.data(),
        d_small_projection.data(), small_projection_count * sizeof(float),
        cudaMemcpyDeviceToHost), "download small-detector projection");

    auto run_fdk = [&](const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& views,
        const std::vector<float>& projection, float* output,
        double& elapsed_ms) {
        FdkPipeline pipeline;
        bool result = pipeline.prepareWithGeometry(params, views, 32, stream);
        const auto begin = std::chrono::steady_clock::now();
        if (result) result = pipeline.processBatchSync({projection.data(),
            nullptr, nullptr, params.scan.NAng}, output, true) &&
            pipeline.complete() && checkCuda(cudaStreamSynchronize(stream),
                "synchronize xFDK comparison FDK");
        elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        pipeline.release();
        return result;
    };

    double fdk_ms = 0.0, xfdk_ms = 0.0, reference_ms = 0.0;
    if (ok) ok = run_fdk(small, small_views, host_small_projection,
        d_fdk.data(), fdk_ms);

    Fdk::XfdkPipeline xfdk;
    if (ok) ok = xfdk.prepare(small, small_views, stream);
    Fdk::detail::SXfdkGeometry xfdk_geometry{};
    if (ok) xfdk_geometry = xfdk.derivedGeometry();
    const auto xfdk_begin = std::chrono::steady_clock::now();
    if (ok) ok = xfdk.reconstruct(d_small_projection.data(), d_xfdk.data(),
        true) && xfdk.wait();
    xfdk_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - xfdk_begin).count();

    ok = ok && run_fp(large, large_views, d_large_projection.data());
    std::vector<float> host_large_projection(large_projection_count);
    if (ok) ok = checkCuda(cudaMemcpy(host_large_projection.data(),
        d_large_projection.data(), large_projection_count * sizeof(float),
        cudaMemcpyDeviceToHost), "download large-detector projection");
    if (ok) ok = run_fdk(large, large_views, host_large_projection,
        d_reference.data(), reference_ms);

    std::vector<float> fdk_values(volume_count), xfdk_values(volume_count),
        reference_values(volume_count);
    auto download = [&](const auto& device, std::vector<float>& values) {
        auto host = memory.allocateCpu3D<float>(small.volume.Nx,
            small.volume.Ny, small.volume.Nz, false);
        memory.download3D(host, device);
        std::copy(host.cdata(), host.cdata() + volume_count, values.begin());
    };
    if (ok) {
        download(d_fdk, fdk_values);
        download(d_xfdk, xfdk_values);
        download(d_reference, reference_values);
    }

    enum class Region { Center, FdkEdge, XfdkExtension };
    struct Metrics {
        size_t count = 0;
        double truth_sum = 0.0;
        double value_sum = 0.0;
        double absolute_error = 0.0;
        double squared_error = 0.0;
        double reference_absolute_error = 0.0;
    };
    struct RegionMetrics {
        Metrics fdk, xfdk, reference;
    };
    RegionMetrics center_metrics, edge_metrics, extension_metrics;
    const float sid = xfdk_geometry.sid_mm;
    const float c = xfdk_geometry.c_cot_gamma;
    const float reconstruction_radius = xfdk_geometry.reconstruction_radius_mm;
    auto accumulate = [](Metrics& m, float value, float expected,
        float reference) {
        const double error = static_cast<double>(value) - expected;
        ++m.count;
        m.truth_sum += expected;
        m.value_sum += value;
        m.absolute_error += std::fabs(error);
        m.squared_error += error * error;
        m.reference_absolute_error += std::fabs(
            static_cast<double>(value) - reference);
    };
    auto add_region = [&](RegionMetrics& metrics, size_t index) {
        accumulate(metrics.fdk, fdk_values[index], truth[index],
            reference_values[index]);
        accumulate(metrics.xfdk, xfdk_values[index], truth[index],
            reference_values[index]);
        accumulate(metrics.reference, reference_values[index], truth[index],
            reference_values[index]);
    };
    if (ok) {
        for (int z = 0; z < small.volume.Nz; ++z) {
            const float wz = origin.z + z * vg.vox_z;
            for (int y = 0; y < small.volume.Ny; ++y) {
                const float wy = origin.y + y * vg.vox_y;
                for (int x = 0; x < small.volume.Nx; ++x) {
                    const float wx = origin.x + x * vg.vox_x;
                    const size_t index = (static_cast<size_t>(z) *
                        small.volume.Ny + y) * small.volume.Nx + x;
                    if (truth[index] <= 5e-4f) continue;
                    const float r = std::hypot(wx, wy);
                    if (r > reconstruction_radius) continue;
                    const float fdk_limit = (sid - r) / c;
                    const float xfdk_limit = (sid * sid - r * r) /
                        (sid * c);
                    const float az = std::fabs(wz);
                    if (az <= 0.5f * fdk_limit)
                        add_region(center_metrics, index);
                    else if (az <= fdk_limit)
                        add_region(edge_metrics, index);
                    else if (az <= xfdk_limit)
                        add_region(extension_metrics, index);
                }
            }
        }
    }

    const fs::path output_dir = fs::path(YKCBCT_TEST_SOURCE_DIR) / "output" /
        "xfdk" / "large_cone";
    fs::create_directories(output_dir);
    const auto write_raw = [&](const char* name,
        const std::vector<float>& values) {
        std::ofstream output(output_dir / name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        return output.good();
    };
    if (ok) ok = write_raw("truth_f32.raw", truth) &&
        write_raw("fdk_f32.raw", fdk_values) &&
        write_raw("xfdk_f32.raw", xfdk_values) &&
        write_raw("large_detector_fdk_f32.raw", reference_values);

    const int center_x = small.volume.Nx / 2;
    const int center_y = small.volume.Ny / 2;
    const int off_axis_y = std::clamp(static_cast<int>(std::lround(
        (50.f - origin.y) / vg.vox_y)), 0, small.volume.Ny - 1);
    const auto make_xz = [&](const std::vector<float>& volume, int y) {
        std::vector<float> slice(static_cast<size_t>(small.volume.Nx) *
            small.volume.Nz);
        for (int z = 0; z < small.volume.Nz; ++z)
            for (int x = 0; x < small.volume.Nx; ++x)
                slice[static_cast<size_t>(z) * small.volume.Nx + x] = volume[
                    (static_cast<size_t>(z) * small.volume.Ny + y) *
                    small.volume.Nx + x];
        return slice;
    };
    const auto make_yz = [&](const std::vector<float>& volume, int x) {
        std::vector<float> slice(static_cast<size_t>(small.volume.Ny) *
            small.volume.Nz);
        for (int z = 0; z < small.volume.Nz; ++z)
            for (int y = 0; y < small.volume.Ny; ++y)
                slice[static_cast<size_t>(z) * small.volume.Ny + y] = volume[
                    (static_cast<size_t>(z) * small.volume.Ny + y) *
                    small.volume.Nx + x];
        return slice;
    };
    std::vector<std::vector<float>> slices;
    const std::vector<const std::vector<float>*> volumes = {
        &truth, &fdk_values, &xfdk_values, &reference_values
    };
    for (const auto* volume : volumes) slices.push_back(make_xz(*volume, center_y));
    for (const auto* volume : volumes) slices.push_back(make_yz(*volume, center_x));
    for (const auto* volume : volumes) slices.push_back(make_xz(*volume, off_axis_y));
    std::vector<TestImage::GrayPanel> panels;
    for (const auto& slice : slices) panels.push_back({&slice,
        small.volume.Nx, small.volume.Nz, 1, 0, 1.f, 0.f, 0.025f});
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "coronal_sagittal_material_window.bmp", panels, 4, 8, 2) && ok;
    panels.clear();
    for (const auto& slice : slices) panels.push_back({&slice,
        small.volume.Nx, small.volume.Nz, 1, 0, 1.f, 0.015f, 0.022f});
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "coronal_sagittal_water_window.bmp", panels, 4, 8, 2) && ok;

    // z=55 mm 穿过本测试刻意布置的扩展区。四幅横断面沿用完全相同的
    // 物理窗，便于直接观察外围材料是否因显示自动归一化而“被补回来”。
    const int extension_z = std::clamp(static_cast<int>(std::lround(
        (55.f - origin.z) / vg.vox_z)), 0, small.volume.Nz - 1);
    const std::vector<TestImage::GrayPanel> axial_panels = {
        {&truth,small.volume.Nx,small.volume.Ny,small.volume.Nz,extension_z,
            1.f,0.f,0.025f},
        {&fdk_values,small.volume.Nx,small.volume.Ny,small.volume.Nz,extension_z,
            1.f,0.f,0.025f},
        {&xfdk_values,small.volume.Nx,small.volume.Ny,small.volume.Nz,extension_z,
            1.f,0.f,0.025f},
        {&reference_values,small.volume.Nx,small.volume.Ny,small.volume.Nz,
            extension_z,1.f,0.f,0.025f}
    };
    ok = TestImage::writeGrayMontageBmp(output_dir / "extended_axial.bmp",
        axial_panels, 4, 8, 2) && ok;

    const auto profile = [&](const std::vector<float>& volume, int z,
        bool peripheral) {
        double sum = 0.0; size_t count = 0;
        for (int y = 0; y < small.volume.Ny; ++y) {
            const float wy = origin.y + y * vg.vox_y;
            for (int x = 0; x < small.volume.Nx; ++x) {
                const float wx = origin.x + x * vg.vox_x;
                const float r = std::hypot(wx, wy);
                if (peripheral && (r < 45.f || r > reconstruction_radius))
                    continue;
                const size_t index = (static_cast<size_t>(z) * small.volume.Ny +
                    y) * small.volume.Nx + x;
                if (truth[index] <= 5e-4f) continue;
                sum += volume[index]; ++count;
            }
        }
        return count ? static_cast<float>(sum / count) : 0.f;
    };
    std::ofstream csv(output_dir / "z_profile.csv");
    csv << "z_mm,truth,fdk,xfdk,large_detector_fdk,"
        "peripheral_truth,peripheral_fdk,peripheral_xfdk,"
        "peripheral_large_detector_fdk\n";
    for (int z = 0; z < small.volume.Nz; ++z) {
        csv << origin.z + z * vg.vox_z << ',' << profile(truth, z, false) << ','
            << profile(fdk_values, z, false) << ','
            << profile(xfdk_values, z, false) << ','
            << profile(reference_values, z, false) << ','
            << profile(truth, z, true) << ','
            << profile(fdk_values, z, true) << ','
            << profile(xfdk_values, z, true) << ','
            << profile(reference_values, z, true) << '\n';
    }
    ok = csv.good() && ok;

    const auto mean = [](const Metrics& m, double sum) {
        return m.count ? sum / m.count : 0.0;
    };
    const auto write_metrics_row = [&](std::ofstream& output,
        const char* region, const char* method, const Metrics& m) {
        output << region << ',' << method << ',' << m.count << ','
            << mean(m, m.truth_sum) << ',' << mean(m, m.value_sum) << ','
            << (m.truth_sum != 0.0 ? m.value_sum / m.truth_sum : 0.0) << ','
            << mean(m, m.absolute_error) << ','
            << (m.count ? std::sqrt(m.squared_error / m.count) : 0.0) << ','
            << mean(m, m.reference_absolute_error) << '\n';
    };
    std::ofstream metrics_csv(output_dir / "region_metrics.csv");
    metrics_csv << "region,method,material_voxels,truth_mean,value_mean,"
        "truth_ratio,mae,rmse,mae_vs_large_detector_fdk\n";
    const auto write_region = [&](const char* name,
        const RegionMetrics& metrics) {
        write_metrics_row(metrics_csv, name, "fdk", metrics.fdk);
        write_metrics_row(metrics_csv, name, "xfdk", metrics.xfdk);
        write_metrics_row(metrics_csv, name, "large_detector_fdk",
            metrics.reference);
    };
    write_region("center", center_metrics);
    write_region("fdk_edge", edge_metrics);
    write_region("xfdk_only_extension", extension_metrics);
    metrics_csv.close();
    ok = metrics_csv.good() && ok;

    const auto log_region = [&](const char* name, const RegionMetrics& metrics) {
        YK_LOGI("[xFDK compare] {} material voxels={}, truth mean={:.6e}",
            name, metrics.fdk.count,
            mean(metrics.fdk, metrics.fdk.truth_sum));
        const auto log_method = [&](const char* method, const Metrics& m) {
            YK_LOGI("[xFDK compare]   {} mean={:.6e}, truth ratio={:.6f}, "
                "MAE={:.6e}, RMSE={:.6e}, MAE vs large-FDK={:.6e}",
                method, mean(m, m.value_sum),
                m.truth_sum != 0.0 ? m.value_sum / m.truth_sum : 0.0,
                mean(m, m.absolute_error),
                m.count ? std::sqrt(m.squared_error / m.count) : 0.0,
                mean(m, m.reference_absolute_error));
        };
        log_method("FDK", metrics.fdk);
        log_method("xFDK", metrics.xfdk);
        log_method("large-detector FDK", metrics.reference);
    };
    log_region("center", center_metrics);
    log_region("FDK edge", edge_metrics);
    log_region("xFDK-only extension", extension_metrics);
    YK_LOGI("[xFDK compare] coverage Rm={:.3f} mm, at r=50 mm "
        "FDK/xFDK z-limit={:.3f}/{:.3f} mm",
        reconstruction_radius, (sid - 50.f) / c,
        (sid * sid - 2500.f) / (sid * c));
    YK_LOGI("[xFDK compare] time FDK/xFDK/large-FDK={:.1f}/{:.1f}/{:.1f} ms",
        fdk_ms, xfdk_ms, reference_ms);

    std::ofstream metadata(output_dir / "metadata.json");
    metadata << "{\n"
        << "  \"data_type\": \"float32 little-endian\",\n"
        << "  \"layout\": \"[z][y][x], x fastest\",\n"
        << "  \"volume_xyz\": [" << small.volume.Nx << ", "
        << small.volume.Ny << ", " << small.volume.Nz << "],\n"
        << "  \"voxel_mm_xyz\": [1, 1, 1],\n"
        << "  \"phantom\": \"ASTRA modified 3-D Shepp-Logan\",\n"
        << "  \"phantom_z_shift_mm\": " << z_shift_voxels * vg.vox_z << ",\n"
        << "  \"small_detector_uv\": [" << small.scan.Nu << ", "
        << small.scan.Nv << "],\n"
        << "  \"large_detector_uv\": [" << large.scan.Nu << ", "
        << large.scan.Nv << "],\n"
        << "  \"detector_pixel_mm_uv\": [" << small.scan.du_mm << ", "
        << small.scan.dv_mm << "],\n"
        << "  \"sid_sdd_mm\": [" << sid << ", " << small.scan.sdd_mm << "],\n"
        << "  \"small_detector_half_cone_deg\": "
        << std::atan(0.5f * (small.scan.Nv - 1) * small.scan.dv_mm /
            small.scan.sdd_mm) * 180.f / CUDA_PI << ",\n"
        << "  \"small_detector_full_cone_deg\": "
        << 2.f * std::atan(0.5f * (small.scan.Nv - 1) * small.scan.dv_mm /
            small.scan.sdd_mm) * 180.f / CUDA_PI << ",\n"
        << "  \"views\": " << small.scan.NAng << ",\n"
        << "  \"xfdk_reconstruction_radius_mm\": " << reconstruction_radius << ",\n"
        << "  \"time_ms\": {\"fdk\": " << fdk_ms << ", \"xfdk\": "
        << xfdk_ms << ", \"large_detector_fdk\": " << reference_ms << "},\n"
        << "  \"montage_columns\": [\"truth\", \"FDK\", \"xFDK\", "
        "\"large-detector FDK\"],\n"
        << "  \"montage_rows\": [\"center coronal\", \"center sagittal\", "
        "\"coronal at y=50 mm\"]\n"
        << "}\n";
    metadata.close();
    ok = metadata.good() && ok;

    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        });
    };
    ok = ok && finite(fdk_values) && finite(xfdk_values) &&
        finite(reference_values) && center_metrics.fdk.count > 100 &&
        edge_metrics.fdk.count > 100 && extension_metrics.fdk.count > 100 &&
        hasSignal(xfdk_values) &&
        extension_metrics.xfdk.absolute_error <
            extension_metrics.fdk.absolute_error &&
        extension_metrics.xfdk.reference_absolute_error <
            extension_metrics.fdk.reference_absolute_error;
    xfdk.release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    YK_LOGI("[xFDK compare] artifacts: {} ({})", output_dir.string(),
        ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}

// 论文式 (25)、(27) 使用 |c| 定义上下对称的渐变曲线。式 (35) 是这些
// 曲线的反解，因此必须验证三个分段的往返、边界连续性和 c(-z)=-c(z)。
// 该测试独立于重建图像，可直接发现负 z 分母误用有符号 z 的问题。
int main_curve_filtered_fdk_piecewise_mapping()
{
    Fdk::detail::SCurveFilteredFdkGeometry g{};
    g.sid_mm = 400.f;
    g.sdd_mm = 800.f;
    const float am = 99.609375f;
    g.bm_mm = 99.609375f;
    const float r2 = g.sid_mm * g.sid_mm;
    const float q_edge = std::sqrt(r2 - am * am);
    g.c0_mm = g.bm_mm * (2.f * (r2 - am * am) -
        g.sid_mm * q_edge) / r2;
    g.s0_mm = g.bm_mm * (r2 - am * am) / r2;

    bool ok = true;
    float max_roundtrip_error = 0.f;
    float max_symmetry_error = 0.f;
    float max_boundary_jump = 0.f;
    float max_ray_error = 0.f;
    int branch_hits[3] = {0, 0, 0};
    const float t_values[] = {0.f, 35.f, 80.f};
    const float v_values[] = {-40.f, 0.f, 45.f};
    const float c_values[] = {
        0.5f * g.c0_mm,
        g.c0_mm,
        0.5f * (g.c0_mm + g.s0_mm),
        g.s0_mm,
        0.5f * (g.s0_mm + g.bm_mm),
        g.bm_mm
    };

    const auto forward_z = [&](float t, float v, float c) {
        const float q = std::sqrt(r2 - t * t);
        const float abs_c = std::fabs(c);
        float sc_prime = 0.f;
        if (abs_c <= g.c0_mm) {
            sc_prime = 2.f * q - g.sid_mm;
        } else if (abs_c <= g.s0_mm) {
            sc_prime = 2.f * q - g.sid_mm +
                (abs_c - g.c0_mm) / (g.s0_mm - g.c0_mm) *
                (g.sid_mm - q);
        } else {
            sc_prime = q + (abs_c - g.s0_mm) /
                (g.bm_mm - g.s0_mm) * t * t / q;
        }
        return c * (q + v) / sc_prime;
    };

    for (float t : t_values) {
        // 验证论文坐标转到项目 builder 后的视角符号。项目虚拟平板的
        // U 轴为 e_beta，beta=theta+asin(t/R) 时，源点和 a 采样点到
        // e_theta 的投影都必须为 t，即二者定义同一条平行射线。
        const float theta = 0.37f;
        const float delta = std::asin(t / g.sid_mm);
        const float beta = theta + delta;
        const float q = std::sqrt(r2 - t * t);
        const float a = t * g.sid_mm / q;
        const float source_projection = g.sid_mm * std::sin(beta - theta);
        const float detector_projection = a * std::cos(beta - theta);
        max_ray_error = std::max(max_ray_error,
            std::max(std::fabs(source_projection - t),
                std::fabs(detector_projection - t)));
        for (float v : v_values) {
            for (float magnitude : c_values) {
                for (float sign : {-1.f, 1.f}) {
                    const float expected_c = sign * magnitude;
                    const float z = forward_z(t, v, expected_c);
                    float actual_c = 0.f;
                    int branch = -1;
                    ok = Fdk::detail::mapCurveFilteredFdkBackprojectionC(
                        g, t, v, z, actual_c, &branch) && ok;
                    if (branch >= 0 && branch < 3) ++branch_hits[branch];
                    max_roundtrip_error = std::max(max_roundtrip_error,
                        std::fabs(actual_c - expected_c));

                    float mirrored_c = 0.f;
                    ok = Fdk::detail::mapCurveFilteredFdkBackprojectionC(
                        g, t, v, -z, mirrored_c) && ok;
                    max_symmetry_error = std::max(max_symmetry_error,
                        std::fabs(actual_c + mirrored_c));
                }
            }

            // 在 c0 和 s0 两侧取极小扰动，检查分段反解没有数值跳变。
            for (float boundary : {g.c0_mm, g.s0_mm}) {
                const float epsilon = 1e-4f * boundary;
                float left = 0.f, right = 0.f;
                const float z_left = forward_z(t, v, boundary - epsilon);
                const float z_right = forward_z(t, v, boundary + epsilon);
                ok = Fdk::detail::mapCurveFilteredFdkBackprojectionC(
                    g, t, v, z_left, left) && ok;
                ok = Fdk::detail::mapCurveFilteredFdkBackprojectionC(
                    g, t, v, z_right, right) && ok;
                max_boundary_jump = std::max(max_boundary_jump,
                    std::fabs((right - left) - 2.f * epsilon));
            }
        }
    }

    ok = ok && branch_hits[0] > 0 && branch_hits[1] > 0 &&
        branch_hits[2] > 0 && max_roundtrip_error < 5e-4f &&
        max_symmetry_error < 5e-5f && max_boundary_jump < 5e-4f &&
        max_ray_error < 5e-5f;
    YK_LOGI("[CurveFilteredFdk] piecewise mapping c0={:.6f} s0={:.6f} "
        "bm={:.6f}; hits={}/{}/{} roundtrip={:.3e} symmetry={:.3e} "
        "boundary={:.3e} ray={:.3e}: {}", g.c0_mm, g.s0_mm, g.bm_mm,
        branch_hits[0], branch_hits[1], branch_hits[2],
        max_roundtrip_error, max_symmetry_error, max_boundary_jump, max_ray_error,
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 大锥角对比：使用论文实验接近的 ±14 度锥角，比较普通 FDK 与 C-FDK
// 在中心层和离中心层的材料绝对值。该测试不做经验比例拟合，结果用于观察
// C-FDK 对圆轨迹锥角伪影和 z 方向强度下降的改善幅度。
int main_curve_filtered_fdk_large_cone_comparison()
{
    SReconstructionParams p = makeSmallParams(360);
    // 论文第 3 节原始仿真参数：400 mm SID、800 mm SDD、400x400 mm
    // 平板、256^3 体积和 0.78 mm 等方体素。
    p.scan.Nu = 256; p.scan.Nv = 256;
    p.scan.du_mm = 400.f / 256.f; p.scan.dv_mm = 400.f / 256.f;
    p.scan.sid_mm = 400.f; p.scan.sdd_mm = 800.f;
    p.volume.Nx = 256; p.volume.Ny = 256; p.volume.Nz = 256;
    p.volume.voxelX_mm = 0.78f; p.volume.voxelY_mm = 0.78f;
    p.volume.voxelZ_mm = 0.78f;
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = 2.f * CUDA_PI * i / p.scan.NAng;

    std::vector<SConeProjGeomVec> views;
    detail::buildCircularViews(p, views);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv;
    // 使用与普通 C-FDK 回归相同的 ASTRA 官方模体，覆盖椭球边缘和低对比结构。
    std::vector<float> phantom = TestPhantom::makeAstraSheppLogan3D(
        p, 0.5f * p.volume.Nx * p.volume.voxelX_mm, false, 1.0f);
    const SVolGeom vg = SVolGeom::make_centered(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, p.volume.voxelX_mm, p.volume.voxelY_mm, p.volume.voxelZ_mm);
    const float3 origin = vg.origin();

    Mem::MemoryController memory;
    auto h_phantom = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    std::copy(phantom.begin(), phantom.end(), h_phantom.data());
    auto d_phantom = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0, false);
    auto d_fdk = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto d_cfdk = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto h_fdk = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    auto h_cfdk = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    memory.upload3D(d_phantom, h_phantom);

    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create large-cone stream");
    GeometryContext geometry;
    ResourceContext resources;
    ok = ok && geometry.initialize(p, views);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && fp->prepare(geometry, resources) &&
        fp->apply(d_phantom.data(), p, d_projection.data(), resources) &&
        checkCuda(cudaStreamSynchronize(stream), "large-cone FP synchronize");

    FdkPipeline fdk;
    Fdk::CurveFilteredFdkPipeline cfdk;
    double fdk_ms = 0.0, cfdk_ms = 0.0;
    if (ok) {
        ok = fdk.prepareWithGeometry(p, views, 32, stream);
        // processBatchSync 的 host 输入契约要求投影位于 host；下载一次后复用，
        // 避免为对比路径引入另一套 FP 实现。
        if (ok) {
            std::vector<float> host_projection(projection_count);
            ok = checkCuda(cudaMemcpy(host_projection.data(), d_projection.data(),
                projection_count * sizeof(float), cudaMemcpyDeviceToHost),
                "download large-cone projection");
            const auto begin = std::chrono::steady_clock::now();
            if (ok) ok = fdk.processBatchSync({host_projection.data(), nullptr,
                nullptr, p.scan.NAng}, d_fdk.data(), true) && fdk.complete();
            fdk_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        }
        if (ok) ok = cfdk.prepare(p, views, stream);
        const auto begin = std::chrono::steady_clock::now();
        if (ok) ok = cfdk.reconstruct(d_projection.data(), d_cfdk.data(), true) &&
            cfdk.wait();
        cfdk_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    }
    if (ok) {
        memory.download3D(h_fdk, d_fdk);
        memory.download3D(h_cfdk, d_cfdk);
        const std::filesystem::path output_dir =
            std::filesystem::path(YKCBCT_TEST_SOURCE_DIR) / "output" / "cfdk";
        std::filesystem::create_directories(output_dir);
        const auto write_raw = [&](const char* name, const float* data) {
            std::ofstream output(output_dir / name, std::ios::binary);
            output.write(reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(volume_count * sizeof(float)));
            return output.good();
        };
        ok = write_raw("paper_shepp_logan_phantom_256x256x256_f32.raw", phantom.data()) &&
            write_raw("paper_shepp_logan_fdk_256x256x256_f32.raw", h_fdk.cdata()) &&
            write_raw("paper_shepp_logan_cfdk_256x256x256_f32.raw", h_cfdk.cdata());
    }

    struct Metrics { double mean = 0.0; double truth_mean = 0.0; double bias = 0.0; double mae = 0.0; size_t n = 0; };
    const auto measure = [&](const auto& volume, float center_z) {
        Metrics m{};
        for (int z = 0; z < p.volume.Nz; ++z) {
            const float wz = origin.z + z * vg.vox_z;
            if (std::fabs(wz - center_z) > 1.5f) continue;
            for (int y = 0; y < p.volume.Ny; ++y) for (int x = 0; x < p.volume.Nx; ++x) {
                const size_t i = (static_cast<size_t>(z) * p.volume.Ny + y) * p.volume.Nx + x;
                const float expected = phantom[i];
                if (expected <= 0.f) continue;
                const float value = volume.cdata()[i];
                m.mean += value; m.truth_mean += expected;
                m.mae += std::fabs(value - expected); ++m.n;
            }
        }
        if (m.n) { m.mean /= m.n; m.truth_mean /= m.n; m.bias = m.mean - m.truth_mean; m.mae /= m.n; }
        return m;
    };
    const Metrics fdk_center = measure(h_fdk, 0.f);
    const Metrics cfdk_center = measure(h_cfdk, 0.f);
    const Metrics fdk_off = measure(h_fdk, 10.f);
    const Metrics cfdk_off = measure(h_cfdk, 10.f);
    ok = ok && fdk_center.n > 0 && cfdk_center.n > 0 &&
        fdk_off.n > 0 && cfdk_off.n > 0;
    YK_LOGI("[LargeCone] FDK center mean={:.6e} bias={:.6e} MAE={:.6e}; "
        "off-z mean={:.6e} bias={:.6e} MAE={:.6e}", fdk_center.mean,
        fdk_center.bias, fdk_center.mae, fdk_off.mean, fdk_off.bias, fdk_off.mae);
    YK_LOGI("[LargeCone] C-FDK center mean={:.6e} bias={:.6e} MAE={:.6e}; "
        "off-z mean={:.6e} bias={:.6e} MAE={:.6e}", cfdk_center.mean,
        cfdk_center.bias, cfdk_center.mae, cfdk_off.mean, cfdk_off.bias, cfdk_off.mae);
    YK_LOGI("[LargeCone] reconstruction time: FDK={:.3f} ms, C-FDK={:.3f} ms",
        fdk_ms, cfdk_ms);

    if (ok) {
        std::vector<float> fdk_values(h_fdk.cdata(), h_fdk.cdata() + volume_count);
        std::vector<float> cfdk_values(h_cfdk.cdata(), h_cfdk.cdata() + volume_count);
        std::vector<float> difference(volume_count, 0.f);
        for (size_t i = 0; i < volume_count; ++i)
            difference[i] = cfdk_values[i] - fdk_values[i];
        int center_slice = 0, off_slice = 0;
        float center_distance = INFINITY, off_distance = INFINITY;
        for (int z = 0; z < p.volume.Nz; ++z) {
            const float wz = origin.z + z * vg.vox_z;
            if (std::fabs(wz) < center_distance) {
                center_distance = std::fabs(wz);
                center_slice = z;
            }
            if (std::fabs(wz - 10.f) < off_distance) {
                off_distance = std::fabs(wz - 10.f);
                off_slice = z;
            }
        }
        const std::filesystem::path output_dir =
            std::filesystem::path(YKCBCT_TEST_SOURCE_DIR) / "output" / "cfdk";
        const std::vector<TestImage::GrayPanel> panels = {
            {&phantom, p.volume.Nx, p.volume.Ny, p.volume.Nz, center_slice, 1.f, 0.f, 2.05f},
            {&fdk_values, p.volume.Nx, p.volume.Ny, p.volume.Nz, center_slice, 1.f, 0.f, 2.05f},
            {&cfdk_values, p.volume.Nx, p.volume.Ny, p.volume.Nz, center_slice, 1.f, 0.f, 2.05f},
            {&difference, p.volume.Nx, p.volume.Ny, p.volume.Nz, center_slice, 1.f, -0.20f, 0.20f},
            {&phantom, p.volume.Nx, p.volume.Ny, p.volume.Nz, off_slice, 1.f, 0.f, 2.05f},
            {&fdk_values, p.volume.Nx, p.volume.Ny, p.volume.Nz, off_slice, 1.f, 0.f, 2.05f},
            {&cfdk_values, p.volume.Nx, p.volume.Ny, p.volume.Nz, off_slice, 1.f, 0.f, 2.05f},
            {&difference, p.volume.Nx, p.volume.Ny, p.volume.Nz, off_slice, 1.f, -0.20f, 0.20f},
        };
        ok = TestImage::writeGrayMontageBmp(
            output_dir / "large_cone_center_offz_fdk_cfdk.bmp", panels, 4, 8, 3) && ok;
        YK_LOGI("[LargeCone] image comparison written: {} (rows: z=0, z=10; "
            "columns: truth, FDK, C-FDK, C-FDK-FDK)",
            (output_dir / "large_cone_center_offz_fdk_cfdk.bmp").string());

        // 论文 Fig.6 使用固定 y=-25 mm 的 X-Z 冠状面。这里同时输出
        // 冠状面和 x=0 的 Y-Z 矢状面，并对 truth/FDK/C-FDK 使用完全相同的
        // 绝对窗宽；禁止分别归一化，否则会掩盖材料值比例和 z 向衰减。
        const int y_slice = std::clamp(static_cast<int>(std::lround(
            (-25.f - origin.y) / vg.vox_y)), 0, p.volume.Ny - 1);
        const int x_slice = std::clamp(static_cast<int>(std::lround(
            (0.f - origin.x) / vg.vox_x)), 0, p.volume.Nx - 1);
        const auto make_coronal = [&](const std::vector<float>& volume) {
            std::vector<float> slice(static_cast<size_t>(p.volume.Nx) * p.volume.Nz);
            for (int z = 0; z < p.volume.Nz; ++z)
                for (int x = 0; x < p.volume.Nx; ++x)
                    slice[static_cast<size_t>(z) * p.volume.Nx + x] =
                        volume[(static_cast<size_t>(z) * p.volume.Ny + y_slice) * p.volume.Nx + x];
            return slice;
        };
        const auto make_sagittal = [&](const std::vector<float>& volume) {
            std::vector<float> slice(static_cast<size_t>(p.volume.Ny) * p.volume.Nz);
            for (int z = 0; z < p.volume.Nz; ++z)
                for (int y = 0; y < p.volume.Ny; ++y)
                    slice[static_cast<size_t>(z) * p.volume.Ny + y] =
                        volume[(static_cast<size_t>(z) * p.volume.Ny + y) * p.volume.Nx + x_slice];
            return slice;
        };
        auto truth_xz = make_coronal(phantom);
        auto fdk_xz = make_coronal(fdk_values);
        auto cfdk_xz = make_coronal(cfdk_values);
        auto truth_yz = make_sagittal(phantom);
        auto fdk_yz = make_sagittal(fdk_values);
        auto cfdk_yz = make_sagittal(cfdk_values);
        std::vector<float> fdk_error_xz(truth_xz.size());
        std::vector<float> cfdk_error_xz(truth_xz.size());
        std::vector<float> fdk_error_yz(truth_yz.size());
        std::vector<float> cfdk_error_yz(truth_yz.size());
        for (size_t i = 0; i < truth_xz.size(); ++i) {
            fdk_error_xz[i] = fdk_xz[i] - truth_xz[i];
            cfdk_error_xz[i] = cfdk_xz[i] - truth_xz[i];
        }
        for (size_t i = 0; i < truth_yz.size(); ++i) {
            fdk_error_yz[i] = fdk_yz[i] - truth_yz[i];
            cfdk_error_yz[i] = cfdk_yz[i] - truth_yz[i];
        }

        const float absolute_lo = 0.f, absolute_hi = 2.05f;
        const float uniformity_lo = 0.85f, uniformity_hi = 1.10f;
        const float error_lo = -0.20f, error_hi = 0.20f;
        const std::vector<TestImage::GrayPanel> absolute_panels = {
            {&truth_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,absolute_lo,absolute_hi},
            {&fdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,absolute_lo,absolute_hi},
            {&cfdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,absolute_lo,absolute_hi},
            {&truth_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,absolute_lo,absolute_hi},
            {&fdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,absolute_lo,absolute_hi},
            {&cfdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,absolute_lo,absolute_hi}
        };
        ok = TestImage::writeGrayMontageBmp(output_dir /
            "large_cone_coronal_sagittal_absolute.bmp", absolute_panels, 3, 8, 2) && ok;
        const std::vector<TestImage::GrayPanel> uniformity_panels = {
            {&truth_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,uniformity_lo,uniformity_hi},
            {&fdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,uniformity_lo,uniformity_hi},
            {&cfdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,uniformity_lo,uniformity_hi},
            {&truth_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,uniformity_lo,uniformity_hi},
            {&fdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,uniformity_lo,uniformity_hi},
            {&cfdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,uniformity_lo,uniformity_hi}
        };
        ok = TestImage::writeGrayMontageBmp(output_dir /
            "large_cone_coronal_sagittal_uniformity.bmp", uniformity_panels, 3, 8, 2) && ok;
        const std::vector<TestImage::GrayPanel> error_panels = {
            {&fdk_error_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,error_lo,error_hi},
            {&cfdk_error_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,error_lo,error_hi},
            {&fdk_error_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,error_lo,error_hi},
            {&cfdk_error_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,error_lo,error_hi}
        };
        ok = TestImage::writeGrayMontageBmp(output_dir /
            "large_cone_coronal_sagittal_error.bmp", error_panels, 2, 8, 2) && ok;

        struct PlaneBandMetrics { double ratio = 0.0; double mae = 0.0; size_t n = 0; };
        const auto measure_plane_band = [&](const std::vector<float>& truth,
            const std::vector<float>& reconstruction, int width,
            float z_min_mm, float z_max_mm) {
            PlaneBandMetrics metrics{};
            for (int z = 0; z < p.volume.Nz; ++z) {
                const float wz = origin.z + z * vg.vox_z;
                if (wz < z_min_mm || wz >= z_max_mm) continue;
                for (int horizontal = 0; horizontal < width; ++horizontal) {
                    const size_t i = static_cast<size_t>(z) * width + horizontal;
                    if (truth[i] <= 0.f) continue;
                    metrics.ratio += reconstruction[i] / truth[i];
                    metrics.mae += std::fabs(reconstruction[i] - truth[i]);
                    ++metrics.n;
                }
            }
            if (metrics.n) {
                metrics.ratio /= metrics.n;
                metrics.mae /= metrics.n;
            }
            return metrics;
        };
        const auto log_plane = [&](const char* name, const std::vector<float>& truth,
            const std::vector<float>& fdk_slice, const std::vector<float>& cfdk_slice,
            int width) {
            const float bands[3][2] = {{-80.f, -65.f}, {-5.f, 5.f}, {65.f, 80.f}};
            PlaneBandMetrics fdk_band[3], cfdk_band[3];
            for (int i = 0; i < 3; ++i) {
                fdk_band[i] = measure_plane_band(truth, fdk_slice, width,
                    bands[i][0], bands[i][1]);
                cfdk_band[i] = measure_plane_band(truth, cfdk_slice, width,
                    bands[i][0], bands[i][1]);
            }
            YK_LOGI("[LargeCone] {} FDK top/center/bottom ratio="
                "{:.6f}/{:.6f}/{:.6f} MAE={:.6f}/{:.6f}/{:.6f}; "
                "C-FDK ratio={:.6f}/{:.6f}/{:.6f} MAE={:.6f}/{:.6f}/{:.6f}",
                name, fdk_band[0].ratio, fdk_band[1].ratio, fdk_band[2].ratio,
                fdk_band[0].mae, fdk_band[1].mae, fdk_band[2].mae,
                cfdk_band[0].ratio, cfdk_band[1].ratio, cfdk_band[2].ratio,
                cfdk_band[0].mae, cfdk_band[1].mae, cfdk_band[2].mae);
        };
        log_plane("coronal(y=-25mm)", truth_xz, fdk_xz, cfdk_xz, p.volume.Nx);
        log_plane("sagittal(x=0mm)", truth_yz, fdk_yz, cfdk_yz, p.volume.Ny);

        // 用稀疏体素/视角统计端到端图实际覆盖的式 (35) 分支。ASTRA 模体
        // 不一定延伸到 c0/s0 外侧，必须把“解析分段已测试”和“图像覆盖分段”
        // 区分开，避免根据一张图错误宣称三个过渡区都已验证。
        size_t branch_hits[3] = {0, 0, 0};
        size_t outside_hits = 0;
        const auto& derived = cfdk.derivedGeometry();
        for (int z = 0; z < p.volume.Nz; z += 4) {
            const float wz = origin.z + z * vg.vox_z;
            for (int y = 0; y < p.volume.Ny; y += 4) {
                const float wy = origin.y + y * vg.vox_y;
                for (int x = 0; x < p.volume.Nx; x += 4) {
                    const size_t voxel = (static_cast<size_t>(z) * p.volume.Ny + y) *
                        p.volume.Nx + x;
                    if (phantom[voxel] <= 0.f) continue;
                    const float wx = origin.x + x * vg.vox_x;
                    for (int view = 0; view < p.scan.NAng; view += 15) {
                        const float theta = derived.beta0_rad +
                            view * derived.signed_dtheta_rad;
                        const float sine = std::sin(theta), cosine = std::cos(theta);
                        const float t = wy * cosine - wx * sine;
                        const float v = -(wx * cosine + wy * sine);
                        float c = 0.f;
                        int branch = -1;
                        if (Fdk::detail::mapCurveFilteredFdkBackprojectionC(
                                derived, t, v, wz, c, &branch)) {
                            if (branch >= 0 && branch < 3) ++branch_hits[branch];
                        } else {
                            ++outside_hits;
                        }
                    }
                }
            }
        }
        YK_LOGI("[LargeCone] sampled Eq.(35) branch hits={}/{}/{}, outside={}",
            branch_hits[0], branch_hits[1], branch_hits[2], outside_hits);

        const auto profile_ratio = [&](const std::vector<float>& v,
            float z_min_mm, float z_max_mm) {
            double sum = 0.0; size_t n = 0;
            // 论文 Fig.7 是中心竖直剖面。取中心附近 3 列降低单像素离散噪声，
            // 并只统计真值约为 1.02 的主体材料，排除空气和外层高值边缘。
            for (int z = 0; z < p.volume.Nz; ++z) {
                const float wz = origin.z + z * vg.vox_z;
                if (wz < z_min_mm || wz > z_max_mm) continue;
                for (int x = p.volume.Nx / 2 - 1; x <= p.volume.Nx / 2 + 1; ++x) {
                    const size_t i = static_cast<size_t>(z) * p.volume.Nx + x;
                    const float expected = truth_xz[i];
                    if (expected < 0.95f || expected > 1.10f) continue;
                    sum += v[i] / expected; ++n;
                }
            }
            return n ? static_cast<float>(sum / n) : 0.f;
        };
        YK_LOGI("[LargeCone] paper center-line recon/truth FDK top/center/bottom={:.6f}/{:.6f}/{:.6f}; "
            "C-FDK={:.6f}/{:.6f}/{:.6f}",
            profile_ratio(fdk_xz, 45.f, 65.f),
            profile_ratio(fdk_xz, -5.f, 5.f),
            profile_ratio(fdk_xz, -65.f, -45.f),
            profile_ratio(cfdk_xz, 45.f, 65.f),
            profile_ratio(cfdk_xz, -5.f, 5.f),
            profile_ratio(cfdk_xz, -65.f, -45.f));

        // 中心线 CSV 保留物理值，不做显示归一化。坐标取论文的 y=-25 mm
        // 冠状面、x=0 附近三列平均，便于逐 z 检查材料真值与上下对称性。
        std::ofstream profile(output_dir / "large_cone_z_profile.csv");
        profile << "z_mm,truth,fdk,cfdk,fdk_over_truth,cfdk_over_truth\n";
        double fdk_symmetry = 0.0, cfdk_symmetry = 0.0;
        size_t symmetry_count = 0;
        for (int z = 0; z < p.volume.Nz; ++z) {
            double truth_sum = 0.0, fdk_sum = 0.0, cfdk_sum = 0.0;
            int count = 0;
            for (int x = p.volume.Nx / 2 - 1; x <= p.volume.Nx / 2 + 1; ++x) {
                const size_t i = static_cast<size_t>(z) * p.volume.Nx + x;
                truth_sum += truth_xz[i]; fdk_sum += fdk_xz[i]; cfdk_sum += cfdk_xz[i];
                ++count;
            }
            const double truth_value = truth_sum / count;
            const double fdk_value = fdk_sum / count;
            const double cfdk_value = cfdk_sum / count;
            profile << origin.z + z * vg.vox_z << ',' << truth_value << ','
                << fdk_value << ',' << cfdk_value << ','
                << (truth_value != 0.0 ? fdk_value / truth_value : 0.0) << ','
                << (truth_value != 0.0 ? cfdk_value / truth_value : 0.0) << '\n';
            const int mirror = p.volume.Nz - 1 - z;
            if (z < mirror) {
                for (int x = p.volume.Nx / 2 - 1; x <= p.volume.Nx / 2 + 1; ++x) {
                    const size_t i0 = static_cast<size_t>(z) * p.volume.Nx + x;
                    const size_t i1 = static_cast<size_t>(mirror) * p.volume.Nx + x;
                    if (truth_xz[i0] <= 0.f || truth_xz[i1] <= 0.f) continue;
                    fdk_symmetry += std::fabs(fdk_xz[i0] - fdk_xz[i1]);
                    cfdk_symmetry += std::fabs(cfdk_xz[i0] - cfdk_xz[i1]);
                    ++symmetry_count;
                }
            }
        }
        ok = profile.good() && ok;
        YK_LOGI("[LargeCone] coronal z symmetry MAE: FDK={:.6e}, C-FDK={:.6e}; "
            "images use shared absolute windows, y-index={}, x-index={}",
            symmetry_count ? fdk_symmetry / symmetry_count : 0.0,
            symmetry_count ? cfdk_symmetry / symmetry_count : 0.0,
            y_slice, x_slice);
        YK_LOGI("[LargeCone] coronal/sagittal outputs: {}, {}, {}",
            (output_dir / "large_cone_coronal_sagittal_absolute.bmp").string(),
            (output_dir / "large_cone_coronal_sagittal_uniformity.bmp").string(),
            (output_dir / "large_cone_coronal_sagittal_error.bmp").string());
    }
    YK_LOGI("[LargeCone] absolute comparison: {}", ok ? "PASS" : "FAIL");

    cfdk.release(); fdk.release(); fp->release(); resources.release();
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// 大水模 C-FDK 回归：使用比常规 large-water 更大的水柱和更高的轴向范围，
// 专门观察冠状面/矢状面的 z 向材料值。这里保留真实的水和壳体衰减系数，
// 不使用相关系数归一化来掩盖绝对量级误差。
int main_curve_filtered_fdk_large_water()
{
    SReconstructionParams p = makeSmallParams(720);
    // 物理探测器宽度约 800 mm；经过 SID/SDD=0.5 的虚拟平面缩放后，
    // 虚拟半宽约 200 mm，对应论文中约 26.6 度的真正大锥角。
    p.scan.Nu = 768; p.scan.Nv = 384;
    p.scan.du_mm = 1.04f; p.scan.dv_mm = 1.00f;
    p.scan.sid_mm = 400.f; p.scan.sdd_mm = 800.f;
    p.volume.Nx = 384; p.volume.Ny = 384; p.volume.Nz = 384;
    p.volume.voxelX_mm = 0.40f;
    p.volume.voxelY_mm = 0.40f;
    p.volume.voxelZ_mm = 0.40f;
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = 2.f * CUDA_PI * i / p.scan.NAng;

    const size_t volume_count = static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) *
        p.scan.Nu * p.scan.Nv;
    std::vector<float> truth(volume_count, 0.f);
    constexpr float water_radius_mm = 70.f;       // 140 mm 直径
    constexpr float water_half_height_mm = 68.f;  // 136 mm 水柱高度
    constexpr float water_mu = 0.020f;
    const SVolGeom vg = SVolGeom::make_centered(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, p.volume.voxelX_mm, p.volume.voxelY_mm,
        p.volume.voxelZ_mm);
    const float3 origin = vg.origin();
    for (int z = 0; z < p.volume.Nz; ++z) {
        const float wz = origin.z + z * vg.vox_z;
        for (int y = 0; y < p.volume.Ny; ++y) {
            const float wy = origin.y + y * vg.vox_y;
            for (int x = 0; x < p.volume.Nx; ++x) {
                const float wx = origin.x + x * vg.vox_x;
                const float radius2 = wx * wx + wy * wy;
                const float value =
                    radius2 <= water_radius_mm * water_radius_mm &&
                    std::fabs(wz) <= water_half_height_mm ? water_mu : 0.f;
                truth[(static_cast<size_t>(z) * p.volume.Ny + y) *
                    p.volume.Nx + x] = value;
            }
        }
    }

    std::vector<SConeProjGeomVec> views;
    detail::buildCircularViews(p, views);
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0, false);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
        p.scan.NAng, 0, false);
    auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, 0, false);
    auto h_reconstruction = memory.allocateCpu3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, false);
    bool ok = checkCuda(cudaMemcpy(d_truth.data(), truth.data(),
        volume_count * sizeof(float), cudaMemcpyHostToDevice),
        "upload large-water phantom");

    cudaStream_t stream = nullptr;
    ok = ok && checkCuda(cudaStreamCreate(&stream),
        "create large-water C-FDK stream");
    GeometryContext geometry;
    ResourceContext resources;
    ok = ok && geometry.initialize(p, views);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && fp->prepare(geometry, resources) &&
        fp->apply(d_truth.data(), p, d_projection.data(), resources) &&
        checkCuda(cudaStreamSynchronize(stream), "large-water FP synchronize");

    std::vector<float> host_projection(projection_count);
    std::vector<float> fdk_values(volume_count, 0.f);
    std::vector<float> cfdk_values(volume_count, 0.f);
    float fdk_ms = 0.f, cfdk_ms = 0.f;
    float derived_am = 0.f, derived_bm = 0.f, derived_c0 = 0.f,
        derived_s0 = 0.f;
    if (ok) {
        ok = checkCuda(cudaMemcpy(host_projection.data(), d_projection.data(),
            projection_count * sizeof(float), cudaMemcpyDeviceToHost),
            "download large-water projection");
        FdkPipeline fdk;
        if (ok) ok = fdk.prepareWithGeometry(p, views, 32, stream);
        const auto fdk_begin = std::chrono::steady_clock::now();
        if (ok) ok = fdk.processBatchSync({host_projection.data(), nullptr,
            nullptr, p.scan.NAng}, d_reconstruction.data(), true) && fdk.complete();
        ok = ok && checkCuda(cudaStreamSynchronize(stream),
            "large-water FDK synchronize");
        fdk_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fdk_begin).count());
        if (ok) {
            memory.download3D(h_reconstruction, d_reconstruction);
            std::copy(h_reconstruction.cdata(), h_reconstruction.cdata() +
                volume_count, fdk_values.begin());
        }
        fdk.release();
    }
    if (ok) {
        Fdk::CurveFilteredFdkPipeline cfdk;
        ok = cfdk.prepare(p, views, stream) &&
            checkCuda(cudaMemsetAsync(d_reconstruction.data(), 0,
                volume_count * sizeof(float), stream), "clear large-water C-FDK output");
        if (ok) {
            const auto& derived = cfdk.derivedGeometry();
            derived_am = 0.5f * (p.scan.Nu - 1) * p.scan.du_mm *
                derived.sid_mm / derived.sdd_mm;
            derived_bm = derived.bm_mm;
            derived_c0 = derived.c0_mm;
            derived_s0 = derived.s0_mm;
        }
        const auto cfdk_begin = std::chrono::steady_clock::now();
        if (ok) ok = cfdk.reconstruct(d_projection.data(), d_reconstruction.data(), true) &&
            cfdk.wait();
        cfdk_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cfdk_begin).count());
        if (ok) {
            memory.download3D(h_reconstruction, d_reconstruction);
            std::copy(h_reconstruction.cdata(), h_reconstruction.cdata() +
                volume_count, cfdk_values.begin());
        }
        cfdk.release();
    }

    const std::filesystem::path output_dir =
        std::filesystem::path(YKCBCT_TEST_SOURCE_DIR) / "output" / "cfdk" /
        "large_water";
    std::filesystem::create_directories(output_dir);
    const auto write_raw = [&](const char* name, const std::vector<float>& data) {
        std::ofstream output(output_dir / name, std::ios::binary);
        output.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
        return output.good();
    };
    ok = ok && write_raw("truth_f32.raw", truth) &&
        write_raw("projection_f32.raw", host_projection) &&
        write_raw("fdk_f32.raw", fdk_values) &&
        write_raw("cfdk_f32.raw", cfdk_values);

    const auto make_water_profile = [&](const std::vector<float>& volume,
        float radius_mm) {
        std::vector<float> result(p.volume.Nz, 0.f);
        for (int z = 0; z < p.volume.Nz; ++z) {
            double sum = 0.0; size_t count = 0;
            for (int y = 0; y < p.volume.Ny; ++y) {
                for (int x = 0; x < p.volume.Nx; ++x) {
                const float wx = origin.x + x * vg.vox_x;
                const float wy = origin.y + y * vg.vox_y;
                if (wx * wx + wy * wy > radius_mm * radius_mm) continue;
                sum += volume[(static_cast<size_t>(z) * p.volume.Ny + y) *
                    p.volume.Nx + x];
                ++count;
                }
            }
            if (count) result[z] = static_cast<float>(sum / count);
        }
        return result;
    };
    const auto truth_profile = make_water_profile(truth, 50.f);
    const auto fdk_profile = make_water_profile(fdk_values, 50.f);
    const auto cfdk_profile = make_water_profile(cfdk_values, 50.f);
    const auto profile_at = [&](const std::vector<float>& profile, float z_mm) {
        const int z = std::clamp(static_cast<int>(std::lround(
            (z_mm - origin.z) / vg.vox_z)), 0, p.volume.Nz - 1);
        return profile[z];
    };
    const float z_probe[3] = {-60.f, 0.f, 60.f};
    YK_LOGI("[LargeWater C-FDK] geometry: water diameter={} mm, water height={} mm, "
        "volume={}x{}x{} @ {} mm, views={}", 2.f * water_radius_mm,
        2.f * water_half_height_mm, p.volume.Nx, p.volume.Ny, p.volume.Nz,
        p.volume.voxelX_mm, p.scan.NAng);
    for (const float z_mm : z_probe) {
        YK_LOGI("[LargeWater C-FDK] z={:.1f} mm water truth/FDK/C-FDK="
            "{:.6e}/{:.6e}/{:.6e}", z_mm, water_mu,
            profile_at(fdk_profile, z_mm), profile_at(cfdk_profile, z_mm));
    }
    const float fdk_center = profile_at(fdk_profile, 0.f);
    const float cfdk_center = profile_at(cfdk_profile, 0.f);
    const float fdk_end = 0.5f * (profile_at(fdk_profile, -60.f) +
        profile_at(fdk_profile, 60.f));
    const float cfdk_end = 0.5f * (profile_at(cfdk_profile, -60.f) +
        profile_at(cfdk_profile, 60.f));
    YK_LOGI("[LargeWater C-FDK] |z|=60/center ratio FDK={:.6f}, "
        "C-FDK={:.6f}", fdk_center != 0.f ? fdk_end / fdk_center : 0.f,
        cfdk_center != 0.f ? cfdk_end / cfdk_center : 0.f);
    YK_LOGI("[LargeWater C-FDK] virtual detector am/bm={:.3f}/{:.3f} mm, "
        "c0/s0/bm={:.3f}/{:.3f}/{:.3f} mm, half-cone={:.3f} deg",
        derived_am, derived_bm, derived_c0, derived_s0, derived_bm,
        std::atan(derived_am / p.scan.sid_mm) * 180.f / CUDA_PI);
    YK_LOGI("[LargeWater C-FDK] time FDK={:.1f} ms, C-FDK={:.1f} ms",
        fdk_ms, cfdk_ms);

    const int y_slice = p.volume.Ny / 2;
    const int x_slice = p.volume.Nx / 2;
    const auto make_coronal = [&](const std::vector<float>& volume) {
        std::vector<float> slice(static_cast<size_t>(p.volume.Nx) * p.volume.Nz);
        for (int z = 0; z < p.volume.Nz; ++z) for (int x = 0; x < p.volume.Nx; ++x)
            slice[static_cast<size_t>(z) * p.volume.Nx + x] = volume[
                (static_cast<size_t>(z) * p.volume.Ny + y_slice) * p.volume.Nx + x];
        return slice;
    };
    const auto make_sagittal = [&](const std::vector<float>& volume) {
        std::vector<float> slice(static_cast<size_t>(p.volume.Ny) * p.volume.Nz);
        for (int z = 0; z < p.volume.Nz; ++z) for (int y = 0; y < p.volume.Ny; ++y)
            slice[static_cast<size_t>(z) * p.volume.Ny + y] = volume[
                (static_cast<size_t>(z) * p.volume.Ny + y) * p.volume.Nx + x_slice];
        return slice;
    };
    auto truth_xz = make_coronal(truth), fdk_xz = make_coronal(fdk_values),
        cfdk_xz = make_coronal(cfdk_values);
    auto truth_yz = make_sagittal(truth), fdk_yz = make_sagittal(fdk_values),
        cfdk_yz = make_sagittal(cfdk_values);
    const std::vector<TestImage::GrayPanel> panels = {
        {&truth_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,0.f,0.025f},
        {&fdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,0.f,0.025f},
        {&cfdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,0.f,0.025f},
        {&truth_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,0.f,0.025f},
        {&fdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,0.f,0.025f},
        {&cfdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,0.f,0.025f}
    };
    ok = TestImage::writeGrayMontageBmp(output_dir / "coronal_sagittal.bmp",
        panels, 3, 8, 2) && ok;
    const std::vector<TestImage::GrayPanel> water_window_panels = {
        {&truth_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,0.017f,0.022f},
        {&fdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,0.017f,0.022f},
        {&cfdk_xz,p.volume.Nx,p.volume.Nz,1,0,1.f,0.017f,0.022f},
        {&truth_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,0.017f,0.022f},
        {&fdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,0.017f,0.022f},
        {&cfdk_yz,p.volume.Ny,p.volume.Nz,1,0,1.f,0.017f,0.022f}
    };
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "coronal_sagittal_water_window.bmp", water_window_panels, 3, 8, 2) && ok;
    std::ofstream profile(output_dir / "z_profile.csv");
    profile << "z_mm,truth,fdk,cfdk,fdk_over_truth,cfdk_over_truth\n";
    for (int z = 0; z < p.volume.Nz; ++z) {
        const float z_mm = origin.z + z * vg.vox_z;
        const float expected = truth_profile[z];
        profile << z_mm << ',' << expected << ',' << fdk_profile[z] << ',' <<
            cfdk_profile[z] << ',' <<
            (expected > 0.f ? fdk_profile[z] / expected : 0.f) << ',' <<
            (expected > 0.f ? cfdk_profile[z] / expected : 0.f) << '\n';
    }
    ok = profile.good() && ok;
    std::ofstream metadata(output_dir / "metadata.json");
    metadata << "{\n"
        << "  \"data_type\": \"float32 little-endian\",\n"
        << "  \"layout\": \"[z][y][x], x fastest\",\n"
        << "  \"volume_xyz\": [" << p.volume.Nx << ", " << p.volume.Ny
        << ", " << p.volume.Nz << "],\n"
        << "  \"voxel_mm_xyz\": [" << p.volume.voxelX_mm << ", "
        << p.volume.voxelY_mm << ", " << p.volume.voxelZ_mm << "],\n"
        << "  \"water_mu_per_mm\": " << water_mu << ",\n"
        << "  \"water_diameter_mm\": " << 2.f * water_radius_mm << ",\n"
        << "  \"water_height_mm\": " << 2.f * water_half_height_mm << ",\n"
        << "  \"detector_uv\": [" << p.scan.Nu << ", " << p.scan.Nv
        << "],\n"
        << "  \"detector_pixel_mm_uv\": [" << p.scan.du_mm << ", "
        << p.scan.dv_mm << "],\n"
        << "  \"sid_mm\": " << p.scan.sid_mm << ",\n"
        << "  \"sdd_mm\": " << p.scan.sdd_mm << ",\n"
        << "  \"views\": " << p.scan.NAng << ",\n"
        << "  \"fdk_time_ms\": " << fdk_ms << ",\n"
        << "  \"cfdk_time_ms\": " << cfdk_ms << ",\n"
        << "  \"end_to_center_ratio\": {\"fdk\": "
        << (fdk_center != 0.f ? fdk_end / fdk_center : 0.f)
        << ", \"cfdk\": "
        << (cfdk_center != 0.f ? cfdk_end / cfdk_center : 0.f) << "},\n"
        << "  \"forward_projector\": \"Joseph x1\",\n"
        << "  \"files\": {\"truth\": \"truth_f32.raw\", "
        << "\"fdk\": \"fdk_f32.raw\", \"cfdk\": \"cfdk_f32.raw\"}\n"
        << "}\n";
    metadata.close();
    ok = metadata.good() && ok;
    ok = ok && std::all_of(fdk_values.begin(), fdk_values.end(),
        [](float value) { return std::isfinite(value); }) &&
        std::all_of(cfdk_values.begin(), cfdk_values.end(),
        [](float value) { return std::isfinite(value); }) &&
        fdk_center > 0.005f && cfdk_center > 0.005f;
    if (fp) fp->release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    YK_LOGI("[LargeWater C-FDK] artifacts: {} ({})", output_dir.string(),
        ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}

// ICRP 真实投影的大锥角对比。原始 1024x1024x720 数据按 2x2 像素、隔一帧
// 降采样为 512x512x360，保持探测器物理宽度和约 19.4 度锥角，避免测试被
// 3 GB 输入和 768^3 输出的显存占用主导。结果输出到 output/cfdk/icrp。
int main_curve_filtered_fdk_icrp_comparison()
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(YKCBCT_TEST_SOURCE_DIR) / "example" /
        "ICRP_female_head_20260827";
    const fs::path projection_path = root /
        "icrp145_female_head_cbct_z720_1024_frame0_1024x1024pixels_720proj.raw";
    const fs::path air_path = root /
        "icrp145_female_head_cbct_z720_1024_air_frame0_1024x1024pixels_1proj.raw";
    if (!fs::exists(projection_path) || !fs::exists(air_path)) {
        YK_LOGE("[ICRP C-FDK] projection/air 文件不存在，跳过真实数据测试。");
        return 1;
    }
    constexpr int in_u = 1024, in_v = 1024, in_views = 720;
    constexpr int nu = 512, nv = 512, views_count = 360;
    constexpr float in_du = 0.787654f, in_dv = 0.787654f;
    constexpr float sid = 709.f, sdd = 1143.699951f;
    std::vector<float> air(static_cast<size_t>(in_u) * in_v);
    {
        std::ifstream stream(air_path, std::ios::binary);
        stream.read(reinterpret_cast<char*>(air.data()),
            static_cast<std::streamsize>(air.size() * sizeof(float)));
        if (!stream) return 1;
    }
    std::vector<float> projection(static_cast<size_t>(views_count) * nu * nv);
    std::vector<float> input(static_cast<size_t>(in_u) * in_v);
    std::ifstream stream(projection_path, std::ios::binary);
    for (int view = 0; view < views_count; ++view) {
        stream.seekg(static_cast<std::streamoff>(view * 2ull * input.size() * sizeof(float)));
        stream.read(reinterpret_cast<char*>(input.data()),
            static_cast<std::streamsize>(input.size() * sizeof(float)));
        if (!stream) return 1;
        for (int y = 0; y < nv; ++y) for (int x = 0; x < nu; ++x) {
            const int iy = 2 * y, ix = 2 * x;
            const float measured = 0.25f * (input[iy * in_u + ix] +
                input[iy * in_u + ix + 1] + input[(iy + 1) * in_u + ix] +
                input[(iy + 1) * in_u + ix + 1]);
            const float reference = 0.25f * (air[iy * in_u + ix] +
                air[iy * in_u + ix + 1] + air[(iy + 1) * in_u + ix] +
                air[(iy + 1) * in_u + ix + 1]);
            projection[(static_cast<size_t>(view) * nv + y) * nu + x] =
                -std::log(std::clamp(measured / std::max(reference, 1e-6f),
                    1e-6f, 1.f));
        }
    }
    SReconstructionParams p{};
    p.scan.Nu = nu; p.scan.Nv = nv; p.scan.NAng = views_count;
    p.scan.totalViews = views_count; p.scan.du_mm = 2.f * in_du;
    p.scan.dv_mm = 2.f * in_dv; p.scan.sid_mm = sid; p.scan.sdd_mm = sdd;
    p.scan.range_rad = 2.f * CUDA_PI; p.scan.angles.resize(views_count);
    for (int i = 0; i < views_count; ++i)
        p.scan.angles[i] = 2.f * CUDA_PI * i / views_count;
    p.volume.Nx = 384; p.volume.Ny = 384; p.volume.Nz = 384;
    p.volume.voxelX_mm = p.volume.voxelY_mm = p.volume.voxelZ_mm = 1.f;

    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    Mem::MemoryController memory;
    auto h_projection = memory.allocateCpu3D<float>(nu, nv, views_count, false);
    std::copy(projection.begin(), projection.end(), h_projection.data());
    auto d_projection = memory.allocateDevice3D<float>(nu, nv, views_count, 0, false);
    auto d_fdk = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto d_cfdk = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto h_fdk = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    auto h_cfdk = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    memory.upload3D(d_projection, h_projection);
    cudaStream_t cuda_stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&cuda_stream), "create ICRP C-FDK stream");
    FdkPipeline fdk;
    Fdk::CurveFilteredFdkPipeline cfdk;
    const auto begin = std::chrono::steady_clock::now();
    ok = ok && fdk.prepareWithGeometry(p, geometry, 32, cuda_stream) &&
        fdk.processBatchSync({projection.data(), nullptr, nullptr, views_count},
            d_fdk.data(), true) && fdk.complete();
    const double fdk_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    const auto cfdk_begin = std::chrono::steady_clock::now();
    ok = ok && cfdk.prepare(p, geometry, cuda_stream) &&
        cfdk.reconstruct(d_projection.data(), d_cfdk.data(), true) && cfdk.wait();
    const double cfdk_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cfdk_begin).count();
    if (ok) { memory.download3D(h_fdk, d_fdk); memory.download3D(h_cfdk, d_cfdk); }
    if (ok) {
        const fs::path out = fs::path(YKCBCT_TEST_SOURCE_DIR) / "output" / "cfdk" / "icrp";
        fs::create_directories(out);
        const auto minmax = [](const auto& h) {
            auto it = std::minmax_element(h.cdata(), h.cdata() +
                static_cast<size_t>(384) * 384 * 384);
            return std::pair<float, float>{*it.first, *it.second};
        };
        const auto f = minmax(h_fdk), c = minmax(h_cfdk);
        const float wmin = std::max(0.f, std::min(f.first, c.first));
        const float wmax = std::min(0.08f, std::max(f.second, c.second));
        std::vector<float> vf(h_fdk.cdata(), h_fdk.cdata() + 384ull * 384 * 384);
        std::vector<float> vc(h_cfdk.cdata(), h_cfdk.cdata() + 384ull * 384 * 384);
        std::vector<float> difference(vf.size(), 0.f);
        for (size_t i = 0; i < difference.size(); ++i)
            difference[i] = std::fabs(vc[i] - vf[i]) * 10.f;
        const std::vector<TestImage::GrayPanel> panels = {
            {&vf,384,384,384,192,1.f,wmin,wmax}, {&vc,384,384,384,192,1.f,wmin,wmax},
            {&difference,384,384,384,192,1.f,0.f,0.01f},
            {&vf,384,384,384,202,1.f,wmin,wmax}, {&vc,384,384,384,202,1.f,wmin,wmax},
            {&difference,384,384,384,202,1.f,0.f,0.01f}};
        ok = TestImage::writeGrayMontageBmp(out / "icrp_fdk_cfdk_center_offz.bmp",
            panels, 3, 8, 2);
        YK_LOGI("[ICRP C-FDK] FDK={:.1f} ms C-FDK={:.1f} ms window=[{:.4g},{:.4g}] "
            "difference=10x absolute, image={}", fdk_ms, cfdk_ms, wmin, wmax,
            (out / "icrp_fdk_cfdk_center_offz.bmp").string());
    }
    cfdk.release(); fdk.release();
    if (cuda_stream) cudaStreamDestroy(cuda_stream);
    return ok ? 0 : 1;
}

// 坐标变换在前端烘焙几何，但体素数组仍固定在 Object 坐标系。该测试使用
// 非零体积中心，验证绕中心旋转的枢轴补偿、点/向量规则和逆变换。
int main_rigid_geometry_transform_smoke()
{
    const float3 volume_center = make_float3(12.f, -7.f, 4.f);
    const float3 motion = make_float3(3.f, 2.f, -1.f);
    const SRigidTransform objectFromScanner = SRigidTransform::aroundAxisPoint(
        make_float3(0.f, 0.f, 1.f), 0.5f * CUDA_PI,
        volume_center, motion);

    const float3 transformed_center = objectFromScanner.transformPoint(volume_center);
    const float3 transformed_offset = objectFromScanner.transformPoint(
        make_float3(volume_center.x + 2.f, volume_center.y, volume_center.z));
    const float3 transformed_vector = objectFromScanner.transformVector(
        make_float3(2.f, 0.f, 0.f));
    const float3 roundtrip = objectFromScanner.inverse().transformPoint(
        transformed_offset);
    const float3 matrix_transformed = SMat4f::from_rigid(objectFromScanner)
        .apply_point(make_float3(volume_center.x + 2.f,
            volume_center.y, volume_center.z));

    const auto near = [](float a, float b) { return std::fabs(a - b) < 1e-5f; };
    bool ok = objectFromScanner.isRigid() &&
        near(transformed_center.x, volume_center.x + motion.x) &&
        near(transformed_center.y, volume_center.y + motion.y) &&
        near(transformed_center.z, volume_center.z + motion.z) &&
        near(transformed_offset.x, volume_center.x + motion.x) &&
        near(transformed_offset.y, volume_center.y + motion.y + 2.f) &&
        near(transformed_vector.x, 0.f) && near(transformed_vector.y, 2.f) &&
        near(matrix_transformed.x, transformed_offset.x) &&
        near(matrix_transformed.y, transformed_offset.y) &&
        near(matrix_transformed.z, transformed_offset.z) &&
        near(roundtrip.x, volume_center.x + 2.f) &&
        near(roundtrip.y, volume_center.y) && near(roundtrip.z, volume_center.z);

    SReconstructionParams desc{};
    std::vector<SConeProjGeomVec> views;
    desc.scan.Nu = 8; desc.scan.Nv = 6; desc.scan.NAng = 1; desc.scan.totalViews = 1;
    desc.scan.sid_mm = 100.f; desc.scan.sdd_mm = 200.f;
    desc.volume.Nx = 8; desc.volume.Ny = 7; desc.volume.Nz = 6;
    desc.volume.centerX_mm = volume_center.x;
    desc.volume.centerY_mm = volume_center.y;
    desc.volume.centerZ_mm = volume_center.z;
    views.push_back({
        make_float4(10.f, -100.f, 4.f, 0.f),
        make_float4(-4.f, 100.f, 1.f, 0.f),
        make_float4(1.f, 0.f, 0.f, 0.f),
        make_float4(0.f, 0.f, 1.f, 0.f),
        make_float4(0.25f, 0.f, 0.f, 0.f) });

    // 该视图同时含 U/V offset 和探测器倾斜：中心射线不等于平面法向。
    // 派生帧经过刚体变换后，点随平移旋转，方向只旋转，平面距离不变。
    SProjectionFrame scanner_frame{};
    SProjectionFrame expected_frame{};
    const SConeProjGeomVec transformed_geometry = transformProjectionGeometry(
        views.front(), objectFromScanner);
    ok = ok && deriveProjectionFrame(views.front(),
        desc.scan.Nu, desc.scan.Nv, scanner_frame) &&
        deriveProjectionFrame(transformed_geometry,
            desc.scan.Nu, desc.scan.Nv, expected_frame);
    if (ok) {
        const float3 expected_center_ray = objectFromScanner.transformVector(
            scanner_frame.centerRay);
        const float3 expected_normal = objectFromScanner.transformVector(
            scanner_frame.detectorNormal);
        const float3 expected_principal = objectFromScanner.transformPoint(
            scanner_frame.principalPoint);
        const float alignment = scanner_frame.centerRay.x * scanner_frame.detectorNormal.x +
            scanner_frame.centerRay.y * scanner_frame.detectorNormal.y +
            scanner_frame.centerRay.z * scanner_frame.detectorNormal.z;
        ok = alignment < 0.9999f &&
            near(expected_frame.centerRay.x, expected_center_ray.x) &&
            near(expected_frame.centerRay.y, expected_center_ray.y) &&
            near(expected_frame.centerRay.z, expected_center_ray.z) &&
            near(expected_frame.detectorNormal.x, expected_normal.x) &&
            near(expected_frame.detectorNormal.y, expected_normal.y) &&
            near(expected_frame.detectorNormal.z, expected_normal.z) &&
            near(expected_frame.principalPoint.x, expected_principal.x) &&
            near(expected_frame.principalPoint.y, expected_principal.y) &&
            near(expected_frame.principalPoint.z, expected_principal.z) &&
            near(expected_frame.planeDistance, scanner_frame.planeDistance);
    }

    // offset 必须沿倾斜后的最终 U/V 轴生效，而不是沿未倾斜的世界 X/Z 轴。
    std::vector<SConeProjGeomVec> offset_geometry;
    buildTestCircularGeometry(offset_geometry, { 0.f },
        1, 8, 6, 1.f, 1.f, 100.f, 100.f,
        make_float3(3.f, 0.f, 4.f), make_float3(20.f, 10.f, 0.f));
    SProjectionFrame offset_frame{};
    ok = ok && deriveProjectionFrame(offset_geometry.front(), 8, 6, offset_frame);
    if (ok) {
        const float4 u4 = offset_geometry.front().detU;
        const float4 v4 = offset_geometry.front().detV;
        const float u_length = std::sqrt(u4.x * u4.x + u4.y * u4.y + u4.z * u4.z);
        const float v_length = std::sqrt(v4.x * v4.x + v4.y * v4.y + v4.z * v4.z);
        const float3 expected_center = make_float3(
            3.f * u4.x / u_length + 4.f * v4.x / v_length,
            100.f + 3.f * u4.y / u_length + 4.f * v4.y / v_length,
            3.f * u4.z / u_length + 4.f * v4.z / v_length);
        ok = near(offset_frame.detectorCenter.x, expected_center.x) &&
            near(offset_frame.detectorCenter.y, expected_center.y) &&
            near(offset_frame.detectorCenter.z, expected_center.z);
    }

    GeometryContext context;
    ok = ok && context.initialize(desc, {transformed_geometry});
    if (ok) {
        const SVolGeom volume = context.volumeGeometry();
        const SConeProjGeomVec expected = transformProjectionGeometry(
            views.front(), objectFromScanner);
        const SConeProjGeomVec& actual = context.allGeometry().front();
        ok = near(volume.center.x, volume_center.x) &&
            near(volume.center.y, volume_center.y) &&
            near(volume.center.z, volume_center.z) &&
            near(actual.src.x, expected.src.x) &&
            near(actual.src.y, expected.src.y) &&
            near(actual.src.z, expected.src.z) &&
            near(actual.detU.x, expected.detU.x) &&
            near(actual.detU.y, expected.detU.y) &&
            near(actual.angle.x, views.front().angle.x);
    }

    // 未提供显式 geometry 时，先在 Scanner 坐标系生成圆轨迹，再应用同一
    // 公共刚体类型；对照恒等姿态生成的 geometry，避免前端存在第二套变换。
    SSystemConfig identity_system{};
    identity_system.circular.total_views = 1;
    identity_system.circular.views_per_turn = 32;
    identity_system.circular.start_angle_rad = 0.25f;
    identity_system.circular.sid_mm = 100.f;
    identity_system.circular.sdd_mm = 200.f;
    identity_system.flat_detector.channels = 8;
    identity_system.flat_detector.rows = 6;
    identity_system.flat_detector.channel_size_mm = 1.f;
    identity_system.flat_detector.row_size_mm = 1.f;
    identity_system.volume = {8, 7, 6, 1.f, 1.f, 1.f, volume_center};
    GeometryContext scanner_context;
    ok = ok && scanner_context.initialize(identity_system);
    auto transformed_views = scanner_context.allGeometry();
    for (auto& view : transformed_views)
        view = transformProjectionGeometry(view, objectFromScanner);
    GeometryContext transformed_context;
    ok = ok && transformed_context.initialize(scanner_context.base(), transformed_views);
    if (ok) {
        const SConeProjGeomVec expected = transformProjectionGeometry(
            scanner_context.allGeometry().front(), objectFromScanner);
        const SConeProjGeomVec& actual = transformed_context.allGeometry().front();
        ok = near(actual.src.x, expected.src.x) &&
            near(actual.src.y, expected.src.y) &&
            near(actual.src.z, expected.src.z) &&
            near(actual.detS.x, expected.detS.x) &&
            near(actual.detS.y, expected.detS.y) &&
            near(actual.detS.z, expected.detS.z) &&
            near(actual.angle.x, 0.25f);
    }
    std::printf("rigid geometry with non-origin volume center: %s\n",
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 新框架的最小闭环：圆轨迹 operator 生成投影，随后 BP operator 消费同一块
// 调用方拥有的设备缓冲。它覆盖了无 runner 的参数流、数据流和累加语义。
int main_operator_roundtrip_smoke()
{
    const SReconstructionParams p = makeSmallParams();
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t sino_n = static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv;
    std::vector<float> h_volume = TestPhantom::makeBasic(p);

    cudaStream_t stream = nullptr;
    float* d_volume = nullptr; float* d_sino = nullptr; float* d_backprojection = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create stream") &&
        checkCuda(cudaMalloc(&d_volume, volume_n * sizeof(float)), "allocate volume") &&
        checkCuda(cudaMalloc(&d_sino, sino_n * sizeof(float)), "allocate projection") &&
        checkCuda(cudaMalloc(&d_backprojection, volume_n * sizeof(float)), "allocate backprojection") &&
        checkCuda(cudaMemcpyAsync(d_volume, h_volume.data(), volume_n * sizeof(float),
            cudaMemcpyHostToDevice, stream), "upload volume");

    GeometryContext geometry;
    ResourceContext resources;
    ok = ok && geometry.initialize(p);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    auto bp = makeBackOperator(ETask::BP_Joseph_v3);
    ok = ok && fp->prepare(geometry, resources) && bp->prepare(geometry, resources) &&
        fp->apply(d_volume, p, d_sino, resources) &&
        bp->apply(d_sino, p, d_backprojection, true, resources) &&
        checkCuda(cudaStreamSynchronize(stream), "operator synchronize");

    std::vector<float> h_sino(sino_n), h_backprojection(volume_n);
    ok = ok && checkCuda(cudaMemcpy(h_sino.data(), d_sino, sino_n * sizeof(float),
        cudaMemcpyDeviceToHost), "download projection") &&
        checkCuda(cudaMemcpy(h_backprojection.data(), d_backprojection, volume_n * sizeof(float),
            cudaMemcpyDeviceToHost), "download backprojection") &&
        hasSignal(h_sino) && hasSignal(h_backprojection);
    std::printf("operator roundtrip: %s\n", ok ? "PASS" : "FAIL");

    fp->release(); bp->release();
    if (d_backprojection) cudaFree(d_backprojection);
    if (d_sino) cudaFree(d_sino);
    if (d_volume) cudaFree(d_volume);
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// FP/BP 的 apply() 是异步接口，但算子拥有 kernel 使用的纹理和几何缓冲。
// 本测试不做外部 stream 同步，直接 release()，用于验证内部完成事件会先等待
// kernel，再销毁这些资源；release() 返回后结果应可立即下载。
int main_operator_release_fence_smoke()
{
    const SReconstructionParams p = makeSmallParams();
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t sino_n = static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv;
    const std::vector<float> h_volume = TestPhantom::makeBasic(p);

    cudaStream_t stream = nullptr;
    float* d_volume = nullptr; float* d_sino = nullptr; float* d_backprojection = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create release-fence stream") &&
        checkCuda(cudaMalloc(&d_volume, volume_n * sizeof(float)), "allocate fence volume") &&
        checkCuda(cudaMalloc(&d_sino, sino_n * sizeof(float)), "allocate fence projection") &&
        checkCuda(cudaMalloc(&d_backprojection, volume_n * sizeof(float)), "allocate fence backprojection") &&
        checkCuda(cudaMemcpyAsync(d_volume, h_volume.data(), volume_n * sizeof(float),
            cudaMemcpyHostToDevice, stream), "upload fence volume");

    GeometryContext geometry;
    ResourceContext resources;
    ok = ok && geometry.initialize(p);
    resources.attach(stream, 0);
    auto fp = makeForwardOperator(ETask::FP_Joseph);
    auto bp = makeBackOperator(ETask::BP_Joseph_v3);
    ok = ok && fp->prepare(geometry, resources) && bp->prepare(geometry, resources) &&
        fp->apply(d_volume, p, d_sino, resources);
    if (ok) fp->release();
    ok = ok && bp->apply(d_sino, p, d_backprojection, true, resources);
    if (ok) bp->release();

    std::vector<float> h_sino(sino_n), h_backprojection(volume_n);
    ok = ok && checkCuda(cudaMemcpy(h_sino.data(), d_sino, sino_n * sizeof(float),
        cudaMemcpyDeviceToHost), "download fenced projection") &&
        checkCuda(cudaMemcpy(h_backprojection.data(), d_backprojection, volume_n * sizeof(float),
            cudaMemcpyDeviceToHost), "download fenced backprojection") &&
        hasSignal(h_sino) && hasSignal(h_backprojection);
    std::printf("operator release fence: %s\n", ok ? "PASS" : "FAIL");

    if (d_backprojection) cudaFree(d_backprojection);
    if (d_sino) cudaFree(d_sino);
    if (d_volume) cudaFree(d_volume);
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// 同一圆轨迹以“内部角度构造”和“外部逐视图 geometry”两种方式进入 operator，
// 结果应一致。这是 geometry 为唯一真源的基础回归。
int main_external_geometry_operator_smoke()
{
    const SReconstructionParams p = makeSmallParams();
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t sino_n = static_cast<size_t>(p.scan.NAng) * p.scan.Nu * p.scan.Nv;
    std::vector<float> h_volume = TestPhantom::makeCatphanLike(p);

    std::vector<SConeProjGeomVec> external_geometry;
    detail::buildCircularViews(p, external_geometry);
    GeometryContext circular, external;
    if (!circular.initialize(p) || !external.initialize(p, external_geometry)) return 1;

    cudaStream_t stream = nullptr;
    float* d_volume = nullptr; float* d_circular = nullptr; float* d_external = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create stream") &&
        checkCuda(cudaMalloc(&d_volume, volume_n * sizeof(float)), "allocate volume") &&
        checkCuda(cudaMalloc(&d_circular, sino_n * sizeof(float)), "allocate circular sino") &&
        checkCuda(cudaMalloc(&d_external, sino_n * sizeof(float)), "allocate external sino") &&
        checkCuda(cudaMemcpyAsync(d_volume, h_volume.data(), volume_n * sizeof(float),
            cudaMemcpyHostToDevice, stream), "upload volume");
    ResourceContext resources;
    resources.attach(stream, 0);
    auto fp_circular = makeForwardOperator(ETask::FP_Joseph);
    auto fp_external = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && fp_circular->prepare(circular, resources) && fp_external->prepare(external, resources) &&
        fp_circular->apply(d_volume, p, d_circular, resources) &&
        fp_external->apply(d_volume, p, d_external, resources) &&
        checkCuda(cudaStreamSynchronize(stream), "external geometry synchronize");

    std::vector<float> h_circular(sino_n), h_external(sino_n);
    ok = ok && checkCuda(cudaMemcpy(h_circular.data(), d_circular, sino_n * sizeof(float),
        cudaMemcpyDeviceToHost), "download circular sino") &&
        checkCuda(cudaMemcpy(h_external.data(), d_external, sino_n * sizeof(float),
            cudaMemcpyDeviceToHost), "download external sino");
    const float diff = ok ? maxAbsDiff(h_circular, h_external) : INFINITY;
    ok = ok && diff < 1e-5f;
    std::printf("external geometry FP: max diff = %.8g, %s\n", diff, ok ? "PASS" : "FAIL");

    fp_circular->release(); fp_external->release();
    if (d_external) cudaFree(d_external);
    if (d_circular) cudaFree(d_circular);
    if (d_volume) cudaFree(d_volume);
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// FDK 在线的关键回归：整批和任意连续分批必须累积出相同体数据。投影完全
// 在测试中生成，所以该检查不依赖本机的原始数据目录。
int main_fdk_batch_consistency_smoke()
{
    const SReconstructionParams p = makeFdkSmokeParams();
    const size_t view_n = static_cast<size_t>(p.scan.Nu) * p.scan.Nv;
    const size_t sino_n = static_cast<size_t>(p.scan.NAng) * view_n;
    const size_t volume_n = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    std::vector<float> h_projection(sino_n);
    for (size_t i = 0; i < sino_n; ++i)
        h_projection[i] = 0.02f + static_cast<float>((i * 17) % 31) * 0.001f;

    cudaStream_t stream = nullptr;
    float* d_full = nullptr; float* d_split = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create stream") &&
        checkCuda(cudaMalloc(&d_full, volume_n * sizeof(float)), "allocate full volume") &&
        checkCuda(cudaMalloc(&d_split, volume_n * sizeof(float)), "allocate split volume");
    FdkPipeline full, split;
    ok = ok && full.prepareWithAngles(p, p.scan.angles, 4, stream) &&
        split.prepareWithAngles(p, p.scan.angles, 4, stream);
    if (ok) {
        const FdkProjectionBatch all{ h_projection.data(), nullptr, nullptr, p.scan.NAng };
        const int first_count = 5;
        const FdkProjectionBatch first{ h_projection.data(), nullptr, nullptr, first_count };
        const FdkProjectionBatch second{ h_projection.data() + first_count * view_n,
            nullptr, nullptr, p.scan.NAng - first_count };
        ok = full.processBatch(all, d_full, true) &&
            split.processBatch(first, d_split, true) &&
            split.processBatch(second, d_split, false) && split.complete() &&
            checkCuda(cudaStreamSynchronize(stream), "FDK synchronize");
    }
    std::vector<float> h_full(volume_n), h_split(volume_n);
    ok = ok && checkCuda(cudaMemcpy(h_full.data(), d_full, volume_n * sizeof(float),
        cudaMemcpyDeviceToHost), "download full FDK") &&
        checkCuda(cudaMemcpy(h_split.data(), d_split, volume_n * sizeof(float),
            cudaMemcpyDeviceToHost), "download split FDK");
    const float diff = ok ? maxAbsDiff(h_full, h_split) : INFINITY;
    ok = ok && diff < 1e-4f;
    std::printf("FDK batch consistency: max diff = %.8g, %s\n", diff, ok ? "PASS" : "FAIL");

    full.release(); split.release();
    if (d_split) cudaFree(d_split);
    if (d_full) cudaFree(d_full);
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// 生成器本身也应有可执行的结构检查：Catphan-like 模体必须含有空气、基材、
// 高对比材料和低对比材料，避免今后改尺寸或体素间距时模块悄悄退化为空体。
int main_catphan_phantom_smoke()
{
    const SReconstructionParams p = makeSmallParams();
    const std::vector<float> phantom = TestPhantom::makeCatphanLike(p);
    const auto [min_it, max_it] = std::minmax_element(phantom.begin(), phantom.end());
    int air = 0, base = 0, high = 0, low = 0;
    for (float value : phantom) {
        air += value == 0.f;
        base += std::fabs(value - 0.020f) < 1e-6f;
        high += value >= 0.050f;
        low += std::fabs(value - 0.022f) < 1e-6f;
    }
    const bool ok = air > 0 && base > 0 && high > 0 && low > 0 &&
        *min_it == 0.f && *max_it >= 0.080f;
    std::printf("Catphan-like phantom: air=%d base=%d high=%d low=%d, %s\n",
        air, base, high, low, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 同步便捷入口应在返回前完成 H2D 和全部 FDK kernel。这里在
// processBatchSync() 返回后立即改写主机投影，再下载体数据，
// 用于回归“输入可复用，输出可消费”的同步契约。
int main_fdk_synchronous_batch_smoke()
{
    const SReconstructionParams p = makeFdkSmokeParams();
    const size_t projection_count = static_cast<size_t>(p.scan.Nu) * p.scan.Nv * p.scan.NAng;
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    std::vector<float> projection(projection_count, 0.03f);
    Mem::MemoryController memory;
    auto device_volume = memory.allocateDevice3D<float>(
        p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto host_volume = memory.allocateCpu3D<float>(
        p.volume.Nx, p.volume.Ny, p.volume.Nz, false);

    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create sync FDK stream");
    FdkPipeline pipeline;
    ok = ok && pipeline.prepareWithAngles(p, p.scan.angles, 4, stream);
    if (ok) {
        const FdkProjectionBatch batch{ projection.data(), nullptr, nullptr, p.scan.NAng };
        ok = pipeline.processBatchSync(batch, device_volume.data(), true) &&
            pipeline.complete();
        std::fill(projection.begin(), projection.end(), 0.f);
        // processBatchSync() 返回后本批输出可直接下载，无需额外同步。
        if (ok) memory.download3D(host_volume, device_volume);
        const std::vector<float> volume(host_volume.cdata(),
            host_volume.cdata() + volume_count);
        ok = ok && hasSignal(volume);
    }
    std::printf("FDK synchronous batch contract: %s\n", ok ? "PASS" : "FAIL");

    pipeline.release();
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// 校准 geometry 下的近似 FDK：源端固定偏移由 builder 烘焙，逐视图漂移直接
// 写入最终 geometry。预加权、Parker 和 BP 都必须使用派生实际几何，不能退回
// params 中的标称 SID/SDD；非退化偏离只报告，不拒绝执行。
int main_fdk_calibrated_geometry_smoke()
{
    SReconstructionParams p = makeFdkSmokeParams();
    p.scan.short_scan = true;
    p.scan.range_rad = 1.2f * CUDA_PI;
    p.scan.start_angle_rad = -0.1f;
    p.scan.sourceOffsetX_mm = 0.35f;
    p.scan.sourceOffsetY_mm = -0.20f;
    p.scan.sourceOffsetZ_mm = 0.15f;
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = p.scan.start_angle_rad + p.scan.range_rad * i /
            static_cast<float>(p.scan.NAng - 1);

    std::vector<SConeProjGeomVec> geometry;
    const auto rad2deg = [](float value) { return value * 180.f / CUDA_PI; };
    buildTestCircularGeometry(geometry, p.scan.angles, p.scan.NAng,
        p.scan.Nu, p.scan.Nv, p.scan.du_mm, p.scan.dv_mm, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
        make_float3(0.4f, 0.f, -0.25f),
        make_float3(rad2deg(0.004f), rad2deg(-0.003f), rad2deg(0.002f)),
        make_float3(p.scan.sourceOffsetX_mm, p.scan.sourceOffsetY_mm, p.scan.sourceOffsetZ_mm));
    for (int i = 0; i < p.scan.NAng; ++i) {
        const float phase = 2.f * CUDA_PI * i / p.scan.NAng;
        geometry[i].src.x += 0.18f * std::sin(phase);
        geometry[i].src.y += 0.12f * std::cos(phase);
        geometry[i].src.z += 0.08f * std::sin(2.f * phase);
    }

    // 从这里开始只允许最终逐视图 geometry 参与重建。故意破坏标称构造参数，
    // 可防止 prepare、Parker、滤波或 BP 日后又偷偷读取这些旧参数。
    p.scan.sid_mm = -1.f;
    p.scan.sdd_mm = -2.f;
    p.scan.du_mm = -1.f;
    p.scan.dv_mm = -1.f;

    const size_t projection_count = static_cast<size_t>(p.scan.Nu) * p.scan.Nv * p.scan.NAng;
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    std::vector<float> projection(projection_count, 0.03f);
    Mem::MemoryController memory;
    auto device_volume = memory.allocateDevice3D<float>(
        p.volume.Nx, p.volume.Ny, p.volume.Nz, 0, false);
    auto host_volume = memory.allocateCpu3D<float>(
        p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create calibrated FDK stream");
    FdkPipeline pipeline;
    ok = ok && pipeline.prepareWithGeometry(p, geometry, 4, stream);
    if (ok) {
        const auto& diagnostic = pipeline.geometryDiagnostics();
        ok = diagnostic.approximate &&
            diagnostic.max_source_to_axis_mm > diagnostic.min_source_to_axis_mm &&
            diagnostic.max_source_to_detector_mm > diagnostic.min_source_to_detector_mm;
        const FdkProjectionBatch batch{ projection.data(), nullptr, nullptr, p.scan.NAng };
        ok = ok && pipeline.processBatchSync(batch, device_volume.data(), true) &&
            pipeline.complete();
        if (ok) memory.download3D(host_volume, device_volume);
        const std::vector<float> volume(host_volume.cdata(),
            host_volume.cdata() + volume_count);
        ok = ok && std::all_of(volume.begin(), volume.end(), [](float value) {
            return std::isfinite(value);
        }) && hasSignal(volume);
    }
    std::printf("FDK calibrated approximate geometry: %s\n", ok ? "PASS" : "FAIL");
    pipeline.release();
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

// 新系统级请求必须能直接初始化 Flat FDK，并由扫描几何自动判断 Parker。
int main_fdk_system_request_smoke()
{
    SSystemConfig system{};
    system.detector = EDetectorKind::Flat;
    system.trajectory = ETrajectoryKind::Circular;
    system.circular.total_views = 6;
    system.circular.views_per_turn = 10; // 1.2π，满足当前小扇角短扫。
    system.circular.sid_mm = 80.f;
    system.circular.sdd_mm = 160.f;
    system.flat_detector.channels = 16;
    system.flat_detector.rows = 8;
    system.flat_detector.channel_size_mm = 1.f;
    system.flat_detector.row_size_mm = 1.f;
    system.volume = {8, 8, 8, 1.f, 1.f, 1.f,
        make_float3(2.f, -1.f, 3.f)};
    SReconstructionSpec reconstruction{};
    reconstruction.fdk.parker.mode = EParkerMode::Auto;
    reconstruction.fdk.filter = EFdkFilter::Hann;

    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create system FDK stream");
    FdkPipeline pipeline;
    ok = ok && resolveParkerEnabled(system, reconstruction) &&
        pipeline.prepare(system, reconstruction, 4, stream);
    pipeline.release();

    system.circular.total_views = system.circular.views_per_turn;
    ok = ok && !resolveParkerEnabled(system, reconstruction) &&
        pipeline.prepare(system, reconstruction, 4, stream);
    pipeline.release();
    if (stream) cudaStreamDestroy(stream);
    std::printf("FDK system reconstruction request: %s\n",
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 探测器倾斜时，投影平面求交分母与 FDK 径向深度分母不再相同。
// 本测试直接核对 GPU 预计算系数，防止以后为了节省字段再次错误复用 Cd。
int main_fdk_depth_denominator_smoke()
{
    SReconstructionParams p = makeFdkSmokeParams();
    std::vector<SConeProjGeomVec> geometry;
    buildTestCircularGeometry(geometry, p.scan.angles, p.scan.NAng,
        p.scan.Nu, p.scan.Nv, p.scan.du_mm, p.scan.dv_mm, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
        make_float3(1.2f, 0.f, -0.8f),
        make_float3(8.f, 5.f, -4.f),
        make_float3(0.7f, -0.4f, 0.3f));

    std::vector<SFDKGeoParamPerView> derived;
    bool ok = GeoDerivedManagerVec{}.build_geo_params(p.scan.Nu, p.scan.Nv,
        p.scan.range_rad, geometry, derived);
    if (!ok) return 1;

    Mem::PodDataController memory;
    auto d_geometry = memory.allocateAndUpload(geometry);
    auto d_derived = memory.allocateAndUpload(derived);
    auto d_coefficients = memory.allocate<FdkAffineCoeff>(p.scan.NAng);
    cudaStream_t stream = nullptr;
    ok = checkCuda(cudaStreamCreate(&stream), "create depth denominator stream");
    if (ok) {
        Fdk::bp_launchPrecomputeCoeffs(d_geometry.data(), d_derived.data(),
            d_coefficients.data(), p.scan.NAng, stream);
        ok = checkCuda(cudaStreamSynchronize(stream), "precompute FDK coefficients");
    }

    std::vector<FdkAffineCoeff> coefficients;
    if (ok) memory.download(coefficients, d_coefficients);
    const float3 points[] = {
        make_float3(13.f, -9.f, 7.f),
        make_float3(-17.f, 11.f, -5.f),
        make_float3(4.f, 19.f, 12.f)
    };
    float max_plane_error = 0.f;
    float max_depth_error = 0.f;
    float max_denominator_separation = 0.f;
    for (int view = 0; ok && view < p.scan.NAng; ++view) {
        const auto& c = coefficients[view];
        const float3 src = f4_to_f3(geometry[view].src);
        const float3 normal = f4_to_f3(derived[view].det_n);
        const float3 radial = f4_to_f3(derived[view].radial_ray);
        for (const float3 point : points) {
            const float plane_coefficient = c.Cd_w + point.x * c.Cd_x +
                point.y * c.Cd_y + point.z * c.Cd_z;
            const float depth_coefficient = c.Cr_w + point.x * c.Cr_x +
                point.y * c.Cr_y + point.z * c.Cr_z;
            const float plane_direct = f3_dot(f3_sub(point, src), normal);
            const float depth_direct = f3_dot(f3_sub(point, src), radial);
            max_plane_error = std::max(max_plane_error,
                std::fabs(plane_coefficient - plane_direct));
            max_depth_error = std::max(max_depth_error,
                std::fabs(depth_coefficient - depth_direct));
            max_denominator_separation = std::max(max_denominator_separation,
                std::fabs(plane_coefficient - depth_coefficient));
        }
    }
    ok = ok && max_plane_error < 1e-4f && max_depth_error < 1e-4f &&
        max_denominator_separation > 0.1f;
    std::printf("FDK depth denominator: plane error=%.8g, depth error=%.8g, "
        "separation=%.8g, %s\n", max_plane_error, max_depth_error,
        max_denominator_separation, ok ? "PASS" : "FAIL");
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
