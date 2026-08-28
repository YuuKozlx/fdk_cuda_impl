#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include <global/YkGlobals.h>
#include <utility>
#include "FDK/YkFdkPipeline.hpp"
#include "FDK/kernels/YkFDKBpPrecompute.cuh"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "test_common.hpp"
#include "util/YkVecOperation.hpp"

namespace {

    using namespace YK;

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

    SessionDesc desc{};
    desc.scan.Nu = 8; desc.scan.Nv = 6; desc.scan.NAng = 1;
    desc.scan.SOD_mm = 100.f; desc.scan.SDD_mm = 200.f;
    desc.volume.Nx = 8; desc.volume.Ny = 7; desc.volume.Nz = 6;
    desc.volume.offsetX_mm = volume_center.x;
    desc.volume.offsetY_mm = volume_center.y;
    desc.volume.offsetZ_mm = volume_center.z;
    desc.geometry.push_back({
        make_float4(10.f, -100.f, 4.f, 0.f),
        make_float4(-4.f, 100.f, 1.f, 0.f),
        make_float4(1.f, 0.f, 0.f, 0.f),
        make_float4(0.f, 0.f, 1.f, 0.f),
        make_float4(0.25f, 0.f, 0.f, 0.f) });
    desc.objectFromScanner.push_back(objectFromScanner);

    // 该视图同时含 U/V offset 和探测器倾斜：中心射线不等于平面法向。
    // 派生帧经过刚体变换后，点随平移旋转，方向只旋转，平面距离不变。
    SProjectionFrame scanner_frame{};
    SProjectionFrame expected_frame{};
    const SConeProjGeomVec transformed_geometry = transformProjectionGeometry(
        desc.geometry.front(), objectFromScanner);
    ok = ok && deriveProjectionFrame(desc.geometry.front(),
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
    build_circular_vec_geometry_from_theta(offset_geometry, { 0.f },
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
    ok = ok && context.initialize(desc);
    if (ok) {
        const SVolGeom volume = context.volumeGeometry();
        const SConeProjGeomVec expected = transformProjectionGeometry(
            desc.geometry.front(), objectFromScanner);
        const SConeProjGeomVec& actual = context.allGeometry().front();
        ok = near(volume.center.x, volume_center.x) &&
            near(volume.center.y, volume_center.y) &&
            near(volume.center.z, volume_center.z) &&
            near(actual.src.x, expected.src.x) &&
            near(actual.src.y, expected.src.y) &&
            near(actual.src.z, expected.src.z) &&
            near(actual.detU.x, expected.detU.x) &&
            near(actual.detU.y, expected.detU.y) &&
            near(actual.angle.x, desc.geometry.front().angle.x);
    }

    // 未提供显式 geometry 时，先在 Scanner 坐标系生成圆轨迹，再应用同一
    // 公共刚体类型；对照恒等姿态生成的 geometry，避免前端存在第二套变换。
    SessionDesc identity_desc = desc;
    identity_desc.geometry.clear();
    identity_desc.angles = { 0.25f };
    identity_desc.objectFromScanner = { SRigidTransform::identity() };
    GeometryContext scanner_context;
    ok = ok && scanner_context.initialize(identity_desc);

    SessionDesc transformed_desc = identity_desc;
    transformed_desc.objectFromScanner = { objectFromScanner };
    GeometryContext transformed_context;
    ok = ok && transformed_context.initialize(transformed_desc);
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
    build_circular_vec_geometry_from_theta(geometry, p.scan.angles, p.scan.NAng,
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

// 探测器倾斜时，投影平面求交分母与 FDK 径向深度分母不再相同。
// 本测试直接核对 GPU 预计算系数，防止以后为了节省字段再次错误复用 Cd。
int main_fdk_depth_denominator_smoke()
{
    SReconstructionParams p = makeFdkSmokeParams();
    std::vector<SConeProjGeomVec> geometry;
    build_circular_vec_geometry_from_theta(geometry, p.scan.angles, p.scan.NAng,
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
