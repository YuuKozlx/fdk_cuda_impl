#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include <global/YkGlobals.h>
#include <utility>
#include "FDK/YkFdkPipeline.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "test_common.hpp"

namespace {

    using namespace YK;

    SCBCTParams makeSmallParams(int views = 12)
    {
        SCBCTParams p{};
        p.iPU = 48; p.iPV = 36;
        p.iPAng = views; p.iPAngTotal = views;
        p.iVX = 32; p.iVY = 32; p.iVZ = 24;
        p.du_mm = 1.f; p.dv_mm = 1.f;
        p.vox_x_mm = 1.f; p.vox_y_mm = 1.f; p.vox_z_mm = 1.f;
        p.SID = 100.f; p.SDD = 200.f;
        p.scan_range_rad = 2.f * CUDA_PI;
        p.scan_start_angle_rad = 0.f;
        p.bShortScan = false;
        p.angle_list.resize(views);
        for (int i = 0; i < views; ++i)
            p.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / views;
        return p;
    }

    SCBCTParams makeFdkSmokeParams()
    {
        SCBCTParams p = makeSmallParams();
        // FDK 分包回归只验证在线状态和 chunk 边界；保持最小已验证尺寸，避免
        // 测试本身把 FFT 工作区扩张成性能/显存压力测试。
        p.iPU = 32; p.iPV = 24;
        p.iVX = 16; p.iVY = 16; p.iVZ = 12;
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

// 新框架的最小闭环：圆轨迹 operator 生成投影，随后 BP operator 消费同一块
// 调用方拥有的设备缓冲。它覆盖了无 runner 的参数流、数据流和累加语义。
int main_operator_roundtrip_smoke()
{
    const SCBCTParams p = makeSmallParams();
    const size_t volume_n = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t sino_n = static_cast<size_t>(p.iPAng) * p.iPU * p.iPV;
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
    const SCBCTParams p = makeSmallParams();
    const size_t volume_n = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t sino_n = static_cast<size_t>(p.iPAng) * p.iPU * p.iPV;
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
    const SCBCTParams p = makeSmallParams();
    const size_t volume_n = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t sino_n = static_cast<size_t>(p.iPAng) * p.iPU * p.iPV;
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
    const SCBCTParams p = makeFdkSmokeParams();
    const size_t view_n = static_cast<size_t>(p.iPU) * p.iPV;
    const size_t sino_n = static_cast<size_t>(p.iPAng) * view_n;
    const size_t volume_n = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    std::vector<float> h_projection(sino_n);
    for (size_t i = 0; i < sino_n; ++i)
        h_projection[i] = 0.02f + static_cast<float>((i * 17) % 31) * 0.001f;

    cudaStream_t stream = nullptr;
    float* d_full = nullptr; float* d_split = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create stream") &&
        checkCuda(cudaMalloc(&d_full, volume_n * sizeof(float)), "allocate full volume") &&
        checkCuda(cudaMalloc(&d_split, volume_n * sizeof(float)), "allocate split volume");
    FdkPipeline full, split;
    ok = ok && full.prepareWithAngles(p, p.angle_list, 4, stream) &&
        split.prepareWithAngles(p, p.angle_list, 4, stream);
    if (ok) {
        const FdkProjectionBatch all{ h_projection.data(), nullptr, nullptr, p.iPAng };
        const int first_count = 5;
        const FdkProjectionBatch first{ h_projection.data(), nullptr, nullptr, first_count };
        const FdkProjectionBatch second{ h_projection.data() + first_count * view_n,
            nullptr, nullptr, p.iPAng - first_count };
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
    const SCBCTParams p = makeSmallParams();
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
    const SCBCTParams p = makeFdkSmokeParams();
    const size_t projection_count = static_cast<size_t>(p.iPU) * p.iPV * p.iPAng;
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    std::vector<float> projection(projection_count, 0.03f);
    Mem::MemoryController memory;
    auto device_volume = memory.allocateDevice3D<float>(
        p.iVX, p.iVY, p.iVZ, 0, false);
    auto host_volume = memory.allocateCpu3D<float>(
        p.iVX, p.iVY, p.iVZ, false);

    cudaStream_t stream = nullptr;
    bool ok = checkCuda(cudaStreamCreate(&stream), "create sync FDK stream");
    FdkPipeline pipeline;
    ok = ok && pipeline.prepareWithAngles(p, p.angle_list, 4, stream);
    if (ok) {
        const FdkProjectionBatch batch{ projection.data(), nullptr, nullptr, p.iPAng };
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
