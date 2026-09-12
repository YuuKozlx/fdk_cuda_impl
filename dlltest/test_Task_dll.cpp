// DLL smoke test for the Session API.  It uses a tiny synthetic acquisition,
// so it runs without external raw-data files.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include "YKCBCT/interface/IYkSession.hpp"

namespace {
constexpr int kNu = 32, kNv = 24, kNx = 16, kNy = 16, kNz = 12;
constexpr int kSmokeAngles = 2;
constexpr int kAnalyticAngles = 32;
constexpr int kWfbpAngles = 120;

void fillProjection(std::vector<float>& projection, int views)
{
    const size_t view_size = size_t(kNu) * kNv;
    projection.resize(size_t(views) * view_size);
    for (int view = 0; view < views; ++view) {
        for (int v = 0; v < kNv; ++v) {
            for (int u = 0; u < kNu; ++u) {
                // 非零且可复现的输入，确保解析滤波路径确实写入重建体。
                const float fu = static_cast<float>(u + 1) / kNu;
                const float fv = static_cast<float>(v + 1) / kNv;
                const float fa = static_cast<float>(view + 1) / views;
                projection[size_t(view) * view_size + size_t(v) * kNu + u] =
                    0.25f + 0.5f * fu +
                    0.125f * std::sin(6.2831853f * fa) * fv;
            }
        }
    }
}

bool finiteAndNonzeroVolume(const float* host, size_t count, const char* name)
{
    double energy = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(host[i])) {
            std::fprintf(stderr, "%s produced a non-finite voxel at %zu\n", name, i);
            return false;
        }
        energy += std::abs(static_cast<double>(host[i]));
    }
    if (!(energy > 0.0)) {
        std::fprintf(stderr, "%s produced an all-zero volume\n", name);
        return false;
    }
    return true;
}

YK::SSystemSpec makeDesc(YK::EPipeline pipeline) {
    YK::SSystemSpec d{};
    d.geometry.detector = YK::EDetectorKind::Flat;
    d.geometry.trajectory = YK::ETrajectoryKind::Circular;
    d.geometry.circular.total_views = kSmokeAngles;
    d.geometry.circular.views_per_turn = kSmokeAngles;
    d.geometry.circular.sid_mm = 500.f;
    d.geometry.circular.sdd_mm = 1000.f;
    d.geometry.flat_detector.channels = kNu;
    d.geometry.flat_detector.rows = kNv;
    d.geometry.flat_detector.channel_size_mm = 1.f;
    d.geometry.flat_detector.row_size_mm = 1.f;
    d.geometry.volume = {kNx, kNy, kNz, 1.f, 1.f, 1.f,
        make_float3(0.f, 0.f, 0.f)};
    d.reconstruction.pipeline = pipeline;
    d.reconstruction.forward_projector = YK::ETask::FP_Joseph;
    d.reconstruction.back_projector = YK::ETask::BP_Joseph_v3;
    return d;
}

YK::SSystemSpec makeCylDesc(YK::EPipeline pipeline) {
    auto d = makeDesc(pipeline);
    d.geometry.detector = YK::EDetectorKind::Cylindrical;
    d.geometry.cylindrical_detector.channels = kNu;
    d.geometry.cylindrical_detector.rows = kNv;
    d.geometry.cylindrical_detector.channel_arc_mm = 1.f;
    d.geometry.cylindrical_detector.row_size_mm = 1.f;
    d.geometry.cylindrical_detector.curvature_radius_mm = 1000.f;
    d.reconstruction.iterative.iterations = 1;
    d.reconstruction.iterative.subsets = 1;
    return d;
}

YK::SSystemSpec makeHelical(YK::SSystemSpec d) {
    d.geometry.trajectory = YK::ETrajectoryKind::Helical;
    d.geometry.helical.total_views = kSmokeAngles;
    d.geometry.helical.views_per_turn = kSmokeAngles;
    d.geometry.helical.sid_mm = 500.f;
    d.geometry.helical.sdd_mm = 1000.f;
    d.geometry.helical.start_z_mm = -1.f;
    d.geometry.helical.pitch_mm_per_turn = 2.f;
    return d;
}

bool cudaOk(cudaError_t e, const char* where) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
    return false;
}

bool testForwardProjection(const YK::SSystemSpec& desc, const char* name) {
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    if (!session->initialize(desc)) {
        std::fprintf(stderr, "%s initialize: %s\n", name,
            session->lastErrorMessage());
        YK::SessionFactory::destroy(session);
        return false;
    }
    const size_t volN = size_t(kNx) * kNy * kNz;
    const int views = desc.geometry.trajectory == YK::ETrajectoryKind::Helical
        ? desc.geometry.helical.total_views : desc.geometry.circular.total_views;
    const size_t sinoN = size_t(views) * kNu * kNv;
    float *dVol = nullptr, *dSino = nullptr;
    bool ok = cudaOk(cudaMalloc(reinterpret_cast<void**>(&dVol), volN * sizeof(float)), "cudaMalloc(volume)") &&
              cudaOk(cudaMalloc(reinterpret_cast<void**>(&dSino), sinoN * sizeof(float)), "cudaMalloc(sinogram)") &&
              cudaOk(cudaMemset(dVol, 0, volN * sizeof(float)), "cudaMemset(volume)");
    if (ok) {
        YK::SExecutionRequest r{};
        r.view_count = views;
        r.volume = { dVol, YK::EMemoryLocation::Device, volN };
        r.projection = { dSino, YK::EMemoryLocation::Device, sinoN };
        ok = session->execute(r) && cudaOk(cudaDeviceSynchronize(), name);
    }
    cudaFree(dSino); cudaFree(dVol);
    session->release(); YK::SessionFactory::destroy(session);
    return ok;
}

bool testValidationError() {
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    auto desc = makeDesc(YK::EPipeline::ForwardProjection);
    desc.struct_size = sizeof(std::uint32_t);
    const bool rejected = !session->initialize(desc) &&
        session->lastError() == YK::EApiErrorCode::InvalidConfig &&
        session->lastErrorMessage()[0] != '\0';
    YK::SessionFactory::destroy(session);
    return rejected;
}

bool testExecutionValidation()
{
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    auto desc = makeDesc(YK::EPipeline::ForwardProjection);
    bool ok = session->initialize(desc);
    YK::SExecutionRequest request{};
    request.projection = {reinterpret_cast<float*>(1),
        YK::EMemoryLocation::Device, 1};
    request.volume = {reinterpret_cast<float*>(1),
        YK::EMemoryLocation::Device, 1};
    ok = ok && !session->execute(request) &&
        session->lastError() == YK::EApiErrorCode::InvalidBuffer;

    request.projection.element_count = size_t(kSmokeAngles) * kNu * kNv;
    request.volume.element_count = size_t(kNx) * kNy * kNz;
    request.view_offset = 1;
    request.view_count = 1;
    ok = ok && !session->execute(request) &&
        session->lastError() == YK::EApiErrorCode::StreamingStateError;
    YK::SessionFactory::destroy(session);
    return ok;
}

bool testUnderTestPolicy() {
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    auto desc = makeDesc(YK::EPipeline::CFDK);
    bool ok = !session->initialize(desc) &&
        session->lastError() == YK::EApiErrorCode::AlgorithmUnderTest;
    desc = makeCylDesc(YK::EPipeline::FDK);
    ok = ok && !session->initialize(desc) &&
        session->lastError() == YK::EApiErrorCode::AlgorithmUnderTest;
    YK::SessionFactory::destroy(session);
    return ok;
}

bool testIterative(YK::SSystemSpec desc, const char* name) {
    desc.reconstruction.pipeline = YK::EPipeline::SIRT;
    desc.reconstruction.iterative.iterations = 1;
    desc.reconstruction.iterative.subsets = 1;
    desc.reconstruction.iterative.relaxation = 0.2f;
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    if (!session->initialize(desc)) {
        std::fprintf(stderr, "%s initialize: %s\n", name,
            session->lastErrorMessage());
        YK::SessionFactory::destroy(session);
        return false;
    }
    const size_t volN = size_t(kNx) * kNy * kNz;
    const int views = desc.geometry.trajectory == YK::ETrajectoryKind::Helical
        ? desc.geometry.helical.total_views : desc.geometry.circular.total_views;
    const size_t sinoN = size_t(views) * kNu * kNv;
    float *dVol = nullptr, *dSino = nullptr;
    bool ok = cudaOk(cudaMalloc(reinterpret_cast<void**>(&dVol), volN * sizeof(float)),
                  "cudaMalloc(iter volume)") &&
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&dSino), sinoN * sizeof(float)),
            "cudaMalloc(iter projection)") &&
        cudaOk(cudaMemset(dVol, 0, volN * sizeof(float)), "cudaMemset(iter volume)") &&
        cudaOk(cudaMemset(dSino, 0, sinoN * sizeof(float)), "cudaMemset(iter projection)");
    if (ok) {
        YK::SExecutionRequest r{};
        r.view_count = views;
        r.volume = {dVol, YK::EMemoryLocation::Device, volN};
        r.projection = {dSino, YK::EMemoryLocation::Device, sinoN};
        ok = session->execute(r);
        if (!ok) std::fprintf(stderr, "%s execute: %s\n", name,
            session->lastErrorMessage());
    }
    cudaFree(dSino); cudaFree(dVol);
    YK::SessionFactory::destroy(session);
    return ok;
}

bool testSessionResetReuse()
{
    auto desc = makeDesc(YK::EPipeline::SIRT);
    desc.geometry.circular.total_views = kAnalyticAngles;
    desc.geometry.circular.views_per_turn = kAnalyticAngles;
    desc.reconstruction.iterative.iterations = 2;
    desc.reconstruction.iterative.subsets = 1;
    desc.reconstruction.iterative.relaxation = 0.2f;
    auto* session = YK::SessionFactory::create();
    if (!session || !session->initialize(desc)) {
        if (session) std::fprintf(stderr, "Session reset initialize: %s\n",
            session->lastErrorMessage());
        YK::SessionFactory::destroy(session);
        return false;
    }

    const size_t volumeN = size_t(kNx) * kNy * kNz;
    const size_t projectionN = size_t(kAnalyticAngles) * kNu * kNv;
    std::vector<float> hProjection;
    fillProjection(hProjection, kAnalyticAngles);
    float* dProjection = nullptr;
    float* dVolume = nullptr;
    bool ok = cudaOk(cudaMalloc(reinterpret_cast<void**>(&dProjection),
            projectionN * sizeof(float)), "cudaMalloc(reset projection)") &&
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&dVolume),
            volumeN * sizeof(float)), "cudaMalloc(reset volume)") &&
        cudaOk(cudaMemcpy(dProjection, hProjection.data(),
            projectionN * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy(reset projection)") &&
        cudaOk(cudaMemset(dVolume, 0, volumeN * sizeof(float)),
            "cudaMemset(reset volume)");

    std::vector<float> first(volumeN), second(volumeN);
    if (ok) {
        YK::SExecutionRequest request{};
        request.view_count = kAnalyticAngles;
        request.projection = {dProjection, YK::EMemoryLocation::Device, projectionN};
        request.volume = {dVolume, YK::EMemoryLocation::Device, volumeN};
        ok = session->execute(request) &&
            cudaOk(cudaDeviceSynchronize(), "reset first execute") &&
            cudaOk(cudaMemcpy(first.data(), dVolume, volumeN * sizeof(float),
                cudaMemcpyDeviceToHost), "cudaMemcpy(reset first)");
        if (ok) {
            session->reset();
            ok = cudaOk(cudaMemset(dVolume, 0, volumeN * sizeof(float)),
                "cudaMemset(reset second volume)") &&
                session->execute(request) &&
                cudaOk(cudaDeviceSynchronize(), "reset second execute") &&
                cudaOk(cudaMemcpy(second.data(), dVolume, volumeN * sizeof(float),
                    cudaMemcpyDeviceToHost), "cudaMemcpy(reset second)");
        }
    }
    double maxDiff = 0.0;
    if (ok) {
        for (size_t i = 0; i < volumeN; ++i)
            maxDiff = std::max(maxDiff, std::abs(static_cast<double>(first[i]) - second[i]));
        ok = maxDiff <= 1e-5;
        if (!ok) std::fprintf(stderr, "Session reset changed result: maxDiff=%g\n", maxDiff);
    }
    cudaFree(dProjection);
    cudaFree(dVolume);
    YK::SessionFactory::destroy(session);
    return ok;
}

YK::SSystemSpec makeAnalyticDesc(YK::EPipeline pipeline, int views)
{
    auto d = makeDesc(pipeline);
    d.geometry.circular.total_views = views;
    d.geometry.circular.views_per_turn = views;
    d.reconstruction.fdk.filter = YK::EFdkFilter::RamLak;
    return d;
}

YK::SSystemSpec makeWfbpDesc()
{
    auto d = makeCylDesc(YK::EPipeline::WFBP);
    d.geometry.trajectory = YK::ETrajectoryKind::Helical;
    d.geometry.helical.total_views = kWfbpAngles;
    d.geometry.helical.views_per_turn = kWfbpAngles;
    d.geometry.helical.sid_mm = 500.f;
    d.geometry.helical.sdd_mm = 1000.f;
    d.geometry.helical.start_z_mm = -1.f;
    d.geometry.helical.pitch_mm_per_turn = 2.f;
    d.reconstruction.wfbp.input_detector =
        YK::EWfbpInputDetectorSpec::CylindricalArc;
    return d;
}

bool testReconstruction(const YK::SSystemSpec& desc, int views,
    const char* name, bool host_projection = false)
{
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    if (!session->initialize(desc)) {
        std::fprintf(stderr, "%s initialize: %s\n", name,
            session->lastErrorMessage());
        YK::SessionFactory::destroy(session);
        return false;
    }

    std::vector<float> hProjection;
    fillProjection(hProjection, views);
    const size_t projectionN = hProjection.size();
    const size_t volumeN = size_t(kNx) * kNy * kNz;
    float *dProjection = nullptr, *dVolume = nullptr;
    std::vector<float> hVolume(volumeN, 0.f);
    bool ok = cudaOk(cudaMalloc(reinterpret_cast<void**>(&dProjection),
            projectionN * sizeof(float)), "cudaMalloc(reconstruction projection)") &&
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&dVolume),
            volumeN * sizeof(float)), "cudaMalloc(reconstruction volume)") &&
        cudaOk(cudaMemcpy(dProjection, hProjection.data(),
            projectionN * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy(reconstruction projection)") &&
        cudaOk(cudaMemset(dVolume, 0, volumeN * sizeof(float)),
            "cudaMemset(reconstruction volume)");
    if (ok) {
        YK::SExecutionRequest request{};
        request.view_count = views;
        request.projection = {host_projection ? hProjection.data() : dProjection,
            host_projection ? YK::EMemoryLocation::Host : YK::EMemoryLocation::Device,
            projectionN};
        request.volume = {dVolume, YK::EMemoryLocation::Device, volumeN};
        ok = session->execute(request);
        if (!ok)
            std::fprintf(stderr, "%s execute: %s\n", name,
                session->lastErrorMessage());
    }
    if (ok)
        ok = cudaOk(cudaDeviceSynchronize(), name) &&
            cudaOk(cudaMemcpy(hVolume.data(), dVolume,
                volumeN * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy(reconstruction volume)") &&
            finiteAndNonzeroVolume(hVolume.data(), volumeN, name);
    cudaFree(dProjection);
    cudaFree(dVolume);
    YK::SessionFactory::destroy(session);
    return ok;
}

bool testFdkOfflineVsStreaming()
{
    const auto desc = makeAnalyticDesc(YK::EPipeline::FDK, kAnalyticAngles);
    auto* offline = YK::SessionFactory::create();
    auto* streaming = YK::SessionFactory::create();
    if (!offline || !streaming) {
        YK::SessionFactory::destroy(offline);
        YK::SessionFactory::destroy(streaming);
        return false;
    }
    bool ok = offline->initialize(desc) && streaming->initialize(desc);
    std::vector<float> projection;
    fillProjection(projection, kAnalyticAngles);
    const size_t view_elements = size_t(kNu) * kNv;
    const size_t volume_elements = size_t(kNx) * kNy * kNz;
    float* d_offline = nullptr;
    float* d_streaming = nullptr;
    ok = ok &&
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_offline),
            volume_elements * sizeof(float)), "cudaMalloc(FDK offline volume)") &&
        cudaOk(cudaMalloc(reinterpret_cast<void**>(&d_streaming),
            volume_elements * sizeof(float)), "cudaMalloc(FDK streaming volume)");
    if (ok) {
        YK::SExecutionRequest request{};
        request.view_count = kAnalyticAngles;
        request.projection = {projection.data(), YK::EMemoryLocation::Host,
            projection.size()};
        request.volume = {d_offline, YK::EMemoryLocation::Device,
            volume_elements};
        ok = offline->execute(request);
    }
    if (ok) {
        constexpr int kFirstBatch = 13;
        YK::SExecutionRequest first{};
        first.view_offset = 0;
        first.view_count = kFirstBatch;
        first.clear_output = true;
        first.projection = {projection.data(), YK::EMemoryLocation::Host,
            size_t(kFirstBatch) * view_elements};
        first.volume = {d_streaming, YK::EMemoryLocation::Device,
            volume_elements};
        ok = streaming->execute(first);

        YK::SExecutionRequest second{};
        second.view_offset = kFirstBatch;
        second.view_count = kAnalyticAngles - kFirstBatch;
        second.clear_output = false;
        second.projection = {projection.data() + size_t(kFirstBatch) * view_elements,
            YK::EMemoryLocation::Host,
            size_t(kAnalyticAngles - kFirstBatch) * view_elements};
        second.volume = {d_streaming, YK::EMemoryLocation::Device,
            volume_elements};
        ok = ok && streaming->execute(second);
    }
    if (ok) {
        std::vector<float> a(volume_elements), b(volume_elements);
        ok = cudaOk(cudaMemcpy(a.data(), d_offline,
                volume_elements * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy(FDK offline volume)") &&
            cudaOk(cudaMemcpy(b.data(), d_streaming,
                volume_elements * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy(FDK streaming volume)");
        double max_diff = 0.0;
        for (size_t i = 0; ok && i < volume_elements; ++i)
            max_diff = std::max(max_diff,
                std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        // 两条路径使用相同的批次处理内核，结果应在单精度累加误差内一致。
        ok = ok && max_diff <= 1e-5;
        if (!ok)
            std::fprintf(stderr, "FDK offline/streaming max difference: %.9g\n",
                max_diff);
    }
    cudaFree(d_offline);
    cudaFree(d_streaming);
    YK::SessionFactory::destroy(offline);
    YK::SessionFactory::destroy(streaming);
    return ok;
}

#if !YKCBCT_HAS_HELICAL
bool testWfbpRejected()
{
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    const bool rejected = !session->initialize(makeWfbpDesc()) &&
        session->lastError() == YK::EApiErrorCode::UnsupportedCombination;
    if (!rejected) {
        std::fprintf(stderr, "wFBP without helical backend: expected explicit rejection, got %s\n",
            session->lastErrorMessage());
    }
    YK::SessionFactory::destroy(session);
    return rejected;
}
#endif

bool testFdkPolicies() {
    auto* session = YK::SessionFactory::create();
    if (!session) return false;
    auto desc = makeDesc(YK::EPipeline::FDK);
    // Flat FDK 需要至少一整圈的几何；这里只验证初始化策略和明确错误，
    // 不把两视图的欠采样结果当作算法数值回归。
    const bool fdkRejected = !session->initialize(makeHelical(desc)) &&
        session->lastError() == YK::EApiErrorCode::AlgorithmUnderTest;
    const bool xfdkRejected = !session->initialize(
        makeDesc(YK::EPipeline::CFDK)) &&
        session->lastError() == YK::EApiErrorCode::AlgorithmUnderTest;
    YK::SessionFactory::destroy(session);
    return fdkRejected && xfdkRejected;
}

bool testOffsetNormalization()
{
    // XFDK 和柱面解析 FDK 当前只消费规范采集几何。公共 Session 应复制
    // 并归零不支持的采集 offset，而不是拒绝整个算法或静默让后端丢失参数。
    auto* session = YK::SessionFactory::create();
    if (!session) return false;

    auto xfdk = makeAnalyticDesc(YK::EPipeline::XFDK, kAnalyticAngles);
    xfdk.geometry.circular.source_offset_mm = make_float3(1.f, 0.f, 0.f);
    xfdk.geometry.flat_detector.pose.offset_unv_mm = make_float3(2.f, 3.f, 4.f);
    const bool xfdkOk = session->initialize(xfdk);

    auto cyl = makeCylDesc(YK::EPipeline::CylAnalyticFDK);
    cyl.geometry.circular.total_views = kAnalyticAngles;
    cyl.geometry.circular.views_per_turn = kAnalyticAngles;
    cyl.geometry.cylindrical_detector.curvature_radius_mm = 1000.f;
    cyl.geometry.circular.source_offset_mm = make_float3(0.f, 0.f, 1.f);
    cyl.geometry.cylindrical_detector.pose.offset_unv_mm = make_float3(2.f, 0.f, 0.f);
    const bool cylOk = session->initialize(cyl);

    if (!xfdkOk || !cylOk)
        std::fprintf(stderr, "offset normalization initialize failed: %s\n",
            session->lastErrorMessage());
    YK::SessionFactory::destroy(session);
    return xfdkOk && cylOk;
}

}

int main() {
    const bool flatCircularFp = testForwardProjection(
        makeDesc(YK::EPipeline::ForwardProjection), "Flat circular FP");
    const bool flatHelicalFp = testForwardProjection(makeHelical(
        makeDesc(YK::EPipeline::ForwardProjection)), "Flat helical FP");
    const bool cylCircularFp = testForwardProjection(
        makeCylDesc(YK::EPipeline::ForwardProjection), "Cyl circular FP");
    const bool cylHelicalFp = testForwardProjection(makeHelical(
        makeCylDesc(YK::EPipeline::ForwardProjection)), "Cyl helical FP");
    const bool validationOk = testValidationError();
    const bool executionValidationOk = testExecutionValidation();
    const bool policyOk = testUnderTestPolicy();
    const bool fdkPolicyOk = testFdkPolicies();
    const bool offsetNormalizationOk = testOffsetNormalization();
    const bool flatHelicalIter = testIterative(makeHelical(
        makeDesc(YK::EPipeline::SIRT)), "Flat helical SIRT");
    const bool cylCircularIter = testIterative(
        makeCylDesc(YK::EPipeline::SIRT), "Cyl circular SIRT");
    const bool cylHelicalIter = testIterative(makeHelical(
        makeCylDesc(YK::EPipeline::SIRT)), "Cyl helical SIRT");
    const bool sessionResetReuse = testSessionResetReuse();
    const bool flatFdk = testReconstruction(
        makeAnalyticDesc(YK::EPipeline::FDK, kAnalyticAngles),
        kAnalyticAngles, "Flat circular FDK", true);
    const bool fdkStreaming = testFdkOfflineVsStreaming();
    const bool flatXfdk = testReconstruction(
        makeAnalyticDesc(YK::EPipeline::XFDK, kAnalyticAngles),
        kAnalyticAngles, "Flat circular XFDK");
    const bool cylFdk = testReconstruction(
        [] {
            auto d = makeCylDesc(YK::EPipeline::CylAnalyticFDK);
            d.geometry.circular.total_views = kAnalyticAngles;
            d.geometry.circular.views_per_turn = kAnalyticAngles;
            d.geometry.cylindrical_detector.curvature_radius_mm = 1000.f;
            return d;
        }(), kAnalyticAngles, "Cyl circular analytic FDK");
    const bool flatTigre = testReconstruction(
        makeAnalyticDesc(YK::EPipeline::TigreGradient, kAnalyticAngles),
        kAnalyticAngles, "Flat circular TIGRE gradient");
#if YKCBCT_HAS_HELICAL
    const bool wfbp = testReconstruction(makeWfbpDesc(), kWfbpAngles,
        "Cyl helical wFBP");
#else
    // 未启用螺旋后端时，初始化明确返回 UnsupportedCombination 是正确行为。
    const bool wfbp = testWfbpRejected();
#endif
    const auto status = [](bool value) { return value ? "OK" : "FAILED"; };
    std::cout << "Session DLL smoke: FlatCir=" << status(flatCircularFp)
        << " FlatHeli=" << status(flatHelicalFp)
        << " CylCir=" << status(cylCircularFp)
        << " CylHeli=" << status(cylHelicalFp)
        << " FlatHeliIter=" << status(flatHelicalIter)
        << " CylCirIter=" << status(cylCircularIter)
        << " CylHeliIter=" << status(cylHelicalIter)
        << " SessionReset=" << status(sessionResetReuse)
        << " validation=" << status(validationOk)
        << " executionValidation=" << status(executionValidationOk)
        << " policy=" << status(policyOk)
        << " fdkPolicy=" << status(fdkPolicyOk)
        << " offsetNormalization=" << status(offsetNormalizationOk)
        << " FlatFDK=" << status(flatFdk)
        << " FDKStreaming=" << status(fdkStreaming)
        << " FlatXFDK=" << status(flatXfdk)
        << " CylFDK=" << status(cylFdk)
        << " FlatTigre=" << status(flatTigre)
        << " WFBP=" << status(wfbp) << '\n';
    return flatCircularFp && flatHelicalFp && cylCircularFp && cylHelicalFp &&
        flatHelicalIter && cylCircularIter && cylHelicalIter && sessionResetReuse && validationOk &&
        executionValidationOk &&
        policyOk && fdkPolicyOk && offsetNormalizationOk && flatFdk && fdkStreaming && flatXfdk && cylFdk &&
        flatTigre && wfbp ? 0 : 1;
}
