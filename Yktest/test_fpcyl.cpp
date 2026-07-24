#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "CylFpBp/YkCylFpBpGeometry.hpp"
#include "CylFpBp/YkCylFpBpOperator.hpp"
#include "Heli/wfbp/YkWfbpPipeline.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

SCBCTParams toCbct(const SHeliCTParam& h)
{
    SCBCTParams p{};
    p.angle_list = h.angle_list;
    p.iPU = h.iPU; p.iPV = h.iPV;
    p.iPAng = static_cast<int>(h.angle_list.size()); p.iPAngTotal = p.iPAng;
    p.du_mm = h.du_mm; p.dv_mm = h.dv_mm;
    p.offsetU_mm = h.offsetU_mm; p.offsetV_mm = h.offsetV_mm;
    p.SID = h.SID; p.SDD = h.SDD;
    p.iVX = h.iVX; p.iVY = h.iVY; p.iVZ = h.iVZ;
    p.vox_x_mm = h.vox_x_mm; p.vox_y_mm = h.vox_y_mm;
    p.vox_z_mm = h.vox_z_mm;
    p.scan_range_rad = h.angle_list.back() - h.angle_list.front();
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
    double nrmse = 0.0;
    float scale = 0.f;
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
    result.scale = yy > 1e-30 ? static_cast<float>(xy / yy) : 0.f;
    double error = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double d = truth[i] - result.scale * recon[i];
        error += d * d;
    }
    result.correlation = xy / std::sqrt(std::max(xx * yy, 1e-30));
    result.nrmse = std::sqrt(error / std::max(xx, 1e-30));
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

    const auto geometry = CylFpBp::buildFreeCtArcGeometry(h);
    CylFpBp::Operator op;
    const bool launched = op.prepare(volumeGeometry(h), h.iPU, h.iPV, geometry) &&
        op.forward(d_x.data(), d_ax.data(), stream) &&
        op.backproject(d_y.data(), d_at_y.data(), stream);
    cudaStreamSynchronize(stream);

    std::vector<float> ax(projection_count), at_y(volume_count);
    cudaMemcpy(ax.data(), d_ax.data(), projection_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(at_y.data(), d_at_y.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    double lhs = 0.0, rhs = 0.0;
    for (size_t i = 0; i < projection_count; ++i)
        lhs += static_cast<double>(ax[i]) * y[i];
    for (size_t i = 0; i < volume_count; ++i)
        rhs += static_cast<double>(x[i]) * at_y[i];
    const double relative_error = std::fabs(lhs - rhs) /
        std::max({ std::fabs(lhs), std::fabs(rhs), 1e-30 });
    const bool ok = launched && std::isfinite(relative_error) && relative_error < 2e-5;
    std::printf("CylFpBp matched FP/BP: <Ax,y>=%.9e <x,ATy>=%.9e rel=%.3e %s\n",
        lhs, rhs, relative_error, ok ? "PASS" : "FAIL");
    op.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_fpcyl_wfbp_comparison()
{
    struct TestCase { const char* name; bool catphan; float offset_v; };
    const TestCase cases[] = {
        { "basic-v0", false, 0.f },
        { "basic-v+1.5", false, 1.5f },
        { "catphan-v0", true, 0.f },
        { "catphan-v-1.5", true, -1.5f }
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
        const SCBCTParams p = toCbct(h);
        Result result{};
        result.name = test_case.name;
        result.truth = test_case.catphan ? TestPhantom::makeCatphanLike(p)
                                         : TestPhantom::makeBasic(p);
        result.recon.resize(volume_count);
        result.error.resize(volume_count);
        cudaMemcpyAsync(d_truth.data(), result.truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream);

        const float arc_step = 2.f * atanf(0.5f * h.du_mm / h.SDD);
        const auto geometry = CylFpBp::buildFreeCtArcGeometry(h, arc_step);
        CylFpBp::Operator projector;
        CylFpBp::Config projector_config{};
        projector_config.samples_per_voxel = 2.f;
        Helical::Wfbp::Config config{};
        config.input_detector = Helical::Wfbp::EInputDetector::EquiangularArc;
        config.arc_channel_angle_step_rad = arc_step;
        config.arc_principal_channel = 0.5f * (h.iPU - 1) - h.offsetU_mm / h.du_mm;
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
            case_ok = projector.forward(d_truth.data(), d_projection.data(), stream);
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
                result.error[i] = result.metrics.scale * result.recon[i] - result.truth[i];
            case_ok = result.metrics.correlation > 0.80 && result.metrics.nrmse < 0.60;
        }
        std::printf("CylFpBp -> wFBP %-15s corr %.6f NRMSE %.6f scale %.6f "
            "FP %.3f ms recon %.3f ms %s\n", result.name.c_str(),
            result.metrics.correlation, result.metrics.nrmse, result.metrics.scale,
            result.fp_ms, result.recon_ms, case_ok ? "PASS" : "FAIL");
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
            base.iVZ / 2, result.metrics.scale, 0.f, maximum, false });
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
