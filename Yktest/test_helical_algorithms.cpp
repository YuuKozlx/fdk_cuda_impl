#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHelicalIcdReconstructor.hpp"
#include "Heli/wfbp/YkWfbpPipeline.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

SHeliCTParam makeHelicalParams()
{
    SHeliCTParam h{};
    h.iPU = 24; h.iPV = 16;
    h.iVX = 12; h.iVY = 12; h.iVZ = 10;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 1.f; h.vox_y_mm = 1.f; h.vox_z_mm = 1.f;
    h.SID = 80.f; h.SDD = 160.f;
    h.pitch_mm = 4.f; h.start_z_mm = -2.f;
    h.views_per_rot = 12;
    h.angle_list.resize(18);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / h.views_per_rot;
    return h;
}

SCBCTParams toCbct(const SHeliCTParam& h)
{
    SCBCTParams p{};
    p.angle_list = h.angle_list;
    p.iPU = h.iPU; p.iPV = h.iPV;
    p.iPAng = static_cast<int>(h.angle_list.size()); p.iPAngTotal = p.iPAng;
    p.du_mm = h.du_mm; p.dv_mm = h.dv_mm;
    p.SID = h.SID; p.SDD = h.SDD;
    p.iVX = h.iVX; p.iVY = h.iVY; p.iVZ = h.iVZ;
    p.vox_x_mm = h.vox_x_mm; p.vox_y_mm = h.vox_y_mm; p.vox_z_mm = h.vox_z_mm;
    p.scan_range_rad = h.angle_list.back() - h.angle_list.front();
    return p;
}

double squaredNorm(const std::vector<float>& values)
{
    double result = 0.0;
    for (float value : values) result += static_cast<double>(value) * value;
    return result;
}

} // namespace

int main_helical_icd_smoke()
{
    const SHeliCTParam h = makeHelicalParams();
    const SCBCTParams p = toCbct(h);
    std::vector<SConeProjGeomVec> geometry;
    build_helical_vec_geometry(geometry, h);
    const auto truth = TestPhantom::makeBasic(p);
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t projection_count = static_cast<size_t>(p.iPAng) * p.iPU * p.iPV;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
    auto d_recon = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
    auto d_measured = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
    auto d_forward = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
    cudaMemcpyAsync(d_truth.data(), truth.data(), volume_count * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_recon.data(), 0, volume_count * sizeof(float), stream);

    ForwardOperatorAdapter fp;
    bool ok = fp.init(p, geometry, ETask::FP_Joseph, 0, stream) &&
        fp.run(d_truth.data(), p, d_measured.data(), stream);
    Helical::IcdConfig config{};
    config.iterations = 4;
    config.relaxation = 0.7f;
    config.regularization = 2e-3f;
    config.bp_task = ETask::BP_Joseph;
    Helical::IcdReconstructor icd;
    ok = ok && icd.prepare(p, geometry, config, stream) &&
        icd.reconstruct(d_measured.data(), d_recon.data()) &&
        fp.run(d_recon.data(), p, d_forward.data(), stream);
    cudaStreamSynchronize(stream);

    std::vector<float> measured(projection_count);
    std::vector<float> forward(projection_count);
    std::vector<float> recon(volume_count);
    cudaMemcpy(measured.data(), d_measured.data(), projection_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(forward.data(), d_forward.data(), projection_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(recon.data(), d_recon.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < projection_count; ++i) forward[i] = measured[i] - forward[i];

    bool finite_nonzero = false;
    for (float value : recon) {
        if (!std::isfinite(value)) ok = false;
        finite_nonzero = finite_nonzero || std::fabs(value) > 1e-8f;
    }
    const double initial_residual = squaredNorm(measured);
    const double final_residual = squaredNorm(forward);
    ok = ok && finite_nonzero && final_residual < initial_residual;
    std::printf("Helical ICD: residual %.8e -> %.8e, %s\n",
        initial_residual, final_residual, ok ? "PASS" : "FAIL");

    icd.release();
    fp.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_helical_wfbp_smoke()
{
    SHeliCTParam h = makeHelicalParams();
    // wFBP 的冗余归一化至少需要一整圈，并在首尾留出插值余量。
    h.views_per_rot = 24;
    h.angle_list.resize(72);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / h.views_per_rot;
    h.pitch_mm = 4.f;
    h.start_z_mm = -6.f;
    h.offsetU_mm = 0.35f;
    const SCBCTParams p = toCbct(h);
    std::vector<SConeProjGeomVec> geometry;
    build_helical_vec_geometry(geometry, h);
    const auto truth = TestPhantom::makeBasic(p);
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
    auto d_recon_flat = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
    auto d_recon_arc = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
    cudaMemcpyAsync(d_truth.data(), truth.data(), volume_count * sizeof(float),
        cudaMemcpyHostToDevice, stream);

    ForwardOperatorAdapter fp;
    Helical::Wfbp::Config flat_config{};
    flat_config.input_detector = Helical::Wfbp::EInputDetector::FlatPanel;
    flat_config.channel_oversampling = 2;
    Helical::Wfbp::Pipeline flat_wfbp;
    bool ok = fp.init(p, geometry, ETask::FP_Joseph, 0, stream) &&
        fp.run(d_truth.data(), p, d_projection.data(), stream) &&
        flat_wfbp.prepare(h, flat_config, stream) &&
        flat_wfbp.reconstruct(d_projection.data(), d_recon_flat.data());

    // The arc entry must reproduce the same result when fed the exact virtual
    // arc generated by the flat entry. This checks that flat-to-arc is a real
    // pipeline stage rather than an implicit approximation in rebinning.
    Helical::Wfbp::Config arc_config = flat_config;
    arc_config.input_detector = Helical::Wfbp::EInputDetector::EquiangularArc;
    arc_config.arc_channel_angle_step_rad = flat_wfbp.geometry().fan_angle_step;
    arc_config.arc_principal_channel = flat_wfbp.geometry().central_channel;
    Helical::Wfbp::Pipeline arc_wfbp;
    ok = ok && flat_wfbp.arcProjectionData() &&
        arc_wfbp.prepare(h, arc_config, stream) &&
        arc_wfbp.reconstruct(flat_wfbp.arcProjectionData(), d_recon_arc.data());
    cudaStreamSynchronize(stream);

    const size_t rebin_count = static_cast<size_t>(flat_wfbp.geometry().views) *
        flat_wfbp.geometry().rows * flat_wfbp.geometry().output_channels;
    std::vector<float> recon(volume_count);
    std::vector<float> recon_arc(volume_count);
    std::vector<float> rebin_flat(rebin_count);
    std::vector<float> rebin_arc(rebin_count);
    cudaMemcpy(recon.data(), d_recon_flat.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(recon_arc.data(), d_recon_arc.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(rebin_flat.data(), flat_wfbp.rebinnedData(), rebin_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(rebin_arc.data(), arc_wfbp.rebinnedData(), rebin_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    double energy = 0.0;
    double truth_energy = 0.0;
    double correlation_sum = 0.0;
    float maximum = 0.f;
    float volume_path_diff = 0.f;
    for (size_t i = 0; i < recon.size(); ++i) {
        const float value = recon[i];
        ok = ok && std::isfinite(value);
        energy += static_cast<double>(value) * value;
        truth_energy += static_cast<double>(truth[i]) * truth[i];
        correlation_sum += static_cast<double>(value) * truth[i];
        maximum = std::max(maximum, std::fabs(value));
        volume_path_diff = std::max(volume_path_diff,
            std::fabs(value - recon_arc[i]));
    }
    float rebin_path_diff = 0.f;
    for (size_t i = 0; i < rebin_count; ++i)
        rebin_path_diff = std::max(rebin_path_diff,
            std::fabs(rebin_flat[i] - rebin_arc[i]));
    const double correlation = correlation_sum /
        std::sqrt(std::max(energy * truth_energy, 1e-30));
    ok = ok && energy > 1e-12 && maximum > 1e-7f && correlation > 0.05 &&
        rebin_path_diff < 1e-6f && volume_path_diff < 1e-6f;
    std::printf("Helical wFBP: energy %.8e, max %.8e, corr %.6f, "
        "path diff(rebin/vol) %.3e/%.3e, %s\n", energy, maximum, correlation,
        rebin_path_diff, volume_path_diff, ok ? "PASS" : "FAIL");

    arc_wfbp.release();
    flat_wfbp.release();
    fp.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
