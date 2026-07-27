#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "CylFpBp/YkCylFpBpGeometry.hpp"
#include "CylFpBp/YkCylFpBpOperator.hpp"
#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHelicalIcdReconstructor.hpp"
#include "Heli/wfbp/YkWfbpPipeline.hpp"
#include "Iter/YkParallelPwlsReconstructor.hpp"
#include "YkTestImage.hpp"
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

SVolGeom volumeGeometry(const SHeliCTParam& h)
{
    SVolGeom geometry = SVolGeom::make_centered(h.iVX, h.iVY, h.iVZ,
        h.vox_x_mm, h.vox_y_mm, h.vox_z_mm);
    geometry.center = make_float3(h.vol_offset_x_mm, h.vol_offset_y_mm,
        h.vol_offset_z_mm);
    return geometry;
}

double squaredNorm(const std::vector<float>& values)
{
    double result = 0.0;
    for (float value : values) result += static_cast<double>(value) * value;
    return result;
}

struct ReconstructionMetrics {
    double correlation = 0.0;
    double nrmse = 0.0;
    double slice_correlation = 0.0;
    float scale = 1.f;
};

ReconstructionMetrics compareReconstruction(const std::vector<float>& truth,
    const std::vector<float>& reconstruction, int nx, int ny, int nz, int slice)
{
    ReconstructionMetrics metrics{};
    double truth_norm = 0.0, recon_norm = 0.0, dot = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        truth_norm += static_cast<double>(truth[i]) * truth[i];
        recon_norm += static_cast<double>(reconstruction[i]) * reconstruction[i];
        dot += static_cast<double>(truth[i]) * reconstruction[i];
    }
    metrics.scale = recon_norm > 1e-30 ? static_cast<float>(dot / recon_norm) : 0.f;
    double error_norm = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double error = metrics.scale * reconstruction[i] - truth[i];
        error_norm += error * error;
    }
    metrics.correlation = dot / std::sqrt(std::max(truth_norm * recon_norm, 1e-30));
    metrics.nrmse = std::sqrt(error_norm / std::max(truth_norm, 1e-30));

    double slice_truth = 0.0, slice_recon = 0.0, slice_dot = 0.0;
    const size_t begin = static_cast<size_t>(slice) * nx * ny;
    const size_t end = begin + static_cast<size_t>(nx) * ny;
    for (size_t i = begin; i < end; ++i) {
        slice_truth += static_cast<double>(truth[i]) * truth[i];
        slice_recon += static_cast<double>(reconstruction[i]) * reconstruction[i];
        slice_dot += static_cast<double>(truth[i]) * reconstruction[i];
    }
    metrics.slice_correlation = slice_dot /
        std::sqrt(std::max(slice_truth * slice_recon, 1e-30));
    return metrics;
}

bool writeFloatRaw(const std::filesystem::path& path,
    const std::vector<float>& values)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

float referenceSignedAngle(float x1, float x2, float y1, float y2)
{
    const float denominator = std::hypot(x1, x2) * std::hypot(y1, y2);
    const float sine = (x1 * y2 - x2 * y1) / denominator;
    return std::asin(std::max(-1.f, std::min(1.f, sine)));
}

void referenceFocalOffsets(const Helical::Wfbp::Geometry& g, int phi_spot,
    int z_spot, float& da, float& dr)
{
    da = g.phi_spot_count == 2
        ? (phi_spot == 0 ? g.phi_ffs_shift : -g.phi_ffs_shift) : 0.f;
    dr = g.z_spot_count == 2
        ? (z_spot == 0 ? -g.z_ffs_shift : g.z_ffs_shift) : 0.f;
}

float referenceFocalBeta(const Helical::Wfbp::Geometry& g, float channel,
    float da, float dr)
{
    const float b0 = (channel - g.central_channel) * g.fan_angle_step;
    return referenceSignedAngle(-(g.sid + dr), -da,
        -(g.sdd * std::cos(b0) + dr), -(g.sdd * std::sin(b0) + da));
}

int referenceSpotIndex(const Helical::Wfbp::Geometry& g, int phi_spot, int z_spot)
{
    if (g.focal_spot_mode == Helical::Wfbp::EFocalSpotMode::PhiAndZ)
        return 2 * z_spot + phi_spot;
    if (g.focal_spot_mode == Helical::Wfbp::EFocalSpotMode::Phi) return phi_spot;
    if (g.focal_spot_mode == Helical::Wfbp::EFocalSpotMode::Z) return z_spot;
    return 0;
}

float referenceSpotSequenceSample(const std::vector<float>& raw,
    const Helical::Wfbp::Geometry& g, int sequence_output_view, int raw_row,
    int phi_spot, int z_spot, int channel)
{
    float da = 0.f, dr = 0.f;
    referenceFocalOffsets(g, phi_spot, z_spot, da, dr);
    const float beta = referenceFocalBeta(g, static_cast<float>(channel), da, dr);
    const float d_alpha = referenceSignedAngle(g.sid, 0.f, g.sid + dr, da);
    const int spot = referenceSpotIndex(g, phi_spot, z_spot);

    // 官方 p/z/a 第一阶段：先把 raw[spot+n_ffs*k] 拆成独立焦点序列，
    // 再在各自序列内按 (alpha_idx-spot)/n_ffs 做角度插值。
    const float sequence_view = sequence_output_view -
        (beta + d_alpha) / g.angle_step -
        static_cast<float>(spot) / g.focal_spot_count;
    if (sequence_view < 0.f || sequence_view > g.sequence_views - 1.f) return 0.f;
    const int v0 = std::min(static_cast<int>(std::floor(sequence_view)),
        g.sequence_views - 1);
    const int v1 = std::min(v0 + 1, g.sequence_views - 1);
    const float t = sequence_view - v0;
    const size_t stride = static_cast<size_t>(g.input_rows) * g.input_channels;
    const size_t offset = static_cast<size_t>(raw_row) * g.input_channels + channel;
    const float a = raw[static_cast<size_t>(v0 * g.focal_spot_count + spot) *
        stride + offset];
    const float b = raw[static_cast<size_t>(v1 * g.focal_spot_count + spot) *
        stride + offset];
    return a + t * (b - a);
}

float referenceFreeCtRebin(const std::vector<float>& raw,
    const Helical::Wfbp::Geometry& g, int output_view, int output_row,
    int output_channel)
{
    const int z_parity = output_row & 1;
    const int z_spot = g.z_spot_count == 1 ? 0
        : (g.reverse_row_interleave ? 1 - z_parity : z_parity);
    const int raw_row = g.z_spot_count == 1 ? output_row : output_row / 2;

    // 官方 z1/z2 与 a1/a2 第二阶段使用 r_f/r_fr(0,+/-dr) 修正目标 beta。
    float sine_beta = (output_channel - g.parallel_center) *
        (g.fan_angle_step * 0.5f);
    if (g.z_spot_count == 2) {
        float da = 0.f, dr = 0.f;
        referenceFocalOffsets(g, 0, z_spot, da, dr);
        sine_beta *= g.sid / std::fabs(g.sid + dr);
    }
    if (std::fabs(sine_beta) >= 1.f) return 0.f;
    const float target_beta = std::asin(sine_beta);

    const int intermediate_count = g.input_channels * g.phi_spot_count;
    int high = 0;
    auto beta_at = [&](int intermediate) {
        const int channel = intermediate / g.phi_spot_count;
        const int phi_spot = intermediate % g.phi_spot_count;
        float da = 0.f, dr = 0.f;
        referenceFocalOffsets(g, phi_spot, z_spot, da, dr);
        return referenceFocalBeta(g, static_cast<float>(channel), da, dr);
    };
    auto stage_one = [&](int intermediate) {
        const int channel = intermediate / g.phi_spot_count;
        const int phi_spot = intermediate % g.phi_spot_count;
        return referenceSpotSequenceSample(raw, g,
            output_view + g.add_projections, raw_row, phi_spot, z_spot, channel);
    };
    const float first_beta = beta_at(0);
    const float last_beta = beta_at(intermediate_count - 1);
    if (target_beta < first_beta - 1e-7f || target_beta > last_beta + 1e-7f)
        return 0.f;
    if (target_beta <= first_beta) return stage_one(0);
    if (target_beta >= last_beta) return stage_one(intermediate_count - 1);
    while (high < intermediate_count && beta_at(high) < target_beta) ++high;
    if (high == 0 || high == intermediate_count) return 0.f;
    const int low = high - 1;
    const float beta_low = beta_at(low);
    const float beta_high = beta_at(high);
    const float t = (target_beta - beta_low) / (beta_high - beta_low);
    const float a = stage_one(low);
    const float b = stage_one(high);
    return a + t * (b - a);
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
    // wFBP 的冗余归一化至少需要一整圈，并在首尾留出 FreeCT
    // add_projections 边界补帧。
    h.views_per_rot = 128;
    h.angle_list.resize(384);
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
    auto d_projection_v = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
    auto d_recon_v = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
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

    // V offset 不能只停留在参数校验：用同一模体和真实偏移后的 vector
    // geometry 重新生成投影，检查 principal row 已进入 wFBP 行坐标映射。
    SHeliCTParam h_v = h;
    h_v.offsetV_mm = 1.25f;
    std::vector<SConeProjGeomVec> geometry_v;
    build_helical_vec_geometry(geometry_v, h_v);
    ForwardOperatorAdapter fp_v;
    Helical::Wfbp::Pipeline wfbp_v;
    ok = ok && fp_v.init(p, geometry_v, ETask::FP_Joseph, 0, stream) &&
        fp_v.run(d_truth.data(), p, d_projection_v.data(), stream) &&
        wfbp_v.prepare(h_v, flat_config, stream) &&
        wfbp_v.reconstruct(d_projection_v.data(), d_recon_v.data());
    cudaStreamSynchronize(stream);

    const size_t rebin_count = static_cast<size_t>(flat_wfbp.geometry().views) *
        flat_wfbp.geometry().rows * flat_wfbp.geometry().output_channels;
    std::vector<float> recon(volume_count);
    std::vector<float> recon_arc(volume_count);
    std::vector<float> recon_v(volume_count);
    std::vector<float> rebin_flat(rebin_count);
    std::vector<float> rebin_arc(rebin_count);
    cudaMemcpy(recon.data(), d_recon_flat.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(recon_arc.data(), d_recon_arc.data(), volume_count * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(recon_v.data(), d_recon_v.data(), volume_count * sizeof(float),
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
    double v_energy = 0.0;
    double v_correlation_sum = 0.0;
    for (size_t i = 0; i < recon.size(); ++i) {
        const float value = recon[i];
        ok = ok && std::isfinite(value);
        energy += static_cast<double>(value) * value;
        truth_energy += static_cast<double>(truth[i]) * truth[i];
        correlation_sum += static_cast<double>(value) * truth[i];
        maximum = std::max(maximum, std::fabs(value));
        volume_path_diff = std::max(volume_path_diff,
            std::fabs(value - recon_arc[i]));
        ok = ok && std::isfinite(recon_v[i]);
        v_energy += static_cast<double>(recon_v[i]) * recon_v[i];
        v_correlation_sum += static_cast<double>(recon_v[i]) * truth[i];
    }
    float rebin_path_diff = 0.f;
    for (size_t i = 0; i < rebin_count; ++i)
        rebin_path_diff = std::max(rebin_path_diff,
            std::fabs(rebin_flat[i] - rebin_arc[i]));
    const double correlation = correlation_sum /
        std::sqrt(std::max(energy * truth_energy, 1e-30));
    const double v_correlation = v_correlation_sum /
        std::sqrt(std::max(v_energy * truth_energy, 1e-30));
    const float expected_principal_row = 0.5f * (h_v.iPV - 1) -
        h_v.offsetV_mm / h_v.dv_mm;
    ok = ok && energy > 1e-12 && maximum > 1e-7f && correlation > 0.05 &&
        rebin_path_diff < 1e-6f && volume_path_diff < 1e-6f &&
        v_energy > 1e-12 && v_correlation > 0.05 &&
        std::fabs(wfbp_v.geometry().central_row - expected_principal_row) < 1e-5f;
    std::printf("Helical wFBP: energy %.8e, max %.8e, corr %.6f, "
        "V-offset corr %.6f, path diff(rebin/vol) %.3e/%.3e, %s\n",
        energy, maximum, correlation, v_correlation, rebin_path_diff,
        volume_path_diff, ok ? "PASS" : "FAIL");

    wfbp_v.release();
    fp_v.release();
    arc_wfbp.release();
    flat_wfbp.release();
    fp.release();
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_helical_wfbp_ffs_smoke()
{
    SHeliCTParam h = makeHelicalParams();
    // 四焦点路径仍需保留足够的有效 views/turn 和 FreeCT 边界补帧。
    h.views_per_rot = 128;
    h.angle_list.resize(384);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / h.views_per_rot;
    h.pitch_mm = 4.f;
    h.start_z_mm = -6.f;
    h.offsetV_mm = 0.75f;

    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
        static_cast<int>(h.angle_list.size()), 0);
    auto d_volume = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);

    // 此测试关注 FreeCT n/p/z/a 四种布局、焦点拆分和输出维度。输入使用
    // 平滑解析信号，确保每条分支都执行插值、滤波和 W(q) 反投，而不依赖
    // 外部厂商原始数据。
    std::vector<float> projection(projection_count);
    for (int view = 0; view < static_cast<int>(h.angle_list.size()); ++view) {
        for (int row = 0; row < h.iPV; ++row) {
            for (int channel = 0; channel < h.iPU; ++channel) {
                const float u = (channel - 0.5f * (h.iPU - 1)) / h.iPU;
                const float v = (row - 0.5f * (h.iPV - 1)) / h.iPV;
                projection[(static_cast<size_t>(view) * h.iPV + row) * h.iPU + channel] =
                    0.3f + 0.15f * std::cos(h.angle_list[view] + 2.f * u) +
                    0.05f * v;
            }
        }
    }
    cudaMemcpyAsync(d_projection.data(), projection.data(),
        projection_count * sizeof(float), cudaMemcpyHostToDevice, stream);

    struct FfsCase {
        Helical::Wfbp::EFocalSpotMode mode;
        bool reverse_rows;
        const char* name;
    };
    const FfsCase cases[] = {
        { Helical::Wfbp::EFocalSpotMode::None, false, "n" },
        { Helical::Wfbp::EFocalSpotMode::Phi, false, "p" },
        { Helical::Wfbp::EFocalSpotMode::Z, false, "z" },
        { Helical::Wfbp::EFocalSpotMode::Z, true, "z-reverse" },
        { Helical::Wfbp::EFocalSpotMode::PhiAndZ, false, "a" },
        { Helical::Wfbp::EFocalSpotMode::PhiAndZ, true, "a-reverse" }
    };
    bool ok = true;
    for (const auto& test_case : cases) {
        const auto mode = test_case.mode;
        Helical::Wfbp::Config config{};
        // FreeCT 原生输入是等角弧形探测器；平板转换已在普通 wFBP 测试
        // 单独覆盖，这里直接验证官方 n/p/z/a 重排公式。
        config.input_detector = Helical::Wfbp::EInputDetector::EquiangularArc;
        config.arc_channel_angle_step_rad = 0.006f;
        config.focal_spot_mode = mode;
        config.anode_angle_rad = 7.f * CUDA_PI / 180.f;
        config.reverse_row_interleave = test_case.reverse_rows;
        Helical::Wfbp::Pipeline pipeline;
        cudaMemsetAsync(d_volume.data(), 0, volume_count * sizeof(float), stream);
        const bool prepared = pipeline.prepare(h, config, stream);
        cudaEvent_t start = nullptr, stop = nullptr;
        float elapsed_ms = 0.f;
        bool reconstructed = false;
        if (prepared && cudaEventCreate(&start) == cudaSuccess &&
            cudaEventCreate(&stop) == cudaSuccess) {
            cudaEventRecord(start, stream);
            reconstructed = pipeline.reconstruct(d_projection.data(), d_volume.data());
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_ms, start, stop);
        }
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        ok = ok && prepared && reconstructed;

        std::vector<float> volume(volume_count);
        cudaMemcpy(volume.data(), d_volume.data(), volume_count * sizeof(float),
            cudaMemcpyDeviceToHost);
        double energy = 0.0;
        for (float value : volume) {
            ok = ok && std::isfinite(value);
            energy += static_cast<double>(value) * value;
        }
        const int expected_spots = mode == Helical::Wfbp::EFocalSpotMode::PhiAndZ
            ? 4 : (mode == Helical::Wfbp::EFocalSpotMode::None ? 1 : 2);
        const int expected_rows = (mode == Helical::Wfbp::EFocalSpotMode::Z ||
            mode == Helical::Wfbp::EFocalSpotMode::PhiAndZ) ? 2 * h.iPV : h.iPV;
        const auto& g = pipeline.geometry();
        const size_t rebin_count = static_cast<size_t>(g.views) * g.rows *
            g.output_channels;
        std::vector<float> rebinned(rebin_count);
        cudaMemcpy(rebinned.data(), pipeline.rebinnedData(),
            rebin_count * sizeof(float), cudaMemcpyDeviceToHost);

        // 与官方两阶段 CPU 公式逐点对照；Pipeline 已按 setup.cu 的
        // add_projections 规则裁掉两端补帧，因此全部输出 view 都应有效。
        float max_rebin_error = 0.f;
        for (int view = 0; view < g.views; ++view) {
            for (int row = 0; row < g.rows; ++row) {
                for (int channel = 0; channel < g.output_channels; ++channel) {
                    const size_t index = (static_cast<size_t>(view) * g.rows + row) *
                        g.output_channels + channel;
                    const float expected = referenceFreeCtRebin(projection, g,
                        view, row, channel);
                    max_rebin_error = std::max(max_rebin_error,
                        std::fabs(rebinned[index] - expected));
                }
            }
        }
        const float raw_principal = 0.5f * (h.iPV - 1) - h.offsetV_mm / h.dv_mm;
        const float expected_principal = g.z_spot_count == 2
            ? 2.f * raw_principal + 0.5f : raw_principal;
        ok = ok && energy > 1e-14 &&
            pipeline.geometry().focal_spot_count == expected_spots &&
            pipeline.geometry().rows == expected_rows &&
            pipeline.geometry().sequence_views ==
                static_cast<int>(h.angle_list.size()) / expected_spots &&
            pipeline.geometry().views ==
                ((pipeline.geometry().sequence_views -
                    2 * pipeline.geometry().add_projections) /
                    (pipeline.geometry().views_per_turn / 2)) *
                (pipeline.geometry().views_per_turn / 2) &&
            std::fabs(g.central_row - expected_principal) < 1e-5f &&
            max_rebin_error < 2e-5f;
        std::printf("  FreeCT %s-FFS: %.3f ms, max rebin error %.3e\n",
            test_case.name, elapsed_ms, max_rebin_error);
        pipeline.release();
    }
    std::printf("Helical wFBP FreeCT n/p/z/a FFS layout: %s\n",
        ok ? "PASS" : "FAIL");
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

int main_helical_wfbp_comparison()
{
    SHeliCTParam base{};
    base.iPU = 96; base.iPV = 32;
    base.iVX = 64; base.iVY = 64; base.iVZ = 32;
    base.du_mm = 1.f; base.dv_mm = 1.f;
    base.vox_x_mm = 0.8f; base.vox_y_mm = 0.8f; base.vox_z_mm = 0.8f;
    base.SID = 160.f; base.SDD = 300.f;
    base.pitch_mm = 8.f; base.start_z_mm = -20.f;
    base.views_per_rot = 128;
    base.angle_list.resize(640); // 五圈，包含 FreeCT 首尾补帧。
    for (int i = 0; i < static_cast<int>(base.angle_list.size()); ++i)
        base.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) /
            base.views_per_rot;

    struct ComparisonCase {
        const char* name;
        bool catphan;
        float pitch_mm;
        float offset_v_mm;
    };
    const ComparisonCase cases[] = {
        { "basic-p6-v0", false, 6.f, 0.f },
        { "basic-p8-v0", false, 8.f, 0.f },
        { "basic-p8-v+1.5", false, 8.f, 1.5f },
        { "basic-p10-v0", false, 10.f, 0.f },
        { "catphan-p8-v0", true, 8.f, 0.f },
        { "catphan-p8-v-1.5", true, 8.f, -1.5f }
    };
    struct ComparisonResult {
        std::string name;
        std::vector<float> truth;
        std::vector<float> reconstruction;
        std::vector<float> error;
        ReconstructionMetrics metrics;
        float fp_ms = 0.f;
        float recon_ms = 0.f;
    };
    std::vector<ComparisonResult> results;
    results.reserve(std::size(cases));

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    const size_t volume_count = static_cast<size_t>(base.iVX) * base.iVY * base.iVZ;
    const size_t projection_count = static_cast<size_t>(base.iPU) * base.iPV *
        base.angle_list.size();
    auto d_truth = memory.allocateDevice3D<float>(base.iVX, base.iVY, base.iVZ, 0);
    auto d_reconstruction = memory.allocateDevice3D<float>(base.iVX, base.iVY,
        base.iVZ, 0);
    auto d_projection = memory.allocateDevice3D<float>(base.iPU, base.iPV,
        static_cast<int>(base.angle_list.size()), 0);

    bool ok = true;
    for (const auto& test_case : cases) {
        SHeliCTParam h = base;
        h.pitch_mm = test_case.pitch_mm;
        // 各 pitch 使用相同圈数，并将完整采集轨迹的中点放在 z=0。
        h.start_z_mm = -0.5f * h.pitch_mm * h.angle_list.back() /
            (2.f * CUDA_PI);
        h.offsetV_mm = test_case.offset_v_mm;
        const SCBCTParams p = toCbct(h);
        std::vector<SConeProjGeomVec> geometry;
        build_helical_vec_geometry(geometry, h);

        ComparisonResult result{};
        result.name = test_case.name;
        result.truth = test_case.catphan ? TestPhantom::makeCatphanLike(p)
                                         : TestPhantom::makeBasic(p);
        result.reconstruction.resize(volume_count);
        result.error.resize(volume_count);
        cudaMemcpyAsync(d_truth.data(), result.truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(d_reconstruction.data(), 0, volume_count * sizeof(float), stream);
        cudaMemsetAsync(d_projection.data(), 0, projection_count * sizeof(float), stream);

        ForwardOperatorAdapter fp;
        Helical::Wfbp::Config config{};
        config.input_detector = Helical::Wfbp::EInputDetector::FlatPanel;
        Helical::Wfbp::Pipeline pipeline;
        cudaEvent_t fp_start = nullptr, fp_stop = nullptr;
        cudaEvent_t recon_start = nullptr, recon_stop = nullptr;
        bool case_ok = cudaEventCreate(&fp_start) == cudaSuccess &&
            cudaEventCreate(&fp_stop) == cudaSuccess &&
            cudaEventCreate(&recon_start) == cudaSuccess &&
            cudaEventCreate(&recon_stop) == cudaSuccess &&
            fp.init(p, geometry, ETask::FP_Joseph, 0, stream) &&
            pipeline.prepare(h, config, stream);
        if (case_ok) {
            cudaEventRecord(fp_start, stream);
            case_ok = fp.run(d_truth.data(), p, d_projection.data(), stream);
            cudaEventRecord(fp_stop, stream);
            cudaEventRecord(recon_start, stream);
            case_ok = case_ok && pipeline.reconstruct(d_projection.data(),
                d_reconstruction.data());
            cudaEventRecord(recon_stop, stream);
            cudaEventSynchronize(recon_stop);
            cudaEventElapsedTime(&result.fp_ms, fp_start, fp_stop);
            cudaEventElapsedTime(&result.recon_ms, recon_start, recon_stop);
        }
        if (fp_start) cudaEventDestroy(fp_start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (recon_start) cudaEventDestroy(recon_start);
        if (recon_stop) cudaEventDestroy(recon_stop);
        if (case_ok) {
            cudaMemcpy(result.reconstruction.data(), d_reconstruction.data(),
                volume_count * sizeof(float), cudaMemcpyDeviceToHost);
            result.metrics = compareReconstruction(result.truth, result.reconstruction,
                h.iVX, h.iVY, h.iVZ, h.iVZ / 2);
            for (size_t i = 0; i < volume_count; ++i)
                result.error[i] = result.metrics.scale * result.reconstruction[i] -
                    result.truth[i];
            case_ok = std::isfinite(result.metrics.correlation) &&
                std::isfinite(result.metrics.nrmse) &&
                result.metrics.correlation > 0.45 && result.metrics.nrmse < 0.90;
        }
        std::printf("wFBP compare %-16s corr %.6f slice-corr %.6f "
            "NRMSE %.6f scale %.6f FP %.3f ms recon %.3f ms %s\n",
            result.name.c_str(), result.metrics.correlation,
            result.metrics.slice_correlation, result.metrics.nrmse,
            result.metrics.scale, result.fp_ms, result.recon_ms,
            case_ok ? "PASS" : "FAIL");
        ok = ok && case_ok;
        pipeline.release();
        fp.release();
        results.push_back(std::move(result));
    }

    // 每行依次输出 truth、同窗宽的尺度校正重建、绝对误差。Basic 展示
    // 两种 offset；Catphan-like 则对每种 offset 展示材料、低对比和分辨率模块。
    std::vector<TestImage::GrayPanel> basic_panels;
    std::vector<TestImage::GrayPanel> catphan_panels;
    const int catphan_slices[] = { 11, 18, 24 };
    for (const auto& result : results) {
        const bool catphan = result.name.find("catphan") != std::string::npos;
        const int* slices = catphan ? catphan_slices : nullptr;
        const int slice_count = catphan ? 3 : 1;
        for (int i = 0; i < slice_count; ++i) {
            const int z = catphan ? slices[i] : base.iVZ / 2;
            auto& panels = catphan ? catphan_panels : basic_panels;
            const float display_max = catphan ? 0.09f : 0.08f;
            panels.push_back({ &result.truth, base.iVX, base.iVY, base.iVZ,
                z, 1.f, 0.f, display_max, false });
            panels.push_back({ &result.reconstruction, base.iVX, base.iVY, base.iVZ,
                z, result.metrics.scale, 0.f, display_max, false });
            panels.push_back({ &result.error, base.iVX, base.iVY, base.iVZ,
                z, 1.f, 0.f, 0.03f, true });
        }
    }
    const std::filesystem::path artifact_dir = std::filesystem::absolute(
        "out/test-artifacts");
    const auto basic_artifact = artifact_dir / "wfbp_basic_comparison.bmp";
    const auto catphan_artifact = artifact_dir / "wfbp_catphan_comparison.bmp";
    const bool basic_image_ok = TestImage::writeGrayMontageBmp(
        basic_artifact, basic_panels, 3, 8, 4);
    const bool catphan_image_ok = TestImage::writeGrayMontageBmp(
        catphan_artifact, catphan_panels, 3, 8, 4);

    const auto metrics_artifact = artifact_dir / "wfbp_comparison_metrics.csv";
    std::filesystem::create_directories(artifact_dir);
    std::ofstream metrics_file(metrics_artifact);
    metrics_file << "case,correlation,slice_correlation,nrmse,scale,fp_ms,recon_ms\n";
    for (const auto& result : results)
        metrics_file << result.name << ',' << result.metrics.correlation << ','
            << result.metrics.slice_correlation << ',' << result.metrics.nrmse << ','
            << result.metrics.scale << ',' << result.fp_ms << ',' << result.recon_ms
            << '\n';
    const bool metrics_ok = metrics_file.good();
    std::printf("wFBP comparison images: %s, %s (%s)\n",
        basic_artifact.string().c_str(), catphan_artifact.string().c_str(),
        basic_image_ok && catphan_image_ok && metrics_ok ? "written" : "FAILED");
    cudaStreamDestroy(stream);
    return ok && basic_image_ok && catphan_image_ok && metrics_ok ? 0 : 1;
}

int main_helical_large_volume()
{
    // 该用例专门覆盖常规 smoke test 无法暴露的较大显存步长、三维索引和
    // wFBP 中间缓冲区问题。规模明显大于功能测试，但仍控制在常见 8 GiB
    // 显卡可运行的范围内；输入完全由合成模体生成，不依赖本地 RAW 数据。
    SHeliCTParam h{};
    h.iPU = 384; h.iPV = 80;
    h.iVX = 160; h.iVY = 160; h.iVZ = 96;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 0.6f; h.vox_y_mm = 0.6f; h.vox_z_mm = 0.6f;
    h.SID = 400.f; h.SDD = 800.f;
    h.pitch_mm = 20.f;
    h.views_per_rot = 192;
    h.angle_list.resize(768); // 四圈，兼顾轴向覆盖和 FreeCT 首尾补帧。
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) /
            h.views_per_rot;
    h.start_z_mm = -0.5f * h.pitch_mm * h.angle_list.back() /
        (2.f * CUDA_PI);

    const SCBCTParams p = toCbct(h);
    const auto truth = TestPhantom::makeArrowDirections(p, true);
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();

    size_t free_before = 0, total_memory = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) {
        if (stream) cudaStreamDestroy(stream);
        std::printf("Helical large volume: CUDA initialization failed\n");
        return 1;
    }

    float fp_ms = 0.f, recon_ms = 0.f;
    ReconstructionMetrics metrics{};
    std::vector<float> projection(projection_count);
    std::vector<float> arc_backprojection(volume_count);
    std::vector<float> reconstruction(volume_count);
    size_t free_during = free_before;
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(h.iVX, h.iVY,
            h.iVZ, 0);
        auto d_arc_backprojection = memory.allocateDevice3D<float>(h.iVX, h.iVY,
            h.iVZ, 0);
        auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) ==
            cudaSuccess;
        ok = ok && cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;

        // 真实等角圆柱探测器：弧长为一个平板像素在探测器中心对应的
        // 圆弧，CylFpBp 正投影输出可直接作为 wFBP 的弧面输入。
        const float arc_step = 2.f * atanf(0.5f * h.du_mm / h.SDD);
        const auto arc_geometry = CylFpBp::buildFreeCtArcGeometry(h, arc_step);
        CylFpBp::Operator projector;
        CylFpBp::Config projector_config{};
        projector_config.samples_per_voxel = 2.f;
        Helical::Wfbp::Config config{};
        config.input_detector = Helical::Wfbp::EInputDetector::EquiangularArc;
        config.arc_channel_angle_step_rad = arc_step;
        config.arc_principal_channel = 0.5f * (h.iPU - 1) -
            h.offsetU_mm / h.du_mm;
        Helical::Wfbp::Pipeline pipeline;
        cudaEvent_t fp_start = nullptr, fp_stop = nullptr;
        cudaEvent_t recon_start = nullptr, recon_stop = nullptr;
        ok = ok && cudaEventCreate(&fp_start) == cudaSuccess &&
            cudaEventCreate(&fp_stop) == cudaSuccess &&
            cudaEventCreate(&recon_start) == cudaSuccess &&
            cudaEventCreate(&recon_stop) == cudaSuccess &&
            projector.prepare(volumeGeometry(h), h.iPU, h.iPV, arc_geometry,
                projector_config) &&
            pipeline.prepare(h, config, stream);
        if (ok) {
            cudaEventRecord(fp_start, stream);
            ok = projector.forward(d_truth.data(), d_projection.data(), stream);
            cudaEventRecord(fp_stop, stream);
            ok = ok && projector.backproject(d_projection.data(),
                d_arc_backprojection.data(), stream);
            cudaEventRecord(recon_start, stream);
            ok = ok && pipeline.reconstruct(d_projection.data(),
                d_reconstruction.data());
            cudaEventRecord(recon_stop, stream);
            ok = ok && cudaEventSynchronize(recon_stop) == cudaSuccess;
            if (ok) {
                cudaEventElapsedTime(&fp_ms, fp_start, fp_stop);
                cudaEventElapsedTime(&recon_ms, recon_start, recon_stop);
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                ok = ok && cudaMemcpy(arc_backprojection.data(),
                    d_arc_backprojection.data(), volume_count * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
                ok = ok && cudaMemcpy(reconstruction.data(), d_reconstruction.data(),
                    volume_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                if (ok) {
                    metrics = compareReconstruction(truth, reconstruction,
                        h.iVX, h.iVY, h.iVZ, h.iVZ / 2);
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                        [](float value) { return std::isfinite(value); }) &&
                        std::all_of(arc_backprojection.begin(),
                            arc_backprojection.end(), [](float value) {
                                return std::isfinite(value);
                            }) &&
                        squaredNorm(arc_backprojection) > 1e-12 &&
                        squaredNorm(reconstruction) > 1e-12 &&
                        std::isfinite(metrics.correlation) &&
                        std::isfinite(metrics.slice_correlation) &&
                        std::isfinite(metrics.nrmse) &&
                        // 箭头模体由细杆和尖锥组成，中心层只有很小的有效
                        // 支撑区域，因此切片指标天然低于大圆柱模体。实测
                        // 基线约为 0.831/0.342/0.556，门槛保留跨 GPU 余量，
                        // 同时仍能拒绝方向错误、空输出和明显的数值退化。
                        metrics.correlation > 0.75 &&
                        metrics.slice_correlation > 0.25 &&
                        metrics.nrmse < 0.70;
                }
            }
        }
        cudaMemGetInfo(&free_during, &total_memory);
        if (fp_start) cudaEventDestroy(fp_start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (recon_start) cudaEventDestroy(recon_start);
        if (recon_stop) cudaEventDestroy(recon_stop);
        pipeline.release();
        projector.release();
    }
    cudaStreamDestroy(stream);

    const std::filesystem::path artifact_dir = std::filesystem::absolute(
        "out/test-artifacts/helical-large-volume");
    const auto phantom_path = artifact_dir /
        "arrow_phantom_f32_160x160x96.raw";
    const auto projection_path = artifact_dir /
        "arrow_arc_projection_f32_384x80x768.raw";
    const auto backprojection_path = artifact_dir /
        "arrow_arc_backprojection_f32_160x160x96.raw";
    const auto reconstruction_path = artifact_dir /
        "arrow_reconstruction_f32_160x160x96.raw";
    const auto metadata_path = artifact_dir / "arrow_large_volume.json";
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(backprojection_path, arc_backprojection) &&
        writeFloatRaw(reconstruction_path, reconstruction);
    std::filesystem::create_directories(artifact_dir);
    std::ofstream metadata(metadata_path);
    metadata << "{\n"
        << "  \"scalar_type\": \"float32-little-endian\",\n"
        << "  \"storage_order\": \"x/u fastest, then y/v, then z/view\",\n"
        << "  \"phantom\": {\"file\": \"" << phantom_path.filename().string()
        << "\", \"dimensions_xyz\": [" << h.iVX << ", " << h.iVY << ", "
        << h.iVZ << "], \"voxel_mm_xyz\": [" << h.vox_x_mm << ", "
        << h.vox_y_mm << ", " << h.vox_z_mm
        << "], \"arrow_mu_xyz\": [0.02, 0.04, 0.06], "
        << "\"boundary_faces\": [\"+X\", \"-X\", \"+Y\", \"-Y\", "
        << "\"+Z\", \"-Z\"], \"edge_axis_labels\": true, "
        << "\"label_mu\": 0.01},\n"
        << "  \"projection\": {\"file\": \""
        << projection_path.filename().string() << "\", \"dimensions_uv_view\": ["
        << h.iPU << ", " << h.iPV << ", " << h.angle_list.size() << "]},\n"
        << "  \"arc_backprojection\": {\"file\": \""
        << backprojection_path.filename().string()
        << "\", \"dimensions_xyz\": [" << h.iVX << ", " << h.iVY << ", "
        << h.iVZ << "], \"filtered\": false},\n"
        << "  \"reconstruction\": {\"file\": \""
        << reconstruction_path.filename().string()
        << "\", \"dimensions_xyz\": [" << h.iVX << ", " << h.iVY << ", "
        << h.iVZ << "]},\n"
        << "  \"helical\": {\"sid_mm\": " << h.SID << ", \"sdd_mm\": "
        << h.SDD << ", \"pitch_mm\": " << h.pitch_mm
        << ", \"views_per_rotation\": " << h.views_per_rot
        << ", \"detector\": \"equiangular-arc\", "
        << "\"arc_channel_angle_step_rad\": "
        << 2.f * atanf(0.5f * h.du_mm / h.SDD) << "}\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double projection_mib = projection_count * sizeof(float) /
        (1024.0 * 1024.0);
    const double volume_mib = volume_count * sizeof(float) / (1024.0 * 1024.0);
    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    std::printf("Helical large volume: volume %dx%dx%d (%.1f MiB), "
        "projection %dx%dx%zu (%.1f MiB), GPU workspace %.1f MiB\n",
        h.iVX, h.iVY, h.iVZ, volume_mib, h.iPU, h.iPV,
        h.angle_list.size(), projection_mib, used_mib);
    std::printf("  corr %.6f, slice-corr %.6f, NRMSE %.6f, "
        "FP %.3f ms, recon %.3f ms: %s\n",
        metrics.correlation, metrics.slice_correlation, metrics.nrmse,
        fp_ms, recon_ms, ok ? "PASS" : "FAIL");
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}

int main_helical_large_volume_icd()
{
    // 与弧面 wFBP 大体积用例保持相同体积、探测器和扫描轨迹，但使用
    // 原生平板 vector geometry、Joseph FP/BP 和螺旋 PWLS-ICD。
    SHeliCTParam h{};
    h.iPU = 384; h.iPV = 80;
    h.iVX = 160; h.iVY = 160; h.iVZ = 96;
    h.du_mm = 1.f; h.dv_mm = 1.f;
    h.vox_x_mm = 0.6f; h.vox_y_mm = 0.6f; h.vox_z_mm = 0.6f;
    h.SID = 400.f; h.SDD = 800.f;
    h.pitch_mm = 20.f;
    h.views_per_rot = 192;
    h.angle_list.resize(768);
    for (int i = 0; i < static_cast<int>(h.angle_list.size()); ++i)
        h.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) /
            h.views_per_rot;
    h.start_z_mm = -0.5f * h.pitch_mm * h.angle_list.back() /
        (2.f * CUDA_PI);

    const SCBCTParams p = toCbct(h);
    std::vector<SConeProjGeomVec> geometry;
    build_helical_vec_geometry(geometry, h);
    const auto truth = TestPhantom::makeArrowDirections(p, true);
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();

    size_t free_before = 0, total_memory = 0, free_during = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) {
        if (stream) cudaStreamDestroy(stream);
        std::printf("Helical large ICD: CUDA initialization failed\n");
        return 1;
    }

    std::vector<float> projection(projection_count);
    std::vector<float> reconstruction(volume_count);
    ReconstructionMetrics metrics{};
    float fp_ms = 0.f, prepare_ms = 0.f, recon_ms = 0.f;
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ, 0);
        auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(h.iVX, h.iVY,
            h.iVZ, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) ==
            cudaSuccess;
        ok = ok && cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;

        ForwardOperatorAdapter fp;
        Iter::ParallelPwlsConfig config{};
        config.iterations = 12;
        config.relaxation = 0.7f;
        config.regularization = 2e-3f;
        config.fp_task = ETask::FP_Joseph;
        config.bp_task = ETask::BP_Joseph_v3;
        Iter::ParallelPwlsReconstructor pwls;
        cudaEvent_t start = nullptr, fp_stop = nullptr;
        cudaEvent_t prepare_stop = nullptr, recon_stop = nullptr;
        ok = ok && cudaEventCreate(&start) == cudaSuccess &&
            cudaEventCreate(&fp_stop) == cudaSuccess &&
            cudaEventCreate(&prepare_stop) == cudaSuccess &&
            cudaEventCreate(&recon_stop) == cudaSuccess &&
            fp.init(p, geometry, ETask::FP_Joseph, 0, stream);
        if (ok) {
            cudaEventRecord(start, stream);
            ok = fp.run(d_truth.data(), p, d_projection.data(), stream);
            cudaEventRecord(fp_stop, stream);
            ok = ok && pwls.prepare(p, geometry, config, stream);
            cudaEventRecord(prepare_stop, stream);
            ok = ok && pwls.reconstruct(d_projection.data(),
                d_reconstruction.data());
            cudaEventRecord(recon_stop, stream);
            ok = ok && cudaEventSynchronize(recon_stop) == cudaSuccess;
            if (ok) {
                cudaEventElapsedTime(&fp_ms, start, fp_stop);
                cudaEventElapsedTime(&prepare_ms, fp_stop, prepare_stop);
                cudaEventElapsedTime(&recon_ms, prepare_stop, recon_stop);
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                ok = ok && cudaMemcpy(reconstruction.data(),
                    d_reconstruction.data(), volume_count * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) {
                    metrics = compareReconstruction(truth, reconstruction,
                        h.iVX, h.iVY, h.iVZ, h.iVZ / 2);
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                        [](float value) { return std::isfinite(value); }) &&
                        squaredNorm(reconstruction) > 1e-12 &&
                        std::isfinite(metrics.correlation) &&
                        std::isfinite(metrics.slice_correlation) &&
                        std::isfinite(metrics.nrmse) &&
                        // 12 次迭代实测基线约 0.784/0.763/0.621。
                        metrics.correlation > 0.70 &&
                        metrics.slice_correlation > 0.65 &&
                        metrics.nrmse < 0.75;
                }
            }
        }
        cudaMemGetInfo(&free_during, &total_memory);
        if (start) cudaEventDestroy(start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (prepare_stop) cudaEventDestroy(prepare_stop);
        if (recon_stop) cudaEventDestroy(recon_stop);
        pwls.release();
        fp.release();
    }
    cudaStreamDestroy(stream);

    const std::filesystem::path artifact_dir = std::filesystem::absolute(
        "out/test-artifacts/helical-large-volume-icd");
    const auto phantom_path = artifact_dir /
        "arrow_phantom_f32_160x160x96.raw";
    const auto projection_path = artifact_dir /
        "arrow_flat_projection_f32_384x80x768.raw";
    const auto reconstruction_path = artifact_dir /
        "arrow_icd_reconstruction_f32_160x160x96.raw";
    const auto metadata_path = artifact_dir / "arrow_large_volume_icd.json";
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(reconstruction_path, reconstruction);
    std::ofstream metadata(metadata_path);
    metadata << "{\n"
        << "  \"scalar_type\": \"float32-little-endian\",\n"
        << "  \"storage_order\": \"x/u fastest, then y/v, then z/view\",\n"
        << "  \"detector\": \"flat-panel\",\n"
        << "  \"phantom_dimensions_xyz\": [160, 160, 96],\n"
        << "  \"projection_dimensions_uv_view\": [384, 80, 768],\n"
        << "  \"iterations\": 12,\n"
        << "  \"fp\": \"Joseph\",\n"
        << "  \"bp\": \"Joseph-v3\"\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    std::printf("Helical large ICD: corr %.6f, slice-corr %.6f, "
        "NRMSE %.6f, FP %.3f ms, prepare %.3f ms, 12 iterations %.3f ms, "
        "GPU workspace %.1f MiB: %s\n", metrics.correlation,
        metrics.slice_correlation, metrics.nrmse, fp_ms, prepare_ms, recon_ms,
        used_mib, ok ? "PASS" : "FAIL");
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}
