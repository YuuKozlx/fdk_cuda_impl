#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include "FDK/YkFdkPipeline.hpp"
#include "Iter/YkAlgebraicReconstructor.hpp"
#include "Iter/YkParallelPwlsReconstructor.hpp"
#include "common/YkProjectionOperators.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

bool writeFloatRaw(const std::filesystem::path& path,
    const std::vector<float>& values)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

SCBCTParams makeParams()
{
    SCBCTParams p{};
    p.iVX = 512; p.iVY = 512; p.iVZ = 400;
    p.vox_x_mm = 0.3f; p.vox_y_mm = 0.3f; p.vox_z_mm = 0.3f;
    p.iPU = 1024; p.iPV = 128;
    p.du_mm = 0.417f; p.dv_mm = 0.417f;
    p.SID = 440.f; p.SDD = 770.f;
    p.iPAng = 720; p.iPAngTotal = 720;
    p.scan_start_angle_rad = 0.f;
    p.scan_range_rad = 2.f * CUDA_PI;
    p.bShortScan = false;
    p.offsetU_mm = 0.f; p.offsetV_mm = 0.f;
    p.angle_list.resize(p.iPAng);
    for (int i = 0; i < p.iPAng; ++i)
        p.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / p.iPAng;
    return p;
}

// 缩小后的封闭水模：水区直径 130 mm，外壳厚 10 mm，外径 150 mm。
// 为避免 Z 边界贴住体积，外部高度 110 mm，内部水柱高度 90 mm。
std::vector<float> makeWaterPhantom(const SCBCTParams& p)
{
    constexpr float water_radius_mm = 65.f;
    constexpr float shell_outer_radius_mm = 75.f;
    constexpr float water_half_height_mm = 45.f;
    constexpr float shell_outer_half_height_mm = 55.f;
    constexpr float water_mu = 0.020f;
    constexpr float shell_mu = 0.040f;
    std::vector<float> volume(static_cast<size_t>(p.iVX) * p.iVY * p.iVZ, 0.f);
    for (int z = 0; z < p.iVZ; ++z) {
        const float pz = (z + 0.5f - 0.5f * p.iVZ) * p.vox_z_mm;
        for (int y = 0; y < p.iVY; ++y) {
            const float py = (y + 0.5f - 0.5f * p.iVY) * p.vox_y_mm;
            for (int x = 0; x < p.iVX; ++x) {
                const float px = (x + 0.5f - 0.5f * p.iVX) * p.vox_x_mm;
                const float radius2 = px * px + py * py;
                float value = 0.f;
                if (radius2 <= shell_outer_radius_mm * shell_outer_radius_mm &&
                    std::fabs(pz) <= shell_outer_half_height_mm) {
                    value = shell_mu;
                    if (radius2 <= water_radius_mm * water_radius_mm &&
                        std::fabs(pz) <= water_half_height_mm)
                        value = water_mu;
                }
                volume[(static_cast<size_t>(z) * p.iVY + y) * p.iVX + x] = value;
            }
        }
    }
    return volume;
}

struct Metrics {
    double correlation = 0.0;
    double nrmse = 0.0;
    float scale = 0.f;
};

struct MaterialMetrics {
    double water_mean = 0.0;
    double shell_mean = 0.0;
    float shell_peak = 0.f;
};

MaterialMetrics measureMaterials(const std::vector<float>& reconstruction,
    const SCBCTParams& p, float scale)
{
    double water_sum = 0.0, shell_sum = 0.0;
    size_t water_count = 0, shell_count = 0;
    MaterialMetrics result{};
    const int z = p.iVZ / 2;
    for (int y = 0; y < p.iVY; ++y) {
        const float py = (y + 0.5f - 0.5f * p.iVY) * p.vox_y_mm;
        for (int x = 0; x < p.iVX; ++x) {
            const float px = (x + 0.5f - 0.5f * p.iVX) * p.vox_x_mm;
            const float radius = std::sqrt(px * px + py * py);
            const float value = scale * reconstruction[
                (static_cast<size_t>(z) * p.iVY + y) * p.iVX + x];
            // 远离两条材料边界取均值，避免插值过渡带影响材料值判断。
            if (radius < 55.f) {
                water_sum += value;
                ++water_count;
            }
            else if (radius > 68.f && radius < 72.f) {
                shell_sum += value;
                result.shell_peak = std::max(result.shell_peak, value);
                ++shell_count;
            }
        }
    }
    result.water_mean = water_count ? water_sum / water_count : 0.0;
    result.shell_mean = shell_count ? shell_sum / shell_count : 0.0;
    return result;
}

Metrics compareCentralSlab(const std::vector<float>& truth,
    const std::vector<float>& reconstruction, const SCBCTParams& p,
    float half_span_mm)
{
    double truth_norm = 0.0, recon_norm = 0.0, dot = 0.0;
    for (int z = 0; z < p.iVZ; ++z) {
        const float pz = (z + 0.5f - 0.5f * p.iVZ) * p.vox_z_mm;
        if (std::fabs(pz) > half_span_mm) continue;
        const size_t begin = static_cast<size_t>(z) * p.iVX * p.iVY;
        const size_t end = begin + static_cast<size_t>(p.iVX) * p.iVY;
        for (size_t i = begin; i < end; ++i) {
            truth_norm += static_cast<double>(truth[i]) * truth[i];
            recon_norm += static_cast<double>(reconstruction[i]) * reconstruction[i];
            dot += static_cast<double>(truth[i]) * reconstruction[i];
        }
    }
    Metrics result{};
    result.scale = recon_norm > 1e-30 ? static_cast<float>(dot / recon_norm) : 0.f;
    double error_norm = 0.0;
    for (int z = 0; z < p.iVZ; ++z) {
        const float pz = (z + 0.5f - 0.5f * p.iVZ) * p.vox_z_mm;
        if (std::fabs(pz) > half_span_mm) continue;
        const size_t begin = static_cast<size_t>(z) * p.iVX * p.iVY;
        const size_t end = begin + static_cast<size_t>(p.iVX) * p.iVY;
        for (size_t i = begin; i < end; ++i) {
            const double error = result.scale * reconstruction[i] - truth[i];
            error_norm += error * error;
        }
    }
    result.correlation = dot /
        std::sqrt(std::max(truth_norm * recon_norm, 1e-30));
    result.nrmse = std::sqrt(error_norm / std::max(truth_norm, 1e-30));
    return result;
}

} // namespace

int main_large_water_pwls()
{
    const SCBCTParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t projection_count = static_cast<size_t>(p.iPU) * p.iPV * p.iPAng;
    std::vector<float> projection(projection_count);
    std::vector<float> reconstruction(volume_count);

    size_t free_before = 0, total_memory = 0, free_during = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) return 1;

    float fp_ms = 0.f, prepare_ms = 0.f, recon_ms = 0.f;
    Metrics metrics{};
    MaterialMetrics material{};
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(), volume_count * sizeof(float),
            cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;

        ForwardOperatorAdapter fp;
        Iter::ParallelPwlsConfig config{};
        config.iterations = 120;
        config.subset_count = 10;
        config.relaxation = 0.25f;
        // 无噪声解析模体用于观察数据项的充分收敛能力；关闭正则，避免把
        // 正则化固有的边缘平滑误判成迭代算法无法恢复高频。
        config.regularizer = Iter::EPwlsRegularizer::None;
        config.regularization = 0.f;
        config.huber_delta = 3e-3f;
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
            ok = ok && pwls.reconstruct(d_projection.data(), d_reconstruction.data());
            cudaEventRecord(recon_stop, stream);
            ok = ok && cudaEventSynchronize(recon_stop) == cudaSuccess;
            if (ok) {
                cudaEventElapsedTime(&fp_ms, start, fp_stop);
                cudaEventElapsedTime(&prepare_ms, fp_stop, prepare_stop);
                cudaEventElapsedTime(&recon_ms, prepare_stop, recon_stop);
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                ok = ok && cudaMemcpy(reconstruction.data(), d_reconstruction.data(),
                    volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) {
                    metrics = compareCentralSlab(truth, reconstruction, p, 12.f);
                    material = measureMaterials(reconstruction, p, metrics.scale);
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                        [](float value) { return std::isfinite(value); }) &&
                        std::isfinite(metrics.correlation) &&
                        std::isfinite(metrics.nrmse) &&
                        // 120x10 无正则 OS-PWLS 的充分收敛回归门槛。
                        metrics.correlation > 0.995 && metrics.nrmse < 0.08;
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

    const auto artifact_dir = std::filesystem::absolute(
        "out/test-artifacts/large-water-pwls");
    const auto phantom_path = artifact_dir / "water_shell_f32_512x512x400.raw";
    const auto projection_path = artifact_dir /
        "water_projection_f32_1024x128x720.raw";
    const auto reconstruction_path = artifact_dir /
        "water_pwls_reconstruction_f32_512x512x400.raw";
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(reconstruction_path, reconstruction);
    std::ofstream metadata(artifact_dir / "water_pwls.json");
    metadata << "{\n"
        << "  \"volume_xyz\": [512, 512, 400],\n"
        << "  \"voxel_mm_xyz\": [0.3, 0.3, 0.3],\n"
        << "  \"water_diameter_mm\": 130,\n"
        << "  \"shell_thickness_mm\": 10,\n"
        << "  \"outer_diameter_mm\": 150,\n"
        << "  \"detector_uv\": [1024, 128],\n"
        << "  \"detector_pixel_mm_uv\": [0.417, 0.417],\n"
        << "  \"sid_mm\": 440, \"sdd_mm\": 770,\n"
        << "  \"views\": 720, \"scan_degrees\": 360,\n"
        << "  \"offset_mm_uv\": [0, 0],\n"
        << "  \"detector_axial_fov_at_isocenter_mm\": 30.50,\n"
        << "  \"coverage_note\": \"128 detector rows do not cover the full 120 mm volume height\",\n"
        << "  \"central_metric_half_span_mm\": 12,\n"
        << "  \"outer_iterations\": 120, \"subsets\": 10, "
        << "\"regularizer\": \"None\", "
        << "\"regularization\": 0.0, "
        << "\"fp\": \"Joseph\", \"bp\": \"Joseph-v3\",\n"
        << "  \"correlation\": " << metrics.correlation
        << ", \"nrmse\": " << metrics.nrmse << ",\n"
        << "  \"water_mean\": " << material.water_mean
        << ", \"shell_mean\": " << material.shell_mean
        << ", \"shell_peak\": " << material.shell_peak << "\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    const double axial_fov_mm = p.iPV * p.dv_mm * p.SID / p.SDD;
    std::printf("Large water PWLS: corr %.6f NRMSE %.6f, "
        "water %.6f shell %.6f peak %.6f, axial FOV %.2f mm, "
        "FP %.1f ms prepare %.1f ms 120x10 subset updates %.1f ms, GPU %.1f MiB: %s\n",
        metrics.correlation, metrics.nrmse, material.water_mean,
        material.shell_mean, material.shell_peak, axial_fov_mm, fp_ms,
        prepare_ms, recon_ms, used_mib, ok ? "PASS" : "FAIL");
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}

int main_large_water_ossart()
{
    const SCBCTParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t projection_count = static_cast<size_t>(p.iPU) * p.iPV * p.iPAng;
    std::vector<float> projection(projection_count);
    std::vector<float> reconstruction(volume_count);

    size_t free_before = 0, total_memory = 0, free_during = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) return 1;

    float fp_ms = 0.f, prepare_ms = 0.f, recon_ms = 0.f;
    Metrics metrics{};
    MaterialMetrics material{};
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.iVX, p.iVY,
            p.iVZ, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(), volume_count * sizeof(float),
            cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;

        ForwardOperatorAdapter fp;
        Iter::AlgebraicReconstructionConfig config{};
        config.method = Iter::EAlgebraicMethod::Ossart;
        config.iterations = 120;
        config.subset_count = 10;
        // 此实现对首轮大步长敏感；沿用已验证稳定的 0.25，并通过增加完整
        // 外循环观察其充分收敛/极限环水平。
        config.relaxation = 0.25f;
        config.relaxation_reduction = 1.f;
        config.epsilon = 1e-6f;
        config.use_min = true;
        config.min_constraint = 0.f;
        config.fp_task = ETask::FP_Joseph;
        config.bp_task = ETask::BP_Joseph_v3;
        Iter::AlgebraicReconstructorEx ossart;
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
            ok = ok && ossart.prepare(p, geometry, config, stream);
            cudaEventRecord(prepare_stop, stream);
            ok = ok && ossart.reconstruct(d_projection.data(), d_reconstruction.data());
            cudaEventRecord(recon_stop, stream);
            ok = ok && cudaEventSynchronize(recon_stop) == cudaSuccess;
            if (ok) {
                cudaEventElapsedTime(&fp_ms, start, fp_stop);
                cudaEventElapsedTime(&prepare_ms, fp_stop, prepare_stop);
                cudaEventElapsedTime(&recon_ms, prepare_stop, recon_stop);
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                ok = ok && cudaMemcpy(reconstruction.data(), d_reconstruction.data(),
                    volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) {
                    metrics = compareCentralSlab(truth, reconstruction, p, 12.f);
                    material = measureMaterials(reconstruction, p, metrics.scale);
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                        [](float value) { return std::isfinite(value); }) &&
                        std::isfinite(metrics.correlation) &&
                        std::isfinite(metrics.nrmse) &&
                        metrics.correlation > 0.997 && metrics.nrmse < 0.07;
                }
            }
        }
        cudaMemGetInfo(&free_during, &total_memory);
        if (start) cudaEventDestroy(start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (prepare_stop) cudaEventDestroy(prepare_stop);
        if (recon_stop) cudaEventDestroy(recon_stop);
        ossart.release();
        fp.release();
    }
    cudaStreamDestroy(stream);

    const auto artifact_dir = std::filesystem::absolute(
        "out/test-artifacts/large-water-ossart");
    const auto phantom_path = artifact_dir / "water_shell_f32_512x512x400.raw";
    const auto projection_path = artifact_dir /
        "water_projection_f32_1024x128x720.raw";
    const auto reconstruction_path = artifact_dir /
        "water_ossart_reconstruction_f32_512x512x400.raw";
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(reconstruction_path, reconstruction);
    std::ofstream metadata(artifact_dir / "water_ossart.json");
    metadata << "{\n"
        << "  \"volume_xyz\": [512, 512, 400],\n"
        << "  \"voxel_mm_xyz\": [0.3, 0.3, 0.3],\n"
        << "  \"water_diameter_mm\": 130, \"shell_thickness_mm\": 10,\n"
        << "  \"detector_uv\": [1024, 128],\n"
        << "  \"detector_pixel_mm_uv\": [0.417, 0.417],\n"
        << "  \"sid_mm\": 440, \"sdd_mm\": 770,\n"
        << "  \"views\": 720, \"scan_degrees\": 360,\n"
        << "  \"outer_iterations\": 120, \"subsets\": 10,\n"
        << "  \"lambda\": 0.25, \"lambda_reduction\": 1.0,\n"
        << "  \"fp\": \"Joseph\", \"bp\": \"Joseph-v3\",\n"
        << "  \"correlation\": " << metrics.correlation
        << ", \"nrmse\": " << metrics.nrmse << ",\n"
        << "  \"water_mean\": " << material.water_mean
        << ", \"shell_mean\": " << material.shell_mean
        << ", \"shell_peak\": " << material.shell_peak << "\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    std::printf("Large water OSSART: corr %.6f NRMSE %.6f, "
        "water %.6f shell %.6f peak %.6f, FP %.1f ms prepare %.1f ms "
        "120x10 updates %.1f ms, GPU %.1f MiB: %s\n", metrics.correlation,
        metrics.nrmse, material.water_mean, material.shell_mean,
        material.shell_peak, fp_ms, prepare_ms, recon_ms, used_mib,
        ok ? "PASS" : "FAIL");
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}

int main_large_water_fdk()
{
    const SCBCTParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t projection_count = static_cast<size_t>(p.iPU) * p.iPV * p.iPAng;
    std::vector<float> projection(projection_count);
    std::vector<float> reconstruction(volume_count);

    size_t free_before = 0, total_memory = 0, free_during = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) return 1;

    float fp_ms = 0.f, prepare_ms = 0.f, recon_ms = 0.f;
    Metrics metrics{};
    MaterialMetrics material{};
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.iVX, p.iVY,
            p.iVZ, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(), volume_count * sizeof(float),
            cudaMemcpyHostToDevice, stream) == cudaSuccess;

        ForwardOperatorAdapter fp;
        FdkPipeline fdk;
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
            ok = ok && cudaEventSynchronize(fp_stop) == cudaSuccess;
            if (ok) {
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
            }
            ok = ok && fdk.prepareWithGeometry(p, geometry, 64, stream);
            cudaEventRecord(prepare_stop, stream);
            if (ok) {
                const FdkProjectionBatch batch{
                    projection.data(), &geometry, nullptr, p.iPAng
                };
                ok = fdk.processBatch(batch, d_reconstruction.data(), true);
                cudaEventRecord(recon_stop, stream);
                ok = ok && cudaEventSynchronize(recon_stop) == cudaSuccess;
            }
            if (ok) {
                cudaEventElapsedTime(&fp_ms, start, fp_stop);
                cudaEventElapsedTime(&prepare_ms, fp_stop, prepare_stop);
                cudaEventElapsedTime(&recon_ms, prepare_stop, recon_stop);
                ok = cudaMemcpy(reconstruction.data(), d_reconstruction.data(),
                    volume_count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) {
                    metrics = compareCentralSlab(truth, reconstruction, p, 12.f);
                    material = measureMaterials(reconstruction, p, metrics.scale);
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                        [](float value) { return std::isfinite(value); }) &&
                        std::isfinite(metrics.correlation) &&
                        std::isfinite(metrics.nrmse) &&
                        metrics.correlation > 0.75 && metrics.nrmse < 0.70;
                }
            }
        }
        cudaMemGetInfo(&free_during, &total_memory);
        if (start) cudaEventDestroy(start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (prepare_stop) cudaEventDestroy(prepare_stop);
        if (recon_stop) cudaEventDestroy(recon_stop);
        fdk.release();
        fp.release();
    }
    cudaStreamDestroy(stream);

    const auto artifact_dir = std::filesystem::absolute(
        "out/test-artifacts/large-water-fdk");
    const auto phantom_path = artifact_dir / "water_shell_f32_512x512x400.raw";
    const auto projection_path = artifact_dir /
        "water_projection_f32_1024x128x720.raw";
    const auto reconstruction_path = artifact_dir /
        "water_fdk_reconstruction_f32_512x512x400.raw";
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(reconstruction_path, reconstruction);
    std::ofstream metadata(artifact_dir / "water_fdk.json");
    metadata << "{\n"
        << "  \"volume_xyz\": [512, 512, 400],\n"
        << "  \"voxel_mm_xyz\": [0.3, 0.3, 0.3],\n"
        << "  \"water_diameter_mm\": 130, \"shell_thickness_mm\": 10,\n"
        << "  \"detector_uv\": [1024, 128],\n"
        << "  \"detector_pixel_mm_uv\": [0.417, 0.417],\n"
        << "  \"sid_mm\": 440, \"sdd_mm\": 770,\n"
        << "  \"views\": 720, \"scan_degrees\": 360,\n"
        << "  \"filter\": \"RamLak\", \"fp\": \"Joseph\",\n"
        << "  \"correlation\": " << metrics.correlation
        << ", \"nrmse\": " << metrics.nrmse << ",\n"
        << "  \"water_mean\": " << material.water_mean
        << ", \"shell_mean\": " << material.shell_mean
        << ", \"shell_peak\": " << material.shell_peak << "\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    std::printf("Large water FDK: corr %.6f NRMSE %.6f, "
        "water %.6f shell %.6f peak %.6f, FP %.1f ms prepare %.1f ms "
        "FDK %.1f ms, GPU %.1f MiB: %s\n", metrics.correlation,
        metrics.nrmse, material.water_mean, material.shell_mean,
        material.shell_peak, fp_ms, prepare_ms, recon_ms, used_mib,
        ok ? "PASS" : "FAIL");
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}
