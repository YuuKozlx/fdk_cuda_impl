#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFdkPipeline.hpp"
#include "Reconstruction/Iterative/Flat/YkAlgebraicReconstructor.hpp"
#include "Reconstruction/Iterative/Flat/YkCglsReconstructor.hpp"
#include "Reconstruction/Iterative/Flat/YkPwlsReconstructor.hpp"
#include "YkTestImage.hpp"
#include "common/YkProjectionOperators.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

struct WaterFdkIterativeOptions {
    std::string method = "ossart";
    int iterations = 10;
    int subsets = 10;
    float relaxation = 0.25f;
    float relative_residual_tolerance = 0.f;
    int minimum_iterations = 1;
    int convergence_check_interval = 1;
    int convergence_patience = 1;
};

WaterFdkIterativeOptions g_water_fdk_iterative_options{};

bool writeFloatRaw(const std::filesystem::path& path,
    const std::vector<float>& values)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

SReconstructionParams makeParams()
{
    SReconstructionParams p{};
    p.volume.Nx = 512; p.volume.Ny = 512; p.volume.Nz = 400;
    p.volume.voxelX_mm = 0.3f; p.volume.voxelY_mm = 0.3f; p.volume.voxelZ_mm = 0.3f;
    p.scan.Nu = 1024; p.scan.Nv = 128;
    p.scan.du_mm = 0.417f; p.scan.dv_mm = 0.417f;
    p.scan.sid_mm = 440.f; p.scan.sdd_mm = 770.f;
    p.scan.NAng = 720; p.scan.totalViews = 720;
    p.scan.start_angle_rad = 0.f;
    p.scan.range_rad = 2.f * CUDA_PI;
    p.scan.short_scan = false;
    p.scan.offsetU_mm = 0.f; p.scan.offsetV_mm = 0.f;
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = 2.f * CUDA_PI * static_cast<float>(i) / p.scan.NAng;
    return p;
}

// 缩小后的封闭水模：水区直径 130 mm，外壳厚 10 mm，外径 150 mm。
// 为避免 Z 边界贴住体积，外部高度 110 mm，内部水柱高度 90 mm。
std::vector<float> makeWaterPhantom(const SReconstructionParams& p)
{
    constexpr float water_radius_mm = 65.f;
    constexpr float shell_outer_radius_mm = 75.f;
    constexpr float water_half_height_mm = 45.f;
    constexpr float shell_outer_half_height_mm = 55.f;
    constexpr float water_mu = 0.020f;
    constexpr float shell_mu = 0.040f;
    std::vector<float> volume(static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz, 0.f);
    for (int z = 0; z < p.volume.Nz; ++z) {
        const float pz = (z + 0.5f - 0.5f * p.volume.Nz) * p.volume.voxelZ_mm;
        for (int y = 0; y < p.volume.Ny; ++y) {
            const float py = (y + 0.5f - 0.5f * p.volume.Ny) * p.volume.voxelY_mm;
            for (int x = 0; x < p.volume.Nx; ++x) {
                const float px = (x + 0.5f - 0.5f * p.volume.Nx) * p.volume.voxelX_mm;
                const float radius2 = px * px + py * py;
                float value = 0.f;
                if (radius2 <= shell_outer_radius_mm * shell_outer_radius_mm &&
                    std::fabs(pz) <= shell_outer_half_height_mm) {
                    value = shell_mu;
                    if (radius2 <= water_radius_mm * water_radius_mm &&
                        std::fabs(pz) <= water_half_height_mm)
                        value = water_mu;
                }
                volume[(static_cast<size_t>(z) * p.volume.Ny + y) * p.volume.Nx + x] = value;
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
    const SReconstructionParams& p, float scale)
{
    double water_sum = 0.0, shell_sum = 0.0;
    size_t water_count = 0, shell_count = 0;
    MaterialMetrics result{};
    const int z = p.volume.Nz / 2;
    for (int y = 0; y < p.volume.Ny; ++y) {
        const float py = (y + 0.5f - 0.5f * p.volume.Ny) * p.volume.voxelY_mm;
        for (int x = 0; x < p.volume.Nx; ++x) {
            const float px = (x + 0.5f - 0.5f * p.volume.Nx) * p.volume.voxelX_mm;
            const float radius = std::sqrt(px * px + py * py);
            const float value = scale * reconstruction[
                (static_cast<size_t>(z) * p.volume.Ny + y) * p.volume.Nx + x];
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
    const std::vector<float>& reconstruction, const SReconstructionParams& p,
    float half_span_mm)
{
    double truth_norm = 0.0, recon_norm = 0.0, dot = 0.0;
    for (int z = 0; z < p.volume.Nz; ++z) {
        const float pz = (z + 0.5f - 0.5f * p.volume.Nz) * p.volume.voxelZ_mm;
        if (std::fabs(pz) > half_span_mm) continue;
        const size_t begin = static_cast<size_t>(z) * p.volume.Nx * p.volume.Ny;
        const size_t end = begin + static_cast<size_t>(p.volume.Nx) * p.volume.Ny;
        for (size_t i = begin; i < end; ++i) {
            truth_norm += static_cast<double>(truth[i]) * truth[i];
            recon_norm += static_cast<double>(reconstruction[i]) * reconstruction[i];
            dot += static_cast<double>(truth[i]) * reconstruction[i];
        }
    }
    Metrics result{};
    result.scale = recon_norm > 1e-30 ? static_cast<float>(dot / recon_norm) : 0.f;
    double error_norm = 0.0;
    for (int z = 0; z < p.volume.Nz; ++z) {
        const float pz = (z + 0.5f - 0.5f * p.volume.Nz) * p.volume.voxelZ_mm;
        if (std::fabs(pz) > half_span_mm) continue;
        const size_t begin = static_cast<size_t>(z) * p.volume.Nx * p.volume.Ny;
        const size_t end = begin + static_cast<size_t>(p.volume.Nx) * p.volume.Ny;
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

void configure_large_water_fdk_iterative_test(const std::string& method,
    int iterations, int subsets, float relaxation,
    float relative_residual_tolerance, int minimum_iterations,
    int convergence_check_interval, int convergence_patience)
{
    g_water_fdk_iterative_options.method = method;
    g_water_fdk_iterative_options.iterations = iterations;
    g_water_fdk_iterative_options.subsets = subsets;
    g_water_fdk_iterative_options.relaxation = relaxation;
    g_water_fdk_iterative_options.relative_residual_tolerance =
        relative_residual_tolerance;
    g_water_fdk_iterative_options.minimum_iterations = minimum_iterations;
    g_water_fdk_iterative_options.convergence_check_interval =
        convergence_check_interval;
    g_water_fdk_iterative_options.convergence_patience = convergence_patience;
}

int main_large_water_pwls()
{
    const SReconstructionParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.Nu) * p.scan.Nv * p.scan.NAng;
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
        auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(), volume_count * sizeof(float),
            cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ok = ok && cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;

        ForwardOperatorAdapter fp;
        Iter::PwlsConfig config{};
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
        Iter::PwlsReconstructor pwls;
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
    const double axial_fov_mm = p.scan.Nv * p.scan.dv_mm * p.scan.sid_mm / p.scan.sdd_mm;
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
    const SReconstructionParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.Nu) * p.scan.Nv * p.scan.NAng;
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
        auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, 0);
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
    const SReconstructionParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.Nu) * p.scan.Nv * p.scan.NAng;
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
        auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, 0);
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
                    projection.data(), &geometry, nullptr, p.scan.NAng
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

int main_large_water_fdk_iterative()
{
    const SReconstructionParams p = makeParams();
    const auto truth = makeWaterPhantom(p);
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);
    const size_t volume_count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    const size_t projection_count = static_cast<size_t>(p.scan.Nu) * p.scan.Nv * p.scan.NAng;
    std::vector<float> projection(projection_count);
    std::vector<float> fdk_initial(volume_count);
    std::vector<float> reconstruction(volume_count);

    size_t free_before = 0, total_memory = 0, free_during = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) return 1;

    float fp_ms = 0.f;
    float fdk_prepare_ms = 0.f, fdk_recon_ms = 0.f;
    float iterative_prepare_ms = 0.f, iterative_recon_ms = 0.f;
    Metrics fdk_metrics{};
    Metrics final_metrics{};
    MaterialMetrics fdk_material{};
    MaterialMetrics final_material{};
    unsigned int subset_updates = 0;
    Iter::IterativeConvergenceStatistics convergence_statistics{};
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) ==
            cudaSuccess;

        ForwardOperatorAdapter fp;
        FdkPipeline fdk;
        cudaEvent_t start = nullptr, fp_stop = nullptr;
        cudaEvent_t fdk_prepare_stop = nullptr, fdk_stop = nullptr;
        cudaEvent_t iterative_prepare_stop = nullptr, iterative_stop = nullptr;
        ok = ok && cudaEventCreate(&start) == cudaSuccess &&
            cudaEventCreate(&fp_stop) == cudaSuccess &&
            cudaEventCreate(&fdk_prepare_stop) == cudaSuccess &&
            cudaEventCreate(&fdk_stop) == cudaSuccess &&
            cudaEventCreate(&iterative_prepare_stop) == cudaSuccess &&
            cudaEventCreate(&iterative_stop) == cudaSuccess &&
            fp.init(p, geometry, ETask::FP_Joseph, 0, stream);
        if (ok) {
            cudaEventRecord(start, stream);
            ok = fp.run(d_truth.data(), p, d_projection.data(), stream);
            cudaEventRecord(fp_stop, stream);
            ok = ok && cudaEventSynchronize(fp_stop) == cudaSuccess;
            if (ok)
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;

            ok = ok && fdk.prepareWithGeometry(p, geometry, 64, stream);
            cudaEventRecord(fdk_prepare_stop, stream);
            if (ok) {
                const FdkProjectionBatch batch{
                    projection.data(), &geometry, nullptr, p.scan.NAng
                };
                ok = fdk.processBatch(batch, d_reconstruction.data(), true);
                cudaEventRecord(fdk_stop, stream);
                ok = ok && cudaEventSynchronize(fdk_stop) == cudaSuccess;
            }
            if (ok) {
                ok = cudaMemcpy(fdk_initial.data(), d_reconstruction.data(),
                    volume_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                fdk_metrics = compareCentralSlab(truth, fdk_initial, p, 12.f);
                fdk_material = measureMaterials(fdk_initial, p, fdk_metrics.scale);
            }

            if (ok && g_water_fdk_iterative_options.method == "ossart") {
                Iter::AlgebraicReconstructionConfig config{};
                config.method = Iter::EAlgebraicMethod::Ossart;
                config.weight_model = Iter::EAlgebraicWeightModel::DetailedSubset;
                config.subset_order = Iter::EAlgebraicSubsetOrder::GoldenRatio;
                config.iterations = g_water_fdk_iterative_options.iterations;
                config.subset_count = g_water_fdk_iterative_options.subsets;
                config.relaxation = g_water_fdk_iterative_options.relaxation;
                config.relaxation_reduction = 1.f;
                config.convergence.relative_residual_tolerance =
                    g_water_fdk_iterative_options.relative_residual_tolerance;
                config.convergence.minimum_iterations =
                    g_water_fdk_iterative_options.minimum_iterations;
                config.convergence.check_interval =
                    g_water_fdk_iterative_options.convergence_check_interval;
                config.convergence.patience =
                    g_water_fdk_iterative_options.convergence_patience;
                config.use_min = true;
                config.min_constraint = 0.f;
                config.fp_task = ETask::FP_Joseph;
                config.bp_task = ETask::BP_Joseph_v3;
                Iter::AlgebraicReconstructorEx reconstructor;
                ok = reconstructor.prepare(p, geometry, config, stream);
                cudaEventRecord(iterative_prepare_stop, stream);
                ok = ok && reconstructor.reconstruct(
                    d_projection.data(), d_reconstruction.data());
                cudaEventRecord(iterative_stop, stream);
                ok = ok && cudaEventSynchronize(iterative_stop) == cudaSuccess;
                subset_updates = reconstructor.totalSubsetUpdates();
                convergence_statistics = reconstructor.convergenceStatistics();
                reconstructor.release();
            }
            else if (ok) {
                Iter::CglsReconstructionConfig config{};
                config.strategy = Iter::ECglsStrategy::RobustRestart;
                config.iterations = g_water_fdk_iterative_options.iterations;
                config.epsilon = 1e-8f;
                config.restart_on_divergence = true;
                config.convergence.relative_residual_tolerance =
                    g_water_fdk_iterative_options.relative_residual_tolerance;
                config.convergence.minimum_iterations =
                    g_water_fdk_iterative_options.minimum_iterations;
                config.convergence.check_interval =
                    g_water_fdk_iterative_options.convergence_check_interval;
                config.convergence.patience =
                    g_water_fdk_iterative_options.convergence_patience;
                config.use_min = true;
                config.min_constraint = 0.f;
                config.fp_task = ETask::FP_Joseph;
                config.bp_task = ETask::BP_Joseph_v3;
                Iter::CglsReconstructorEx reconstructor;
                ok = reconstructor.prepare(p, geometry, config, stream);
                cudaEventRecord(iterative_prepare_stop, stream);
                ok = ok && reconstructor.reconstruct(
                    d_projection.data(), d_reconstruction.data());
                cudaEventRecord(iterative_stop, stream);
                ok = ok && cudaEventSynchronize(iterative_stop) == cudaSuccess;
                convergence_statistics = reconstructor.convergenceStatistics();
                reconstructor.release();
            }

            if (ok) {
                cudaEventElapsedTime(&fp_ms, start, fp_stop);
                cudaEventElapsedTime(&fdk_prepare_ms, fp_stop, fdk_prepare_stop);
                cudaEventElapsedTime(&fdk_recon_ms, fdk_prepare_stop, fdk_stop);
                cudaEventElapsedTime(&iterative_prepare_ms, fdk_stop,
                    iterative_prepare_stop);
                cudaEventElapsedTime(&iterative_recon_ms, iterative_prepare_stop,
                    iterative_stop);
                ok = cudaMemcpy(reconstruction.data(), d_reconstruction.data(),
                    volume_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                if (ok) {
                    final_metrics = compareCentralSlab(truth, reconstruction, p, 12.f);
                    final_material = measureMaterials(reconstruction, p,
                        final_metrics.scale);
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                            [](float value) { return std::isfinite(value); }) &&
                        std::isfinite(final_metrics.correlation) &&
                        std::isfinite(final_metrics.nrmse) &&
                        final_metrics.correlation > 0.70 &&
                        final_metrics.nrmse < 0.75;
                }
            }
        }
        cudaMemGetInfo(&free_during, &total_memory);
        if (start) cudaEventDestroy(start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (fdk_prepare_stop) cudaEventDestroy(fdk_prepare_stop);
        if (fdk_stop) cudaEventDestroy(fdk_stop);
        if (iterative_prepare_stop) cudaEventDestroy(iterative_prepare_stop);
        if (iterative_stop) cudaEventDestroy(iterative_stop);
        fdk.release();
        fp.release();
    }
    cudaStreamDestroy(stream);

    const std::string method = g_water_fdk_iterative_options.method;
    const auto artifact_dir = std::filesystem::absolute(
        std::string("out/test-artifacts/large-water-fdk-") + method);
    const auto phantom_path = artifact_dir / "water_shell_f32_512x512x400.raw";
    const auto projection_path = artifact_dir /
        "water_projection_f32_1024x128x720.raw";
    const auto fdk_path = artifact_dir /
        "water_fdk_initial_f32_512x512x400.raw";
    const auto reconstruction_path = artifact_dir /
        (std::string("water_fdk_") + method + "_f32_512x512x400.raw");
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(fdk_path, fdk_initial) &&
        writeFloatRaw(reconstruction_path, reconstruction);
    std::vector<TestImage::GrayPanel> panels;
    for (const int z : { p.volume.Nz / 2 - 20, p.volume.Nz / 2, p.volume.Nz / 2 + 20 }) {
        panels.push_back({ &truth, p.volume.Nx, p.volume.Ny, p.volume.Nz, z,
            1.f, 0.f, 0.045f, false });
        panels.push_back({ &fdk_initial, p.volume.Nx, p.volume.Ny, p.volume.Nz, z,
            fdk_metrics.scale, 0.f, 0.045f, false });
        panels.push_back({ &reconstruction, p.volume.Nx, p.volume.Ny, p.volume.Nz, z,
            final_metrics.scale, 0.f, 0.045f, false });
    }
    const auto montage_path = artifact_dir /
        (std::string("water_truth_fdk_") + method + ".bmp");
    artifacts_ok = artifacts_ok &&
        TestImage::writeGrayMontageBmp(montage_path, panels, 3, 4, 1);
    std::filesystem::create_directories(artifact_dir);
    std::ofstream metadata(artifact_dir /
        (std::string("water_fdk_") + method + ".json"));
    const float total_recon_ms = fdk_prepare_ms + fdk_recon_ms +
        iterative_prepare_ms + iterative_recon_ms;
    metadata << "{\n"
        << "  \"volume_xyz\": [512, 512, 400],\n"
        << "  \"voxel_mm_xyz\": [0.3, 0.3, 0.3],\n"
        << "  \"detector_uv\": [1024, 128], \"views\": 720,\n"
        << "  \"sid_mm\": 440, \"sdd_mm\": 770, \"scan_degrees\": 360,\n"
        << "  \"initialization\": \"FDK\", \"iterative_method\": \""
        << method << "\",\n"
        << "  \"iterations\": " << g_water_fdk_iterative_options.iterations
        << ", \"subsets\": " << g_water_fdk_iterative_options.subsets
        << ", \"relaxation\": " << g_water_fdk_iterative_options.relaxation
        << ", \"relative_residual_tolerance\": "
        << g_water_fdk_iterative_options.relative_residual_tolerance
        << ", \"minimum_iterations\": "
        << g_water_fdk_iterative_options.minimum_iterations
        << ", \"convergence_check_interval\": "
        << g_water_fdk_iterative_options.convergence_check_interval
        << ", \"convergence_patience\": "
        << g_water_fdk_iterative_options.convergence_patience
        << ",\n"
        << "  \"fp_ms\": " << fp_ms
        << ", \"fdk_prepare_ms\": " << fdk_prepare_ms
        << ", \"fdk_reconstruction_ms\": " << fdk_recon_ms
        << ", \"iterative_prepare_ms\": " << iterative_prepare_ms
        << ", \"iterative_reconstruction_ms\": " << iterative_recon_ms
        << ", \"total_reconstruction_ms\": " << total_recon_ms << ",\n"
        << "  \"subset_updates\": " << subset_updates << ",\n"
        << "  \"completed_iterations\": "
        << convergence_statistics.completed_iterations
        << ", \"convergence_checks\": "
        << convergence_statistics.convergence_checks
        << ", \"projection_residual_l2\": "
        << convergence_statistics.projection_residual_l2
        << ", \"relative_projection_residual\": "
        << convergence_statistics.relative_projection_residual
        << ", \"stopped_by_relative_residual\": "
        << (convergence_statistics.stopped_by_relative_residual ? "true" : "false")
        << ", \"stopped_by_relative_update\": "
        << (convergence_statistics.stopped_by_relative_update ? "true" : "false")
        << ", \"stopped_by_stagnation\": "
        << (convergence_statistics.stopped_by_stagnation ? "true" : "false")
        << ",\n"
        << "  \"fdk_correlation\": " << fdk_metrics.correlation
        << ", \"fdk_nrmse\": " << fdk_metrics.nrmse << ",\n"
        << "  \"final_correlation\": " << final_metrics.correlation
        << ", \"final_nrmse\": " << final_metrics.nrmse << ",\n"
        << "  \"fdk_water_mean\": " << fdk_material.water_mean
        << ", \"fdk_shell_mean\": " << fdk_material.shell_mean << ",\n"
        << "  \"final_water_mean\": " << final_material.water_mean
        << ", \"final_shell_mean\": " << final_material.shell_mean << "\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    std::printf("Large water FDK -> %s: FDK corr %.6f NRMSE %.6f, "
        "final corr %.6f NRMSE %.6f\n", method.c_str(),
        fdk_metrics.correlation, fdk_metrics.nrmse,
        final_metrics.correlation, final_metrics.nrmse);
    std::printf("  FP %.1f ms, FDK prepare %.1f ms recon %.1f ms, "
        "iter prepare %.1f ms recon %.1f ms, total recon %.1f ms, "
        "GPU %.1f MiB: %s\n", fp_ms, fdk_prepare_ms, fdk_recon_ms,
        iterative_prepare_ms, iterative_recon_ms, total_recon_ms,
        used_mib, ok ? "PASS" : "FAIL");
    std::printf("  convergence: completed %d/%d, checks %d, rel-residual %.6g, "
        "stopped(residual=%d, update=%d, stagnation=%d)\n",
        convergence_statistics.completed_iterations,
        g_water_fdk_iterative_options.iterations,
        convergence_statistics.convergence_checks,
        convergence_statistics.relative_projection_residual,
        convergence_statistics.stopped_by_relative_residual,
        convergence_statistics.stopped_by_relative_update,
        convergence_statistics.stopped_by_stagnation);
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}
