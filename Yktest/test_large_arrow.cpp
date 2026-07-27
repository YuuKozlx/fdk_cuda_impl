#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "Iter/YkTigreGradientReconstructor.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

struct LargeArrowOptions {
    std::string method = "os-asd-pocs";
    int iterations = 6;
    int block_size = 20;
    float lambda = 0.25f;
    int tv_iterations = 5;
};

LargeArrowOptions g_options{};

struct MethodSpec {
    Iter::ETigreGradientAlgorithm algorithm;
    const char* slug;
    const char* display_name;
    bool uses_os_block;
    bool uses_tv;
};

MethodSpec resolveMethod(const std::string& method)
{
    using Algorithm = Iter::ETigreGradientAlgorithm;
    if (method == "sart")
        return { Algorithm::Sart, "sart", "TIGRE SART", false, false };
    if (method == "os-sart")
        return { Algorithm::OsSart, "os-sart", "TIGRE OS-SART", true, false };
    if (method == "sirt")
        return { Algorithm::Sirt, "sirt", "TIGRE SIRT", false, false };
    if (method == "asd-pocs")
        return { Algorithm::AsdPocs, "asd-pocs", "TIGRE ASD-POCS", false, true };
    if (method == "b-asd-pocs-beta")
        return { Algorithm::BAsdPocsBeta, "b-asd-pocs-beta",
            "TIGRE B-ASD-POCS-beta", false, true };
    if (method == "pcsd")
        return { Algorithm::Pcsd, "pcsd", "TIGRE PCSD", false, true };
    if (method == "os-pcsd")
        return { Algorithm::OsPcsd, "os-pcsd", "TIGRE OS-PCSD", true, true };
    if (method == "aw-pcsd")
        return { Algorithm::AwPcsd, "aw-pcsd", "TIGRE AwPCSD", false, true };
    if (method == "os-aw-pcsd")
        return { Algorithm::OsAwPcsd, "os-aw-pcsd",
            "TIGRE OS-AwPCSD", true, true };
    if (method == "aw-asd-pocs")
        return { Algorithm::AwAsdPocs, "aw-asd-pocs",
            "TIGRE Aw-ASD-POCS", false, true };
    if (method == "os-aw-asd-pocs")
        return { Algorithm::OsAwAsdPocs, "os-aw-asd-pocs",
            "TIGRE OS-Aw-ASD-POCS", true, true };
    return { Algorithm::OsAsdPocs, "os-asd-pocs",
        "TIGRE OS-ASD-POCS", true, true };
}

SCBCTParams makeLargeArrowParams()
{
    SCBCTParams p{};
    p.iVX = 160; p.iVY = 160; p.iVZ = 96;
    p.vox_x_mm = 0.6f; p.vox_y_mm = 0.6f; p.vox_z_mm = 0.6f;
    p.iPU = 384; p.iPV = 128;
    p.du_mm = 1.f; p.dv_mm = 1.f;
    p.SID = 400.f; p.SDD = 800.f;
    p.iPAng = 360; p.iPAngTotal = 360;
    p.scan_start_angle_rad = 0.f;
    p.scan_range_rad = 2.f * CUDA_PI;
    p.nDirSign = 1;
    p.bShortScan = false;
    p.offsetU_mm = 0.f; p.offsetV_mm = 0.f;
    p.angle_list.resize(p.iPAng);
    for (int i = 0; i < p.iPAng; ++i)
        p.angle_list[i] = 2.f * CUDA_PI * static_cast<float>(i) / p.iPAng;
    return p;
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

struct Metrics {
    double correlation = 0.0;
    double nrmse = 0.0;
    float scale = 0.f;
};

Metrics compareVolumes(const std::vector<float>& truth,
    const std::vector<float>& reconstruction)
{
    double truth_norm = 0.0, reconstruction_norm = 0.0, dot = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        truth_norm += static_cast<double>(truth[i]) * truth[i];
        reconstruction_norm +=
            static_cast<double>(reconstruction[i]) * reconstruction[i];
        dot += static_cast<double>(truth[i]) * reconstruction[i];
    }
    Metrics result{};
    result.scale = reconstruction_norm > 1e-30
        ? static_cast<float>(dot / reconstruction_norm) : 0.f;
    double error_norm = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double error = result.scale * reconstruction[i] - truth[i];
        error_norm += error * error;
    }
    result.correlation = dot /
        std::sqrt(std::max(truth_norm * reconstruction_norm, 1e-30));
    result.nrmse = std::sqrt(error_norm / std::max(truth_norm, 1e-30));
    return result;
}

} // namespace

void configure_large_arrow_test(const std::string& method, int iterations,
    int block_size, float lambda, int tv_iterations)
{
    g_options.method = method;
    g_options.iterations = iterations;
    g_options.block_size = block_size;
    g_options.lambda = lambda;
    g_options.tv_iterations = tv_iterations;
}

int main_large_arrow_tigre()
{
    const MethodSpec method = resolveMethod(g_options.method);
    const SCBCTParams p = makeLargeArrowParams();
    const auto truth = TestPhantom::makeArrowDirections(p, true);
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t projection_count = static_cast<size_t>(p.iPU) * p.iPV * p.iPAng;
    std::vector<float> projection(projection_count);
    std::vector<float> reconstruction(volume_count);
    std::vector<float> error(volume_count);

    size_t free_before = 0, total_memory = 0, free_during = 0;
    bool ok = cudaMemGetInfo(&free_before, &total_memory) == cudaSuccess;
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) return 1;

    float fp_ms = 0.f, prepare_ms = 0.f, reconstruction_ms = 0.f;
    Iter::TigreGradientStatistics statistics{};
    Metrics metrics{};
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ, 0);
        auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng, 0);
        auto d_reconstruction = memory.allocateDevice3D<float>(p.iVX, p.iVY,
            p.iVZ, 0);
        ok = cudaMemcpyAsync(d_truth.data(), truth.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) ==
            cudaSuccess;
        ok = ok && cudaMemsetAsync(d_reconstruction.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;

        std::vector<SConeProjGeomVec> geometry;
        detail::buildCircularViews(p, geometry);
        ForwardOperatorAdapter fp;
        Iter::TigreGradientReconstructor::Config config{};
        config.algorithm = method.algorithm;
        config.iterations = g_options.iterations;
        config.block_size = g_options.block_size;
        config.lambda = g_options.lambda;
        config.lambda_reduction = 0.98f;
        config.initialization = Iter::ETigreInitialization::Zero;
        config.non_negative = true;
        config.tv_iterations = g_options.tv_iterations;
        config.alpha = 0.002f;
        config.alpha_reduction = 0.95f;
        config.maximum_update_ratio = 0.95f;
        // 固定为 0，使无噪声合成数据的每个外循环都执行数据一致性更新。
        config.max_l2_error = 0.f;
        config.bregman_interval = 1;
        config.fp_task = ETask::FP_Joseph;
        config.bp_task = ETask::BP_Joseph_v3;
        Iter::TigreGradientReconstructor reconstructor;

        cudaEvent_t start = nullptr, fp_stop = nullptr;
        cudaEvent_t prepare_stop = nullptr, reconstruction_stop = nullptr;
        ok = ok && cudaEventCreate(&start) == cudaSuccess &&
            cudaEventCreate(&fp_stop) == cudaSuccess &&
            cudaEventCreate(&prepare_stop) == cudaSuccess &&
            cudaEventCreate(&reconstruction_stop) == cudaSuccess &&
            fp.init(p, geometry, ETask::FP_Joseph, 0, stream);
        if (ok) {
            cudaEventRecord(start, stream);
            ok = fp.run(d_truth.data(), p, d_projection.data(), stream);
            cudaEventRecord(fp_stop, stream);
            ok = ok && reconstructor.prepare(p, config, stream);
            cudaEventRecord(prepare_stop, stream);
            ok = ok && reconstructor.reconstruct(d_projection.data(),
                d_reconstruction.data());
            cudaEventRecord(reconstruction_stop, stream);
            ok = ok && cudaEventSynchronize(reconstruction_stop) == cudaSuccess;
            if (ok) {
                cudaEventElapsedTime(&fp_ms, start, fp_stop);
                cudaEventElapsedTime(&prepare_ms, fp_stop, prepare_stop);
                cudaEventElapsedTime(&reconstruction_ms, prepare_stop,
                    reconstruction_stop);
                ok = cudaMemcpy(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess;
                ok = ok && cudaMemcpy(reconstruction.data(),
                    d_reconstruction.data(), volume_count * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) {
                    statistics = reconstructor.statistics();
                    metrics = compareVolumes(truth, reconstruction);
                    for (size_t i = 0; i < volume_count; ++i)
                        error[i] = metrics.scale * reconstruction[i] - truth[i];
                    const unsigned int expected_updates =
                        static_cast<unsigned int>(statistics.completed_iterations *
                            reconstructor.actualSubsetCount());
                    ok = std::all_of(reconstruction.begin(), reconstruction.end(),
                            [](float value) { return std::isfinite(value); }) &&
                        std::isfinite(metrics.correlation) &&
                        std::isfinite(metrics.nrmse) &&
                        statistics.completed_iterations >= 1 &&
                        statistics.completed_iterations <= config.iterations &&
                        statistics.subset_updates == expected_updates &&
                        metrics.correlation > 0.05 && metrics.nrmse < 1.0;
                }
            }
        }
        cudaMemGetInfo(&free_during, &total_memory);
        if (start) cudaEventDestroy(start);
        if (fp_stop) cudaEventDestroy(fp_stop);
        if (prepare_stop) cudaEventDestroy(prepare_stop);
        if (reconstruction_stop) cudaEventDestroy(reconstruction_stop);
        reconstructor.release();
        fp.release();
    }
    cudaStreamDestroy(stream);

    const auto artifact_dir = std::filesystem::absolute(
        std::string("out/test-artifacts/large-arrow-") + method.slug);
    const auto phantom_path = artifact_dir /
        "arrow_labels_phantom_f32_160x160x96.raw";
    const auto projection_path = artifact_dir /
        "arrow_flat_projection_f32_384x128x360.raw";
    const auto reconstruction_path = artifact_dir /
        (std::string("arrow_") + method.slug + "_f32_160x160x96.raw");
    const auto error_path = artifact_dir /
        (std::string("arrow_") + method.slug + "_error_f32_160x160x96.raw");
    bool artifacts_ok = writeFloatRaw(phantom_path, truth) &&
        writeFloatRaw(projection_path, projection) &&
        writeFloatRaw(reconstruction_path, reconstruction) &&
        writeFloatRaw(error_path, error);

    std::vector<TestImage::GrayPanel> panels;
    for (const int z : { 2, p.iVZ / 2, p.iVZ - 3 }) {
        panels.push_back({ &truth, p.iVX, p.iVY, p.iVZ, z,
            1.f, 0.f, 0.06f, false });
        panels.push_back({ &reconstruction, p.iVX, p.iVY, p.iVZ, z,
            metrics.scale, 0.f, 0.06f, false });
        panels.push_back({ &error, p.iVX, p.iVY, p.iVZ, z,
            1.f, 0.f, 0.03f, true });
    }
    const auto montage_path = artifact_dir / "arrow_z_slices_truth_recon_error.bmp";
    artifacts_ok = artifacts_ok &&
        TestImage::writeGrayMontageBmp(montage_path, panels, 3, 4, 2);

    std::filesystem::create_directories(artifact_dir);
    std::ofstream metadata(artifact_dir /
        (std::string("large_arrow_") + method.slug + ".json"));
    const auto writeFiniteOrNull = [&](float value) {
        if (std::isfinite(value)) metadata << value;
        else metadata << "null";
    };
    metadata << "{\n"
        << "  \"scalar_type\": \"float32-little-endian\",\n"
        << "  \"storage_order\": \"x/u fastest, then y/v, then z/view\",\n"
        << "  \"volume_xyz\": [160, 160, 96],\n"
        << "  \"voxel_mm_xyz\": [0.6, 0.6, 0.6],\n"
        << "  \"phantom\": \"three-axis arrows with labels on six boundary faces\",\n"
        << "  \"arrow_mu_xyz\": [0.02, 0.04, 0.06], \"label_mu\": 0.01,\n"
        << "  \"detector_uv\": [384, 128],\n"
        << "  \"detector_pixel_mm_uv\": [1.0, 1.0],\n"
        << "  \"sid_mm\": 400, \"sdd_mm\": 800,\n"
        << "  \"views\": 360, \"scan_degrees\": 360, \"offset_mm_uv\": [0, 0],\n"
        << "  \"algorithm\": \"" << method.display_name << "\",\n"
        << "  \"method_slug\": \"" << method.slug << "\",\n"
        << "  \"iterations\": " << g_options.iterations
        << ", \"block_size_views\": "
        << (method.uses_os_block ? g_options.block_size :
            (method.algorithm == Iter::ETigreGradientAlgorithm::Sirt ? p.iPAng : 1))
        << ",\n"
        << "  \"lambda\": " << g_options.lambda
        << ", \"lambda_reduction\": 0.98,\n"
        << "  \"tv_iterations\": " << g_options.tv_iterations
        << ", \"uses_tv\": " << (method.uses_tv ? "true" : "false")
        << ", \"alpha\": 0.002,\n"
        << "  \"alpha_reduction\": 0.95, \"maximum_update_ratio\": 0.95,\n"
        << "  \"fp\": \"Joseph\", \"bp\": \"Joseph-v3\",\n"
        << "  \"correlation\": " << metrics.correlation
        << ", \"nrmse\": " << metrics.nrmse
        << ", \"scale\": " << metrics.scale << ",\n"
        << "  \"projection_l2\": ";
    writeFiniteOrNull(statistics.projection_l2);
    metadata << ", \"data_update_l2\": ";
    writeFiniteOrNull(statistics.data_update_l2);
    metadata << ", \"regularization_update_l2\": ";
    writeFiniteOrNull(statistics.regularization_update_l2);
    metadata << ",\n"
        << "  \"fp_ms\": " << fp_ms << ", \"prepare_ms\": " << prepare_ms
        << ", \"reconstruction_ms\": " << reconstruction_ms << "\n"
        << "}\n";
    metadata.close();
    artifacts_ok = artifacts_ok && metadata.good();
    ok = ok && artifacts_ok;

    const double projection_mib = projection_count * sizeof(float) /
        (1024.0 * 1024.0);
    const double volume_mib = volume_count * sizeof(float) /
        (1024.0 * 1024.0);
    const double used_mib = free_before >= free_during
        ? (free_before - free_during) / (1024.0 * 1024.0) : 0.0;
    std::printf("Large arrow %s: volume %dx%dx%d (%.1f MiB), "
        "projection %dx%dx%d (%.1f MiB), GPU workspace %.1f MiB\n",
        method.display_name, p.iVX, p.iVY, p.iVZ, volume_mib,
        p.iPU, p.iPV, p.iPAng,
        projection_mib, used_mib);
    std::printf("  corr %.6f NRMSE %.6f scale %.6f, FP %.1f ms, "
        "prepare %.1f ms, recon %.1f ms, subset updates %u: %s\n",
        metrics.correlation, metrics.nrmse, metrics.scale, fp_ms, prepare_ms,
        reconstruction_ms, statistics.subset_updates, ok ? "PASS" : "FAIL");
    std::printf("  artifacts: %s (%s)\n", artifact_dir.string().c_str(),
        artifacts_ok ? "written" : "FAILED");
    return ok ? 0 : 1;
}
