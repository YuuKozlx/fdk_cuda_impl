#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include "common/YkExecutionContext.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "YkTestImage.hpp"

namespace {

YK::SSystemConfig makeSystem(float rotation_x_rad, bool enabled = true)
{
    YK::SSystemConfig system{};
    system.detector = YK::EDetectorKind::Flat;
    system.trajectory = YK::ETrajectoryKind::Circular;
    system.circular.total_views = 1;
    system.circular.views_per_turn = 2;
    system.circular.sid_mm = 80.f;
    system.circular.sdd_mm = 160.f;
    system.flat_detector.channels = 256;
    system.flat_detector.rows = 256;
    system.flat_detector.channel_size_mm = 1.f;
    system.flat_detector.row_size_mm = 1.f;
    system.volume = {256, 256, 256, 0.5f, 0.5f, 0.5f,
        make_float3(0.f, 0.f, 0.f)};
    system.forward_projection_pose.enabled = enabled;
    system.forward_projection_pose.rotation_x_rad = rotation_x_rad;
    return system;
}

bool run(const YK::SSystemConfig& system, const std::vector<float>& volume,
    std::vector<float>& projection)
{
    YK::PreparedGeometry geometry;
    YK::ResourceContext resources;
    if (!geometry.initialize(system) || !resources.initialize(0)) return false;
    YK::Mem::MemoryController memory;
    auto d_volume = memory.allocateDevice3D<float>(system.volume.nx,
        system.volume.ny, system.volume.nz, 0, false);
    auto h_volume = memory.allocateCpu3D<float>(system.volume.nx,
        system.volume.ny, system.volume.nz, false);
    std::copy(volume.begin(), volume.end(), h_volume.data());
    memory.upload3D(d_volume, h_volume);
    auto d_projection = memory.allocateDevice3D<float>(
        system.flat_detector.channels, system.flat_detector.rows, 1, 0, true);
    auto forward = YK::makeForwardOperator(YK::ETask::FP_Joseph);
    const auto params = geometry.base();
    const bool ok = forward && forward->prepare(geometry, resources) &&
        forward->apply(d_volume.data(), params, d_projection.data(), resources);
    if (ok) {
        YK_CUDA_CHECK(cudaStreamSynchronize(resources.stream()));
        YK_CUDA_CHECK(cudaMemcpy(projection.data(), d_projection.data(),
            projection.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (forward) forward->release();
    resources.release();
    return ok;
}

bool writeRaw(const std::filesystem::path& path,
    const std::vector<float>& values)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

} // namespace

int main_fp_volume_pose_api()
{
    constexpr int nx = 256, ny = 256, nz = 256;
    constexpr int nu = 256, nv = 256;
    constexpr float pi = 3.14159265358979323846f;
    std::vector<float> volume(static_cast<size_t>(nx) * ny * nz, 0.f);
    // Finite cylindrical water phantom. Rotation about X changes its axial
    // extent in the projection while retaining a standard water phantom.
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            const float dx = static_cast<float>(x - nx / 2);
            const float dy = static_cast<float>(y - ny / 2);
            if (dx * dx + dy * dy <= 120.f * 120.f)
                volume[(static_cast<size_t>(z) * ny + y) * nx + x] = 1.f;
        }
    std::vector<float> baseline(static_cast<size_t>(nu) * nv, 0.f);
    std::vector<float> repeated(static_cast<size_t>(nu) * nv, 0.f);
    std::vector<float> rotated(static_cast<size_t>(nu) * nv, 0.f);

    const bool launched = run(makeSystem(0.f, false), volume, baseline) &&
        run(makeSystem(0.f), volume, repeated) &&
        run(makeSystem(0.25f * pi), volume, rotated);
    if (!launched) return 1;

    float baseline_delta = 0.f;
    float rotation_delta = 0.f;
    float projection_max = 0.f;
    std::vector<float> difference(baseline.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        if (!std::isfinite(baseline[i]) || !std::isfinite(rotated[i])) return 1;
        baseline_delta = std::max(baseline_delta,
            std::fabs(baseline[i] - repeated[i]));
        rotation_delta = std::max(rotation_delta,
            std::fabs(baseline[i] - rotated[i]));
        difference[i] = std::fabs(baseline[i] - rotated[i]);
        projection_max = std::max(projection_max,
            std::max(baseline[i], rotated[i]));
    }
    const auto output_dir = std::filesystem::path("out/test-artifacts/fp-volume-pose");
    const bool saved = writeRaw(output_dir / "water-cylinder-no-pose-256x256x1-f32.raw",
            baseline) &&
        writeRaw(output_dir / "water-cylinder-rot-x45-256x256x1-f32.raw", rotated) &&
        writeRaw(output_dir / "water-cylinder-abs-difference-256x256x1-f32.raw",
            difference) &&
        YK::TestImage::writeGrayMontageBmp(output_dir /
            "water-cylinder-no-pose-vs-rot-x45.bmp", {
                {&baseline, nu, nv, 1, 0, 1.f, 0.f, projection_max, false},
                {&rotated, nu, nv, 1, 0, 1.f, 0.f, projection_max, false},
                {&difference, nu, nv, 1, 0, 1.f, 0.f, rotation_delta, false}},
            3, 8, 2);
    const bool ok = baseline_delta <= 1e-5f && rotation_delta > 1e-4f && saved;
    std::printf("FP volume pose API: zero_delta=%.6e rotation_delta=%.6e, %s\n",
        baseline_delta, rotation_delta, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
