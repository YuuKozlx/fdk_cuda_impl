#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

#include "YkTestImage.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtGeometry.hpp"
#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtPipeline.hpp"

namespace {
using namespace YK;

bool allFinite(const std::vector<float>& values)
{
    for (float value : values)
        if (!std::isfinite(value)) return false;
    return true;
}

float positiveDisplayMaximum(const std::vector<float>& values)
{
    std::vector<float> positive;
    positive.reserve(values.size());
    for (float value : values)
        if (std::isfinite(value) && value > 0.f) positive.push_back(value);
    if (positive.empty()) return 1.f;
    const size_t index = std::min(positive.size() - 1,
        static_cast<size_t>(0.995 * positive.size()));
    std::nth_element(positive.begin(), positive.begin() + index, positive.end());
    return std::max(positive[index], 1e-8f);
}

std::vector<float> makeHorizontalPcb(const SReconstructionParams& p)
{
    std::vector<float> volume(static_cast<size_t>(p.volume.Nx) *
        p.volume.Ny * p.volume.Nz, 0.f);
    constexpr int samples = 4;
    constexpr float board_width = 100.f, board_height = 100.f;
    constexpr float spacing_x = 2.4f * (board_width / 14.f);
    constexpr float spacing_y = 2.4f * (board_height / 10.f);
    constexpr float feature_scale = board_width / 14.f;
    constexpr float pad_radius = 0.60f * feature_scale;
    for (int z = 0; z < p.volume.Nz; ++z) {
        for (int y = 0; y < p.volume.Ny; ++y) {
            for (int x = 0; x < p.volume.Nx; ++x) {
                float sum = 0.f;
                for (int sz = 0; sz < samples; ++sz) {
                    for (int sy = 0; sy < samples; ++sy) {
                        for (int sx = 0; sx < samples; ++sx) {
                            const float wx = (x + (sx + 0.5f) / samples -
                                0.5f * p.volume.Nx) * p.volume.voxelX_mm;
                            const float wy = (y + (sy + 0.5f) / samples -
                                0.5f * p.volume.Ny) * p.volume.voxelY_mm;
                            const float wz = (z + (sz + 0.5f) / samples -
                                0.5f * p.volume.Nz) * p.volume.voxelZ_mm;
                            float value = std::fabs(wx) <= 0.5f * board_width &&
                                std::fabs(wy) <= 0.5f * board_height &&
                                std::fabs(wz) <= 0.25f ? 0.03f : 0.f;
                            for (int iy = -2; iy <= 2; ++iy) {
                                for (int ix = -2; ix <= 2; ++ix) {
                                    const float dx = wx - spacing_x * ix;
                                    const float dy = wy - spacing_y * iy;
                                    const bool in_pad = dx * dx + dy * dy <=
                                        pad_radius * pad_radius &&
                                        wz >= 0.25f && wz <= 0.65f;
                                    if (in_pad) value = 0.60f;
                                    const float da_x = dx - 0.12f * feature_scale;
                                    const float da_y = dy + 0.08f * feature_scale;
                                    const float db_x = dx + 0.10f * feature_scale;
                                    const float db_y = dy - 0.10f * feature_scale;
                                    const bool defect_a = ix == -1 && iy == 1 &&
                                        da_x * da_x + da_y * da_y <=
                                        std::pow(0.24f * feature_scale, 2.f);
                                    const bool defect_b = ix == 1 && iy == -1 &&
                                        db_x * db_x + db_y * db_y <=
                                        std::pow(0.18f * feature_scale, 2.f);
                                    if ((defect_a || defect_b) &&
                                        wz >= 0.25f && wz <= 0.65f)
                                        value = 0.f;
                                }
                            }
                            if (std::fabs(wx) <= 25.f && std::fabs(wy) <= 2.2f &&
                                wz >= 0.25f && wz <= 0.42f)
                                value = 0.60f;
                            sum += value;
                        }
                    }
                }
                const size_t index = (static_cast<size_t>(z) * p.volume.Ny + y) *
                    p.volume.Nx + x;
                volume[index] = sum /
                    static_cast<float>(samples * samples * samples);
            }
        }
    }
    return volume;
}

} // namespace

int main_nikon_tilted_pcb_reconstruction()
{
    namespace fs = std::filesystem;
    constexpr int detector_u = 512, detector_v = 512, views = 180;
    SReconstructionParams p{};
    p.scan.Nu = detector_u; p.scan.Nv = detector_v;
    p.scan.NAng = p.scan.totalViews = views;
    p.scan.du_mm = p.scan.dv_mm = 0.417f;
    p.scan.sid_mm = 200.f; p.scan.sdd_mm = 300.f;
    p.scan.range_rad = 2.f * CUDA_PI;
    p.scan.angles.resize(views);
    for (int i = 0; i < views; ++i)
        p.scan.angles[i] = 2.f * CUDA_PI * i / views;
    p.volume.Nx = 416; p.volume.Ny = 416; p.volume.Nz = 24;
    p.volume.voxelX_mm = p.volume.voxelY_mm = 0.25f;
    p.volume.voxelZ_mm = 0.1f;

    NonStandardFpBp::NikonTiltedCt::GeometryConfig config{};
    config.detector_u = detector_u; config.detector_v = detector_v;
    config.detector_pixel_u_mm = config.detector_pixel_v_mm = 0.417f;
    config.sid_mm = p.scan.sid_mm; config.sdd_mm = p.scan.sdd_mm;
    config.tilt_rad = 30.f * CUDA_PI / 180.f;
    config.tilt_axis_azimuth_rad = 0.f;
    config.source = make_float3(0.f, -p.scan.sid_mm, 0.f);
    config.rotation_axis_pivot = make_float3(0.f, 0.f, 0.f);
    std::vector<SConeProjGeomVec> geometry;
    std::vector<NonStandardFpBp::NikonTiltedCt::ViewPose> poses;
    bool ok = NonStandardFpBp::NikonTiltedCt::buildTiltedAxisGeometry(
        config, p.scan.angles, poses, geometry);

    const auto truth = makeHorizontalPcb(p);
    const size_t volume_n = truth.size();
    const size_t projection_n = static_cast<size_t>(detector_u) * detector_v * views;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
        p.volume.Nz, 0);
    auto d_projection = memory.allocateDevice3D<float>(detector_u, detector_v,
        views, 0);
    auto d_reconstruction = memory.allocateDevice3D<float>(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, 0, true);
    cudaStream_t stream = nullptr;
    ok = ok && cudaStreamCreate(&stream) == cudaSuccess;
    GeometryContext context;
    ResourceContext resources;
    ok = ok && context.initialize(p, geometry);
    resources.attach(stream, 0);
    auto forward = makeForwardOperator(ETask::FP_Joseph);
    ok = ok && cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_n * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    ok = ok && forward->prepare(context, resources) &&
        forward->apply(d_truth.data(), p, d_projection.data(), resources) &&
        cudaStreamSynchronize(stream) == cudaSuccess;

    NonStandardFpBp::NikonTiltedCt::Pipeline pipeline;
    ok = ok && pipeline.prepare(p, geometry, stream, 0) &&
        pipeline.process(d_projection.data(), d_reconstruction.data(), true);
    std::vector<float> projection(projection_n), reconstruction(volume_n);
    ok = ok && cudaMemcpy(projection.data(), d_projection.data(),
        projection_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(reconstruction.data(), d_reconstruction.data(),
            volume_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
        allFinite(projection) && allFinite(reconstruction);
    bool projection_nonzero = false, reconstruction_nonzero = false;
    for (float value : projection)
        projection_nonzero = projection_nonzero || std::abs(value) > 1e-7f;
    for (float value : reconstruction)
        reconstruction_nonzero = reconstruction_nonzero || std::abs(value) > 1e-7f;
    ok = ok && projection_nonzero && reconstruction_nonzero;

    const fs::path output_dir = fs::path(YKCBCT_TEST_SOURCE_DIR) /
        "output" / "nikon_tilted_ct";
    fs::create_directories(output_dir);
    const auto write_raw = [](const fs::path& path,
        const std::vector<float>& values) {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        return output.good();
    };
    ok = write_raw(output_dir / "pcb_tilt30_projection_f32.raw", projection) && ok;
    ok = write_raw(output_dir / "pcb_tilt30_reconstruction_f32.raw",
        reconstruction) && ok;
    const int selected[] = {0, views / 4, views / 2, 3 * views / 4};
    std::vector<TestImage::GrayPanel> projection_panels;
    const float projection_max = positiveDisplayMaximum(projection);
    for (int view : selected)
        projection_panels.push_back({&projection, detector_u, detector_v,
            views, view, 1.f, 0.f, projection_max});
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "pcb_tilt30_projection_views_0_90_180_270.bmp",
        projection_panels, 2, 8, 1) && ok;
    const int pad_slice = p.volume.Nz / 2 + 4;
    const float reconstruction_max = positiveDisplayMaximum(reconstruction);
    const std::vector<TestImage::GrayPanel> reconstruction_panels = {
        {&truth, p.volume.Nx, p.volume.Ny, p.volume.Nz, pad_slice,
            1.f, 0.f, 0.60f},
        {&reconstruction, p.volume.Nx, p.volume.Ny, p.volume.Nz, pad_slice,
            1.f, 0.f, reconstruction_max}
    };
    ok = TestImage::writeGrayMontageBmp(output_dir /
        "pcb_tilt30_truth_reconstruction_xy.bmp",
        reconstruction_panels, 2, 8, 1) && ok;
    std::ofstream metadata(output_dir / "metadata.txt");
    metadata << "views=180\ntilt_deg=30\nsid_mm=200\nsdd_mm=300\n"
        << "detector=512x512\npixel_mm=0.417\npcb_size_mm=100x100\n"
        << "pcb_plane=world XY, normal=world +Z\n"
        << "scan_axis=world +Z through volume center\n"
        << "initial_pose_tilt_axis=+X, tilt_deg=30\n"
        << "reconstruction=nikon tilted direct ConeVec analytic pipeline\n"
        << "reconstruction_volume=416x416x24\n"
        << "reconstruction_voxel_mm=0.25x0.25x0.1\n";
    ok = metadata.good() && ok;

    std::printf("nikon tilted PCB projection/reconstruction: %s output=%s\n",
        ok ? "PASS" : "FAIL", output_dir.string().c_str());
    pipeline.release();
    forward->release();
    resources.release();
    if (stream) cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
