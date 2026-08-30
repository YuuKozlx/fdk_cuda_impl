#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "CylFpBp/analytic/YkCylAnalyticReconstruction.hpp"
#include "CylFpBp/fp/YkCylForwardOperator.hpp"
#include "FDK/YkFdkPipeline.hpp"
#include "Heli/iter/YkHelicalCylIterativeReconstructor.hpp"
#include "Heli/iter/YkHelicalFlatIterativeReconstructor.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "YKCBCT/interface/YkSystemReconstruction.hpp"
#include "common/YkExecutionContext.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace {

using namespace YK;

SVolumeGridSpec sampleVolume()
{
    return {64, 64, 64, 1.f, 1.f, 1.f, make_float3(0.f, 0.f, 0.f)};
}

SFlatDetectorSpec sampleFlatDetector()
{
    SFlatDetectorSpec detector{};
    detector.channels = 128;
    detector.rows = 96;
    detector.channel_size_mm = 1.f;
    detector.row_size_mm = 1.f;
    return detector;
}

SCylDetectorSpec sampleCylDetector()
{
    SCylDetectorSpec detector{};
    detector.channels = 128;
    detector.rows = 96;
    detector.channel_arc_mm = 1.f;
    detector.row_size_mm = 1.f;
    // 解析路径直接使用规范的源中心等角探测器 R=SDD；迭代路径本身不要求。
    detector.curvature_radius_mm = 300.f;
    return detector;
}

SRegularCircularScanSpec sampleCircularScan()
{
    SRegularCircularScanSpec scan{};
    scan.total_views = 90;
    scan.views_per_turn = 90;
    scan.sid_mm = 160.f;
    scan.sdd_mm = 300.f;
    return scan;
}

SRegularHelicalScanSpec sampleHelicalScan()
{
    SRegularHelicalScanSpec scan{};
    scan.total_views = 120;
    scan.views_per_turn = 90;
    scan.sid_mm = 160.f;
    scan.sdd_mm = 300.f;
    scan.start_z_mm = -16.f;
    scan.pitch_mm_per_turn = 24.f;
    return scan;
}

template <typename System, typename Geometry>
SReconstructionParams makeInternalParams(const System& system,
    const std::vector<Geometry>& geometry)
{
    SReconstructionParams params{};
    params.scan.Nu = system.detector.channels;
    params.scan.Nv = system.detector.rows;
    params.scan.NAng = static_cast<int>(geometry.size());
    params.scan.totalViews = params.scan.NAng;
    params.scan.sid_mm = system.scan.sid_mm;
    params.scan.sdd_mm = system.scan.sdd_mm;
    params.scan.start_angle_rad = system.scan.start_angle_rad;
    params.scan.direction = system.scan.rotation_direction;
    params.scan.range_rad = regularScanRangeRad(system.scan);
    params.scan.angles.resize(geometry.size());
    for (size_t i = 0; i < geometry.size(); ++i) {
        if constexpr (std::is_same_v<Geometry, SConeProjGeomVec>)
            params.scan.angles[i] = geometry[i].angle.x;
        else
            params.scan.angles[i] = cylViewAngle(geometry[i]);
    }
    SVolGeom volume{};
    buildVolumeGeometry(system.volume, volume);
    params.volume.Nx = volume.Nx; params.volume.Ny = volume.Ny;
    params.volume.Nz = volume.Nz;
    params.volume.voxelX_mm = volume.vox_x;
    params.volume.voxelY_mm = volume.vox_y;
    params.volume.voxelZ_mm = volume.vox_z;
    params.volume.centerX_mm = volume.center.x;
    params.volume.centerY_mm = volume.center.y;
    params.volume.centerZ_mm = volume.center.z;
    return params;
}

Mem::Tex3DHandle makeVolumeTexture(const float* device_volume,
    const SVolGeom& volume, cudaStream_t stream)
{
    auto texture = Mem::TextureController::createEmptyTex3D(volume.Nx,
        volume.Ny, volume.Nz, cudaFilterModeLinear, cudaAddressModeBorder);
    Mem::TextureController::updateTex3DFromDeviceAsync(texture, device_volume,
        volume.Nx, volume.Ny, volume.Nz, stream);
    return texture;
}

bool simulateFlat(const SReconstructionParams& params,
    const std::vector<SConeProjGeomVec>& geometry, const float* truth,
    float* projection, cudaStream_t stream)
{
    GeometryContext geometry_context;
    ResourceContext resources;
    resources.attach(stream, 0);
    auto forward = makeForwardOperator(ETask::FP_Joseph);
    const bool ok = geometry_context.initialize(params, geometry) &&
        forward && forward->prepare(geometry_context, resources) &&
        forward->apply(truth, params, projection, resources);
    if (forward) forward->release();
    return ok;
}

bool simulateCyl(const SVolGeom& volume, int channels, int rows,
    const std::vector<SCylConeProjGeomVec>& geometry, const float* truth,
    float* projection, cudaStream_t stream)
{
    auto texture = makeVolumeTexture(truth, volume, stream);
    CylFpBp::Config config{};
    config.samples_per_voxel = 2.f;
    CylFpBp::CylForwardOperator forward;
    const bool ok = forward.prepare(volume, channels, rows, geometry, config) &&
        forward.forward(texture, projection, stream);
    forward.release();
    return ok;
}

bool writeRaw(const std::filesystem::path& path,
    const std::vector<float>& data)
{
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(output);
}

} // namespace

// 从宏观系统参数开始的四几何重建样例：
//   Static Flat -> FDK；Static Cyl -> Cyl analytic FDK；
//   Heli Flat/Cyl -> OSSART。输入统一为 ASTRA modified 3-D Shepp-Logan。
int main_system_reconstruction_four_geometries()
{
    using namespace YK;
    SStaticFlatReconstructionRequest static_flat{};
    static_flat.system.scan = sampleCircularScan();
    static_flat.system.detector = sampleFlatDetector();
    static_flat.system.volume = sampleVolume();
    static_flat.reconstruction.fdk.filter = EFdkFilter::SheppLogan;
    static_flat.reconstruction.parker.mode = EParkerMode::Auto;

    SStaticCylReconstructionRequest static_cyl{};
    static_cyl.system.scan = sampleCircularScan();
    static_cyl.system.detector = sampleCylDetector();
    static_cyl.system.volume = sampleVolume();
    // 当前 Cyl analytic FDK 使用柱面采样修正的离散 Ram-Lak，尚未实现
    // Shepp-Logan 窗；这里的 Shepp-Logan 指四条路径共用的输入模体。
    static_cyl.reconstruction.fdk.filter = EFdkFilter::RamLak;

    SHelicalFlatReconstructionRequest helical_flat{};
    helical_flat.system.scan = sampleHelicalScan();
    helical_flat.system.detector = sampleFlatDetector();
    helical_flat.system.volume = sampleVolume();
    helical_flat.reconstruction.pipeline = EPipeline::OSSART;
    helical_flat.reconstruction.iterative = {3, 0.25f, 12};

    SHelicalCylReconstructionRequest helical_cyl{};
    helical_cyl.system.scan = sampleHelicalScan();
    helical_cyl.system.detector = sampleCylDetector();
    helical_cyl.system.volume = sampleVolume();
    helical_cyl.reconstruction.pipeline = EPipeline::OSSART;
    helical_cyl.reconstruction.iterative = {3, 0.2f, 12};

    std::vector<SConeProjGeomVec> static_flat_geometry, helical_flat_geometry;
    std::vector<SCylConeProjGeomVec> static_cyl_geometry, helical_cyl_geometry;
    SVolGeom volume{};
    bool ok = buildStaticFlatGeometry(static_flat.system,
            static_flat_geometry, volume) &&
        buildStaticCylGeometry(static_cyl.system, static_cyl_geometry) &&
        buildHelicalFlatGeometry(helical_flat.system, helical_flat_geometry) &&
        buildHelicalCylGeometry(helical_cyl.system, helical_cyl_geometry);
    if (!ok) return 1;

    const auto phantom_params = makeInternalParams(static_flat.system,
        static_flat_geometry);
    const auto truth = TestPhantom::makeAstraSheppLogan3D(
        phantom_params, 28.f, true, 0.02f);
    const size_t volume_elements = truth.size();
    const size_t static_projection_elements = static_cast<size_t>(
        static_flat.system.detector.channels) *
        static_flat.system.detector.rows * static_flat_geometry.size();
    const size_t helical_projection_elements = static_cast<size_t>(
        helical_flat.system.detector.channels) *
        helical_flat.system.detector.rows * helical_flat_geometry.size();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    Mem::MemoryController memory;
    auto d_truth = memory.allocateDevice3D<float>(volume.Nx, volume.Ny,
        volume.Nz, 0, false);
    auto d_static_flat_projection = memory.allocateDevice3D<float>(
        static_flat.system.detector.channels, static_flat.system.detector.rows,
        static_cast<int>(static_flat_geometry.size()), 0, false);
    auto d_static_cyl_projection = memory.allocateDevice3D<float>(
        static_cyl.system.detector.channels, static_cyl.system.detector.rows,
        static_cast<int>(static_cyl_geometry.size()), 0, false);
    auto d_helical_flat_projection = memory.allocateDevice3D<float>(
        helical_flat.system.detector.channels, helical_flat.system.detector.rows,
        static_cast<int>(helical_flat_geometry.size()), 0, false);
    auto d_helical_cyl_projection = memory.allocateDevice3D<float>(
        helical_cyl.system.detector.channels, helical_cyl.system.detector.rows,
        static_cast<int>(helical_cyl_geometry.size()), 0, false);
    YK_CUDA_CHECK(cudaMemcpyAsync(d_truth.data(), truth.data(),
        volume_elements * sizeof(float), cudaMemcpyHostToDevice, stream));

    const auto static_flat_params = makeInternalParams(static_flat.system,
        static_flat_geometry);
    const auto helical_flat_params = makeInternalParams(helical_flat.system,
        helical_flat_geometry);
    const bool static_flat_fp = simulateFlat(static_flat_params,
        static_flat_geometry, d_truth.data(), d_static_flat_projection.data(),
        stream);
    const bool static_cyl_fp = simulateCyl(volume,
            static_cyl.system.detector.channels,
            static_cyl.system.detector.rows, static_cyl_geometry,
            d_truth.data(), d_static_cyl_projection.data(), stream);
    const bool helical_flat_fp = simulateFlat(helical_flat_params,
        helical_flat_geometry, d_truth.data(), d_helical_flat_projection.data(),
        stream);
    const bool helical_cyl_fp = simulateCyl(volume,
            helical_cyl.system.detector.channels,
            helical_cyl.system.detector.rows, helical_cyl_geometry,
            d_truth.data(), d_helical_cyl_projection.data(), stream);
    const bool fp_sync = cudaStreamSynchronize(stream) == cudaSuccess;
    YK_LOGI("[SystemExample] FP static-flat={} static-cyl={} heli-flat={} heli-cyl={} sync={}",
        static_flat_fp, static_cyl_fp, helical_flat_fp, helical_cyl_fp, fp_sync);
    ok = static_flat_fp && static_cyl_fp && helical_flat_fp &&
        helical_cyl_fp && fp_sync;
    if (!ok) { cudaStreamDestroy(stream); return 1; }

    std::vector<float> host_static_flat_projection(static_projection_elements);
    YK_CUDA_CHECK(cudaMemcpy(host_static_flat_projection.data(),
        d_static_flat_projection.data(), static_projection_elements *
        sizeof(float), cudaMemcpyDeviceToHost));

    auto d_static_flat = memory.allocateDevice3D<float>(volume.Nx, volume.Ny,
        volume.Nz, 0, false);
    auto d_static_cyl = memory.allocateDevice3D<float>(volume.Nx, volume.Ny,
        volume.Nz, 0, false);
    auto d_helical_flat = memory.allocateDevice3D<float>(volume.Nx, volume.Ny,
        volume.Nz, 0, false);
    auto d_helical_cyl = memory.allocateDevice3D<float>(volume.Nx, volume.Ny,
        volume.Nz, 0, false);

    FdkPipeline flat_fdk;
    const FdkProjectionBatch flat_batch{host_static_flat_projection.data(),
        nullptr, nullptr, static_cast<int>(static_flat_geometry.size())};
    const bool static_flat_recon = flat_fdk.prepare(static_flat, 32, stream) &&
        flat_fdk.processBatchSync(flat_batch, d_static_flat.data(), true);
    ok = static_flat_recon && ok;
    flat_fdk.release();

    CylFpBp::CylFdkPipeline cyl_fdk;
    const bool static_cyl_recon = cyl_fdk.prepare(volume,
            static_cyl.system.detector.channels,
            static_cyl.system.detector.rows, static_cyl_geometry,
            SFilterKernelDesc::RamLak(), stream) &&
        cyl_fdk.reconstruct(d_static_cyl_projection.data(),
            d_static_cyl.data(), true);
    const bool static_cyl_sync = cudaStreamSynchronize(stream) == cudaSuccess;
    ok = static_cyl_recon && static_cyl_sync && ok;
    cyl_fdk.release();

    Helical::Iterative::FlatConfig flat_iter_config{};
    flat_iter_config.method = Helical::Iterative::EMethod::Ossart;
    flat_iter_config.algebraic.iterations =
        helical_flat.reconstruction.iterative.iterations;
    flat_iter_config.algebraic.subset_count =
        helical_flat.reconstruction.iterative.subsets;
    flat_iter_config.algebraic.relaxation =
        helical_flat.reconstruction.iterative.relaxation;
    Helical::Iterative::FlatReconstructor flat_iter;
    const bool helical_flat_recon = flat_iter.prepare(helical_flat_params,
            helical_flat_geometry,
            flat_iter_config, stream) &&
        flat_iter.reconstruct(d_helical_flat_projection.data(),
            d_helical_flat.data());
    ok = helical_flat_recon && ok;
    flat_iter.release();

    Helical::Iterative::CylConfig cyl_iter_config{};
    cyl_iter_config.method = Helical::Iterative::EMethod::Ossart;
    cyl_iter_config.algebraic.iterations =
        helical_cyl.reconstruction.iterative.iterations;
    cyl_iter_config.algebraic.subset_count =
        helical_cyl.reconstruction.iterative.subsets;
    cyl_iter_config.algebraic.relaxation =
        helical_cyl.reconstruction.iterative.relaxation;
    Helical::Iterative::CylReconstructor cyl_iter;
    const bool helical_cyl_recon = cyl_iter.prepare(volume,
            helical_cyl.system.detector.channels,
            helical_cyl.system.detector.rows, helical_cyl_geometry,
            cyl_iter_config, stream) &&
        cyl_iter.reconstruct(d_helical_cyl_projection.data(),
            d_helical_cyl.data());
    cyl_iter.release();
    const bool reconstruction_sync = cudaStreamSynchronize(stream) == cudaSuccess;
    ok = helical_cyl_recon && reconstruction_sync && ok;
    YK_LOGI("[SystemExample] Recon static-flat={} static-cyl={}/{} heli-flat={} heli-cyl={} sync={}",
        static_flat_recon, static_cyl_recon, static_cyl_sync,
        helical_flat_recon, helical_cyl_recon, reconstruction_sync);

    std::vector<std::vector<float>> images(4,
        std::vector<float>(volume_elements));
    const float* device_results[] = {d_static_flat.data(), d_static_cyl.data(),
        d_helical_flat.data(), d_helical_cyl.data()};
    for (size_t i = 0; ok && i < images.size(); ++i)
        ok = cudaMemcpy(images[i].data(), device_results[i],
            volume_elements * sizeof(float), cudaMemcpyDeviceToHost) ==
            cudaSuccess;
    for (size_t i = 0; i < images.size(); ++i) {
        float maximum = 0.f;
        bool finite = true;
        for (const float value : images[i]) {
            finite = finite && std::isfinite(value);
            maximum = std::max(maximum, std::fabs(value));
        }
        YK_LOGI("[SystemExample] result {} finite={} max={:.6e}",
            i, finite, maximum);
        ok = finite && maximum > 1e-8f && ok;
    }

    const auto output = std::filesystem::absolute(
        "out/test-artifacts/system-reconstruction-four-geometries");
    std::filesystem::create_directories(output);
    const char* names[] = {"static-flat-fdk", "static-cyl-fdk",
        "helical-flat-ossart", "helical-cyl-ossart"};
    ok = writeRaw(output / "truth-shepp-logan.raw", truth) && ok;
    for (size_t i = 0; i < images.size(); ++i)
        ok = writeRaw(output / (std::string(names[i]) + ".raw"), images[i]) && ok;
    std::vector<TestImage::GrayPanel> panels = {
        {&truth, 64, 64, 64, 32, 1.f, 0.f, 0.022f, false},
        {&images[0], 64, 64, 64, 32, 1.f, 0.f, 0.022f, false},
        {&images[1], 64, 64, 64, 32, 1.f, 0.f, 0.022f, false},
        {&images[2], 64, 64, 64, 32, 1.f, 0.f, 0.022f, false},
        {&images[3], 64, 64, 64, 32, 1.f, 0.f, 0.022f, false}
    };
    ok = TestImage::writeGrayMontageBmp(output / "comparison.bmp", panels,
        5, 8, 4) && ok;
    YK_LOGI("[SystemExample] outputs: {}", output.string());
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
