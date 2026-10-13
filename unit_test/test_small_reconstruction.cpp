#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda_runtime_api.h>

#include "YKCBCT/interface/YkReconstructionApi.hpp"

namespace {

using Session = std::unique_ptr<YK::IReconstructionSession,
    decltype(&YK::ReconstructionSessionFactory::destroy)>;

Session makeSession()
{
    return Session(YK::ReconstructionSessionFactory::create(),
        &YK::ReconstructionSessionFactory::destroy);
}

YK::SSystemSpec smallCircularSystem(YK::EPipeline pipeline)
{
    YK::SSystemSpec system{};
    system.geometry.detector = YK::EDetectorKind::Flat;
    system.geometry.trajectory = YK::ETrajectoryKind::Circular;
    system.geometry.circular.total_views = 96;
    system.geometry.circular.views_per_turn = 96;
    system.geometry.circular.sid_mm = 100.f;
    system.geometry.circular.sdd_mm = 200.f;
    system.geometry.flat_detector.channels = 48;
    system.geometry.flat_detector.rows = 32;
    system.geometry.flat_detector.channel_size_mm = 1.f;
    system.geometry.flat_detector.row_size_mm = 1.f;
    system.geometry.volume = {24, 24, 16, 1.f, 1.f, 1.f,
        make_float3(0.f, 0.f, 0.f)};
    system.reconstruction.pipeline = pipeline;
    system.reconstruction.projection_model = YK::EProjectionModel::Joseph;
    system.reconstruction.fdk.filter = YK::EFdkFilter::RamLak;
    return system;
}

std::vector<float> makeAnalyticPhantom(const YK::SVolumeGridSpec& grid)
{
    std::vector<float> result(static_cast<size_t>(grid.nx) * grid.ny * grid.nz,
        0.f);
    for (int z = 0; z < grid.nz; ++z) {
        const float pz = (z + 0.5f - 0.5f * grid.nz) * grid.voxel_z_mm;
        for (int y = 0; y < grid.ny; ++y) {
            const float py = (y + 0.5f - 0.5f * grid.ny) * grid.voxel_y_mm;
            for (int x = 0; x < grid.nx; ++x) {
                const float px = (x + 0.5f - 0.5f * grid.nx) * grid.voxel_x_mm;
                float value = px * px / 49.f + py * py / 64.f +
                    pz * pz / 25.f <= 1.f ? 0.02f : 0.f;
                if ((px - 3.f) * (px - 3.f) + (py + 2.f) * (py + 2.f) +
                        pz * pz <= 2.25f)
                    value = 0.06f;
                result[(static_cast<size_t>(z) * grid.ny + y) * grid.nx + x] =
                    value;
            }
        }
    }
    return result;
}

YK::Buffer hostBuffer(std::vector<float>& values)
{
    return {values.data(), YK::EMemoryLocation::Host, values.size()};
}

double cosineSimilarity(const std::vector<float>& lhs,
    const std::vector<float>& rhs)
{
    double product = 0.0, lhs_energy = 0.0, rhs_energy = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        product += static_cast<double>(lhs[i]) * rhs[i];
        lhs_energy += static_cast<double>(lhs[i]) * lhs[i];
        rhs_energy += static_cast<double>(rhs[i]) * rhs[i];
    }
    return product / std::sqrt(std::max(lhs_energy * rhs_energy, 1e-30));
}

TEST(SmallReconstruction, AnalyticEllipsoidsRoundTripThroughFpAndFdk)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device is unavailable";

    auto system = smallCircularSystem(YK::EPipeline::ForwardProjection);
    auto truth = makeAnalyticPhantom(system.geometry.volume);
    const size_t view_elements =
        static_cast<size_t>(system.geometry.flat_detector.channels) *
        system.geometry.flat_detector.rows;
    std::vector<float> projection(view_elements *
        system.geometry.circular.total_views, 0.f);
    std::vector<float> reconstruction(truth.size(), 0.f);

    auto forward = makeSession();
    ASSERT_NE(forward, nullptr);
    ASSERT_TRUE(forward->initialize(system)) << forward->lastErrorMessage();
    YK::SExecutionRequest request{};
    request.volume = hostBuffer(truth);
    request.projection = hostBuffer(projection);
    request.view_count = system.geometry.circular.total_views;
    ASSERT_TRUE(forward->execute(request)) << forward->lastErrorMessage();
    EXPECT_TRUE(std::all_of(projection.begin(), projection.end(),
        [](float value) { return std::isfinite(value) && value >= 0.f; }));
    EXPECT_GT(*std::max_element(projection.begin(), projection.end()), 0.f);
    forward.reset();

    system.reconstruction.pipeline = YK::EPipeline::FDK;
    auto fdk = makeSession();
    ASSERT_NE(fdk, nullptr);
    ASSERT_TRUE(fdk->initialize(system)) << fdk->lastErrorMessage();
    request.volume = hostBuffer(reconstruction);
    ASSERT_TRUE(fdk->execute(request)) << fdk->lastErrorMessage();
    ASSERT_TRUE(std::all_of(reconstruction.begin(), reconstruction.end(),
        [](float value) { return std::isfinite(value); }));
    EXPECT_GT(*std::max_element(reconstruction.begin(), reconstruction.end()),
        0.f);
    EXPECT_GT(cosineSimilarity(truth, reconstruction), 0.70);
}

TEST(SmallReconstruction, RejectsUndersizedBuffersBeforeExecution)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device is unavailable";

    auto system = smallCircularSystem(YK::EPipeline::ForwardProjection);
    auto session = makeSession();
    ASSERT_NE(session, nullptr);
    ASSERT_TRUE(session->initialize(system)) << session->lastErrorMessage();
    float placeholder = 0.f;
    YK::SExecutionRequest request{};
    request.view_count = system.geometry.circular.total_views;
    request.volume = {&placeholder, YK::EMemoryLocation::Host, 1};
    request.projection = {&placeholder, YK::EMemoryLocation::Host, 1};
    EXPECT_FALSE(session->execute(request));
    EXPECT_EQ(session->lastError(), YK::EApiErrorCode::InvalidBuffer);
}

#if YKCBCT_TEST_HAS_HELICAL
TEST(SmallReconstruction, WfbpAcceptsBothPhysicalBedFeedDirections)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device is unavailable";

    for (float bed_feed : {24.f, -24.f}) {
        YK::SSystemSpec system{};
        system.geometry.detector = YK::EDetectorKind::Flat;
        system.geometry.trajectory = YK::ETrajectoryKind::Helical;
        system.geometry.helical.total_views = 192;
        system.geometry.helical.views_per_turn = 64;
        system.geometry.helical.start_angle_rad = 0.37f;
        system.geometry.helical.start_z_mm = 11.f;
        system.geometry.helical.pitch_mm_per_turn = bed_feed;
        system.geometry.helical.sid_mm = 500.f;
        system.geometry.helical.sdd_mm = 1000.f;
        system.geometry.flat_detector.channels = 16;
        system.geometry.flat_detector.rows = 8;
        system.geometry.flat_detector.channel_size_mm = 1.f;
        system.geometry.flat_detector.row_size_mm = 1.f;
        system.geometry.volume = {8, 8, 8, 1.f, 1.f, 1.f,
            make_float3(0.f, 0.f, 0.f)};
        system.reconstruction.pipeline = YK::EPipeline::WFBP;

        auto session = makeSession();
        ASSERT_NE(session, nullptr);
        ASSERT_TRUE(session->initialize(system))
            << "bed feed " << bed_feed << ": " << session->lastErrorMessage();
        const size_t projection_count = static_cast<size_t>(
            system.geometry.flat_detector.channels) *
            system.geometry.flat_detector.rows *
            system.geometry.helical.total_views;
        const size_t voxel_count = static_cast<size_t>(
            system.geometry.volume.nx) * system.geometry.volume.ny *
            system.geometry.volume.nz;
        std::vector<float> projection(projection_count, 0.f);
        std::vector<float> volume(voxel_count, 0.f);
        YK::SExecutionRequest request{};
        request.projection = hostBuffer(projection);
        request.volume = hostBuffer(volume);
        request.view_count = system.geometry.helical.total_views;
        ASSERT_TRUE(session->execute(request))
            << "bed feed " << bed_feed << ": " << session->lastErrorMessage();
        EXPECT_TRUE(std::all_of(volume.begin(), volume.end(), [](float value) {
            return std::isfinite(value) && std::fabs(value) <= 1e-6f;
        }));
    }
}
#endif

} // namespace
