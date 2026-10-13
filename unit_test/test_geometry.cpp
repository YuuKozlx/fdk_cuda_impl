#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"
#include "YKCBCT/interface/YkSystemReconstruction.hpp"

namespace {

constexpr float kTwoPi = 6.28318530717958647692f;

YK::SFlatDetectorSpec smallFlatDetector()
{
    YK::SFlatDetectorSpec detector{};
    detector.channels = 32;
    detector.rows = 16;
    detector.channel_size_mm = 0.8f;
    detector.row_size_mm = 1.2f;
    return detector;
}

YK::SCylDetectorSpec smallCylDetector()
{
    YK::SCylDetectorSpec detector{};
    detector.channels = 32;
    detector.rows = 16;
    detector.channel_arc_mm = 0.8f;
    detector.row_size_mm = 1.2f;
    detector.curvature_radius_mm = 850.f;
    return detector;
}

struct BedFeedCase {
    float bed_feed_mm;
    int rotation_direction;
    float start_angle_rad;
    float start_z_mm;
    float source_offset_z_mm;
};

class HelicalBedFeedTest : public ::testing::TestWithParam<BedFeedCase> {};

TEST_P(HelicalBedFeedTest, ConvertsPhysicalBedFeedToFixedObjectVectors)
{
    using namespace YK;
    const BedFeedCase test = GetParam();
    SRegularHelicalScanSpec scan{};
    scan.total_views = 11;
    scan.views_per_turn = 8;
    scan.start_angle_rad = test.start_angle_rad;
    scan.rotation_direction = test.rotation_direction;
    scan.sid_mm = 500.f;
    scan.sdd_mm = 1000.f;
    scan.start_z_mm = test.start_z_mm;
    scan.pitch_mm_per_turn = test.bed_feed_mm;
    scan.source_offset_mm = make_float3(0.f, 0.f, test.source_offset_z_mm);

    SHelicalTrajectorySpec trajectory{};
    ASSERT_TRUE(buildHelicalTrajectory(scan, trajectory));
    const float vector_pitch = -test.bed_feed_mm * test.rotation_direction;
    EXPECT_FLOAT_EQ(trajectory.angles_rad.front(), test.start_angle_rad);
    EXPECT_FLOAT_EQ(trajectory.pitch_mm_per_turn, vector_pitch);
    EXPECT_NEAR(trajectory.start_z_mm,
        test.start_z_mm - vector_pitch * test.start_angle_rad / kTwoPi, 1e-5f);

    std::vector<SConeProjGeomVec> flat;
    std::vector<SCylConeProjGeomVec> cylindrical;
    ASSERT_TRUE(buildProjectionGeometry(trajectory, smallFlatDetector(), flat));
    ASSERT_TRUE(buildProjectionGeometry(trajectory, smallCylDetector(), cylindrical));
    ASSERT_EQ(flat.size(), static_cast<size_t>(scan.total_views));
    ASSERT_EQ(cylindrical.size(), flat.size());
    for (int view = 0; view < scan.total_views; ++view) {
        const float expected_source_z = test.start_z_mm +
            test.source_offset_z_mm - test.bed_feed_mm * view /
            scan.views_per_turn;
        const float relative_z = -test.bed_feed_mm * view / scan.views_per_turn;
        EXPECT_NEAR(flat[view].src.z, expected_source_z, 1e-4f);
        EXPECT_NEAR(cylindrical[view].source.z, expected_source_z, 1e-4f);
        EXPECT_NEAR(flat[view].detS.z - flat.front().detS.z, relative_z, 1e-4f);
        EXPECT_NEAR(cylindrical[view].detectorCenter.z -
            cylindrical.front().detectorCenter.z, relative_z, 1e-4f);
    }
}

INSTANTIATE_TEST_SUITE_P(AllSigns, HelicalBedFeedTest,
    ::testing::Values(
        BedFeedCase{24.f, 1, 0.37f, 11.f, 2.5f},
        BedFeedCase{24.f, -1, -0.41f, -7.f, -1.25f},
        BedFeedCase{-18.f, 1, 1.13f, 5.f, 0.75f},
        BedFeedCase{-18.f, -1, -0.83f, -3.f, -2.f}));

TEST(HelicalGeometry, ExplicitTrajectoryKeepsSignedVectorPitch)
{
    using namespace YK;
    for (float pitch : {-30.f, 30.f}) {
        SHelicalTrajectorySpec trajectory{};
        trajectory.angles_rad = {0.4f, 1.1f, 2.3f};
        trajectory.sid_mm = 500.f;
        trajectory.sdd_mm = 1000.f;
        trajectory.start_z_mm = -4.f;
        trajectory.pitch_mm_per_turn = pitch;
        std::vector<SConeProjGeomVec> geometry;
        ASSERT_TRUE(buildProjectionGeometry(trajectory, smallFlatDetector(),
            geometry));
        for (size_t view = 0; view < geometry.size(); ++view) {
            EXPECT_NEAR(geometry[view].src.z, trajectory.start_z_mm +
                pitch * trajectory.angles_rad[view] / kTwoPi, 1e-5f);
        }
    }
}

TEST(GeometryValidation, RejectsInvalidTrajectoryDetectorAndVolume)
{
    using namespace YK;
    SRegularHelicalScanSpec scan{};
    scan.total_views = 8;
    scan.views_per_turn = 8;
    scan.rotation_direction = 1;
    scan.sid_mm = 500.f;
    scan.sdd_mm = 1000.f;
    scan.pitch_mm_per_turn = 20.f;
    SHelicalTrajectorySpec trajectory{};
    EXPECT_TRUE(buildHelicalTrajectory(scan, trajectory));

    auto invalid_scan = scan;
    invalid_scan.total_views = 0;
    EXPECT_FALSE(buildHelicalTrajectory(invalid_scan, trajectory));
    invalid_scan = scan;
    invalid_scan.rotation_direction = 0;
    EXPECT_FALSE(buildHelicalTrajectory(invalid_scan, trajectory));
    invalid_scan = scan;
    invalid_scan.sdd_mm = invalid_scan.sid_mm;
    EXPECT_FALSE(buildHelicalTrajectory(invalid_scan, trajectory));
    invalid_scan = scan;
    invalid_scan.pitch_mm_per_turn = 0.f;
    EXPECT_FALSE(buildHelicalTrajectory(invalid_scan, trajectory));
    invalid_scan = scan;
    invalid_scan.start_angle_rad = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(buildHelicalTrajectory(invalid_scan, trajectory));

    ASSERT_TRUE(buildHelicalTrajectory(scan, trajectory));
    std::vector<SConeProjGeomVec> flat_geometry;
    auto flat = smallFlatDetector();
    flat.channel_size_mm = 0.f;
    EXPECT_FALSE(buildProjectionGeometry(trajectory, flat, flat_geometry));
    std::vector<SCylConeProjGeomVec> cylindrical_geometry;
    auto cylindrical = smallCylDetector();
    cylindrical.curvature_radius_mm = -1.f;
    EXPECT_FALSE(buildProjectionGeometry(trajectory, cylindrical,
        cylindrical_geometry));

    SVolGeom volume{};
    EXPECT_TRUE(buildVolumeGeometry({8, 9, 10, 0.5f, 0.6f, 0.7f,
        make_float3(1.f, 2.f, 3.f)}, volume));
    EXPECT_EQ(volume.Nx, 8);
    EXPECT_FLOAT_EQ(volume.center.z, 3.f);
    EXPECT_FALSE(buildVolumeGeometry({0, 9, 10, 0.5f, 0.6f, 0.7f,
        make_float3(0.f, 0.f, 0.f)}, volume));
}

TEST(GeometryConventions, DetectorFramesAreRightHandedTowardSource)
{
    using namespace YK;
    SCircularTrajectorySpec trajectory{};
    trajectory.angles_rad = {0.f, 0.7f};
    trajectory.sid_mm = 500.f;
    trajectory.sdd_mm = 1000.f;
    std::vector<SConeProjGeomVec> flat;
    std::vector<SCylConeProjGeomVec> cylindrical;
    ASSERT_TRUE(buildProjectionGeometry(trajectory, smallFlatDetector(), flat));
    ASSERT_TRUE(buildProjectionGeometry(trajectory, smallCylDetector(), cylindrical));

    const auto check = [](const float4& source, const float4& center,
        const float4& u, const float4& v) {
        const float3 normal = make_float3(u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
        const float3 toward_source = make_float3(source.x - center.x,
            source.y - center.y, source.z - center.z);
        return normal.x * toward_source.x + normal.y * toward_source.y +
            normal.z * toward_source.z;
    };
    const auto flat_detector = smallFlatDetector();
    const float center_u = 0.5f * (flat_detector.channels - 1);
    const float center_v = 0.5f * (flat_detector.rows - 1);
    for (size_t view = 0; view < flat.size(); ++view) {
        const float4 center = make_float4(
            flat[view].detS.x + center_u * flat[view].detU.x +
                center_v * flat[view].detV.x,
            flat[view].detS.y + center_u * flat[view].detU.y +
                center_v * flat[view].detV.y,
            flat[view].detS.z + center_u * flat[view].detU.z +
                center_v * flat[view].detV.z, 0.f);
        EXPECT_GT(check(flat[view].src, center, flat[view].detU,
            flat[view].detV), 0.f);
        EXPECT_GT(check(cylindrical[view].source,
            cylindrical[view].detectorCenter, cylindrical[view].detectorU,
            cylindrical[view].detectorV), 0.f);
    }
}

TEST(ReconstructionPolicy, ParkerAutoUsesOnlyShortFlatCircularScans)
{
    using namespace YK;
    SSystemConfig system{};
    system.detector = EDetectorKind::Flat;
    system.trajectory = ETrajectoryKind::Circular;
    system.circular.total_views = 180;
    system.circular.views_per_turn = 360;
    SReconstructionSpec reconstruction{};
    reconstruction.fdk.parker.mode = EParkerMode::Auto;
    EXPECT_TRUE(resolveParkerEnabled(system, reconstruction));
    system.circular.total_views = 360;
    EXPECT_FALSE(resolveParkerEnabled(system, reconstruction));
    system.detector = EDetectorKind::Cylindrical;
    EXPECT_FALSE(resolveParkerEnabled(system, reconstruction));
    reconstruction.fdk.parker.mode = EParkerMode::Enabled;
    EXPECT_FALSE(resolveParkerEnabled(system, reconstruction));
}

} // namespace
