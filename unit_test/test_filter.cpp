#include <gtest/gtest.h>
#include <global/YkMacro.hpp>
#include "Yktest/Yktest_fdkflter.hpp"
#include <global/YkLog.h>

// ----------------------------------------------------------------
// FilterTest：滤波权重正确性验证
// 数值验证类，有明确阈值
// ----------------------------------------------------------------

TEST(FilterTest, RamLak_Nu512_DefaultParams)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpectra_RamLak(512));
}

TEST(FilterTest, RamLak_Nu256_DefaultParams)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpectra_RamLak(256));
}

TEST(FilterTest, RamLak_Nu1024_DefaultParams)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpectra_RamLak(1024));
}

TEST(FilterTest, RamLak_NoBakeInvN)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpectra_RamLak(
        512,
        /*dump_bins=*/64,
        /*bake_invN=*/false));
}

TEST(FilterTest, SpatialRampFiniteKernel)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpatialRamp(
        512,
        /*radius=*/31,
        /*dump_bins=*/16,
        /*bake_invN=*/true));
}
