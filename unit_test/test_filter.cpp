#include <gtest/gtest.h>
#include <global/YkMacro.hpp>
#include "Yktest/Yktest_fdkflter.hpp"

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

TEST(FilterTest, RamLak_ForceDcZero)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpectra_RamLak(
        512,
        /*dump_bins=*/64,
        /*force_dc_zero=*/true));
}

TEST(FilterTest, RamLak_NoBakeInvN)
{
    EXPECT_TRUE(YKTest::testFilterWeightsSpectra_RamLak(
        512,
        /*dump_bins=*/64,
        /*force_dc_zero=*/false,
        /*bake_invN=*/false));
}
