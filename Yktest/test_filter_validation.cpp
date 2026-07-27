#include <cstdio>

#include "Yktest_fdkflter.hpp"
#include "global/YkLog.h"

int main_filter_spatial_ramp_validation()
{
    const bool ok = YKTest::testFilterWeightsSpatialRamp(
        512, /*radius=*/31, /*dump_bins=*/16, /*bake_invN=*/true);
    YK_LOGI("filter spatial-ramp validation: {}", ok ? "PASS" : "FAIL");
    std::printf("filter spatial-ramp validation: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
