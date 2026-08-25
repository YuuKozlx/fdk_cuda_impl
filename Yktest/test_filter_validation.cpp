#include <cstdio>
#include <cmath>

#include "Yktest_fdkflter.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

int main_filter_spatial_ramp_validation()
{
    const bool ok = YKTest::testFilterWeightsSpatialRamp(
        512, /*radius=*/31, /*dump_bins=*/16, /*bake_invN=*/true);
    YK_LOGI("filter spatial-ramp validation: {}", ok ? "PASS" : "FAIL");
    std::printf("filter spatial-ramp validation: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main_filter_discrete_ramlak_dc_zero()
{
    constexpr int padded_n = 1024;
    constexpr int complex_n = padded_n / 2 + 1;
    YK::Mem::MemoryController memory;
    auto d_weights = memory.allocateDevice3D<float>(complex_n, 1, 1, 0);

    YK::Filter::CreateFilterKernelFromFFT kernel;
    kernel.prepare(padded_n);

    YK::SFilterKernelDesc desc{};
    desc.kind = YK::EFilterKernel::RamLak;
    desc.source = YK::EWeightsBuildSource::DiscreteRLFFT;
    desc.cutoff = 0.5f;
    desc.gain = 1.0f;
    desc.extract_mode = YK::ERampExtractMode::Magnitude;

    bool ok = true;
    for (bool bake_inv_n : {true, false}) {
        kernel.build_weights(d_weights.data(), desc, /*du_real=*/1.0f, bake_inv_n);
        float dc = NAN;
        YK_CUDA_CHECK(cudaMemcpy(&dc, d_weights.data(), sizeof(float), cudaMemcpyDeviceToHost));
        const bool case_ok = std::isfinite(dc) && dc == 0.0f;
        YK_LOGI("finite discrete RamLak DC={:.9e}, bake_invN={}: {}",
            dc, bake_inv_n, case_ok ? "PASS" : "FAIL");
        ok = ok && case_ok;
    }
    kernel.release();
    return ok ? 0 : 1;
}
