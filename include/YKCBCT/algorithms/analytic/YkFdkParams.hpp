#pragma once
#include <cstdint>

namespace YK {

enum class EParkerMode : int32_t { Auto, Disabled, Enabled };

struct SParkerScanSpec {
    EParkerMode mode = EParkerMode::Auto;
};

enum class EFdkFilter : int32_t {
    None = 0,
    RamLak = 1,
    SheppLogan = 2,
    Cosine = 3,
    Hann = 4,
    Hamming = 5,
    Blackman = 6,
    // Parameterized windows.  Keep existing numeric values stable because
    // this enum is part of the public task interface.
    Butterworth = 7,
    Kaiser = 8,
    Tukey = 9,
};

struct SFdkAlgoParams {
    SParkerScanSpec parker{};
    EFdkFilter filter = EFdkFilter::RamLak;

    // Shared frequency-domain window parameters.  cutoff is normalized
    // to Nyquist: 0.5 means the complete representable band.
    float cutoff = 0.5f;
    float gain = 1.f;
    // Finite discrete Ram-Lak always suppresses DC; this is fixed
    // algorithm behavior and is not exposed as a runtime option.

    // Filter-specific parameters.  Unused fields are ignored.
    float butterworth_order = 2.f;
    float kaiser_beta = 8.6f;
    float tukey_alpha = 0.5f;
};

} // namespace YK
