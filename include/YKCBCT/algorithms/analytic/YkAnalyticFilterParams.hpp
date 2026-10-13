#pragma once
#include <cstdint>
namespace YK {
// Shared by FDK and WFBP. Keep the established public enum values stable.
enum class EFdkFilter : int32_t {
    None=0, RamLak=1, SheppLogan=2, Cosine=3, Hann=4, Hamming=5,
    Blackman=6, Butterworth=7, Kaiser=8, Tukey=9
};
struct SAnalyticFilterParams {
    EFdkFilter type=EFdkFilter::RamLak;
    // Cycles/sample. Nyquist (the full representable band) is 0.5.
    float cutoff=.5f, gain=1.f;
    float butterworth_order=2.f, kaiser_beta=8.6f, tukey_alpha=.5f;
};
} // namespace YK
