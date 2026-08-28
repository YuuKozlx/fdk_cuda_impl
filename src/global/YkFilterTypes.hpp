#pragma once

#include <utility>
#include <vector>

namespace YK {

enum class EFilterKernel {
    None,
    RamLak,
    SheppLogan,
    Cosine,
    Hann,
    Hamming,
    Blackman,
    Butterworth,
    Kaiser,
    Tukey,
    Custom
};

enum class EWeightsBuildSource {
    AnalyticFreq,
    DiscreteRLFFT,
    SpatialRampFFT
};

enum class ERampExtractMode {
    RealPart = 0,
    Magnitude = 1
};

struct SFilterKernelDesc {
    EFilterKernel kind = EFilterKernel::RamLak;
    EWeightsBuildSource source = EWeightsBuildSource::AnalyticFreq;
    ERampExtractMode extract_mode = ERampExtractMode::RealPart;
    float gain = 1.f;
    float cutoff = 0.5f;
    float order = 2.f;
    float beta = 8.6f;
    float tukey_alpha = 0.5f;
    std::vector<float> custom_weights;
    std::vector<float> spatial_ramp;

    static SFilterKernelDesc Custom(std::vector<float> weights,
        float gain = 1.f)
    {
        SFilterKernelDesc result;
        result.kind = EFilterKernel::Custom;
        result.custom_weights = std::move(weights);
        result.gain = gain;
        return result;
    }

    static SFilterKernelDesc RamLak(
        EWeightsBuildSource source = EWeightsBuildSource::DiscreteRLFFT,
        float gain = 1.f)
    {
        SFilterKernelDesc result;
        result.kind = EFilterKernel::RamLak;
        result.source = source;
        result.gain = gain;
        return result;
    }

    static SFilterKernelDesc Hamming(float cutoff = 0.5f, float gain = 1.f)
    {
        auto result = RamLak(EWeightsBuildSource::DiscreteRLFFT, gain);
        result.kind = EFilterKernel::Hamming;
        result.cutoff = cutoff;
        return result;
    }

    static SFilterKernelDesc Hann(float cutoff = 0.5f, float gain = 1.f)
    {
        auto result = RamLak(EWeightsBuildSource::DiscreteRLFFT, gain);
        result.kind = EFilterKernel::Hann;
        result.cutoff = cutoff;
        return result;
    }

    static SFilterKernelDesc SpatialRamp(std::vector<float> kernel,
        float gain = 1.f)
    {
        auto result = RamLak(EWeightsBuildSource::SpatialRampFFT, gain);
        result.spatial_ramp = std::move(kernel);
        return result;
    }

    static SFilterKernelDesc Butterworth(float cutoff = 0.5f,
        float order = 2.f, float gain = 1.f)
    {
        auto result = RamLak(EWeightsBuildSource::AnalyticFreq, gain);
        result.kind = EFilterKernel::Butterworth;
        result.cutoff = cutoff;
        result.order = order;
        return result;
    }

    static SFilterKernelDesc Kaiser(float cutoff = 0.5f, float beta = 8.6f,
        float gain = 1.f)
    {
        auto result = RamLak(EWeightsBuildSource::AnalyticFreq, gain);
        result.kind = EFilterKernel::Kaiser;
        result.cutoff = cutoff;
        result.beta = beta;
        return result;
    }

    static SFilterKernelDesc Tukey(float cutoff = 0.5f, float alpha = 0.5f,
        float gain = 1.f)
    {
        auto result = RamLak(EWeightsBuildSource::AnalyticFreq, gain);
        result.kind = EFilterKernel::Tukey;
        result.cutoff = cutoff;
        result.tukey_alpha = alpha;
        return result;
    }

    static SFilterKernelDesc Identity(float gain = 1.f)
    {
        SFilterKernelDesc result;
        result.kind = EFilterKernel::None;
        result.gain = gain;
        return result;
    }
};

} // namespace YK
