#include "pipeline/BuiltinPluginInfo.hpp"
#pragma once
#include "recon/preprocess/PreprocessPipeline.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
namespace yk::spectral::examples {
// CPU teaching example: log projection p -> calibrated polynomial P(p).
// Coefficients are [c0,c1,...]; this stage does not fit a spectrum/water plate.
class PolynomialBhcStage final : public preprocess::IStage {
    std::vector<double> coefficients_;
    double low_, high_;

public:
    PolynomialBhcStage(std::vector<double> coefficients, double low, double high)
        : coefficients_(std::move(coefficients))
        , low_(low)
        , high_(high)
    {
        if (coefficients_.empty() || coefficients_.size() > 17 || !std::isfinite(low_) || !std::isfinite(high_) || low_ > high_)
            throw std::invalid_argument("invalid example BHC parameters");
        for (double c : coefficients_)
            if (!std::isfinite(c))
                throw std::invalid_argument("nonfinite BHC coefficient");
    }
    const pipeline::PluginInfo& info()const noexcept override{return pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::ExampleBhc);}
    pipeline::Contract<preprocess::Domain> contract() const override
    {
        return { preprocess::Domain::LogAttenuation, preprocess::Domain::LogAttenuation,
            pipeline::Location::Host, pipeline::Location::Host };
    }
    void process(preprocess::Frame& frame, const preprocess::Context&) override
    {
        if (!frame.host_data || frame.host_data->size() != frame.pixels)
            throw std::runtime_error("invalid BHC host frame");
        for (float& p : *frame.host_data) {
            if (!std::isfinite(p))
                throw std::runtime_error("nonfinite BHC input");
            double corrected = 0;
            // Double Horner: c0 + c1*p + c2*p*p + ...
            for (auto it = coefficients_.rbegin(); it != coefficients_.rend(); ++it)
                corrected = corrected * double(p) + *it;
            if (!std::isfinite(corrected))
                throw std::runtime_error("BHC polynomial overflow");
            corrected = std::clamp(corrected, low_, high_);
            if (std::abs(corrected) > std::numeric_limits<float>::max())
                throw std::runtime_error("BHC result outside float32 range");
            p = float(corrected);
        }
        // domain/location remain owned by the common executor.
        // No workspace: inherited prepare/reset are sufficient.
    }
};
} // namespace yk::spectral::examples
