#include "recon/postprocess/stages/segmentation/HuThresholds.hpp"
#include "../IntensityOps.hpp"
#include "recon/postprocess/stages/IntensityPluginInfo.hpp"
#include <cmath>
#include <map>
#include <stdexcept>
namespace yk::spectral::postprocess {
namespace {
    class ThresholdStage final : public IStage {
        HuThresholdSegmentation options_;
        Location location_;
        std::shared_ptr<const LabelMapping> mapping_;

    public:
        ThresholdStage(HuThresholdSegmentation s, Location l)
            : options_(std::move(s))
            , location_(l)
        {
            intensity_detail::validateLocation(l);
            if (options_.thresholds_hu.size() >= 16777216u)
                throw std::invalid_argument("too many HU thresholds");
            for (size_t i = 0; i < options_.thresholds_hu.size(); ++i)
                if (!std::isfinite(options_.thresholds_hu[i]) || (i && options_.thresholds_hu[i] <= options_.thresholds_hu[i - 1]))
                    throw std::invalid_argument("HU thresholds must be finite and strictly increasing");
            if (options_.classes.empty())
                for (size_t i = 0; i <= options_.thresholds_hu.size(); ++i)
                    options_.classes.push_back({ int(i), {} });
            if (options_.classes.size() != options_.thresholds_hu.size() + 1)
                throw std::invalid_argument("N thresholds require N+1 label classes");
            std::map<int, std::string> names;
            for (const auto& c : options_.classes) {
                if (c.value < 0 || c.value > 16777216)
                    throw std::invalid_argument("label ID is not an exact nonnegative float32 integer");
                auto found = names.emplace(c.value, c.name);
                if (!found.second && found.first->second != c.name)
                    throw std::invalid_argument("same label ID has conflicting names");
            }
            mapping_ = std::make_shared<LabelMapping>(LabelMapping { options_.thresholds_hu, options_.classes });
        }
        const pipeline::PluginInfo& info() const noexcept override { return intensityPluginDescriptions()[(location_ == Location::Device ? 5 : 0) + 4]; }
        pipeline::Contract<Domain> contract() const override { return { Domain::HounsfieldUnits, Domain::Labels, location_, location_ }; }
        void validateContext(const Context& c) const override
        {
            if (location_ == Location::Device && c.stream)
                throw std::invalid_argument("segmentation requires default CUDA stream");
        }
        void process(Volume& v, const Context&) override
        {
            if (v.domain != Domain::HounsfieldUnits)
                throw std::invalid_argument("threshold segmentation requires HU values");
            intensity_detail::classify(v, location_, options_);
            v.label_mapping = mapping_;
        }
    };
}
std::unique_ptr<IStage> makeHuThresholdStage(HuThresholdSegmentation s, Location l) { return std::make_unique<ThresholdStage>(std::move(s), l); }
}
