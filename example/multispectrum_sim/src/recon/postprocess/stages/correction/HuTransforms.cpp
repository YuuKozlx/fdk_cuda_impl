#include "recon/postprocess/stages/correction/HuTransforms.hpp"
#include "../IntensityOps.hpp"
#include "pipeline/BuiltinPluginInfo.hpp"
#include <cmath>
#include <stdexcept>
namespace yk::spectral::postprocess {
namespace {
    HuLinearCoefficients conversion(MuHuReference r, bool inverse)
    {
        if (!std::isfinite(r.water_mu_mm_inv) || !std::isfinite(r.air_mu_mm_inv) || r.air_mu_mm_inv < 0 || r.water_mu_mm_inv <= r.air_mu_mm_inv || !std::isfinite(r.water_hu) || !std::isfinite(r.air_hu) || r.water_hu <= r.air_hu)
            throw std::invalid_argument("invalid mu/HU air-water reference anchors");
        const double mu_span = r.water_mu_mm_inv - r.air_mu_mm_inv, hu_span = r.water_hu - r.air_hu;
        HuLinearCoefficients c;
        c.scale = inverse ? mu_span / hu_span : hu_span / mu_span;
        c.offset = inverse ? r.air_mu_mm_inv - c.scale * r.air_hu : r.air_hu - c.scale * r.air_mu_mm_inv;
        intensity_detail::validateCoefficients(c);
        return c;
    }
    class LinearStage final : public IStage {
        Domain input_, output_;
        Location location_;
        HuLinearCoefficients coefficients_;
        std::shared_ptr<const HuCalibrationResult> calibration_;
        int metadata_index_;
        bool basic_;

    public:
        LinearStage(Domain in, Domain out, Location l, HuLinearCoefficients c, int op, bool basic = false, std::shared_ptr<const HuCalibrationResult> calibration = {})
            : input_(in)
            , output_(out)
            , location_(l)
            , coefficients_(c)
            , calibration_(std::move(calibration))
            , metadata_index_((l == Location::Device ? 5 : 0) + op)
            , basic_(basic)
        {
            intensity_detail::validateLocation(l);
            intensity_detail::validateCoefficients(c);
        }
        const pipeline::PluginInfo& info() const noexcept override { return basic_ ? pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::HuConversion) : intensityPluginDescriptions()[metadata_index_]; }
        pipeline::Contract<Domain> contract() const override { return { input_, output_, location_, location_ }; }
        void validateContext(const Context& c) const override
        {
            if (location_ == Location::Device && c.stream)
                throw std::invalid_argument("HU stages require the pool's default stream");
        }
        void process(Volume& v, const Context&) override
        {
            if (v.domain != input_)
                throw std::invalid_argument("HU stage input domain mismatch");
            auto c = coefficients_;
            if (calibration_) {
                if (!calibration_->valid)
                    throw std::runtime_error("HU calibration is not available or failed");
                c = calibration_->coefficients;
            }
            intensity_detail::applyLinear(v, location_, c);
        }
    };
}
std::unique_ptr<IStage> makeMuToHuStage(MuHuReference r, Location l) { return std::make_unique<LinearStage>(Domain::AttenuationMmInv, Domain::HounsfieldUnits, l, conversion(r, false), 0); }
std::unique_ptr<IStage> makeHuToMuStage(MuHuReference r, Location l) { return std::make_unique<LinearStage>(Domain::HounsfieldUnits, Domain::AttenuationMmInv, l, conversion(r, true), 1); }
std::unique_ptr<IStage> makeHuCorrectionStage(HuLinearCoefficients c, Location l) { return std::make_unique<LinearStage>(Domain::HounsfieldUnits, Domain::HounsfieldUnits, l, c, 3); }
std::unique_ptr<IStage> makeHuCorrectionStage(std::shared_ptr<const HuCalibrationResult> c, Location l)
{
    if (!c)
        throw std::invalid_argument("missing HU calibration result");
    return std::make_unique<LinearStage>(Domain::HounsfieldUnits, Domain::HounsfieldUnits, l, HuLinearCoefficients {}, 3, false, std::move(c));
}
// Existing water-only CPU convenience factory now shares the same numerical engine.
std::unique_ptr<IStage> makeHuConversionStage(double water) { return std::make_unique<LinearStage>(Domain::AttenuationMmInv, Domain::HounsfieldUnits, Location::Host, conversion({ water }, false), 0, true); }
}
