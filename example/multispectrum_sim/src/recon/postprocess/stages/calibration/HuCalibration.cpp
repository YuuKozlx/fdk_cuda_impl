#include "recon/postprocess/stages/calibration/HuCalibration.hpp"
#include "../IntensityOps.hpp"
#include "recon/postprocess/stages/IntensityPluginInfo.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace yk::spectral::postprocess {
HuCalibrationResult fitAirWaterHu(double air, double water, AirWaterTargets targets)
{
    intensity_detail::validateTargets(targets);
    const double gap = water - air;
    if (!std::isfinite(air) || !std::isfinite(water) || !std::isfinite(gap) || gap < targets.minimum_measured_gap_hu)
        throw std::invalid_argument("air/water HU measurements are reversed or too close");
    HuCalibrationResult result;
    result.coefficients.scale = (targets.water_hu - targets.air_hu) / gap;
    result.coefficients.offset = targets.air_hu - result.coefficients.scale * air;
    intensity_detail::validateCoefficients(result.coefficients);
    result.measured_air_hu = air;
    result.measured_water_hu = water;
    result.target_air_hu = targets.air_hu;
    result.target_water_hu = targets.water_hu;
    result.valid = true;
    return result;
}
namespace {
    class CalibrationStage final : public IStage {
        CalibrationRoi air_, water_;
        std::shared_ptr<HuCalibrationResult> result_;
        AirWaterTargets targets_;
        Location location_;
        void validateBoxes(const Grid& g) const
        {
            intensity_detail::validateRoi(air_, g);
            intensity_detail::validateRoi(water_, g);
            bool overlap = true;
            for (int a = 0; a < 3; ++a)
                overlap = overlap && std::max(air_.begin[a], water_.begin[a]) < std::min(air_.end[a], water_.end[a]);
            if (overlap)
                throw std::invalid_argument("air and water calibration ROIs overlap");
        }

    public:
        CalibrationStage(CalibrationRoi air, CalibrationRoi water, std::shared_ptr<HuCalibrationResult> result, AirWaterTargets targets, Location l)
            : air_(air)
            , water_(water)
            , result_(std::move(result))
            , targets_(targets)
            , location_(l)
        {
            if (!result_)
                throw std::invalid_argument("missing HU calibration output");
            intensity_detail::validateTargets(targets);
            intensity_detail::validateLocation(l);
        }
        const pipeline::PluginInfo& info() const noexcept override { return intensityPluginDescriptions()[(location_ == Location::Device ? 5 : 0) + 2]; }
        pipeline::Contract<Domain> contract() const override { return { Domain::HounsfieldUnits, Domain::HounsfieldUnits, location_, location_ }; }
        void prepare(const Context& c) override
        {
            *result_ = {};
            if (location_ == Location::Device && c.stream)
                throw std::invalid_argument("calibration requires default CUDA stream");
            validateBoxes(c.grid);
        }
        void process(Volume& v, const Context&) override
        {
            *result_ = {};
            if (v.domain != Domain::HounsfieldUnits)
                throw std::invalid_argument("calibration requires HU values");
            validateBoxes(v.grid);
            const double air = intensity_detail::meanRoi(v, location_, air_), water = intensity_detail::meanRoi(v, location_, water_);
            auto result = fitAirWaterHu(air, water, targets_);
            result.air_voxels = intensity_detail::roiCount(air_);
            result.water_voxels = intensity_detail::roiCount(water_);
            *result_ = result;
        }
    };
}
std::unique_ptr<IStage> makeAirWaterHuCalibrationStage(CalibrationRoi air, CalibrationRoi water, std::shared_ptr<HuCalibrationResult> out, AirWaterTargets targets, Location l) { return std::make_unique<CalibrationStage>(air, water, std::move(out), targets, l); }
}
