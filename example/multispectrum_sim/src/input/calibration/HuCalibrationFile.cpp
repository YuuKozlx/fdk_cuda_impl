#include "recon/postprocess/stages/calibration/HuCalibration.hpp"
#include "recon/postprocess/stages/correction/HuTransforms.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <toml++/toml.hpp>
namespace yk::spectral::postprocess {
namespace {
    void validate(const HuCalibrationResult& r)
    {
        if (!r.valid || !std::isfinite(r.coefficients.scale) || r.coefficients.scale <= 0 || !std::isfinite(r.coefficients.offset))
            throw std::invalid_argument("Invalid HU calibration coefficients");
        for (double x : { r.measured_air_hu, r.measured_water_hu, r.target_air_hu, r.target_water_hu })
            if (!std::isfinite(x))
                throw std::invalid_argument("Non-finite HU calibration anchor");
        if (r.measured_air_hu >= r.measured_water_hu || r.target_air_hu >= r.target_water_hu)
            throw std::invalid_argument("Reversed HU calibration anchors");
        for (auto pair : { std::pair { r.measured_air_hu, r.target_air_hu }, std::pair { r.measured_water_hu, r.target_water_hu } })
            if (std::abs(pair.first * r.coefficients.scale + r.coefficients.offset - pair.second) > 1e-6 * std::max(1., std::abs(pair.second)))
                throw std::invalid_argument("HU coefficients disagree with calibration anchors");
    }
}
void HuCalibrationResult::save(const std::filesystem::path& path) const
{
    validate(*this);
    toml::table t { { "format", "air_water_hu" }, { "version", 1 }, { "input_domain", "HU" }, { "output_domain", "HU" },
        { "scale", coefficients.scale }, { "offset", coefficients.offset }, { "measured_air_hu", measured_air_hu },
        { "measured_water_hu", measured_water_hu }, { "target_air_hu", target_air_hu }, { "target_water_hu", target_water_hu },
        { "air_voxels", int64_t(air_voxels) }, { "water_voxels", int64_t(water_voxels) } };
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f << t;
    f.close();
}
HuCalibrationResult HuCalibrationResult::load(const std::filesystem::path& path)
{
    const auto t = toml::parse_file(path.string());
    if (t["format"].value_or(std::string{}) != "air_water_hu" || t["version"].value_or(0) != 1 || t["input_domain"].value_or(std::string{}) != "HU" || t["output_domain"].value_or(std::string{}) != "HU")
        throw std::runtime_error("Unsupported HU calibration format/domain");
    auto number = [&](const char* key) {auto v=t[key].value<double>();if(!v)throw std::runtime_error(std::string("Missing HU field: ")+key);return *v; };
    HuCalibrationResult r;
    r.valid = true;
    r.coefficients = { number("scale"), number("offset") };
    r.measured_air_hu = number("measured_air_hu");
    r.measured_water_hu = number("measured_water_hu");
    r.target_air_hu = number("target_air_hu");
    r.target_water_hu = number("target_water_hu");
    const auto air = t["air_voxels"].value<int64_t>(), water = t["water_voxels"].value<int64_t>();
    if (!air || !water || *air < 0 || *water < 0)
        throw std::runtime_error("Invalid HU calibration sample counts");
    r.air_voxels = size_t(*air);
    r.water_voxels = size_t(*water);
    validate(r);
    return r;
}
std::unique_ptr<IVolumePostprocessStage> makeHuCorrectionStage(const std::filesystem::path& path, Location location)
{
    return makeHuCorrectionStage(HuCalibrationResult::load(path).coefficients, location);
}
}
