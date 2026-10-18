#pragma once
#include <toml++/toml.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
namespace yk::spectral::config_detail {
inline constexpr double degrees_per_radian=57.2957795130823208768;
inline std::string degreeKey(std::string rad_key){return rad_key.substr(0,rad_key.size()-4)+"_deg";}
// Explicit suffixes preserve old files without changing their physical meaning.
inline double readAngle(const toml::table& table,const char* rad_key,double fallback){
    const auto deg_key=degreeKey(rad_key);
    const auto rad=table[rad_key],deg=table[deg_key];
    if(rad && deg)throw std::runtime_error("Specify only one angle unit: "+deg_key+" / "+rad_key);
    const auto value=deg?deg:rad;
    if(!value)return fallback;
    auto number=value.value<double>();
    if(!number || !std::isfinite(*number))throw std::runtime_error("Invalid angle: "+(deg?deg_key:std::string(rad_key)));
    return deg?*number/degrees_per_radian:*number;
}
}
