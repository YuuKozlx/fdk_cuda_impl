#pragma once
#include "SimTypes.hpp"
#include <vector>
#include <string>
namespace yk::spectral {
class IMassAttenuationProvider {
public:
    virtual ~IMassAttenuationProvider() = default;
    // 返回 cm^2/g，顺序与输入能量一致。
    virtual bool query(const MaterialSpec&, const std::vector<double>&,
                       std::vector<double>&, std::string& diagnostic) const = 0;
};
}
