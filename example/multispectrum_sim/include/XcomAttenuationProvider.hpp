#pragma once
#include "AttenuationProvider.hpp"
#include <filesystem>
namespace yk::spectral {
class XcomAttenuationProvider final : public IMassAttenuationProvider {
public:
    explicit XcomAttenuationProvider(std::filesystem::path data_directory);
    bool query(const MaterialSpec&, const std::vector<double>&,
               std::vector<double>&, std::string& error) const override;
private: std::filesystem::path data_directory_;
};
}
