#pragma once
#include "config/SimTypes.hpp"
#include "../../../visualization/FloatVolume.hpp"
namespace yk::spectral {
yk::visualization::FloatVolume loadReconstructionDisplay(const std::filesystem::path&,const GeometryConfig&);
}
