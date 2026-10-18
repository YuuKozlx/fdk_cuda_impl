#pragma once
#include "input/config/SimTypes.hpp"
#include "client/visualization/FloatVolume.hpp"
namespace yk::spectral {
yk::visualization::FloatVolume loadReconstructionDisplay(const std::filesystem::path&,const GeometryConfig&);
}
