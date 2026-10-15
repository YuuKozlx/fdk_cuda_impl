#pragma once
#include "recon/preprocess/PreprocessPipeline.hpp"
namespace yk::spectral::preprocess {
// Providers may reproject a mask, read cached paths, or borrow GPU paths.
// writeDevice supplies exactly width*height bone thicknesses in mm on default stream.
class IBoneThicknessProvider {
public:
    virtual ~IBoneThicknessProvider() = default;
    virtual void prepare(const Context&) { }
    virtual void writeDevice(int view, int width, int height, float* destination) = 0;
    virtual void reset() noexcept { }
};
}
