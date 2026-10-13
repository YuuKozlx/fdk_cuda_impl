#pragma once
#include "ImageKernels.hpp"
#include "image/ImageProcessing.hpp"
namespace yk::spectral::image::detail {
void cudaAxis(float*, size_t, AxisFilter);
void cudaResize(float*, size_t, ResizeAxis);
void cudaBilateral(float*, size_t, Bilateral);
void cudaPointwise(float*, size_t, Pointwise);
} // namespace
