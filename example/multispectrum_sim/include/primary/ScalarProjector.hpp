#pragma once
#include <cuda_runtime.h>
#include <vector>
namespace yk::spectral::scalar_fp {
enum class Method { Siddon,
    Joseph };
enum class DetectorAnchor { Center,
    Pixel00 };
// All positions and pixel step vectors are millimetres, in the volume's local axes.
// detector defaults to the complete panel centre (ASTRA cone_vec convention).
// With DetectorAnchor::Pixel00 it is the centre of pixel (u=0,v=0).
struct View {
    float3 source {}, detector {}, u {}, v {};
};
struct Grid {
    int nx = 0, ny = 0, nz = 0;
    float3 spacing { 1, 1, 1 }, center {};
};
// Prepared by prepareViews: world coordinates for Siddon, centred voxel coordinates
// for Joseph. Upload unchanged; do not interchange preparations between methods/grids.
struct DeviceView {
    float3 src {}, detS {}, detU {}, detV {};
    int axis = 0;
};
std::vector<DeviceView> prepareViews(const std::vector<View>&, const Grid&, int nu, int nv, Method, DetectorAnchor = DetectorAnchor::Center);
// Caller owns all CUDA resources and stream completion. No allocation or transfer.
// Input, prepared views and output must not overlap. Geometry assumes a source
// outside the volume and a detector beyond it. Caller transforms any volume pose.
// [z][y][x] contiguous float volume; [view][v][u] contiguous output.
// Joseph texture: float32 CUDA array, unnormalised coordinates, linear filtering,
// zero-border addressing on all axes. Texture values equal the input volume.
// A subset is passed by offsetting prepared_views and output pointers and reducing views.
// Steps 1/2/4 are supported for Joseph; Siddon requires 1. No detector supersampling.
void launch(Method, const float* device_volume, cudaTextureObject_t volume_texture,
    const DeviceView* prepared_views, float* output, const Grid&, int nu, int nv, int views,
    bool accumulate = false, cudaStream_t stream = nullptr, int steps = 1);
} // namespace yk::spectral::scalar_fp
