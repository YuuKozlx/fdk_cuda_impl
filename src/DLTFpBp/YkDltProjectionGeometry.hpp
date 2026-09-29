#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK::DltFpBp {

// Row-major 3x4 projective camera matrix.  World/object coordinates must use
// the same millimetre coordinate system as the reconstruction volume.
struct SDltProjectionMatrix {
    std::array<double, 12> value{};

    double operator()(int row, int column) const
    { return value[static_cast<size_t>(row) * 4 + column]; }

    double& operator()(int row, int column)
    { return value[static_cast<size_t>(row) * 4 + column]; }
};

// A ray camera, not a physical detector plane.  For detector sample (u,v):
//   ray(u,v) = ray00 + u*rayU + v*rayV
//   X(t)     = source + t*ray(u,v), t >= 0.
// Multiplying all three ray vectors by one non-zero scalar changes neither
// the line nor the Siddon path lengths.
struct SDltRayGeometry {
    float4 source{};
    float4 ray00{};
    float4 rayU{};
    float4 rayV{};
};

static_assert(sizeof(SDltRayGeometry) == 4 * sizeof(float4),
    "SDltRayGeometry must remain a packed four-float4 type");

// Host-prepared geometry for voxel-driven DLT backprojection. For a voxel
// point X, the inverse basis maps d = X - source to
// [a,b,c]^T = inverse_basis * d, with detector coordinates u=b/a and v=c/a.
// The inverse is prepared once on the host instead of recomputed for every
// voxel/view pair in the CUDA kernel.
struct SDltVoxelBackGeometry {
    float4 source{};
    float4 ray00{};
    float4 rayU{};
    float4 rayV{};
    float4 inverseRow0{};
    float4 inverseRow1{};
    float4 inverseRow2{};
};

inline SDltProjectionMatrix flipImageV(const SDltProjectionMatrix& input,
    int detector_rows)
{
    SDltProjectionMatrix output = input;
    if (detector_rows <= 0) return output;
    // [u, v_down, 1]^T = H [u, v_up, 1]^T,
    // H row 1 = (detector_rows-1)*P.row(2) - P.row(1).
    for (int column = 0; column < 4; ++column)
        output(1, column) = (detector_rows - 1.0) * input(2, column)
            - input(1, column);
    return output;
}

namespace detail {

inline bool invert3x3(const double m[9], double inverse[9],
    double epsilon = 1e-14)
{
    const double c00 = m[4] * m[8] - m[5] * m[7];
    const double c01 = m[2] * m[7] - m[1] * m[8];
    const double c02 = m[1] * m[5] - m[2] * m[4];
    const double c10 = m[5] * m[6] - m[3] * m[8];
    const double c11 = m[0] * m[8] - m[2] * m[6];
    const double c12 = m[2] * m[3] - m[0] * m[5];
    const double c20 = m[3] * m[7] - m[4] * m[6];
    const double c21 = m[1] * m[6] - m[0] * m[7];
    const double c22 = m[0] * m[4] - m[1] * m[3];
    const double determinant = m[0] * c00 + m[1] * c10 + m[2] * c20;
    double scale = 0.0;
    for (double item : std::array<double, 9>{m[0], m[1], m[2], m[3], m[4],
             m[5], m[6], m[7], m[8]})
        scale = std::max(scale, std::fabs(item));
    if (!std::isfinite(determinant) || std::fabs(determinant) <=
        epsilon * std::max(1.0, scale * scale * scale)) return false;
    const double reciprocal = 1.0 / determinant;
    const double adjugate[9] = {
        c00, c01, c02, c10, c11, c12, c20, c21, c22
    };
    for (int i = 0; i < 9; ++i) inverse[i] = adjugate[i] * reciprocal;
    return true;
}

inline std::array<double, 3> multiply(const double matrix[9],
    const std::array<double, 3>& vector)
{
    return {
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]
    };
}

inline bool finite3(const std::array<double, 3>& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

inline float4 toFloat4(const std::array<double, 3>& value)
{
    return make_float4(static_cast<float>(value[0]),
        static_cast<float>(value[1]), static_cast<float>(value[2]), 0.f);
}

} // namespace detail

inline bool buildRayGeometry(const SDltProjectionMatrix& projection,
    int detector_channels, int detector_rows, const float3& volume_center_mm,
    SDltRayGeometry& output)
{
    if (detector_channels <= 0 || detector_rows <= 0) return false;
    double block[9]{};
    double translation[3]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            block[row * 3 + column] = projection(row, column);
        translation[row] = projection(row, 3);
    }
    double inverse[9]{};
    if (!detail::invert3x3(block, inverse)) return false;
    auto source = detail::multiply(inverse,
        {-translation[0], -translation[1], -translation[2]});
    auto ray_u = detail::multiply(inverse, {1.0, 0.0, 0.0});
    auto ray_v = detail::multiply(inverse, {0.0, 1.0, 0.0});
    auto ray_00 = detail::multiply(inverse, {0.0, 0.0, 1.0});
    const double center_u = 0.5 * (detector_channels - 1.0);
    const double center_v = 0.5 * (detector_rows - 1.0);
    std::array<double, 3> center_ray{};
    for (int axis = 0; axis < 3; ++axis)
        center_ray[axis] = ray_00[axis] + center_u * ray_u[axis]
            + center_v * ray_v[axis];
    const double length = std::sqrt(center_ray[0] * center_ray[0] +
        center_ray[1] * center_ray[1] + center_ray[2] * center_ray[2]);
    if (!detail::finite3(source) || !detail::finite3(center_ray) ||
        !std::isfinite(length) || length <= 1e-14) return false;
    const std::array<double, 3> to_volume = {
        static_cast<double>(volume_center_mm.x) - source[0],
        static_cast<double>(volume_center_mm.y) - source[1],
        static_cast<double>(volume_center_mm.z) - source[2]
    };
    const double orientation = center_ray[0] * to_volume[0] +
        center_ray[1] * to_volume[1] + center_ray[2] * to_volume[2];
    const double common_scale = (orientation < 0.0 ? -1.0 : 1.0) / length;
    for (int axis = 0; axis < 3; ++axis) {
        ray_00[axis] *= common_scale;
        ray_u[axis] *= common_scale;
        ray_v[axis] *= common_scale;
    }
    if (!detail::finite3(ray_00) || !detail::finite3(ray_u) ||
        !detail::finite3(ray_v)) return false;
    output.source = detail::toFloat4(source);
    output.ray00 = detail::toFloat4(ray_00);
    output.rayU = detail::toFloat4(ray_u);
    output.rayV = detail::toFloat4(ray_v);
    return true;
}

inline bool buildRayGeometry(const std::vector<SDltProjectionMatrix>& projections,
    int detector_channels, int detector_rows, const float3& volume_center_mm,
    std::vector<SDltRayGeometry>& output)
{
    if (projections.empty()) return false;
    output.resize(projections.size());
    for (size_t view = 0; view < projections.size(); ++view) {
        if (!buildRayGeometry(projections[view], detector_channels, detector_rows,
                volume_center_mm, output[view])) {
            output.clear();
            return false;
        }
    }
    return true;
}

inline bool buildVoxelBackGeometry(
    const std::vector<SDltRayGeometry>& geometry,
    std::vector<SDltVoxelBackGeometry>& output)
{
    if (geometry.empty()) return false;
    output.resize(geometry.size());
    for (size_t i = 0; i < geometry.size(); ++i) {
        const auto& input = geometry[i];
        const double basis[9] = {
            input.ray00.x, input.rayU.x, input.rayV.x,
            input.ray00.y, input.rayU.y, input.rayV.y,
            input.ray00.z, input.rayU.z, input.rayV.z
        };
        double inverse[9]{};
        if (!detail::invert3x3(basis, inverse)) {
            output.clear();
            return false;
        }
        auto& item = output[i];
        item.source = input.source;
        item.ray00 = input.ray00;
        item.rayU = input.rayU;
        item.rayV = input.rayV;
        item.inverseRow0 = make_float4(
            static_cast<float>(inverse[0]), static_cast<float>(inverse[1]),
            static_cast<float>(inverse[2]), 0.f);
        item.inverseRow1 = make_float4(
            static_cast<float>(inverse[3]), static_cast<float>(inverse[4]),
            static_cast<float>(inverse[5]), 0.f);
        item.inverseRow2 = make_float4(
            static_cast<float>(inverse[6]), static_cast<float>(inverse[7]),
            static_cast<float>(inverse[8]), 0.f);
    }
    return true;
}

// Convert the library's explicit flat-detector representation to the same
// canonical ray camera used by the DLT kernels.  detS is the centre of pixel
// (0,0), so no half-pixel correction is needed.  A common per-view scale and
// sign are removed because they do not change any source-to-pixel ray.
inline bool buildRayGeometry(const std::vector<SConeProjGeomVec>& geometry,
    int detector_channels, int detector_rows, const float3& volume_center_mm,
    std::vector<SDltRayGeometry>& output)
{
    if (geometry.empty() || detector_channels <= 0 || detector_rows <= 0)
        return false;
    const double center_u = 0.5 * (detector_channels - 1.0);
    const double center_v = 0.5 * (detector_rows - 1.0);
    output.resize(geometry.size());
    for (size_t view_index = 0; view_index < geometry.size(); ++view_index) {
        const auto& view = geometry[view_index];
        std::array<double, 3> source = {
            view.src.x, view.src.y, view.src.z
        };
        std::array<double, 3> ray_00 = {
            static_cast<double>(view.detS.x) - view.src.x,
            static_cast<double>(view.detS.y) - view.src.y,
            static_cast<double>(view.detS.z) - view.src.z
        };
        std::array<double, 3> ray_u = {
            view.detU.x, view.detU.y, view.detU.z
        };
        std::array<double, 3> ray_v = {
            view.detV.x, view.detV.y, view.detV.z
        };
        std::array<double, 3> center_ray{};
        for (int axis = 0; axis < 3; ++axis)
            center_ray[axis] = ray_00[axis] + center_u * ray_u[axis]
                + center_v * ray_v[axis];
        const double length = std::sqrt(center_ray[0] * center_ray[0] +
            center_ray[1] * center_ray[1] + center_ray[2] * center_ray[2]);
        if (!detail::finite3(source) || !detail::finite3(center_ray) ||
            !detail::finite3(ray_u) || !detail::finite3(ray_v) ||
            !std::isfinite(length) || length <= 1e-14) {
            output.clear();
            return false;
        }
        const std::array<double, 3> to_volume = {
            static_cast<double>(volume_center_mm.x) - source[0],
            static_cast<double>(volume_center_mm.y) - source[1],
            static_cast<double>(volume_center_mm.z) - source[2]
        };
        const double orientation = center_ray[0] * to_volume[0] +
            center_ray[1] * to_volume[1] + center_ray[2] * to_volume[2];
        const double common_scale = (orientation < 0.0 ? -1.0 : 1.0) / length;
        for (int axis = 0; axis < 3; ++axis) {
            ray_00[axis] *= common_scale;
            ray_u[axis] *= common_scale;
            ray_v[axis] *= common_scale;
        }
        auto& ray = output[view_index];
        ray.source = detail::toFloat4(source);
        ray.ray00 = detail::toFloat4(ray_00);
        ray.rayU = detail::toFloat4(ray_u);
        ray.rayV = detail::toFloat4(ray_v);
    }
    return true;
}

// A projective camera is also an equivalent flat detector: multiplying every
// ray direction by one common positive distance changes detector-plane scale,
// but not a single source-to-pixel ray.
inline bool buildConeProjectionGeometry(
    const std::vector<SDltRayGeometry>& rays,
    std::vector<SConeProjGeomVec>& output,
    float detector_plane_scale_mm = 1000.f)
{
    if (rays.empty() || !std::isfinite(detector_plane_scale_mm) ||
        detector_plane_scale_mm <= 0.f) return false;
    constexpr float two_pi = 6.2831853071795864769f;
    output.resize(rays.size());
    for (size_t i = 0; i < rays.size(); ++i) {
        const auto& ray = rays[i];
        auto& view = output[i];
        view.src = ray.source;
        view.detS = make_float4(
            ray.source.x + detector_plane_scale_mm * ray.ray00.x,
            ray.source.y + detector_plane_scale_mm * ray.ray00.y,
            ray.source.z + detector_plane_scale_mm * ray.ray00.z, 0.f);
        view.detU = make_float4(detector_plane_scale_mm * ray.rayU.x,
            detector_plane_scale_mm * ray.rayU.y,
            detector_plane_scale_mm * ray.rayU.z, 0.f);
        view.detV = make_float4(detector_plane_scale_mm * ray.rayV.x,
            detector_plane_scale_mm * ray.rayV.y,
            detector_plane_scale_mm * ray.rayV.z, 0.f);
        view.angle = make_float4(two_pi * static_cast<float>(i) /
            static_cast<float>(rays.size()), 0.f, 0.f, 0.f);
    }
    return true;
}

inline bool buildConeProjectionGeometry(
    const std::vector<SDltProjectionMatrix>& projections,
    int detector_channels, int detector_rows,
    const float3& volume_center_mm,
    std::vector<SConeProjGeomVec>& output,
    float detector_plane_scale_mm = 1000.f)
{
    std::vector<SDltRayGeometry> rays;
    return buildRayGeometry(projections, detector_channels, detector_rows,
            volume_center_mm, rays) &&
        buildConeProjectionGeometry(rays, output, detector_plane_scale_mm);
}

} // namespace YK::DltFpBp
