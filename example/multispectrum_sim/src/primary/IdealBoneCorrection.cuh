#pragma once
#include "MaterialPaths.cuh"

namespace yk::spectral::primary_detail {
// One central ray and one Siddon material traversal per detector pixel.
// Both spectral and reference-energy signals reuse precisely the same paths.
// No focal sampling, detector blur/lag, scatter, or noise is applied here.
__global__ void idealBoneCorrectionKernel(DeviceGeometry g,
    const unsigned char* labels, const int* map, const double* mu,
    const double* weights, const double* mono_mu, const double* coefficients,
    int coefficient_count, double low, double high, int first_view,
    const float* measured, float* corrected, float* residual, float* model_poly = nullptr)
{
    const int pixel = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int pixels = g.nu * g.nv;
    if (pixel >= pixels)
        return;
    const int view = first_view + int(blockIdx.y);
    const auto ray = scannerRay(g, view, float(pixel % g.nu), float(pixel / g.nu));
    float3 direction = make_float3(ray.detector.x - ray.source.x,
        ray.detector.y - ray.source.y, ray.detector.z - ray.source.z);
    const float length = sqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    direction = scale3(direction, 1.f / length);
    const float3 lo = make_float3(-.5f * g.nx * g.vx, -.5f * g.ny * g.vy, -.5f * g.nz * g.vz);
    const float3 hi = make_float3(-lo.x, -lo.y, -lo.z);
    float paths[3] = {};
    accumulatePaths<3>(g, labels, map, phantomInverse(g, ray.source, false),
        phantomInverse(g, direction, true), lo, hi, paths);
    double mono = 0, transmission = 0;
    for (int m = 0; m < 3; ++m)
        mono += double(paths[m]) * mono_mu[m];
    for (int e = 0; e < g.ne; ++e) {
        double exponent = 0;
        for (int m = 0; m < 3; ++m)
            exponent += double(paths[m]) * mu[m * g.ne + e];
        transmission += weights[e] * exp(-exponent);
    }
    const double poly = -log(fmax(transmission, 1e-30));
    double water = coefficient_count == 0 ? poly : 0;
    for (int j = coefficient_count - 1; j >= 0; --j)
        water = __dadd_rn(__dmul_rn(water, poly), coefficients[j]);
    water = fmin(fmax(water, low), high);
    const size_t index = size_t(blockIdx.y) * pixels + pixel;
    const float delta = float(mono - water);
    residual[index] = delta;
    corrected[index] = measured[index] + delta;
    if (model_poly)
        model_poly[index] = float(poly);
}
} // namespace yk::spectral::primary_detail
