#pragma once

// Deterministic spatial sampling and photon-count noise, including scatter noise.

#include <cuda_runtime.h>

namespace yk::spectral::primary_detail {

static __device__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16); }
static __device__ float uniform(unsigned seed) { return (hash32(seed) & 0x00ffffffU) / 16777216.0f; }
static __device__ float normal(unsigned seed) {
    const float u1 = fmaxf(uniform(seed + 1), 1e-7f);
    const float u2 = uniform(seed + 2);
    return sqrtf(-2.f * logf(u1)) * cosf(6.28318530718f * u2);
}
static __device__ float poissonSample(float lambda, unsigned seed) {
    if (lambda <= 0.f) return 0.f;
    if (lambda > 32.f) return max(0.f, floorf(lambda + sqrtf(lambda) * normal(seed) + 0.5f));
    float product = 1.f;
    const float limit = expf(-lambda);
    unsigned n = 0;
    do { product *= fmaxf(uniform(seed + n * 2 + 3), 1e-7f); ++n; } while (product > limit && n < 256);
    return static_cast<float>(n - 1);
}

static __global__ void applyQuantumNoiseAfterScatter(float* energy, int pixels,
    float mean_energy_keV, unsigned seed, int view) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels || !(energy[pixel] > 0.f) || !(mean_energy_keV > 0.f)) return;
    const float expected_photons = energy[pixel] / mean_energy_keV;
    const unsigned noise_seed = seed ^ static_cast<unsigned>(view * pixels + pixel) * 0x9e3779b9U;
    energy[pixel] = poissonSample(expected_photons, noise_seed) * mean_energy_keV;
}

} // namespace yk::spectral::primary_detail
