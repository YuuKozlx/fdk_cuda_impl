#pragma once

// Material path integration and spectral transmission. Included only by the
// projector CUDA translation unit, which owns these kernels and constant tables.

#include "ProjectionGeometry.cuh"
#include "QuantumSampling.cuh"

namespace yk::spectral::primary_detail {

__constant__ float c_mu[15000];
__constant__ float c_spectrum[256];
__constant__ float c_energies[256];

__device__ float readMu(const DeviceGeometry& g, const float* mu, int index) {
    return g.use_constant_tables ? c_mu[index] : __ldg(mu + index);
}
__device__ float readSpectrum(const DeviceGeometry& g, const float* spectrum, int index) {
    return g.use_constant_tables ? c_spectrum[index] : __ldg(spectrum + index);
}
__device__ float readEnergy(const DeviceGeometry& g, const float* energies, int index) {
    return g.use_constant_tables ? c_energies[index] : __ldg(energies + index);
}

__device__ bool boxHit(float3 s, float3 d, float3 lo, float3 hi, float& a, float& b) {
    a = 0.f; b = 1e30f;
    for (int k = 0; k < 3; ++k) {
        float o = k == 0 ? s.x : (k == 1 ? s.y : s.z);
        float q = k == 0 ? d.x : (k == 1 ? d.y : d.z);
        float l = k == 0 ? lo.x : (k == 1 ? lo.y : lo.z);
        float h = k == 0 ? hi.x : (k == 1 ? hi.y : hi.z);
        if (fabsf(q) < 1e-12f) { if (o < l || o > h) return false; continue; }
        float x = (l-o)/q, y = (h-o)/q; if (x > y) { float t=x; x=y; y=t; }
        a = fmaxf(a, x); b = fminf(b, y); if (a >= b) return false;
    }
    return b > 0.f;
}

// Traverse from the box entry in local coordinates. Keeping the traversal
// parameter relative to the entry avoids subtracting sub-millimeter voxel
// boundaries from the ~500 mm source coordinates.
template <int MaterialCount>
__device__ bool accumulatePaths(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, float3 source, float3 d, float3 lo, float3 hi,
    float* paths) {
    float entry, exit;
    if (!boxHit(source, d, lo, hi, entry, exit)) return false;
    const float3 entry_point = make_float3(source.x + d.x * entry,
        source.y + d.y * entry, source.z + d.z * entry);
    const float3 q = make_float3(entry_point.x - lo.x,
        entry_point.y - lo.y, entry_point.z - lo.z);
    int ix = max(0, min(g.nx - 1, static_cast<int>(floorf(q.x / g.vx))));
    int iy = max(0, min(g.ny - 1, static_cast<int>(floorf(q.y / g.vy))));
    int iz = max(0, min(g.nz - 1, static_cast<int>(floorf(q.z / g.vz))));
    const int sx = d.x >= 0 ? 1 : -1, sy = d.y >= 0 ? 1 : -1, sz = d.z >= 0 ? 1 : -1;
    const float inf = 1e30f;
    float tx = d.x != 0 ? ((d.x > 0 ? (ix + 1) * g.vx : ix * g.vx) - q.x) / d.x : inf;
    float ty = d.y != 0 ? ((d.y > 0 ? (iy + 1) * g.vy : iy * g.vy) - q.y) / d.y : inf;
    float tz = d.z != 0 ? ((d.z > 0 ? (iz + 1) * g.vz : iz * g.vz) - q.z) / d.z : inf;
    const float dx = d.x != 0 ? g.vx / fabsf(d.x) : inf;
    const float dy = d.y != 0 ? g.vy / fabsf(d.y) : inf;
    const float dz = d.z != 0 ? g.vz / fabsf(d.z) : inf;
    float s = 0.f;
    bool hit_material = false;
    const float length = exit - entry;
    while (s < length && ix >= 0 && ix < g.nx && iy >= 0 && iy < g.ny && iz >= 0 && iz < g.nz) {
        const float next = fminf(tx, fminf(ty, tz));
        const float segment = fminf(next, length) - s;
        const int label = g.use_label_texture
            ? static_cast<int>(tex3D<unsigned char>(g.label_texture,
                ix + 0.5f, iy + 0.5f, iz + 0.5f))
            : static_cast<int>(__ldg(labels + (iz * g.ny + iy) * g.nx + ix));
        const int m = __ldg(label_to_material + label);
        if (m >= 0 && m < MaterialCount && segment > 0.f) {
            paths[m] += segment * 0.1f;
            hit_material = true;
        }
        s = next;
        if (tx <= next) { tx += dx; ix += sx; }
        if (ty <= next) { ty += dy; iy += sy; }
        if (tz <= next) { tz += dz; iz += sz; }
    }
    return hit_material;
}

template <int MaterialCount>
__global__ void primaryKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, float incident, int quantum_noise,
    int view, unsigned seed, unsigned quantum_seed,
    float* expected_energy_out, float* sampled_energy_out, float* air_out,
    unsigned char* object_mask_out) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = g.nu * g.nv; if (pixel >= pixels) return;
    const int u0 = pixel % g.nu, v0 = pixel / g.nu;
    if (g.ne > 256) return;
    float expected_counts[256] = {};
    float expected_air_energy = 0.f;
    bool object_hit = false;
    float3 lo = make_float3(-0.5f*g.nx*g.vx, -0.5f*g.ny*g.vy, -0.5f*g.nz*g.vz);
    float3 hi = make_float3(-lo.x, -lo.y, -lo.z);
    // Integrate each component separately so a small defocus dose cannot be
    // lost through random component selection with few spatial samples.
    for (int component=0; component<2; ++component) {
        const bool defocused = component == 1;
        const float exposure_fraction = defocused ? g.focal_epsilon : 1.f-g.focal_epsilon;
        if (exposure_fraction <= 0.f) continue;
        for (int sample=0; sample<g.samples; ++sample) {
        unsigned base = seed ^ (unsigned)(view*pixels + pixel)*0x9e3779b9U ^ (unsigned)sample*0x85ebca6bU;
        const float fu = defocused ? g.defocus_fu : g.normal_fu;
        const float fv = defocused ? g.defocus_fv : g.normal_fv;
        float fsu = (uniform(base+1)-.5f)*fu;
        float fsv = (uniform(base+2)-.5f)*fv;
        const float jitter_u = uniform(base+3)-.5f;
        const float jitter_v = uniform(base+4)-.5f;
        const auto ray = scannerRay(g, view, static_cast<float>(u0),
            static_cast<float>(v0), fsu, fsv, jitter_u, jitter_v);
        const float3 source = ray.source, det = ray.detector;
        float3 d = make_float3(det.x-source.x, det.y-source.y, det.z-source.z);
        float length = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z); d.x/=length; d.y/=length; d.z/=length;
        float paths[MaterialCount] = {};
        object_hit = accumulatePaths<MaterialCount>(g, labels, label_to_material,
            phantomInverse(g, source, false), phantomInverse(g, d, true),
            lo, hi, paths) || object_hit;
        float r2=length*length; float flux=g.geometry_flux ? 1000000.f/fmaxf(r2,1e-6f) : 1.f;
        // Geometry samples share one pixel exposure. They are not separate
        // exposures; each sample carries 1/g.samples of the pixel photons.
        const float photons_per_sample = exposure_fraction / static_cast<float>(g.samples);
        for(int e=0;e<g.ne;++e){float exponent=0.f;
            #pragma unroll
            for(int m=0;m<MaterialCount;++m) exponent+=paths[m]*readMu(g, mu, m*g.ne+e);
            const float transmission=expf(-exponent);
            const float response=g.detector_response ? __ldg(g.detector_efficiency + (v0*g.nu+u0)*g.ne+e) : 1.f;
            const float weight=photons_per_sample*readSpectrum(g, spectrum, e)*flux*response;
            const float lambda = weight * transmission;
            expected_counts[e] += lambda;
            expected_air_energy += weight * energies[e];
        }
        }
    }
    float expected_energy = 0.f;
    float sampled_energy = 0.f;
    for (int e=0; e<g.ne; ++e) {
        const unsigned noise_seed = quantum_seed
            ^ (unsigned)(view * pixels + pixel) * 0x9e3779b9U
            ^ (unsigned)e * 0x27d4eb2dU;
        const float sampled_count = quantum_noise ? poissonSample(expected_counts[e], noise_seed)
                                                  : expected_counts[e];
        expected_energy += expected_counts[e] * readEnergy(g, energies, e);
        sampled_energy += sampled_count * readEnergy(g, energies, e);
    }
    const float sum_air_energy = expected_air_energy;
    // Actual transmitted counts fluctuate; normalize only by the configured
    // incident expectation so the Poisson count fluctuation remains visible.
    // Return energy per incident photon. The configured exposure is applied
    // only inside the sample weights above, so increasing g.samples does not
    // increase dose.
    const float avg_air_energy = sum_air_energy;
    expected_energy_out[pixel]=expected_energy;
    sampled_energy_out[pixel]=sampled_energy;
    air_out[pixel]=avg_air_energy;
    object_mask_out[pixel] = object_hit ? 1 : 0;
}

template <int MaterialCount>
void launchPrimaryKernel(const DeviceGeometry& geometry,
    const unsigned char* labels, const int* label_to_material,
    const float* mu, const float* spectrum, const float* energies,
    float incident, int quantum_noise, int view, unsigned seed,
    unsigned quantum_seed, float* expected_energy_out,
    float* sampled_energy_out, float* air_out, unsigned char* object_mask_out) {
    const int pixels = geometry.nu * geometry.nv;
    primaryKernel<MaterialCount><<< (pixels + 255) / 256, 256 >>>(
        geometry, labels, label_to_material, mu, spectrum, energies,
        incident, quantum_noise, view, seed, quantum_seed,
        expected_energy_out, sampled_energy_out, air_out, object_mask_out);
}


} // namespace yk::spectral::primary_detail
