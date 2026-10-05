#include "CudaPrimaryProjector.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace yk::spectral {
namespace {
void writeMetadata(const std::filesystem::path& file, int u, int v, int views, const char* signal) {
    std::ofstream f(file.string()+".json");
    if (!f) throw std::runtime_error("cannot create projection metadata");
    f << "{\n  \"columns\": " << u << ",\n  \"rows\": " << v
      << ",\n  \"frames\": " << views
      << ",\n  \"data_type\": \"float32\",\n  \"byte_order\": \"little_endian\",\n"
      << "  \"layout\": \"frame_row_column\",\n  \"signal\": \"" << signal << "\"\n}\n";
}
struct DeviceGeometry { int nu, nv, views, views_per_turn, rotation_direction, samples, nx, ny, nz, nm, ne; float pu, pv, sid, sdd, vx, vy, vz, fu, fv; float offset_u, offset_n, offset_v, source_x, source_y, source_z; float tilt_u, tilt_v, tilt_n; float start, pitch, start_z; float phantom_x, phantom_y, phantom_z, phantom_rx, phantom_ry, phantom_rz; int cylindrical, apply_flux, detector_response, use_constant_tables, use_label_texture; const float* detector_efficiency; cudaTextureObject_t label_texture; };
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

__device__ float3 add3(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ float3 scale3(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ float3 cross3(float3 a, float3 b) { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ float3 rotateAxis(float3 value, float3 axis, float angle) {
    const float c=cosf(angle), s=sinf(angle);
    return add3(add3(scale3(value,c),scale3(cross3(axis,value),s)),scale3(axis,(axis.x*value.x+axis.y*value.y+axis.z*value.z)*(1.f-c)));
}
__device__ void detectorFrame(DeviceGeometry g, float angle, float3& u, float3& v, float3& n) {
    const float ca=cosf(angle), sa=sinf(angle);
    u=make_float3(-sa,ca,0.f); v=make_float3(0.f,0.f,1.f); n=make_float3(ca,sa,0.f);
    v=rotateAxis(v,u,g.tilt_u); n=rotateAxis(n,u,g.tilt_u);
    u=rotateAxis(u,v,g.tilt_v); n=rotateAxis(n,v,g.tilt_v);
    u=rotateAxis(u,n,g.tilt_n); v=rotateAxis(v,n,g.tilt_n);
}
std::vector<float> loadLayerThickness(const DetectorLayerConfig& layer, int pixels) {
    std::vector<float> thickness(static_cast<size_t>(pixels), static_cast<float>(layer.thickness_mm));
    if (layer.thickness_map_file.empty()) return thickness;
    std::ifstream input(layer.thickness_map_file, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open detector thickness map: " + layer.thickness_map_file.string());
    input.read(reinterpret_cast<char*>(thickness.data()),
        static_cast<std::streamsize>(thickness.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(thickness.size() * sizeof(float)) ||
        input.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("detector thickness map must contain exactly detector_u * detector_v float32 values");
    for (float value : thickness)
        if (!std::isfinite(value) || value < 0.f)
            throw std::runtime_error("detector thickness map contains a negative or non-finite value");
    return thickness;
}

__device__ float3 phantomInverse(DeviceGeometry g, float3 value, bool direction) {
    const float3 x = make_float3(1.f, 0.f, 0.f);
    const float3 y = make_float3(0.f, 1.f, 0.f);
    const float3 z = make_float3(0.f, 0.f, 1.f);
    value = rotateAxis(value, z, -g.phantom_rz);
    value = rotateAxis(value, y, -g.phantom_ry);
    value = rotateAxis(value, x, -g.phantom_rx);
    // Forward pose order is offset first, then XYZ rotation. The inverse ray
    // transform therefore applies inverse rotation before removing the offset.
    if (!direction) value = make_float3(value.x-g.phantom_x, value.y-g.phantom_y, value.z-g.phantom_z);
    return value;
}

__device__ unsigned hash32(unsigned x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16); }
__device__ float uniform(unsigned seed) { return (hash32(seed) & 0x00ffffffU) / 16777216.0f; }
__device__ float normal(unsigned seed) {
    const float u1 = fmaxf(uniform(seed + 1), 1e-7f);
    const float u2 = uniform(seed + 2);
    return sqrtf(-2.f * logf(u1)) * cosf(6.28318530718f * u2);
}
__device__ float poissonSample(float lambda, unsigned seed) {
    if (lambda <= 0.f) return 0.f;
    if (lambda > 32.f) return max(0.f, floorf(lambda + sqrtf(lambda) * normal(seed) + 0.5f));
    float product = 1.f;
    const float limit = expf(-lambda);
    unsigned n = 0;
    do { product *= fmaxf(uniform(seed + n * 2 + 3), 1e-7f); ++n; } while (product > limit && n < 256);
    return static_cast<float>(n - 1);
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
__device__ void accumulatePaths(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, float3 source, float3 d, float3 lo, float3 hi,
    float* paths) {
    float entry, exit;
    if (!boxHit(source, d, lo, hi, entry, exit)) return;
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
    const float length = exit - entry;
    while (s < length && ix >= 0 && ix < g.nx && iy >= 0 && iy < g.ny && iz >= 0 && iz < g.nz) {
        const float next = fminf(tx, fminf(ty, tz));
        const float segment = fminf(next, length) - s;
        const int label = g.use_label_texture
            ? static_cast<int>(tex3D<unsigned char>(g.label_texture,
                ix + 0.5f, iy + 0.5f, iz + 0.5f))
            : static_cast<int>(__ldg(labels + (iz * g.ny + iy) * g.nx + ix));
        const int m = __ldg(label_to_material + label);
        if (m >= 0 && m < MaterialCount && segment > 0.f)
            paths[m] += segment * 0.1f;
        s = next;
        if (tx <= next) { tx += dx; ix += sx; }
        if (ty <= next) { ty += dy; iy += sy; }
        if (tz <= next) { tz += dz; iz += sz; }
    }
}

template <int MaterialCount>
__global__ void primaryKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, float incident, int quantum_noise,
    int view, unsigned seed, unsigned quantum_seed,
    float* energy_out, float* air_out) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = g.nu * g.nv; if (pixel >= pixels) return;
    const int u0 = pixel % g.nu, v0 = pixel / g.nu;
    if (g.ne > 256) return;
    float expected_counts[256] = {};
    float expected_air_energy = 0.f;
    const float pi = 3.14159265358979323846f;
    const float angle = g.start + static_cast<float>(g.rotation_direction) * 2.f*pi*view / g.views_per_turn;
    // The library frame at angle zero has source at -Y, U at +X, and N at
    // +Y. Apply the same phase to both flat and cylindrical detectors.
    const float frame_angle = angle - 0.5f*pi;
    const float ca = cosf(frame_angle), sa = sinf(frame_angle);
    const float z = g.start_z + g.pitch * (angle - g.start) / (2.f*pi);
    float3 lo = make_float3(-0.5f*g.nx*g.vx, -0.5f*g.ny*g.vy, -0.5f*g.nz*g.vz);
    float3 hi = make_float3(-lo.x, -lo.y, -lo.z);
    for (int sample=0; sample<g.samples; ++sample) {
        unsigned base = seed ^ (unsigned)(view*pixels + pixel)*0x9e3779b9U ^ (unsigned)sample*0x85ebca6bU;
        float fsu = (uniform(base+1)-.5f)*g.fu;
        float fsv = (uniform(base+2)-.5f)*g.fv;
        float du = ((float)u0 - .5f*(g.nu-1) + uniform(base+3)-.5f)*g.pu;
        float dv = ((float)v0 - .5f*(g.nv-1) + uniform(base+4)-.5f)*g.pv;
        float3 uaxis, vaxis, naxis; detectorFrame(g, frame_angle, uaxis, vaxis, naxis);
        const float3 nominal_naxis = make_float3(ca, sa, 0.f);
        float3 source = make_float3((g.sid+fsu)*ca + g.source_x*ca - g.source_y*sa,
            (g.sid+fsu)*sa + g.source_x*sa + g.source_y*ca, z+fsv+g.source_z);
        const float center_u = du + g.offset_u;
        const float center_v = dv + g.offset_v;
        // Keep the detector fixed in the scanner frame. Source offsets and
        // focal-spot samples affect only the source; the panel starts at the
        // isocenter and is SDD-SID away from it.
        const float detector_distance = g.sdd - g.sid;
        const float3 isocenter = make_float3(0.f, 0.f, z);
        const float3 detector_principal = add3(isocenter,
            scale3(nominal_naxis, -detector_distance));
        float3 detector_center = add3(add3(add3(detector_principal,
            scale3(naxis, g.offset_n)),
            scale3(uaxis, center_u)), scale3(vaxis, center_v));
        float3 det;
        if (g.cylindrical) {
            const float gamma = center_u / g.sdd;
            // Match the DLL cylindrical builder: detector_principal is
            // SDD-SID from isocenter, offset_n moves the cylindrical surface
            // along its normal, offset_u selects an arc position, and
            // offset_v moves along the detector axis.
            const float3 cylinder_center = add3(detector_principal,
                scale3(naxis, g.offset_n + g.sdd));
            const float3 radial = add3(scale3(naxis, -cosf(gamma)),
                scale3(uaxis, sinf(gamma)));
            det = add3(add3(cylinder_center, scale3(radial, g.sdd)),
                scale3(vaxis, center_v));
        } else {
            det = detector_center;
        }
        float3 d = make_float3(det.x-source.x, det.y-source.y, det.z-source.z);
        float length = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z); d.x/=length; d.y/=length; d.z/=length;
        float paths[MaterialCount] = {};
        accumulatePaths<MaterialCount>(g, labels, label_to_material,
            phantomInverse(g, source, false), phantomInverse(g, d, true),
            lo, hi, paths);
        float r2=length*length; float flux=g.apply_flux ? 1000000.f/fmaxf(r2,1e-6f) : 1.f;
        // Geometry samples share one pixel exposure. They are not separate
        // exposures; each sample carries 1/g.samples of the pixel photons.
        const float photons_per_sample = 1.f / static_cast<float>(g.samples);
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
    float sum_energy = 0.f;
    for (int e=0; e<g.ne; ++e) {
        const unsigned noise_seed = quantum_seed
            ^ (unsigned)(view * pixels + pixel) * 0x9e3779b9U
            ^ (unsigned)e * 0x27d4eb2dU;
        const float count = quantum_noise ? poissonSample(expected_counts[e], noise_seed)
                                          : expected_counts[e];
        sum_energy += count * readEnergy(g, energies, e);
    }
    const float sum_air_energy = expected_air_energy;
    // Actual transmitted counts fluctuate; normalize only by the configured
    // incident expectation so the Poisson count fluctuation remains visible.
    // Return energy per incident photon. The configured exposure is applied
    // only inside the sample weights above, so increasing g.samples does not
    // increase dose.
    const float avg_energy = sum_energy;
    const float avg_air_energy = sum_air_energy;
energy_out[pixel]=avg_energy;
    air_out[pixel]=avg_air_energy;
}

template <int MaterialCount>
void launchPrimaryKernel(const DeviceGeometry& geometry,
    const unsigned char* labels, const int* label_to_material,
    const float* mu, const float* spectrum, const float* energies,
    float incident, int quantum_noise, int view, unsigned seed,
    unsigned quantum_seed, float* energy_out, float* air_out) {
    const int pixels = geometry.nu * geometry.nv;
    primaryKernel<MaterialCount><<< (pixels + 255) / 256, 256 >>>(
        geometry, labels, label_to_material, mu, spectrum, energies,
        incident, quantum_noise, view, seed, quantum_seed,
        energy_out, air_out);
}

__global__ void detectorEfficiencyKernel(DeviceGeometry g,
    const float* protective_thickness, const float* impurity_thickness,
    const float* scintillator_thickness, const float* protective_mu,
    const float* impurity_mu, const float* scintillator_mu,
    float protective_density, float impurity_density,
    float scintillator_density, int impurity_enabled,
    int oblique_path_correction, float* efficiency) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = g.nu * g.nv * g.ne;
    if (index >= total) return;
    const int e = index % g.ne;
    const int pixel = index / g.ne;
    const int u0 = pixel % g.nu;
    const int v0 = pixel / g.nu;
    const float pi = 3.14159265358979323846f;
    const float frame_angle = g.start - 0.5f * pi;
    const float ca = cosf(frame_angle), sa = sinf(frame_angle);
    float3 uaxis, vaxis, naxis;
    detectorFrame(g, frame_angle, uaxis, vaxis, naxis);
    const float3 source = make_float3(
        g.sid * ca + g.source_x * ca - g.source_y * sa,
        g.sid * sa + g.source_x * sa + g.source_y * ca,
        g.start_z + g.source_z);
    const float detector_distance = g.sdd - g.sid;
    const float3 detector_principal = scale3(make_float3(ca, sa, 0.f),
        -detector_distance);
    const float du = (u0 - 0.5f * (g.nu - 1)) * g.pu;
    const float dv = (v0 - 0.5f * (g.nv - 1)) * g.pv;
    const float3 center = add3(add3(add3(detector_principal,
        scale3(naxis, g.offset_n)), scale3(uaxis, du + g.offset_u)),
        scale3(vaxis, dv + g.offset_v));
    const float3 ray = make_float3(center.x - source.x, center.y - source.y,
        center.z - source.z);
    const float ray_len = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    const float cosine = fabsf(ray.x * naxis.x + ray.y * naxis.y + ray.z * naxis.z)
        / fmaxf(ray_len, 1e-6f);
    const float path_factor = oblique_path_correction
        ? 1.f / fmaxf(cosine, 1e-4f) : 1.f;
    const float mm_to_cm = path_factor * 0.1f;
    float entrance_exponent = mm_to_cm * __ldg(protective_thickness + pixel)
        * protective_density * __ldg(protective_mu + e);
    if (impurity_enabled)
        entrance_exponent += mm_to_cm * __ldg(impurity_thickness + pixel)
            * impurity_density * __ldg(impurity_mu + e);
    const float sensor_exponent = mm_to_cm * __ldg(scintillator_thickness + pixel)
        * scintillator_density * __ldg(scintillator_mu + e);
    efficiency[index] = expf(-entrance_exponent) * (-expm1f(-sensor_exponent));
}

bool ok(cudaError_t e, const char* what, std::string& err) { if(e==cudaSuccess)return true; err=std::string(what)+": "+cudaGetErrorString(e); return false; }

void convolveSeparable(std::vector<float>& image, int width, int height,
                       const std::vector<double>& kernel_u,
                       const std::vector<double>& kernel_v) {
    std::vector<float> temporary(image.size());
    const int radius_u = static_cast<int>(kernel_u.size() / 2);
    for (int v=0; v<height; ++v) for (int u=0; u<width; ++u) {
        double value = 0.0, weight = 0.0;
        for (int k=-radius_u; k<=radius_u; ++k) {
            const int source_u = u + k;
            if (source_u < 0 || source_u >= width) continue;
            const double w = kernel_u[static_cast<size_t>(k + radius_u)];
            value += w * image[static_cast<size_t>(v*width + source_u)];
            weight += w;
        }
        temporary[static_cast<size_t>(v*width + u)] = static_cast<float>(value / std::max(weight, 1e-30));
    }
    const int radius_v = static_cast<int>(kernel_v.size() / 2);
    for (int v=0; v<height; ++v) for (int u=0; u<width; ++u) {
        double value = 0.0, weight = 0.0;
        for (int k=-radius_v; k<=radius_v; ++k) {
            const int source_v = v + k;
            if (source_v < 0 || source_v >= height) continue;
            const double w = kernel_v[static_cast<size_t>(k + radius_v)];
            value += w * temporary[static_cast<size_t>(source_v*width + u)];
            weight += w;
        }
        image[static_cast<size_t>(v*width + u)] = static_cast<float>(value / std::max(weight, 1e-30));
}
}
void applyCrosstalkSignal(std::vector<float>& signal, int width, int height,
                          const CrosstalkConfig& config) {
    if (!config.enabled) return;
    convolveSeparable(signal, width, height, config.kernel_u, config.kernel_v);
}

void applyAfterglowSignal(std::vector<float>& signal,
                          const DetectorPostprocessConfig::AfterglowConfig& config,
                          std::vector<std::vector<float>>& state) {
    if (!config.enabled) return;
    const bool initialize_steady_state = state.empty() &&
        config.initialization == "steady_state";
    if (state.empty())
        state.assign(config.weights.size(), std::vector<float>(signal.size(), 0.f));
    if (initialize_steady_state) {
        const float exposure = static_cast<float>(config.exposure_time_ms);
        const float readout = static_cast<float>(config.frame_time_ms - config.exposure_time_ms);
        for (size_t component=0; component<config.weights.size(); ++component) {
            const float tau = static_cast<float>(config.time_constants_ms[component]);
            const float de = std::exp(-exposure / tau);
            const float dr = std::exp(-readout / tau);
            const float cycle_decay = de * dr;
            const float input_scale = static_cast<float>(config.p) *
                static_cast<float>(config.weights[component]) * tau * (1.f - de) / exposure;
            const float steady_scale = dr * input_scale / std::max(1.f - cycle_decay, 1e-6f);
            for (size_t p=0; p<signal.size(); ++p)
                state[component][p] = steady_scale * signal[p];
        }
    }
    std::vector<float> output(signal.size(), static_cast<float>(0.0));
    const float prompt = static_cast<float>(1.0 - config.p);
    for (size_t p=0; p<signal.size(); ++p) output[p] = prompt * signal[p];
    for (size_t component=0; component<config.weights.size(); ++component) {
        const float exposure = static_cast<float>(config.exposure_time_ms);
        const float readout = static_cast<float>(
            config.frame_time_ms - config.exposure_time_ms);
        const float weight = static_cast<float>(config.weights[component]);
        const float tau = static_cast<float>(config.time_constants_ms[component]);
        const float exposure_decay = std::exp(-exposure / tau);
        const float readout_decay = std::exp(-readout / tau);
        const float released_fraction = 1.f - exposure_decay;
        for (size_t p=0; p<signal.size(); ++p) {
            const float input_trapped = static_cast<float>(config.p) * weight * signal[p];
            output[p] += state[component][p] * released_fraction
                + input_trapped * (1.f - tau * released_fraction / exposure);
            const float state_after_exposure = state[component][p] * exposure_decay
                + input_trapped * tau * released_fraction / exposure;
            state[component][p] = state_after_exposure * readout_decay;
        }
    }
    signal.swap(output);
}

void applyDasSignal(std::vector<float>& signal,
                    const DetectorPostprocessConfig::DasConfig& config,
                    std::mt19937& random) {
    if (!config.enabled) return;
    std::normal_distribution<double> noise(0.0, config.electronic_noise_std_electrons);
    const double gain = config.gain_electrons_per_keV;
    auto digitize = [&](double electrons) {
        if (config.saturation_electrons > 0)
            electrons = std::min(electrons, config.saturation_electrons);
        if (config.adc_lsb_electrons > 0)
            electrons = std::round(electrons / config.adc_lsb_electrons) * config.adc_lsb_electrons;
        return electrons;
    };
    for (size_t p=0; p<signal.size(); ++p) {
        const double electrons = digitize(
            signal[p] * gain + config.offset_electrons + noise(random));
        signal[p] = static_cast<float>(electrons / gain);
    }
}
}

bool CudaPrimaryProjector::run(const SimulationConfig& c, const std::vector<std::uint8_t>& labels,
    const SpectralTransmissionModel& model, const std::filesystem::path& out_file, std::string& err) const {
    try {
        const auto& g=c.geometry.parameters; const int pixels=g.detector_u*g.detector_v; const int nm=(int)c.projection.materials.size(), ne=(int)model.spectrum().size();
        if(g.volume_x<=0||g.volume_y<=0||g.volume_z<=0||labels.size()!=(size_t)g.volume_x*g.volume_y*g.volume_z) throw std::runtime_error("CUDA前投标签体尺寸不匹配");
        if(nm > 32) throw std::runtime_error("CUDA primary projector supports at most 32 materials");
        const int cylindrical = (c.geometry.kind == GeometryKind::CylCbct || c.geometry.kind == GeometryKind::CylHelical) ? 1 : 0;
        DeviceGeometry dg{g.detector_u,g.detector_v,g.views,g.views_per_turn,g.rotation_direction,c.projection.sampling.samples_per_pixel,g.volume_x,g.volume_y,g.volume_z,nm,ne,(float)g.pixel_u_mm,(float)g.pixel_v_mm,(float)g.sid_mm,(float)g.sdd_mm,(float)g.voxel_x_mm,(float)g.voxel_y_mm,(float)g.voxel_z_mm,(float)(c.projection.focal_spot.enabled?c.projection.focal_spot.size_u_mm:0),(float)(c.projection.focal_spot.enabled?c.projection.focal_spot.size_v_mm:0),(float)g.offset_u_mm,(float)g.offset_n_mm,(float)g.offset_v_mm,(float)g.source_offset_x_mm,(float)g.source_offset_y_mm,(float)g.source_offset_z_mm,(float)g.tilt_u_rad,(float)g.tilt_v_rad,(float)g.tilt_n_rad,(float)g.start_angle_rad,(float)g.pitch_mm_per_turn,(float)g.start_z_mm,(float)g.phantom_offset_x_mm,(float)g.phantom_offset_y_mm,(float)g.phantom_offset_z_mm,(float)g.phantom_rotation_x_rad,(float)g.phantom_rotation_y_rad,(float)g.phantom_rotation_z_rad,cylindrical,c.projection.apply_geometry_flux ? 1 : 0,0,0,0,nullptr,0};
        std::vector<int> map(256,-1); for(int m=0;m<nm;++m)map[c.projection.materials[m].label]=m;
        std::vector<float> mu(nm*ne), sw(ne), en(ne); float incident=0;
        const double pixel_area_cm2 = g.pixel_u_mm * g.pixel_v_mm / 100.0;
        for(int e=0;e<ne;++e){const auto& point=model.spectrum()[e];sw[e]=(float)(point.fluence_per_keV_cm2_mAs_at_1m*point.bin_width_keV*c.projection.mAs_per_view*pixel_area_cm2);en[e]=(float)point.energy_keV;incident+=sw[e]*en[e];for(int m=0;m<nm;++m)mu[m*ne+e]=(float)(c.projection.materials[m].density_g_cm3*model.massAttenuation(c.projection.materials[m].label)[e]);}
        const bool detector_response = c.projection.detector_response.enabled;
        std::vector<float> protective_mu, impurity_mu, scintillator_mu;
        std::vector<float> protective_thickness, impurity_thickness, scintillator_thickness;
        if (detector_response) {
            const auto& protective = c.projection.detector_response.protective_layer;
            const auto& impurity = c.projection.detector_response.impurity_layer;
            const auto& scintillator = c.projection.detector_response.scintillator_layer;
            const auto protective_mu_double = model.detectorMassAttenuation(protective);
            const auto scintillator_mu_double = model.detectorMassAttenuation(scintillator);
            const auto impurity_mu_double = impurity.enabled
                ? model.detectorMassAttenuation(impurity) : std::vector<double>(static_cast<size_t>(ne), 0.0);
            protective_mu.assign(protective_mu_double.begin(), protective_mu_double.end());
            impurity_mu.assign(impurity_mu_double.begin(), impurity_mu_double.end());
            scintillator_mu.assign(scintillator_mu_double.begin(), scintillator_mu_double.end());
            protective_thickness = loadLayerThickness(protective, pixels);
            scintillator_thickness = loadLayerThickness(scintillator, pixels);
            impurity_thickness = impurity.enabled
                ? loadLayerThickness(impurity, pixels) : std::vector<float>(static_cast<size_t>(pixels), 0.f);
        }
        unsigned char* dl=nullptr; int* dm=nullptr; float *dmu=nullptr,*ds=nullptr,*de=nullptr,*dresponse=nullptr,*do1=nullptr,*do2=nullptr;
        cudaArray_t label_array=nullptr; cudaTextureObject_t label_texture=0;
        float *dprotective_thickness=nullptr,*dimpurity_thickness=nullptr,*dscintillator_thickness=nullptr;
        float *dprotective_mu=nullptr,*dimpurity_mu=nullptr,*dscintillator_mu=nullptr;
        auto ck=[&](cudaError_t x,const char* w){return ok(x,w,err);};
        if(!ck(cudaMalloc(&dl,labels.size()),"cudaMalloc labels")||!ck(cudaMalloc(&dm,256*sizeof(int)),"cudaMalloc map")||!ck(cudaMalloc(&dmu,mu.size()*sizeof(float)),"cudaMalloc mu")||!ck(cudaMalloc(&ds,sw.size()*sizeof(float)),"cudaMalloc spectrum")||!ck(cudaMalloc(&de,en.size()*sizeof(float)),"cudaMalloc energies")||!ck(cudaMalloc(&do1,pixels*sizeof(float)),"cudaMalloc projection")||!ck(cudaMalloc(&do2,pixels*sizeof(float)),"cudaMalloc energy")) throw std::runtime_error(err);
        cudaMemcpy(dl,labels.data(),labels.size(),cudaMemcpyHostToDevice);cudaMemcpy(dm,map.data(),256*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dmu,mu.data(),mu.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(ds,sw.data(),sw.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(de,en.data(),en.size()*sizeof(float),cudaMemcpyHostToDevice);
        const bool use_constant_tables = ne <= 256 && nm * ne <= 15000;
        if (use_constant_tables) {
            cudaMemcpyToSymbol(c_mu,mu.data(),mu.size()*sizeof(float));
            cudaMemcpyToSymbol(c_spectrum,sw.data(),sw.size()*sizeof(float));
            cudaMemcpyToSymbol(c_energies,en.data(),en.size()*sizeof(float));
            dg.use_constant_tables=1;
        }
        {
            const cudaChannelFormatDesc desc=cudaCreateChannelDesc<unsigned char>();
            const cudaExtent extent=make_cudaExtent(g.volume_x,g.volume_y,g.volume_z);
            cudaMalloc3DArray(&label_array,&desc,extent);
            cudaMemcpy3DParms copy={}; copy.srcPtr=make_cudaPitchedPtr(const_cast<unsigned char*>(labels.data()),g.volume_x,g.volume_x,g.volume_y); copy.dstArray=label_array; copy.extent=extent; copy.kind=cudaMemcpyHostToDevice; cudaMemcpy3D(&copy);
            cudaResourceDesc resource={}; resource.resType=cudaResourceTypeArray; resource.res.array.array=label_array;
            cudaTextureDesc texture={}; texture.readMode=cudaReadModeElementType; texture.filterMode=cudaFilterModePoint; texture.addressMode[0]=cudaAddressModeClamp; texture.addressMode[1]=cudaAddressModeClamp; texture.addressMode[2]=cudaAddressModeClamp; texture.normalizedCoords=0;
            cudaCreateTextureObject(&label_texture,&resource,&texture,nullptr);
            dg.label_texture=label_texture; dg.use_label_texture=1;
        }
        if (detector_response) {
            const size_t pixel_bytes = static_cast<size_t>(pixels) * sizeof(float);
            const size_t energy_bytes = static_cast<size_t>(ne) * sizeof(float);
            const size_t response_bytes = static_cast<size_t>(pixels) * ne * sizeof(float);
            if(!ck(cudaMalloc(&dresponse,response_bytes),"cudaMalloc detector response")||
               !ck(cudaMalloc(&dprotective_thickness,pixel_bytes),"cudaMalloc protective thickness")||
               !ck(cudaMalloc(&dimpurity_thickness,pixel_bytes),"cudaMalloc impurity thickness")||
               !ck(cudaMalloc(&dscintillator_thickness,pixel_bytes),"cudaMalloc scintillator thickness")||
               !ck(cudaMalloc(&dprotective_mu,energy_bytes),"cudaMalloc protective mu")||
               !ck(cudaMalloc(&dimpurity_mu,energy_bytes),"cudaMalloc impurity mu")||
               !ck(cudaMalloc(&dscintillator_mu,energy_bytes),"cudaMalloc scintillator mu")) throw std::runtime_error(err);
            cudaMemcpy(dprotective_thickness,protective_thickness.data(),pixel_bytes,cudaMemcpyHostToDevice);
            cudaMemcpy(dimpurity_thickness,impurity_thickness.data(),pixel_bytes,cudaMemcpyHostToDevice);
            cudaMemcpy(dscintillator_thickness,scintillator_thickness.data(),pixel_bytes,cudaMemcpyHostToDevice);
            cudaMemcpy(dprotective_mu,protective_mu.data(),energy_bytes,cudaMemcpyHostToDevice);
            cudaMemcpy(dimpurity_mu,impurity_mu.data(),energy_bytes,cudaMemcpyHostToDevice);
            cudaMemcpy(dscintillator_mu,scintillator_mu.data(),energy_bytes,cudaMemcpyHostToDevice);
            const auto& response=c.projection.detector_response;
            const int total=pixels*ne;
            detectorEfficiencyKernel<<<(total+255)/256,256>>>(dg,
                dprotective_thickness,dimpurity_thickness,dscintillator_thickness,
                dprotective_mu,dimpurity_mu,dscintillator_mu,
                static_cast<float>(response.protective_layer.density_g_cm3),
                static_cast<float>(response.impurity_layer.density_g_cm3),
                static_cast<float>(response.scintillator_layer.density_g_cm3),
                response.impurity_layer.enabled?1:0,
                response.oblique_path_correction?1:0,dresponse);
            if(!ck(cudaGetLastError(),"detector efficiency kernel")||
               !ck(cudaDeviceSynchronize(),"detector efficiency synchronize")) throw std::runtime_error(err);
            dg.detector_response=1; dg.detector_efficiency=dresponse;
        }
        const bool write_energy = !c.projection.energy_output_file.empty();
        if (const auto parent = out_file.parent_path(); !parent.empty())
            std::filesystem::create_directories(parent);
        if (write_energy) {
            const auto parent = c.projection.energy_output_file.parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
        }
        const bool write_air = !c.projection.air_output_file.empty();
        if (write_air) {
            const auto parent = c.projection.air_output_file.parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
        }
        std::vector<float> ho(pixels),he(pixels),air(pixels); std::ofstream fo(out_file,std::ios::binary);
        std::vector<std::vector<float>> object_afterglow_state;
        std::vector<std::vector<float>> air_afterglow_state;
        std::vector<float> processed_air(pixels);
        bool have_air = false;
        std::mt19937 object_das_random(c.projection.detector_postprocess.das.noise_seed);
        std::mt19937 air_das_random(c.projection.detector_postprocess.das.noise_seed + 1u);
        std::ofstream fe;
        if (write_energy) fe.open(c.projection.energy_output_file, std::ios::binary);
        std::ofstream fa;
        if (write_air) fa.open(c.projection.air_output_file, std::ios::binary);
        if (write_air && !fa) throw std::runtime_error("cannot open air output");
        if(!fo || (write_energy && !fe)) throw std::runtime_error("无法创建CUDA前投输出");
        for(int v=0;v<g.views;++v){
            const int quantum_noise = c.projection.quantum_noise_enabled ? 1 : 0;
            switch (dg.nm) {
            case 1: launchPrimaryKernel<1>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 2: launchPrimaryKernel<2>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 3: launchPrimaryKernel<3>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 4: launchPrimaryKernel<4>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 5: launchPrimaryKernel<5>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 6: launchPrimaryKernel<6>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 7: launchPrimaryKernel<7>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 8: launchPrimaryKernel<8>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 9: launchPrimaryKernel<9>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            case 10: launchPrimaryKernel<10>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            default: launchPrimaryKernel<32>(dg,dl,dm,dmu,ds,de,incident,quantum_noise,v,c.projection.sampling.seed,c.projection.quantum_noise_seed,do1,do2); break;
            }
            if(!ck(cudaGetLastError(),"primary kernel")||!ck(cudaDeviceSynchronize(),"primary synchronize"))throw std::runtime_error(err);cudaMemcpy(he.data(),do1,pixels*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(air.data(),do2,pixels*sizeof(float),cudaMemcpyDeviceToHost);
            applyCrosstalkSignal(he, g.detector_u, g.detector_v,
                c.projection.detector_postprocess.optical_crosstalk);
            applyAfterglowSignal(he, c.projection.detector_postprocess.afterglow,
                object_afterglow_state);
            applyCrosstalkSignal(he, g.detector_u, g.detector_v,
                c.projection.detector_postprocess.electronic_crosstalk);
            applyDasSignal(he, c.projection.detector_postprocess.das,
                object_das_random);
            if (!have_air) {
                processed_air = air;
                applyCrosstalkSignal(processed_air, g.detector_u, g.detector_v,
                    c.projection.detector_postprocess.optical_crosstalk);
                applyAfterglowSignal(processed_air,
                    c.projection.detector_postprocess.afterglow,
                    air_afterglow_state);
                applyCrosstalkSignal(processed_air, g.detector_u, g.detector_v,
                    c.projection.detector_postprocess.electronic_crosstalk);
                applyDasSignal(processed_air,
                    c.projection.detector_postprocess.das, air_das_random);
                have_air = true;
                if (write_air)
                    fa.write(reinterpret_cast<const char*>(processed_air.data()), pixels*sizeof(float));
            }
            for (int p=0; p<pixels; ++p)
                ho[p] = -std::log(std::max(he[p] / std::max(processed_air[p], 1e-30f), 1e-30f));
            fo.write(reinterpret_cast<const char*>(ho.data()), pixels*sizeof(float));
            if (write_energy)
                fe.write(reinterpret_cast<const char*>(he.data()), pixels*sizeof(float));
            if ((v + 1) == 1 || (v + 1) == g.views || (v + 1) % 10 == 0)
                std::cerr << "projection frame " << (v + 1) << "/" << g.views << std::endl;
        }
        cudaFree(dl);cudaFree(dm);cudaFree(dmu);cudaFree(ds);cudaFree(de);cudaFree(dresponse);
        cudaFree(dprotective_thickness);cudaFree(dimpurity_thickness);cudaFree(dscintillator_thickness);
        cudaFree(dprotective_mu);cudaFree(dimpurity_mu);cudaFree(dscintillator_mu);
        cudaFree(do1);cudaFree(do2); if(label_texture) cudaDestroyTextureObject(label_texture); if(label_array) cudaFreeArray(label_array); fo.close();fe.close();fa.close();
        if (c.projection.save_metadata)
            writeMetadata(out_file,g.detector_u,g.detector_v,g.views,"air_corrected_negative_log_attenuation");
        if (write_energy)
            writeMetadata(c.projection.energy_output_file,g.detector_u,g.detector_v,g.views,"detected_energy_keV_per_pixel_per_view");
        if (write_air)
            writeMetadata(c.projection.air_output_file,g.detector_u,g.detector_v,1,"processed_air_energy_keV_per_pixel_per_view");
        return true;
    } catch(const std::exception& e){err=e.what();return false;}
}
}

