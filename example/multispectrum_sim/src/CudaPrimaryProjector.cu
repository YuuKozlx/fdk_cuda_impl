#include "CudaPrimaryProjector.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
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
struct DeviceGeometry { int nu, nv, views, views_per_turn, rotation_direction, samples, nx, ny, nz, nm, ne; float pu, pv, sid, sdd, vx, vy, vz, fu, fv; float offset_u, offset_n, offset_v, source_x, source_y, source_z; float tilt_u, tilt_v, tilt_n; float start, pitch, start_z; float phantom_x, phantom_y, phantom_z, phantom_rx, phantom_ry, phantom_rz; int cylindrical, apply_flux; };

__device__ float3 add3(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ float3 scale3(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ float3 cross3(float3 a, float3 b) { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ float3 rotateAxis(float3 value, float3 axis, float angle) {
    const float c=cosf(angle), s=sinf(angle);
    return add3(add3(scale3(value,c),scale3(cross3(axis,value),s)),scale3(axis,(axis.x*value.x+axis.y*value.y+axis.z*value.z)*(1.f-c)));
}
__device__ void detectorFrame(DeviceGeometry g, float angle, float3& u, float3& v, float3& n) {
    const float ca=cosf(angle), sa=sinf(angle);
    u=make_float3(-sa,ca,0.f); v=make_float3(0.f,0.f,1.f); n=make_float3(-ca,-sa,0.f);
    v=rotateAxis(v,u,g.tilt_u); n=rotateAxis(n,u,g.tilt_u);
    u=rotateAxis(u,v,g.tilt_v); n=rotateAxis(n,v,g.tilt_v);
    u=rotateAxis(u,n,g.tilt_n); v=rotateAxis(v,n,g.tilt_n);
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
__device__ unsigned poisson(float lambda, unsigned seed) {
    if (lambda <= 0.f) return 0;
    if (lambda > 32.f) return (unsigned)max(0.f, floorf(lambda + sqrtf(lambda) * normal(seed) + 0.5f));
    float product = 1.f;
    const float limit = expf(-lambda);
    unsigned n = 0;
    do { product *= fmaxf(uniform(seed + n * 2 + 3), 1e-7f); ++n; } while (product > limit && n < 256);
    return n - 1;
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
        const int m = label_to_material[labels[(iz * g.ny + iy) * g.nx + ix]];
        if (m >= 0 && segment > 0.f)
            paths[m] += segment * 0.1f;
        s = next;
        if (tx <= next) { tx += dx; ix += sx; }
        if (ty <= next) { ty += dy; iy += sy; }
        if (tz <= next) { tz += dz; iz += sz; }
    }
}

__global__ void primaryKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, float incident, float photons_per_pixel, int use_poisson,
    int view, unsigned seed, unsigned photon_seed,
    float* log_out, float* energy_out) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = g.nu * g.nv; if (pixel >= pixels) return;
    const int u0 = pixel % g.nu, v0 = pixel / g.nu;
    float sum_energy = 0.f;
    float sum_air_energy = 0.f;
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
        const float3 nominal_naxis = make_float3(-ca, -sa, 0.f);
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
            scale3(nominal_naxis, detector_distance));
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
                scale3(naxis, g.offset_n - g.sdd));
            const float3 radial = add3(scale3(naxis, cosf(gamma)),
                scale3(uaxis, sinf(gamma)));
            det = add3(add3(cylinder_center, scale3(radial, g.sdd)),
                scale3(vaxis, center_v));
        } else {
            det = detector_center;
        }
        float3 d = make_float3(det.x-source.x, det.y-source.y, det.z-source.z);
        float length = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z); d.x/=length; d.y/=length; d.z/=length;
        float paths[32] = {}; if (g.nm > 32) return;
        accumulatePaths(g, labels, label_to_material, phantomInverse(g, source, false), phantomInverse(g, d, true), lo, hi, paths);
        float r2=length*length; float flux=(!g.cylindrical && g.apply_flux) ? g.sdd*g.sdd/fmaxf(r2,1e-6f) : 1.f;
        sum_air_energy += photons_per_pixel / static_cast<float>(g.samples) * flux * incident;
        float noisy_energy = 0.f;
        const float photons_per_sample = photons_per_pixel / static_cast<float>(g.samples);
        for(int e=0;e<g.ne;++e){float exponent=0.f;for(int m=0;m<g.nm;++m) exponent+=paths[m]*mu[m*g.ne+e]; const float transmission=expf(-exponent); const float lambda=photons_per_sample*spectrum[e]*flux*transmission; const float count=use_poisson ? static_cast<float>(poisson(lambda, photon_seed ^ base ^ (unsigned)e*0x27d4eb2dU)) : lambda; noisy_energy += count*energies[e];}
        sum_energy += noisy_energy;
    }
    // Actual transmitted counts fluctuate; normalize only by the configured
    // incident expectation so the Poisson count fluctuation remains visible.
    const float avg_energy = sum_energy / photons_per_pixel;
    const float avg_air_energy = sum_air_energy / photons_per_pixel;
    // Use the same ray flux for flat-field normalization.
    log_out[pixel]=-logf(fmaxf(avg_energy / fmaxf(avg_air_energy, 1e-30f), 1e-30f));
    energy_out[pixel]=avg_energy;
}

__global__ void globalRandomKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, float incident, float photons_per_sample, std::uint64_t total_samples,
    int use_poisson, int view, unsigned seed, unsigned photon_seed, float* energy_accum, float* air_accum,
    unsigned int* sample_count) {
    const std::uint64_t thread_id = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const float pi = 3.14159265358979323846f;
    const float angle = g.start + static_cast<float>(g.rotation_direction) * 2.f*pi*view / g.views_per_turn;
    const float frame_angle = angle - 0.5f*pi;
    const float ca = cosf(frame_angle), sa = sinf(frame_angle);
    const float z = g.start_z + g.pitch * (angle - g.start) / (2.f*pi);
    const float3 lo = make_float3(-0.5f*g.nx*g.vx, -0.5f*g.ny*g.vy, -0.5f*g.nz*g.vz);
    const float3 hi = make_float3(-lo.x, -lo.y, -lo.z);
    for (std::uint64_t sample = thread_id; sample < total_samples; sample += stride) {
        const unsigned base = seed ^ static_cast<unsigned>(sample * 0x9e3779b9ULL)
            ^ static_cast<unsigned>(view * 0x85ebca6bU);
        const float detector_u = (uniform(base + 1) - .5f) * g.nu * g.pu;
        const float detector_v = (uniform(base + 2) - .5f) * g.nv * g.pv;
        const int u = max(0, min(g.nu - 1, static_cast<int>(floorf(detector_u / g.pu + .5f*g.nu))));
        const int v = max(0, min(g.nv - 1, static_cast<int>(floorf(detector_v / g.pv + .5f*g.nv))));
        const int pixel = v * g.nu + u;
        const float fsu = (uniform(base + 3) - .5f) * g.fu;
        const float fsv = (uniform(base + 4) - .5f) * g.fv;
        float3 uaxis, vaxis, naxis; detectorFrame(g, frame_angle, uaxis, vaxis, naxis);
        const float3 nominal_naxis = make_float3(-ca, -sa, 0.f);
        const float3 source = make_float3((g.sid+fsu)*ca + g.source_x*ca - g.source_y*sa,
            (g.sid+fsu)*sa + g.source_x*sa + g.source_y*ca, z+fsv+g.source_z);
        const float center_u = detector_u + g.offset_u;
        const float center_v = detector_v + g.offset_v;
        float3 det;
        if (g.cylindrical) {
            const float gamma = center_u / g.sdd;
            const float detector_distance = g.sdd - g.sid;
            const float3 isocenter = make_float3(0.f, 0.f, z);
            const float3 detector_principal = add3(isocenter,
                scale3(nominal_naxis, detector_distance));
            const float3 cylinder_center = add3(detector_principal,
                scale3(naxis, g.offset_n - g.sdd));
            const float3 radial = add3(scale3(naxis, cosf(gamma)),
                scale3(uaxis, sinf(gamma)));
            det = add3(add3(cylinder_center, scale3(radial, g.sdd)),
                scale3(vaxis, center_v));
        } else {
            const float detector_distance = g.sdd - g.sid;
            const float3 isocenter = make_float3(0.f, 0.f, z);
            const float3 detector_principal = add3(isocenter,
                scale3(nominal_naxis, detector_distance));
            det = add3(add3(add3(detector_principal,
                scale3(naxis, g.offset_n)),
                scale3(uaxis, center_u)), scale3(vaxis, center_v));
        }
        float3 d = make_float3(det.x-source.x, det.y-source.y, det.z-source.z);
        const float length = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);
        d.x/=length; d.y/=length; d.z/=length;
        float paths[32] = {}; if (g.nm > 32) return;
        accumulatePaths(g, labels, label_to_material, phantomInverse(g, source, false), phantomInverse(g, d, true), lo, hi, paths);
        const float r2=length*length;
        const float flux=(!g.cylindrical && g.apply_flux) ? g.sdd*g.sdd/fmaxf(r2,1e-6f) : 1.f;
        atomicAdd(&air_accum[pixel], photons_per_sample * flux * incident);
        float noisy_energy=0.f;
        for(int e=0;e<g.ne;++e){float exponent=0.f;for(int m=0;m<g.nm;++m) exponent+=paths[m]*mu[m*g.ne+e]; const float lambda=photons_per_sample*spectrum[e]*flux*expf(-exponent); const float count=use_poisson ? static_cast<float>(poisson(lambda, photon_seed ^ base ^ (unsigned)e*0x27d4eb2dU)) : lambda; noisy_energy += count*energies[e];}
        atomicAdd(&energy_accum[pixel], noisy_energy);
        atomicAdd(&sample_count[pixel], 1U);
    }
}

bool ok(cudaError_t e, const char* what, std::string& err) { if(e==cudaSuccess)return true; err=std::string(what)+": "+cudaGetErrorString(e); return false; }
}

bool CudaPrimaryProjector::run(const SimulationConfig& c, const std::vector<std::uint8_t>& labels,
    const SpectralTransmissionModel& model, const std::filesystem::path& out_file, std::string& err) const {
    try {
        const auto& g=c.geometry.parameters; const int pixels=g.detector_u*g.detector_v; const int nm=(int)c.projection.materials.size(), ne=(int)model.spectrum().size();
        if(g.volume_x<=0||g.volume_y<=0||g.volume_z<=0||labels.size()!=(size_t)g.volume_x*g.volume_y*g.volume_z) throw std::runtime_error("CUDA前投标签体尺寸不匹配");
        if(nm > 32) throw std::runtime_error("CUDA primary projector supports at most 32 materials");
        const int cylindrical = (c.geometry.kind == GeometryKind::CylCbct || c.geometry.kind == GeometryKind::CylHelical) ? 1 : 0;
        DeviceGeometry dg{g.detector_u,g.detector_v,g.views,g.views_per_turn,g.rotation_direction,c.projection.sampling.samples_per_pixel,g.volume_x,g.volume_y,g.volume_z,nm,ne,(float)g.pixel_u_mm,(float)g.pixel_v_mm,(float)g.sid_mm,(float)g.sdd_mm,(float)g.voxel_x_mm,(float)g.voxel_y_mm,(float)g.voxel_z_mm,(float)(c.projection.focal_spot.enabled?c.projection.focal_spot.size_u_mm:0),(float)(c.projection.focal_spot.enabled?c.projection.focal_spot.size_v_mm:0),(float)g.offset_u_mm,(float)g.offset_n_mm,(float)g.offset_v_mm,(float)g.source_offset_x_mm,(float)g.source_offset_y_mm,(float)g.source_offset_z_mm,(float)g.tilt_u_rad,(float)g.tilt_v_rad,(float)g.tilt_n_rad,(float)g.start_angle_rad,(float)g.pitch_mm_per_turn,(float)g.start_z_mm,(float)g.phantom_offset_x_mm,(float)g.phantom_offset_y_mm,(float)g.phantom_offset_z_mm,(float)g.phantom_rotation_x_rad,(float)g.phantom_rotation_y_rad,(float)g.phantom_rotation_z_rad,cylindrical,c.projection.apply_geometry_flux ? 1 : 0};
        std::vector<int> map(256,-1); for(int m=0;m<nm;++m)map[c.projection.materials[m].label]=m;
        std::vector<float> mu(nm*ne), sw(ne), en(ne); float incident=0, spectrum_sum=0;
        for(int e=0;e<ne;++e) spectrum_sum+=(float)model.spectrum()[e].relative_photons;
        if (!(spectrum_sum > 0.f)) throw std::runtime_error("spectrum photon weights must sum to a positive value");
        for(int e=0;e<ne;++e){sw[e]=(float)model.spectrum()[e].relative_photons/spectrum_sum;en[e]=(float)model.spectrum()[e].energy_keV;incident+=sw[e]*en[e];for(int m=0;m<nm;++m)mu[m*ne+e]=(float)(c.projection.materials[m].density_g_cm3*model.massAttenuation(c.projection.materials[m].label)[e]);}
        unsigned char* dl=nullptr; int* dm=nullptr; float *dmu=nullptr,*ds=nullptr,*de=nullptr,*do1=nullptr,*do2=nullptr,*denergy_accum=nullptr,*dair_accum=nullptr; unsigned int* dcount=nullptr; auto ck=[&](cudaError_t x,const char* w){return ok(x,w,err);};
        if(!ck(cudaMalloc(&dl,labels.size()),"cudaMalloc labels")||!ck(cudaMalloc(&dm,256*sizeof(int)),"cudaMalloc map")||!ck(cudaMalloc(&dmu,mu.size()*sizeof(float)),"cudaMalloc mu")||!ck(cudaMalloc(&ds,sw.size()*sizeof(float)),"cudaMalloc spectrum")||!ck(cudaMalloc(&de,en.size()*sizeof(float)),"cudaMalloc energies")||!ck(cudaMalloc(&do1,pixels*sizeof(float)),"cudaMalloc projection")||!ck(cudaMalloc(&do2,pixels*sizeof(float)),"cudaMalloc energy")) throw std::runtime_error(err);
        cudaMemcpy(dl,labels.data(),labels.size(),cudaMemcpyHostToDevice);cudaMemcpy(dm,map.data(),256*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dmu,mu.data(),mu.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(ds,sw.data(),sw.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(de,en.data(),en.size()*sizeof(float),cudaMemcpyHostToDevice);
        if (c.projection.sampling.mode == "detector_global_random") {
            if (!ck(cudaMalloc(&denergy_accum,pixels*sizeof(float)),"cudaMalloc global energy") ||
                !ck(cudaMalloc(&dair_accum,pixels*sizeof(float)),"cudaMalloc global air") ||
                !ck(cudaMalloc(&dcount,pixels*sizeof(unsigned int)),"cudaMalloc global count")) throw std::runtime_error(err);
        }
        std::vector<float> ho(pixels),he(pixels); std::ofstream fo(out_file,std::ios::binary), fe(c.projection.energy_output_file,std::ios::binary); if(!fo||!fe)throw std::runtime_error("无法创建CUDA前投输出");
        for(int v=0;v<g.views;++v){
            const int use_poisson = c.projection.sampling.photon_count_mode == "poisson" ? 1 : 0;
            if (c.projection.sampling.mode == "detector_global_random") {
                const std::uint64_t total = c.projection.sampling.total_samples;
                const float photons_per_sample = (float)(c.projection.sampling.photons_per_pixel * pixels / static_cast<double>(total));
                cudaMemset(denergy_accum,0,pixels*sizeof(float)); cudaMemset(dair_accum,0,pixels*sizeof(float)); cudaMemset(dcount,0,pixels*sizeof(unsigned int));
                const int blocks = 65535;
                globalRandomKernel<<<blocks,256>>>(dg,dl,dm,dmu,ds,de,incident,photons_per_sample,total,use_poisson,v,c.projection.sampling.seed,c.projection.sampling.photon_seed,denergy_accum,dair_accum,dcount);
                if(!ck(cudaGetLastError(),"global random kernel")||!ck(cudaDeviceSynchronize(),"global random synchronize"))throw std::runtime_error(err);
                std::vector<float> air(pixels); std::vector<unsigned int> counts(pixels);
                cudaMemcpy(he.data(),denergy_accum,pixels*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(air.data(),dair_accum,pixels*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(counts.data(),dcount,pixels*sizeof(unsigned int),cudaMemcpyDeviceToHost);
                const float expected_samples_per_pixel = static_cast<float>(total) / pixels;
                for(int p=0;p<pixels;++p){
                    const float spatial_scale = counts[p] > 0 ? expected_samples_per_pixel / counts[p] : 0.f;
                    const float energy = he[p] * spatial_scale / (float)c.projection.sampling.photons_per_pixel;
                    const float air_energy = air[p] * spatial_scale / (float)c.projection.sampling.photons_per_pixel;
                    he[p]=energy;
                    ho[p]=-std::log(std::max(energy/std::max(air_energy,1e-30f),1e-30f));
                }
                fe.write((char*)he.data(),pixels*sizeof(float)); fo.write((char*)ho.data(),pixels*sizeof(float));
            } else {
                primaryKernel<<<(pixels+255)/256,256>>>(dg,dl,dm,dmu,ds,de,incident,(float)c.projection.sampling.photons_per_pixel,use_poisson,v,c.projection.sampling.seed,c.projection.sampling.photon_seed,do1,do2);if(!ck(cudaGetLastError(),"primary kernel")||!ck(cudaDeviceSynchronize(),"primary synchronize"))throw std::runtime_error(err);cudaMemcpy(ho.data(),do1,pixels*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(he.data(),do2,pixels*sizeof(float),cudaMemcpyDeviceToHost);fo.write((char*)ho.data(),pixels*sizeof(float));fe.write((char*)he.data(),pixels*sizeof(float));
            }
        }
        cudaFree(dl);cudaFree(dm);cudaFree(dmu);cudaFree(ds);cudaFree(de);cudaFree(do1);cudaFree(do2);cudaFree(denergy_accum);cudaFree(dair_accum);cudaFree(dcount); fo.close();fe.close();
        writeMetadata(out_file,g.detector_u,g.detector_v,g.views,"negative_log_transmission");
        writeMetadata(c.projection.energy_output_file,g.detector_u,g.detector_v,g.views,"transmitted_energy_keV_per_incident_photon");
        return true;
    } catch(const std::exception& e){err=e.what();return false;}
}
}
