#pragma once

// Material path integration and spectral transmission. Included only by the
// projector CUDA translation unit, which owns these kernels and constant tables.

#include "MaterialPaths.cuh"
#include "QuantumSampling.cuh"
#include "FocalSampling.cuh"
#include <iostream>

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

template <int MaterialCount,bool Instrument=false,int EnergyCapacity=256>
__global__ void primaryKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, float incident, int quantum_noise,
    int view, unsigned seed, unsigned quantum_seed,
    float* expected_energy_out, float* sampled_energy_out, float* air_out,
    unsigned char* object_mask_out,unsigned long long* timing=nullptr) {
    view += int(blockIdx.y);
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = g.nu * g.nv; if (pixel >= pixels) return;
    const int u0 = pixel % g.nu, v0 = pixel / g.nu;
    const bool profile=Instrument&&((hash32(unsigned(pixel)^unsigned(view)*0x85ebca6bU)&1023U)==0);
    unsigned long long geometry_cycles=0,path_cycles=0,spectrum_cycles=0,noise_cycles=0;
    unsigned ray_count=0,hit_count=0,visited=0;
    if (g.ne > EnergyCapacity) return;
    float expected_counts[EnergyCapacity] = {};
    float expected_air_energy = 0.f;
    bool object_hit = false;
    float3 lo = make_float3(-0.5f*g.nx*g.vx, -0.5f*g.ny*g.vy, -0.5f*g.nz*g.vz);
    float3 hi = make_float3(-lo.x, -lo.y, -lo.z);
    const auto focal_partition=makeFocalPartition(g);
    // N inner rays serve both normal and broad-focus inner exposure; N outer
    // rays complete the area-weighted broad focus without increasing dose.
    for (int component=0; component<2; ++component) {
        const bool defocused = component == 1;
        const float exposure_fraction = focalPartitionWeight(focal_partition,defocused);
        if (exposure_fraction <= 0.f) continue;
        for (int sample=0; sample<g.samples; ++sample) {
        const auto geometry_start=profile?clock64():0ULL;
        unsigned base = seed ^ (unsigned(view)*unsigned(pixels)+unsigned(pixel))*0x9e3779b9U ^ (unsigned)sample*0x85ebca6bU;
        const auto focal=sampleFocalPoint(g,focal_partition,defocused,base);
        const float fsu=focal.x,fsv=focal.y;
        const float jitter_u = uniform(base+3)-.5f;
        const float jitter_v = uniform(base+4)-.5f;
        const auto ray = scannerRay(g, view, static_cast<float>(u0),
            static_cast<float>(v0), fsu, fsv, jitter_u, jitter_v);
        const float3 source = ray.source, det = ray.detector;
        float3 d = make_float3(det.x-source.x, det.y-source.y, det.z-source.z);
        float length = sqrtf(d.x*d.x+d.y*d.y+d.z*d.z); d.x/=length; d.y/=length; d.z/=length;
        float paths[MaterialCount] = {};
        const auto local_source=phantomInverse(g,source,false),local_direction=phantomInverse(g,d,true);
        if(profile)geometry_cycles+=clock64()-geometry_start;
        const auto path_start=profile?clock64():0ULL;
        const bool hit=accumulatePaths<MaterialCount>(g,labels,label_to_material,local_source,local_direction,lo,hi,paths,profile?&visited:nullptr);
        object_hit=hit||object_hit;
        if(profile){path_cycles+=clock64()-path_start;++ray_count;if(hit)++hit_count;}
        const auto spectrum_start=profile?clock64():0ULL;
        float r2=length*length; float flux=g.geometry_flux ? 1000000.f/fmaxf(r2,1e-6f) : 1.f;
        // Geometry samples share one pixel exposure. They are not separate
        // exposures; each sample carries 1/g.samples of the pixel photons.
        const float photons_per_sample = exposure_fraction / static_cast<float>(g.samples);
        for(int e=0;e<g.ne;++e){float exponent=0.f;
            // A ray through vacuum has exactly unit transmission. Keep its
            // original weights and accumulation order, but skip attenuation.
            if(hit){
                if constexpr(MaterialCount>=32){
                    for(int m=0;m<g.nm;++m)exponent+=paths[m]*readMu(g,mu,m*g.ne+e);
                }else{
                    #pragma unroll
                    for(int m=0;m<MaterialCount;++m) exponent+=paths[m]*readMu(g, mu, m*g.ne+e);
                }
            }
            const float transmission=hit ? expf(-exponent) : 1.f;
            const float response=g.detector_response ? __ldg(g.detector_efficiency + (v0*g.nu+u0)*g.ne+e) : 1.f;
            const float weight=photons_per_sample*readSpectrum(g, spectrum, e)*flux*response;
            const float lambda = weight * transmission;
            expected_counts[e] += lambda;
            expected_air_energy += weight * readEnergy(g, energies, e);
        }
        if(profile)spectrum_cycles+=clock64()-spectrum_start;
        }
    }
    const auto noise_start=profile?clock64():0ULL;
    float expected_energy = 0.f;
    float sampled_energy = 0.f;
    for (int e=0; e<g.ne; ++e) {
        const unsigned noise_seed = quantum_seed
            ^ (unsigned(view)*unsigned(pixels)+unsigned(pixel)) * 0x9e3779b9U
            ^ (unsigned)e * 0x27d4eb2dU;
        const float sampled_count = quantum_noise ? poissonSample(expected_counts[e], noise_seed)
                                                  : expected_counts[e];
        expected_energy += expected_counts[e] * readEnergy(g, energies, e);
        sampled_energy += sampled_count * readEnergy(g, energies, e);
    }
    if(profile){noise_cycles=clock64()-noise_start;
        atomicAdd(timing+0,geometry_cycles);atomicAdd(timing+1,path_cycles);
        atomicAdd(timing+2,spectrum_cycles);atomicAdd(timing+3,noise_cycles);
        atomicAdd(timing+4,1ULL);atomicAdd(timing+5,(unsigned long long)ray_count);
        atomicAdd(timing+6,(unsigned long long)hit_count);atomicAdd(timing+7,(unsigned long long)visited);
    }
    const float sum_air_energy = expected_air_energy;
    // Actual transmitted counts fluctuate; normalize only by the configured
    // incident expectation so the Poisson count fluctuation remains visible.
    // Return energy per incident photon. The configured exposure is applied
    // only inside the sample weights above, so increasing g.samples does not
    // increase dose.
    const float avg_air_energy = sum_air_energy;
    const size_t output=size_t(blockIdx.y)*pixels+pixel;
    expected_energy_out[output]=expected_energy;
    sampled_energy_out[output]=sampled_energy;
    air_out[output]=avg_air_energy;
    object_mask_out[output] = object_hit ? 1 : 0;
}

template <int MaterialCount,int EnergyCapacity>
void launchPrimaryKernelSized(const DeviceGeometry& geometry,
    const unsigned char* labels, const int* label_to_material,
    const float* mu, const float* spectrum, const float* energies,
    float incident, int quantum_noise, int view, unsigned seed,
    unsigned quantum_seed, float* expected_energy_out,
    float* sampled_energy_out, float* air_out, unsigned char* object_mask_out, int views=1,unsigned long long* timing=nullptr) {
    const int pixels = geometry.nu * geometry.nv;
    if(timing){
        static bool reported=false;
        if(!reported){cudaFuncAttributes normal{},instrumented{};
            cudaFuncGetAttributes(&normal,primaryKernel<MaterialCount,false,EnergyCapacity>);cudaFuncGetAttributes(&instrumented,primaryKernel<MaterialCount,true,EnergyCapacity>);
            std::cout<<"primary_resources energy_capacity="<<EnergyCapacity<<" materials="<<MaterialCount<<" normal_registers="<<normal.numRegs<<" normal_local_bytes="<<normal.localSizeBytes
                <<" profiled_registers="<<instrumented.numRegs<<" profiled_local_bytes="<<instrumented.localSizeBytes<<std::endl;reported=true;}
        primaryKernel<MaterialCount,true,EnergyCapacity><<<dim3((pixels+255)/256,views),256>>>(geometry,labels,label_to_material,mu,spectrum,energies,incident,quantum_noise,view,seed,quantum_seed,expected_energy_out,sampled_energy_out,air_out,object_mask_out,timing);
    }else{
    primaryKernel<MaterialCount,false,EnergyCapacity><<< dim3((pixels + 255) / 256,views), 256 >>>(
        geometry, labels, label_to_material, mu, spectrum, energies,
        incident, quantum_noise, view, seed, quantum_seed,
        expected_energy_out, sampled_energy_out, air_out, object_mask_out);
    }
}

// The path and photon accumulation order is unchanged. The common <=128-bin
// spectra need half the dynamically indexed per-thread counter storage.
template <int MaterialCount>
void launchPrimaryKernel(const DeviceGeometry& geometry,
    const unsigned char* labels, const int* label_to_material,
    const float* mu, const float* spectrum, const float* energies,
    float incident, int quantum_noise, int view, unsigned seed,
    unsigned quantum_seed, float* expected_energy_out,
    float* sampled_energy_out, float* air_out, unsigned char* object_mask_out,
    int views=1,unsigned long long* timing=nullptr) {
    if(geometry.ne<=128)
        launchPrimaryKernelSized<MaterialCount,128>(geometry,labels,label_to_material,
            mu,spectrum,energies,incident,quantum_noise,view,seed,quantum_seed,
            expected_energy_out,sampled_energy_out,air_out,object_mask_out,views,timing);
    else
        launchPrimaryKernelSized<MaterialCount,256>(geometry,labels,label_to_material,
            mu,spectrum,energies,incident,quantum_noise,view,seed,quantum_seed,
            expected_energy_out,sampled_energy_out,air_out,object_mask_out,views,timing);
}

} // namespace yk::spectral::primary_detail
