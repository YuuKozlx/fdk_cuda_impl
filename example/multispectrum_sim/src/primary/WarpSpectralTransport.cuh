#pragma once

// Default independent warp projector. SpectralTransport.cuh's scalar kernel
// remains available as a selectable fallback and numeric reference.
#include "SpectralTransport.cuh"

namespace yk::spectral::primary_detail {

template<int MaterialCount, int EnergyCapacity=128, int Warps=4>
__global__ void warpPrimaryKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, int quantum_noise, int first_view,
    unsigned seed, unsigned quantum_seed, float* expected_out,
    float* sampled_out, float* air_out, unsigned char* mask_out) {
    constexpr int Slots=EnergyCapacity/32;
    const int lane=threadIdx.x&31, warp=threadIdx.x/32;
    const int pixel=blockIdx.x*Warps+warp, pixels=g.nu*g.nv;
    // Each complete warp owns one pixel; partial final blocks discard full warps.
    if(pixel>=pixels||g.ne>EnergyCapacity)return;
    const int view=first_view+blockIdx.y;
    __shared__ float paths_tile[Warps][MaterialCount][32];
    __shared__ float flux_tile[Warps][32];
    __shared__ unsigned hit_tile[Warps][32];
    float counts[Slots]={},air_counts[Slots]={};
    float energy[Slots],sw[Slots],response[Slots];
    #pragma unroll
    for(int k=0;k<Slots;++k){
        const int e=lane+k*32;
        // Adjacent lanes read adjacent energies. Global read-only tables avoid
        // constant-memory serialization of different addresses within a warp.
        energy[k]=e<g.ne?__ldg(energies+e):0.f;
        sw[k]=e<g.ne?__ldg(spectrum+e):0.f;
        response[k]=(e<g.ne&&g.detector_response)?__ldg(g.detector_efficiency+size_t(pixel)*g.ne+e):1.f;
    }
    bool object_hit=false;
    const float3 lo=make_float3(-.5f*g.nx*g.vx,-.5f*g.ny*g.vy,-.5f*g.nz*g.vz);
    const float3 hi=make_float3(-lo.x,-lo.y,-lo.z);
    for(int component=0;component<2;++component){
        const bool defocused=component==1;
        const float fraction=defocused?g.focal_epsilon:1.f-g.focal_epsilon;
        if(fraction<=0.f)continue;
        const float photons=fraction/static_cast<float>(g.samples);
        for(int start=0;start<g.samples;start+=32){
            const int sample=start+lane, active=min(32,g.samples-start);
            float paths[MaterialCount]={};float flux=0.f;bool hit=false;
            if(sample<g.samples){
                const unsigned base=seed^(unsigned(view)*unsigned(pixels)+unsigned(pixel))*0x9e3779b9U^unsigned(sample)*0x85ebca6bU;
                const float fu=defocused?g.defocus_fu:g.normal_fu,fv=defocused?g.defocus_fv:g.normal_fv;
                const auto ray=scannerRay(g,view,float(pixel%g.nu),float(pixel/g.nu),
                    (uniform(base+1)-.5f)*fu,(uniform(base+2)-.5f)*fv,uniform(base+3)-.5f,uniform(base+4)-.5f);
                float3 d=make_float3(ray.detector.x-ray.source.x,ray.detector.y-ray.source.y,ray.detector.z-ray.source.z);
                const float length=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=length;d.y/=length;d.z/=length;
                hit=accumulatePaths<MaterialCount>(g,labels,label_to_material,
                    phantomInverse(g,ray.source,false),phantomInverse(g,d,true),lo,hi,paths);
                flux=g.geometry_flux?1000000.f/fmaxf(length*length,1e-6f):1.f;
            }
            #pragma unroll
            for(int m=0;m<MaterialCount;++m)paths_tile[warp][m][lane]=paths[m];
            flux_tile[warp][lane]=flux;hit_tile[warp][lane]=hit;
            object_hit=object_hit||(__ballot_sync(0xffffffffU,hit)!=0);
            __syncwarp();
            // Preserve the original component/sample order independently for
            // every energy. No averaging of paths before the exponential.
            for(int j=0;j<active;++j){
                const bool material=hit_tile[warp][j]!=0;
                const float ray_flux=flux_tile[warp][j];
                #pragma unroll
                for(int k=0;k<Slots;++k){
                    const int e=lane+k*32;
                    if(e<g.ne){
                        float exponent=0.f;
                        if(material){
                            if constexpr(MaterialCount==32){
                                for(int m=0;m<g.nm;++m)exponent+=paths_tile[warp][m][j]*__ldg(mu+m*g.ne+e);
                            }else{
                                #pragma unroll
                                for(int m=0;m<MaterialCount;++m)exponent+=paths_tile[warp][m][j]*__ldg(mu+m*g.ne+e);
                            }
                        }
                        const float transmission=material?expf(-exponent):1.f;
                        const float weight=photons*sw[k]*ray_flux*response[k];
                        counts[k]+=weight*transmission;
                        air_counts[k]+=weight*energy[k];
                    }
                }
            }
            // Protect the shared tile before lanes start traversing next rays.
            __syncwarp();
        }
    }
    float expected=0.f,sampled=0.f,air=0.f;
    #pragma unroll
    for(int k=0;k<Slots;++k){
        const int e=lane+k*32;
        if(e<g.ne){
            const unsigned noise_seed=quantum_seed^(unsigned(view)*unsigned(pixels)+unsigned(pixel))*0x9e3779b9U^unsigned(e)*0x27d4eb2dU;
            const float n=quantum_noise?poissonSample(counts[k],noise_seed):counts[k];
            expected+=counts[k]*energy[k];sampled+=n*energy[k];air+=air_counts[k];
        }
    }
    // Different final reduction order can produce floating-point last-bit
    // differences versus the scalar reference, including the air reference.
    for(int offset=16;offset>0;offset/=2){
        expected+=__shfl_down_sync(0xffffffffU,expected,offset);
        sampled+=__shfl_down_sync(0xffffffffU,sampled,offset);
        air+=__shfl_down_sync(0xffffffffU,air,offset);
    }
    if(lane==0){const size_t out=size_t(blockIdx.y)*pixels+pixel;
        expected_out[out]=expected;sampled_out[out]=sampled;air_out[out]=air;mask_out[out]=object_hit?1:0;}
}

template<int MaterialCount,int EnergyCapacity>
void launchWarpPrimarySized(const DeviceGeometry& g,const unsigned char* labels,
    const int* map,const float* mu,const float* spectrum,const float* energies,
    int quantum_noise,int view,unsigned seed,unsigned quantum_seed,float* expected,
    float* sampled,float* air,unsigned char* mask,int views,bool report){
    constexpr int Warps=4;
    if(report){static bool reported=false;if(!reported){cudaFuncAttributes a{};
        cudaFuncGetAttributes(&a,warpPrimaryKernel<MaterialCount,EnergyCapacity,Warps>);
        std::cout<<"warp_primary_resources materials="<<MaterialCount<<" energy_capacity="<<EnergyCapacity
            <<" registers="<<a.numRegs<<" local_bytes="<<a.localSizeBytes<<" shared_bytes="<<a.sharedSizeBytes<<std::endl;
        reported=true;}}
    warpPrimaryKernel<MaterialCount,EnergyCapacity,Warps><<<dim3((g.nu*g.nv+Warps-1)/Warps,views),32*Warps>>>(
        g,labels,map,mu,spectrum,energies,quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask);
}

template<int MaterialCount>
void launchWarpPrimaryKernel(const DeviceGeometry& g,const unsigned char* labels,
    const int* map,const float* mu,const float* spectrum,const float* energies,
    int quantum_noise,int view,unsigned seed,unsigned quantum_seed,float* expected,
    float* sampled,float* air,unsigned char* mask,int views=1,bool report=false){
    if(g.ne<=128)launchWarpPrimarySized<MaterialCount,128>(g,labels,map,mu,spectrum,energies,
        quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report);
    else launchWarpPrimarySized<MaterialCount,256>(g,labels,map,mu,spectrum,energies,
        quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report);
}

} // namespace yk::spectral::primary_detail
