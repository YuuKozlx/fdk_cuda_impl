#pragma once

// Default independent warp projector. SpectralTransport.cuh's scalar kernel
// remains available as a selectable fallback and numeric reference.
#include "SpectralTransport.cuh"
#include "primary/PrimaryLaunchPolicy.hpp"

namespace yk::spectral::primary_detail {
// Sampled warp residence cycles, not additive GPU wall-clock milliseconds.
// setup, geometry/path/init, shared exchange, spectrum, noise/energy sum,
// reduction/output, sampled warps, sampled rays, hit rays, traversed voxels.
inline constexpr int WarpProfileCounterCount=10;

template<int MaterialCount, int EnergyCapacity=128, int Warps=4, bool CacheCoefficients=false, bool SparseMaterials=false, bool Instrument=false, bool SharedPaths=false, int GroupWidth=32>
__global__ void warpPrimaryKernel(DeviceGeometry g, const unsigned char* labels,
    const int* label_to_material, const float* mu, const float* spectrum,
    const float* energies, int quantum_noise, int first_view,
    unsigned seed, unsigned quantum_seed, float* expected_out,
    float* sampled_out, float* air_out, unsigned char* mask_out, unsigned long long* timing=nullptr) {
    static_assert(GroupWidth==4||GroupWidth==8||GroupWidth==16||GroupWidth==32,"supported spectral thread group sizes");
    constexpr int Slots=EnergyCapacity/GroupWidth;
    const int lane=threadIdx.x%GroupWidth, warp=threadIdx.x/GroupWidth;
    const unsigned group_mask=(0xffffffffU>>(32-GroupWidth))<<((threadIdx.x&31)/GroupWidth*GroupWidth);
    const int pixel=blockIdx.x*Warps+warp, pixels=g.nu*g.nv;
    // Each complete thread group owns one pixel; groups outside the image return.
    if(pixel>=pixels||g.ne>EnergyCapacity)return;
    const int view=first_view+blockIdx.y;
    const bool profile=Instrument&&((hash32(unsigned(pixel)^unsigned(view)*0x85ebca6bU)&1023U)==0);
    unsigned long long cycles[6]={},tick=0,rays=0,hits=0;
    unsigned visited=0;
    if(profile){__syncwarp(group_mask);if(lane==0)tick=clock64();}
    // Large comparison blocks opt in to >48 KiB dynamic shared memory.
    // Baseline keeps its original static layout and launch requirements.
    extern __shared__ float dynamic_paths[];
    __shared__ float static_paths[SharedPaths?1:Warps][MaterialCount][GroupWidth];
    auto paths_tile=SharedPaths?reinterpret_cast<float (*)[MaterialCount][GroupWidth]>(dynamic_paths):static_paths;
    __shared__ float flux_tile[Warps][GroupWidth];
    __shared__ unsigned hit_tile[Warps][GroupWidth];
    __shared__ unsigned long long material_masks[SparseMaterials?Warps:1][SparseMaterials?32:1];
    float counts[Slots]={},air_counts[Slots]={};
    float energy[Slots],sw[Slots],response[Slots];
    #pragma unroll
    for(int k=0;k<Slots;++k){
        const int e=lane+k*GroupWidth;
        // Adjacent lanes read adjacent energies. Global read-only tables avoid
        // constant-memory serialization of different addresses within a warp.
        energy[k]=e<g.ne?__ldg(energies+e):0.f;
        sw[k]=e<g.ne?__ldg(spectrum+e):0.f;
        response[k]=(e<g.ne&&g.detector_response)?__ldg(g.detector_efficiency+size_t(pixel)*g.ne+e):1.f;
    }
    // Only the cached specialization allocates these statically indexed slots.
    // No coefficient reads remain inside its per-ray spectral loop.
    float coefficients[CacheCoefficients?Slots:1][CacheCoefficients?MaterialCount:1];
    if constexpr(CacheCoefficients){
        #pragma unroll
        for(int k=0;k<Slots;++k){
            #pragma unroll
            for(int m=0;m<MaterialCount;++m)
                coefficients[k][m]=(lane+k*GroupWidth<g.ne&&m<g.nm)?__ldg(mu+m*g.ne+lane+k*GroupWidth):0.f;
        }
    }
    const auto focal_partition=makeFocalPartition(g);
    bool object_hit=false;
    const float3 lo=make_float3(-.5f*g.nx*g.vx,-.5f*g.ny*g.vy,-.5f*g.nz*g.vz);
    const float3 hi=make_float3(-lo.x,-lo.y,-lo.z);
    if(profile){__syncwarp(group_mask);if(lane==0)cycles[0]+=clock64()-tick;}
    for(int component=0;component<2;++component){
        const bool defocused=component==1;
        const float fraction=focalPartitionWeight(focal_partition,defocused);
        if(fraction<=0.f)continue;
        const float photons=fraction/static_cast<float>(g.samples);
        for(int start=0;start<g.samples;start+=GroupWidth){
            if(profile){__syncwarp(group_mask);if(lane==0)tick=clock64();}
            const int sample=start+lane, active=min(GroupWidth,g.samples-start);
            // The independent shared-path specialization avoids the dynamically
            // indexed per-thread array (and its local-memory traffic).
            float local_paths[SharedPaths?1:MaterialCount]={};
            float* paths=SharedPaths?&paths_tile[warp][0][lane]:local_paths;
            constexpr int PathStride=SharedPaths?GroupWidth:1;
            if constexpr(SharedPaths){
                #pragma unroll
                for(int m=0;m<MaterialCount;++m)paths[m*PathStride]=0.f;
            }
            float flux=0.f;bool hit=false;
            if(sample<g.samples){
                const unsigned base=seed^(unsigned(view)*unsigned(pixels)+unsigned(pixel))*0x9e3779b9U^unsigned(sample)*0x85ebca6bU;
                const auto focal=sampleFocalPoint(g,focal_partition,defocused,base);
                const auto ray=scannerRay(g,view,float(pixel%g.nu),float(pixel/g.nu),
                    focal.x,focal.y,uniform(base+3)-.5f,uniform(base+4)-.5f);
                float3 d=make_float3(ray.detector.x-ray.source.x,ray.detector.y-ray.source.y,ray.detector.z-ray.source.z);
                const float length=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z);d.x/=length;d.y/=length;d.z/=length;
                hit=accumulatePaths<MaterialCount,PathStride>(g,labels,label_to_material,
                    phantomInverse(g,ray.source,false),phantomInverse(g,d,true),lo,hi,paths,profile?&visited:nullptr);
                flux=g.geometry_flux?1000000.f/fmaxf(length*length,1e-6f):1.f;
            }
            if constexpr(!SharedPaths){
                #pragma unroll
                for(int m=0;m<MaterialCount;++m)paths_tile[warp][m][lane]=paths[m];
            }
            if constexpr(SparseMaterials){
                unsigned long long mask=0;
                for(int m=0;m<g.nm;++m)if(paths[m*PathStride]!=0.f)mask|=1ULL<<m;
                material_masks[warp][lane]=mask;
            }
            if(profile){__syncwarp(group_mask);if(lane==0){cycles[1]+=clock64()-tick;tick=clock64();rays+=active;}}
            flux_tile[warp][lane]=flux;hit_tile[warp][lane]=hit;
            const unsigned hit_mask=__ballot_sync(group_mask,hit);
            object_hit=object_hit||(hit_mask!=0);
            __syncwarp(group_mask);
            if(profile&&lane==0){cycles[2]+=clock64()-tick;tick=clock64();hits+=__popc(hit_mask);}
            // Preserve the original component/sample order independently for
            // every energy. No averaging of paths before the exponential.
            for(int j=0;j<active;++j){
                const bool material=hit_tile[warp][j]!=0;
                const float ray_flux=flux_tile[warp][j];
                #pragma unroll
                for(int k=0;k<Slots;++k){
                    const int e=lane+k*GroupWidth;
                    if(e<g.ne){
                        float exponent=0.f;
                        if(material){
                            if constexpr(SparseMaterials){
                                auto remaining=material_masks[warp][j];
                                while(remaining){
                                    const int m=__ffsll(static_cast<long long>(remaining))-1;
                                    exponent+=paths_tile[warp][m][j]*__ldg(mu+m*g.ne+e);
                                    remaining&=remaining-1;
                                }
                            }else if constexpr((SharedPaths&&!CacheCoefficients)||MaterialCount>=32){
                                for(int m=0;m<g.nm;++m)exponent+=paths_tile[warp][m][j]*(CacheCoefficients?coefficients[k][m]:__ldg(mu+m*g.ne+e));
                            }else{
                                #pragma unroll
                                for(int m=0;m<MaterialCount;++m)exponent+=paths_tile[warp][m][j]*(CacheCoefficients?coefficients[k][m]:__ldg(mu+m*g.ne+e));
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
            __syncwarp(group_mask);
            if(profile&&lane==0)cycles[3]+=clock64()-tick;
        }
    }
    if(profile){__syncwarp(group_mask);if(lane==0)tick=clock64();}
    float expected=0.f,sampled=0.f,air=0.f;
    #pragma unroll
    for(int k=0;k<Slots;++k){
        const int e=lane+k*GroupWidth;
        if(e<g.ne){
            const unsigned noise_seed=quantum_seed^(unsigned(view)*unsigned(pixels)+unsigned(pixel))*0x9e3779b9U^unsigned(e)*0x27d4eb2dU;
            const float n=quantum_noise?poissonSample(counts[k],noise_seed):counts[k];
            expected+=counts[k]*energy[k];sampled+=n*energy[k];air+=air_counts[k];
        }
    }
    if(profile){__syncwarp(group_mask);if(lane==0){cycles[4]+=clock64()-tick;tick=clock64();}}
    // Different final reduction order can produce floating-point last-bit
    // differences versus the scalar reference, including the air reference.
    for(int offset=GroupWidth/2;offset>0;offset/=2){
        expected+=__shfl_down_sync(group_mask,expected,offset,GroupWidth);
        sampled+=__shfl_down_sync(group_mask,sampled,offset,GroupWidth);
        air+=__shfl_down_sync(group_mask,air,offset,GroupWidth);
    }
    if(lane==0){const size_t out=size_t(blockIdx.y)*pixels+pixel;
        expected_out[out]=expected;sampled_out[out]=sampled;air_out[out]=air;mask_out[out]=object_hit?1:0;}
    if(profile){
        __syncwarp(group_mask);if(lane==0)cycles[5]+=clock64()-tick;
        for(int offset=GroupWidth/2;offset>0;offset/=2)visited+=__shfl_down_sync(group_mask,visited,offset,GroupWidth);
        if(lane==0){
            for(int k=0;k<6;++k)atomicAdd(timing+k,cycles[k]);
            atomicAdd(timing+6,1ULL);atomicAdd(timing+7,rays);atomicAdd(timing+8,hits);atomicAdd(timing+9,static_cast<unsigned long long>(visited));
        }
    }
}

template<int MaterialCount,int EnergyCapacity,bool CacheCoefficients=false,bool SparseMaterials=false,bool SharedPaths=false,int GroupWidth=SharedPaths?8:32,int BlockSize=128>
void launchWarpPrimarySized(const DeviceGeometry& g,const unsigned char* labels,
    const int* map,const float* mu,const float* spectrum,const float* energies,
    int quantum_noise,int view,unsigned seed,unsigned quantum_seed,float* expected,
    float* sampled,float* air,unsigned char* mask,int views,bool report,unsigned long long* profile=nullptr){
    static_assert(BlockSize%32==0&&BlockSize%GroupWidth==0,"complete physical warps and pixel groups required");
    constexpr int Warps=BlockSize/GroupWidth;
    constexpr size_t SharedBytes=SharedPaths?Warps*MaterialCount*GroupWidth*sizeof(float):0;
    if(report||profile){static bool reported=false;if(!reported){cudaFuncAttributes a{};
        if(profile)cudaFuncGetAttributes(&a,warpPrimaryKernel<MaterialCount,EnergyCapacity,Warps,CacheCoefficients,SparseMaterials,true,SharedPaths,GroupWidth>);
        else cudaFuncGetAttributes(&a,warpPrimaryKernel<MaterialCount,EnergyCapacity,Warps,CacheCoefficients,SparseMaterials,false,SharedPaths,GroupWidth>);
        std::cout<<"warp_primary_resources materials="<<MaterialCount<<" energy_capacity="<<EnergyCapacity
            <<" group_threads="<<GroupWidth<<" block_threads="<<BlockSize
            <<" coefficient_cache="<<CacheCoefficients<<" sparse_materials="<<SparseMaterials<<" shared_paths="<<SharedPaths<<" profiled="<<(profile!=nullptr)<<" registers="<<a.numRegs<<" local_bytes="<<a.localSizeBytes<<" shared_bytes="<<a.sharedSizeBytes+SharedBytes<<" dynamic_shared_bytes="<<SharedBytes<<std::endl;
        reported=true;}}
    if(profile){
        warpPrimaryKernel<MaterialCount,EnergyCapacity,Warps,CacheCoefficients,SparseMaterials,true,SharedPaths,GroupWidth><<<dim3((g.nu*g.nv+Warps-1)/Warps,views),BlockSize,SharedBytes>>>(
            g,labels,map,mu,spectrum,energies,quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,profile);
        return;
    }
    warpPrimaryKernel<MaterialCount,EnergyCapacity,Warps,CacheCoefficients,SparseMaterials,false,SharedPaths,GroupWidth><<<dim3((g.nu*g.nv+Warps-1)/Warps,views),GroupWidth*Warps,SharedBytes>>>(
        g,labels,map,mu,spectrum,energies,quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask);
}

template<int MaterialCount>
void launchWarpPrimaryKernel(const DeviceGeometry& g,const unsigned char* labels,
    const int* map,const float* mu,const float* spectrum,const float* energies,
    int quantum_noise,int view,unsigned seed,unsigned quantum_seed,float* expected,
    float* sampled,float* air,unsigned char* mask,int views=1,bool report=false,unsigned long long* profile=nullptr){
    // Small material sets share the same capacity specialization. Its material
    // loop uses g.nm; absent coefficient-cache rows are initialized to zero.
    constexpr int Capacity=MaterialCount<=10?10:MaterialCount;
    const auto layout=selectPrimaryLaunchLayout(g.nm,g.ne,g.samples);
    if(g.ne>128){
        launchWarpPrimarySized<Capacity,256,false,false,true,32>(g,labels,map,mu,spectrum,energies,
            quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report,profile);
        return;
    }
    switch(layout.threads_per_pixel){
    case 4:launchWarpPrimarySized<Capacity,128,false,false,true,4>(g,labels,map,mu,spectrum,energies,
        quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report,profile);break;
    case 8:launchWarpPrimarySized<Capacity,128,false,false,true,8>(g,labels,map,mu,spectrum,energies,
        quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report,profile);break;
    case 16:launchWarpPrimarySized<Capacity,128,false,false,true,16,(Capacity>32?64:128)>(g,labels,map,mu,spectrum,energies,
        quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report,profile);break;
    // With few materials and many rays, private paths plus cached coefficients
    // outperform shared path updates. This is one layout of the same kernel.
    default:launchWarpPrimarySized<Capacity,128,Capacity==10,false,Capacity!=10,32>(g,labels,map,mu,spectrum,energies,
        quantum_noise,view,seed,quantum_seed,expected,sampled,air,mask,views,report,profile);break;
    }
}

} // namespace yk::spectral::primary_detail
